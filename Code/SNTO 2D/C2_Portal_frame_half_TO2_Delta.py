"""
2D stress-constrained TO driver -- ANALYTICAL hard-constraints, NO GP.

Multi-GPU domain decomposition (DDP) with a merged-lag adjoint force + N_MASK
void-exclusion. Displacement (u,v) and adjoint (t1,t2) satisfy their homogeneous
Dirichlet BCs EXACTLY via field(x) = A(x)*NN(x); density satisfies rho = 1 on the
external loading region EXACTLY via rho(x) = (1 - A_rho(x)) + A_rho(x)*proj(NN(x)).
All three A-weights share the same shape/K (AFUNC). No Gaussian-process correction
is built or used anywhere; the networks are LMGP objects used only as NN containers.

Launch with torchrun, one rank per GPU (single GPU -> plain `python <file>.py`,
world_size=1, no distributed init):
    torchrun --standalone --nproc_per_node=N <file>.py

Every rank holds a FULL replica of all networks (identical, same seed) and owns a
contiguous block of SUBSLICE_PER_RANK element-disjoint slices. Per step: accumulate
each slice's loss + a LOCAL partial adjoint force F_adj; all-reduce F_adj across
ranks and finalize the global normalizer (cached for next step's lag); all-reduce
gradients so the replicas step in sync.
"""
import os
# Pin each rank to its own GPU BEFORE importing torch, so the single-GPU data
# pipeline keeps working: each process then sees exactly one GPU as "cuda:0".
if "LOCAL_RANK" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import copy
import numpy as np
import time
import shutil
from models.lmgp_updated import LMGP
from tqdm import tqdm
from utils.utils_general import set_seed0, get_multiGPU, projectDensity
from utils.get_training_data_2D_updated import get_data_Portal_frame_half as get_data
from utils.domain_slicing import (attach_slices_to_mesh_data,
                                  accumulate_merged_lag_nmask,
                                  _finalize_fadj)

# 2D model layout: u,v (disp) + t1,t2 (adjoint) + density at index 4.
N_DISP = 2
N_ADJ = 2
PHASE_OFFSET = N_DISP + N_ADJ

# Domain decomposition with SUB-SLICING inside each rank.
# Each rank owns SUBSLICE_PER_RANK element-disjoint slices and accumulates over
# them (bounded peak memory); grads are all-reduced across ranks.
# Total slices = world_size * SUBSLICE_PER_RANK.
# Overridable from the PBS/env: `export SUBSLICE_PER_RANK=1` (=1 -> ONE slice per
# rank / max per-GPU work; increase only if a full per-rank block OOMs).
SUBSLICE_PER_RANK = int(os.environ.get("SUBSLICE_PER_RANK", 1))
N_SLICE = int(os.environ.get("WORLD_SIZE", 1)) * SUBSLICE_PER_RANK

# VOID EXCLUSION (see the Option 1 exclude_void file for the rationale):
# elements with density < VOID_THRESH are dropped from the element graphs each
# step (mass still summed over all elements). Controlled approximation; density
# is a neural field so excluded elements are not frozen. Set None for exact.
VOID_THRESH = 1e-3

# N_MASK (Option B): refresh each rank's active-element masks only every N_MASK
# epochs (per mesh) and cache in between; density forward over active only.
N_MASK = 20


# --- PER-PHASE TIMING INSTRUMENTATION ------------------------------------------
# Optional profiling: decompose one training iteration into its sequential pieces
# to see which part grows with N_SLICE / is redundant within an N_MASK window. We
# monkey-patch the domain_slicing helpers + Tensor.backward on the MODULE object,
# so accumulate_merged_lag_nmask resolves the timed versions at call time. Each
# piece is bracketed by cuda.synchronize(), which lands async GPU work in the right
# bucket but INFLATES the absolute total vs a normal run -> read PERCENTAGES /
# per-call counts. Set TIMING_ON=False (default) for a clean bottom-line run (the
# _phase timers are no-ops then).
import utils.domain_slicing as _ds
import time as _time
from collections import defaultdict as _dd
from contextlib import contextmanager as _cm

TIMING_ON = False           # True -> per-phase decomposition in Results_summary.txt
_PH_T = _dd(float)          # phase name -> accumulated seconds
_PH_N = _dd(int)            # phase name -> number of calls
_CUDA = torch.cuda.is_available()

def _sync():
    if _CUDA:
        torch.cuda.synchronize()

def _timed(name, fn):
    def _w(*a, **k):
        _sync(); _t0 = _time.perf_counter()
        r = fn(*a, **k)
        _sync(); _PH_T[name] += _time.perf_counter() - _t0; _PH_N[name] += 1
        return r
    return _w

@_cm
def _phase(name):
    if not (TIMING_ON and _CUDA):
        yield; return
    _sync(); _t0 = _time.perf_counter()
    try:
        yield
    finally:
        _sync(); _PH_T[name] += _time.perf_counter() - _t0; _PH_N[name] += 1

if TIMING_ON:
    # nmask (void path) calls _restrict_to_active + _view_from_sub DIRECTLY via
    # _slice_density_and_view; patch them individually (no nesting/double-count).
    _ds._phase_fields         = _timed('density_fwd', _ds._phase_fields)
    _ds._disp_fields_analytic = _timed('disp_fwd',    _ds._disp_fields_analytic)
    _ds._adj_fields_analytic  = _timed('adj_fwd',     _ds._adj_fields_analytic)
    _ds._elem_strains         = _timed('strains',     _ds._elem_strains)
    _ds._restrict_to_active = _timed('restrict',      _ds._restrict_to_active)
    _ds._view_from_sub      = _timed('view_gather',   _ds._view_from_sub)
    _ds._assemble_fadj      = _timed('fadj_assemble', _ds._assemble_fadj)
    _ds._finalize_fadj      = _timed('fadj_assemble', _ds._finalize_fadj)
    _orig_backward = torch.Tensor.backward
    def _timed_backward(self, *a, **k):
        _sync(); _t0 = _time.perf_counter()
        r = _orig_backward(self, *a, **k)
        _sync(); _PH_T['backward'] += _time.perf_counter() - _t0; _PH_N['backward'] += 1
        return r
    torch.Tensor.backward = _timed_backward

# --- ANALYTICAL HARD-CONSTRAINTS (no GP) ---------------------------------------
# Displacement (u,v) and adjoint (t1,t2) satisfy their homogeneous Dirichlet BCs
# EXACTLY via field(x) = A(x)*NN(x), A = 0 on the BC edge and -> 1 away. A_u, A_v
# are precomputed per mesh in the loader (_distance_weight), sliced in
# build_domain_slices, and applied as A*NN each iteration (_disp_fields_analytic /
# _adj_fields_analytic). DENSITY is analytical too: rho = (1 - A_rho) +
# A_rho*projectDensity(NN), so rho = 1 on the loading region. A_rho is built once in
# the loader (per element, like A_u/A_v) and applied in domain_slicing._phase_fields.
# This driver only sets AFUNC (below); all three weights share it.
# ---- analytical hard-BC weight A(d): d = normalized min-distance to the BC edge
#      (0 ON the BC -> 1 away). Choose ONE (hard-coded, no env var):
#        {'type': 'linear'}        A = d
#        {'type': 'quadratic'}     A = d**2
#        {'type': 'exp', 'k': K}   A = 1 - exp(-K*d)   (larger K -> thinner BC layer)
AFUNC = {'type': 'exp', 'k': 50.0}
AFUNC_RHO = {'type': 'exp', 'k': 500.0}   # SHARPER exp K for the density A_rho only:
# thinner un-adjustable (1-A_rho) transition band in rho=(1-A_rho)+A_rho*proj(NN), so the
# NN controls rho closer to the pinned region. Disp/adjoint A_u,A_v keep AFUNC (K=50).
# ---- objective / adjoint-scaling knobs (hard-coded per run) ----
INCLUDE_BODY = True    # objective = comp only  (True -> comp + alpha_b * 0.5*h^2 body term)
ALPHA_B_0 = 1.0
ALPHA_B_F = 1.0
frac_decrease_alpha_b = 0.5  # fraction of epoch to increase alpha_b from alpha_b_0 to alpha_b_f

F_SCALE_GOAL = 2.5e6      # target max|F_adj| after scaling (adjoint force normalizer)
LOG_BODY_COMP = False    # log body vs comp density-gradient magnitudes (for plot_history)

# When True, log the per-element density change between consecutive iterations (max /
# min / 95%-quantile of |rho_cur - rho_prev|) into timeHistory for an early-stop
# diagnostic. Costs one extra full-mesh density forward per epoch (rank 0 only); set
# False to skip it entirely for faster training.
difference_rho = False
difference_rho_skip = 100   # measure the density diff only every N epochs (cost / N); off-stride epochs store NaN

# --- minimum stress-violation design tracker -------------------------------
# After MIN_SEARCH_ITER epochs, whenever the stress-violation fraction reaches
# a new minimum, keep a CPU copy of the 3 prediction NNs (u, adj, rho). The
# copy is overwritten in place -> constant memory; cloning a small MLP
# state_dict is sub-millisecond, so doing it up to ~15k times is negligible.
# Set MIN_SEARCH_ITER < 0 to DISABLE the search entirely (no dict kept on CUDA/CPU).
MIN_SEARCH_ITER = 15000

# --- STRESS-VIOLATION METRICS + BEST-DESIGN TRACKERS (stored in timeHistory) ------
# All 'over yield' tests use the PHYSICAL C0 stress: over yield <=> svm0/sy > 1.
# Per epoch we count:
#   stress_viol_count_solid : # solids (rho>=0.5) whose MEAN-over-IP svm0/sy > 1
#   solid_count             : # solids (rho>=0.5)
#   n_h_activate            : # active elements (ANY rho) with ANY-IP h>0
#   n_h_activate_solid      : # solids (rho>=0.5) with ANY-IP h>0
# and three violation ratios (N_elem = total elements, massfrac = mass fraction):
#   stress_violation_1 = stress_viol_count_solid / solid_count
#   stress_violation_2 = n_h_activate            / (massfrac * N_elem)
#   stress_violation_3 = n_h_activate_solid      / (massfrac * N_elem)
# All five counts and three ratios are saved in timeHistory. For epochs >= MIN_SEARCH_ITER
# we keep the running MINIMUM design for six criteria, each written to disk with its epoch
# in the filename ( Trained_models_best{k}_{epoch}.pth ):
#   best1: min stress_violation_1   best4: min stress_viol_count_solid
#   best2: min stress_violation_2   best5: min n_h_activate
#   best3: min stress_violation_3   best6: min n_h_activate_solid
#
# NOTE (integration order): with REDUCED integration (wt=[4.0], single Gauss
# point, N_int=1) 'MEAN-over-IP > 1' and 'ANY-IP h>0' both collapse to the same
# per-element test svm0/sy > 1, so stress_viol_count_solid == n_h_activate_solid
# EXACTLY, and best4 == best6 (same design). The two only diverge under FULL
# integration (wt=[1,1,1,1]), where an element can exceed yield at some points
# but not on average. The three ratios still differ (different denominators), and
# n_h_activate (includes void) always differs from n_h_activate_solid.

# --- ANALYTICAL HARD-CONSTRAINT DENSITY ----------------------------------------
# rho = (1 - A_rho)*1 + A_rho*projectDensity(NN), with A_rho = 0 on the loading
# region (rho pinned to 1) -> 1 away. A_rho is built ONCE in the loader (per element,
# like A_u/A_v) and applied in domain_slicing._phase_fields via the cached per-slice
# weight -- this driver does not recompute it.

def _plot_A_weights(X, conn, A_u, A_v, A_rho, example, fname, afunc_bc=None, afunc_rho=None):
    """Filled contours of the three analytical hard-constraint weights on ONE figure:
    A_u, A_v (displacement/adjoint BCs) and A_rho (density). Each is 0 on its
    constraint (BC edge / loading region -> that field is pinned) and -> 1 away.
    Shared mesh triangulation (each CPS4 quad -> 2 triangles via the ACTUAL
    connectivity so holes/concavities are respected). Headless-safe; the caller wraps
    it in try/except so it never crashes training."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    x = X[:, 0].detach().cpu().numpy(); y = X[:, 1].detach().cpu().numpy()
    c = conn.detach().cpu().numpy()
    tris = np.vstack([c[:, [0, 1, 2]], c[:, [0, 2, 3]]])
    triang = mtri.Triangulation(x, y, tris)
    levels = np.linspace(0.0, 1.0, 21)          # A-weights are normalized to [0,1]
    def _klabel(af):
        if not af: return ''
        tt = af.get('type', 'linear')
        return f" (exp K={float(af.get('k', 10.0)):g})" if tt == 'exp' else f" ({tt})"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, A, nm, af in zip(axes, (A_u, A_v, A_rho), ('A_u', 'A_v', 'A_rho'),
                             (afunc_bc, afunc_bc, afunc_rho)):
        tcf = ax.tricontourf(triang, A.detach().cpu().numpy(), levels=levels,
                             cmap='viridis', vmin=0.0, vmax=1.0, extend='both')
        fig.colorbar(tcf, ax=ax, label=nm, ticks=np.linspace(0, 1, 6))
        ax.set_aspect('equal'); ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
        ax.set_title(f'{example}: {nm}{_klabel(af)}')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[A_weights] saved {fname}", flush=True)

# (label, bucket-key) tables used by the timing-report writer.
_TOP_PHASES = [("accumulate (nmask sweep)", "PHASE_accumulate"),
               ("optimizer step",           "PHASE_optim_step")]
_OP_PHASES  = [("disp_fwd  (_disp_fields_analytic)", "disp_fwd"),
               ("adj_fwd   (_adj_fields_analytic)",  "adj_fwd"),
               ("density_fwd(_phase_fields)",        "density_fwd"),
               ("strains   (_elem_strains)",         "strains"),
               ("restrict  [geo, CACHE-2]",          "restrict"),
               ("view_gather[geo, CACHE-2]",         "view_gather"),
               ("fadj_assemble+finalize",            "fadj_assemble"),
               ("backward  (Tensor.backward)",       "backward")]

def _write_timing_report(f, total_time, n_epochs, n_slice):
    f.write("\n================ TIMING DECOMPOSITION ================\n")
    f.write(f"N_SLICE = {n_slice}   N_MASK = {N_MASK}   epochs = {n_epochs}   "
            f"CUDA_sync = {_CUDA}   TIMING_ON = {TIMING_ON}\n")
    if not TIMING_ON:
        f.write("(TIMING_ON is False -> per-phase timers were no-ops. Set TIMING_ON=True\n"
                " at the top of the driver for the full per-phase breakdown.)\n")
        f.write("======================================================\n")
        return
    f.write("GPU-synchronized wall time accumulated over ALL epochs.\n")
    f.write("NOTE: per-phase cuda.synchronize() INFLATES the absolute total vs a\n")
    f.write("normal run. Compare PERCENTAGES and per-call counts (calls scale\n")
    f.write("~linearly with N_SLICE). restrict+view_gather = geometry rebuild\n")
    f.write("(CACHE-2), redundant within a mask window. NN forwards + backward are\n")
    f.write("the irreducible per-step cost.\n")
    tot = max(total_time, 1e-9)
    def line(label, key):
        t = _PH_T.get(key, 0.0); n = _PH_N.get(key, 0)
        f.write(f"  {label:<30s} {t:9.2f}s {100*t/tot:6.2f}%  calls={n:>9d}  {1e3*t/max(n,1):8.3f} ms/call\n")
    f.write("\n-- top-level phases (non-overlapping) --\n")
    for lab, key in _TOP_PHASES:
        line(lab, key)
    f.write("\n-- operation breakdown (GPU ops; subsets of the phases above) --\n")
    for lab, key in _OP_PHASES:
        line(lab, key)
    ph_sum = sum(_PH_T.get(k, 0.0) for _, k in _TOP_PHASES)
    op_sum = sum(_PH_T.get(k, 0.0) for _, k in _OP_PHASES)
    geo = _PH_T.get('restrict', 0.0) + _PH_T.get('view_gather', 0.0)
    f.write(f"\n  {'geometry rebuild (CACHE-2)':<30s} {geo:9.2f}s {100*geo/tot:6.2f}%  (restrict + view_gather)\n")
    f.write(f"  {'loss-math + py (phases-ops)':<30s} {ph_sum-op_sum:9.2f}s  (inline stress/comp math + overhead)\n")
    f.write(f"  {'sum of top-level phases':<30s} {ph_sum:9.2f}s\n")
    f.write(f"  {'measured total_time':<30s} {total_time:9.2f}s  (loop remainder = logging/mesh-assign/schedule)\n")
    f.write("======================================================\n")


def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size
    )
    # Each rank sees exactly one GPU (CUDA_VISIBLE_DEVICES pinned at import).
    torch.cuda.set_device(0)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_grads(models):
    """Sum gradients of the given models' parameters across all ranks, so every
    replica holds the full-domain gradient and steps in sync."""
    for m in models:
        for prm in m.parameters():
            if prm.grad is not None:
                dist.all_reduce(prm.grad, op=dist.ReduceOp.SUM)


def calculate_massfrac(model_list):
    """Full-mesh mass fraction using the SAME analytical density blend as training
    (rho = (1 - A_rho) + A_rho * projectDensity(NN)); NO GP. Uses the precomputed
    per-element weight model_list[0].A_rho_elem (set in the loop when the mesh is bound)."""
    elem_x = model_list[0].elem_x.clone()
    A_rho_elem = model_list[0].A_rho_elem
    elem_vol = model_list[0].elem_vol

    for model in model_list:
        model.eval()

    # Chunk the full-mesh density forward: the PGCAN grid-sample materializes large
    # per-point intermediates, which OOMs when elem_x is the whole mesh. Element
    # phases are row-independent, so we accumulate mass over element chunks.
    CHUNK = 20000  # element centroids per forward; lower if memory is still tight
    b = model_list[0].MP['b']
    rho_model = model_list[PHASE_OFFSET]

    with torch.no_grad():
        mass = elem_x.new_zeros(())
        for s in range(0, elem_x.shape[0], CHUNK):
            ex = elem_x[s:s + CHUNK]
            ev = elem_vol[s:s + CHUNK]
            p = projectDensity(rho_model.mean_module_NN_All(ex), b)   # [c, num_phase] in (0,1)
            A = A_rho_elem[s:s + CHUNK].unsqueeze(-1)                 # [c, 1] in [0,1]
            phase_weights = (1.0 - A) + A * p                        # blend, in [0,1]
            mass = mass + (phase_weights * ev.unsqueeze(-1)).sum()

    M0 = (elem_vol).sum()

    massfrac = mass / M0

    return massfrac


def compute_rho_full(model_list):
    """Full-mesh per-element density rho using the SAME analytical blend as
    calculate_massfrac (rho = (1 - A_rho) + A_rho * projectDensity(NN)); returns a
    detached 1-D tensor over all elements (num_phase columns flattened). Used only for
    the `difference_rho` early-stop diagnostic, so the extra forward is skipped unless
    that flag is on. The mesh is fixed (num_CP == 1), so element ordering is stable
    across epochs and the per-element diff is meaningful."""
    elem_x = model_list[0].elem_x
    A_rho_elem = model_list[0].A_rho_elem

    for model in model_list:
        model.eval()

    CHUNK = 20000  # match calculate_massfrac; lower if memory is tight
    b = model_list[0].MP['b']
    rho_model = model_list[PHASE_OFFSET]

    rho_chunks = []
    with torch.no_grad():
        for s in range(0, elem_x.shape[0], CHUNK):
            ex = elem_x[s:s + CHUNK]
            p = projectDensity(rho_model.mean_module_NN_All(ex), b)   # [c, num_phase] in (0,1)
            A = A_rho_elem[s:s + CHUNK].unsqueeze(-1)                 # [c, 1] in [0,1]
            rho_chunks.append(((1.0 - A) + A * p).reshape(-1))       # blend, flattened
    return torch.cat(rho_chunks)

def adjust_density_bias_bisection(model_list, target_massfrac, tol=1e-4, max_iter=50):
    with torch.no_grad():
        last_bias = model_list[4].mean_module_NN_All.network.last.bias
        original_bias = last_bias.clone()

        def eval_massfrac(shift):
            last_bias.copy_(original_bias + shift)
            return calculate_massfrac(model_list).item()

        # Current mass fraction at zero shift
        mf0 = eval_massfrac(0.0)
        f0 = mf0 - target_massfrac

        if abs(f0) < tol:
            return mf0, 0.0

        # Decide search direction automatically
        # Usually increasing bias increases density/mass fraction
        direction = 1.0 if f0 < 0.0 else -1.0

        # Exponentially expand until bracket is found
        step = 1e-3
        lo = 0.0
        hi = None
        f_lo = f0
        f_hi = None

        found_bracket = False
        for _ in range(40):
            trial = direction * step
            mf_trial = eval_massfrac(trial)
            f_trial = mf_trial - target_massfrac

            if abs(f_trial) < tol:
                last_bias.copy_(original_bias + trial)
                return mf_trial, trial

            # Check if target is bracketed between lo and trial
            if f0 * f_trial < 0.0:
                lo = 0.0
                hi = trial
                f_lo = f0
                f_hi = f_trial
                found_bracket = True
                break

            step *= 2.0

        if not found_bracket:
            # restore
            last_bias.copy_(original_bias)
            mf_restore = calculate_massfrac(model_list).item()
            return mf_restore, None

        # Make sure lo < hi for bisection
        if lo > hi:
            lo, hi = hi, lo
            f_lo, f_hi = f_hi, f_lo

        # Standard bisection
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            mf_mid = eval_massfrac(mid)
            f_mid = mf_mid - target_massfrac

            if abs(f_mid) < tol:
                last_bias.copy_(original_bias + mid)
                return mf_mid, mid

            if f_lo * f_mid < 0.0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid

        final_shift = 0.5 * (lo + hi)
        last_bias.copy_(original_bias + final_shift)
        mf_final = calculate_massfrac(model_list).item()
        return mf_final, final_shift
    
# ========================
# Worker: single GPU debug
# ========================
def run_worker(MP, NN_config_disp, NN_config_disp_adj, NN_config_rho,
               u_X_train, u_train, t1_X_train, t1_train,
               v_X_train, v_train, t2_X_train, t2_train,
               phase_train, index_CP, mesh_data,
               rank=0, world_size=1):

    is_main = (rank == 0)
    device = torch.device("cuda:0")   # each rank pinned to its GPU via CUDA_VISIBLE_DEVICES
    # save the initial MP:
    MP_init = copy.deepcopy(MP)
    Example = MP["Example"]
    Case = NN_config_disp["Case"]
    base_folder = MP["base_folder"]
    run_folder = f"{base_folder}/Results/{Example}/{Case}/"

    num_phase = MP["num_phase"]
    frac_decrease = MP['frac_decrease']
    frac_decrease_p = MP['frac_decrease_p']
    frac_decrease_sy = MP['frac_decrease_sy']
    DEM_frac = MP['DEM_frac']
    Adjoint_frac = MP['Adjoint_frac']
    TO_num_iter = NN_config_disp["TO_num_iter"]
    plotting_interval_TO = NN_config_disp["plotting_interval_TO"]
    delta = NN_config_disp["delta"]
    wc = NN_config_disp["wc"]
    wd = NN_config_disp["wd"]
    wm = NN_config_disp["wm"]
    wp = NN_config_disp["wp"]
    gradient_clip = NN_config_disp["gradient_clip"]
    nrmThreshold = NN_config_disp["nrmThreshold"]
    dynamic_weight = NN_config_disp["dynamic_weight"]
    random_state = NN_config_disp["random_state"]
    learning_rate_disp = NN_config_disp["learning_rate_disp"]
    learning_rate_rho = NN_config_disp["learning_rate_rho"]
    learning_rate_adj = NN_config_disp["learning_rate_adj"]
    Diff_type = MP["Diff_type"]

    # loop over random states (multiple runs)
    for i, state in enumerate(random_state, start=1):
        run_idx = i   # stable run index; the `for i in range(num_phase)` loop below reuses/clobbers i
        set_seed0(state)
        NN_config_disp["state"] = state
        # read the initial MP
        MP = copy.deepcopy(MP_init)
        # save folder (only rank 0 touches the filesystem)
        save_folder = f"{run_folder}Run_{i}/"
        if is_main:
            if os.path.exists(save_folder):
                shutil.rmtree(save_folder)
            os.makedirs(save_folder)
        if world_size > 1:
            dist.barrier()
        NN_config_disp["save_folder"] = save_folder
        NN_config_disp_adj["save_folder"] = save_folder
        NN_config_rho["save_folder"] = save_folder

        # ========================
        # Build models
        # ========================
        model_u = LMGP(u_X_train, u_train, NN_config_disp,
                       name_output="u", MP=MP, num_output=2).to(device)
        model_v = LMGP(v_X_train, v_train, NN_config_disp,
                       name_output="v", MP=MP, num_output=2).to(device)
        model_t1 = LMGP(t1_X_train, t1_train, NN_config_disp_adj,
                       name_output="t1", MP=MP, num_output=2).to(device)
        model_t2 = LMGP(t2_X_train, t2_train, NN_config_disp_adj,
                       name_output="t2", MP=MP, num_output=2).to(device)

        # One LMGP (NN container) per phase
        model_phases = []
        for i in range(num_phase):
            phase_X, phase_y = phase_train[i]   # from your phase_train list
            model_phase_i = LMGP(phase_X, phase_y, NN_config_rho,
                                name_output=f"rho", MP=MP, num_output=num_phase).to(device)
            model_phases.append(model_phase_i)

        # Keep all in list for BC handling
        model_list = [model_u, model_v, model_t1, model_t2] + model_phases

        # define the time history dict
        timeHistory = {
            'loss_total': [], 'loss_obj': [], 'loss_dem': [],'loss_adj':[],
            'loss_mConstraint': [], 'loss_cConstraint': [], 'loss_tv': [], 
            'strain_energy': [], 'external_work': [], 'spring_energy':[],
            'strain_energy_adj': [], 'external_work_adj': [], 'body_energy_adj': [],'spring_energy_adj':[],
            'loss_pde_1': [], 'loss_pde_2': [],
            'massfrac': [], 'costfrac': [],
            'wc': [], 'wd': [], 'wm': [], 'wp': [],
            'grey': [], 'p': [],'b':[],'alpha':[], 'ALPHA_B':[], 'sy':[],
            'stress_violation_1': [], 'stress_violation_2': [], 'stress_violation_3': [],
            'stress_viol_count_solid': [], 'solid_count': [],
            'n_h_activate': [], 'n_h_activate_solid': [],
            'rho_diff_max': [], 'rho_diff_avg': [], 'rho_diff_q95': []
        }

        # ========================
        # Optimizers + schedulers
        # ========================
        optimizer_disp = torch.optim.Adam(model_list[0].parameters(),
                                          lr=learning_rate_disp, amsgrad=True)
        optimizer_adj = torch.optim.Adam(model_list[2].parameters(),
                                          lr=learning_rate_adj, amsgrad=True)
        optimizer_rho = torch.optim.Adam(model_list[4].parameters(),
                                         lr=learning_rate_rho, amsgrad=True)

        # scheduler_disp = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer_disp,
        #     milestones=torch.linspace(0, TO_num_iter, 4).tolist(),
        #     gamma=0.75
        # )
        # scheduler_adj = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer_adj,
        #     milestones=torch.linspace(0, TO_num_iter, 4).tolist(),
        #     gamma=0.75
        # )
        
        # TO14: decay ONLY the density-NN lr, reaching ~1e-5 by ~50% of epochs.
        # 4 segments (3 steps), gamma=0.8 -> 2e-5, 1.6e-5, 1.28e-5, 1.024e-5. disp/adj stay fixed.
        scheduler_rho = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_rho,
            milestones=[int(0.20 * TO_num_iter), int(0.35 * TO_num_iter), int(0.50 * TO_num_iter)],
            gamma=0.8,
        )

        # ========================
        # Training loop
        # ========================
        epochs_iter = tqdm(range(TO_num_iter),
                           desc="GPU0 TO Epoch",
                           position=0, leave=True)
        total_time, start_time = 0, time.time()

        # snapshot helper: CPU copy of the 3 prediction NNs (u, adj, rho). Cloning a
        # small MLP state_dict is sub-ms; disk writes are deferred to after training.
        def _snapshot_pred_models():
            return {
                'model_u_state_dict':   {k: v.detach().cpu().clone() for k, v in model_list[0].mean_module_NN_All.state_dict().items()},
                'model_adj_state_dict': {k: v.detach().cpu().clone() for k, v in model_list[2].mean_module_NN_All.state_dict().items()},
                'model_rho_state_dict': {k: v.detach().cpu().clone() for k, v in model_list[4].mean_module_NN_All.state_dict().items()},
            }

        # min stress-violation design tracker (reset each run)
        # six best-design trackers (reset each run): running MINIMUM per criterion.
        # each holds {'val','epoch','state'}; the snapshot is shared across criteria that
        # improve in the same epoch (identical weights). Saved to disk after training.
        best_trackers = {k: {'val': float('inf'), 'epoch': -1, 'state': None}
                         for k in ('best1','best2','best3','best4','best5','best6')}
        best_meaning = {'best1':'stress_violation_1','best2':'stress_violation_2',
                        'best3':'stress_violation_3','best4':'stress_viol_count_solid',
                        'best5':'n_h_activate','best6':'n_h_activate_solid'}
        # gray-area milestone snapshots: capture the FIRST epoch the gray fraction
        # drops to <= each target. Written to disk after training (time NOT counted).
        gray_targets = [(0.05, 'gray5p'), (0.04, 'gray4p'), (0.03, 'gray3p'),
                        (0.02, 'gray2p'), (0.01, 'gray1p')]
        gray_states = {}   # tag -> (epoch, cpu state_dict)

        # --- GPU capacity report (per rank) + reset peak-memory tracking ---
        _prop = torch.cuda.get_device_properties(0)
        _cap_gb = _prop.total_memory / 1024**3
        print(f"[rank {rank}] GPU '{_prop.name}'  capacity = {_cap_gb:.1f} GiB  "
              f"(SUBSLICE_PER_RANK={SUBSLICE_PER_RANK}, N_SLICE={N_SLICE})", flush=True)
        torch.cuda.reset_peak_memory_stats(0)

        # Per-mesh caches: GLOBAL lagged F_adj (assembled via cross-rank all-reduce)
        # and this rank's active-mask cache (Option B, N_MASK).
        fadj_cache = {}
        mask_cache_by_mesh = {}
        mask_age = {}

        # previous-epoch full-mesh density, for the difference_rho early-stop diagnostic
        # (None until the first epoch produces a design to compare against)
        rho_prev = None

        for epoch in epochs_iter:
            optimizer_disp.zero_grad()
            optimizer_adj.zero_grad()
            optimizer_rho.zero_grad()

            # --- mesh slice ---
            index = index_CP[epoch]
            mesh_key = f"Mesh_{index:02d}"
            GPU_key = "GPU0"  # single GPU

            # assign mesh data to model_u
            model_list[0].collocation_x = mesh_data[GPU_key][mesh_key]["X_node"]
            model_list[0].elem_x        = mesh_data[GPU_key][mesh_key]["X_elem"]
            model_list[0].A_rho_elem    = mesh_data[GPU_key][mesh_key]["A_rho"]
            model_list[0].elem_vol        = mesh_data[GPU_key][mesh_key]["elem_vol"]
            model_list[0].f_index       = mesh_data[GPU_key][mesh_key]["f_index"]
            model_list[0].f_magnitude   = mesh_data[GPU_key][mesh_key]["f_magnitude"]
            model_list[0].conn          = mesh_data[GPU_key][mesh_key]["conn"]
            model_list[0].B             = mesh_data[GPU_key][mesh_key]["B"]
            model_list[0].detJ          = mesh_data[GPU_key][mesh_key]["detJ"]

            # visualize the analytical weights A_u, A_v, A_rho on ONE figure. The A field
            # is geometry-only and identical across the 5 runs, so write it ONCE (first
            # run, first epoch, rank 0) to the Case folder (run_folder), not per-run. Never
            # crashes training. Each panel title shows that weight's K; file named by Case.
            if is_main and run_idx == 1 and epoch == 0:
                try:
                    _md = mesh_data[GPU_key][mesh_key]
                    _Xn = _md["X_node"]
                    _plot_A_weights(_Xn, _md["conn"], _md["A_u"], _md["A_v"], _md["A_rho_node"],
                                    MP['Example'],
                                    run_folder + f"A_weights_{Case}.pdf",
                                    afunc_bc=MP['AFUNC'], afunc_rho=MP.get('AFUNC_RHO', MP['AFUNC']))
                except Exception as e:
                    print(f"[A_weights] plot skipped: {e}", flush=True)

            # offset the last layer to change the initial massfrac
            if epoch == 0:
                mf_final, shift = adjust_density_bias_bisection(
                                        model_list,
                                        target_massfrac=MP['massfrac0'],
                                        tol=1e-4,
                                        max_iter=50,
                                    )
                if shift is None:
                    print(f"Failed to bracket target. Current mass fraction = {mf_final:.6f}")
                else:
                    print(f"Adjusted mass fraction: {mf_final:.6f}, bias shift: {shift:.6f}")

                # record the bias shift (persisted in MP.pt + summary)
                model_list[0].MP['initial_bias_shift'] = shift
                model_list[0].MP['initial_massfrac_shift'] = mf_final

                # Save the INITIAL (post-bias-shift) parameters -- the exact state
                # training starts from -- for later init -> topology correlation.
                if is_main:
                    torch.save(
                        {
                            'seed': state,
                            'bias_shift': shift,
                            'massfrac_after_shift': mf_final,
                            'model_u_state_dict': model_list[0].mean_module_NN_All.state_dict(),
                            'model_adj_state_dict': model_list[N_DISP].mean_module_NN_All.state_dict(),
                            'model_rho_state_dict': model_list[PHASE_OFFSET].mean_module_NN_All.state_dict(),
                        },
                        save_folder + 'Initial_models_0.pth'
                    )

            # --- decide training phase & weights (same schedule as original) ---
            if epoch <= DEM_frac:
                include_adj, include_obj_mass = False, False
            elif (epoch > DEM_frac) and (epoch <= DEM_frac + Adjoint_frac):
                include_adj, include_obj_mass = True, False
            else:
                include_adj, include_obj_mass = True, True
                wc = model_list[0].MP['wc']

            # ============================================================
            # merged-lag + N_MASK single sweep:
            #   * each rank owns SUBSLICE_PER_RANK element-disjoint slices;
            #   * the GLOBAL lagged F_adj is assembled by ALL-REDUCING each rank's
            #     partial F_adj (raw), then finalized (global alpha) for next step;
            #   * grads are all-reduced so every replica steps in sync.
            # ============================================================
            all_slices = mesh_data[GPU_key][mesh_key]["slices"]
            k = SUBSLICE_PER_RANK
            my_slice = all_slices[rank * k:(rank + 1) * k]   # this rank's block
            Nn_mesh = mesh_data[GPU_key][mesh_key]["X_node"].shape[0]

            # GLOBAL lagged F_adj for this mesh (zeros the first time it is seen)
            F_adj_lag, alpha_lag = fadj_cache.get(
                mesh_key,
                (torch.zeros(Nn_mesh, 2, device=device),
                 torch.tensor(1.0, device=device)),
            )
            # N_MASK: refresh THIS rank's active masks every N_MASK epochs (all ranks
            # share the epoch schedule -> they refresh in lock-step).
            refresh = (mesh_key not in mask_cache_by_mesh) or \
                      (epoch - mask_age[mesh_key] >= N_MASK)
            mc = None if refresh else mask_cache_by_mesh[mesh_key]

            # --- per-epoch compute+comm timer (synced); should DECREASE as void
            #     exclusion shrinks the active set. Covers forwards, backward, and
            #     (multi-GPU) the cross-rank all-reduces + optimizer step. ---
            _sync(); _ep_t0 = _time.perf_counter()

            # density is the analytical A_rho blend (patched into
            # domain_slicing._phase_fields); displacement/adjoint are A(x)*NN.
            with _phase('PHASE_accumulate'):
                log, F_adj_raw, new_masks = accumulate_merged_lag_nmask(
                    model_list, my_slice,
                    wd=wd, wc=wc, wm=wm, delta=delta,
                    include_adj=include_adj, include_obj_mass=include_obj_mass,
                    F_adj_scaled_lag=F_adj_lag, alpha_lag=alpha_lag,
                    void_thresh=VOID_THRESH, mask_cache=mc, return_raw_fadj=True,
                )

            if refresh:
                mask_cache_by_mesh[mesh_key] = new_masks
                mask_age[mesh_key] = epoch

            # --- assemble the GLOBAL F_adj for next step's lag: all-reduce each
            #     rank's partial (raw) F_adj, then finalize (global-max alpha) ---
            if world_size > 1:
                dist.all_reduce(F_adj_raw, op=dist.ReduceOp.SUM)
            F_adj_scaled_new, alpha_new = _finalize_fadj(F_adj_raw, Nn_mesh, F_SCALE_GOAL)
            fadj_cache[mesh_key] = (F_adj_scaled_new.detach(), alpha_new.detach())
            alpha_val = float(alpha_new.item())

            # sum gradients across ranks -> full-domain gradient on every replica
            if world_size > 1:
                reduce_grads([model_list[0], model_list[N_DISP], model_list[PHASE_OFFSET]])

            # reduce the scalar diagnostics so rank 0 logs GLOBAL quantities
            scal = torch.tensor([
                log['strain_energy'], log['external_work'], log['strain_energy_adj'],
                log['body_energy_adj'], log['obj'], log['mass'], log['loss_total'],
                float(log.get('stress_viol_count_solid', 0.0)),  # solids mean-IP over-yield (global after all-reduce)
                float(log.get('solid_count', 0.0)),              # solids rho>=0.5 (global after all-reduce)
                float(log.get('n_h_activate', 0.0)),             # any-IP h>0, all active (global after all-reduce)
                float(log.get('n_h_activate_solid', 0.0)),       # any-IP h>0, solids only (global after all-reduce)
            ], device=device)
            grey = torch.tensor(log['grey_counts'] + [log['count_total'], log['active_total']],
                                device=device)
            if world_size > 1:
                dist.all_reduce(scal, op=dist.ReduceOp.SUM)
                dist.all_reduce(grey, op=dist.ReduceOp.SUM)
            (g_SE, g_EW, g_SEadj, g_body, g_obj, g_mass, g_loss, g_viol, g_solid, g_hact, g_hact_solid) = scal.tolist()
            massfrac = g_mass / log['M0']
            g_count = grey[-2]; g_active = float(grey[-1].item())
            grey_fraction = (grey[:-2] / torch.clamp(g_count, min=1)).cpu().tolist()
            # stress-violation monitor: DEFINITION CONSISTENT WITH plot_stress_figure.py.
            # (# violating solids) / (# solids); both count rho>=0.5 elements, a solid
            # violating when its physical mean-over-IP von-Mises ratio svm/sy > 1.
            N_elem = float(g_count.item())                   # total elements (all-reduced count)
            massfrac_N = massfrac * N_elem                   # ~ expected # solids (denom of viol_2/3)
            stress_violation_1 = (g_viol / g_solid) if g_solid > 0 else 0.0
            stress_violation_2 = (g_hact / massfrac_N) if massfrac_N > 0 else 0.0
            stress_violation_3 = (g_hact_solid / massfrac_N) if massfrac_N > 0 else 0.0

            # --- keep the running-MINIMUM design for all six criteria (rank 0; NNs
            #     identical across ranks). Captured before the optimizer step so the
            #     weights match these values. One snapshot per epoch, shared. ---
            if is_main and MIN_SEARCH_ITER >= 0 and epoch >= MIN_SEARCH_ITER:
                _best_vals = {'best1': stress_violation_1, 'best2': stress_violation_2,
                              'best3': stress_violation_3, 'best4': g_viol,
                              'best5': g_hact, 'best6': g_hact_solid}
                _snap = None
                for _tag, _val in _best_vals.items():
                    if _val < best_trackers[_tag]['val']:
                        if _snap is None:
                            _snap = _snapshot_pred_models()
                        best_trackers[_tag] = {'val': _val, 'epoch': epoch, 'state': _snap}

            # --- gray-area milestone capture: first epoch the gray fraction hits each
            #     target (captured before the optim step so weights match the gray value) ---
            if is_main:
                gray_val = float(sum(grey_fraction))
                for _g_thr, _g_tag in gray_targets:
                    if _g_tag not in gray_states and gray_val <= _g_thr:
                        gray_states[_g_tag] = (epoch, _snapshot_pred_models())

            if epoch == 0:
                initial_massfrac = massfrac
                frac_step_mass = (initial_massfrac - massfrac_f)/frac_decrease
                frac_step_p = (MP['pf'] - MP['p0'])/frac_decrease_p
                frac_step_sy = (MP['syf'] - MP['sy0'])/frac_decrease_sy
                frac_step_alpha_b = (MP['ALPHA_B_F'] - MP['ALPHA_B_0'])/frac_decrease_alpha_b
                frac_step_b = (MP['b_f'] - MP['b_0'])/frac_decrease
                frac_step_wc = (MP['wc_f'] - MP['wc_0'])/frac_decrease
                model_list[0].MP['p'] = MP['p0']
                model_list[0].MP['sy'] = MP['sy0']
                model_list[0].MP['ALPHA_B'] = MP['ALPHA_B']
                model_list[0].MP['b'] = MP['b_0']
                model_list[0].MP['wc'] = MP['wc_0']
                model_list[0].MP['massfrac_star'] = initial_massfrac
                model_list[0].MP['frac_step_p'] = frac_step_p
                model_list[0].MP['frac_step_sy'] = frac_step_sy
                model_list[0].MP['frac_step_alpha_b'] = frac_step_alpha_b
                model_list[0].MP['frac_step_b'] = frac_step_b
                model_list[0].MP['frac_step_wc'] = frac_step_wc
                model_list[0].MP['frac_step_mass'] = frac_step_mass
                MP['frac_step_mass'] = frac_step_mass
                MP['frac_step_p'] = frac_step_p
                MP['frac_step_sy'] = frac_step_sy
                MP['frac_step_alpha_b'] = frac_step_alpha_b
                MP['frac_step_b'] = frac_step_b
                MP['frac_step_wc'] = frac_step_wc

            # --- gradient clip on disp/adj/density NNs + optimizer steps (all ranks) ---
            with _phase('PHASE_optim_step'):
                torch.nn.utils.clip_grad_norm_(model_list[0].parameters(), nrmThreshold)
                torch.nn.utils.clip_grad_norm_(model_list[N_DISP].parameters(), nrmThreshold)
                torch.nn.utils.clip_grad_norm_(model_list[PHASE_OFFSET].parameters(), nrmThreshold/10)
                optimizer_disp.step()
                optimizer_adj.step()
                optimizer_rho.step()
                scheduler_rho.step()

            # --- per-epoch time (synced) + active count; rank 0 logs to timeHistory ---
            _sync(); _ep_dt = _time.perf_counter() - _ep_t0
            if is_main:
                timeHistory.setdefault('epoch_time', []).append(_ep_dt)
                if epoch < 5 or (epoch + 1) % 200 == 0:
                    _c = float(g_count.item())
                    print(f"[epoch {epoch+1:6d}] epoch_time={_ep_dt*1e3:7.1f} ms   "
                          f"active={int(g_active)}/{int(_c)} ({100*g_active/max(_c,1):4.1f}%)", flush=True)

            # --- save the loss histories (rank 0 only, GLOBAL quantities) ---
            if is_main:
                timeHistory['loss_total'].append(g_loss)
                timeHistory['loss_obj'].append(g_obj)
                timeHistory['loss_dem'].append(g_SE - g_EW)
                timeHistory['loss_adj'].append(g_SEadj + g_body)
                timeHistory['loss_mConstraint'].append(massfrac)
                timeHistory['massfrac'].append(massfrac)
                timeHistory['strain_energy'].append(g_SE)
                timeHistory['external_work'].append(g_EW)
                timeHistory['strain_energy_adj'].append(g_SEadj)
                timeHistory['body_energy_adj'].append(g_body)
                timeHistory['wc'].append(model_list[0].MP['wc'])
                timeHistory['wd'].append(wd)
                timeHistory['wm'].append(wm)
                timeHistory['wp'].append(wp)
                timeHistory['grey'].append(grey_fraction)
                timeHistory['stress_violation_1'].append(stress_violation_1)
                timeHistory['stress_violation_2'].append(stress_violation_2)
                timeHistory['stress_violation_3'].append(stress_violation_3)
                timeHistory['stress_viol_count_solid'].append(g_viol)
                timeHistory['solid_count'].append(g_solid)
                timeHistory['n_h_activate'].append(g_hact)
                timeHistory['n_h_activate_solid'].append(g_hact_solid)
                timeHistory['p'].append(model_list[0].MP['p'])
                timeHistory['sy'].append(model_list[0].MP['sy'])
                timeHistory['ALPHA_B'].append(model_list[0].MP['ALPHA_B'])
                timeHistory['b'].append(model_list[0].MP['b'])
                timeHistory['alpha'].append(alpha_val)   # GLOBAL alpha (post all-reduce)
                timeHistory.setdefault('active_total', []).append(g_active)
                timeHistory.setdefault('count_total', []).append(float(g_count.item()))
                # body vs comp density-gradient magnitudes (single-GPU: rank-0 log = global)
                _bcn = max(log.get('bc_n', 0), 1)
                timeHistory.setdefault('grad_comp_norm', []).append(log['bc_csq'] ** 0.5 if 'bc_csq' in log else float('nan'))
                timeHistory.setdefault('grad_body_norm', []).append(log['bc_bsq'] ** 0.5 if 'bc_bsq' in log else float('nan'))
                timeHistory.setdefault('grad_comp_max', []).append(log.get('bc_cmax', float('nan')))
                timeHistory.setdefault('grad_body_max', []).append(log.get('bc_bmax', float('nan')))
                timeHistory.setdefault('grad_comp_mean', []).append(log['bc_cabs'] / _bcn if 'bc_cabs' in log else float('nan'))
                timeHistory.setdefault('grad_body_mean', []).append(log['bc_babs'] / _bcn if 'bc_babs' in log else float('nan'))
                # --- rho-change early-stop diagnostic: compare the current full-mesh
                #     density to the previous epoch's and log max / min / 95%-quantile of
                #     the per-element change magnitude |rho_cur - rho_prev|. Guarded by
                #     `difference_rho`; adds one full-mesh density forward (rank 0 only). ---
                if difference_rho:
                    if epoch % difference_rho_skip == 0:
                        rho_cur = compute_rho_full(model_list)
                        if rho_prev is None:
                            _mx = _av = _q = float('nan')
                        else:
                            d = (rho_cur - rho_prev).abs()
                            _mx = d.max().item(); _av = d.mean().item(); _q = torch.quantile(d, 0.95).item()
                        rho_prev = rho_cur
                    else:
                        _mx = _av = _q = float('nan')      # off-stride epoch: keep arrays aligned
                    timeHistory['rho_diff_max'].append(_mx)
                    timeHistory['rho_diff_avg'].append(_av)
                    timeHistory['rho_diff_q95'].append(_q)
            # --- checkpoint (rank 0 only) ---
            if is_main and (epoch+1) % plotting_interval_TO == 0:
                end_time = time.time()
                torch.save(
                    {
                        'model_u_state_dict': model_list[0].mean_module_NN_All.state_dict(),
                        'model_adj_state_dict': model_list[2].mean_module_NN_All.state_dict(),
                        'model_rho_state_dict': model_list[4].mean_module_NN_All.state_dict(),
                    }, 
                     save_folder + f'Trained_models_{epoch}.pth'
                    )
                total_time = total_time + (end_time - start_time)
                start_time = time.time()

            model_list[0].MP['massfrac_star'] = max(model_list[0].MP['massfrac_f'], model_list[0].MP['massfrac_star'] - model_list[0].MP['frac_step_mass'])
            # model_list[0].MP['costfrac_star'] = max(model_list[0].MP['costfrac_f'], model_list[0].MP['costfrac_star'] - model_list[0].MP['frac_step_cost'])
            if epoch > (DEM_frac + Adjoint_frac):
                model_list[0].MP['p'] = min(model_list[0].MP['pf'], model_list[0].MP['p'] + model_list[0].MP['frac_step_p'])
                model_list[0].MP['sy'] = max(model_list[0].MP['syf'], model_list[0].MP['sy'] + model_list[0].MP['frac_step_sy'])
                model_list[0].MP['ALPHA_B'] = min(model_list[0].MP['ALPHA_B_F'], model_list[0].MP['ALPHA_B'] + model_list[0].MP['frac_step_alpha_b'])
                model_list[0].MP['b'] = min(model_list[0].MP['b_f'], model_list[0].MP['b'] + model_list[0].MP['frac_step_b'])
                model_list[0].MP['wc'] = min(model_list[0].MP['wc_f'], model_list[0].MP['wc'] + model_list[0].MP['frac_step_wc'])

        end_time = time.time()
        total_time += (end_time - start_time)

        # --- peak GPU memory per rank (use this to tune SUBSLICE_PER_RANK / how
        #     large a mesh each GPU can hold). Every rank must reach the all-reduce. ---
        _peak_alloc = torch.cuda.max_memory_allocated(0) / 1024**3
        _peak_resv = torch.cuda.max_memory_reserved(0) / 1024**3
        print(f"[rank {rank}] peak GPU mem: allocated {_peak_alloc:.2f} GiB, "
              f"reserved {_peak_resv:.2f} GiB / {_cap_gb:.1f} GiB "
              f"({100*_peak_resv/_cap_gb:.0f}% reserved)", flush=True)
        _peak_glob = torch.tensor(_peak_resv, device=device)
        if world_size > 1:
            dist.all_reduce(_peak_glob, op=dist.ReduceOp.MAX)
        _peak_glob = float(_peak_glob.item())

        # Only rank 0 writes results; other ranks move on to the next run/state.
        if not is_main:
            continue
        print(f"Run {i} finished in {total_time:.2f}s")

        # -----------------------------
        # Save metadata and results
        # -----------------------------
        torch.save(MP, save_folder + "MP.pt")
        with open(save_folder + "NN_config_disp.json", "w") as file:
            json.dump(NN_config_disp, file, indent=4)
        with open(save_folder + "NN_config_disp_adj.json", "w") as file:
            json.dump(NN_config_disp_adj, file, indent=4)
        with open(save_folder + "NN_config_rho.json", "w") as file:
            json.dump(NN_config_rho, file, indent=4)

        torch.save({
            "u_X_train": u_X_train, "u_train": u_train,
            "v_X_train": v_X_train, "v_train": v_train,
            "t1_X_train": t1_X_train, "t1_train": t1_train,
            "t2_X_train": t2_X_train, "t2_train": t2_train,
            "phase_train": phase_train,
        }, save_folder + "Training.pth")

        # --- six best designs (each the MINIMUM of its criterion over epochs >=
        #     MIN_SEARCH_ITER). Same keys as periodic checkpoints -> prediction loads
        #     them identically. Epoch embedded in each filename. ---
        _any_best = False
        for _tag, _tr in best_trackers.items():
            if _tr['state'] is not None:
                torch.save(_tr['state'], save_folder + f"Trained_models_{_tag}_{_tr['epoch']}.pth")
                timeHistory[f'{_tag}_value'] = _tr['val']
                timeHistory[f'{_tag}_epoch'] = _tr['epoch']
                timeHistory[f'{_tag}_metric'] = best_meaning[_tag]
                print(f"[{_tag}] min {best_meaning[_tag]} = {_tr['val']:.6g} at epoch {_tr['epoch']+1} "
                      f"-> Trained_models_{_tag}_{_tr['epoch']}.pth", flush=True)
                _any_best = True
        if MIN_SEARCH_ITER < 0:
            print("[best] MIN_SEARCH_ITER < 0 -> best-model search disabled; nothing saved.", flush=True)
        elif not _any_best:
            print(f"[best] no epoch >= MIN_SEARCH_ITER ({MIN_SEARCH_ITER}); no best model saved.", flush=True)

        # --- gray-area milestone snapshots to disk (save time NOT counted in total_time) ---
        for _g_tag, (_g_ep, _g_state) in gray_states.items():
            torch.save(_g_state, save_folder + f"Trained_models_{_g_tag}_{_g_ep}.pth")
            timeHistory[f'{_g_tag}_epoch'] = _g_ep
            print(f"[gray] {_g_tag} reached at epoch {_g_ep+1} -> Trained_models_{_g_tag}_{_g_ep}.pth", flush=True)

        with open(save_folder + "timeHistory.json", "w") as file:
            json.dump(timeHistory, file)

        result_file_path = f"{save_folder}Results_summary.txt"
        with open(result_file_path, 'w') as f:
            f.write(f"The total training time in second is: {total_time}\n")
            f.write(f"GPU: {_prop.name}  capacity: {_cap_gb:.1f} GiB  "
                    f"peak reserved (max over {world_size} ranks): {_peak_glob:.2f} GiB "
                    f"({100*_peak_glob/_cap_gb:.0f}%)  "
                    f"[world_size={world_size}, SUBSLICE_PER_RANK={SUBSLICE_PER_RANK}]\n")
            f.write(f"Final value strain energy: {timeHistory['strain_energy'][-1]}\n")
            f.write(f"Final value external work: {timeHistory['external_work'][-1]}\n")
            f.write(f"Random seed (state): {state}\n")
            f.write(f"omega (raw lengthscale): {NN_config_disp['omega']}\n")
            f.write(f"Analytical hard-BC weight: {MP.get('AFUNC', 'n/a')}\n")
            f.write(f"Jitter used (max over networks): {MP.get('jitter', 'n/a')}\n")
            f.write(f"Initial density bias shift: {MP.get('initial_bias_shift', 'n/a')}\n")
            f.write(f"Initial mass fraction after shift: {MP.get('initial_massfrac_shift', 'n/a')}\n")

            # --- per-epoch compute time summary + (optional) per-phase decomposition ---
            _ets = timeHistory.get('epoch_time', [])
            if _ets:
                f.write(f"Epoch compute time: mean {1e3*np.mean(_ets):.1f} ms, "
                        f"median {1e3*np.median(_ets):.1f} ms  (over {len(_ets)} epochs)\n")
            _write_timing_report(f, total_time, len(timeHistory['loss_total']), N_SLICE)

# ========================
# Launch block (torchrun)
# ========================
if __name__ == "__main__":
    # Try to get ranks from torchrun, else default to single-GPU debug mode
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    ############################### Define Parameters ##############################################
    # base_folder = f"/home/alexsunuci/Aug_20_2D/"
    base_folder = f"/u/xsun7/Aug_21_2D/"
    # base_folder = os.getcwd()
    random_state = [1,3,5,7,9]   
    num_CP = 1   # 250K L_beam data has ONE (large) mesh -> index_CP is always Mesh_01
    Example = 'Portal_frame_half'
    Case = 'C2_Portal_frame_half_TO2_Delta'
    Diff_type = 'CPS4_reduced_250K'   # Portal_frame_half 250K data
    N_worker = 1  # for local debugging with 1 GPU
    # define material properties and
    # wt = [1.0, 1.0, 1.0, 1.0] # full integration
    wt = [4.0] # 4 for CPS4 reduced integration, 0.5 for CPS3
    D = [1.0]  # density
    E = [7.0]  # modulus
    P = [1.0]  # cost
    nu = 0.25  # Poisson's ratio
    p0 = 3.0  # initial penalty
    pf = 5.0  # final penalty
    sy0 = 0.01   # initial yield stress
    syf = 0.009   # final yield stress
    f_factor = 0.15
    a_R = 1 
    massfrac_0 = 0.6  # initial mass fraction (match TO3)
    massfrac_f = 0.3  # final mass fraction
    costfrac_f = 0.6  # final mass fraction
    frac_decrease = 0.5  # fraction of epoch to decrease volume fraction from VF0 to VF_f
    frac_decrease_p = 0.5  # fraction of epoch to increase p from p0 to p_f
    frac_decrease_sy = 0.5  # fraction of epoch to decrease sy from sy0 to syf

    DEM_frac = 0.05 # epoch fraction of DEM for primary displacements
    Adjoint_frac = 0.05 # epoch fraction of adjoint problems
    b_0 = 4  # sharpness parameters
    b_f = 8  # sharpness parameters (match TO3)
    thres_static = 0.5  # static threshold values to binarize the density field
    rho_min = 0.1  # lower limit to define grey element (rho_min,rho_max)
    rho_max = 0.9  # upper limit to define grey element (rho_min,rho_max)

    # define domain
    Nelx = 60
    Nely = 60
    xmin, xmax = 0.0, 60.0
    ymin, ymax = 0.0, 60.0
    thick = 10
    num_bc = 200
    num_load = 25

    # define model parameters and training parameters
    init_method = 'kaiming_uniform_'
    dynamic_weight = False
    gradient_clip = False
    omega = 0.0   # LMGP kernel lengthscale (kernel unused; needed only to build the LMGP NN container)
    learning_rate_disp = 1e-3
    learning_rate_adj = 1e-2
    learning_rate_rho = 2e-5
    nrmThreshold = 0.1
    TO_num_iter = 30000
    plot_num = 30
    delta = 1e-1
    alpha_0 = 0.1
    alpha_f = 0.1
    wc_0 = 1.0 
    wc_f = 1000.0 
    wd = 1e3
    wm = 1.0    # mass fraction
    wp = 0.0
    wg = 1e3
    basis = 'PGCAN'
    basis_rho = 'PGCAN'
    quant_correlation_class = 'Rough_RBF'
    activation = 'tanh'
    if basis == 'PGCAN':
        n_features = 128
        n_cells = 3
        res = [24, 24]
        n_neurons = int(n_features / 2)
        n_layers = 3
        NN_arch = [n_neurons] * n_layers
        kernel_size = (3, 3)   # match TO3: even kernel breaks SAME padding
    else:
        n_features = None
        n_cells = None
        res = None
        n_neurons = 64
        n_layers = 6
        NN_arch = [n_neurons] * n_layers
        kernel_size = None

    ############################### Pre-processing ##############################################
    num_phase = len(D)
    tkwargs = get_multiGPU(N_worker)
    D_tensor = [torch.tensor(D, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    E_tensor = [torch.tensor(E, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    P_tensor = [torch.tensor(P, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    wt_tensor = [torch.tensor(wt, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    domain = {'x': [xmin, xmax], 'y': [ymin, ymax]}
    frac_decrease = int(frac_decrease * TO_num_iter)
    frac_decrease_p = int(frac_decrease_p * TO_num_iter)
    frac_decrease_sy = int(frac_decrease_sy * TO_num_iter)
    frac_decrease_alpha_b = int(frac_decrease_alpha_b * TO_num_iter)
    DEM_frac = int(DEM_frac * TO_num_iter)
    Adjoint_frac = int(Adjoint_frac * TO_num_iter)
    plotting_interval_TO = TO_num_iter // plot_num
    frac_step = None
    alpha_step =  (alpha_f - alpha_0)/(1e-6+TO_num_iter - DEM_frac - Adjoint_frac)
    MP = {'num_CP': num_CP, 'Example': Example, 'N_worker': N_worker,'num_bc': num_bc, 'num_load': num_load,
          'AFUNC': AFUNC,   # analytical hard-BC weight (see AFUNC at top of file)
          'AFUNC_RHO': AFUNC_RHO,   # sharper density weight (see AFUNC_RHO at top)
          'INCLUDE_BODY': INCLUDE_BODY, 'F_scale_goal': F_SCALE_GOAL, 'LOG_BODY_COMP': LOG_BODY_COMP,
          'base_folder': base_folder, 'domain': domain,
          'thres_static': thres_static, 'Diff_type': Diff_type,
          'num_phase': num_phase, 'Nelx': Nelx, 'Nely': Nely,
          'D': D_tensor, 'E': E_tensor, 'P': P_tensor, 'wt':wt_tensor,'nu': nu,'sy':sy0,'sy0':sy0,'syf':syf,
          'p': p0, 'p0': p0, 'pf': pf, 'b': b_0,'b_0': b_0,'b_f': b_f, 
          'ALPHA_B':ALPHA_B_0,'ALPHA_B_0':ALPHA_B_0,'ALPHA_B_F':ALPHA_B_F,'frac_decrease_alpha_b': frac_decrease_alpha_b,
          'rho_min': rho_min, 'rho_max': rho_max, 'a_R':a_R,'f_factor':f_factor,
          'massfrac_star': massfrac_f, 'massfrac0': massfrac_0,
          'massfrac_f': massfrac_f, 'frac_decrease': frac_decrease, 'frac_decrease_p': frac_decrease_p,'frac_decrease_sy': frac_decrease_sy,
          'DEM_frac': DEM_frac,'Adjoint_frac': Adjoint_frac,
          'costfrac_star': costfrac_f, 'costfrac0': costfrac_f,
          'costfrac_f': costfrac_f, 'thick':thick, 'alpha':alpha_0,'alpha_0':alpha_0, 'alpha_step':alpha_step,
          'wc':wc_0,'wc_0':wc_0,'wc_f':wc_f,
          'frac_step_mass': frac_step, 'frac_step_cost': frac_step,
          'frac_step_p': frac_step}
    MP['DEM_ADJ_PENAL_P'] = False   # TO2: fixed p=3 to penalize DEM + Adj stiffness; density NN uses p=3->5 (mE2 unchanged)

    NN_config_disp = {'init_method': init_method, 'random_state': random_state, 'state': [],
                      'Example': Example, 'Case': Case,
                      'dynamic_weight': dynamic_weight, 'gradient_clip': gradient_clip,
                      'omega': omega, 'learning_rate_disp': learning_rate_disp,'learning_rate_adj': learning_rate_adj,
                      'learning_rate_rho': learning_rate_rho, 'nrmThreshold': nrmThreshold,
                      'TO_num_iter': TO_num_iter, 'plot_num': plot_num,
                      'plotting_interval_TO': plotting_interval_TO,
                      'delta': delta,'wc':wc_0, 'wd': wd, 'wm': wm, 'wp': wp,'wg':wg,
                      'kernel_size': kernel_size,
                      'basis': basis, 'quant_correlation_class': quant_correlation_class,
                      'activation': activation,
                      'n_features': n_features, 'n_cells': n_cells,
                      'res': res, 'NN_arch': NN_arch, 'save_folder': []}
    NN_config_disp_adj = copy.deepcopy(NN_config_disp)
    NN_config_disp_adj['n_cells'] = 3

    NN_config_rho = copy.deepcopy(NN_config_disp)
    NN_config_rho['basis'] = basis_rho
    if basis_rho == 'PGCAN':
        NN_config_rho['n_features'] = 128
        NN_config_rho['n_cells'] = 3
        NN_config_rho['res'] = [24, 24]
        n_neurons = int(NN_config_rho['n_features'] / 2)
        n_layers = 3
        NN_config_rho['NN_arch'] = [n_neurons] * n_layers
        NN_config_rho['kernel_size'] = (3, 3)   # match TO3
    else:
        NN_config_rho['n_features'] = None
        NN_config_rho['n_cells'] = None
        NN_config_rho['res'] = None
        n_neurons = 64
        n_layers = 6
        NN_config_rho['NN_arch'] = [n_neurons] * n_layers

    ############################### DDP setup ##############################################
    # Initialise the process group before building models so every rank starts
    # from an identical (seeded) replica of the networks.
    if world_size > 1:
        setup(rank, world_size)

    ############################### Generate Data ##############################################
    Training, mesh_data = get_data(MP)

    # --- DATA PREP: build one contiguous block of sub-slices per rank ---
    # Total slices = world_size * SUBSLICE_PER_RANK (N_SLICE, read at import).
    attach_slices_to_mesh_data(mesh_data, N_SLICE, axis=0)
    u_X_train = Training['u_X_train']
    v_X_train = Training['v_X_train']
    u_train = Training['u_train']
    v_train = Training['v_train']
    t1_X_train = Training['t1_X_train']
    t1_train = Training['t1_train']
    t2_X_train = Training['t2_X_train']
    t2_train = Training['t2_train']

    # create list phase_train: each element is a tuple (X_train, y_train) for a phase
    phase_train = []
    for i in range(num_phase):
        phase_X = Training[f'phase{i}_X_train']
        phase_y = Training[f'phase{i}_train']
        phase_train.append((phase_X, phase_y))

    # Generate random integers -- SEED first so every rank picks the SAME mesh
    # each epoch (all ranks must decompose the identical global mesh).
    np.random.seed(12345)
    index_CP = np.random.randint(1, num_CP + 1, size=TO_num_iter)

    # torchrun spawns the ranks; each runs its slice block, grads all-reduced.
    run_worker(MP, NN_config_disp, NN_config_disp_adj, NN_config_rho,
               u_X_train, u_train, t1_X_train, t1_train,
               v_X_train, v_train, t2_X_train, t2_train,
               phase_train,
               index_CP, mesh_data,
               rank=rank, world_size=world_size)

    cleanup()







