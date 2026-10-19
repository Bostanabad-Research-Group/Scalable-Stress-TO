"""
3D L_beam TO driver -- ANALYTICAL hard-constraints, NO GP.

Merged-lag + N_MASK with optional DDP. Displacement (u,v,w) and adjoint (t1,t2,t3)
satisfy their homogeneous Dirichlet BCs via field(x) = A(x)*NN(x); density satisfies
rho = 1 on the loading region via rho = (1 - A_rho) + A_rho*projectDensity(NN). A_u/
A_v/A_w use AFUNC (K_disp=50); A_rho uses AFUNC_RHO (sharper K_rho=500). All weights
are built ONCE in the loader and cached (A_rho sliced per slice). No Gaussian-process
correction anywhere; the LMGP objects are used only as NN containers.

Each step is a SINGLE sweep over N_SLICE element-disjoint slices: disp + adjoint +
density forwarded once over the ACTIVE (non-void) set; the adjoint body term uses the
PREVIOUS step's GLOBAL F_adj (cached per mesh, LAGGED); the active mask refreshes
every N_MASK steps. Knobs (top of file): N_SLICE, N_MASK, VOID_THRESH, TIMING_ON.
"""
import os
import json
import torch
import torch.distributed as dist
import copy
import numpy as np
import time
import shutil
from models.lmgp_updated import LMGP 
from tqdm import tqdm
from utils.utils_general import set_seed0, get_multiGPU, projectDensity
from utils.get_training_data_2D_updated import get_data_L_beam_3D as get_data
from utils.domain_slicing import (attach_slices_to_mesh_data,
                                  accumulate_merged_lag_nmask,
                                  _finalize_fadj)

N_DISP = 3
N_ADJ = 3
PHASE_OFFSET = N_DISP + N_ADJ

# Analytical hard-BC weight A(x) (0 on the fixed support surface, ->1 away): field = A(x)*NN(x).
# Read by get_data_L_beam_3D (adds A_u/A_v/A_w) and used by accumulate_merged_lag_nmask.
AFUNC = {'type': 'exp', 'k': 50.0}
AFUNC_RHO = {'type': 'exp', 'k': 500.0}   # SHARPER density weight A_rho (K_rho=500 vs K_disp=50)

# Each rank owns SUBSLICE_PER_RANK element-disjoint slices and accumulates over
# them locally. Total slices = world_size * SUBSLICE_PER_RANK. Keep the default
# at 4 to preserve the previous single-GPU/multiple-slice behavior.
SUBSLICE_PER_RANK = int(os.environ.get("SUBSLICE_PER_RANK", 1))
N_SLICE = int(os.environ.get("WORLD_SIZE", 1)) * SUBSLICE_PER_RANK

# RESUME-SAFETY: RESUME=1 resumes from Run_i/resume_ckpt.pth if present (else fresh start);
# a full-state checkpoint is atomically (over)written every RESUME_INTERVAL epochs.
RESUME = int(os.environ.get("RESUME", "0"))
RESUME_INTERVAL = int(os.environ.get("RESUME_INTERVAL", "1000"))

# VOID EXCLUSION: elements with density < VOID_THRESH are dropped from the element
# graphs each step (mass still summed over the active set). None -> exact.
VOID_THRESH = 1e-3

# N_MASK (Option B): refresh the active-element mask only every N_MASK steps (per
# mesh) and cache in between; density forward over active only. N_MASK=1 = exact.
N_MASK = 20

# ---- objective / adjoint-scaling knobs (hard-coded per run) ----
INCLUDE_BODY = True    # False -> comp only; True -> comp + ALPHA_B * 0.5*h^2 body term
ALPHA_B_0 = 1.0
ALPHA_B_F = 1.0
frac_decrease_alpha_b = 0.5  # fraction of epoch to increase alpha_b from alpha_b_0 to alpha_b_f
F_SCALE_GOAL = 4e6      # target max|F_adj| after scaling (adjoint force normalizer)  [C7 TO9: 1 x 4M elems]

# When True, log the per-element density change between consecutive iterations (max /
# avg / 95%-quantile of |rho_cur - rho_prev|) into timeHistory for an early-stop
# diagnostic. Costs one extra full-mesh density forward per epoch (rank 0 only); set
# False to skip it entirely for faster training.
difference_rho = False
difference_rho_skip = 100   # measure the density diff only every N epochs (cost / N); off-stride epochs store NaN

# --- minimum-search / best-design tracker ---------------------------------------
# For epochs >= MIN_SEARCH_ITER we keep the running MINIMUM design for six criteria.
# Set MIN_SEARCH_ITER < 0 to DISABLE the search entirely. (Mirrors Aug_15_2D.)
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
# NOTE (integration order): with REDUCED integration (single Gauss point, N_int=1)
# 'MEAN-over-IP > 1' and 'ANY-IP h>0' both collapse to the same per-element test
# svm0/sy > 1, so stress_viol_count_solid == n_h_activate_solid EXACTLY, and
# best4 == best6. They only diverge under FULL integration. The three ratios still
# differ (denominators), and n_h_activate (includes void) differs from n_h_activate_solid.


def _plot_A_weights(X, A_u, A_v, A_w, A_rho, example, fname, afunc_bc=None, afunc_rho=None):
    """3D scatter of the four analytical hard-constraint weights (A_u, A_v, A_w, A_rho)
    on ONE figure. Each is 0 on its constraint surface (fixed support / loading region)
    and -> 1 away. A_u = A_v = A_w (distance to support); A_rho is the sharper density
    weight. Nodes are subsampled for a fast, legible scatter. Headless-safe; the caller
    wraps this in try/except so it never crashes training."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    x = X[:, 0].detach().cpu().numpy(); y = X[:, 1].detach().cpu().numpy(); z = X[:, 2].detach().cpu().numpy()
    n = x.shape[0]
    max_pts = 20000
    idx = (np.random.default_rng(0).choice(n, size=max_pts, replace=False)
           if n > max_pts else np.arange(n))
    def _klabel(af):
        if not af: return ''
        tt = af.get('type', 'linear')
        return f" (exp K={float(af.get('k', 10.0)):g})" if tt == 'exp' else f" ({tt})"
    fig = plt.figure(figsize=(24, 6))
    panels = [(A_u, 'A_u', afunc_bc), (A_v, 'A_v', afunc_bc),
              (A_w, 'A_w', afunc_bc), (A_rho, 'A_rho', afunc_rho)]
    for k, (A, nm, af) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 4, k, projection='3d')
        a = A.detach().cpu().numpy()
        sc = ax.scatter(x[idx], y[idx], z[idx], c=a[idx], cmap='viridis',
                        vmin=0.0, vmax=1.0, s=3)
        fig.colorbar(sc, ax=ax, label=nm, shrink=0.6, ticks=np.linspace(0, 1, 6))
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'{example}: {nm}{_klabel(af)}')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[A_weights] saved {fname}", flush=True)


# ======================================================================
# TIMING INSTRUMENTATION  (this driver ONLY -- utils/
# is untouched at rest; we monkey-patch the domain_slicing helpers on the MODULE
# object at import). torch.cuda.synchronize() around each piece INFLATES the
# absolute total vs a normal run -- read PERCENTAGES / per-call counts.
# TIMING_ON=False -> run uninstrumented (production).
# ======================================================================
import time as _time
from collections import defaultdict as _dd
from contextlib import contextmanager as _cm
import utils.domain_slicing as _ds

TIMING_ON = False
_PH_T = _dd(float)
_PH_N = _dd(int)
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
    _ds._phase_fields         = _timed('density_fwd', _ds._phase_fields)
    _ds._disp_fields_analytic = _timed('disp_fwd',    _ds._disp_fields_analytic)
    _ds._adj_fields_analytic  = _timed('adj_fwd',     _ds._adj_fields_analytic)
    _ds._elem_strains         = _timed('strains',     _ds._elem_strains)
    _ds._restrict_to_active = _timed('restrict',     _ds._restrict_to_active)
    _ds._view_from_sub     = _timed('view_gather',   _ds._view_from_sub)
    _ds._assemble_fadj     = _timed('fadj_assemble', _ds._assemble_fadj)
    _ds._finalize_fadj     = _timed('fadj_assemble', _ds._finalize_fadj)
    _orig_backward = torch.Tensor.backward
    def _timed_backward(self, *a, **k):
        _sync(); _t0 = _time.perf_counter()
        r = _orig_backward(self, *a, **k)
        _sync(); _PH_T['backward'] += _time.perf_counter() - _t0; _PH_N['backward'] += 1
        return r
    torch.Tensor.backward = _timed_backward

# DENSITY is the analytical A_rho blend (built once in the loader, cached per slice,
# applied in domain_slicing._phase_fields); disp/adjoint are A(x)*NN. NO GP.

_TOP_PHASES = [("accumulate (nmask sweep)", "PHASE_accumulate"),
               ("optimizer step",           "PHASE_optim_step")]
_OP_PHASES  = [("disp_fwd  (_disp_fields_analytic)", "disp_fwd"),
               ("adj_fwd   (_adj_fields_analytic)",  "adj_fwd"),
               ("density_fwd(_phase_fields)", "density_fwd"),
               ("strains   (_elem_strains)",  "strains"),
               ("restrict  [geo, CACHE-2]",   "restrict"),
               ("view_gather[geo, CACHE-2]",  "view_gather"),
               ("fadj_assemble+finalize",     "fadj_assemble"),
               ("backward  (Tensor.backward)","backward")]
def _write_timing_report(f, total_time, n_epochs, n_slice):
    f.write("\n================ TIMING DECOMPOSITION (3D) ================\n")
    f.write(f"N_SLICE = {n_slice}   N_MASK = {N_MASK}   epochs = {n_epochs}   CUDA_sync = {_CUDA}\n")
    f.write("GPU-synchronized wall time accumulated over ALL epochs. Per-phase\n")
    f.write("cuda.synchronize() INFLATES the absolute total -- read PERCENTAGES and\n")
    f.write("per-call counts. restrict+view_gather = geometry rebuild (CACHE-2);\n")
    f.write("gp_covar = GP eval (CACHE-1). gp_mean_nn_X + backward = irreducible.\n")
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
    f.write(f"  {'gp_covar (CACHE-1)':<30s} {_PH_T.get('gp_covar',0.0):9.2f}s {100*_PH_T.get('gp_covar',0.0)/tot:6.2f}%\n")
    f.write(f"  {'loss-math + py (phases-ops)':<30s} {ph_sum-op_sum:9.2f}s  (inline stress/comp math + overhead)\n")
    f.write(f"  {'sum of top-level phases':<30s} {ph_sum:9.2f}s\n")
    f.write(f"  {'measured total_time':<30s} {total_time:9.2f}s\n")
    f.write("==========================================================\n")

def setup(rank, world_size, local_rank=0):
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size
    )
    torch.cuda.set_device(local_rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def reduce_grads(models):
    """Sum parameter gradients across ranks so every replica applies the same full-domain update."""
    for model in models:
        for prm in model.parameters():
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

    CHUNK = 20000  # element centroids per forward; lower if memory is still tight
    b = model_list[0].MP['b']
    rho_model = model_list[PHASE_OFFSET]

    with torch.no_grad():
        mass = elem_x.new_zeros(())
        for s in range(0, elem_x.shape[0], CHUNK):
            ex = elem_x[s:s + CHUNK]
            ev = elem_vol[s:s + CHUNK]
            p = projectDensity(rho_model.mean_module_NN_All(ex), b)   # [c, num_phase]
            A = A_rho_elem[s:s + CHUNK].unsqueeze(-1)                 # [c, 1]
            phase_weights = (1.0 - A) + A * p                        # blend in [0,1]
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
        last_bias = model_list[PHASE_OFFSET].mean_module_NN_All.network.last.bias
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
# Worker: single GPU or torchrun DDP
# ========================
def run_worker(MP, NN_config_disp, NN_config_disp_adj, NN_config_rho,
               u_X_train, u_train, t1_X_train, t1_train,
               v_X_train, v_train, t2_X_train, t2_train,
               w_X_train, w_train, t3_X_train, t3_train,
               phase_train, index_CP, mesh_data, train_data,
               rank=0, world_size=1):

    is_main = (rank == 0)
    device = torch.device("cuda")
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
    frac_decrease_alpha_b = MP['frac_decrease_alpha_b']
    massfrac_f = MP['massfrac_f']
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
        # save folder (rank 0 only touches the filesystem)
        save_folder = f"{run_folder}Run_{i}/"
        resume_path = save_folder + "resume_ckpt.pth"
        do_resume = (RESUME == 1) and os.path.exists(resume_path)
        if is_main:
            if do_resume:
                print(f"[resume] found {resume_path}; keeping folder and resuming", flush=True)
                os.makedirs(save_folder, exist_ok=True)
            else:
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
                       name_output="u", MP=MP, num_output=3).to(device)
        model_v = LMGP(v_X_train, v_train, NN_config_disp,
                       name_output="v", MP=MP, num_output=3).to(device)
        model_w = LMGP(w_X_train, w_train, NN_config_disp,
                       name_output="w", MP=MP, num_output=3).to(device)
        model_t1 = LMGP(t1_X_train, t1_train, NN_config_disp_adj,
                       name_output="t1", MP=MP, num_output=3).to(device)
        model_t2 = LMGP(t2_X_train, t2_train, NN_config_disp_adj,
                       name_output="t2", MP=MP, num_output=3).to(device)
        model_t3 = LMGP(t3_X_train, t3_train, NN_config_disp_adj,
                       name_output="t3", MP=MP, num_output=3).to(device)

        # One GP model per phase
        model_phases = []
        for i in range(num_phase):
            phase_X, phase_y = phase_train[i]   # from your phase_train list
            model_phase_i = LMGP(phase_X, phase_y, NN_config_rho,
                                name_output=f"rho", MP=MP, num_output=num_phase).to(device)
            model_phases.append(model_phase_i)

        # Keep all in list for BC handling
        model_list = [model_u, model_v, model_w, model_t1, model_t2, model_t3] + model_phases

        # define the time history dict
        timeHistory = {
            'loss_total': [], 'loss_obj': [], 'loss_dem': [],'loss_adj':[],
            'loss_mConstraint': [], 'loss_cConstraint': [], 'loss_tv': [], 
            'strain_energy': [], 'external_work': [], 'spring_energy':[],
            'strain_energy_adj': [], 'external_work_adj': [], 'body_energy_adj': [],'spring_energy_adj':[],
            'loss_pde_1': [], 'loss_pde_2': [],
            'massfrac': [], 'costfrac': [],
            'wc': [], 'wd': [], 'wm': [], 'wp': [],
            'grey': [], 'p': [],'b':[],'alpha':[], 'ALPHA_B':[],
            'stress_violation_1': [], 'stress_violation_2': [], 'stress_violation_3': [],
            'stress_viol_count_solid': [], 'solid_count': [],
            'n_h_activate': [], 'n_h_activate_solid': [], 'sy': [],
            'rho_diff_max': [], 'rho_diff_avg': [], 'rho_diff_q95': []
        }

        # ========================
        # Optimizers + schedulers
        # ========================
        optimizer_disp = torch.optim.Adam(model_list[0].parameters(),
                                          lr=learning_rate_disp, amsgrad=True)
        optimizer_adj = torch.optim.Adam(model_list[N_DISP].parameters(),
                                          lr=learning_rate_adj, amsgrad=True)
        optimizer_rho = torch.optim.Adam(model_list[PHASE_OFFSET].parameters(),
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
        
        # TO7: same as TO2 but with the density-NN lr scheduler restored.
        # 4 segments (3 steps), gamma=0.8 -> 8e-6, 6.4e-6, 5.12e-6. disp/adj lr stay constant.
        scheduler_rho = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer_rho,
                    milestones=[int(0.20 * TO_num_iter), int(0.35 * TO_num_iter), int(0.50 * TO_num_iter)],
                    gamma=0.8,
                )

        # ========================
        # Training loop
        # ========================
        epochs_iter = tqdm(range(TO_num_iter),
                           desc=f"rank {rank} TO Epoch",
                           position=rank, leave=is_main, disable=not is_main)
        total_time, start_time = 0, time.time()

        # --- GPU capacity report + reset peak-memory tracking (does N_SLICE fit?) ---
        _dev_idx = torch.cuda.current_device()
        _prop = torch.cuda.get_device_properties(_dev_idx)
        _cap_gb = _prop.total_memory / 1024**3
        print(f"[rank {rank}] GPU '{_prop.name}'  capacity = {_cap_gb:.1f} GiB  "
              f"(SUBSLICE_PER_RANK={SUBSLICE_PER_RANK}, N_SLICE={N_SLICE}, N_MASK={N_MASK})", flush=True)
        torch.cuda.reset_peak_memory_stats(_dev_idx)

        # Per-mesh caches: LAGGED F_adj (nodal, mesh-specific) + active-mask cache.
        fadj_cache = {}
        mask_cache_by_mesh = {}
        mask_age = {}

        # previous-epoch full-mesh density, for the difference_rho early-stop diagnostic
        # (None until the first epoch produces a design to compare against)
        rho_prev = None

        # snapshot helper: CPU copy of the 3 prediction NNs (u, adj, rho). Cloning a
        # small MLP state_dict is sub-ms; disk writes are deferred to after training.
        def _snapshot_pred_models():
            return {
                'model_u_state_dict':   {k: v.detach().cpu().clone() for k, v in model_list[0].mean_module_NN_All.state_dict().items()},
                'model_adj_state_dict': {k: v.detach().cpu().clone() for k, v in model_list[N_DISP].mean_module_NN_All.state_dict().items()},
                'model_rho_state_dict': {k: v.detach().cpu().clone() for k, v in model_list[PHASE_OFFSET].mean_module_NN_All.state_dict().items()},
            }

        # six best-design trackers (reset each run): running MINIMUM per criterion.
        best_trackers = {k: {'val': float('inf'), 'epoch': -1, 'state': None}
                         for k in ('best1','best2','best3','best4','best5','best6')}
        best_meaning = {'best1':'stress_violation_1','best2':'stress_violation_2',
                        'best3':'stress_violation_3','best4':'stress_viol_count_solid',
                        'best5':'n_h_activate','best6':'n_h_activate_solid'}
        # gray-area milestone snapshots: first epoch the gray fraction drops <= each target.
        gray_targets = [(0.05, 'gray5p'), (0.04, 'gray4p'), (0.03, 'gray3p'),
                        (0.02, 'gray2p'), (0.01, 'gray1p')]
        gray_states = {}   # tag -> (epoch, cpu state_dict)

        # ---- RESUME: restore full training state if a checkpoint was found ----
        start_epoch = 0
        if do_resume:
            _ck = torch.load(resume_path, map_location=device)
            model_list[0].mean_module_NN_All.load_state_dict(_ck['model_u_state_dict'])
            model_list[N_DISP].mean_module_NN_All.load_state_dict(_ck['model_adj_state_dict'])
            model_list[PHASE_OFFSET].mean_module_NN_All.load_state_dict(_ck['model_rho_state_dict'])
            optimizer_disp.load_state_dict(_ck['opt_disp'])
            optimizer_adj.load_state_dict(_ck['opt_adj'])
            optimizer_rho.load_state_dict(_ck['opt_rho'])
            scheduler_rho.load_state_dict(_ck['sched_rho'])
            model_list[0].MP.update(_ck['MP'])
            total_time = _ck.get('total_time', total_time)
            if _ck.get('timeHistory') is not None:
                timeHistory = _ck['timeHistory']
            if _ck.get('best_trackers') is not None:
                best_trackers = _ck['best_trackers']
            if _ck.get('gray_states') is not None:
                gray_states = _ck['gray_states']
            start_epoch = int(_ck['epoch']) + 1
            start_time = time.time()
            if is_main:
                print(f"[resume] loaded {resume_path}; resuming at epoch {start_epoch}/{TO_num_iter}", flush=True)
            epochs_iter = tqdm(range(start_epoch, TO_num_iter), initial=start_epoch, total=TO_num_iter,
                               desc=f"rank {rank} TO Epoch", position=rank, leave=is_main, disable=not is_main)
            if world_size > 1:
                dist.barrier()

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
            model_list[0].elem_vol      = mesh_data[GPU_key][mesh_key]["elem_vol"]
            model_list[0].f_index       = mesh_data[GPU_key][mesh_key]["f_index"]
            model_list[0].f_magnitude   = mesh_data[GPU_key][mesh_key]["f_magnitude"]
            model_list[0].conn          = mesh_data[GPU_key][mesh_key]["conn"]
            model_list[0].B             = mesh_data[GPU_key][mesh_key]["B"]
            model_list[0].detJ          = mesh_data[GPU_key][mesh_key]["detJ"]

            # visualize the analytical weights A_u, A_v, A_w, A_rho on ONE figure. The A
            # field is geometry-only and identical across the 5 runs, so write it ONCE
            # (first run, first epoch, rank 0) to the Case folder (run_folder), not per-run.
            # Never crashes training. File named by Case.
            if is_main and run_idx == 1 and epoch == 0:
                try:
                    _md = mesh_data[GPU_key][mesh_key]
                    _plot_A_weights(_md["X_node"], _md["A_u"], _md["A_v"], _md["A_w"],
                                    _md["A_rho_node"], MP['Example'],
                                    run_folder + f"A_weights_{Case}.pdf",
                                    afunc_bc=MP['AFUNC'], afunc_rho=MP.get('AFUNC_RHO', MP['AFUNC']))
                except Exception as e:
                    print(f"[A_weights] plot skipped: {e}", flush=True)

            mesh_train = train_data[index - 1]

            model_list[0].train_inputs = (mesh_train.train_inputs[0],)
            model_list[0].train_targets = mesh_train.train_targets[0]

            model_list[1].train_inputs = (mesh_train.train_inputs[1],)
            model_list[1].train_targets = mesh_train.train_targets[1]

            model_list[2].train_inputs = (mesh_train.train_inputs[2],)
            model_list[2].train_targets = mesh_train.train_targets[2]

            model_list[3].train_inputs = (mesh_train.train_inputs[3],)
            model_list[3].train_targets = mesh_train.train_targets[3]

            model_list[4].train_inputs = (mesh_train.train_inputs[4],)
            model_list[4].train_targets = mesh_train.train_targets[4]

            model_list[5].train_inputs = (mesh_train.train_inputs[5],)
            model_list[5].train_targets = mesh_train.train_targets[5]

            for p in range(num_phase):
                model_list[PHASE_OFFSET + p].train_inputs = (mesh_train.train_inputs[6 + p],)
                model_list[PHASE_OFFSET + p].train_targets = mesh_train.train_targets[6 + p]

            
            # offset the last layer to change the initial massfrac
            if epoch == 0:
                rho_last_bias = model_list[PHASE_OFFSET].mean_module_NN_All.network.last.bias
                rho_bias_before_shift = rho_last_bias.detach().clone()
                mf_final, shift = adjust_density_bias_bisection(
                                        model_list,
                                        target_massfrac=MP['massfrac0'],
                                        tol=1e-4,
                                        max_iter=50,
                                    )
                if world_size > 1:
                    shift_tensor = torch.tensor(
                        0.0 if shift is None else float(shift), device=device
                    )
                    found_tensor = torch.tensor(0 if shift is None else 1, device=device)
                    mf_tensor = torch.tensor(float(mf_final), device=device)
                    dist.broadcast(shift_tensor, src=0)
                    dist.broadcast(found_tensor, src=0)
                    dist.broadcast(mf_tensor, src=0)
                    shift = None if int(found_tensor.item()) == 0 else float(shift_tensor.item())
                    mf_final = float(mf_tensor.item())
                    with torch.no_grad():
                        if shift is None:
                            rho_last_bias.copy_(rho_bias_before_shift)
                        else:
                            rho_last_bias.copy_(rho_bias_before_shift + shift)
                if shift is None:
                    if is_main:
                        print(f"Failed to bracket target. Current mass fraction = {mf_final:.6f}")
                else:
                    if is_main:
                        print(f"Adjusted mass fraction: {mf_final:.6f}, bias shift: {shift:.6f}")
                # record the bias shift (persisted in the results summary, like 2D)
                model_list[0].MP['initial_bias_shift'] = shift
                model_list[0].MP['initial_massfrac_shift'] = mf_final
                MP['initial_bias_shift'] = shift
                MP['initial_massfrac_shift'] = mf_final
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
            # OPTION 1 (3D): MERGED single-displacement-forward, LAGGED F_adj,
            # + N_MASK density-restriction. Single sweep:
            # disp + adjoint + density each forwarded ONCE over active elements;
            # the adjoint body term uses the PREVIOUS F_adj (cached per mesh); a
            # fresh (detached) F_adj is assembled for the next step. Peak autograd
            # memory ~ one slice's active elements.
            # ============================================================
            all_slices = mesh_data[GPU_key][mesh_key]["slices"]
            k = SUBSLICE_PER_RANK
            my_slices = all_slices[rank * k:(rank + 1) * k]
            Nn_mesh = mesh_data[GPU_key][mesh_key]["X_node"].shape[0]
            # LAGGED GLOBAL F_adj for this mesh (zeros first time; 3 DOF -> [Nn,3])
            F_adj_lag, alpha_lag = fadj_cache.get(
                mesh_key,
                (torch.zeros(Nn_mesh, 3, device=device),
                 torch.tensor(1.0, device=device)),
            )
            refresh = (mesh_key not in mask_cache_by_mesh) or \
                      (epoch - mask_age[mesh_key] >= N_MASK)
            mc = None if refresh else mask_cache_by_mesh[mesh_key]

            # --- per-epoch compute+comm timer (synced); should DECREASE as void
            #     exclusion shrinks the active set. Covers forwards, backward, and
            #     (multi-GPU) the cross-rank all-reduces + optimizer step. ---
            _sync(); _ep_t0 = _time.perf_counter()

            # NO GP: density is the analytical A_rho blend (cached per slice),
            # disp/adjoint are A(x)*NN. Straight into the accumulate sweep.
            with _phase('PHASE_accumulate'):
                log, F_adj_raw, new_masks = accumulate_merged_lag_nmask(
                    model_list, my_slices,
                    wd=wd, wc=wc, wm=wm, delta=delta,
                    include_adj=include_adj, include_obj_mass=include_obj_mass,
                    F_adj_scaled_lag=F_adj_lag, alpha_lag=alpha_lag,
                    void_thresh=VOID_THRESH, mask_cache=mc, return_raw_fadj=True,
                )

            if refresh:
                mask_cache_by_mesh[mesh_key] = new_masks
                mask_age[mesh_key] = epoch

            # Assemble the GLOBAL F_adj for the next lag: all-reduce each rank's
            # raw partial force, then finalize using the same global normalizer.
            if world_size > 1:
                dist.all_reduce(F_adj_raw, op=dist.ReduceOp.SUM)
            F_adj_scaled_new, alpha_new = _finalize_fadj(F_adj_raw, Nn_mesh, F_SCALE_GOAL)
            fadj_cache[mesh_key] = (F_adj_scaled_new.detach(), alpha_new.detach())
            alpha_val = float(alpha_new.item())

            # Sum gradients across ranks -> full-domain gradient on every replica.
            if world_size > 1:
                reduce_grads([model_list[0], model_list[N_DISP], model_list[PHASE_OFFSET]])

            # Reduce scalar diagnostics so rank 0 logs GLOBAL quantities.
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
            g_count = grey[-2]
            g_active = float(grey[-1].item())
            grey_fraction = (grey[:-2] / torch.clamp(g_count, min=1)).cpu().tolist()
            # --- three stress-violation ratios (see header for definitions) ---
            N_elem = float(g_count.item())                   # total elements (all-reduced count)
            massfrac_N = massfrac * N_elem                   # ~ expected # solids (denom of viol_2/3)
            stress_violation_1 = (g_viol / g_solid) if g_solid > 0 else 0.0
            stress_violation_2 = (g_hact / massfrac_N) if massfrac_N > 0 else 0.0
            stress_violation_3 = (g_hact_solid / massfrac_N) if massfrac_N > 0 else 0.0
            # --- running-MINIMUM design for all six criteria (rank 0; NNs identical
            #     across ranks). One snapshot per epoch, shared across improved criteria. ---
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
            # --- gray-area milestone capture: first epoch gray fraction hits each target ---
            if is_main:
                gray_val = float(sum(grey_fraction))
                for _g_thr, _g_tag in gray_targets:
                    if _g_tag not in gray_states and gray_val <= _g_thr:
                        gray_states[_g_tag] = (epoch, _snapshot_pred_models())

            # --- one-time schedule setup (unchanged, now sourced from log) ---
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

            # --- gradient clip on density NN + optimizer steps ---
            # (grads already accumulated inside accumulate_merged_lag_nmask; the loop
            #  top already zero_grad-ed all three optimizers.)
            with _phase('PHASE_optim_step'):
                torch.nn.utils.clip_grad_norm_(model_list[0].parameters(), nrmThreshold)
                torch.nn.utils.clip_grad_norm_(model_list[N_DISP].parameters(), nrmThreshold)
                torch.nn.utils.clip_grad_norm_(model_list[PHASE_OFFSET].parameters(), nrmThreshold/10)
                optimizer_disp.step()
                optimizer_adj.step()
                optimizer_rho.step()
                scheduler_rho.step()   # TO7: density-NN lr scheduler restored (vs TO2)

            # --- per-epoch time (synced) + active count; rank 0 logs to timeHistory ---
            _sync(); _ep_dt = _time.perf_counter() - _ep_t0
            if is_main:
                timeHistory.setdefault('epoch_time', []).append(_ep_dt)
                if epoch < 5 or (epoch + 1) % 200 == 0:
                    _c = float(g_count.item()) if torch.is_tensor(g_count) else float(g_count)
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
                timeHistory['p'].append(model_list[0].MP['p'])
                timeHistory['sy'].append(model_list[0].MP['sy'])
                timeHistory['ALPHA_B'].append(model_list[0].MP['ALPHA_B'])
                timeHistory['b'].append(model_list[0].MP['b'])
                timeHistory['alpha'].append(alpha_val)
                timeHistory['stress_violation_1'].append(stress_violation_1)
                timeHistory['stress_violation_2'].append(stress_violation_2)
                timeHistory['stress_violation_3'].append(stress_violation_3)
                timeHistory['stress_viol_count_solid'].append(g_viol)
                timeHistory['solid_count'].append(g_solid)
                timeHistory['n_h_activate'].append(g_hact)
                timeHistory['n_h_activate_solid'].append(g_hact_solid)
                timeHistory.setdefault('active_total', []).append(g_active)
                timeHistory.setdefault('count_total', []).append(float(g_count.item()))
                # --- rho-change early-stop diagnostic: compare the current full-mesh
                #     density to the previous epoch's and log max / avg / 95%-quantile of
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

            # --- early peak-memory readout (peak is reached in the first epochs;
            #     watch this to pick N_SLICE without waiting for the full run) ---
            if (epoch + 1) % 200 == 0 or epoch == 50:
                _pk = torch.cuda.max_memory_reserved(_dev_idx) / 1024**3
                print(f"[rank {rank} epoch {epoch+1}] peak GPU reserved so far: {_pk:.2f} GiB "
                      f"/ {_cap_gb:.1f} GiB ({100*_pk/_cap_gb:.0f}%)  "
                      f"SUBSLICE_PER_RANK={SUBSLICE_PER_RANK}", flush=True)

            if is_main and (epoch+1) % plotting_interval_TO == 0:
                end_time = time.time()
                torch.save(
                    {
                        'model_u_state_dict': model_list[0].mean_module_NN_All.state_dict(),
                        'model_adj_state_dict': model_list[N_DISP].mean_module_NN_All.state_dict(),
                        'model_rho_state_dict': model_list[PHASE_OFFSET].mean_module_NN_All.state_dict(),
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

            # ---- RESUME: periodic full-state checkpoint (atomic overwrite) ----
            if is_main and RESUME == 1 and (epoch + 1) % RESUME_INTERVAL == 0:
                _rtmp = resume_path + ".tmp"
                torch.save({
                    'epoch': epoch,
                    'model_u_state_dict':   model_list[0].mean_module_NN_All.state_dict(),
                    'model_adj_state_dict': model_list[N_DISP].mean_module_NN_All.state_dict(),
                    'model_rho_state_dict': model_list[PHASE_OFFSET].mean_module_NN_All.state_dict(),
                    'opt_disp': optimizer_disp.state_dict(),
                    'opt_adj':  optimizer_adj.state_dict(),
                    'opt_rho':  optimizer_rho.state_dict(),
                    'sched_rho': scheduler_rho.state_dict(),
                    'MP': model_list[0].MP,
                    'total_time': total_time,
                    'timeHistory': timeHistory,
                    'best_trackers': best_trackers,
                    'gray_states': gray_states,
                }, _rtmp)
                os.replace(_rtmp, resume_path)
                print(f"[resume] saved checkpoint at epoch {epoch + 1} -> {resume_path}", flush=True)

        end_time = time.time()
        total_time += (end_time - start_time)

        # --- peak GPU memory (does N_SLICE fit the GPU? tune N_SLICE from this) ---
        _peak_alloc = torch.cuda.max_memory_allocated(_dev_idx) / 1024**3
        _peak_resv = torch.cuda.max_memory_reserved(_dev_idx) / 1024**3
        print(f"[rank {rank}] peak GPU mem: allocated {_peak_alloc:.2f} GiB, "
              f"reserved {_peak_resv:.2f} GiB / {_cap_gb:.1f} GiB "
              f"({100*_peak_resv/_cap_gb:.0f}% reserved)", flush=True)
        _peak_glob = torch.tensor(_peak_resv, device=device)
        if world_size > 1:
            dist.all_reduce(_peak_glob, op=dist.ReduceOp.MAX)
        _peak_glob = float(_peak_glob.item())

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

        train_data_to_save = {}
        for mesh_id, mesh_train in train_data.items():
            train_data_to_save[mesh_id] = {
                "gpu_key": mesh_train.gpu_key,
                "mesh_key": mesh_train.mesh_key,
                "train_inputs": mesh_train.train_inputs,
                "train_targets": mesh_train.train_targets,
            }

        torch.save({
            "u_X_train": u_X_train, "u_train": u_train,
            "v_X_train": v_X_train, "v_train": v_train,
            "w_X_train": w_X_train, "w_train": w_train,
            "t1_X_train": t1_X_train, "t1_train": t1_train,
            "t2_X_train": t2_X_train, "t2_train": t2_train,
            "t3_X_train": t3_X_train, "t3_train": t3_train,
            "phase_train": phase_train,
            "train_data": train_data_to_save,
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
                    f"peak reserved max across ranks: {_peak_glob:.2f} GiB ({100*_peak_glob/_cap_gb:.0f}%)  "
                    f"[world_size={world_size}, SUBSLICE_PER_RANK={SUBSLICE_PER_RANK}, N_SLICE={N_SLICE}]\n")
            f.write(f"Final value strain energy: {timeHistory['strain_energy'][-1]}\n")
            f.write(f"Final value external work: {timeHistory['external_work'][-1]}\n")
            f.write(f"Random seed (state): {state}\n")
            f.write(f"omega (raw lengthscale): {NN_config_disp.get('omega', 'n/a')}\n")
            f.write(f"Initial density bias shift: {MP.get('initial_bias_shift', 'n/a')}\n")
            f.write(f"Initial mass fraction after shift: {MP.get('initial_massfrac_shift', 'n/a')}\n")
            _write_timing_report(f, total_time, len(timeHistory['loss_total']), N_SLICE)

# ========================
# Launch block (torchrun)
# ========================
if __name__ == "__main__":
    # Try to get ranks from torchrun, else default to single-GPU debug mode
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    ############################### DDP setup ##############################################
    # Initialise before data/model tensors are created so torch.device("cuda") maps
    # to the correct local rank under torchrun.
    if world_size > 1:
        setup(rank, world_size, local_rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    ############################### Define Parameters ##############################################
    base_folder = f"/home/alexsunuci/Aug_21_3D"   # Sophia
    # base_folder = f"/u/xsun7/Aug_21_3D"   # Delta
    # base_folder = os.getcwd()
    random_state = [11,13]   # 2 seeds
    num_CP = 1
    Example = 'L_beam_3D'
    Case = 'C7_L_beam_TO9_Sophia'   # 3D analytical hard-BC (A*NN), merged-lag + N_mask
    Diff_type = 'C3D8R_4M'  # loads Data/L_beam_3D_GPU1_C3D8R_4M.pt
    N_worker = 1  # for local debugging with 1 GPU
    # define material properties and
    # wt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # full: 8 points × weight 1
    wt = [8.0]
    D = [1.0]  # density
    E = [7.0]  # modulus
    P = [1.0]  # cost
    nu = 0.25  # Poisson's ratio
    p0 = 3.0  # initial penalty
    pf = 5.0  # final penalty
    sy0 = 0.01   # initial yield stress
    syf = 0.009   # final yield stress
    f_factor = 0.2
    epsilon = 0.2 # relaxation
    a_R = 1 
    massfrac_0 = 0.6  # initial mass fraction
    massfrac_f = 0.3  # final mass fraction
    costfrac_f = 0.6  # final mass fraction
    frac_decrease = 0.5  # fraction of epoch to decrease volume fraction from VF0 to VF_f
    frac_decrease_p = 0.5  # fraction of epoch to decrease volume fraction from VF0 to VF_f
    frac_decrease_sy = 0.5  # fraction of epoch to decrease sy from sy0 to syf

    DEM_frac = 0.05 # epoch fraction of DEM for primary displacements
    Adjoint_frac = 0.05 # epoch fraction of adjoint problems
    b_0 = 4  # sharpness parameters
    b_f = 8  # sharpness parameters
    thres_static = 0.5  # static threshold values to binarize the density field
    rho_min = 0.1  # lower limit to define grey element (rho_min,rho_max)
    rho_max = 0.9  # upper limit to define grey element (rho_min,rho_max)

    # define domain
    Nelx = 100
    Nely = 100
    Nelz = 10
    xmin, xmax = 0.0, 100.0
    ymin, ymax = 0.0, 100.0
    zmin, zmax = 0.0, 10.0
    num_bc = 1000
    num_load = 50

    # define model parameters and training parameters
    init_method = 'kaiming_uniform_'
    dynamic_weight = False
    gradient_clip = False
    omega = 0.0
    learning_rate_disp = 1e-3
    learning_rate_adj = 1e-2
    learning_rate_rho = 1e-5
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
    basis = 'PGCAN3D'
    basis_rho = 'PGCAN3D'
    quant_correlation_class = 'Rough_RBF'
    activation = 'tanh'
    if basis == 'PGCAN3D':
        n_features = 128
        n_cells = 3
        res = [36, 36, 5]
        n_neurons = int(n_features / 2)
        n_layers = 3
        NN_arch = [n_neurons] * n_layers
        groups = n_features
        kernel_size = (3, 3, 3)
    else:
        n_features = None
        n_cells = None
        res = None
        n_neurons = 64
        n_layers = 6
        NN_arch = [n_neurons] * n_layers
        groups = None
        kernel_size = None

    ############################### Pre-processing ##############################################
    num_phase = len(D)
    tkwargs = get_multiGPU(N_worker)
    D_tensor = [torch.tensor(D, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    E_tensor = [torch.tensor(E, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    P_tensor = [torch.tensor(P, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    wt_tensor = [torch.tensor(wt, dtype=tkwargs["dtype"], device=device) for device in tkwargs['devices']]
    domain = {'x': [xmin, xmax], 'y': [ymin, ymax], 'z': [zmin, zmax]}
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
          'base_folder': base_folder, 'domain': domain,
          'AFUNC': AFUNC,   # analytical hard-BC weight -> triggers A*NN fields (get_data + accumulate)
          'AFUNC_RHO': AFUNC_RHO,   # sharper density weight A_rho (K_rho=500)
          'INCLUDE_BODY': INCLUDE_BODY, 'F_scale_goal': F_SCALE_GOAL,
          'thres_static': thres_static, 'Diff_type': Diff_type,
          'num_phase': num_phase, 'Nelx': Nelx, 'Nely': Nely,'Nelz':Nelz,
          'D': D_tensor, 'E': E_tensor, 'P': P_tensor, 'wt':wt_tensor,'nu': nu,
          'p': p0, 'p0': p0, 'pf': pf, 'b': b_0,'b_0': b_0,'b_f': b_f, 'sy':sy0,'sy0':sy0,'syf':syf,'epsilon':epsilon,
          'ALPHA_B':ALPHA_B_0,'ALPHA_B_0':ALPHA_B_0,'ALPHA_B_F':ALPHA_B_F,'frac_decrease_alpha_b': frac_decrease_alpha_b,
          'rho_min': rho_min, 'rho_max': rho_max, 'a_R':a_R,'f_factor':f_factor,
          'massfrac_star': massfrac_f, 'massfrac0': massfrac_0,
          'massfrac_f': massfrac_f, 'frac_decrease': frac_decrease, 'frac_decrease_p': frac_decrease_p,'frac_decrease_sy': frac_decrease_sy,
          'DEM_frac': DEM_frac,'Adjoint_frac': Adjoint_frac,
          'costfrac_star': costfrac_f, 'costfrac0': costfrac_f,
          'costfrac_f': costfrac_f, 'alpha':alpha_0,'alpha_0':alpha_0, 'alpha_step':alpha_step,
          'wc':wc_0,'wc_0':wc_0,'wc_f':wc_f,
          'frac_step_mass': frac_step, 'frac_step_cost': frac_step,
          'frac_step_p': frac_step}

    NN_config_disp = {'init_method': init_method, 'random_state': random_state, 'state': [],
                      'Example': Example, 'Case': Case,
                      'dynamic_weight': dynamic_weight, 'gradient_clip': gradient_clip,
                      'omega': omega, 'learning_rate_disp': learning_rate_disp,'learning_rate_adj': learning_rate_adj,
                      'learning_rate_rho': learning_rate_rho, 'nrmThreshold': nrmThreshold,
                      'TO_num_iter': TO_num_iter, 'plot_num': plot_num,
                      'plotting_interval_TO': plotting_interval_TO,
                      'delta': delta,'wc':wc_0, 'wd': wd, 'wm': wm, 'wp': wp,'wg':wg,
                      'groups': groups,
                      'kernel_size': kernel_size,
                      'basis': basis, 'quant_correlation_class': quant_correlation_class,
                      'activation': activation,
                      'n_features': n_features, 'n_cells': n_cells,
                      'res': res, 'NN_arch': NN_arch, 'save_folder': []}
    NN_config_disp_adj = copy.deepcopy(NN_config_disp)
    NN_config_disp_adj['n_cells'] = 3

    NN_config_rho = copy.deepcopy(NN_config_disp)
    NN_config_rho['basis'] = basis_rho
    if basis_rho == 'PGCAN3D':
        NN_config_rho['n_features'] = 128
        NN_config_rho['n_cells'] = 3
        NN_config_rho['res'] = [36, 36, 5]
        NN_config_rho['groups'] = NN_config_rho['n_features']
        n_neurons = int(NN_config_rho['n_features'] / 2)
        n_layers = 3
        NN_config_rho['NN_arch'] = [n_neurons] * n_layers
        NN_config_rho['kernel_size'] = (3, 3, 3)
    else:
        NN_config_rho['n_features'] = None
        NN_config_rho['n_cells'] = None
        NN_config_rho['res'] = None
        NN_config_rho['groups'] = None
        n_neurons = 64
        n_layers = 6
        NN_config_rho['NN_arch'] = [n_neurons] * n_layers

    ############################### Generate Data ##############################################
    Training, mesh_data, train_data = get_data(MP)

    # --- DATA PREP: build element-disjoint slices for every mesh (once) ---
    # Total slices = world_size * SUBSLICE_PER_RANK; each rank owns one block.
    attach_slices_to_mesh_data(mesh_data, N_SLICE, axis=0)
    u_X_train = Training['u_X_train']
    v_X_train = Training['v_X_train']
    w_X_train = Training['w_X_train']
    u_train = Training['u_train']
    v_train = Training['v_train']
    w_train = Training['w_train']
    t1_X_train = Training['t1_X_train']
    t1_train = Training['t1_train']
    t2_X_train = Training['t2_X_train']
    t2_train = Training['t2_train']
    t3_X_train = Training['t3_X_train']
    t3_train = Training['t3_train']

    # create list phase_train: each element is a tuple (X_train, y_train) for a phase
    phase_train = []
    for i in range(num_phase):
        phase_X = Training[f'phase{i}_X_train']
        phase_y = Training[f'phase{i}_train']
        phase_train.append((phase_X, phase_y))

    # Generate random integers -- every rank must choose the same mesh per epoch.
    np.random.seed(12345)
    index_CP = np.random.randint(1, num_CP + 1, size=TO_num_iter)

    # torchrun handles ranks; each rank trains on its slice block, then all-reduces.
    run_worker(MP, NN_config_disp, NN_config_disp_adj, NN_config_rho,
               u_X_train, u_train, t1_X_train, t1_train,
               v_X_train, v_train, t2_X_train, t2_train,
               w_X_train, w_train, t3_X_train, t3_train,
               phase_train,
               index_CP, mesh_data, train_data,
               rank=rank, world_size=world_size)

    cleanup()
