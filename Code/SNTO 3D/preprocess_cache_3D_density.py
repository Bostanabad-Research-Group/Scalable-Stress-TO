"""
preprocess_cache_3D_density.py  (TO env -- needs torch)
=======================================================
Step 1 of the 3D FEniCS / smooth-remesh stress pipeline -- the 3D analogue of
Aug_15_2D/preprocess_cache_2D_density.py.

For each (case folder x run x selection) it reconstructs, from the trained density
NN, everything the downstream FE re-solve + plotting need, and writes a plain
NumPy cache:

    <run>/density_field_<tag>.npz  with
        points          [Nn,3]  node coordinates
        conn            [Ne,8]  C3D8 hex connectivity (Abaqus/VTK node order)
        rho             [Ne]    density  projectDensity(NN)  (+ GP corr if design_flag)
        support_u_nodes [nu]    node ids where u is fixed (A_u ~ 0)   -- component-wise
        support_v_nodes [nv]    node ids where v is fixed (A_v ~ 0)
        support_w_nodes [nw]    node ids where w is fixed (A_w ~ 0)   (differ for mixed BC)
        load_nodes      [nl]    node ids where the load is applied (= f_index)
        load_forces     [nl,3]  nodal force vectors (= f_magnitude * f_factor)
        E, nu, sy               material + stress limit (scalars)
        thick                   plate thickness if any (else 1.0)
        example                 example name (string)

The density is the SAME projectDensity(NN) the existing 3D plotters use
(plot_3D_density / plot_3D_stress) -- NO analytical A_rho blend -- so this cache is
a drop-in for those and for plot_3D_stress_FEniCS_thres.py. The support/load node
coordinates let the FE re-solve (and the FEM1-3 tet pipeline) re-impose the exact TO
boundary conditions by coordinate matching, so no per-example geometric markers are
needed (works for L_beam, Portal_frame_half, ...).

Runs torch-only (never imports pyvista) so it is safe on macOS.

Run:  /opt/anaconda3/envs/TO/bin/python preprocess_cache_3D_density.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import glob
import json
import numpy as np

# =========================================================================
#                              PARAMETERS
# =========================================================================
EXAMPLES = [
    'Results/L_beam_3D/C2_L_beam_TO10_Sophia',
    'Results/L_beam_3D/C2_L_beam_TO11_Sophia',
    'Results/Portal_frame_half_3D/C2_Portal_frame_half_TO10_Sophia',
    'Results/Portal_frame_half_3D/C2_Portal_frame_half_TO11_Sophia',
]
RUNS       = [1, 2, 3, 4, 5, 6]
RUNS       = [1,2,3,4]

SELECTIONS = ['iter:29999', 'best:1', 'best:2', 'best:3',
              'gray:1', 'gray:2', 'gray:3', 'gray:4', 'gray:5']
SELECTIONS = ['iter:29999', 'best:1']
MESH_KEY    = 'Mesh_01'
MODE        = ''        # '' or 'pred' (uses the *_pred.pt mesh)
A_TOL       = 1e-6      # support = nodes where the disp weight A_u/A_v/A_w <= A_TOL
RECON_CHUNK = 50000     # elements per density-NN chunk (lower if RAM-tight)
FORCE       = False      # True -> re-write caches even if they already exist
# =========================================================================


def resolve_selection(run_folder, sel):
    """(checkpoint_path or None, tag). Mirrors the 2D cache naming; best/gray globbed."""
    kind, _, val = sel.partition(':')
    kind = kind.strip().lower(); val = val.strip()
    if kind == 'iter':
        return os.path.join(run_folder, f'Trained_models_{val}.pth'), str(val)
    if kind == 'best':
        pat = os.path.join(run_folder, f'Trained_models_best{val}_*.pth'); stem = f'best{val}'
    elif kind == 'gray':
        pat = os.path.join(run_folder, f'Trained_models_gray{val}p_*.pth'); stem = f'gray{val}p'
    else:
        raise ValueError(f'unknown selection kind: {sel!r}')
    matches = sorted(glob.glob(pat),
                     key=lambda p: int(p.rsplit('_', 1)[1].split('.')[0]))
    if not matches:
        return None, stem
    ckpt = matches[-1]
    tag = os.path.basename(ckpt)[len('Trained_models_'):-len('.pth')]
    return ckpt, tag


def reconstruct(run_folder, ckpt_path):
    """torch-only: rebuild the 3D density (projectDensity(NN), + GP correction when the
    run used design_flag -- exactly as plot_3D_density/_stress do) plus the mesh, the
    component-wise support node sets (A_u/A_v/A_w ~ 0) and the nodal load, and return a
    dict of numpy arrays for the .npz."""
    import torch
    from models.lmgp_updated import LMGP
    from utils.utils_general import projectDensity
    from utils.get_training_data_2D_updated import analytic_weights_for_mesh

    MP = torch.load(run_folder + 'MP.pt', map_location='cpu')
    Example = MP['Example']; Diff_type = MP['Diff_type']
    num_phase = MP['num_phase']; b = MP['b']
    design_flag = MP.get('design_flag', False)

    with open(run_folder + 'NN_config_rho.json') as fh:
        NN_config_rho = json.load(fh)
    Training = torch.load(run_folder + 'Training.pth', map_location='cpu')
    phase_train = Training['phase_train']

    model_phases = [LMGP(phase_train[i][0], phase_train[i][1], NN_config_rho,
                         name_output='rho', MP=MP, num_output=num_phase)
                    for i in range(num_phase)]
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_phases[0].mean_module_NN_All.load_state_dict(ckpt['model_rho_state_dict'])
    rho_NN = model_phases[0].mean_module_NN_All

    file_loc = 'Data/' + f'{Example}_GPU1_{Diff_type}' + ('_pred' if MODE == 'pred' else '') + '.pt'
    md = torch.load(file_loc, map_location='cpu')['GPU0'][MESH_KEY]
    X_node = md['X_node'].detach(); X_elem = md['X_elem'].detach()
    conn = md['conn'].detach()
    f_index = md['f_index'].detach(); f_magnitude = md['f_magnitude'].detach()

    # --- density GP correction pieces (only if this run used design_flag) ---
    K_inv = None
    if design_flag:
        from gpytorch.settings import cholesky_jitter
        K_inv = []
        with torch.no_grad(), cholesky_jitter(1e-4):
            for i in range(num_phase):
                m = model_phases[i]
                chol = m.covar_module(m.train_inputs[0]).cholesky()
                resid = m.train_targets.unsqueeze(-1) - rho_NN(m.train_inputs[0])[:, i].unsqueeze(-1)
                K_inv.append(chol._cholesky_solve(resid))

    # --- density projectDensity(NN), chunked over elements (matches plot_3D_density) ---
    N = X_elem.shape[0]; rho_parts = []
    with torch.no_grad():
        for start in range(0, N, RECON_CHUNK):
            xe = X_elem[start:start + RECON_CHUNK]
            m_phase = rho_NN(xe)
            if design_flag:
                cols = [m_phase[:, i].unsqueeze(-1)
                        + model_phases[i].covar_module(model_phases[i].train_inputs[0], xe).evaluate().t() @ K_inv[i]
                        for i in range(num_phase)]
                pw = projectDensity(torch.cat(cols, dim=1), b)
            else:
                pw = projectDensity(m_phase, b)
            rho_parts.append(pw[:, 0].cpu())
    rho = torch.cat(rho_parts).numpy()

    # --- analytical hard-BC weights A_u, A_v, A_w (component-wise support) ---
    _pts, A_u, A_v, A_w = analytic_weights_for_mesh(X_node, MP)
    A_u_np = A_u.detach().cpu().numpy().ravel()
    A_v_np = A_v.detach().cpu().numpy().ravel()
    A_w_np = A_w.detach().cpu().numpy().ravel()

    f_factor = float(MP['f_factor'])
    E = float(MP['E'][0]) if isinstance(MP['E'], (list, tuple)) else float(MP['E'])
    nu = float(MP['nu']); sy = float(MP['sy'])
    thick = MP.get('thick', None)
    thick = 1.0 if thick is None else float(thick)

    return dict(
        points=X_node.cpu().numpy()[:, :3].astype(np.float64),
        conn=conn.cpu().numpy().astype(np.int64),
        rho=rho.astype(np.float64),
        support_u_nodes=np.where(A_u_np <= A_TOL)[0].astype(np.int64),
        support_v_nodes=np.where(A_v_np <= A_TOL)[0].astype(np.int64),
        support_w_nodes=np.where(A_w_np <= A_TOL)[0].astype(np.int64),
        load_nodes=f_index.cpu().numpy().astype(np.int64).ravel(),
        load_forces=(f_magnitude.cpu().numpy().reshape(-1, 3) * f_factor).astype(np.float64),
        E=np.float64(E), nu=np.float64(nu), sy=np.float64(sy), thick=np.float64(thick),
        example=np.array(Example),
    )


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for case in EXAMPLES:
        for run in RUNS:
            run_folder = os.path.join(here, case, f'Run_{run}') + os.sep
            if not os.path.isdir(run_folder):
                print(f'[skip] no run folder: {run_folder}'); continue
            for sel in SELECTIONS:
                ckpt, tag = resolve_selection(run_folder, sel)
                out = os.path.join(run_folder, f'density_field_{tag}.npz')
                if ckpt is None or not os.path.isfile(ckpt):
                    print(f'[skip] {case} Run_{run} {sel}: checkpoint not found'); continue
                if os.path.isfile(out) and not FORCE:
                    print(f'[cache] exists: {out}'); continue
                print(f'[recon] {case} Run_{run} {sel} -> {os.path.basename(out)}', flush=True)
                try:
                    d = reconstruct(run_folder, ckpt)
                except Exception as e:
                    print(f'  [error] {e}'); continue
                np.savez_compressed(out, **d)
                print(f'  wrote {out}  ({d["conn"].shape[0]} elems, '
                      f'{d["support_u_nodes"].size}/{d["support_v_nodes"].size}/'
                      f'{d["support_w_nodes"].size} u/v/w-support, '
                      f'{d["load_nodes"].size} load nodes)', flush=True)


if __name__ == '__main__':
    main()
