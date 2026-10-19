import torch
import numpy as np
from utils.utils_general import get_multiGPU
import math

# =============================================================================
# EXACT-NODE BOUNDARY / LOAD SAMPLING
# -----------------------------------------------------------------------------
# For each mesh we select, from that mesh's own X_node, the boundary nodes on the
# u=v=0 edges and the external loading region, stored in mesh_dict['point_sets'].
# These point sets feed two things:
#   * the analytical hard-constraint weights A_u, A_v (displacement/adjoint BCs)
#     and A_rho (density = 1 on the loading region) -- all built from min-distance
#     to these points (see _distance_weight / the drivers' _A_rho);
#   * the LMGP models' train points (the networks are LMGP objects used only as
#     NN containers; NO GP correction is evaluated anywhere).
# Selecting from the actual mesh nodes (not an independent grid) keeps the
# constraints aligned with where the physics is evaluated and scales the point
# count automatically with mesh refinement.
#
# DYNAMIC / MULTI-MESH: the boundary nodes differ from mesh to mesh, so each mesh
# gets its OWN point set. Nothing here assumes a single mesh -- it loops over every
# (GPU, mesh) present.
#
# TOLERANCES: the meshes are unstructured in the interior (node coordinates are
# not on a regular grid), so a "unique coordinate spacing" tolerance is just
# float noise. We instead use a characteristic element size h = sqrt(area / N)
# and select boundary nodes within a fraction of h of the boundary line/arc.
# The boundary node rows sit cleanly on the domain edges (the next interior row
# is ~h away), so any tol in ( ~0.1 h , ~0.9 h ) captures exactly the edge nodes.
# =============================================================================

_TOL_LINE = 0.40     # boundary-line tolerance as a fraction of h
_TOL_CIRC = 0.60     # circular-rim tolerance as a fraction of h (a touch wider)

# Hook fixed-support hole (u=v=0 BC is the rim of this circle). Used both to select
# the exact rim nodes (_points_Hook) and to build the CIRCULAR analytical weight A_u/A_v.
_HOOK_BC_CIRCLE = (0.0, 80.6226, 10.0)   # (center_x, center_y, radius)


def _char_h(X):
    """Characteristic node spacing h = sqrt(bbox_area / N_node)."""
    mn = X.min(dim=0).values
    mx = X.max(dim=0).values
    area = ((mx[0] - mn[0]) * (mx[1] - mn[1])).clamp(min=1e-12)
    return float(torch.sqrt(area / X.shape[0]).clamp(min=1e-9))


def _sel_line(X, fixed_axis, fixed_value, tol, rng_axis=None, rng_lo=None, rng_hi=None):
    """Mesh nodes on the line {coord[fixed_axis] == fixed_value}, optionally
    restricted to rng_lo <= coord[rng_axis] <= rng_hi (bounds padded by tol)."""
    m = (X[:, fixed_axis] - fixed_value).abs() < tol
    if rng_axis is not None:
        if rng_lo is not None:
            m = m & (X[:, rng_axis] >= rng_lo - tol)
        if rng_hi is not None:
            m = m & (X[:, rng_axis] <= rng_hi + tol)
    return X[m]


def _sel_corner(X, x_val, y_val, tol):
    """Mesh node(s) at the corner (x_val, y_val)."""
    m = ((X[:, 0] - x_val).abs() < tol) & ((X[:, 1] - y_val).abs() < tol)
    return X[m]


def _sel_circle(X, cx, cy, R, tol, half=None):
    """Mesh nodes on the arc of radius R about (cx, cy); half in {'upper','lower'}."""
    dist = torch.sqrt((X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2)
    m = (dist - R).abs() < tol
    if half == 'upper':
        m = m & (X[:, 1] >= cy)
    elif half == 'lower':
        m = m & (X[:, 1] <= cy)
    return X[m]


def _lin(a, b, fixed, axis, n, dtype, device):
    """Analytic linspace edge (fallback only): axis='x' varies x in [a,b] at
    y=fixed; axis='y' varies y in [a,b] at x=fixed."""
    t = torch.linspace(a, b, n, dtype=dtype, device=device)
    f = torch.full((n,), float(fixed), dtype=dtype, device=device)
    return torch.stack((t, f), dim=1) if axis == 'x' else torch.stack((f, t), dim=1)


def _ensure(sel, fb, name):
    """If node selection came up empty (geometry mismatch), fall back to the
    analytic linspace points so the run does not crash -- with a loud warning."""
    if sel.shape[0] == 0:
        pts = fb()
        print(f"[get_data] WARNING: boundary '{name}' matched 0 mesh nodes; "
              f"using {pts.shape[0]} analytic linspace points instead.", flush=True)
        return pts
    return sel


def _min_dist(X, edge, chunk=8192):
    """min Euclidean distance from each row of X to the edge-node set (chunked so
    the [|X|, |edge|] temporary stays bounded)."""
    outs = []
    for i in range(0, X.shape[0], chunk):
        outs.append(torch.cdist(X[i:i + chunk], edge).min(dim=1).values)
    return torch.cat(outs, dim=0)


def _apply_shape(dn, afunc):
    """Map the NORMALIZED min-distance dn in [0,1] (0 ON the BC) to the weight A in
    [0,1]. All shapes give A=0 at dn=0 (exact hard BC) and A->1 away from the BC.
    afunc is a dict picking the shape:
        {'type': 'linear'}        A = dn
        {'type': 'quadratic'}     A = dn**2
        {'type': 'exp', 'k': K}   A = 1 - exp(-K*dn)  (saturating; larger K -> thinner
                                  boundary layer, so A ~ 1 over most of the domain)."""
    t = afunc.get('type', 'linear')
    if t == 'exp':
        A = 1.0 - torch.exp(-float(afunc.get('k', 10.0)) * dn)
    elif t == 'quadratic':
        A = dn * dn
    else:  # linear
        A = dn
    return A.detach()


def _distance_weight(X, edge, afunc):
    """Hard-BC weight A(x) for a STRAIGHT (vertical/horizontal) BC edge: 0 ON the edge
    (min-distance to the edge nodes), shaped by `afunc`. Once per mesh; sliced/cached
    downstream. Used for L_beam / Crack / Portal_frame_half (line edges)."""
    d = _min_dist(X.detach(), edge)
    maxd = max(float(d.max()), 1e-12)
    dn = (d / maxd).clamp(max=1.0)
    return _apply_shape(dn, afunc)


def _half_circle_weight(X, cx, cy, R, afunc, upper=True):
    """ANALYTICAL HARD-BC WEIGHT A(x) in [0,1] for a HALF-circle arc BC (the Hook
    fixed support is only the UPPER half of the hole rim, so the LOWER half must NOT
    be constrained). A = (minimum distance from x to the half-circle arc), scaled.

    The arc is {(cx+R cos t, cy+R sin t): t in [0, pi]} (upper half, y>=cy).
    Closest arc point:
      * if x projects radially onto the arc (upper side, dy>=0) -> |dist(x,c) - R|;
      * else the nearer arc ENDPOINT (cx+-R, cy).
    So A=0 on the upper arc only; the lower half of the circle is left free."""
    dx = X[:, 0] - cx
    dy = X[:, 1] - cy
    r = torch.sqrt(dx * dx + dy * dy)
    d_radial = (r - R).abs()                                   # on-arc (radial) distance
    e0 = torch.sqrt((X[:, 0] - (cx + R)) ** 2 + dy * dy)       # endpoint t=0  (cx+R, cy)
    e1 = torch.sqrt((X[:, 0] - (cx - R)) ** 2 + dy * dy)       # endpoint t=pi (cx-R, cy)
    d_end = torch.minimum(e0, e1)
    on_arc = (dy >= 0) if upper else (dy <= 0)                 # radial projection lands on the arc
    d = torch.where(on_arc, d_radial, d_end)
    maxd = max(float(d.max()), 1e-12)
    dn = (d / maxd).clamp(max=1.0)
    return _apply_shape(dn, afunc)


def _assemble_point_sets(u_X, v_X, t1_X, t2_X, phase_specs, dtype, device):
    """Build the per-mesh point dict (mesh_dict['point_sets']) from selected boundary
    and loading coordinate tensors: the u/v/t1/t2 BC node sets plus the phase
    (loading-region) sets. These feed the analytical A-weights and the LMGP models'
    train points. Targets: u=v=t1=t2 -> 0; each phase spec is (coords, value_or_tensor)."""
    def mk(P):
        return P.detach().to(device=device, dtype=dtype).contiguous().requires_grad_(False)

    def zeros(P):
        return torch.zeros(P.shape[0], dtype=dtype, device=device)

    pts = {
        'u_X_train': mk(u_X), 'v_X_train': mk(v_X),
        't1_X_train': mk(t1_X), 't2_X_train': mk(t2_X),
    }
    pts['u_train'] = zeros(pts['u_X_train'])
    pts['v_train'] = zeros(pts['v_X_train'])
    pts['t1_train'] = zeros(pts['t1_X_train'])
    pts['t2_train'] = zeros(pts['t2_X_train'])
    for j, (pX, pval) in enumerate(phase_specs):
        pXm = mk(pX)
        pts[f'phase{j}_X_train'] = pXm
        if torch.is_tensor(pval):
            pts[f'phase{j}_train'] = pval.detach().to(device=device, dtype=dtype)
        else:
            pts[f'phase{j}_train'] = torch.full((pXm.shape[0],), float(pval),
                                               dtype=dtype, device=device)
    return pts


# -----------------------------------------------------------------------------
# Per-example builders: given ONE mesh's node coordinates X (on device), return
# the exact-node point dict for that mesh. Boundary lines are snapped to
# the mesh's actual edges (min/max), feature bounds kept as literals.
# -----------------------------------------------------------------------------
def _points_L_beam(X, MP, dtype, device):
    xmin = MP['domain']['x'][0]
    num_bc, num_load = MP['num_bc'], MP['num_load']
    tol = _TOL_LINE * _char_h(X)
    y_top = float(X[:, 1].max())
    x_right = float(X[:, 0].max())
    # fixed edge (u=v=0): top edge y=ymax, x in [xmin, 40]
    bc = _sel_line(X, 1, y_top, tol, rng_axis=0, rng_lo=xmin, rng_hi=40.0)
    bc = _ensure(bc, lambda: _lin(xmin, 40.0, y_top, 'x', num_bc, dtype, device), 'L_beam:BC')
    # loaded region (density=1): right edge x=xmax, y in [35, 40]
    load = _sel_line(X, 0, x_right, tol, rng_axis=1, rng_lo=35.0, rng_hi=40.0)
    load = _ensure(load, lambda: _lin(35.0, 40.0, x_right, 'y', num_load, dtype, device), 'L_beam:load')
    return _assemble_point_sets(bc, bc, bc, bc, [(load, 1.0)], dtype, device)


def _points_Crack(X, MP, dtype, device):
    xmin = MP['domain']['x'][0]
    ymin = MP['domain']['y'][0]
    num_bc, num_load = MP['num_bc'], MP['num_load']
    tol = _TOL_LINE * _char_h(X)
    x_left = float(X[:, 0].min())
    y_bot = float(X[:, 1].min())
    x_right = float(X[:, 0].max())
    # u=0 on left edge x=xmin, y in [ymin, 100]
    LB = _sel_line(X, 0, x_left, tol, rng_axis=1, rng_lo=ymin, rng_hi=100.0)
    LB = _ensure(LB, lambda: _lin(ymin, 100.0, x_left, 'y', num_bc, dtype, device), 'Crack:LB')
    # v=0 pinned at the bottom-left corner node
    corner = _sel_corner(X, x_left, y_bot, tol)
    corner = _ensure(corner, lambda: torch.tensor([[x_left, y_bot]], dtype=dtype, device=device), 'Crack:corner')
    # density=1 on right edge x=xmax, y in [190, 200]
    load = _sel_line(X, 0, x_right, tol, rng_axis=1, rng_lo=190.0, rng_hi=200.0)
    load = _ensure(load, lambda: _lin(190.0, 200.0, x_right, 'y', num_load, dtype, device), 'Crack:load')
    return _assemble_point_sets(LB, corner, LB, corner, [(load, 1.0)], dtype, device)


def _points_Portal_frame_half(X, MP, dtype, device):
    xmin = MP['domain']['x'][0]
    num_bc, num_load = MP['num_bc'], MP['num_load']
    tol = _TOL_LINE * _char_h(X)
    x_left = float(X[:, 0].min())
    y_bot = float(X[:, 1].min())
    y_top = float(X[:, 1].max())
    # u=0 on the left symmetry edge x=xmin, y in [35, 60]
    LE = _sel_line(X, 0, x_left, tol, rng_axis=1, rng_lo=35.0, rng_hi=60.0)
    LE = _ensure(LE, lambda: _lin(35.0, 60.0, x_left, 'y', num_bc, dtype, device), 'Portal_half:LE')
    # v=0 on the right foot, bottom edge y=ymin, x in [54.5, 60]
    RB = _sel_line(X, 1, y_bot, tol, rng_axis=0, rng_lo=54.5, rng_hi=60.0)
    RB = _ensure(RB, lambda: _lin(54.5, 60.0, y_bot, 'x', max(1, num_bc // 5), dtype, device), 'Portal_half:RB')
    # density=1 on the top edge y=ymax, x in [0, 5]
    load = _sel_line(X, 1, y_top, tol, rng_axis=0, rng_lo=0.0, rng_hi=5.0)
    load = _ensure(load, lambda: _lin(0.0, 5.0, y_top, 'x', num_load, dtype, device), 'Portal_half:load')
    return _assemble_point_sets(LE, RB, LE, RB, [(load, 1.0)], dtype, device)


def _points_Hook(X, MP, dtype, device):
    num_bc, num_load = MP['num_bc'], MP['num_load']
    tol = _TOL_CIRC * _char_h(X)
    # fixed hole rim (u=v=0): upper half of circle R=10 about (0, 80.6226)
    cx0, cy0, R0 = 0.0, 80.6226, 10.0
    top_circ = _sel_circle(X, cx0, cy0, R0, tol, half='upper')

    def _fb_circle(cx, cy, R, a0, a1, n):
        th = torch.linspace(a0, a1, n, dtype=dtype, device=device)
        return torch.stack([cx + R * torch.cos(th), cy + R * torch.sin(th)], dim=1)

    top_circ = _ensure(top_circ, lambda: _fb_circle(cx0, cy0, R0, 0.0, math.pi, num_bc), 'Hook:fixed')
    # loaded hole rim (density=1): lower half of circle R=19 about (0, 0)
    cx1, cy1, R1 = 0.0, 0.0, 19.0
    load = _sel_circle(X, cx1, cy1, R1, tol, half='lower')
    load = _ensure(load, lambda: _fb_circle(cx1, cy1, R1, 0.0, -math.pi, num_load), 'Hook:load')
    return _assemble_point_sets(top_circ, top_circ, top_circ, top_circ, [(load, 1.0)], dtype, device)


# Example -> per-mesh point builder (mirrors the get_data_* wrappers). Used by
# tooling that must rebuild the analytical weights from a RAW mesh (e.g. plot_rho)
# without re-running the CUDA data pipeline.
_POINT_BUILDERS = {'L_beam': _points_L_beam, 'Crack': _points_Crack,
                   'Portal_frame_half': _points_Portal_frame_half, 'Hook': _points_Hook}


def analytic_weights_for_mesh(X, MP, dtype=torch.float32, device=None):
    """Rebuild (point_sets, A_u, A_v) for a mesh's node coords X using the SAME
    construction as _process_meshes below. point_sets['phase0_X_train'] gives
    the loading-region points for A_rho. CPU-safe (pure tensor ops -- does NOT call
    get_multiGPU), so the plotting code can reconstruct the analytical fields."""
    device = X.device if device is None else device
    pts = _POINT_BUILDERS[MP['Example']](X, MP, dtype, device)
    afunc = MP.get('AFUNC', {'type': 'linear'})
    if MP.get('Example', '') == 'Hook':
        cx, cy, R = _HOOK_BC_CIRCLE
        A = _half_circle_weight(X, cx, cy, R, afunc, upper=True)
        A_u, A_v = A, A
    else:
        A_u = _distance_weight(X, pts['u_X_train'], afunc)
        A_v = _distance_weight(X, pts['v_X_train'], afunc)
    return pts, A_u, A_v


def density_weight_for_mesh(X_elem, X_node, point_sets, MP):
    """A_rho at element centroids (for training) and at nodes (for plotting), built
    identically for the loader and the plotting code so there is ONE A_rho
    definition. A_rho = shape( max(min_dist(x, load) - RHO_SLAB_H, 0) / maxd ): 0
    within RHO_SLAB_H of the loading region (rho pinned to 1) -> 1 away. Uses
    MP['AFUNC_RHO'] if set (e.g. a SHARPER exp K than the disp/adjoint A_u/A_v, so
    the un-adjustable (1-A_rho) transition band in rho=(1-A_rho)+A_rho*proj(NN) is
    thinner), else falls back to MP['AFUNC']. Shared maxd (from the element
    distances). Returns (A_rho_elem, A_rho_node)."""
    afunc = MP.get('AFUNC_RHO', MP.get('AFUNC', {'type': 'linear'}))
    H = MP.get('RHO_SLAB_H', 0.0)
    load = point_sets['phase0_X_train']
    de = (_min_dist(X_elem.detach(), load) - H).clamp(min=0.0)
    maxd = max(float(de.max()), 1e-12)
    A_elem = _apply_shape((de / maxd).clamp(max=1.0), afunc)
    dn = (_min_dist(X_node.detach(), load) - H).clamp(min=0.0)
    A_node = _apply_shape((dn / maxd).clamp(max=1.0), afunc)
    return A_elem, A_node


# -----------------------------------------------------------------------------
# Shared mesh-processing + per-mesh point assembly. Moves each mesh's tensors to
# the right device/dtype and attaches mesh_dict['point_sets'] (the boundary/loading
# point sets) built from that mesh's own nodes, plus the analytical weights A_u,
# A_v. Returns (Training, mesh_data) where Training is the FIRST mesh's point dict
# (used only to CONSTRUCT the LMGP NN containers).
# -----------------------------------------------------------------------------
def _process_meshes(mesh_data, tkwargs, MP, points_builder):
    Training = None
    for i, device in enumerate(tkwargs["devices"]):
        gpu_key = f"GPU{i}"
        for mesh_key, mesh_dict in mesh_data[gpu_key].items():
            for k, v in mesh_dict.items():
                if k in ["X_node", "X_elem"]:
                    mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(True)
                elif k in ["f_index", "f_adj_index", "conn", "K_in_index", "K_out_index"]:
                    mesh_dict[k] = v.to(device, dtype=torch.long)
                else:
                    mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(False)
            # exact-node boundary/loading point set for THIS mesh (from its nodes)
            X = mesh_dict["X_node"].detach()
            pts = points_builder(X, MP, tkwargs["dtype"], device)
            mesh_dict["point_sets"] = pts
            # ANALYTICAL HARD-BC WEIGHTS A_u, A_v over ALL mesh nodes, computed ONCE
            # here (0 on the u/v BC edges, ->1 away). Sliced onto each slice in
            # build_domain_slices and just multiplied into the disp/adjoint NN
            # output every iteration -- no per-iteration distance recompute.
            # analytical hard-BC weight shape (single spec from the driver, e.g.
            # {'type':'linear'} | {'type':'quadratic'} | {'type':'exp','k':20}).
            afunc = MP.get('AFUNC', {'type': 'linear'})
            if MP.get('Example', '') == 'Hook':
                # HALF-circle fixed-hole BC: only the UPPER half of the rim is fixed,
                # so A = min-distance to the upper arc (lower half left free). u and v
                # share the same arc. See _half_circle_weight / _HOOK_BC_CIRCLE.
                cx, cy, R = _HOOK_BC_CIRCLE
                mesh_dict["A_u"] = _half_circle_weight(X, cx, cy, R, afunc, upper=True)
                mesh_dict["A_v"] = _half_circle_weight(X, cx, cy, R, afunc, upper=True)
            else:
                # STRAIGHT (vertical/horizontal) BC edges: min-distance to edge nodes.
                mesh_dict["A_u"] = _distance_weight(X, pts['u_X_train'], afunc)
                mesh_dict["A_v"] = _distance_weight(X, pts['v_X_train'], afunc)
            # ANALYTICAL DENSITY WEIGHT A_rho, built ONCE here just like A_u/A_v:
            #   * "A_rho"      at ELEMENT centroids -> sliced per slice in
            #     build_domain_slices and used every density forward (no recompute);
            #   * "A_rho_node" at NODES -> only for the combined A_u/A_v/A_rho plot.
            # density_weight_for_mesh is shared with the plotting code (plot_rho).
            mesh_dict["A_rho"], mesh_dict["A_rho_node"] = density_weight_for_mesh(
                mesh_dict["X_elem"], X, pts, MP)
            # First mesh's point dict is returned as `Training` (used only to build
            # the LMGP NN containers). A_u/A_v/A_rho are plotted together on a single
            # figure in the driver (see _plot_A_weights).
            if Training is None:
                Training = pts
    return Training, mesh_data


def get_data_L_beam(MP):
    Example = MP['Example']
    Diff_type = MP['Diff_type']
    N_worker = MP['N_worker']
    base_folder = MP['base_folder']
    file_loc = base_folder + '/Data/' + f'{Example}_GPU{N_worker}_{Diff_type}.pt'
    mesh_data = torch.load(file_loc)
    tkwargs = get_multiGPU(N_worker)
    return _process_meshes(mesh_data, tkwargs, MP, _points_L_beam)


def get_data_Crack(MP):
    Example = MP['Example']
    Diff_type = MP['Diff_type']
    N_worker = MP['N_worker']
    base_folder = MP['base_folder']
    file_loc = base_folder + '/Data/' + f'{Example}_GPU{N_worker}_{Diff_type}.pt'
    mesh_data = torch.load(file_loc)
    tkwargs = get_multiGPU(N_worker)
    return _process_meshes(mesh_data, tkwargs, MP, _points_Crack)


def get_data_Portal_frame_half(MP):
    Example = MP['Example']
    Diff_type = MP['Diff_type']
    N_worker = MP['N_worker']
    base_folder = MP['base_folder']
    file_loc = base_folder + '/Data/' + f'{Example}_GPU{N_worker}_{Diff_type}.pt'
    mesh_data = torch.load(file_loc)
    tkwargs = get_multiGPU(N_worker)
    return _process_meshes(mesh_data, tkwargs, MP, _points_Portal_frame_half)


def get_data_Hook(MP):
    Example = MP['Example']
    Diff_type = MP['Diff_type']
    N_worker = MP['N_worker']
    base_folder = MP['base_folder']
    file_loc = base_folder + '/Data/' + f'{Example}_GPU{N_worker}_{Diff_type}.pt'
    mesh_data = torch.load(file_loc)
    tkwargs = get_multiGPU(N_worker)
    return _process_meshes(mesh_data, tkwargs, MP, _points_Hook)


