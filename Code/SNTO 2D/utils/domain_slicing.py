"""
Domain slicing helper for memory-scaled 2D stress-constrained TO.

This module contains the active merged-lag + N_MASK implementation used by the
Aug_5 2D L-beam driver. The physics are plane stress with two displacement
components u,v; two adjoint components t1,t2; three strain components
E11,E22,E12; von-Mises stress in 2D; and two DOF per node.

Core idea (unchanged from 3D)
-----------------------------
The deep-energy TO objective (strain energy, adjoint energy, stress objective,
mass) is a SUM over finite elements. Because it is a sum, we partition the
elements into `n_slice` disjoint groups, evaluate each group's contribution
independently, and accumulate the gradients -- the result is *identical* to the
single-shot full-domain gradient, but the peak autograd graph only ever holds
one slice at a time.

Merged-lag evaluation per training step
---------------------------------------
`accumulate_merged_lag_nmask` uses the previous iteration's globally assembled
adjoint stress force for the adjoint body-energy term, while assembling the next
raw force from the current detached displacement strains. The driver all-reduces
that raw force across ranks, finalizes the global normalizer, and caches it for
the next step.

Optional void exclusion
-----------------------
Elements below `void_thresh` are removed from the active element graphs. With
N_MASK, active masks are refreshed periodically; between refreshes, the density
forward and mass term are evaluated only on the cached active set.
"""

import torch

from utils.utils_general import projectDensity

# 2D layout: displacement u,v (2) + adjoint t1,t2 (2); density starts at index 4.
N_DISP = 2
N_ADJ = 2
PHASE_OFFSET = N_DISP + N_ADJ


def build_domain_slices(mesh_dict, n_slice, axis=0):
    """Partition ONE full-domain mesh into `n_slice` element-disjoint sub-meshes.

    Elements are sorted by their centroid coordinate along `axis` (0=x, 1=y) and
    split into `n_slice` contiguous bands of (nearly) equal element count.

    Returns a list of dicts (one per slice), each with:
        elem_ids    LongTensor [ne]     global element indices (into full arrays)
        node_ids    LongTensor [nn]     global node indices touched by the slice
        X_node      Tensor     [nn,2]   local node coordinates
        X_elem      Tensor     [ne,2]   local element centroids
        conn        LongTensor [ne,4]   connectivity remapped to 0..nn-1
        B, detJ, elem_vol                sliced element operators
        f_local     LongTensor [nf_s]   load nodes remapped to local numbering
        f_magnitude Tensor     [nf_s,2] load magnitudes for this slice
        own_node    BoolTensor [nn]     True where this slice is the canonical
                                        owner of the (globally shared) node --
                                        use for NODAL sum terms (body_energy_adj)
                                        so interface nodes are counted once.
    """
    device = mesh_dict["X_node"].device
    X_node = mesh_dict["X_node"].detach()
    X_elem = mesh_dict["X_elem"].detach()
    conn = mesh_dict["conn"]
    Nn = X_node.shape[0]

    order = torch.argsort(X_elem[:, axis])
    elem_chunks = torch.chunk(order, n_slice)

    node_owner = torch.full((Nn,), -1, dtype=torch.long, device=device)

    slices = []
    for s, elem_ids in enumerate(elem_chunks):
        elem_ids = elem_ids.clone()
        conn_sub = conn[elem_ids]                         # [ne, 4] global node ids
        node_ids = torch.unique(conn_sub)                 # sorted, [nn]

        remap = torch.full((Nn,), -1, dtype=torch.long, device=device)
        remap[node_ids] = torch.arange(node_ids.shape[0], device=device)
        conn_local = remap[conn_sub]                      # [ne, 4] in 0..nn-1

        unowned = node_owner[node_ids] < 0
        node_owner[node_ids[unowned]] = s

        f_index = mesh_dict["f_index"]
        in_slice = (remap[f_index] >= 0)
        f_local = remap[f_index[in_slice]]
        f_magnitude = mesh_dict["f_magnitude"][in_slice]

        slices.append({
            "elem_ids": elem_ids,
            "node_ids": node_ids,
            "X_node": X_node[node_ids].clone(),
            "X_elem": X_elem[elem_ids].clone(),
            "conn": conn_local,
            "B": mesh_dict["B"][elem_ids],
            "detJ": mesh_dict["detJ"][elem_ids],
            "elem_vol": mesh_dict["elem_vol"][elem_ids],
            "f_local": f_local,
            "f_magnitude": f_magnitude,
            # SLICED analytical hard-constraint weights (cached per slice; None if
            # unused). A_u/A_v are NODAL (indexed by node_ids); A_rho is per-ELEMENT
            # (indexed by elem_ids) -- built once in the loader, reused every epoch.
            "A_u": mesh_dict["A_u"][node_ids] if "A_u" in mesh_dict else None,
            "A_v": mesh_dict["A_v"][node_ids] if "A_v" in mesh_dict else None,
            "A_rho": mesh_dict["A_rho"][elem_ids] if "A_rho" in mesh_dict else None,
        })

    for s, sl in enumerate(slices):
        sl["own_node"] = (node_owner[sl["node_ids"]] == s)

    return slices


def attach_slices_to_mesh_data(mesh_data, n_slice, axis=0):
    """Data-prep entry point: build slices for EVERY mesh, store them in place.

    After this call each mesh_data[gpu_key][mesh_key] gains a "slices" entry
    (list of length n_slice). Geometry is fixed per mesh, so this is done once
    before the training loop.
    """
    for gpu_key in sorted(mesh_data.keys()):
        for mesh_key in sorted(mesh_data[gpu_key].keys()):
            mesh_data[gpu_key][mesh_key]["slices"] = build_domain_slices(
                mesh_data[gpu_key][mesh_key], n_slice, axis=axis
            )
    return mesh_data


# --- ANALYTICAL HARD-BC displacement/adjoint ----------------------------------
# field = A(x) * NN(x). A_u, A_v are the PRECOMPUTED, SLICED, cached distance
# weights (0 on the BC edge) passed in from the current view -- no per-iteration
# distance computation, just a NN forward + an elementwise multiply. Evaluated at
# an ARBITRARY point set X (a slice's local nodes); per-slice graphs are
# self-contained so summing per-slice gradients reproduces the full-domain gradient.
def _disp_fields_analytic(model_list, X, A_u, A_v):
    m = model_list[0].mean_module_NN_All(X)                 # [|X|, 2]
    return torch.stack([A_u * m[:, 0], A_v * m[:, 1]], dim=1)


def _adj_fields_analytic(model_list, X, A_u, A_v):
    m = model_list[N_DISP].mean_module_NN_All(X)            # [|X|, 2]
    return torch.stack([A_u * m[:, 0], A_v * m[:, 1]], dim=1)


def _phase_fields(model_list, X, num_phase, b, A_rho=None):
    # DENSITY = analytical rho=1 hard-constraint blend using the PRECOMPUTED, sliced
    # A_rho (built once in the loader, gathered to the current elements):
    #     rho = (1 - A_rho) * 1 + A_rho * projectDensity(NN(X)).
    # A_rho = 0 pins rho = 1 (with zero NN gradient there); A_rho -> 1 leaves rho
    # free. If A_rho is None (no density constraint), fall back to plain projected NN.
    p = projectDensity(model_list[PHASE_OFFSET].mean_module_NN_All(X), b)
    if A_rho is None:
        return p
    A = A_rho.unsqueeze(-1)
    return (1.0 - A) + A * p


def _elem_strains(Node_field, conn, B):
    """Plane-stress FEM strains for a slice: gather nodal field into elements
    (4 nodes x 2 DOF = 8), apply B -> [ne, N_int, 3] -> (E11, E22, E12)."""
    ne = conn.shape[0]
    u_elem = Node_field[conn].reshape(ne, -1)               # [ne, 8]
    u_elem = u_elem.unsqueeze(1).unsqueeze(-1).expand(-1, B.shape[1], -1, -1)
    eps = torch.matmul(B, u_elem).squeeze(-1)               # [ne, N_int, 3]
    return eps[..., 0], eps[..., 1], eps[..., 2]


def _restrict_to_active(sl, active_mask):
    """Build a sub-slice keeping only elements with active_mask==True, with the
    node set and connectivity remapped to the retained nodes. Used by the
    void-exclusion path so the element AND node graphs both shrink."""
    device = sl["conn"].device
    elem_keep = active_mask.nonzero(as_tuple=False).squeeze(-1)
    conn_sub = sl["conn"][elem_keep]                        # local (slice) node ids
    node_keep = torch.unique(conn_sub)
    remap = torch.full((sl["X_node"].shape[0],), -1, dtype=torch.long, device=device)
    remap[node_keep] = torch.arange(node_keep.shape[0], device=device)
    # load nodes that survive
    if sl["f_local"].numel() > 0:
        f_in = remap[sl["f_local"]] >= 0
        f_local = remap[sl["f_local"][f_in]]
        f_magnitude = sl["f_magnitude"][f_in]
    else:
        f_local = sl["f_local"]
        f_magnitude = sl["f_magnitude"]
    return {
        "elem_local": elem_keep,                            # index into slice elems
        "node_local": node_keep,                            # index into slice nodes
        "conn": remap[conn_sub],
        "X_node": sl["X_node"][node_keep],
        "X_elem": sl["X_elem"][elem_keep],
        "B": sl["B"][elem_keep],
        "detJ": sl["detJ"][elem_keep],
        "own_node": sl["own_node"][node_keep],
        "f_local": f_local,
        "f_magnitude": f_magnitude,
    }


def _assemble_fadj(F_adj, view, E11d, E22d, E12d, pw_det, MP, C1_0c, C2_0c):
    nu = MP['nu']; sy = MP['sy']; p = MP['p']
    s11_0 = C1_0c * (E11d + nu * E22d)
    s22_0 = C1_0c * (nu * E11d + E22d)
    s12_0 = C2_0c * E12d
    s_vm0 = torch.sqrt(s11_0 ** 2 + s22_0 ** 2 - s11_0 * s22_0 + 3.0 * s12_0 ** 2)
    s_vm_safe0 = torch.clamp(s_vm0, min=1e-8)
    s = s_vm_safe0 / sy - 1.0
    mE2 = (1e-6 + (1 - 1e-6) * pw_det ** p)
    ps = 2
    temp1 = (s + 1) ** ps
    h = torch.clamp(mE2 * s * temp1, min=0.0)
    dg = mE2 * (temp1 + ps * s * (s + 1) ** (ps - 1)) / sy
    hdh = h * (dg * (s > 0))
    dsvm_ds11 = (s11_0 - 0.5 * s22_0) / s_vm_safe0
    dsvm_ds22 = (s22_0 - 0.5 * s11_0) / s_vm_safe0
    dsvm_ds12 = (3.0 * s12_0) / s_vm_safe0
    dsvm_de = torch.stack([C1_0c * (dsvm_ds11 + nu * dsvm_ds22),
                           C1_0c * (nu * dsvm_ds11 + dsvm_ds22),
                           C2_0c * dsvm_ds12], dim=-1)
    dsvm_dU = torch.matmul(view["B"].transpose(-2, -1), dsvm_de.unsqueeze(-1)).squeeze(-1)
    f_adj_elem = -torch.sum(hdh.unsqueeze(-1) * dsvm_dU, dim=1)          # [ne, 4*2]
    conn_g = view["node_ids_global"][view["conn"]]                       # [ne, 4] global ids
    dof_idx = torch.stack([2 * conn_g, 2 * conn_g + 1], dim=-1).reshape(-1)
    F_adj.scatter_add_(0, dof_idx, f_adj_elem.reshape(-1))
    return s11_0, s22_0, s12_0, s        # detached, reused by the stress objective


def _finalize_fadj(F_adj, Nn, F_scale_goal=1e6, alpha_const=None):
    Fx = F_adj[0::2].unsqueeze(-1)
    Fy = F_adj[1::2].unsqueeze(-1)
    F2 = torch.cat((Fx, Fy), dim=1)
    if alpha_const is not None:                    # constant-alpha mode: fixed scale, F_adj/alpha_const
        alpha = torch.as_tensor(float(alpha_const), dtype=F2.dtype, device=F2.device)
    else:                                          # dynamic: normalize so max|F_adj_scaled| = F_scale_goal
        alpha = (torch.max(torch.abs(F2)) + 1e-8) / F_scale_goal
    return F2 / alpha, alpha


def _finish_merged_log(log, grey_counts, count_total, active_total, M0, alpha_val):
    log["massfrac"] = log["mass"] / M0
    log["alpha"] = alpha_val
    log["grey_counts"] = grey_counts.detach().cpu().tolist()
    log["count_total"] = count_total
    log["active_total"] = active_total
    log["M0"] = M0
    log["grey_fraction"] = (grey_counts / max(count_total, 1)).detach().cpu().tolist()
    return log

# ======================================================================
# N_MASK variants (Option B): the active-element mask is REFRESHED only every
# N_MASK steps and cached in between. Between refreshes the density forward is
# restricted to the cached active set -> the full-mesh density forward is
# skipped, so mass is summed over ACTIVE only (approximate, ~2e-4 rel. error;
# void < 1e-4 contributes ~nothing). Grey counts are exact (void < rho_min is
# never grey). Pass mask_cache=None to REFRESH (full density forward, exact
# mass, recompute mask); pass the cached per-slice masks to reuse them.
# ======================================================================
def _view_from_sub(sl, sub):
    return {"conn": sub["conn"], "B": sub["B"], "detJ": sub["detJ"],
            "X_node": sub["X_node"], "own": sub["own_node"],
            "node_ids_global": sl["node_ids"][sub["node_local"]],
            "X_elem": sl["X_elem"][sub["elem_local"]],
            "elem_vol": sl["elem_vol"][sub["elem_local"]],
            "f_local": sub["f_local"], "f_magnitude": sub["f_magnitude"],
            # sliced hard-BC weights gathered to the void-active nodes
            "A_u": sl["A_u"][sub["node_local"]] if sl.get("A_u") is not None else None,
            "A_v": sl["A_v"][sub["node_local"]] if sl.get("A_v") is not None else None}


def _slice_density_and_view(model_list, sl, void_thresh, num_phase, b, cached_mask):
    """Return (view, pw_active, mass, active_mask).
    cached_mask is None -> REFRESH: full density forward, EXACT mass, recompute mask.
    cached_mask given   -> density forward over ACTIVE only, mass APPROX over active."""
    elem_vol = sl["elem_vol"]
    A_rho = sl.get("A_rho")                                    # per-slice cached weight
    if cached_mask is None:                                    # ---- REFRESH ----
        pw_full = _phase_fields(model_list, sl["X_elem"], num_phase, b, A_rho)
        mass = torch.sum(pw_full * elem_vol.unsqueeze(-1))     # exact over all
        if void_thresh is None:
            view = {"conn": sl["conn"], "B": sl["B"], "detJ": sl["detJ"],
                    "X_node": sl["X_node"], "own": sl["own_node"],
                    "node_ids_global": sl["node_ids"], "X_elem": sl["X_elem"],
                    "elem_vol": sl["elem_vol"],
                    "f_local": sl["f_local"], "f_magnitude": sl["f_magnitude"],
                    "A_u": sl.get("A_u"), "A_v": sl.get("A_v")}
            return view, pw_full, mass, None
        active = pw_full.detach()[:, 0] >= void_thresh
        if active.sum() == 0:
            return None, None, mass, active
        sub = _restrict_to_active(sl, active)
        return _view_from_sub(sl, sub), pw_full[sub["elem_local"]], mass, active
    else:                                                      # ---- CACHED ----
        active = cached_mask
        if active.sum() == 0:
            return None, None, torch.zeros((), device=sl["conn"].device), active
        sub = _restrict_to_active(sl, active)
        pw = _phase_fields(model_list, sl["X_elem"][active], num_phase, b,
                           A_rho[active] if A_rho is not None else None)
        mass = torch.sum(pw * elem_vol[active].unsqueeze(-1))  # approx over active
        return _view_from_sub(sl, sub), pw, mass, active


def accumulate_merged_lag_nmask(model_list, slices, wd, wc, wm, delta,
                                include_adj, include_obj_mass, F_adj_scaled_lag,
                                alpha_lag, void_thresh, mask_cache=None,
                                return_raw_fadj=False):
    """Merged LAGGED-F_adj (single sweep) with N_MASK density-restriction.
    Returns (log, F_adj_scaled_new, alpha_new, new_mask_cache)."""
    m0 = model_list[0]; MP = m0.MP
    num_phase = MP['num_phase']; E_tensor = MP['E'][0]; wt = MP['wt'][0]
    thick = MP['thick']; nu = MP['nu']; p = MP['p']; ALPHA_B = MP['ALPHA_B']; b = MP['b']
    f_factor = MP['f_factor']; rho_min = MP['rho_min']; rho_max = MP['rho_max']
    dev = m0.collocation_x.device; dtp = m0.collocation_x.dtype
    Nn = m0.collocation_x.shape[0]; N_elem_global = m0.elem_x.shape[0]
    M0 = m0.elem_vol.sum()
    C1_0c = E_tensor / (1.0 - nu ** 2); C2_0c = E_tensor / (2.0 * (1.0 + nu))
    for model in model_list:
        model.train()
    log = {k: 0.0 for k in ["loss_total", "strain_energy", "external_work",
                            "strain_energy_adj", "body_energy_adj", "obj", "mass"]}
    grey_counts = torch.zeros(num_phase, device=dev); count_total = 0; active_total = 0
    stress_viol_count_solid = 0                            # solids (rho>=0.5) with MEAN-IP physical svm/sy>1 (matches plot_stress_figure)
    solid_count = 0                                        # solids (rho>=0.5); denominator of stress_violation_1
    n_h_activate = 0                                       # active elements (any rho) with ANY-IP h>0 (physical over-yield)
    n_h_activate_solid = 0                                # solids (rho>=0.5) with ANY-IP h>0
    F_adj = torch.zeros(2 * Nn, dtype=dtp, device=dev); new_masks = []

    # --- option 1: EMA density-move damping (default off; only fires if a driver
    #     sets MP['DENSITY_DAMP']=True). Adds friction on the mass/density swing. ---
    _damp_on = MP.get('DENSITY_DAMP', False)
    _damp_lam = MP.get('DENSITY_DAMP_LAMBDA', 0.0)
    _damp_beta = MP.get('DENSITY_DAMP_BETA', 0.99)
    if _damp_on and not hasattr(m0, '_rho_ema'):
        m0._rho_ema = {}                                   # per-slice EMA buffer, persists across epochs

    for i, sl in enumerate(slices):
        cm = mask_cache[i] if mask_cache is not None else None
        view, pw, mass, active = _slice_density_and_view(
            model_list, sl, void_thresh, num_phase, b, cm)
        new_masks.append(active)
        count_total += sl["conn"].shape[0]
        if view is None:
            L = (wm * mass / M0) if include_obj_mass else mass * 0.0
            if L.requires_grad:
                L.backward()
            log["mass"] += float(mass.item()); continue
        wl = pw.detach()
        for j in range(num_phase):
            grey_counts[j] += ((wl[:, j] > rho_min) & (wl[:, j] < rho_max)).sum()
        active_total += view["conn"].shape[0]

        # DEM (primal) + Adjoint stiffness penalization exponent. Default: fixed cubic (3)
        # -> matches TO14. If MP['DEM_ADJ_PENAL_P'] is True, use the ramping p (= MP['p'],
        # 3->5) so the primal/adjoint stiffness follows the SAME schedule as mE2 (TO15).
        p_dem = p if MP.get('DEM_ADJ_PENAL_P', False) else 3.0
        mE_no_grad = (1e-6 + (1 - 1e-6) * pw ** p_dem).detach()
        E = E_tensor * mE_no_grad; C1 = E / (1 - nu ** 2); C2 = E / (2 * (1 + nu))
        Node_disp = _disp_fields_analytic(model_list, view["X_node"], view["A_u"], view["A_v"])
        E11, E22, E12 = _elem_strains(Node_disp, view["conn"], view["B"])
        S11 = C1 * (E11 + nu * E22); S22 = C1 * (nu * E11 + E22); S12 = C2 * E12
        SE = torch.sum(torch.sum(0.5 * (S11 * E11 + S22 * E22 + S12 * E12) * wt * view["detJ"] * thick, dim=1))
        L = wd * SE
        EW = Node_disp.new_zeros(())
        if view["f_local"].numel() > 0:
            EW = torch.sum(Node_disp[view["f_local"]] * view["f_magnitude"] * f_factor)
            L = L - wd * EW + wd * (1 + delta) / 2 * EW.detach()

        if include_adj:
            Node_test = _adj_fields_analytic(model_list, view["X_node"], view["A_u"], view["A_v"])
            E11a, E22a, E12a = _elem_strains(Node_test, view["conn"], view["B"])
            S11a = C1 * (E11a + nu * E22a); S22a = C1 * (nu * E11a + E22a); S12a = C2 * E12a
            SE_adj = torch.sum(torch.sum(0.5 * (S11a * E11a + S22a * E22a + S12a * E12a) * wt * view["detJ"] * thick, dim=1))
            F_own = F_adj_scaled_lag[view["node_ids_global"]][view["own"]]
            body_energy_adj = -(1 / N_elem_global) * torch.sum(Node_test[view["own"]] * F_own)
            L = L + wd * (SE_adj + body_energy_adj)
            log["strain_energy_adj"] += SE_adj.item(); log["body_energy_adj"] += body_energy_adj.item()

        s_det = _assemble_fadj(F_adj, view, E11.detach(), E22.detach(), E12.detach(),
                               pw.detach(), MP, C1_0c, C2_0c)
        if include_obj_mass:
            s11_0, s22_0, s12_0, s = s_det
            mE2 = (1e-6 + (1 - 1e-6) * pw ** p)
            e11a, e22a, e12a = E11a.detach(), E22a.detach(), E12a.detach()
            comp = torch.sum(torch.sum(mE2 * (s11_0 * e11a + s22_0 * e22a + s12_0 * e12a) * wt * view["detJ"] * thick * alpha_lag, dim=1))
            h = torch.clamp(mE2 * (s * (s + 1) ** 2).detach(), min=0.0)
            # stress-violation monitor: DEFINITION CONSISTENT WITH plot_stress_figure.py.
            #   solid            = element density rho >= 0.5 (SOLID_THRESH, phase 0)
            #   violating solid  = solid whose PHYSICAL von-Mises ratio, meaned over
            #                      integration points, exceeds 1  (svm/sy = s+1;
            #                      VIOL_USE_PENALIZED=False in the figure).
            # Count solids and violating solids per slice so the driver divides by the
            # TRUE solid count (not the massfrac*N_total estimate).
            # All 'over yield' tests use the PHYSICAL C0 von-Mises ratio  s+1 = svm0/sy
            # (h>0 <=> s>0, i.e. physical over-yield; mE2 only scales h's magnitude, not
            # the count). Two IP-rules: MEAN-over-IP (>1) for the figure-consistent solid
            # count, ANY-IP (h>0) for the activation counts.
            _solid = (pw[:, 0].detach() >= 0.5)                     # [N_elem] solid mask (rho>=0.5)
            _phys_ratio_mean = (s + 1.0).mean(dim=1)                # MEAN-over-IP physical svm/sy
            _h_active = (h > 0).any(dim=1)                          # ANY-IP h>0 (physical over-yield)
            stress_viol_count_solid += int((_solid & (_phys_ratio_mean > 1.0)).sum().item())  # mean-IP, solids
            solid_count             += int(_solid.sum().item())                                # solids
            n_h_activate            += int(_h_active.sum().item())                             # any-IP, all active
            n_h_activate_solid      += int((_h_active & _solid).sum().item())                  # any-IP, solids
            body = torch.sum((1 / N_elem_global) * torch.sum(0.5 * h ** 2, dim=1))
            if MP.get('LOG_BODY_COMP', False):
                # diagnostic: per-element density-gradient magnitude of comp vs body
                # (d comp/d rho and d body/d rho -- analog of PolyStress dJ/dz split).
                gc = torch.autograd.grad(comp, pw, retain_graph=True, allow_unused=True)[0] if comp.requires_grad else None
                gb = torch.autograd.grad(body, pw, retain_graph=True, allow_unused=True)[0] if body.requires_grad else None
                gc = torch.zeros_like(pw) if gc is None else gc.detach()
                gb = torch.zeros_like(pw) if gb is None else gb.detach()
                log['bc_csq'] = log.get('bc_csq', 0.0) + float((gc ** 2).sum())
                log['bc_bsq'] = log.get('bc_bsq', 0.0) + float((gb ** 2).sum())
                log['bc_cabs'] = log.get('bc_cabs', 0.0) + float(gc.abs().sum())
                log['bc_babs'] = log.get('bc_babs', 0.0) + float(gb.abs().sum())
                log['bc_cmax'] = max(log.get('bc_cmax', 0.0), float(gc.abs().max()))
                log['bc_bmax'] = max(log.get('bc_bmax', 0.0), float(gb.abs().max()))
                log['bc_n'] = log.get('bc_n', 0) + gc.numel()
            body_obj = body
            if MP.get('BODY_CLIP', False):
                clip_ratio = MP.get('BODY_CLIP_RATIO', 0.1)
                clip_eps = MP.get('BODY_CLIP_EPS', 1e-12)
                cap = clip_ratio * (comp.detach().abs() + clip_eps)
                weighted_body = ALPHA_B * body.detach().abs()
                body_scale = torch.clamp(cap / (weighted_body + clip_eps), max=1.0)
                body_obj = body_scale * body
            obj = (comp + ALPHA_B * body_obj) if MP.get('INCLUDE_BODY', False) else comp   # INCLUDE_BODY: add 0.5*h^2 body term (default off; destabilizes in 2D)
            L = L + wc * obj + wm * mass / M0
            log["obj"] += obj.item(); log["mass"] += float(mass.item())
        else:
            log["mass"] += float(mass.item())
        if _damp_on and _damp_lam > 0.0:
            # density at the slice's FIXED element centroids (stable shape, mask-agnostic)
            pw_dmp = _phase_fields(model_list, sl["X_elem"], num_phase, b, sl.get("A_rho"))[:, 0]
            ref = m0._rho_ema.get(i)
            if ref is None or ref.shape != pw_dmp.shape:
                ref = pw_dmp.detach().clone()             # init on first sight -> zero penalty
            L = L + _damp_lam * torch.mean((pw_dmp - ref) ** 2)   # pull rho toward its EMA -> friction
            m0._rho_ema[i] = (_damp_beta * ref + (1.0 - _damp_beta) * pw_dmp.detach()).detach()
        L.backward()
        log["strain_energy"] += SE.item(); log["external_work"] += float(EW.item()); log["loss_total"] += L.item()

    log["stress_viol_count_solid"] = stress_viol_count_solid  # local; driver all-reduces (SUM)
    log["solid_count"] = solid_count                          # local; driver all-reduces (SUM)
    log["n_h_activate"] = n_h_activate                        # local; driver all-reduces (SUM)
    log["n_h_activate_solid"] = n_h_activate_solid            # local; driver all-reduces (SUM)

    if return_raw_fadj:
        # DDP: return the LOCAL raw F_adj (flat 2*Nn). The driver all-reduces it
        # across ranks, then calls _finalize_fadj on the GLOBAL sum (so alpha is a
        # global max). log['alpha'] is a placeholder here, filled after the reduce.
        out = _finish_merged_log(log, grey_counts, count_total, active_total, float(M0), 0.0)
        return out, F_adj, new_masks
    F_adj_scaled_new, alpha_new = _finalize_fadj(F_adj, Nn, MP.get('F_scale_goal', 1e6))
    out = _finish_merged_log(log, grey_counts, count_total, active_total, float(M0), float(alpha_new.item()))
    return out, F_adj_scaled_new, alpha_new, new_masks
