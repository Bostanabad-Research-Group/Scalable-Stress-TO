"""
Domain slicing helper for memory-scaled 3D stress-constrained TO.

The full-domain physics loss (strain energy, adjoint energy, stress objective,
mass) is a SUM over finite elements.  Because it is a sum, we can partition the
elements into `n_slice` disjoint groups, evaluate each group's contribution
independently, and accumulate the gradients -- the result is *identical* to the
single-shot full-domain gradient, but the peak autograd graph only ever holds
one slice at a time.

This module is shared by:
  * L_beam_TO_option1_elembatch.py  (single GPU, loop over slices, accumulate)
  * L_beam_TO_option3_ddp.py        (one slice per GPU rank, all-reduce grads)

`build_domain_slices` is a *data-preparation* step: run it ONCE before training
for every mesh.  It computes, per slice, the local X_node / X_elem / conn plus
everything `calculate_TO_loss` needs, with node indices remapped to a compact
local numbering.
"""

import torch

from utils.utils_general import projectDensity

N_DISP = 3
N_ADJ = 3
PHASE_OFFSET = N_DISP + N_ADJ


def build_domain_slices(mesh_dict, n_slice, axis=0):
    """Partition ONE full-domain mesh into `n_slice` element-disjoint sub-meshes.

    Elements are sorted by their centroid coordinate along `axis` (0=x, 1=y,
    2=z) and split into `n_slice` contiguous bands of (nearly) equal element
    count, so each slice carries a comparable memory load.

    Parameters
    ----------
    mesh_dict : dict
        One full mesh, i.e. mesh_data[gpu_key][mesh_key].  Must contain
        X_node [Nn,3], X_elem [Ne,3], conn [Ne,8], B [Ne,1,6,24],
        detJ [Ne,1], elem_vol [Ne], f_index [nf], f_magnitude [nf,3].
    n_slice : int
        Number of slices (typically 2..8).
    axis : int
        Coordinate axis used to order elements before splitting.

    Returns
    -------
    slices : list[dict]  (length n_slice), each with
        elem_ids    LongTensor [ne]     global element indices (into full arrays)
        node_ids    LongTensor [nn]     global node indices touched by the slice
        X_node      Tensor     [nn,3]   local node coordinates
        X_elem      Tensor     [ne,3]   local element centroids
        conn        LongTensor [ne,8]   connectivity remapped to 0..nn-1
        B, detJ, elem_vol                sliced element operators
        f_local     LongTensor [nf_s]   load nodes remapped to local numbering
        f_magnitude Tensor     [nf_s,3] load magnitudes for this slice
        own_node    BoolTensor [nn]     True where this slice is the canonical
                                        owner of the (globally shared) node.
                                        Use this mask for NODAL sum terms
                                        (e.g. body_energy_adj) so interface
                                        nodes are counted exactly once.
    """
    device = mesh_dict["X_node"].device
    X_node = mesh_dict["X_node"].detach()
    X_elem = mesh_dict["X_elem"].detach()
    conn = mesh_dict["conn"]
    Nn = X_node.shape[0]

    # --- order elements along `axis` and cut into n_slice balanced chunks ---
    order = torch.argsort(X_elem[:, axis])
    elem_chunks = torch.chunk(order, n_slice)

    # canonical owner of every global node (first slice that references it)
    node_owner = torch.full((Nn,), -1, dtype=torch.long, device=device)

    slices = []
    for s, elem_ids in enumerate(elem_chunks):
        elem_ids = elem_ids.clone()
        conn_sub = conn[elem_ids]                        # [ne, 8] global node ids
        node_ids = torch.unique(conn_sub)                # sorted, [nn]

        # global -> local node map (dense scatter, -1 = not in this slice)
        remap = torch.full((Nn,), -1, dtype=torch.long, device=device)
        remap[node_ids] = torch.arange(node_ids.shape[0], device=device)
        conn_local = remap[conn_sub]                     # [ne, 8] in 0..nn-1

        # claim ownership of any node not yet owned by an earlier slice
        unowned = node_owner[node_ids] < 0
        node_owner[node_ids[unowned]] = s

        # load nodes that live inside this slice, remapped to local numbering
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
            "A_u": mesh_dict["A_u"][node_ids] if "A_u" in mesh_dict else None,
            "A_v": mesh_dict["A_v"][node_ids] if "A_v" in mesh_dict else None,
            "A_w": mesh_dict["A_w"][node_ids] if "A_w" in mesh_dict else None,
            "A_rho": mesh_dict["A_rho"][elem_ids] if "A_rho" in mesh_dict else None,
        })

    # second pass: fill in ownership masks now that node_owner is final
    for s, sl in enumerate(slices):
        sl["own_node"] = (node_owner[sl["node_ids"]] == s)

    return slices


def attach_slices_to_mesh_data(mesh_data, n_slice, axis=0):
    """Data-prep entry point: build slices for EVERY mesh, store them in place.

    After this call each mesh_data[gpu_key][mesh_key] gains a "slices" entry
    (list of length n_slice).  Geometry is fixed per mesh, so this is done once
    before the training loop.
    """
    for gpu_key in sorted(mesh_data.keys()):
        for mesh_key in sorted(mesh_data[gpu_key].keys()):
            mesh_data[gpu_key][mesh_key]["slices"] = build_domain_slices(
                mesh_data[gpu_key][mesh_key], n_slice, axis=axis
            )
    return mesh_data


# --- ANALYTICAL HARD-BC displacement/adjoint (3D): field = A(x) * NN(x). A_u,A_v,A_w
#     are precomputed distance weights (0 on the BC surface), built once in the loader
#     and sliced per slice; just a NN forward + elementwise multiply (no GP). ---
def _disp_fields_analytic(model_list, X, A_u, A_v, A_w):
    m = model_list[0].mean_module_NN_All(X)                 # [|X|, 3]
    return torch.stack([A_u * m[:, 0], A_v * m[:, 1], A_w * m[:, 2]], dim=1)


def _adj_fields_analytic(model_list, X, A_u, A_v, A_w):
    m = model_list[N_DISP].mean_module_NN_All(X)            # [|X|, 3]
    return torch.stack([A_u * m[:, 0], A_v * m[:, 1], A_w * m[:, 2]], dim=1)


def _phase_fields(model_list, X, num_phase, b, A_rho=None):
    # DENSITY = analytical rho=1 hard-constraint blend using the PRECOMPUTED, sliced
    # A_rho (built once in the loader, gathered to the current elements):
    #     rho = (1 - A_rho) * 1 + A_rho * projectDensity(NN(X)).
    # A_rho = 0 pins rho = 1 (NN gets no gradient there); A_rho -> 1 leaves rho free.
    p = projectDensity(model_list[PHASE_OFFSET].mean_module_NN_All(X), b)
    if A_rho is None:
        return p
    A = A_rho.unsqueeze(-1)
    return (1.0 - A) + A * p


def _elem_strains(Node_field, conn, B):
    """FEM strains for a slice: gather nodal field into elements, apply B."""
    ne = conn.shape[0]
    u_elem = Node_field[conn].reshape(ne, -1)
    u_elem = u_elem.unsqueeze(1).unsqueeze(-1).expand(-1, B.shape[1], -1, -1)
    eps = torch.matmul(B, u_elem).squeeze(-1)               # [ne, N_int, 6]
    return [eps[..., i] for i in range(6)]


def _restrict_to_active(sl, active_mask):
    """Sub-slice keeping only elements with active_mask==True; node set and conn
    remapped to the retained nodes (both element AND node graphs shrink)."""
    device = sl["conn"].device
    elem_keep = active_mask.nonzero(as_tuple=False).squeeze(-1)
    conn_sub = sl["conn"][elem_keep]                        # local (slice) node ids
    node_keep = torch.unique(conn_sub)
    remap = torch.full((sl["X_node"].shape[0],), -1, dtype=torch.long, device=device)
    remap[node_keep] = torch.arange(node_keep.shape[0], device=device)
    if sl["f_local"].numel() > 0:
        f_in = remap[sl["f_local"]] >= 0
        f_local = remap[sl["f_local"][f_in]]
        f_magnitude = sl["f_magnitude"][f_in]
    else:
        f_local = sl["f_local"]
        f_magnitude = sl["f_magnitude"]
    return {
        "elem_local": elem_keep,
        "node_local": node_keep,
        "conn": remap[conn_sub],
        "X_node": sl["X_node"][node_keep],
        "X_elem": sl["X_elem"][elem_keep],
        "B": sl["B"][elem_keep],
        "detJ": sl["detJ"][elem_keep],
        "own_node": sl["own_node"][node_keep],
        "f_local": f_local,
        "f_magnitude": f_magnitude,
    }


def _view_from_sub(sl, sub):
    return {"conn": sub["conn"], "B": sub["B"], "detJ": sub["detJ"],
            "X_node": sub["X_node"], "own": sub["own_node"],
            "node_ids_global": sl["node_ids"][sub["node_local"]],
            "X_elem": sl["X_elem"][sub["elem_local"]],
            "elem_vol": sl["elem_vol"][sub["elem_local"]],
            "f_local": sub["f_local"], "f_magnitude": sub["f_magnitude"],
            "A_u": sl["A_u"][sub["node_local"]] if sl.get("A_u") is not None else None,
            "A_v": sl["A_v"][sub["node_local"]] if sl.get("A_v") is not None else None,
            "A_w": sl["A_w"][sub["node_local"]] if sl.get("A_w") is not None else None}


def _slice_view(sl, void_thresh, phase_weights_full):
    """Element/node 'view' this step operates on (all elems, or active non-void
    subset) plus the density restricted to it (grad kept)."""
    if void_thresh is None:
        return {"conn": sl["conn"], "B": sl["B"], "detJ": sl["detJ"],
                "X_node": sl["X_node"], "own": sl["own_node"],
                "node_ids_global": sl["node_ids"], "X_elem": sl["X_elem"],
                "elem_vol": sl["elem_vol"],
                "f_local": sl["f_local"], "f_magnitude": sl["f_magnitude"],
                "A_u": sl.get("A_u"), "A_v": sl.get("A_v"), "A_w": sl.get("A_w")}, phase_weights_full
    am = phase_weights_full.detach()[:, 0] >= void_thresh
    if am.sum() == 0:
        return None, None
    sub = _restrict_to_active(sl, am)
    return _view_from_sub(sl, sub), phase_weights_full[sub["elem_local"]]


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
            view, pw = _slice_view(sl, None, pw_full)
            return view, pw, mass, None
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


def _assemble_fadj(F_adj, view, E11d, E22d, E33d, G12d, G23d, G13d, pw_det, MP, lam0_c, mu0_c):
    """3D adjoint (stress) force from DETACHED primal strains (mirrors the
    precompute block). Scatter-adds this view's contribution into the flat
    3*Nn F_adj. Returns the 6 detached stresses + the stress-violation s for the
    compliance objective."""
    nu = MP['nu']; sy = MP['sy']; p = MP['p']
    ne = view["conn"].shape[0]
    dev = E11d.device; dtp = E11d.dtype
    lam0 = lam0_c * torch.ones((ne, 1), dtype=dtp, device=dev)
    mu0 = mu0_c * torch.ones((ne, 1), dtype=dtp, device=dev)
    tre0 = E11d + E22d + E33d
    s11_0 = lam0 * tre0 + 2.0 * mu0 * E11d
    s22_0 = lam0 * tre0 + 2.0 * mu0 * E22d
    s33_0 = lam0 * tre0 + 2.0 * mu0 * E33d
    s12_0 = mu0 * G12d
    s23_0 = mu0 * G23d
    s13_0 = mu0 * G13d
    s_vm0 = torch.sqrt(
        0.5 * ((s11_0 - s22_0) ** 2 + (s22_0 - s33_0) ** 2 + (s33_0 - s11_0) ** 2)
        + 3.0 * (s12_0 ** 2 + s23_0 ** 2 + s13_0 ** 2))
    s_vm_safe0 = torch.clamp(s_vm0, min=1e-8)
    s = s_vm_safe0 / sy - 1.0
    mE2 = (1e-6 + (1 - 1e-6) * pw_det ** p)
    ps = 2
    temp1 = (s + 1) ** ps
    h = torch.clamp(mE2 * s * temp1, min=0.0)
    dg = mE2 * (temp1 + ps * s * (s + 1) ** (ps - 1)) / sy
    hdh = h * (dg * (s > 0))
    dsvm_ds11 = (2.0 * s11_0 - s22_0 - s33_0) / (2.0 * s_vm_safe0)
    dsvm_ds22 = (2.0 * s22_0 - s11_0 - s33_0) / (2.0 * s_vm_safe0)
    dsvm_ds33 = (2.0 * s33_0 - s11_0 - s22_0) / (2.0 * s_vm_safe0)
    dsvm_ds12 = (3.0 * s12_0) / s_vm_safe0
    dsvm_ds23 = (3.0 * s23_0) / s_vm_safe0
    dsvm_ds13 = (3.0 * s13_0) / s_vm_safe0
    dsvm_de = torch.stack([
        (lam0 + 2.0 * mu0) * dsvm_ds11 + lam0 * dsvm_ds22 + lam0 * dsvm_ds33,
        lam0 * dsvm_ds11 + (lam0 + 2.0 * mu0) * dsvm_ds22 + lam0 * dsvm_ds33,
        lam0 * dsvm_ds11 + lam0 * dsvm_ds22 + (lam0 + 2.0 * mu0) * dsvm_ds33,
        mu0 * dsvm_ds12, mu0 * dsvm_ds23, mu0 * dsvm_ds13], dim=-1)
    dsvm_dU = torch.matmul(view["B"].transpose(-2, -1), dsvm_de.unsqueeze(-1)).squeeze(-1)
    f_adj_elem = -torch.sum(hdh.unsqueeze(-1) * dsvm_dU, dim=1)          # [ne, 24]
    conn_g = view["node_ids_global"][view["conn"]]                      # [ne, 8] global
    dof_idx = torch.stack([3 * conn_g, 3 * conn_g + 1, 3 * conn_g + 2], dim=-1).reshape(-1)
    F_adj.scatter_add_(0, dof_idx, f_adj_elem.reshape(-1))
    return s11_0, s22_0, s33_0, s12_0, s23_0, s13_0, s


def _finalize_fadj(F_adj, Nn, F_scale_goal=1e5):
    Fx = F_adj[0::3].unsqueeze(-1)
    Fy = F_adj[1::3].unsqueeze(-1)
    Fz = F_adj[2::3].unsqueeze(-1)
    F3 = torch.cat((Fx, Fy, Fz), dim=1)
    alpha = (torch.max(torch.abs(F3)) + 1e-8) / F_scale_goal    # target max|F_adj_scaled| = F_scale_goal (3D default 1e5)
    return F3 / alpha, alpha


def _finish_merged_log(log, grey_counts, count_total, active_total, M0, alpha_val):
    log["massfrac"] = log["mass"] / M0
    log["alpha"] = alpha_val
    log["grey_counts"] = grey_counts.detach().cpu().tolist()
    log["count_total"] = count_total
    log["active_total"] = active_total
    log["M0"] = M0
    log["grey_fraction"] = (grey_counts / max(count_total, 1)).detach().cpu().tolist()
    return log


def accumulate_merged_lag_nmask(model_list, slices, wd, wc, wm, delta,
                                include_adj, include_obj_mass, F_adj_scaled_lag,
                                alpha_lag, void_thresh, mask_cache=None,
                                return_raw_fadj=False):
    """3D merged LAGGED-F_adj (single sweep) + N_MASK density-restriction.
    Returns (log, F_adj_scaled_new, alpha_new, new_mask_cache), or with
    return_raw_fadj=True returns (log, F_adj_raw [3*Nn], new_mask_cache)."""
    m0 = model_list[0]; MP = m0.MP
    num_phase = MP['num_phase']; E_tensor = MP['E'][0]; wt = MP['wt'][0]
    nu = MP['nu']; p = MP['p']; b = MP['b']
    f_factor = MP['f_factor']; rho_min = MP['rho_min']; rho_max = MP['rho_max']
    dev = m0.collocation_x.device; dtp = m0.collocation_x.dtype
    Nn = m0.collocation_x.shape[0]; N_elem_global = m0.elem_x.shape[0]
    M0 = m0.elem_vol.sum()
    lam0_c = E_tensor * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu0_c = E_tensor / (2.0 * (1.0 + nu))
    for model in model_list:
        model.train()
    log = {k: 0.0 for k in ["loss_total", "strain_energy", "external_work",
                            "strain_energy_adj", "body_energy_adj", "obj", "mass"]}
    grey_counts = torch.zeros(num_phase, device=dev); count_total = 0; active_total = 0
    F_adj = torch.zeros(3 * Nn, dtype=dtp, device=dev); new_masks = []
    stress_viol_count_solid = 0   # solids (rho>=0.5) with MEAN-IP physical svm/sy>1 (matches plot_stress_figure)
    solid_count = 0               # solids (rho>=0.5); denominator of stress_violation_1
    n_h_activate = 0             # active elements (any rho) with ANY-IP h>0 (physical over-yield)
    n_h_activate_solid = 0       # solids (rho>=0.5) with ANY-IP h>0

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

        mE_no_grad = (1e-6 + (1 - 1e-6) * pw ** 3).detach()
        E = E_tensor * mE_no_grad
        lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))

        # single sweep: disp + adjoint forwards
        Node_disp = _disp_fields_analytic(model_list, view["X_node"], view["A_u"], view["A_v"], view["A_w"])
        E11, E22, E33, G12, G23, G13 = _elem_strains(Node_disp, view["conn"], view["B"])
        trE = E11 + E22 + E33
        S11 = lam * trE + 2.0 * mu * E11; S22 = lam * trE + 2.0 * mu * E22
        S33 = lam * trE + 2.0 * mu * E33
        S12 = mu * G12; S23 = mu * G23; S13 = mu * G13
        SE = torch.sum(torch.sum(0.5 * (S11 * E11 + S22 * E22 + S33 * E33
                                        + S12 * G12 + S23 * G23 + S13 * G13) * wt * view["detJ"], dim=1))
        L = wd * SE
        EW = Node_disp.new_zeros(())
        if view["f_local"].numel() > 0:
            EW = torch.sum(Node_disp[view["f_local"]] * view["f_magnitude"] * f_factor)
            L = L - wd * EW + wd * (1 + delta) / 2 * EW.detach()

        if include_adj:
            Node_test = _adj_fields_analytic(model_list, view["X_node"], view["A_u"], view["A_v"], view["A_w"])
            E11a, E22a, E33a, G12a, G23a, G13a = _elem_strains(Node_test, view["conn"], view["B"])
            trEa = E11a + E22a + E33a
            S11a = lam * trEa + 2.0 * mu * E11a; S22a = lam * trEa + 2.0 * mu * E22a
            S33a = lam * trEa + 2.0 * mu * E33a
            S12a = mu * G12a; S23a = mu * G23a; S13a = mu * G13a
            SE_adj = torch.sum(torch.sum(0.5 * (S11a * E11a + S22a * E22a + S33a * E33a
                                                + S12a * G12a + S23a * G23a + S13a * G13a) * wt * view["detJ"], dim=1))
            F_own = F_adj_scaled_lag[view["node_ids_global"]][view["own"]]   # LAGGED F_adj
            body_energy_adj = -(1 / N_elem_global) * torch.sum(Node_test[view["own"]] * F_own)
            L = L + wd * (SE_adj + body_energy_adj)
            log["strain_energy_adj"] += SE_adj.item(); log["body_energy_adj"] += body_energy_adj.item()

        # assemble the NEW F_adj (detached) for next iteration
        s_det = _assemble_fadj(F_adj, view, E11.detach(), E22.detach(), E33.detach(),
                               G12.detach(), G23.detach(), G13.detach(),
                               pw.detach(), MP, lam0_c, mu0_c)

        if include_obj_mass:
            s11_0, s22_0, s33_0, s12_0, s23_0, s13_0, s = s_det
            mE2 = (1e-6 + (1 - 1e-6) * pw ** p)                    # active density, grad kept
            e11a, e22a, e33a = E11a.detach(), E22a.detach(), E33a.detach()
            g12a, g23a, g13a = G12a.detach(), G23a.detach(), G13a.detach()
            comp = torch.sum(torch.sum(mE2 * (s11_0 * e11a + s22_0 * e22a + s33_0 * e33a
                                              + s12_0 * g12a + s23_0 * g23a + s13_0 * g13a)
                                       * wt * view["detJ"] * alpha_lag, dim=1))
            h = torch.clamp(mE2 * (s * (s + 1) ** 2).detach(), min=0.0)
            # --- stress-violation counts (all PHYSICAL C0: over yield <=> svm0/sy>1;
            #     h>0 <=> s>0). MEAN-over-IP for the solid count, ANY-IP (h>0) for the
            #     activation counts. See driver header for definitions/ratios. ---
            _solid = (pw[:, 0].detach() >= 0.5)                    # [N_elem] solid mask (rho>=0.5)
            _phys_ratio_mean = (s + 1.0).mean(dim=1)               # MEAN-over-IP physical svm/sy
            _h_active = (h > 0).any(dim=1)                         # ANY-IP h>0 (physical over-yield)
            stress_viol_count_solid += int((_solid & (_phys_ratio_mean > 1.0)).sum().item())
            solid_count             += int(_solid.sum().item())
            n_h_activate            += int(_h_active.sum().item())
            n_h_activate_solid      += int((_h_active & _solid).sum().item())
            body = torch.sum((1 / N_elem_global) * torch.sum(0.5 * h ** 2, dim=1))
            obj = (comp + MP.get('ALPHA_B', 1.0) * body) if MP.get('INCLUDE_BODY', True) else comp   # INCLUDE_BODY on/off + ALPHA_B body factor
            L = L + wc * obj + wm * mass / M0
            log["obj"] += obj.item(); log["mass"] += float(mass.item())
        else:
            log["mass"] += float(mass.item())

        L.backward()
        log["strain_energy"] += SE.item(); log["external_work"] += float(EW.item()); log["loss_total"] += L.item()

    log["stress_viol_count_solid"] = stress_viol_count_solid   # driver all-reduces (SUM)
    log["solid_count"] = solid_count                           # driver all-reduces (SUM)
    log["n_h_activate"] = n_h_activate                         # driver all-reduces (SUM)
    log["n_h_activate_solid"] = n_h_activate_solid             # driver all-reduces (SUM)
    if return_raw_fadj:
        out = _finish_merged_log(log, grey_counts, count_total, active_total, float(M0), 0.0)
        return out, F_adj, new_masks
    F_adj_scaled_new, alpha_new = _finalize_fadj(F_adj, Nn, MP.get('F_scale_goal', 1e5))
    out = _finish_merged_log(log, grey_counts, count_total, active_total, float(M0), float(alpha_new.item()))
    return out, F_adj_scaled_new, alpha_new, new_masks
