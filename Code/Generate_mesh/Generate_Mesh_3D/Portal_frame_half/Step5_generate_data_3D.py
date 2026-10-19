import math
import os

import torch


def read_abaqus_rpt_disp_3d(filepath, device="cpu"):
    """
    Read an Abaqus .rpt file containing nodal U1, U2, U3.

    Returns
    -------
    node_ids : torch.LongTensor [N_nodes]
    disp     : torch.Tensor [N_nodes, 3]
    """
    node_map = {}

    with open(filepath, "r") as f:
        header_found = False
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "Part Instance" in line and "U, U1" in line:
                header_found = True
                continue
            if not header_found or line.startswith("-"):
                continue

            parts = line.split()
            try:
                node_id = int(parts[1])
                values = [float(v) for v in parts[-3:]]
                if node_id not in node_map:
                    node_map[node_id] = values
            except Exception:
                continue

    node_ids = sorted(node_map.keys())
    disp_vals = [node_map[n] for n in node_ids]
    return (
        torch.tensor(node_ids, dtype=torch.long, device=device),
        torch.tensor(disp_vals, dtype=torch.float32, device=device),
    )


def read_abaqus_rpt_tensor_3d(filepath, field_prefix, device="cpu"):
    """
    Read 3D element tensor output from an Abaqus .rpt file.

    The returned component order is:
      11, 22, 33, 12, 13, 23
    """
    from collections import defaultdict

    elem_map = defaultdict(dict)

    with open(filepath, "r") as f:
        header_found = False
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "Part Instance" in line and f"{field_prefix}, {field_prefix}11" in line:
                header_found = True
                continue
            if not header_found or line.startswith("-"):
                continue

            parts = line.split()
            try:
                elem_id = int(parts[1])
                int_pt = int(parts[3])
                values = [float(v) for v in parts[-6:]]
                elem_map[elem_id][int_pt] = values
            except Exception:
                continue

    elem_ids = sorted(elem_map.keys())
    if len(elem_ids) == 0:
        raise ValueError(f"No {field_prefix} tensor data found in {filepath}")

    n_int = max(len(v) for v in elem_map.values())
    values = torch.zeros((len(elem_ids), n_int, 6), dtype=torch.float32, device=device)

    for i, eid in enumerate(elem_ids):
        for ip, vals in elem_map[eid].items():
            values[i, ip - 1, :] = torch.tensor(vals, dtype=torch.float32, device=device)

    return torch.tensor(elem_ids, dtype=torch.long, device=device), values


def read_abaqus_rpt_strain_3d(filepath, device="cpu"):
    return read_abaqus_rpt_tensor_3d(filepath, field_prefix="E", device=device)


def read_abaqus_rpt_stress_3d(filepath, device="cpu"):
    return read_abaqus_rpt_tensor_3d(filepath, field_prefix="S", device=device)


def build_B_C3D4(X_node, conn):
    """
    Build the constant strain-displacement B matrix for linear tetrahedra.

    Returns
    -------
    B_batch : torch.Tensor [N_elems, 6, 12]
        Strain order: exx, eyy, ezz, gamma_xy, gamma_yz, gamma_xz.
    detJ : torch.Tensor [N_elems]
        Signed Jacobian determinant. The physical tet volume is abs(detJ) / 6.
    """
    elem_nodes = X_node[conn]  # [N_elems, 4, 3]
    n_elems = elem_nodes.shape[0]
    dtype = X_node.dtype
    device = X_node.device

    dN_dnatural = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )

    edge_1 = elem_nodes[:, 1, :] - elem_nodes[:, 0, :]
    edge_2 = elem_nodes[:, 2, :] - elem_nodes[:, 0, :]
    edge_3 = elem_nodes[:, 3, :] - elem_nodes[:, 0, :]
    jac = torch.stack((edge_1, edge_2, edge_3), dim=2)  # [N_elems, 3, 3]

    detJ = torch.linalg.det(jac)
    if torch.any(torch.isclose(detJ, torch.zeros((), dtype=dtype, device=device))):
        raise ValueError("At least one C3D4 element has near-zero volume.")

    inv_jac = torch.linalg.inv(jac)
    dN_dx = torch.matmul(dN_dnatural.unsqueeze(0).expand(n_elems, -1, -1), inv_jac)

    dN_x = dN_dx[:, :, 0]
    dN_y = dN_dx[:, :, 1]
    dN_z = dN_dx[:, :, 2]

    B_batch = torch.zeros((n_elems, 6, 12), dtype=dtype, device=device)

    B_batch[:, 0, 0::3] = dN_x
    B_batch[:, 1, 1::3] = dN_y
    B_batch[:, 2, 2::3] = dN_z

    B_batch[:, 3, 0::3] = dN_y
    B_batch[:, 3, 1::3] = dN_x

    B_batch[:, 4, 1::3] = dN_z
    B_batch[:, 4, 2::3] = dN_y

    B_batch[:, 5, 0::3] = dN_z
    B_batch[:, 5, 2::3] = dN_x

    return B_batch, detJ


def build_B_C3D8(X_node, conn, reduced=False):
    """
    Build the strain-displacement B matrices for linear (trilinear) hexahedra.

    reduced=False -> full 2x2x2 Gauss rule, 8 integration points (Abaqus C3D8),
                     each Gauss weight is 1.
    reduced=True  -> single centroid integration point (Abaqus C3D8R, reduced
                     integration), with integration weight 8 (= 2x2x2).

    Node ordering follows Gmsh/Abaqus C3D8:
        0:(-1,-1,-1) 1:(+1,-1,-1) 2:(+1,+1,-1) 3:(-1,+1,-1)
        4:(-1,-1,+1) 5:(+1,-1,+1) 6:(+1,+1,+1) 7:(-1,+1,+1)

    Returns
    -------
    B_batch : torch.Tensor [N_elems, n_int, 6, 24]
        n_int = 8 (full) or 1 (reduced).
        Strain order: exx, eyy, ezz, gamma_xy, gamma_yz, gamma_xz.
        DOF order: [u0x, u0y, u0z, u1x, ...] (node-major, 3 dof/node).
    detJ : torch.Tensor [N_elems, n_int]
        Jacobian determinant at each integration point. The element volume is
        sum_ip(w_ip * detJ_ip), with w_ip = 1 (full) or 8 (reduced).
    """
    elem_nodes = X_node[conn]  # [N_elems, 8, 3]
    n_elems = elem_nodes.shape[0]
    dtype = X_node.dtype
    device = X_node.device

    # Natural coordinates of the 8 corner nodes.
    nat = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )

    if reduced:
        # Single centroid integration point (reduced integration, C3D8R).
        gauss_pts = torch.zeros((1, 3), dtype=dtype, device=device)
    else:
        # 2-point Gauss rule per direction (weight 1 each) -> 8 points.
        g = 1.0 / math.sqrt(3.0)
        gauss_pts = torch.tensor(
            [
                [sx * g, sy * g, sz * g]
                for sz in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sx in (-1.0, 1.0)
            ],
            dtype=dtype,
            device=device,
        )

    n_int = gauss_pts.shape[0]

    xi_i = nat[:, 0]
    eta_i = nat[:, 1]
    zeta_i = nat[:, 2]

    B_batch = torch.zeros((n_elems, n_int, 6, 24), dtype=dtype, device=device)
    detJ = torch.zeros((n_elems, n_int), dtype=dtype, device=device)

    for gp, (xi, eta, zeta) in enumerate(gauss_pts):
        a = 1.0 + xi * xi_i
        b = 1.0 + eta * eta_i
        c = 1.0 + zeta * zeta_i

        # dN/d(natural) at this Gauss point, shape [8, 3].
        dN_dnat = torch.stack(
            (
                0.125 * xi_i * b * c,
                0.125 * a * eta_i * c,
                0.125 * a * b * zeta_i,
            ),
            dim=1,
        )

        # Jacobian J[n,i,j] = d x_j / d xi_i = sum_k dN_dnat[k,i] * x[n,k,j]
        jac = torch.einsum("ki,nkj->nij", dN_dnat, elem_nodes)  # [N, 3, 3]
        detJ_gp = torch.linalg.det(jac)
        if torch.any(torch.isclose(detJ_gp, torch.zeros((), dtype=dtype, device=device))):
            raise ValueError("At least one C3D8 element has a degenerate Jacobian.")

        inv_jac = torch.linalg.inv(jac)
        # d xi_i / d x_j = (J^-1)^T[i,j] = inv_jac[j,i], so
        # dN/dx[n,k,j] = sum_i dN_dnat[k,i] * d xi_i / d x_j = sum_i dN_dnat[k,i] * inv_jac[n,j,i]
        dN_dx = torch.einsum("ki,nji->nkj", dN_dnat, inv_jac)  # [N, 8, 3]

        dN_x = dN_dx[:, :, 0]
        dN_y = dN_dx[:, :, 1]
        dN_z = dN_dx[:, :, 2]

        B_gp = torch.zeros((n_elems, 6, 24), dtype=dtype, device=device)
        B_gp[:, 0, 0::3] = dN_x
        B_gp[:, 1, 1::3] = dN_y
        B_gp[:, 2, 2::3] = dN_z

        B_gp[:, 3, 0::3] = dN_y
        B_gp[:, 3, 1::3] = dN_x

        B_gp[:, 4, 1::3] = dN_z
        B_gp[:, 4, 2::3] = dN_y

        B_gp[:, 5, 0::3] = dN_z
        B_gp[:, 5, 2::3] = dN_x

        B_batch[:, gp, :, :] = B_gp
        detJ[:, gp] = detJ_gp

    return B_batch, detJ


def find_l_beam_3d_load_nodes(X_node, tol=1e-6):
    x = X_node[:, 0]
    y = X_node[:, 1]

    x_target = torch.tensor(100.0, dtype=X_node.dtype, device=X_node.device)
    mask = (
        torch.isclose(x, x_target, atol=tol)
        & (y >= 35.0 - tol)
        & (y <= 40.0 + tol)
    )
    return torch.nonzero(mask, as_tuple=False).squeeze(-1)


def find_portal_frame_half_3d_load_nodes(X_node, tol=1e-6):
    x = X_node[:, 0]
    y = X_node[:, 1]

    y_target = torch.tensor(60.0, dtype=X_node.dtype, device=X_node.device)
    mask = (
        torch.isclose(y, y_target, atol=tol)
        & (x >= 0.0 - tol)
        & (x <= 5.0 + tol)
    )
    return torch.nonzero(mask, as_tuple=False).squeeze(-1)


def build_mesh_slices_3d(
    mesh_data,
    example="L_beam",
    n_worker=1,
    diff_type="C3D4",
):
    if diff_type not in ("C3D4", "C3D8", "C3D8R"):
        raise NotImplementedError(
            "The 3D data generator supports C3D4, C3D8 and C3D8R, "
            f"got {diff_type!r}."
        )
    expected_nodes = 4 if diff_type == "C3D4" else 8

    all_gpu_mesh_slices = {f"GPU{w}": {} for w in range(n_worker)}

    for mesh_name, mesh in mesh_data.items():
        print(f"Processing {mesh_name} ...")

        X_node = mesh["X_node"]
        conn = mesh["conn"]
        X_elem = mesh["X_elem"]
        node_ids = mesh.get("node_ids", None)
        elem_ids = mesh.get("elem_ids", None)
        n_elems = X_elem.shape[0]

        if X_node.shape[1] != 3:
            raise ValueError(f"{mesh_name} X_node is not 3D: shape={X_node.shape}")
        if X_elem.shape[1] != 3:
            raise ValueError(f"{mesh_name} X_elem is not 3D: shape={X_elem.shape}")
        if conn.shape[1] != expected_nodes:
            raise ValueError(
                f"{mesh_name} connectivity is not {diff_type} "
                f"(expected {expected_nodes} nodes/element): shape={conn.shape}"
            )

        tol = 1e-6
        if example == "L_beam":
            f_index = find_l_beam_3d_load_nodes(X_node, tol=tol)
            total_force = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)
        elif example == "Portal_frame_half":
            f_index = find_portal_frame_half_3d_load_nodes(X_node, tol=tol)
            total_force = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown 3D example: {example}")

        n_loaded = f_index.numel()
        if n_loaded == 0:
            raise ValueError(f"Did not find 3D loading nodes for {mesh_name}.")

        f_magnitude = total_force.unsqueeze(0).repeat(n_loaded, 1) / n_loaded
        f_index_dup = f_index.clone()
        f_mag_dup = f_magnitude.clone()

        elems_per_worker = n_elems // n_worker

        for w in range(n_worker):
            print(f"  worker {w}")

            start = w * elems_per_worker
            end = (w + 1) * elems_per_worker if w < n_worker - 1 else n_elems

            X_elem_w = X_elem[start:end]
            conn_w_global = conn[start:end]

            node_idx_w, inv = torch.unique(
                conn_w_global.flatten(), sorted=True, return_inverse=True
            )
            X_node_w = X_node[node_idx_w]
            conn_w = inv.view_as(conn_w_global)

            if diff_type in ("C3D8", "C3D8R"):
                # Full integration (C3D8): 8 Gauss points, weight 1 each.
                # Reduced integration (C3D8R): 1 centroid point, weight 8.
                reduced = diff_type == "C3D8R"
                hex_weight = 8.0 if reduced else 1.0
                B_batch_w, detJ_w_all = build_B_C3D8(
                    X_node_w, conn_w, reduced=reduced
                )
                B_w = B_batch_w  # [N_elem_w, n_int, 6, 24]
                detJ_w = detJ_w_all  # [N_elem_w, n_int]
                n_int = detJ_w_all.shape[1]
                wt_int = torch.full((n_int,), hex_weight, dtype=torch.float32)
                elem_vol_w = (detJ_w_all.abs() * hex_weight).sum(dim=1)
            else:
                # Single constant-strain integration point for linear tets.
                B_batch_w, detJ_w_single = build_B_C3D4(X_node_w, conn_w)
                B_w = B_batch_w.unsqueeze(1)  # [N_elem_w, 1, 6, 12]
                detJ_w = detJ_w_single.unsqueeze(1)  # [N_elem_w, 1]
                wt_int = torch.tensor([1.0 / 6.0], dtype=torch.float32)
                elem_vol_w = torch.abs(detJ_w_single) / 6.0

            if f_index_dup.numel() == 0:
                f_index_w = torch.zeros((0,), dtype=torch.long)
                f_magnitude_w = torch.zeros((0, 3), dtype=torch.float32)
            else:
                f_index_w_list = []
                f_magnitude_w_list = []
                index_remove = []

                for j, node in enumerate(f_index_dup.tolist()):
                    hits = (node_idx_w == node).nonzero(as_tuple=True)[0]
                    if hits.numel() > 0:
                        f_index_w_list.append(hits.item())
                        f_magnitude_w_list.append(f_mag_dup[j].tolist())
                        index_remove.append(j)

                if len(index_remove) > 0:
                    f_index_w = torch.tensor(f_index_w_list, dtype=torch.long)
                    f_magnitude_w = torch.tensor(f_magnitude_w_list, dtype=torch.float32)

                    keep = torch.ones(f_index_dup.shape[0], dtype=torch.bool)
                    keep[index_remove] = False
                    f_index_dup = f_index_dup[keep]
                    f_mag_dup = f_mag_dup[keep]
                else:
                    f_index_w = torch.zeros((0,), dtype=torch.long)
                    f_magnitude_w = torch.zeros((0, 3), dtype=torch.float32)

            slice_data = {
                "X_elem": X_elem_w,
                "X_node": X_node_w,
                "node_idx": node_idx_w,
                "conn": conn_w,
                "f_index": f_index_w,
                "f_magnitude": f_magnitude_w,
                "B": B_w,
                "detJ": detJ_w,
                "elem_vol": elem_vol_w,
                "elem_type": diff_type,
                "dof_per_node": 3,
                "strain_components": [
                    "exx",
                    "eyy",
                    "ezz",
                    "gamma_xy",
                    "gamma_yz",
                    "gamma_xz",
                ],
                "wt_int": wt_int,
            }

            if node_ids is not None:
                slice_data["abaqus_node_ids"] = node_ids[node_idx_w]
            if elem_ids is not None:
                slice_data["abaqus_elem_ids"] = elem_ids[start:end]

            all_gpu_mesh_slices[f"GPU{w}"][mesh_name] = slice_data

    return all_gpu_mesh_slices


if __name__ == "__main__":
    example = "Portal_frame_half"
    mode = ""
    N_worker = 1
    Diff_type = "C3D8R" # C3D4 C3D8 (full) C3D8R (reduced)

    if mode == "":
        mesh_path = f"{example}_3D_mesh.pt"
        filename_save = f"{example}_3D_GPU{N_worker}_{Diff_type}"
    else:
        mesh_path = f"{example}_3D_mesh_{mode}.pt"
        filename_save = f"{example}_3D_GPU{N_worker}_{Diff_type}_{mode}"

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(
            f"Missing {mesh_path}. Run Step4_process_mesh_3D.py first."
        )

    mesh_data = torch.load(mesh_path)
    all_gpu_mesh_slices = build_mesh_slices_3d(
        mesh_data=mesh_data,
        example=example,
        n_worker=N_worker,
        diff_type=Diff_type,
    )

    torch.save(all_gpu_mesh_slices, f"{filename_save}.pt")
    print(f"Saved all 3D mesh slices to {filename_save}.pt")
