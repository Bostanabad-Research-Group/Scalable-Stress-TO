import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def read_abaqus_rpt_disp(filepath, device="cpu"):
    """
    Read Abaqus .rpt file containing nodal displacements (U1, U2).
    
    Args:
        filepath (str): Path to the .rpt file.
        device (str): Torch device ("cpu" or "cuda").
    
    Returns:
        node_ids (torch.LongTensor): [N_nodes]
        disp (torch.Tensor): [N_nodes, 2] displacements (U1, U2)
    """
    node_map = {}  # NodeID -> (U1, U2)

    with open(filepath, "r") as f:
        header_found = False
        for line in f:
            line = line.strip()
            if not line:
                continue

            # detect header line
            if "Part Instance" in line and "U, U1" in line:
                header_found = True
                continue
            if not header_found or line.startswith("-"):
                continue

            parts = line.split()
            try:
                node_id = int(parts[1])
                u1 = float(parts[-2])
                u2 = float(parts[-1])
                # keep only the first entry for each node
                if node_id not in node_map:
                    node_map[node_id] = (u1, u2)
            except Exception:
                continue

    # sort by node ID to keep consistent ordering
    node_ids = sorted(node_map.keys())
    disp_vals = [node_map[n] for n in node_ids]

    node_ids = torch.tensor(node_ids, dtype=torch.long, device=device)
    disp = torch.tensor(disp_vals, dtype=torch.float32, device=device)

    return node_ids, disp

def read_abaqus_rpt_strain(filepath, device="cpu"):
    """
    Read Abaqus .rpt file containing element strains (E11, E22, E12).
    Works for both reduced integration (1 IP) and full integration (4 IP).

    Args:
        filepath (str): Path to the .rpt file.
        device (str): Torch device ("cpu" or "cuda").

    Returns:
        elem_ids (torch.LongTensor): [N_elems]
        strains (torch.Tensor): [N_elems, N_int_pts, 3]
                               where N_int_pts = 1 for CPS4R, 4 for CPS4, etc.
    """
    from collections import defaultdict
    elem_map = defaultdict(dict)  # elem_id -> {int_pt: (E11,E22,E12)}

    with open(filepath, "r") as f:
        header_found = False
        for line in f:
            line = line.strip()
            if not line:
                continue

            # detect header line
            if "Part Instance" in line and "E, E11" in line:
                header_found = True
                continue
            if not header_found or line.startswith("-"):
                continue

            parts = line.split()
            try:
                elem_id = int(parts[1])
                int_pt = int(parts[3])
                e11 = float(parts[4])
                e22 = float(parts[5])
                e12 = float(parts[7])  # skip E33
                elem_map[elem_id][int_pt] = (e11, e22, e12)
            except Exception:
                continue

    # sort element IDs
    elem_ids = sorted(elem_map.keys())
    N_elem = len(elem_ids)
    # determine number of integration points (should be uniform across all elems)
    N_int = max(len(v) for v in elem_map.values())

    # allocate tensor [N_elem, N_int, 3]
    strains = torch.zeros((N_elem, N_int, 3), dtype=torch.float32, device=device)

    for i, eid in enumerate(elem_ids):
        for ip, vals in elem_map[eid].items():
            strains[i, ip-1, :] = torch.tensor(vals, dtype=torch.float32)

    elem_ids = torch.tensor(elem_ids, dtype=torch.long, device=device)
    return elem_ids, strains

def read_abaqus_rpt_stress(filepath, device="cpu"):
    """
    Read Abaqus .rpt file containing element stresses (S11, S22, S12).
    Works for both reduced integration (1 IP) and full integration (4 IP).

    Args:
        filepath (str): Path to the .rpt file.
        device (str): Torch device ("cpu" or "cuda").

    Returns:
        elem_ids (torch.LongTensor): [N_elems]
        stresses (torch.Tensor): [N_elems, N_int_pts, 3]
                                where N_int_pts = 1 for CPS4R, 4 for CPS4, etc.
                                The last dimension is (S11, S22, S12).
    """
    from collections import defaultdict
    elem_map = defaultdict(dict)  # elem_id -> {int_pt: (S11,S22,S12)}

    with open(filepath, "r") as f:
        header_found = False
        for line in f:
            line = line.strip()
            if not line:
                continue

            # detect header line
            if "Part Instance" in line and "S, S11" in line:
                header_found = True
                continue
            if not header_found or line.startswith("-"):
                continue

            parts = line.split()
            try:
                elem_id = int(parts[1])
                int_pt = int(parts[3])
                s11 = float(parts[4])
                s22 = float(parts[5])
                s12 = float(parts[7])  # skip S33
                elem_map[elem_id][int_pt] = (s11, s22, s12)
            except Exception:
                continue

    # sort element IDs
    elem_ids = sorted(elem_map.keys())
    N_elem = len(elem_ids)
    N_int = max(len(v) for v in elem_map.values())

    stresses = torch.zeros((N_elem, N_int, 3), dtype=torch.float32, device=device)
    for i, eid in enumerate(elem_ids):
        for ip, vals in elem_map[eid].items():
            stresses[i, ip-1, :] = torch.tensor(vals, dtype=torch.float32)

    elem_ids = torch.tensor(elem_ids, dtype=torch.long, device=device)
    return elem_ids, stresses

def build_B_2D(X_node, conn, elem_type="CPS4", int_pt=(0.0, 0.0), device="cpu"):
    """
    Build strain-displacement B matrices for 2D elements at given integration points.

    Args:
        X_node (torch.Tensor): [N_nodes, 2] nodal coordinates
        conn   (torch.LongTensor): [N_elems, n_nodes_per_elem] connectivity (0-based indices)
        elem_type (str): Element type string (currently supports 'CPS4')
        int_pt (tuple): Integration point coordinates (xi, eta) in parent domain
                        Default = (0,0) = reduced integration for CPS4
        device (str): torch device ("cpu" or "cuda")

    Returns:
        B_batch (torch.Tensor): [N_elems, 3, n_dofs_per_elem] strain-displacement matrices
        detJ    (torch.Tensor): [N_elems] determinant of Jacobian (for integration weights)
    """

    # ---------- Check supported element ----------
    if elem_type in {"CPS4", "CPS4_reduced"}:
        # Integration point in parent domain
        xi, eta = int_pt

        # Shape function derivatives wrt xi, eta
        dN_dxi = torch.tensor([
            -(1 - eta),  (1 - eta),
             (1 + eta), -(1 + eta)
        ], dtype=torch.float32, device=device) * 0.25

        dN_deta = torch.tensor([
            -(1 - xi), -(1 + xi),
             (1 + xi),  (1 - xi)
        ], dtype=torch.float32, device=device) * 0.25

        # Gather nodal coords for each element: [N_elems, 4, 2]
        elem_nodes = X_node[conn]

        # Build Jacobian per element
        J11 = torch.sum(dN_dxi * elem_nodes[:,:,0], dim=1)   # ∂x/∂ξ
        J12 = torch.sum(dN_dxi * elem_nodes[:,:,1], dim=1)   # ∂y/∂ξ
        J21 = torch.sum(dN_deta * elem_nodes[:,:,0], dim=1)  # ∂x/∂η
        J22 = torch.sum(dN_deta * elem_nodes[:,:,1], dim=1)  # ∂y/∂η

        detJ = J11 * J22 - J12 * J21
        invJ11 =  J22 / detJ
        invJ12 = -J12 / detJ
        invJ21 = -J21 / detJ
        invJ22 =  J11 / detJ

        # Map to global coordinates: [N_elems, 4]
        dN_dx = invJ11.unsqueeze(1) * dN_dxi + invJ12.unsqueeze(1) * dN_deta
        dN_dy = invJ21.unsqueeze(1) * dN_dxi + invJ22.unsqueeze(1) * dN_deta

        # Build B matrices: [N_elems, 3, 8]
        N_elems = conn.shape[0]
        B_batch = torch.zeros((N_elems, 3, 8), dtype=torch.float32, device=device)

        # Fill rows
        B_batch[:,0,0::2] = dN_dx  # εxx
        B_batch[:,1,1::2] = dN_dy  # εyy
        B_batch[:,2,0::2] = dN_dy  # γxy
        B_batch[:,2,1::2] = dN_dx  # γxy

    elif elem_type_upper == "CPS3":
        # ------------------------------
        # 3-node linear triangle
        # ------------------------------
        # Standard linear triangle shape functions:
        # N1 = 1 - xi - eta
        # N2 = xi
        # N3 = eta
        #
        # Their derivatives wrt parent coordinates are constant:
        dN_dxi = torch.tensor([-1.0, 1.0, 0.0], dtype=torch.float32, device=device)
        dN_deta = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, device=device)

        # Gather nodal coords for each element: [N_elems, 3, 2]
        elem_nodes = X_node[conn]

        # Build Jacobian per element
        J11 = torch.sum(dN_dxi  * elem_nodes[:, :, 0], dim=1)   # dx/dxi
        J12 = torch.sum(dN_dxi  * elem_nodes[:, :, 1], dim=1)   # dy/dxi
        J21 = torch.sum(dN_deta * elem_nodes[:, :, 0], dim=1)   # dx/deta
        J22 = torch.sum(dN_deta * elem_nodes[:, :, 1], dim=1)   # dy/deta

        detJ = J11 * J22 - J12 * J21
        if torch.any(detJ <= 0):
            raise ValueError("Some CPS3 elements have non-positive detJ. Check node ordering.")
        invJ11 =  J22 / detJ
        invJ12 = -J12 / detJ
        invJ21 = -J21 / detJ
        invJ22 =  J11 / detJ

        # Map to global coordinates: [N_elems, 3]
        dN_dx = (
            invJ11.unsqueeze(1) * dN_dxi.unsqueeze(0)
            + invJ12.unsqueeze(1) * dN_deta.unsqueeze(0)
        )
        dN_dy = (
            invJ21.unsqueeze(1) * dN_dxi.unsqueeze(0)
            + invJ22.unsqueeze(1) * dN_deta.unsqueeze(0)
        )

        # Build B matrices: [N_elems, 3, 6]
        N_elems = conn.shape[0]
        B_batch = torch.zeros((N_elems, 3, 6), dtype=torch.float32, device=device)

        B_batch[:, 0, 0::2] = dN_dx  # εxx
        B_batch[:, 1, 1::2] = dN_dy  # εyy
        B_batch[:, 2, 0::2] = dN_dy  # γxy
        B_batch[:, 2, 1::2] = dN_dx  # γxy
    else:
        raise NotImplementedError(f"Element type {elem_type} not supported yet.")

    return B_batch, detJ

example = 'Portal_frame_half'   # L_beam Hook Portal_frame Crack Portal_frame_half
mode = '' # empty is for regular training grids while pred is for prediction
N_worker = 1  # number of GPU
Diff_type = 'CPS4_reduced' # element type
gauss_coord = 1.0 / np.sqrt(3.0)
wt_int = (1.0, 1.0, 1.0, 1.0)
if Diff_type == 'CPS4_reduced':
    int_points = [(0.0, 0.0)]
    wt_int = (4.0,)
else:
    int_points = [
        (-gauss_coord, -gauss_coord),
        ( gauss_coord, -gauss_coord),
        ( gauss_coord,  gauss_coord),
        (-gauss_coord,  gauss_coord),
    ]
if mode == '':
        filename_save = f'{example}_GPU{N_worker}_{Diff_type}'
else:
        filename_save = f'{example}_GPU{N_worker}_{Diff_type}_{mode}'

# read the mesh data
if mode == '':
    mesh_data = torch.load(f'{example}_mesh.pt')
else:
    mesh_data = torch.load(f'{example}_mesh_{mode}.pt')

# storage for all GPUs
all_gpu_mesh_slices = {f"GPU{w}": {} for w in range(N_worker)}

for mesh_name, mesh in mesh_data.items():
    print(f"Processing {mesh_name} ...")

    X_node = mesh['X_node']
    conn = mesh['conn']
    X_elem = mesh['X_elem']
    N_elems = X_elem.shape[0]

    # === Define global force (per mesh) ===
    tol = 1e-6
    if example == 'L_beam':
        mask = (X_node[:, 1] >= 35.0) & (X_node[:, 1] <= 40.0) & torch.isclose(X_node[:, 0], torch.tensor(100.0), atol=tol)
        f_index = torch.nonzero(mask, as_tuple=False).squeeze(-1)   # global indices of loaded nodes
        total_force = torch.tensor([0.0, -1.0], dtype=torch.float32)
    
    elif example == 'Crack':
        mask = (X_node[:, 0] == 100.0) &  (X_node[:, 1] >= 190.0) & (X_node[:, 0] <= 200.0)
        f_index = torch.nonzero(mask, as_tuple=False).squeeze(-1)   # global indices of loaded nodes
        total_force = torch.tensor([1.0, 0.0], dtype=torch.float32)

    elif example == 'Portal_frame':
        mask = (X_node[:, 0] >= -5.0) &  (X_node[:, 0] <= 5.0) & (X_node[:, 1] == 60.0)
        f_index = torch.nonzero(mask, as_tuple=False).squeeze(-1)   # global indices of loaded nodes
        total_force = torch.tensor([0.0, -1.0], dtype=torch.float32)

    elif example == 'Portal_frame_half':
        mask = (X_node[:, 0] >= 0.0) &  (X_node[:, 0] <= 5.0) & (X_node[:, 1] == 60.0)
        f_index = torch.nonzero(mask, as_tuple=False).squeeze(-1)   # global indices of loaded nodes
        total_force = torch.tensor([0.0, -1.0], dtype=torch.float32)

    elif example == 'Hook':
        x = X_node[:, 0]
        y = X_node[:, 1]
        r = torch.sqrt(x**2 + y**2)
        mask = torch.isclose(r, torch.tensor(19.0, dtype=X_node.dtype, device=X_node.device), atol=tol) & (y <= 0.0)
        f_index = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        total_force = torch.tensor([0.0, -1.0], dtype=torch.float32)

    else:
        raise ValueError(f"Unknown example: {example}")
    
    # evenly distribute among loaded nodes
    n_loaded = f_index.numel()
    if n_loaded > 0:
        f_magnitude = total_force.unsqueeze(0).repeat(n_loaded, 1) / n_loaded
    else:
        raise ValueError(f"Did not find loading points!")

     # === Plot ===
    X_node_np = X_node.detach().cpu().numpy()
    f_index_np = f_index.detach().cpu().numpy() if n_loaded > 0 else []

    # plt.figure(figsize=(8, 6))

    # # all nodes
    # plt.scatter(X_node_np[:, 0], X_node_np[:, 1], s=10, label='All nodes')

    # # highlight external load points
    # if n_loaded > 0:
    #     plt.scatter(X_node_np[f_index_np, 0], X_node_np[f_index_np, 1],
    #                 s=40, marker='o', label='External load points')

    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Mesh nodes and external load points')
    # plt.axis('equal')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # duplicate global force arrays for assignment (duplication removal logic)
    f_index_dup = f_index.clone()
    f_mag_dup = f_magnitude.clone()

    elems_per_worker = N_elems // N_worker

    for w in range(N_worker):
        print(f"  worker {w}")

        # slice elements
        start = w * elems_per_worker
        end = (w+1) * elems_per_worker if w < N_worker-1 else N_elems

        X_elem_w = X_elem[start:end]
        conn_w_local = conn[start:end]
        N_elem_w = conn_w_local.shape[0]

        # unique local nodes
        node_ids_w, inv = torch.unique(conn_w_local.flatten(), sorted=True, return_inverse=True)
        X_node_w = X_node[node_ids_w]
        conn_w = inv.view_as(conn_w_local)

        # estimate the element volume (area)
        if Diff_type in {'CPS4', 'CPS4_reduced'}:
            X_elem_nodes = X_node_w[conn_w]  # [N_elem_w, 4, 2]
            x = X_elem_nodes[..., 0]  # shape [N_elem_w, 4]
            y = X_elem_nodes[..., 1]  # shape [N_elem_w, 4]
            # Shoelace formula (polygon area)
            # A = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|, i from 0 to n-1, with wrap-around
            x_next = torch.roll(x, shifts=-1, dims=1)
            y_next = torch.roll(y, shifts=-1, dims=1)
            areas = 0.5 * torch.abs(torch.sum(x * y_next - x_next * y, dim=1))  # shape [N_elem_w]
        elif Diff_type == 'CPS3':
            X_elem_nodes = X_node_w[conn_w]  # [N_elem_w, 3, 2]
            x1 = X_elem_nodes[:, 0, 0]
            y1 = X_elem_nodes[:, 0, 1]
            x2 = X_elem_nodes[:, 1, 0]
            y2 = X_elem_nodes[:, 1, 1]
            x3 = X_elem_nodes[:, 2, 0]
            y3 = X_elem_nodes[:, 2, 1]
            # Triangle area:
            # A = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
            areas = 0.5 * torch.abs(
                (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            )  # [N_elem_w]
            print('ok')
        else:
            raise NotImplementedError(f"Element type {Diff_type} not supported yet.")

        # === External work (duplication removal preserved) ===
        if f_index_dup.numel() == 0:
            f_index_w = torch.zeros((0,), dtype=torch.long)
            f_magnitude_w = torch.zeros((0, 2), dtype=torch.float32)
        else:
            f_index_w = []
            f_magnitude_w = []
            index_remove = []

            for j, node in enumerate(f_index_dup.tolist()):
                hits = (node_ids_w == node).nonzero(as_tuple=True)[0]
                if hits.numel() > 0:
                    local_idx = hits.item()
                    f_index_w.append(local_idx)
                    f_magnitude_w.append(f_mag_dup[j].tolist())
                    index_remove.append(j)

            if len(index_remove) > 0:
                f_index_w = torch.tensor(f_index_w, dtype=torch.long)
                f_magnitude_w = torch.tensor(f_magnitude_w, dtype=torch.float32)

                keep = torch.ones(f_index_dup.shape[0], dtype=torch.bool)
                keep[index_remove] = False
                f_index_dup = f_index_dup[keep]
                f_mag_dup = f_mag_dup[keep]
            else:
                f_index_w = torch.zeros((0,), dtype=torch.long)
                f_magnitude_w = torch.zeros((0, 2), dtype=torch.float32)

        # === loop over integration points ===
        B_list = []
        detJ_list = []
        for xi, eta in int_points:
            B_batch_w, detJ = build_B_2D(X_node_w, conn_w, elem_type=Diff_type, int_pt=(xi, eta))
            B_list.append(B_batch_w)
            detJ_list.append(detJ)
        B_w = torch.stack(B_list, dim=1)       # [n_elem_w, N_int, 3, dof_per_elem]
        detJ_w = torch.stack(detJ_list, dim=1) # [n_elem_w, N_int]

        # === Save slice info under GPU → Mesh ===
        all_gpu_mesh_slices[f"GPU{w}"][mesh_name] = {
            "X_elem": X_elem_w,
            "X_node": X_node_w,
            "node_idx": node_ids_w,
            "conn": conn_w,
            "f_index": f_index_w,
            "f_magnitude": f_magnitude_w,
            "B": B_w,
            "detJ": detJ_w,
            'elem_vol':areas,
        }

# === Save all precomputed slices ===
torch.save(all_gpu_mesh_slices, f"{filename_save}.pt")
print(f"Saved all mesh slices to {filename_save}.pt")




