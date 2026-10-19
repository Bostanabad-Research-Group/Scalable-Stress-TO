import torch
import numpy as np
from utils.utils_general import get_multiGPU
import matplotlib.pyplot as plt
import math
from types import SimpleNamespace

def _coordinate_tolerance(values):
    unique_vals = torch.unique(values.detach().cpu())
    unique_vals, _ = torch.sort(unique_vals)
    if unique_vals.numel() <= 1:
        return 1e-6

    diffs = unique_vals[1:] - unique_vals[:-1]
    diffs = diffs[diffs > 0]
    if diffs.numel() == 0:
        return 1e-6
    return max(1e-6, 0.25 * float(diffs.min().item()))

def _sort_and_sample_points(points, num_points):
    points = torch.unique(points.detach().cpu(), dim=0)
    if points.numel() == 0:
        return points

    sort_cols = [points[:, dim].numpy() for dim in range(points.shape[1] - 1, -1, -1)]
    order = np.lexsort(sort_cols)
    points = points[torch.as_tensor(order, dtype=torch.long)]

    if num_points is None or num_points <= 0 or points.shape[0] <= num_points:
        return points

    index = torch.linspace(0, points.shape[0] - 1, steps=num_points).round().long()
    return points[index]

def _make_regular_surface_grid(axis_ranges, fixed_axis, fixed_value, target_num_points):
    varying_axes = [axis for axis in ["x", "y", "z"] if axis != fixed_axis]
    lengths = []
    for axis in varying_axes:
        axis_min, axis_max = axis_ranges[axis]
        lengths.append(max(float(axis_max - axis_min), 1e-12))

    if target_num_points is None or target_num_points <= 1:
        counts = [1, 1]
    else:
        aspect = lengths[0] / lengths[1]
        n0 = max(2, int(round(math.sqrt(target_num_points * aspect))))
        n1 = max(2, int(round(target_num_points / n0)))
        best_counts = [n0, n1]
        best_error = abs(n0 * n1 - target_num_points)

        for dn0 in range(-2, 3):
            for dn1 in range(-2, 3):
                c0 = max(2, n0 + dn0)
                c1 = max(2, n1 + dn1)
                error = abs(c0 * c1 - target_num_points)
                if error < best_error:
                    best_counts = [c0, c1]
                    best_error = error
        counts = best_counts

    coord_values = {}
    for axis in ["x", "y", "z"]:
        if axis == fixed_axis:
            coord_values[axis] = torch.full((1,), float(fixed_value), dtype=torch.float32)
        else:
            idx = varying_axes.index(axis)
            axis_min, axis_max = axis_ranges[axis]
            coord_values[axis] = torch.linspace(
                float(axis_min), float(axis_max), steps=counts[idx], dtype=torch.float32
            )

    xx, yy, zz = torch.meshgrid(
        coord_values["x"], coord_values["y"], coord_values["z"], indexing="ij"
    )
    return torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)

def _min_dist(X, edge, chunk=8192):
    """min Euclidean distance from each row of X to the edge/surface-node set (chunked)."""
    outs = []
    for i in range(0, X.shape[0], chunk):
        outs.append(torch.cdist(X[i:i + chunk], edge).min(dim=1).values)
    return torch.cat(outs, dim=0)


def _apply_shape(dn, afunc):
    """Map normalized min-distance dn in [0,1] (0 ON the BC) to weight A in [0,1].
    {'type':'linear'} A=dn | {'type':'quadratic'} A=dn**2 | {'type':'exp','k':K} A=1-exp(-K*dn)."""
    t = afunc.get('type', 'linear')
    if t == 'exp':
        A = 1.0 - torch.exp(-float(afunc.get('k', 10.0)) * dn)
    elif t == 'quadratic':
        A = dn * dn
    else:
        A = dn
    return A.detach()


def _distance_weight(X, edge, afunc):
    """Analytical hard-BC weight A(x) in [0,1]: 0 ON the BC nodes (min-distance), ->1 away,
    shaped by afunc. Dimension-agnostic (cdist works on [N,3]). Once per mesh, cached."""
    d = _min_dist(X.detach(), edge)
    maxd = max(float(d.max()), 1e-12)
    dn = (d / maxd).clamp(max=1.0)
    return _apply_shape(dn, afunc)


def density_weight_for_mesh(X_elem, X_node, load_points, MP):
    """Analytical DENSITY weight A_rho at element centroids (for training) and at
    nodes (for plotting): A_rho = shape( max(min_dist(x, load) - RHO_SLAB_H, 0)/maxd ),
    0 within RHO_SLAB_H of the loading region (rho pinned to 1) -> 1 away. Uses
    MP['AFUNC_RHO'] if set (a SHARPER exp K than the disp A_u/A_v), else MP['AFUNC'].
    Shared maxd (from the element distances). Returns (A_rho_elem, A_rho_node)."""
    afunc = MP.get('AFUNC_RHO', MP.get('AFUNC', {'type': 'linear'}))
    H = MP.get('RHO_SLAB_H', 0.0)
    de = (_min_dist(X_elem.detach(), load_points) - H).clamp(min=0.0)
    maxd = max(float(de.max()), 1e-12)
    A_elem = _apply_shape((de / maxd).clamp(max=1.0), afunc)
    dn = (_min_dist(X_node.detach(), load_points) - H).clamp(min=0.0)
    A_node = _apply_shape((dn / maxd).clamp(max=1.0), afunc)
    return A_elem, A_node


def analytic_weights_for_mesh(X_node, MP):
    """Rebuild the analytical hard-BC weights for a mesh's NODE coords, using the SAME
    support/load node selection as get_data_L_beam_3D. CPU-safe (no CUDA / get_data),
    for the plotting code. Returns (point_sets, A_u, A_v, A_w) where
    point_sets['phase0_X_train'] = the loading nodes (fed to density_weight_for_mesh
    for A_rho). A_u = A_v = A_w (distance to the support surface, shaped by MP['AFUNC'])."""
    dom = MP['domain']
    xmin = dom['x'][0]
    zmin, zmax = dom['z'][0], dom['z'][1]
    Xu = torch.unique(X_node.detach().cpu(), dim=0)
    y_support = Xu[:, 1].max()
    x_phase = Xu[:, 0].max()

    support_mask = (
        (Xu[:, 0] >= xmin)
        & (Xu[:, 0] <= 40.0)
        & (Xu[:, 1] == y_support)
        & (Xu[:, 2] >= zmin)
        & (Xu[:, 2] <= zmax)
    )
    support = Xu[support_mask].to(device=X_node.device, dtype=X_node.dtype)

    phase_mask = (
        (Xu[:, 0] == x_phase)
        & (Xu[:, 1] >= 35.0)
        & (Xu[:, 1] <= 40.0)
        & (Xu[:, 2] >= zmin)
        & (Xu[:, 2] <= zmax)
    )
    load = Xu[phase_mask].to(device=X_node.device, dtype=X_node.dtype)

    afunc = MP.get('AFUNC', {'type': 'linear'})
    A = _distance_weight(X_node.detach(), support, afunc)   # A_u = A_v = A_w
    return {'phase0_X_train': load}, A, A, A


def _process_mesh_data_multi_gpu(mesh_data, tkwargs):
    for i, device in enumerate(tkwargs["devices"]):
        gpu_key = f"GPU{i}"
        for mesh_key, mesh_dict in mesh_data[gpu_key].items():
            for k, v in mesh_dict.items():
                if k in ["X_node", "X_elem"]:
                    mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(True)
                elif k in ["f_index", "f_adj_index", "conn", "K_in_index", "K_out_index"]:
                    mesh_dict[k] = v.to(device, dtype=torch.long)
                elif k in ["f_magnitude", "B", "detJ", "elem_vol"]:
                    mesh_dict[k] = v.to(device, dtype=tkwargs["dtype"]).requires_grad_(False)
    return mesh_data

def get_data_L_beam_3D(MP):
    Example = MP["Example"]
    Diff_type = MP["Diff_type"]
    N_worker = MP["N_worker"]
    base_folder = MP["base_folder"]
    file_loc = base_folder + "/Data/" + f"{Example}_GPU{N_worker}_{Diff_type}.pt"
    mesh_data = torch.load(file_loc, map_location="cpu")

    tkwargs = get_multiGPU(N_worker)
    num_phase = MP["num_phase"]
    num_pts_bc = MP["num_bc"]
    num_pts_load = MP["num_load"]

    first_gpu_key = "GPU0"
    first_mesh_key = sorted(mesh_data[first_gpu_key].keys())[0]
    mesh0 = mesh_data[first_gpu_key][first_mesh_key]

    X_node = mesh0["X_node"].detach().cpu()
    X_col = torch.unique(X_node, dim=0)

    xmin = 0.0
    xmax = 100.0
    ymin = 0.0
    ymax = 100.0
    zmin = 0.0
    zmax = 10.0

    tol_x = _coordinate_tolerance(X_col[:, 0])
    support_axis_ranges = {
        "x": (xmin, 40.0),
        "y": (ymax, ymax),
        "z": (zmin, zmax),
    }
    load_axis_ranges = {
        "x": (xmax, xmax),
        "y": (35.0, 40.0),
        "z": (zmin, zmax),
    }

    support_points = _make_regular_surface_grid(
        support_axis_ranges, fixed_axis="y", fixed_value=ymax, target_num_points=num_pts_bc
    )
    load_points = _make_regular_surface_grid(
        load_axis_ranges, fixed_axis="x", fixed_value=xmax, target_num_points=num_pts_load
    )
    # visualize_points = MP.get("visualize_training_points", True)
    # if visualize_points:
    #     step = max(1, X_col.shape[0] // 5000)
    #     fig = plt.figure(figsize=(8, 6))
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.scatter(
    #         X_col[::step, 0].numpy(),
    #         X_col[::step, 1].numpy(),
    #         X_col[::step, 2].numpy(),
    #         alpha=0.12,
    #         s=4,
    #         color="tab:blue",
    #         label="All mesh nodes",
    #     )
    #     ax.scatter(
    #         support_points[:, 0].numpy(),
    #         support_points[:, 1].numpy(),
    #         support_points[:, 2].numpy(),
    #         alpha=0.95,
    #         s=14,
    #         color="tab:cyan",
    #         label="Support points",
    #     )
    #     ax.scatter(
    #         load_points[:, 0].numpy(),
    #         load_points[:, 1].numpy(),
    #         load_points[:, 2].numpy(),
    #         alpha=0.95,
    #         s=18,
    #         color="black",
    #         label="Load points",
    #     )
    #     ax.set_xlabel("X (mm)")
    #     ax.set_ylabel("Y (mm)")
    #     ax.set_zlabel("Z (mm)")
    #     ax.set_title("L-beam 3D training points")
    #     ax.set_box_aspect((xmax - xmin, ymax - MP["domain"]["y"][0], zmax - zmin))
    #     ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    #     plt.tight_layout()
    #     plt.show()

    u_X_train = support_points.type(tkwargs["dtype"]).requires_grad_(False)
    u_train = torch.zeros(u_X_train.shape[0], dtype=tkwargs["dtype"])
    v_X_train = u_X_train.clone()
    v_train = torch.zeros(v_X_train.shape[0], dtype=tkwargs["dtype"])
    w_X_train = u_X_train.clone()
    w_train = torch.zeros(w_X_train.shape[0], dtype=tkwargs["dtype"])

    t1_X_train = u_X_train.clone()
    t1_train = torch.zeros(t1_X_train.shape[0], dtype=tkwargs["dtype"])
    t2_X_train = v_X_train.clone()
    t2_train = torch.zeros(t2_X_train.shape[0], dtype=tkwargs["dtype"])
    t3_X_train = w_X_train.clone()
    t3_train = torch.zeros(t3_X_train.shape[0], dtype=tkwargs["dtype"])

    Training = {
        "X_col": X_col.type(tkwargs["dtype"]),
        "support_points": support_points.type(tkwargs["dtype"]),
        "load_points": load_points.type(tkwargs["dtype"]),
        "u_X_train": u_X_train,
        "u_train": u_train,
        "v_X_train": v_X_train,
        "v_train": v_train,
        "w_X_train": w_X_train,
        "w_train": w_train,
        "t1_X_train": t1_X_train,
        "t1_train": t1_train,
        "t2_X_train": t2_X_train,
        "t2_train": t2_train,
        "t3_X_train": t3_X_train,
        "t3_train": t3_train,
    }

    phase_X_train = load_points.type(tkwargs["dtype"]).requires_grad_(False)
    for i in range(num_phase):
        Training[f"phase{i}_X_train"] = phase_X_train
        if i == 0:
            Training[f"phase{i}_train"] = torch.ones(phase_X_train.shape[0], dtype=tkwargs["dtype"])
        else:
            Training[f"phase{i}_train"] = torch.zeros(phase_X_train.shape[0], dtype=tkwargs["dtype"])

    mesh_data = _process_mesh_data_multi_gpu(mesh_data, tkwargs)
    train_data = {}
    mesh_counter = 0
    for gpu_key in sorted(mesh_data.keys()):
        for mesh_key in sorted(mesh_data[gpu_key].keys()):
            mesh = mesh_data[gpu_key][mesh_key]
            device = mesh["X_node"].device
            X_node_mesh = torch.unique(mesh["X_node"].detach().cpu(), dim=0)
            X_elem_mesh = torch.unique(mesh["X_elem"].detach().cpu(), dim=0)
            y_support = X_node_mesh[:, 1].max()
            x_phase = X_node_mesh[:, 0].max()

            support_mask = (
                (X_node_mesh[:, 0] >= xmin)
                & (X_node_mesh[:, 0] <= 40.0)
                & (X_node_mesh[:, 1] == y_support)
                & (X_node_mesh[:, 2] >= zmin)
                & (X_node_mesh[:, 2] <= zmax)
            )
            support_points_mesh = X_node_mesh[support_mask].to(
                device=device, dtype=tkwargs["dtype"]
            ).requires_grad_(False)
            if support_points_mesh.shape[0] == 0:
                raise ValueError(f"No support training nodes found for {gpu_key}/{mesh_key}.")

            # ANALYTICAL HARD-BC WEIGHTS (3D): A_u=A_v=A_w = distance from every mesh node to
            # the support surface (u=v=w=0 there), 0 on the BC -> 1 away. field = A(x)*NN(x).
            _afunc = MP.get('AFUNC', {'type': 'linear'})
            _Xq = mesh["X_node"].detach()
            mesh["A_u"] = _distance_weight(_Xq, support_points_mesh, _afunc)
            mesh["A_v"] = _distance_weight(_Xq, support_points_mesh, _afunc)
            mesh["A_w"] = _distance_weight(_Xq, support_points_mesh, _afunc)

            phase_mask = (
                (X_node_mesh[:, 0] == x_phase)
                & (X_node_mesh[:, 1] >= 35.0)
                & (X_node_mesh[:, 1] <= 40.0)
                & (X_node_mesh[:, 2] >= zmin)
                & (X_node_mesh[:, 2] <= zmax)
            )
            phase_points_mesh = X_node_mesh[phase_mask].to(
                device=device, dtype=tkwargs["dtype"]
            ).requires_grad_(False)
            if phase_points_mesh.shape[0] == 0:
                raise ValueError(f"No phase training elements found for {gpu_key}/{mesh_key}.")

            # ANALYTICAL DENSITY WEIGHT A_rho (built ONCE here, like A_u/A_v/A_w):
            # 0 on the loading region (rho pinned to 1) -> 1 away, sharper AFUNC_RHO.
            # Element values -> sliced per slice for training; node values -> plotting.
            mesh["A_rho"], mesh["A_rho_node"] = density_weight_for_mesh(
                mesh["X_elem"].detach(), _Xq, phase_points_mesh, MP)

            train_inputs = [
                support_points_mesh,
                support_points_mesh.clone(),
                support_points_mesh.clone(),
                support_points_mesh.clone(),
                support_points_mesh.clone(),
                support_points_mesh.clone(),
            ]
            train_targets = [
                torch.zeros(support_points_mesh.shape[0], device=device, dtype=tkwargs["dtype"])
                for _ in range(6)
            ]

            for phase_idx in range(num_phase):
                train_inputs.append(phase_points_mesh.clone())
                if phase_idx == 0:
                    phase_target = torch.ones(phase_points_mesh.shape[0], device=device, dtype=tkwargs["dtype"])
                else:
                    phase_target = torch.zeros(phase_points_mesh.shape[0], device=device, dtype=tkwargs["dtype"])
                train_targets.append(phase_target)

            train_data[mesh_counter] = SimpleNamespace(
                gpu_key=gpu_key,
                mesh_key=mesh_key,
                train_inputs=train_inputs,
                train_targets=train_targets,
            )
            mesh_counter += 1
    return Training, mesh_data, train_data

def get_data_L_beam_3D_w(MP):
    Example = MP["Example"]
    Diff_type = MP["Diff_type"]
    N_worker = MP["N_worker"]
    base_folder = MP["base_folder"]
    file_loc = base_folder + "/Data/" + f"{Example}_GPU{N_worker}_{Diff_type}.pt"
    mesh_data = torch.load(file_loc, map_location="cpu")

    tkwargs = get_multiGPU(N_worker)
    num_phase = MP["num_phase"]
    num_pts_bc = 10
    num_pts_load = MP["num_load"]

    first_gpu_key = "GPU0"
    first_mesh_key = sorted(mesh_data[first_gpu_key].keys())[0]
    mesh0 = mesh_data[first_gpu_key][first_mesh_key]

    X_node = mesh0["X_node"].detach().cpu()
    X_col = torch.unique(X_node, dim=0)

    xmin = 0.0
    xmax = 100.0
    ymin = 0.0
    ymax = 100.0
    zmin = 0.0
    zmax = 10.0

    tol_x = _coordinate_tolerance(X_col[:, 0])
    support_axis_ranges = {
        "x": (xmin, 40.0),
        "y": (ymax, ymax),
        "z": (zmin, zmax),
    }
    load_axis_ranges = {
        "x": (xmax, xmax),
        "y": (35.0, 40.0),
        "z": (zmin, zmax),
    }

    support_points = _make_regular_surface_grid(
        support_axis_ranges, fixed_axis="y", fixed_value=ymax, target_num_points=num_pts_bc
    )
    load_points = _make_regular_surface_grid(
        load_axis_ranges, fixed_axis="x", fixed_value=xmax, target_num_points=num_pts_load
    )
    w_support_y = torch.linspace(35.0, 40.0, steps=num_pts_bc, dtype=torch.float32)
    w_support_points = torch.stack(
        [
            torch.full_like(w_support_y, float(xmax)),
            w_support_y,
            torch.full_like(w_support_y, 5.0),
        ],
        dim=1,
    )
    support_points_w = torch.unique(torch.cat([support_points, w_support_points], dim=0), dim=0)

    # visualize_points = MP.get("visualize_training_points", True)
    # if visualize_points:
    #     step = max(1, X_col.shape[0] // 5000)
    #     fig = plt.figure(figsize=(8, 6))
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.scatter(
    #         X_col[::step, 0].numpy(),
    #         X_col[::step, 1].numpy(),
    #         X_col[::step, 2].numpy(),
    #         alpha=0.12,
    #         s=4,
    #         color="tab:blue",
    #         label="All mesh nodes",
    #     )
    #     ax.scatter(
    #         support_points[:, 0].numpy(),
    #         support_points[:, 1].numpy(),
    #         support_points[:, 2].numpy(),
    #         alpha=0.95,
    #         s=14,
    #         color="tab:cyan",
    #         label="Support points",
    #     )
    #     ax.scatter(
    #         load_points[:, 0].numpy(),
    #         load_points[:, 1].numpy(),
    #         load_points[:, 2].numpy(),
    #         alpha=0.95,
    #         s=18,
    #         color="black",
    #         label="Load points",
    #     )
    #     ax.set_xlabel("X (mm)")
    #     ax.set_ylabel("Y (mm)")
    #     ax.set_zlabel("Z (mm)")
    #     ax.set_title("L-beam 3D training points")
    #     ax.set_box_aspect((xmax - xmin, ymax - MP["domain"]["y"][0], zmax - zmin))
    #     ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    #     plt.tight_layout()
    #     plt.show()

    u_X_train = support_points.type(tkwargs["dtype"]).requires_grad_(False)
    u_train = torch.zeros(u_X_train.shape[0], dtype=tkwargs["dtype"])
    v_X_train = u_X_train.clone()
    v_train = torch.zeros(v_X_train.shape[0], dtype=tkwargs["dtype"])
    w_X_train = support_points_w.type(tkwargs["dtype"]).requires_grad_(False)
    w_train = torch.zeros(w_X_train.shape[0], dtype=tkwargs["dtype"])

    t1_X_train = u_X_train.clone()
    t1_train = torch.zeros(t1_X_train.shape[0], dtype=tkwargs["dtype"])
    t2_X_train = v_X_train.clone()
    t2_train = torch.zeros(t2_X_train.shape[0], dtype=tkwargs["dtype"])
    t3_X_train = w_X_train.clone()
    t3_train = torch.zeros(t3_X_train.shape[0], dtype=tkwargs["dtype"])

    Training = {
        "X_col": X_col.type(tkwargs["dtype"]),
        "support_points": support_points.type(tkwargs["dtype"]),
        "load_points": load_points.type(tkwargs["dtype"]),
        "u_X_train": u_X_train,
        "u_train": u_train,
        "v_X_train": v_X_train,
        "v_train": v_train,
        "w_X_train": w_X_train,
        "w_train": w_train,
        "t1_X_train": t1_X_train,
        "t1_train": t1_train,
        "t2_X_train": t2_X_train,
        "t2_train": t2_train,
        "t3_X_train": t3_X_train,
        "t3_train": t3_train,
    }

    phase_X_train = load_points.type(tkwargs["dtype"]).requires_grad_(False)
    for i in range(num_phase):
        Training[f"phase{i}_X_train"] = phase_X_train
        if i == 0:
            Training[f"phase{i}_train"] = torch.ones(phase_X_train.shape[0], dtype=tkwargs["dtype"])
        else:
            Training[f"phase{i}_train"] = torch.zeros(phase_X_train.shape[0], dtype=tkwargs["dtype"])

    mesh_data = _process_mesh_data_multi_gpu(mesh_data, tkwargs)
    return Training, mesh_data

def get_data_L_beam_3D_wide(MP):
    Example = MP["Example"]
    Diff_type = MP["Diff_type"]
    N_worker = MP["N_worker"]
    base_folder = MP["base_folder"]
    file_loc = base_folder + "/Data/" + f"{Example}_GPU{N_worker}_{Diff_type}.pt"
    mesh_data = torch.load(file_loc, map_location="cpu")

    tkwargs = get_multiGPU(N_worker)
    num_phase = MP["num_phase"]
    num_pts_bc = MP["num_bc"]
    num_pts_load = MP["num_load"]

    first_gpu_key = "GPU0"
    first_mesh_key = sorted(mesh_data[first_gpu_key].keys())[0]
    mesh0 = mesh_data[first_gpu_key][first_mesh_key]

    X_node = mesh0["X_node"].detach().cpu()
    X_col = torch.unique(X_node, dim=0)

    xmin = 0.0
    xmax = 100.0
    ymin = 0.0
    ymax = 100.0
    zmin = 0.0
    zmax = 25.0

    tol_x = _coordinate_tolerance(X_col[:, 0])
    support_axis_ranges = {
        "x": (xmin, 40.0),
        "y": (ymax, ymax),
        "z": (zmin, zmax),
    }
    load_axis_ranges = {
        "x": (xmax, xmax),
        "y": (35.0, 40.0),
        "z": (10.0, 15.0),
    }

    support_points = _make_regular_surface_grid(
        support_axis_ranges, fixed_axis="y", fixed_value=ymax, target_num_points=num_pts_bc
    )
    load_points = _make_regular_surface_grid(
        load_axis_ranges, fixed_axis="x", fixed_value=xmax, target_num_points=num_pts_load
    )
    # visualize_points = MP.get("visualize_training_points", True)
    # if visualize_points:
    #     step = max(1, X_col.shape[0] // 5000)
    #     fig = plt.figure(figsize=(8, 6))
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.scatter(
    #         X_col[::step, 0].numpy(),
    #         X_col[::step, 1].numpy(),
    #         X_col[::step, 2].numpy(),
    #         alpha=0.12,
    #         s=4,
    #         color="tab:blue",
    #         label="All mesh nodes",
    #     )
    #     ax.scatter(
    #         support_points[:, 0].numpy(),
    #         support_points[:, 1].numpy(),
    #         support_points[:, 2].numpy(),
    #         alpha=0.95,
    #         s=14,
    #         color="tab:cyan",
    #         label="Support points",
    #     )
    #     ax.scatter(
    #         load_points[:, 0].numpy(),
    #         load_points[:, 1].numpy(),
    #         load_points[:, 2].numpy(),
    #         alpha=0.95,
    #         s=18,
    #         color="black",
    #         label="Load points",
    #     )
    #     ax.set_xlabel("X (mm)")
    #     ax.set_ylabel("Y (mm)")
    #     ax.set_zlabel("Z (mm)")
    #     ax.set_title("L-beam 3D training points")
    #     ax.set_box_aspect((xmax - xmin, ymax - MP["domain"]["y"][0], zmax - zmin))
    #     ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    #     plt.tight_layout()
    #     plt.show()

    u_X_train = support_points.type(tkwargs["dtype"]).requires_grad_(False)
    u_train = torch.zeros(u_X_train.shape[0], dtype=tkwargs["dtype"])
    v_X_train = u_X_train.clone()
    v_train = torch.zeros(v_X_train.shape[0], dtype=tkwargs["dtype"])
    w_X_train = u_X_train.clone()
    w_train = torch.zeros(w_X_train.shape[0], dtype=tkwargs["dtype"])

    t1_X_train = u_X_train.clone()
    t1_train = torch.zeros(t1_X_train.shape[0], dtype=tkwargs["dtype"])
    t2_X_train = v_X_train.clone()
    t2_train = torch.zeros(t2_X_train.shape[0], dtype=tkwargs["dtype"])
    t3_X_train = w_X_train.clone()
    t3_train = torch.zeros(t3_X_train.shape[0], dtype=tkwargs["dtype"])

    Training = {
        "X_col": X_col.type(tkwargs["dtype"]),
        "support_points": support_points.type(tkwargs["dtype"]),
        "load_points": load_points.type(tkwargs["dtype"]),
        "u_X_train": u_X_train,
        "u_train": u_train,
        "v_X_train": v_X_train,
        "v_train": v_train,
        "w_X_train": w_X_train,
        "w_train": w_train,
        "t1_X_train": t1_X_train,
        "t1_train": t1_train,
        "t2_X_train": t2_X_train,
        "t2_train": t2_train,
        "t3_X_train": t3_X_train,
        "t3_train": t3_train,
    }

    phase_X_train = load_points.type(tkwargs["dtype"]).requires_grad_(False)
    for i in range(num_phase):
        Training[f"phase{i}_X_train"] = phase_X_train
        if i == 0:
            Training[f"phase{i}_train"] = torch.ones(phase_X_train.shape[0], dtype=tkwargs["dtype"])
        else:
            Training[f"phase{i}_train"] = torch.zeros(phase_X_train.shape[0], dtype=tkwargs["dtype"])

    mesh_data = _process_mesh_data_multi_gpu(mesh_data, tkwargs)
    return Training, mesh_data

def get_data_Portal_frame_half_3D(MP):
    Example = MP["Example"]
    Diff_type = MP["Diff_type"]
    N_worker = MP["N_worker"]
    base_folder = MP["base_folder"]
    file_loc = base_folder + "/Data/" + f"{Example}_GPU{N_worker}_{Diff_type}.pt"
    mesh_data = torch.load(file_loc, map_location="cpu")

    tkwargs = get_multiGPU(N_worker)
    num_phase = MP["num_phase"]
    num_pts_bc = MP["num_bc"]
    num_pts_load = MP["num_load"]

    first_gpu_key = "GPU0"
    first_mesh_key = sorted(mesh_data[first_gpu_key].keys())[0]
    mesh0 = mesh_data[first_gpu_key][first_mesh_key]

    X_node = mesh0["X_node"].detach().cpu()
    X_col = torch.unique(X_node, dim=0)

    xmin = 0.0
    xmax = 60.0
    ymin = 0.0
    ymax = 60.0
    zmin = 0.0
    zmax = 10.0

    u_support_axis_ranges = {
        "x": (xmin, xmin),
        "y": (35.0, ymax),
        "z": (zmin, zmax),
    }
    v_support_axis_ranges = {
        "x": (54.5, xmax),
        "y": (ymin, ymin),
        "z": (zmin, zmax),
    }
    load_axis_ranges = {
        "x": (xmin, 5.0),
        "y": (ymax, ymax),
        "z": (zmin, zmax),
    }

    u_support_points = _make_regular_surface_grid(
        u_support_axis_ranges, fixed_axis="x", fixed_value=xmin, target_num_points=num_pts_bc
    )
    v_support_points = _make_regular_surface_grid(
        v_support_axis_ranges,
        fixed_axis="y",
        fixed_value=ymin,
        target_num_points=max(1, num_pts_bc // 5),
    )
    load_points = _make_regular_surface_grid(
        load_axis_ranges, fixed_axis="y", fixed_value=ymax, target_num_points=num_pts_load
    )
    support_points = torch.unique(torch.cat([u_support_points, v_support_points], dim=0), dim=0)

    # visualize_points = MP.get("visualize_training_points", True)
    # if visualize_points:
    #     step = max(1, X_col.shape[0] // 5000)
    #     fig = plt.figure(figsize=(8, 6))
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.scatter(
    #         X_col[::step, 0].numpy(),
    #         X_col[::step, 1].numpy(),
    #         X_col[::step, 2].numpy(),
    #         alpha=0.12,
    #         s=4,
    #         color="tab:blue",
    #         label="All mesh nodes",
    #     )
    #     ax.scatter(
    #         u_support_points[:, 0].numpy(),
    #         u_support_points[:, 1].numpy(),
    #         u_support_points[:, 2].numpy(),
    #         alpha=0.95,
    #         s=14,
    #         color="tab:cyan",
    #         label="U Support points",
    #     )
    #     ax.scatter(
    #         v_support_points[:, 0].numpy(),
    #         v_support_points[:, 1].numpy(),
    #         v_support_points[:, 2].numpy(),
    #         alpha=0.95,
    #         s=14,
    #         color="tab:red",
    #         label="V,W Support points",
    #     )
    #     ax.scatter(
    #         load_points[:, 0].numpy(),
    #         load_points[:, 1].numpy(),
    #         load_points[:, 2].numpy(),
    #         alpha=0.95,
    #         s=18,
    #         color="black",
    #         label="Load points",
    #     )
    #     ax.set_xlabel("X (mm)")
    #     ax.set_ylabel("Y (mm)")
    #     ax.set_zlabel("Z (mm)")
    #     ax.set_title("L-beam 3D training points")
    #     ax.set_box_aspect((xmax - xmin, ymax - MP["domain"]["y"][0], zmax - zmin))
    #     ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    #     plt.tight_layout()
    #     plt.show()

    u_X_train = u_support_points.type(tkwargs["dtype"]).requires_grad_(False)
    u_train = torch.zeros(u_X_train.shape[0], dtype=tkwargs["dtype"])
    v_X_train = v_support_points.type(tkwargs["dtype"]).requires_grad_(False)
    v_train = torch.zeros(v_X_train.shape[0], dtype=tkwargs["dtype"])
    # Keep one out-of-plane anchor so w does not remain a completely free rigid-body mode.
    w_X_train = v_support_points.type(tkwargs["dtype"]).requires_grad_(False)
    w_train = torch.zeros(w_X_train.shape[0], dtype=tkwargs["dtype"])

    t1_X_train = u_X_train.clone()
    t1_train = torch.zeros(t1_X_train.shape[0], dtype=tkwargs["dtype"])
    t2_X_train = v_X_train.clone()
    t2_train = torch.zeros(t2_X_train.shape[0], dtype=tkwargs["dtype"])
    t3_X_train = w_X_train.clone()
    t3_train = torch.zeros(t3_X_train.shape[0], dtype=tkwargs["dtype"])

    Training = {
        "X_col": X_col.type(tkwargs["dtype"]),
        "support_points": support_points.type(tkwargs["dtype"]),
        "load_points": load_points.type(tkwargs["dtype"]),
        "u_X_train": u_X_train,
        "u_train": u_train,
        "v_X_train": v_X_train,
        "v_train": v_train,
        "w_X_train": w_X_train,
        "w_train": w_train,
        "t1_X_train": t1_X_train,
        "t1_train": t1_train,
        "t2_X_train": t2_X_train,
        "t2_train": t2_train,
        "t3_X_train": t3_X_train,
        "t3_train": t3_train,
    }

    phase_X_train = load_points.type(tkwargs["dtype"]).requires_grad_(False)
    for i in range(num_phase):
        Training[f"phase{i}_X_train"] = phase_X_train
        if i == 0:
            Training[f"phase{i}_train"] = torch.ones(phase_X_train.shape[0], dtype=tkwargs["dtype"])
        else:
            Training[f"phase{i}_train"] = torch.zeros(phase_X_train.shape[0], dtype=tkwargs["dtype"])

    mesh_data = _process_mesh_data_multi_gpu(mesh_data, tkwargs)

    # ---- ANALYTICAL HARD-BC weights + train_data (mirror get_data_L_beam_3D) ----
    # Portal_frame_half has COMPONENT-WISE support: u is fixed on the x=xmin face
    # (y in [35, ymax]); v and w are fixed on the y=ymin face (x in [54.5, xmax]).
    # So A_u = distance to the u-support surface, while A_v = A_w = distance to the
    # v/w-support surface (they DIFFER here, unlike L_beam's full clamp). A_rho is
    # pinned to 1 on the loading region (y=ymax face, x in [xmin, 5]).
    train_data = {}
    mesh_counter = 0
    _afunc = MP.get('AFUNC', {'type': 'linear'})
    for gpu_key in sorted(mesh_data.keys()):
        for mesh_key in sorted(mesh_data[gpu_key].keys()):
            mesh = mesh_data[gpu_key][mesh_key]
            device = mesh["X_node"].device
            X_node_mesh = torch.unique(mesh["X_node"].detach().cpu(), dim=0)
            x_face = X_node_mesh[:, 0].min()   # x=xmin face (u support)
            y_lo   = X_node_mesh[:, 1].min()   # y=ymin face (v/w support)
            y_hi   = X_node_mesh[:, 1].max()   # y=ymax face (load)

            # u-support: x = xmin face, y in [35, ymax], full thickness
            u_support_mask = (
                (X_node_mesh[:, 0] == x_face)
                & (X_node_mesh[:, 1] >= 35.0)
                & (X_node_mesh[:, 2] >= zmin) & (X_node_mesh[:, 2] <= zmax)
            )
            # v/w-support: y = ymin face, x in [54.5, xmax], full thickness
            v_support_mask = (
                (X_node_mesh[:, 1] == y_lo)
                & (X_node_mesh[:, 0] >= 54.5)
                & (X_node_mesh[:, 2] >= zmin) & (X_node_mesh[:, 2] <= zmax)
            )
            u_support_mesh = X_node_mesh[u_support_mask].to(
                device=device, dtype=tkwargs["dtype"]).requires_grad_(False)
            v_support_mesh = X_node_mesh[v_support_mask].to(
                device=device, dtype=tkwargs["dtype"]).requires_grad_(False)
            if u_support_mesh.shape[0] == 0 or v_support_mesh.shape[0] == 0:
                raise ValueError(f"No support nodes (u={u_support_mesh.shape[0]}, "
                                 f"v={v_support_mesh.shape[0]}) for {gpu_key}/{mesh_key}.")

            # ANALYTICAL HARD-BC WEIGHTS (component-wise): field = A(x)*NN(x), A=0 on the
            # component's support surface (that DOF pinned to 0) -> 1 away.
            _Xq = mesh["X_node"].detach()
            mesh["A_u"] = _distance_weight(_Xq, u_support_mesh, _afunc)   # u on u-support
            mesh["A_v"] = _distance_weight(_Xq, v_support_mesh, _afunc)   # v on v-support
            mesh["A_w"] = _distance_weight(_Xq, v_support_mesh, _afunc)   # w anchored on v-support

            # load region: y = ymax face, x in [xmin, 5]
            phase_mask = (
                (X_node_mesh[:, 1] == y_hi)
                & (X_node_mesh[:, 0] >= xmin) & (X_node_mesh[:, 0] <= 5.0)
                & (X_node_mesh[:, 2] >= zmin) & (X_node_mesh[:, 2] <= zmax)
            )
            phase_points_mesh = X_node_mesh[phase_mask].to(
                device=device, dtype=tkwargs["dtype"]).requires_grad_(False)
            if phase_points_mesh.shape[0] == 0:
                raise ValueError(f"No load/phase nodes found for {gpu_key}/{mesh_key}.")

            # ANALYTICAL DENSITY WEIGHT A_rho: 0 on the loading region (rho pinned to 1)
            # -> 1 away (sharper AFUNC_RHO). Element values for training, node for plotting.
            mesh["A_rho"], mesh["A_rho_node"] = density_weight_for_mesh(
                mesh["X_elem"].detach(), _Xq, phase_points_mesh, MP)

            # train_data per model order [u, v, w, t1, t2, t3] then phases. u/t1 sit on the
            # u-support, v/w/t2/t3 on the v-support (component-wise). Targets 0 on support
            # (homogeneous Dirichlet); phase0 -> 1 (solid seed), other phases -> 0.
            supp = [u_support_mesh, v_support_mesh, v_support_mesh,
                    u_support_mesh, v_support_mesh, v_support_mesh]
            train_inputs = [s.clone() for s in supp]
            train_targets = [torch.zeros(s.shape[0], device=device, dtype=tkwargs["dtype"])
                             for s in supp]
            for phase_idx in range(num_phase):
                train_inputs.append(phase_points_mesh.clone())
                _mk = torch.ones if phase_idx == 0 else torch.zeros
                train_targets.append(_mk(phase_points_mesh.shape[0], device=device,
                                         dtype=tkwargs["dtype"]))

            train_data[mesh_counter] = SimpleNamespace(
                gpu_key=gpu_key, mesh_key=mesh_key,
                train_inputs=train_inputs, train_targets=train_targets)
            mesh_counter += 1
    return Training, mesh_data, train_data



