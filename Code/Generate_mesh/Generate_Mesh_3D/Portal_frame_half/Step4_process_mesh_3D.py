import glob
import os

import torch


def read_inp_mesh_3d(filepath, device="cpu"):
    """
    Read a 3D Abaqus .inp orphan mesh.

    Handles linear tetrahedra (C3D4, 4 nodes/element) and linear hexahedra
    (C3D8, 8 nodes/element); the connectivity width follows the element type.

    Returns
    -------
    X_node  : torch.Tensor [N_nodes, 3]
    conn    : torch.LongTensor [N_elems, 4 or 8], zero-based row indices into X_node
    node_ids: torch.LongTensor [N_nodes], original Abaqus node labels
    elem_ids: torch.LongTensor [N_elems], original Abaqus element labels
    """
    nodes = []
    node_ids = []
    elems = []
    elem_ids = []
    reading_nodes = False
    reading_elems = False

    with open(filepath, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("**"):
                continue

            line_lower = line.lower()
            if line_lower.startswith("*node"):
                reading_nodes, reading_elems = True, False
                continue
            if line_lower.startswith("*element,"):
                reading_nodes, reading_elems = False, True
                continue
            if line.startswith("*"):
                reading_nodes, reading_elems = False, False
                continue

            if reading_nodes:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) < 4:
                    continue
                try:
                    node_ids.append(int(parts[0]))
                    nodes.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    continue

            elif reading_elems:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) < 5:
                    continue
                try:
                    elem_ids.append(int(parts[0]))
                    elems.append([int(n) for n in parts[1:]])
                except ValueError:
                    continue

    if not nodes:
        raise ValueError(f"No nodes found in file: {filepath}")
    if not elems:
        raise ValueError(f"No elements found in file: {filepath}")

    node_id_to_row = {nid: i for i, nid in enumerate(node_ids)}
    conn_rows = []
    for elem in elems:
        try:
            conn_rows.append([node_id_to_row[nid] for nid in elem])
        except KeyError as exc:
            raise ValueError(
                f"Element references node {exc.args[0]}, which is missing from *Node."
            ) from exc

    X_node = torch.tensor(nodes, dtype=torch.float32, device=device)
    conn = torch.tensor(conn_rows, dtype=torch.long, device=device)
    node_ids = torch.tensor(node_ids, dtype=torch.long, device=device)
    elem_ids = torch.tensor(elem_ids, dtype=torch.long, device=device)

    return X_node, conn, node_ids, elem_ids


def nodal_average_centroid(X_node, conn):
    elem_nodes = X_node[conn]
    return elem_nodes.mean(dim=1)


if __name__ == "__main__":
    example = "Portal_frame_half"
    sampling_size = 0.28
    mode = ""

    if mode == "pred":
        folder = f"{example}_pred"
        inp_files = sorted(glob.glob(os.path.join(folder, "Mesh_*_3D_pred.inp")))
    else:
        folder = f"{example}_sampling_size{sampling_size}"
        inp_files = sorted(glob.glob(os.path.join(folder, "Mesh_*_3D.inp")))

    if len(inp_files) == 0:
        raise FileNotFoundError(
            f"No 3D Abaqus input files found in {folder}. "
            "Run Step3_write_abq_inp_3D.py first."
        )

    all_meshes = {}
    for i, fname in enumerate(inp_files, start=1):
        X_node, conn, node_ids, elem_ids = read_inp_mesh_3d(fname)
        X_elem = nodal_average_centroid(X_node, conn)
        mesh_name = f"Mesh_{i:02d}"

        all_meshes[mesh_name] = {
            "X_node": X_node,
            "conn": conn,
            "X_elem": X_elem,
            "node_ids": node_ids,
            "elem_ids": elem_ids,
            "source_inp": fname,
        }

        print(
            f"Loaded {mesh_name}: nodes={X_node.shape}, "
            f"elems={conn.shape}, centroids={X_elem.shape}"
        )

    if mode == "pred":
        save_path = f"{example}_3D_mesh_pred.pt"
    else:
        save_path = f"{example}_3D_mesh.pt"

    torch.save(all_meshes, save_path)
    print(f"Saved mesh data to {save_path}")
