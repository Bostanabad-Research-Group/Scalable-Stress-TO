import os
import torch
import matplotlib.pyplot as plt
import torch

def read_inp_mesh(filepath, device="cpu"):
    """
    Robust Abaqus .inp mesh reader.

    Supports:
      *Node block: reads x,y,(z) coordinates into X_node [N_nodes, ndim]
      *Element block: reads connectivity into conn [N_elems, n_nodes_per_elem]
    
    Ignores all other keywords (*Nset, *Elset, *EndPart, etc.).

    Args:        filepath (str): Path to Abaqus .inp file
        device (str): Torch device ("cpu" or "cuda")

    Returns:
        X_node: torch.Tensor [N_nodes, ndim]
        conn:   torch.LongTensor [N_elems, n_nodes_per_elem]
    """
    nodes = []
    elems = []
    reading_nodes = False
    reading_elems = False

    with open(filepath, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("**"):  # skip comments/blank
                continue

            # ====== Section switches ======
            if line.lower().startswith("*node"):
                reading_nodes, reading_elems = True, False
                continue
            elif line.lower().startswith("*element,"):
                reading_nodes, reading_elems = False, True
                continue
            elif line.startswith("*"):  # any other keyword (*nset, *elset, etc.)
                reading_nodes, reading_elems = False, False
                continue

            # ====== Read node coordinates ======
            if reading_nodes:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                # Must have: id, x, y, (z)
                if len(parts) < 3:
                    continue
                try:
                    coords = list(map(float, parts[1:]))
                    nodes.append(coords)
                except ValueError:
                    continue  # skip malformed numeric line

            # ====== Read element connectivity ======
            elif reading_elems:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                # Must have: id, n1, n2, n3, ...
                if len(parts) < 2:
                    continue
                # Ensure first entry is integer (element id)
                if not parts[0].isdigit():
                    continue
                try:
                    conn = [int(n) - 1 for n in parts[1:]]
                    elems.append(conn)
                except ValueError:
                    continue  # skip invalid lines

    # ====== Convert to tensors ======
    if not nodes:
        raise ValueError("No nodes found in file.")
    if not elems:
        raise ValueError("No elements found in file.")

    # Ensure consistent dimensions (2D or 3D)
    ndim = len(nodes[0])
    nodes = [n[:ndim] for n in nodes if len(n) == ndim]  # filter out malformed

    X_node = torch.tensor(nodes, dtype=torch.float32, device=device)
    X_node = X_node[:,0:2]
    conn = torch.tensor(elems, dtype=torch.long, device=device)

    return X_node, conn

def nodal_average_centroid(X_node: torch.Tensor, conn: torch.Tensor) -> torch.Tensor:
    """
    Compute element centroids using nodal average (geometric centroid).
    
    Args:
        X_node (torch.Tensor): [N_nodes, ndim] node coordinates (2D or 3D).
        conn   (torch.LongTensor): [N_elems, n_nodes_per_elem] connectivity, 
                                   with 0-based node indices.
    
    Returns:
        X_elem (torch.Tensor): [N_elems, ndim] centroid coordinates of each element.
    """
    # Gather nodal coordinates for each element
    elem_nodes = X_node[conn]              # [N_elems, n_nodes_per_elem, ndim]
    # Average over element nodes
    X_elem = elem_nodes.mean(dim=1)        # [N_elems, ndim]
    return X_elem

# === Example usage ===
if __name__ == "__main__":
    example = 'Portal_frame_half'   # L_beam Hook Portal_frame Crack Portal_frame_half
    sampling_size = 0.05148
    mode = '' # empty for training, pred for prediction with a lot of elements
    num_inp = 1 # 4 for  pred, 20 for training
    if mode == 'pred':
        folder = f"{example}_pred"
    else:
        folder = f"{example}_sampling_size{sampling_size}"
    all_meshes = {}
    count_lines = []
    for i in range(1, num_inp + 1):  # read first 2 meshes
        # fname = os.path.join(folder, f"Mesh_{i:02d}_pred.inp")
        fname = os.path.join(folder, f"Mesh_{i:02d}.inp")
        X_node, conn = read_inp_mesh(fname)
        X_elem = nodal_average_centroid(X_node, conn)
        # # make 2D scatter plot for X_node (blue) and X_elem (red)
        # plt.figure(figsize=(6, 6))
        # plt.scatter(X_node[:, 0].cpu(), X_node[:, 1].cpu(),
        #             c="blue", s=3, label="Nodes")
        # plt.scatter(X_elem[:, 0].cpu(), X_elem[:, 1].cpu(),
        #             c="red", s=3, label="Element Centroids")
        # plt.gca().set_aspect("equal", adjustable="box")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.title(f"Mesh {i:02d}")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        
        # print(f"Mesh {i}:")
        # print("  Node array shape:", X_node.shape)   # [N_nodes, ndim]
        # print("  Connectivity shape:", conn.shape)   # [N_elems, nodes_per_elem]
        # print("  Element array shape:", X_elem.shape)   # [N_elems, nodes_per_elem]
        # print("  First 5 nodes:\n", X_node[:5])
        # print("  First 5 elements:\n", conn[:5])

        # store in dict
        all_meshes[f"Mesh_{i:02d}"] = {
            "X_node": X_node,
            "conn": conn,
            "X_elem": X_elem
        }

        print(f"Loaded Mesh_{i:02d}: "
              f"nodes={X_node.shape}, elems={conn.shape}, centroids={X_elem.shape}")

        count_lines.append(f"Mesh_{i:02d}: nodes={X_node.shape[0]}, elements={conn.shape[0]}")

    # write final node/element counts into the same folder as the .inp files
    counts_path = os.path.join(folder, "mesh_node_element_counts.txt")
    with open(counts_path, "w") as f:
        f.write("\n".join(count_lines) + "\n")
    print(f"[saved] {counts_path}")

    # save the dictionary into a file
    # save_path = f"{folder}_mesh_pred.pt"
    if mode == 'pred':
        save_path = f"{example}_mesh_pred.pt"
    else:
        save_path = f"{example}_mesh.pt"
    torch.save(all_meshes, save_path)
        














