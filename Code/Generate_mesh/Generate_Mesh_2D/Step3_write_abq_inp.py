import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import gmsh
import matplotlib.tri as mtri
import glob

colors_density = ['#FFFFFF', '#addc30', '#5ec962', '#28ae80', '#21918c', '#2c72e8', '#3b526b', '#472d7b', '#440154']  # White to the purple end
cmap_density = LinearSegmentedColormap.from_list('custom_cmap', colors_density, N=128)

colors_phase = ['#FFFFFF', '#33a645', '#3b4cc0', '#b40426'] 
Density = [0, 0.4, 0.6, 1.0]
D_norm = (np.array(Density) - min(Density)) / (max(Density) - min(Density))  # [0.0, 0.4, 0.6, 1.0] -> [0.0, 0.4, 0.6, 1.0] (already normalized in this case)
color_tuples = list(zip(D_norm, colors_phase))
cmap_phases = LinearSegmentedColormap.from_list('custom_density_cmap', color_tuples, N=256)

colors_contours = ["#3b4fbf", "#79a2f2", "#c2d7f2", "#f2c9bb", "#f2856d", "#bf0637"]
cmap_contours = LinearSegmentedColormap.from_list("custom_cmap", colors_contours, N=128)

cmap_curves = {'blue_dark': "#3b4cc0",'blue_light': "#8db0fe",
                 'red_dark':"#b40426",'red_light':"#f4987a",
                 'green_dark': "#33a645",'green_light': "#afd991",
                 'gray_dark': "#555555",'gray_light': "#bababa",
                 'purple_dark': "#6659a7",'purple_light': "#c2bdde",
                 'orange_dark': "#ef7e21",'orange_light': "#f6c092",
                 'yellow_dark': "#f6d31b",'yellow_light': "#fae98f",
                 'cyan_dark': "#7bddf3",'cyan_light': "#bbedf2",
                 'magenta_dark': "#fe3da7",'magenta_light': "#fe9fd3",
                 }

def hex_to_rgba(hex_color, alpha=1.0):
    """    Convert hex color to RGBA format and adjust alpha (transparency).
    Default alpha is 1.0 (fully opaque).
    
    Parameters:
    - hex_color: str, color in hex format (e.g., '#3b4cc0')
    - alpha: float, transparency level (0.0 to 1.0)
    
    Returns:
    - rgba: tuple, (r, g, b, a)
    """
    rgba = mcolors.to_rgba(hex_color)
    return (rgba[0], rgba[1], rgba[2], alpha)

def merge_triangle_pairs_to_quads(triangles, node_tag_to_xy):
    triangles = np.asarray(triangles, dtype=int)
    if triangles.size == 0:
        return np.zeros((0, 4), dtype=int)

    edge_to_triangles = {}
    for tri_id, tri in enumerate(triangles):
        edges = [
            tuple(sorted((int(tri[0]), int(tri[1])))),
            tuple(sorted((int(tri[1]), int(tri[2])))),
            tuple(sorted((int(tri[2]), int(tri[0])))),
        ]
        for edge in edges:
            edge_to_triangles.setdefault(edge, []).append(tri_id)

    used = np.zeros(len(triangles), dtype=bool)
    quads = []

    for tri_ids in edge_to_triangles.values():
        if len(tri_ids) != 2:
            continue

        t0, t1 = tri_ids
        if used[t0] or used[t1]:
            continue

        tri0 = triangles[t0]
        tri1 = triangles[t1]
        unique_nodes = list(dict.fromkeys(np.concatenate([tri0, tri1]).tolist()))
        if len(unique_nodes) != 4:
            continue

        pts = np.asarray([node_tag_to_xy[int(nid)] for nid in unique_nodes], dtype=float)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        order = np.argsort(angles)
        quads.append(np.asarray(unique_nodes, dtype=int)[order])
        used[t0] = True
        used[t1] = True

    if np.any(~used):
        raise RuntimeError(
            f"Unable to convert {np.count_nonzero(~used)} residual triangle(s) into quads."
        )

    return np.asarray(quads, dtype=int)

def write_abaqus_inp(
    out_inp,
    part_name,
    node_tags,
    node_coords,
    elem_conn,
    outer_curve_nodes=None,
    hole_curve_nodes=None,
    element_type="CPE4",
):
    """
    Write a 2D orphan-mesh Abaqus input file.
    node_tags: shape (N,)
    node_coords: shape (N,3)
    elem_conn: shape (Ne, nnode) with Abaqus node ids
    outer_curve_nodes: dict like {"OUTER_0": np.array([...]), ...}
    hole_curve_nodes: dict like {"HOLE_0": np.array([...]), ...}
    """
    with open(out_inp, "w") as f:
        f.write("*Heading\n")
        f.write("** Generated from contours by Gmsh -> Abaqus orphan mesh pipeline\n")
        f.write("*Part, name={}\n".format(part_name))

        f.write("*Node\n")
        for nid, xyz in zip(node_tags, node_coords):
            f.write("{}, {}, {}, {}\n".format(int(nid), xyz[0], xyz[1], xyz[2]))

        f.write("*Element, type={}\n".format(element_type))
        for eid, conn in enumerate(elem_conn, start=1):
            conn_str = ", ".join(str(int(v)) for v in conn)
            f.write(f"{eid}, {conn_str}\n")

        # all elements
        f.write("*Elset, elset=EALL, generate\n")
        f.write("1, {}, 1\n".format(len(elem_conn)))

        # all nodes
        node_tags_sorted = np.asarray(node_tags, dtype=int)
        f.write("*Nset, nset=NALL\n")
        write_id_block(f, node_tags_sorted)

        if outer_curve_nodes:
            for name, ids in outer_curve_nodes.items():
                ids = np.unique(np.asarray(ids, dtype=int))
                if len(ids) == 0:
                    continue
                f.write("*Nset, nset={}\n".format(name))
                write_id_block(f, ids)

        if hole_curve_nodes:
            for name, ids in hole_curve_nodes.items():
                ids = np.unique(np.asarray(ids, dtype=int))
                if len(ids) == 0:
                    continue
                f.write("*Nset, nset={}\n".format(name))
                write_id_block(f, ids)

        f.write("*End Part\n")

def write_id_block(f, ids, per_line=16):
    ids = list(map(int, ids))
    for i in range(0, len(ids), per_line):
        chunk = ids[i:i + per_line]
        f.write(", ".join(str(v) for v in chunk) + "\n")

def msh_to_inp(
    msh_path,
    out_inp,
    part_name="PART-1",
    element_type="CPE4",
):
    """
    Read an existing Gmsh .msh file and write an Abaqus orphan-mesh .inp file.

    Returns
    -------
    nodes_xy : (N_node, 2) np.ndarray
    elements_plot : (N_elem, 3) or (N_elem, 4) np.ndarray
    outers : list
    holes  : list
        Empty lists here, kept only for compatibility with plot_mesh_with_contours.
    """
    if not os.path.exists(msh_path):
        raise FileNotFoundError(f"Missing mesh file: {msh_path}")

    gmsh.initialize()
    gmsh.model.add("read_msh")

    try:
        gmsh.open(msh_path)

        # all nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=int)
        node_coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)

        if len(node_tags) == 0:
            raise RuntimeError(f"No nodes found in mesh: {msh_path}")

        # get 2D elements
        elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(2)

        elem_conn = None
        elements_plot = []
        triangles_raw = []

        # map gmsh node tag -> row index
        tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

        for et, e_tags, e_nodes in zip(elem_types, elem_tags_list, elem_node_tags_list):
            name, dim, order, num_nodes, local_coords, num_primary_nodes = \
                gmsh.model.mesh.getElementProperties(et)

            if num_nodes == 3:
                arr = np.asarray(e_nodes, dtype=int).reshape(-1, 3)
                arr_plot = np.vectorize(tag_to_idx.get)(arr)
                triangles_raw.append(arr)
            elif num_nodes == 4:
                arr = np.asarray(e_nodes, dtype=int).reshape(-1, 4)
                arr_plot = np.vectorize(tag_to_idx.get)(arr)
            elif num_nodes == 6:
                arr6 = np.asarray(e_nodes, dtype=int).reshape(-1, 6)
                arr = arr6[:, :3]
                arr_plot = np.vectorize(tag_to_idx.get)(arr)
                triangles_raw.append(arr)
            elif num_nodes == 8:
                arr8 = np.asarray(e_nodes, dtype=int).reshape(-1, 8)
                arr = arr8[:, :4]
                arr_plot = np.vectorize(tag_to_idx.get)(arr)
            else:
                continue

            if num_nodes in (4, 8):
                if elem_conn is None:
                    elem_conn = arr.copy()
                elif elem_conn.shape[1] != arr.shape[1]:
                    raise RuntimeError(
                        f"Mixed 2D element topologies found in mesh: {msh_path}"
                    )
                else:
                    elem_conn = np.vstack([elem_conn, arr])

                elements_plot.append(arr_plot)

        if len(triangles_raw) > 0:
            node_tag_to_xy = {
                int(tag): node_coords[idx, :2] for idx, tag in enumerate(node_tags)
            }
            tri_conn = np.vstack(triangles_raw)
            merged_quads = merge_triangle_pairs_to_quads(tri_conn, node_tag_to_xy)
            merged_quads_plot = np.vectorize(tag_to_idx.get)(merged_quads)

            if elem_conn is None:
                elem_conn = merged_quads
            else:
                elem_conn = np.vstack([elem_conn, merged_quads])

            elements_plot.append(merged_quads_plot)

        if elem_conn is None:
            raise RuntimeError(f"No supported 2D elements found in mesh: {msh_path}")

        if len(elements_plot) > 0:
            elements_plot = np.vstack(elements_plot)
        else:
            elements_plot = np.zeros((0, elem_conn.shape[1]), dtype=int)

        # collect node sets from physical curve groups
        outer_curve_nodes = {}
        hole_curve_nodes = {}

        phys_groups = gmsh.model.getPhysicalGroups()

        for dim, tag in phys_groups:
            if dim != 1:
                continue

            name = gmsh.model.getPhysicalName(dim, tag)
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)

            ids = []
            for ent in entities:
                node_ids, _, _ = gmsh.model.mesh.getNodes(dim, ent, includeBoundary=True)
                if len(node_ids) > 0:
                    ids.append(np.asarray(node_ids, dtype=int))

            if len(ids) > 0:
                ids = np.unique(np.concatenate(ids))
            else:
                ids = np.array([], dtype=int)

            if name.startswith("OUTER_"):
                outer_curve_nodes[name] = ids
            elif name.startswith("HOLE_"):
                hole_curve_nodes[name] = ids

        # write Abaqus input
        write_abaqus_inp(
            out_inp=out_inp,
            part_name=part_name,
            node_tags=node_tags,
            node_coords=node_coords,
            elem_conn=elem_conn,
            outer_curve_nodes=outer_curve_nodes,
            hole_curve_nodes=hole_curve_nodes,
            element_type=element_type,
        )

        nodes_xy = node_coords[:, :2]
        outers = []
        holes = []

        return nodes_xy, elements_plot, outers, holes

    finally:
        gmsh.finalize()

sampling_size = 0.05148
example = 'Portal_frame_half'   # L_beam Hook Portal_frame Crack Portal_frame_half
mesh_size = sampling_size
folder_name = f"{example}_sampling_size{sampling_size}"

# find all msh files
msh_files = sorted(glob.glob(os.path.join(folder_name, "*.msh")))

if len(msh_files) == 0:
    raise FileNotFoundError(f"No .msh files found in {folder_name}")

for i, msh_path in enumerate(msh_files, start=1):
    out_inp = os.path.join(folder_name, f"Mesh_{i:02d}.inp")

    print(f"[read] {msh_path}")

    nodes, elements, outers, holes = msh_to_inp(
        msh_path=msh_path,
        out_inp=out_inp,
        part_name="PART-1",
        element_type="CPS4R",
    )

    print(f"[done] {out_inp}")

print("OK")
