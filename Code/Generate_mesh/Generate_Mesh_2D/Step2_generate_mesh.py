import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import gmsh
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
    """
    Convert hex color to RGBA format and adjust alpha (transparency).
    Default alpha is 1.0 (fully opaque).
    
    Parameters:
    - hex_color: str, color in hex format (e.g., '#3b4cc0')
    - alpha: float, transparency level (0.0 to 1.0)
    
    Returns:
    - rgba: tuple, (r, g, b, a)
    """
    rgba = mcolors.to_rgba(hex_color)
    return (rgba[0], rgba[1], rgba[2], alpha)

def plot_contours(contours_outer, contours_holes, ax=None):
    """
    Plot extracted contours.
    """
    if ax is None:
        fig, ax = plt.subplots()

    for c in contours_outer:
        ax.plot(c[:, 0], c[:, 1], "-", linewidth=2.0, label="outer")
    for c in contours_holes:
        ax.plot(c[:, 0], c[:, 1], "--", linewidth=2.0, label="hole")

    ax.set_aspect("equal")
    return ax

def load_contours_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    contours_outer = []
    contours_holes = []

    if "outer" in data:
        for key in sorted(data["outer"].keys()):
            contours_outer.append(np.asarray(data["outer"][key], dtype=float))

    if "holes" in data:
        for key in sorted(data["holes"].keys()):
            contours_holes.append(np.asarray(data["holes"][key], dtype=float))

    return contours_outer, contours_holes

def load_mesh_controls_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("mesh_controls", {})

def plot_contours_from_json(json_path, ax=None, outer_style=None, hole_style=None):
    if ax is None:
        fig, ax = plt.subplots()

    if outer_style is None:
        outer_style = {"color": "r", "linestyle": "-", "linewidth": 2.0}
    if hole_style is None:
        hole_style = {"color": "b", "linestyle": "--", "linewidth": 2.0}

    contours_outer, contours_holes = load_contours_from_json(json_path)

    for c in contours_outer:
        ax.plot(c[:, 0], c[:, 1], **outer_style)

    for c in contours_holes:
        ax.plot(c[:, 0], c[:, 1], **hole_style)

    ax.set_aspect("equal")
    return ax

def remove_duplicate_last_point(loop, tol=1e-12):
    if len(loop) >= 2 and np.linalg.norm(loop[0] - loop[-1]) < tol:
        return loop[:-1]
    return loop

def remove_consecutive_duplicate_points(loop, tol=1e-12):
    if len(loop) == 0:
        return loop
    cleaned = [loop[0]]
    for p in loop[1:]:
        if np.linalg.norm(p - cleaned[-1]) >= tol:
            cleaned.append(p)
    return np.asarray(cleaned, dtype=float)

def polygon_signed_area(loop):
    x = loop[:, 0]
    y = loop[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

def ensure_loop_orientation(loop, ccw=True):
    area = polygon_signed_area(loop)
    if ccw and area < 0:
        return loop[::-1].copy()
    if (not ccw) and area > 0:
        return loop[::-1].copy()
    return loop

def preprocess_loop(loop, ccw=True):
    loop = np.asarray(loop, dtype=float)
    loop = remove_duplicate_last_point(loop)
    loop = remove_consecutive_duplicate_points(loop)

    if len(loop) < 3:
        raise ValueError("A contour loop has fewer than 3 unique points.")

    loop = ensure_loop_orientation(loop, ccw=ccw)
    return loop

def get_loop_edge_controls(mesh_controls, loop_name):
    edge_segments = mesh_controls.get("edge_segments", {})
    return edge_segments.get(loop_name, [])

def add_refined_edge_size_fields(
    edge_controls,
    line_tags,
    interior_size,
    default_transition_distance,
):
    """
    Create graded size fields that refine selected boundary-edge groups and
    smoothly transition back to the interior size.
    """
    threshold_fields = []
    line_start = 0

    for edge_control in edge_controls:
        num_curve_segments = int(edge_control.get("num_curve_segments", 0))
        edge_line_tags = line_tags[line_start:line_start + num_curve_segments]
        line_start += num_curve_segments

        edge_size = float(edge_control.get("sampling_size", interior_size))
        transition_distance = edge_control.get("transition_distance", None)

        if edge_size >= interior_size or len(edge_line_tags) == 0:
            continue

        if transition_distance is None:
            transition_distance = default_transition_distance

        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", edge_line_tags)
        gmsh.model.mesh.field.setNumber(
            distance_field, "Sampling", max(100, 20 * len(edge_line_tags))
        )

        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", edge_size)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", interior_size)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", transition_distance)
        threshold_fields.append(threshold_field)

    if not threshold_fields:
        return None

    if len(threshold_fields) == 1:
        return threshold_fields[0]

    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", threshold_fields)
    return min_field

def add_polyline_loop_geo(loop, lc):
    point_tags = []
    for p in loop:
        tag = gmsh.model.geo.addPoint(float(p[0]), float(p[1]), 0.0, lc)
        point_tags.append(tag)

    line_tags = []
    n = len(point_tags)
    for i in range(n):
        a = point_tags[i]
        b = point_tags[(i + 1) % n]
        ltag = gmsh.model.geo.addLine(a, b)
        line_tags.append(ltag)

    cloop = gmsh.model.geo.addCurveLoop(line_tags)
    return point_tags, line_tags, cloop

def get_nodes_on_curve(curve_tag):
    """
    Return unique node tags currently lying on a given curve.
    Works after meshing.
    """
    node_tags, _, _ = gmsh.model.mesh.getNodes(1, curve_tag, includeBoundary=True)
    if len(node_tags) == 0:
        return np.array([], dtype=int)
    return np.unique(np.asarray(node_tags, dtype=int))

def write_abaqus_inp(
    out_inp,
    part_name,
    node_tags,
    node_coords,
    tri_conn,
    outer_curve_nodes=None,
    hole_curve_nodes=None,
    element_type="CPE3",
):
    """
    Write a 2D orphan-mesh Abaqus input file.
    node_tags: shape (N,)
    node_coords: shape (N,3)
    tri_conn: shape (Ne,3) with Abaqus node ids
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
        for eid, conn in enumerate(tri_conn, start=1):
            f.write("{}, {}, {}, {}\n".format(eid, int(conn[0]), int(conn[1]), int(conn[2])))

        # all elements
        f.write("*Elset, elset=EALL, generate\n")
        f.write("1, {}, 1\n".format(len(tri_conn)))

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

def ensure_even_loop_point_count(loop, edge_controls):
    """
    Gmsh's Blossom recombination algorithm requires an even total triangle
    count in the surface; for a simply-connected polygon that count's parity
    is fixed by the parity of the boundary point count alone. An odd boundary
    point count makes Blossom silently fall back to a much weaker
    recombination pass, leaving thousands of un-mergeable residual triangles.
    Split one boundary edge with a midpoint to force an even count.
    """
    n = len(loop)
    if n % 2 == 0:
        return loop, edge_controls

    edge_controls = [dict(ec) for ec in edge_controls]

    if len(edge_controls) == 0:
        split_idx = n // 2
    else:
        group_idx = max(
            range(len(edge_controls)),
            key=lambda k: int(edge_controls[k].get("num_curve_segments", 0)),
        )
        line_start = sum(
            int(edge_controls[k].get("num_curve_segments", 0))
            for k in range(group_idx)
        )
        num_segments = int(edge_controls[group_idx].get("num_curve_segments", 0))
        split_idx = line_start + num_segments // 2
        edge_controls[group_idx]["num_curve_segments"] = num_segments + 1

    p0 = loop[split_idx % n]
    p1 = loop[(split_idx + 1) % n]
    midpoint = 0.5 * (p0 + p1)
    new_loop = np.insert(loop, split_idx + 1, midpoint, axis=0)
    return new_loop, edge_controls

def merge_triangle_pairs_to_quads(triangles, node_coords_xy):
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

        pts = np.asarray([node_coords_xy[nid] for nid in unique_nodes], dtype=float)
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

def mesh_contours_json_to_msh(
    json_path,
    out_msh,
    mesh_size=0.02,
    algorithm=8,
    recombine=True,
    msh_version=4.1,
    randomize=False,
    mesh_size_delta=0.0,
    random_factor=0.0,
    seed=None,
    interactive_view=True,
):
    outer_loops, hole_loops = load_contours_from_json(json_path)
    mesh_controls = load_mesh_controls_from_json(json_path)
    if len(outer_loops) == 0:
        raise ValueError(f"No outer contour found in {json_path}")

    if seed is not None:
        np.random.seed(seed)

    base_interior_size = float(mesh_controls.get("interior_size", mesh_size))
    if randomize and mesh_size_delta > 0.0:
        interior_size = np.random.uniform(
            base_interior_size - mesh_size_delta,
            base_interior_size + mesh_size_delta,
        )
    else:
        interior_size = base_interior_size
    default_transition_distance = float(
        mesh_controls.get("default_transition_distance", 4.0 * interior_size)
    )

    edge_segments = dict(mesh_controls.get("edge_segments", {}))

    refined_sizes = []
    for loop_edges in edge_segments.values():
        for edge_control in loop_edges:
            edge_size = edge_control.get("sampling_size", interior_size)
            if edge_size is not None:
                refined_sizes.append(float(edge_size))

    mesh_size_min = min([interior_size] + refined_sizes)
    mesh_size_max = max([interior_size] + refined_sizes)

    gmsh.initialize()
    gmsh.model.add("contour_mesh")

    try:
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        gmsh.option.setNumber("Mesh.MshFileVersion", msh_version)
        if recombine:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)

        if randomize:
            gmsh.option.setNumber("Mesh.RandomFactor", random_factor)

        # Keep contour points exactly as stored in contour.json
        outer_loops = [preprocess_loop(loop, ccw=True) for loop in outer_loops]
        hole_loops = [preprocess_loop(loop, ccw=False) for loop in hole_loops]

        # Force an even boundary point count per loop so Gmsh's Blossom
        # recombination doesn't fall back and leave residual triangles.
        fixed_outer_loops = []
        for i, loop in enumerate(outer_loops):
            loop_name = f"outer_{i}"
            loop, ec = ensure_even_loop_point_count(
                loop, edge_segments.get(loop_name, [])
            )
            edge_segments[loop_name] = ec
            fixed_outer_loops.append(loop)
        outer_loops = fixed_outer_loops

        fixed_hole_loops = []
        for i, loop in enumerate(hole_loops):
            loop_name = f"hole_{i}"
            loop, ec = ensure_even_loop_point_count(
                loop, edge_segments.get(loop_name, [])
            )
            edge_segments[loop_name] = ec
            fixed_hole_loops.append(loop)
        hole_loops = fixed_hole_loops

        mesh_controls = dict(mesh_controls)
        mesh_controls["edge_segments"] = edge_segments

        outer_curve_tags = []
        hole_curve_tags = []
        plane_surfaces = []
        outer_cloops = []
        hole_cloops = []

        for loop in outer_loops:
            _, line_tags, cloop = add_polyline_loop_geo(loop, interior_size)
            outer_curve_tags.append(line_tags)
            outer_cloops.append(cloop)

        for loop in hole_loops:
            _, line_tags, cloop = add_polyline_loop_geo(loop, interior_size)
            hole_curve_tags.append(line_tags)
            hole_cloops.append(cloop)

        gmsh.model.geo.synchronize()

        if len(outer_cloops) == 1:
            surf = gmsh.model.geo.addPlaneSurface([outer_cloops[0]] + hole_cloops)
            plane_surfaces.append(surf)
        else:
            for cl in outer_cloops:
                surf = gmsh.model.geo.addPlaneSurface([cl])
                plane_surfaces.append(surf)

        gmsh.model.geo.synchronize()

        size_fields = []
        for i, line_tags in enumerate(outer_curve_tags):
            loop_name = f"outer_{i}"
            edge_controls = get_loop_edge_controls(mesh_controls, loop_name)
            field_tag = add_refined_edge_size_fields(
                edge_controls=edge_controls,
                line_tags=line_tags,
                interior_size=interior_size,
                default_transition_distance=default_transition_distance,
            )
            if field_tag is not None:
                size_fields.append(field_tag)

        for i, line_tags in enumerate(hole_curve_tags):
            loop_name = f"hole_{i}"
            edge_controls = get_loop_edge_controls(mesh_controls, loop_name)
            field_tag = add_refined_edge_size_fields(
                edge_controls=edge_controls,
                line_tags=line_tags,
                interior_size=interior_size,
                default_transition_distance=default_transition_distance,
            )
            if field_tag is not None:
                size_fields.append(field_tag)

        if len(size_fields) == 1:
            gmsh.model.mesh.field.setAsBackgroundMesh(size_fields[0])
        elif len(size_fields) > 1:
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", size_fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        if recombine:
            for s in plane_surfaces:
                gmsh.model.mesh.setRecombine(2, s)

        # Force each boundary segment to have only its two endpoints as mesh nodes
        for lines in outer_curve_tags:
            for ltag in lines:
                gmsh.model.mesh.setTransfiniteCurve(ltag, 2)

        for lines in hole_curve_tags:
            for ltag in lines:
                gmsh.model.mesh.setTransfiniteCurve(ltag, 2)

        pg_surfs = gmsh.model.addPhysicalGroup(2, plane_surfaces)
        gmsh.model.setPhysicalName(2, pg_surfs, "DOMAIN")

        for i, lines in enumerate(outer_curve_tags):
            pg = gmsh.model.addPhysicalGroup(1, lines)
            gmsh.model.setPhysicalName(1, pg, f"OUTER_{i}")

        for i, lines in enumerate(hole_curve_tags):
            pg = gmsh.model.addPhysicalGroup(1, lines)
            gmsh.model.setPhysicalName(1, pg, f"HOLE_{i}")

        gmsh.model.mesh.generate(2)
        gmsh.write(out_msh)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=int)
        node_coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)
        nodes_xy = node_coords[:, :2]

        tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

        elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(2)

        elements_plot = []
        triangles_plot = []
        for et, e_tags, e_nodes in zip(elem_types, elem_tags_list, elem_node_tags_list):
            name, dim, order, num_nodes, local_coords, num_primary_nodes = \
                gmsh.model.mesh.getElementProperties(et)

            if num_nodes == 4:
                arr = np.asarray(e_nodes, dtype=int).reshape(-1, 4)
                arr = np.vectorize(tag_to_idx.get)(arr)
                elements_plot.append(arr)

            elif num_nodes == 8:
                arr = np.asarray(e_nodes, dtype=int).reshape(-1, 8)[:, :4]
                arr = np.vectorize(tag_to_idx.get)(arr)
                elements_plot.append(arr)

            elif num_nodes in (3, 6):
                arr = np.asarray(e_nodes, dtype=int).reshape(-1, num_nodes)[:, :3]
                arr = np.vectorize(tag_to_idx.get)(arr)
                triangles_plot.append(arr)

        if len(triangles_plot) > 0:
            merged_quads = merge_triangle_pairs_to_quads(
                np.vstack(triangles_plot), nodes_xy
            )
            if len(merged_quads) > 0:
                elements_plot.append(merged_quads)

        if len(elements_plot) > 0:
            elements_plot = np.vstack(elements_plot)
        else:
            elements_plot = np.zeros((0, 4), dtype=int)

        if interactive_view:
            out_png = (
                os.path.splitext(out_msh)[0]
                + f"_nodes{len(node_tags)}_elems{len(elements_plot)}.png"
            )
            gmsh.fltk.initialize()
            gmsh.graphics.draw()
            gmsh.write(out_png)
            gmsh.fltk.run()

        return nodes_xy, elements_plot, outer_loops, hole_loops, interior_size

    finally:
        gmsh.finalize()

def batch_build_msh_from_contours(
    foldername,
    filename,
    mesh_size=0.02,
    interactive_view=True,
    num_mesh=20,
    mesh_size_delta=0.0,
    random_factor=0.0,
):
    """
    Generate multiple random Gmsh meshes from one contour JSON file.
    Save meshes from 001 to num_mesh.
    Output names include the contour file stem to avoid overwriting.
    """
    msh_files = []

    json_path = os.path.join(foldername, filename)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing contour file: {json_path}")

    print(f"[mesh] {json_path}")

    contour_tag = os.path.splitext(os.path.basename(filename))[0]
    mesh_tag = f"{mesh_size:.3f}".rstrip("0").rstrip(".")

    for i in range(1, num_mesh + 1):
        seed = i

        msh_name = f"{contour_tag}_mesh_{mesh_tag}_{i:03d}.msh"
        out_msh = os.path.join(foldername, msh_name)

        nodes, elements, outers, holes, h_used = mesh_contours_json_to_msh(
            json_path=json_path,
            out_msh=out_msh,
            mesh_size=mesh_size,
            algorithm=8,
            recombine=True,
            randomize=False,
            mesh_size_delta=mesh_size_delta,
            random_factor=random_factor,
            seed=seed,
            interactive_view=interactive_view,
        )

        msh_files.append(out_msh)

        print(f"[done] {out_msh} | h_used = {h_used:.6f}")

    return msh_files

sampling_size = 0.05148
example = 'Portal_frame_half'   # L_beam Hook Portal_frame Crack Portal_frame_half
plot_mesh = False   # set to False to skip mesh visualization and save time
mesh_size = sampling_size
mesh_size_delta = 0.0   # actual h between about 2.9 and 3.1
random_factor = 0.0
num_mesh_each_contour = 1   # set to 1 if you want one mesh per contour

folder_name = f"{example}_sampling_size{sampling_size}"

# find all contour json files
contour_files = sorted(glob.glob(os.path.join(folder_name, "contour_*.json")))

# count contours
num_contours = len(contour_files)

if num_contours == 0:
    raise FileNotFoundError(f"No contour files found in {folder_name}")

print(f"Found {num_contours} contour files in {folder_name}")

all_msh_files = []

# loop over each contour file and generate mesh(es)
for contour_path in contour_files:
    contour_file = os.path.basename(contour_path)
    print(f"\nProcessing {contour_file} ...")

    msh_files = batch_build_msh_from_contours(
        foldername=folder_name,
        filename=contour_file,
        mesh_size=mesh_size,
        interactive_view=plot_mesh,
        num_mesh=num_mesh_each_contour,
        mesh_size_delta=mesh_size_delta,
        random_factor=random_factor,
    )

    all_msh_files.extend(msh_files)

print(f"\nGenerated {len(all_msh_files)} mesh files:")
for f in all_msh_files:
    print(f)
