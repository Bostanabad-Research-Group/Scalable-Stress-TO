import json
import os
import glob

import numpy as np


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

    return ensure_loop_orientation(loop, ccw=ccw)


def get_loop_edge_controls(mesh_controls, loop_name):
    edge_segments = mesh_controls.get("edge_segments", {})
    return edge_segments.get(loop_name, [])


def add_polyline_loop_geo(gmsh, loop, lc):
    point_tags = []
    for p in loop:
        tag = gmsh.model.geo.addPoint(float(p[0]), float(p[1]), 0.0, lc)
        point_tags.append(tag)

    line_tags = []
    n = len(point_tags)
    for i in range(n):
        a = point_tags[i]
        b = point_tags[(i + 1) % n]
        line_tags.append(gmsh.model.geo.addLine(a, b))

    cloop = gmsh.model.geo.addCurveLoop(line_tags)
    return point_tags, line_tags, cloop


def add_refined_edge_size_fields(
    gmsh,
    edge_controls,
    line_tags,
    interior_size,
    default_transition_distance,
):
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


def mesh_contours_json_to_3d_msh(
    json_path,
    out_msh,
    thickness=10.0,
    mesh_size=1.0,
    element_type="tet",
    num_layers=None,
    algorithm_2d=9,
    algorithm_3d=1,
    msh_version=4.1,
    randomize=False,
    mesh_size_delta=0.0,
    random_factor=0.0,
    seed=None,
    show_plot=False,
    save_png=False,
):
    """
    Build a 3D mesh by extruding a Step 1 2D contour in +z.

    The input contour remains 2D. Gmsh creates the required 3D volume by
    extruding the plane surface through ``thickness``.

    element_type:
        "tet" -> linear tetrahedra (Abaqus C3D4), volume filled by the 3D
                 meshing algorithm.
        "hex" -> linear hexahedra (Abaqus C3D8), produced by recombining the
                 2D mesh into quadrangles and extruding through ``num_layers``
                 structured layers. A pure hex mesh is required: if any
                 non-hexahedral 3D element is produced (i.e. a mixed mesh), an
                 error is raised.
    """
    if element_type not in ("tet", "hex"):
        raise ValueError(
            f"element_type must be 'tet' or 'hex', got {element_type!r}."
        )
    try:
        import gmsh
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Python package 'gmsh' is required. Install it in this Python "
            "environment before running Step2_generate_mesh_3D.py."
        ) from exc

    outer_loops, hole_loops = load_contours_from_json(json_path)
    mesh_controls = load_mesh_controls_from_json(json_path)
    if len(outer_loops) == 0:
        raise ValueError(f"No outer contour found in {json_path}")
    if thickness <= 0:
        raise ValueError("thickness must be positive.")

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

    refined_sizes = []
    edge_segments = mesh_controls.get("edge_segments", {})
    for loop_edges in edge_segments.values():
        for edge_control in loop_edges:
            edge_size = edge_control.get("sampling_size", interior_size)
            if edge_size is not None:
                refined_sizes.append(float(edge_size))

    mesh_size_min = min([interior_size] + refined_sizes)
    mesh_size_max = max([interior_size] + refined_sizes)

    gmsh.initialize()
    gmsh.model.add("contour_mesh_3d")

    try:
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm_2d)
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm_3d)
        gmsh.option.setNumber("Mesh.MshFileVersion", msh_version)

        if element_type == "hex":
            # Guarantee a pure-hex mesh.
            #
            # The structured extrusion (numElements + recombine) needs the 2D
            # base mesh recombined into quads. Simple Blossom recombination (1)
            # is the only recombiner compatible with the structured extrude --
            # full-quad (2/3) tries to split boundary edges and fails with
            # "1D mesh cannot be divided by 2". Blossom alone, however, leaves a
            # few stray triangles that extrude into prisms (a mixed mesh).
            #
            # SubdivisionAlgorithm = 2 ("all hexahedra") is therefore enabled as
            # the final step: it subdivides every 3D element into hexahedra, so
            # any residual prism becomes hexes and the result is guaranteed pure
            # C3D8. Note this also subdivides the regular hexes (each split into
            # 8), so the effective element size is ~half of `mesh_size`; use a
            # correspondingly larger `mesh_size` if you need a target resolution.
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)

        if randomize:
            gmsh.option.setNumber("Mesh.RandomFactor", random_factor)

        outer_loops = [preprocess_loop(loop, ccw=True) for loop in outer_loops]
        hole_loops = [preprocess_loop(loop, ccw=False) for loop in hole_loops]

        outer_curve_tags = []
        hole_curve_tags = []
        outer_cloops = []
        hole_cloops = []

        for loop in outer_loops:
            _, line_tags, cloop = add_polyline_loop_geo(gmsh, loop, interior_size)
            outer_curve_tags.append(line_tags)
            outer_cloops.append(cloop)

        for loop in hole_loops:
            _, line_tags, cloop = add_polyline_loop_geo(gmsh, loop, interior_size)
            hole_curve_tags.append(line_tags)
            hole_cloops.append(cloop)

        gmsh.model.geo.synchronize()

        if len(outer_cloops) != 1:
            raise ValueError("3D extrusion currently expects one outer contour.")

        front_surface = gmsh.model.geo.addPlaneSurface([outer_cloops[0]] + hole_cloops)
        gmsh.model.geo.synchronize()

        size_fields = []
        for i, line_tags in enumerate(outer_curve_tags):
            loop_name = f"outer_{i}"
            edge_controls = get_loop_edge_controls(mesh_controls, loop_name)
            field_tag = add_refined_edge_size_fields(
                gmsh=gmsh,
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
                gmsh=gmsh,
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

        # For tets, pin one element per contour segment so the boundary mesh
        # follows the Step 1 contour points exactly. For hexes this constraint
        # is incompatible with full-quad recombination (it forbids the edge
        # splitting needed to guarantee all-quads), so it is skipped there; the
        # contour points are geometric points and remain mesh nodes regardless.
        if element_type != "hex":
            for lines in outer_curve_tags + hole_curve_tags:
                for ltag in lines:
                    gmsh.model.mesh.setTransfiniteCurve(ltag, 2)

        if element_type == "hex":
            if num_layers is None:
                n_layers = max(1, int(round(abs(thickness) / interior_size)))
            else:
                n_layers = max(1, int(num_layers))
            extruded = gmsh.model.geo.extrude(
                [(2, front_surface)],
                0.0,
                0.0,
                thickness,
                numElements=[n_layers],
                recombine=True,
            )
        else:
            extruded = gmsh.model.geo.extrude([(2, front_surface)], 0.0, 0.0, thickness)
        gmsh.model.geo.synchronize()

        volume_tags = [tag for dim, tag in extruded if dim == 3]
        new_surface_tags = [tag for dim, tag in extruded if dim == 2]
        if len(volume_tags) != 1:
            raise RuntimeError("Gmsh did not create exactly one extruded volume.")

        top_surface_tags = []
        side_surface_tags = []
        z_tol = max(1e-8, abs(thickness) * 1e-8)
        for surface_tag in new_surface_tags:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surface_tag)
            if abs(zmin - thickness) < z_tol and abs(zmax - thickness) < z_tol:
                top_surface_tags.append(surface_tag)
            else:
                side_surface_tags.append(surface_tag)

        pg_volume = gmsh.model.addPhysicalGroup(3, volume_tags)
        gmsh.model.setPhysicalName(3, pg_volume, "DOMAIN")

        pg_front = gmsh.model.addPhysicalGroup(2, [front_surface])
        gmsh.model.setPhysicalName(2, pg_front, "FRONT_Z0")

        if top_surface_tags:
            pg_top = gmsh.model.addPhysicalGroup(2, top_surface_tags)
            gmsh.model.setPhysicalName(2, pg_top, "BACK_ZT")

        if side_surface_tags:
            pg_side = gmsh.model.addPhysicalGroup(2, side_surface_tags)
            gmsh.model.setPhysicalName(2, pg_side, "SIDE")

        gmsh.model.mesh.generate(3)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        elem_types, elem_tags_list, _ = gmsh.model.mesh.getElements(3)
        num_tets = 0
        num_hexes = 0
        num_other_3d = 0
        for et, e_tags in zip(elem_types, elem_tags_list):
            name, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(et)
            if dim != 3:
                continue
            if num_nodes == 4:
                num_tets += len(e_tags)
            elif num_nodes == 8:
                num_hexes += len(e_tags)
            else:
                # prisms (6 nodes) / pyramids (5 nodes) -> mixed mesh
                num_other_3d += len(e_tags)

        num_elements = num_tets + num_hexes + num_other_3d

        # Enforce single-element-type meshes (no mixed elements).
        if element_type == "hex":
            if num_tets > 0 or num_other_3d > 0 or num_hexes == 0:
                raise RuntimeError(
                    "Mixed/non-hexahedral mesh produced for a hex-only request "
                    f"({json_path}): C3D8={num_hexes}, C3D4={num_tets}, "
                    f"other_3d={num_other_3d}. Adjust the contour/mesh size so the "
                    "2D mesh recombines into all quadrangles."
                )
        else:
            if num_hexes > 0 or num_other_3d > 0 or num_tets == 0:
                raise RuntimeError(
                    "Mixed/non-tetrahedral mesh produced for a tet-only request "
                    f"({json_path}): C3D4={num_tets}, C3D8={num_hexes}, "
                    f"other_3d={num_other_3d}."
                )

        # Write the mesh FIRST, before any graphics. The FLTK/OpenGL render
        # buffers (below) allocate memory proportional to the element count and
        # can exhaust RAM for very large meshes (millions of elements), aborting
        # the process. Writing here guarantees the .msh is saved regardless of
        # whether visualization succeeds. For large meshes leave both
        # ``show_plot`` and ``save_png`` off to skip the render entirely.
        gmsh.write(out_msh)

        # Optional visualization for both element types. ``save_png`` writes a
        # screenshot tagged with the node/element counts; ``show_plot`` opens
        # the interactive viewer (blocking). Both build full render buffers, so
        # they are disabled by default -- enable only for small meshes.
        out_png = (
            os.path.splitext(out_msh)[0]
            + f"_nodes{len(node_tags)}_elems{num_elements}.png"
        )
        if save_png or show_plot:
            gmsh.fltk.initialize()
            gmsh.graphics.draw()
            if save_png:
                gmsh.write(out_png)
            if show_plot:
                gmsh.fltk.run()

        return {
            "nodes": len(node_tags),
            "element_type": element_type,
            "linear_tets": num_tets,
            "linear_hexes": num_hexes,
            "image": out_png if save_png else None,
            "mesh_size": interior_size,
            "thickness": thickness,
        }

    finally:
        gmsh.finalize()


def batch_build_3d_msh_from_contours(
    foldername,
    mesh_size=1.0,
    thickness=10.0,
    element_type="tet",
    num_mesh_each_contour=1,
    mesh_size_delta=0.0,
    random_factor=0.0,
    show_plot=False,
    save_png=False,
):
    if element_type not in ("tet", "hex"):
        raise ValueError(
            f"element_type must be 'tet' or 'hex', got {element_type!r}."
        )

    contour_files = sorted(glob.glob(os.path.join(foldername, "contour_*.json")))
    if len(contour_files) == 0:
        raise FileNotFoundError(
            f"No contour_*.json files found in {foldername}. "
            "Run Step1_generate_contours.py with example = 'L_beam' first."
        )

    all_msh_files = []
    mesh_tag = f"{mesh_size:.3f}".rstrip("0").rstrip(".")
    thick_tag = f"{thickness:.3f}".rstrip("0").rstrip(".")
    elem_tag = "hex3d" if element_type == "hex" else "tet3d"

    for contour_path in contour_files:
        contour_stem = os.path.splitext(os.path.basename(contour_path))[0]

        for i in range(1, num_mesh_each_contour + 1):
            seed = i
            out_msh = os.path.join(
                foldername,
                f"{contour_stem}_{elem_tag}_h{mesh_tag}_t{thick_tag}_{i:03d}.msh",
            )

            info = mesh_contours_json_to_3d_msh(
                json_path=contour_path,
                out_msh=out_msh,
                thickness=thickness,
                mesh_size=mesh_size,
                element_type=element_type,
                randomize=True,
                mesh_size_delta=mesh_size_delta,
                random_factor=random_factor,
                seed=seed,
                show_plot=show_plot,
                save_png=save_png,
            )

            all_msh_files.append(out_msh)
            print(
                f"[done] {out_msh} | nodes={info['nodes']} "
                f"linear_tets={info['linear_tets']} linear_hexes={info['linear_hexes']} "
                f"h={info['mesh_size']:.6f} | image={info['image'] or 'skipped'}"
            )

    return all_msh_files


def mesh_l_beam_structured_3d_msh(
    out_msh,
    thickness=10.0,
    mesh_size=1.0,
    num_layers=None,
    msh_version=4.1,
    show_plot=False,
    save_png=False,
):
    """
    Build a *uniform, structured* all-hex (C3D8) mesh of the L-beam by
    transfinite multi-block meshing, then extruding in +z.

    Unlike ``mesh_contours_json_to_3d_msh`` (unstructured Frontal-Delaunay +
    Blossom recombination + all-hex subdivision), this routine decomposes the
    L-shape into rectangular blocks and meshes each as a transfinite surface.
    Every block yields a perfectly regular quad grid, so:

      * the 2D mesh is all-quads with NO stray boundary triangles, hence
      * NO SubdivisionAlgorithm is needed (that 8x split was the source of the
        finer-near-boundary band), and
      * the extrusion produces pure C3D8 hexes at the target size everywhere.

    Geometry (matches Step 1 'L_beam' vertices):

        y3=100  o-----o
                |  E  |
        y2= 40  o-----o-----------o
                |  C  |     D     |
        y1= 35  o-----o-----------o
                |  A  |     B     |
        y0=  0  o-----o-----------o
              x0=0  x1=40      x2=100

    The internal split lines at y=35 and y=40 guarantee node lines there, so
    Step 5's coordinate-based load patch (x==100, 35<=y<=40) is clean.
    """
    try:
        import gmsh
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Python package 'gmsh' is required. Install it in this Python "
            "environment before running Step2_generate_mesh_3D.py."
        ) from exc

    if thickness <= 0:
        raise ValueError("thickness must be positive.")

    # L-beam grid lines.
    x0, x1, x2 = 0.0, 40.0, 100.0
    y0, y1, y2, y3 = 0.0, 35.0, 40.0, 100.0

    h = float(mesh_size)

    def ndiv(length):
        # Round to the nearest EVEN number of intervals (>= 2). Structured
        # recombination of a transfinite surface into quads requires an even
        # subdivision on each side; if given an odd count gmsh silently bumps it
        # to even, which makes the element size unpredictable. Forcing even here
        # keeps the spacing deterministic and close to the target size h.
        n = int(round(length / h))
        n = max(2, n)
        if n % 2 == 1:
            n += 1
        return n

    # Division counts. Shared block edges use identical counts, so every block
    # interface is conformal.
    nx_left = ndiv(x1 - x0)   # x in [0, 40]
    nx_right = ndiv(x2 - x1)  # x in [40, 100]
    ny_bot = ndiv(y1 - y0)    # y in [0, 35]
    ny_mid = ndiv(y2 - y1)    # y in [35, 40]
    ny_top = ndiv(y3 - y2)    # y in [40, 100]
    n_layers = ndiv(thickness) if num_layers is None else max(1, int(num_layers))

    gmsh.initialize()
    gmsh.model.add("l_beam_structured_3d")
    try:
        gmsh.option.setNumber("Mesh.MshFileVersion", msh_version)

        geo = gmsh.model.geo
        lc = h  # ignored under transfinite control, kept for completeness

        # Points (only block corners of the L-shape).
        p00 = geo.addPoint(x0, y0, 0.0, lc)
        p10 = geo.addPoint(x1, y0, 0.0, lc)
        p20 = geo.addPoint(x2, y0, 0.0, lc)
        p01 = geo.addPoint(x0, y1, 0.0, lc)
        p11 = geo.addPoint(x1, y1, 0.0, lc)
        p21 = geo.addPoint(x2, y1, 0.0, lc)
        p02 = geo.addPoint(x0, y2, 0.0, lc)
        p12 = geo.addPoint(x1, y2, 0.0, lc)
        p22 = geo.addPoint(x2, y2, 0.0, lc)
        p03 = geo.addPoint(x0, y3, 0.0, lc)
        p13 = geo.addPoint(x1, y3, 0.0, lc)

        # Lines. Shared edges are single entities reused by both neighbours, so
        # the resulting volume mesh is conformal across block boundaries.
        H0a = geo.addLine(p00, p10)
        H0b = geo.addLine(p10, p20)
        H35a = geo.addLine(p01, p11)
        H35b = geo.addLine(p11, p21)
        H40a = geo.addLine(p02, p12)
        H40b = geo.addLine(p12, p22)
        H100a = geo.addLine(p03, p13)

        V0a = geo.addLine(p00, p01)
        V0b = geo.addLine(p01, p02)
        V0c = geo.addLine(p02, p03)
        V40a = geo.addLine(p10, p11)
        V40b = geo.addLine(p11, p12)
        V40c = geo.addLine(p12, p13)
        V100a = geo.addLine(p20, p21)
        V100b = geo.addLine(p21, p22)

        # Blocks A..E as transfinite quad surfaces.
        clA = geo.addCurveLoop([H0a, V40a, -H35a, -V0a])
        clB = geo.addCurveLoop([H0b, V100a, -H35b, -V40a])
        clC = geo.addCurveLoop([H35a, V40b, -H40a, -V0b])
        clD = geo.addCurveLoop([H35b, V100b, -H40b, -V40b])
        clE = geo.addCurveLoop([H40a, V40c, -H100a, -V0c])
        sA = geo.addPlaneSurface([clA])
        sB = geo.addPlaneSurface([clB])
        sC = geo.addPlaneSurface([clC])
        sD = geo.addPlaneSurface([clD])
        sE = geo.addPlaneSurface([clE])
        surfaces = [sA, sB, sC, sD, sE]

        # Transfinite constraints MUST be set on the geo kernel BEFORE
        # synchronize/extrude. Setting them via the post-synchronize
        # gmsh.model.mesh.* API is silently ignored by the geo-kernel structured
        # extrusion, which then remeshes the base surfaces unstructured (yielding
        # a distorted, non-uniform interior).
        def tf_curve(tag, ndiv_):
            geo.mesh.setTransfiniteCurve(tag, ndiv_ + 1)

        for t in (H0a, H35a, H40a, H100a):
            tf_curve(t, nx_left)
        for t in (H0b, H35b, H40b):
            tf_curve(t, nx_right)
        for t in (V0a, V40a, V100a):
            tf_curve(t, ny_bot)
        for t in (V0b, V40b, V100b):
            tf_curve(t, ny_mid)
        for t in (V0c, V40c):
            tf_curve(t, ny_top)

        for s in surfaces:
            geo.mesh.setTransfiniteSurface(s)
            geo.mesh.setRecombine(2, s)

        # Extrude ALL surfaces in one call so shared side faces stay conformal.
        extruded = geo.extrude(
            [(2, s) for s in surfaces],
            0.0,
            0.0,
            thickness,
            numElements=[n_layers],
            recombine=True,
        )
        geo.synchronize()

        volume_tags = [tag for dim, tag in extruded if dim == 3]
        if len(volume_tags) != len(surfaces):
            raise RuntimeError(
                f"Expected {len(surfaces)} extruded volumes, got {len(volume_tags)}."
            )

        # Classify boundary surfaces. FRONT/BACK are the z=const caps; SIDE is
        # the *exterior* lateral faces only (interior faces shared by two blocks
        # are adjacent to two volumes and are excluded).
        z_tol = max(1e-8, abs(thickness) * 1e-8)
        front_tags, back_tags, side_tags = [], [], []
        for dim, tag in gmsh.model.getEntities(2):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
            if abs(zmin) < z_tol and abs(zmax) < z_tol:
                front_tags.append(tag)
            elif abs(zmin - thickness) < z_tol and abs(zmax - thickness) < z_tol:
                back_tags.append(tag)
            else:
                up_volumes, _ = gmsh.model.getAdjacencies(2, tag)
                if len(up_volumes) == 1:  # exterior lateral face
                    side_tags.append(tag)

        pg_volume = gmsh.model.addPhysicalGroup(3, volume_tags)
        gmsh.model.setPhysicalName(3, pg_volume, "DOMAIN")
        pg_front = gmsh.model.addPhysicalGroup(2, front_tags)
        gmsh.model.setPhysicalName(2, pg_front, "FRONT_Z0")
        pg_back = gmsh.model.addPhysicalGroup(2, back_tags)
        gmsh.model.setPhysicalName(2, pg_back, "BACK_ZT")
        if side_tags:
            pg_side = gmsh.model.addPhysicalGroup(2, side_tags)
            gmsh.model.setPhysicalName(2, pg_side, "SIDE")

        gmsh.model.mesh.generate(3)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        elem_types, elem_tags_list, _ = gmsh.model.mesh.getElements(3)
        num_hexes = 0
        num_other_3d = 0
        for et, e_tags in zip(elem_types, elem_tags_list):
            _, dim, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(et)
            if dim != 3:
                continue
            if num_nodes == 8:
                num_hexes += len(e_tags)
            else:
                num_other_3d += len(e_tags)

        if num_other_3d > 0 or num_hexes == 0:
            raise RuntimeError(
                "Structured L-beam mesh is not pure hexahedral "
                f"(C3D8={num_hexes}, other_3d={num_other_3d})."
            )

        gmsh.write(out_msh)

        out_png = (
            os.path.splitext(out_msh)[0]
            + f"_nodes{len(node_tags)}_elems{num_hexes}.png"
        )
        if save_png or show_plot:
            gmsh.fltk.initialize()
            gmsh.graphics.draw()
            if save_png:
                gmsh.write(out_png)
            if show_plot:
                gmsh.fltk.run()

        return {
            "nodes": len(node_tags),
            "element_type": "hex",
            "linear_tets": 0,
            "linear_hexes": num_hexes,
            "image": out_png if save_png else None,
            "mesh_size": h,
            "thickness": thickness,
            "divisions": {
                "nx_left": nx_left,
                "nx_right": nx_right,
                "ny_bot": ny_bot,
                "ny_mid": ny_mid,
                "ny_top": ny_top,
                "n_layers": n_layers,
            },
        }

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    example = "L_beam"
    sampling_size = 0.203
    thickness = 10.0
    element_type = "hex"   # "tet" -> C3D4, "hex" -> C3D8 (hex-only, errors on mixed)

    # "structured" -> uniform transfinite multi-block hex (Option A, L_beam only)
    # "contour"    -> original unstructured contour-based mesher
    method = "structured"

    folder_name = f"{example}_sampling_size{sampling_size}"
    mesh_size_delta = 0.0
    random_factor = 0.00
    num_mesh_each_contour = 1

    # Visualization is OFF for large meshes: FLTK/OpenGL render buffers scale
    # with element count and can exhaust RAM (the ~10M-element crash). Turn
    # these on only when meshing a small case you want to inspect.
    show_plot = True
    save_png = True

    if method == "structured":
        if example != "L_beam":
            raise ValueError("The structured mesher is implemented for 'L_beam' only.")
        os.makedirs(folder_name, exist_ok=True)
        mesh_tag = f"{sampling_size:.3f}".rstrip("0").rstrip(".")
        thick_tag = f"{thickness:.3f}".rstrip("0").rstrip(".")
        out_msh = os.path.join(
            folder_name,
            f"{example}_structured_hex3d_h{mesh_tag}_t{thick_tag}_001.msh",
        )
        info = mesh_l_beam_structured_3d_msh(
            out_msh=out_msh,
            thickness=thickness,
            mesh_size=sampling_size,
            show_plot=show_plot,
            save_png=save_png,
        )
        all_msh_files = [out_msh]
        print(
            f"[done] {out_msh} | nodes={info['nodes']} "
            f"linear_hexes={info['linear_hexes']} h={info['mesh_size']:.6f} "
            f"divisions={info['divisions']} | image={info['image'] or 'skipped'}"
        )
    else:
        all_msh_files = batch_build_3d_msh_from_contours(
            foldername=folder_name,
            mesh_size=sampling_size,
            thickness=thickness,
            element_type=element_type,
            num_mesh_each_contour=num_mesh_each_contour,
            mesh_size_delta=mesh_size_delta,
            random_factor=random_factor,
            show_plot=show_plot,
            save_png=save_png,
        )

    print(f"\nGenerated {len(all_msh_files)} 3D mesh files:")
    for msh_file in all_msh_files:
        print(msh_file)
