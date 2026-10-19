import glob
import os

import numpy as np


def write_id_block(f, ids, per_line=16):
    ids = list(map(int, ids))
    for i in range(0, len(ids), per_line):
        chunk = ids[i:i + per_line]
        f.write(", ".join(str(v) for v in chunk) + "\n")


def collect_nodes_for_physical_surface_groups(gmsh):
    surface_nodes = {}

    for dim, tag in gmsh.model.getPhysicalGroups():
        if dim != 2:
            continue

        name = gmsh.model.getPhysicalName(dim, tag)
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)

        ids = []
        for ent in entities:
            node_ids, _, _ = gmsh.model.mesh.getNodes(dim, ent, includeBoundary=True)
            if len(node_ids) > 0:
                ids.append(np.asarray(node_ids, dtype=int))

        if len(ids) > 0:
            surface_nodes[name] = np.unique(np.concatenate(ids))
        else:
            surface_nodes[name] = np.array([], dtype=int)

    return surface_nodes


def write_abaqus_inp_3d(
    out_inp,
    part_name,
    node_tags,
    node_coords,
    elem_conn,
    surface_nodes=None,
    element_type="C3D4",
):
    """
    Write a 3D orphan-mesh Abaqus input file.

    Supports linear tetrahedra (C3D4, 4 nodes/element) and linear hexahedra
    (C3D8, 8 nodes/element). Gmsh and Abaqus share the same corner-node
    ordering for both, so the connectivity is written as-is.

    node_tags: shape (N,)
    node_coords: shape (N, 3)
    elem_conn: shape (Ne, 4) for C3D4 or (Ne, 8) for C3D8, with Abaqus node ids
    surface_nodes: dict like {"FRONT_Z0": np.array([...]), ...}
    """
    with open(out_inp, "w") as f:
        f.write("*Heading\n")
        f.write("** Generated from Gmsh 3D mesh ({})\n".format(element_type))
        f.write("*Part, name={}\n".format(part_name))

        f.write("*Node\n")
        for nid, xyz in zip(node_tags, node_coords):
            f.write("{}, {}, {}, {}\n".format(int(nid), xyz[0], xyz[1], xyz[2]))

        f.write("*Element, type={}\n".format(element_type))
        for eid, conn in enumerate(elem_conn, start=1):
            conn_str = ", ".join(str(int(v)) for v in conn)
            f.write(f"{eid}, {conn_str}\n")

        f.write("*Elset, elset=EALL, generate\n")
        f.write("1, {}, 1\n".format(len(elem_conn)))

        f.write("*Nset, nset=NALL\n")
        write_id_block(f, np.asarray(node_tags, dtype=int))

        if surface_nodes:
            for name, ids in surface_nodes.items():
                ids = np.unique(np.asarray(ids, dtype=int))
                if len(ids) == 0:
                    continue
                f.write("*Nset, nset={}\n".format(name))
                write_id_block(f, ids)

        f.write("*End Part\n")
        f.write("*Assembly, name=Assembly\n")
        f.write("*Instance, name={}-1, part={}\n".format(part_name, part_name))
        f.write("*End Instance\n")
        f.write("*End Assembly\n")


def msh_to_inp_3d(
    msh_path,
    out_inp,
    part_name="PART-1",
    element_type=None,
):
    """
    Read a Gmsh .msh file containing 3D elements and write Abaqus .inp.

    The element geometry (tet vs hex) is auto-detected from the mesh; mixed
    meshes (both tets and hexes) are rejected. The Abaqus element keyword is
    chosen as follows:
      - tet geometry  -> C3D4
      - hex geometry  -> C3D8 (full integration) by default, or C3D8R (reduced
                         integration) if element_type="C3D8R" is requested.
    Reduced vs full integration cannot be inferred from geometry, so it must be
    selected explicitly via ``element_type``. If ``element_type`` is given it is
    validated against the detected geometry.

    Returns
    -------
    node_coords  : (N_node, 3) np.ndarray
    elem_conn    : (N_elem, 4 or 8) np.ndarray, using Abaqus node ids
    element_type : str, the detected Abaqus element type ("C3D4" or "C3D8")
    """
    try:
        import gmsh
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Python package 'gmsh' is required to convert .msh to .inp."
        ) from exc

    if not os.path.exists(msh_path):
        raise FileNotFoundError(f"Missing mesh file: {msh_path}")

    gmsh.initialize()
    gmsh.model.add("read_msh_3d")

    try:
        gmsh.open(msh_path)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=int)
        node_coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)

        if len(node_tags) == 0:
            raise RuntimeError(f"No nodes found in mesh: {msh_path}")

        elem_types, _, elem_node_tags_list = gmsh.model.mesh.getElements(3)
        tet_conn_blocks = []
        hex_conn_blocks = []

        for et, e_nodes in zip(elem_types, elem_node_tags_list):
            name, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(et)

            if dim != 3:
                continue
            if num_nodes == 4:
                tet_conn_blocks.append(np.asarray(e_nodes, dtype=int).reshape(-1, 4))
            elif num_nodes == 8:
                hex_conn_blocks.append(np.asarray(e_nodes, dtype=int).reshape(-1, 8))
            elif num_nodes in (10, 20, 27):
                raise RuntimeError(
                    f"{msh_path} contains second-order elements ({name}). "
                    "Use linear meshing or extend this script for C3D10/C3D20."
                )
            else:
                raise RuntimeError(
                    f"{msh_path} contains an unsupported 3D element ({name}, "
                    f"{num_nodes} nodes)."
                )

        n_tet = sum(b.shape[0] for b in tet_conn_blocks)
        n_hex = sum(b.shape[0] for b in hex_conn_blocks)
        if n_tet > 0 and n_hex > 0:
            raise RuntimeError(
                f"{msh_path} is a mixed mesh (C3D4={n_tet}, C3D8={n_hex}). "
                "Only single-element-type meshes are supported."
            )
        if n_tet == 0 and n_hex == 0:
            raise RuntimeError(f"No linear tet/hex 3D elements found in {msh_path}")

        if n_hex > 0:
            elem_conn = np.vstack(hex_conn_blocks)
            geometry = "hex"
            allowed_types = ("C3D8", "C3D8R")
        else:
            elem_conn = np.vstack(tet_conn_blocks)
            geometry = "tet"
            allowed_types = ("C3D4",)

        if element_type is None:
            written_type = allowed_types[0]
        elif element_type in allowed_types:
            written_type = element_type
        else:
            raise RuntimeError(
                f"{msh_path} contains {geometry} elements; element_type must be "
                f"one of {allowed_types}, got {element_type!r}."
            )

        surface_nodes = collect_nodes_for_physical_surface_groups(gmsh)

        write_abaqus_inp_3d(
            out_inp=out_inp,
            part_name=part_name,
            node_tags=node_tags,
            node_coords=node_coords,
            elem_conn=elem_conn,
            surface_nodes=surface_nodes,
            element_type=written_type,
        )

        return node_coords, elem_conn, written_type

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    example = "Portal_frame_half"
    sampling_size = 0.28
    # "tet" -> C3D4, "hex" -> C3D8 (full), "hexR" -> C3D8R (reduced integration)
    element_type = "hex"
    folder_name = f"{example}_sampling_size{sampling_size}"

    abq_type = {"tet": "C3D4", "hex": "C3D8", "hexR": "C3D8R"}[element_type]
    msh_glob = "*_tet3d_*.msh" if element_type == "tet" else "*_hex3d_*.msh"

    msh_files = sorted(glob.glob(os.path.join(folder_name, msh_glob)))
    if len(msh_files) == 0:
        raise FileNotFoundError(
            f"No 3D {element_type} .msh files found in {folder_name}. "
            "Run Step2_generate_mesh_3D.py first."
        )

    for i, msh_path in enumerate(msh_files, start=1):
        out_inp = os.path.join(folder_name, f"Mesh_{i:02d}_3D.inp")
        print(f"[read] {msh_path}")

        nodes, elements, detected_type = msh_to_inp_3d(
            msh_path=msh_path,
            out_inp=out_inp,
            part_name="PART-1",
            element_type=abq_type,
        )

        print(
            f"[done] {out_inp} | type={detected_type} nodes={nodes.shape[0]} "
            f"elements={elements.shape[0]}"
        )

    print("OK")
