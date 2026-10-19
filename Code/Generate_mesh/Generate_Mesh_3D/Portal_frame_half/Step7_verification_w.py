"""
Step 7: FEniCSx (dolfinx) reference solution for the +z loaded L_beam.

Same as Step 6, but the external load is applied along +z (out-of-plane)
instead of -y, to provide the reference solution for the +z DEM case
(L_beam_DEM6). For each L_beam .msh file produced by Step2_generate_mesh_3D.py
this script runs a small linear-elasticity simulation and saves separate
contour images of the three displacement components (u, v, w).

Boundary conditions (L_beam):
  - Fixed (u = v = w = 0): nodes with 0 <= x <= 40 and y == 100, for all z
    (the top of the vertical arm).
  - Nodal load: nodes with x == 100 and 35 <= y <= 40, for all z (the tip of
    the horizontal arm). The total load magnitude is 2.0, applied in +z and
    split equally over those nodes (mirrors the L_beam_DEM6 load definition).

Run SERIALLY (the nodal point-load distribution assumes a single process):
    /opt/anaconda3/envs/FEM/bin/python Step7_verification_w.py

Requires a FEniCSx environment with dolfinx, petsc4py, gmsh and pyvista.
"""

import glob
import os

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

colors_contours = ["#3b4fbf", "#79a2f2", "#c2d7f2", "#f2c9bb", "#f2856d", "#bf0637"]
cmap_contours = LinearSegmentedColormap.from_list("custom_cmap", colors_contours, N=128)

# ----------------------------------------------------------------------------
# Geometric markers for the L_beam boundary conditions and load.
# -----------------------------------
def l_beam_fixed_marker(x, tol=1e-6):
    return (x[0] >= -tol) & (x[0] <= 40.0 + tol) & (np.isclose(x[1], 100.0, atol=tol))


def l_beam_load_marker(x, tol=1e-6):
    return (
        np.isclose(x[0], 100.0, atol=tol)
        & (x[1] >= 35.0 - tol)
        & (x[1] <= 40.0 + tol)
    )


def solve_linear_elasticity(
    domain,
    E,
    nu,
    total_force,
    fixed_marker,
    load_marker,
):
    """
    Solve small-strain linear elasticity with zero-displacement Dirichlet BCs
    on ``fixed_marker`` and equally-distributed nodal point loads summing to
    ``total_force`` on the nodes selected by ``load_marker``.

    Returns the displacement Function ``uh`` and an info dict.
    """
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import (
        apply_lifting,
        assemble_matrix,
        assemble_vector,
        set_bc,
    )
    from petsc4py import PETSc

    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    bs = V.dofmap.index_map_bs
    n_dofs = V.dofmap.index_map.size_global * bs

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(3) + 2.0 * mu * epsilon(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = fem.form(ufl.inner(sigma(u), epsilon(v)) * ufl.dx)
    L = fem.form(ufl.inner(fem.Constant(domain, (0.0, 0.0, 0.0)), v) * ufl.dx)

    # Dirichlet BC: clamp all three components on the fixed region.
    fixed_dofs = fem.locate_dofs_geometrical(V, fixed_marker)
    bc = fem.dirichletbc(fem.Constant(domain, (0.0, 0.0, 0.0)), fixed_dofs, V)

    # Nodal point loads on the loaded region.
    load_blocks = fem.locate_dofs_geometrical(V, load_marker)
    n_loaded = len(load_blocks)
    if n_loaded == 0:
        raise RuntimeError("No loaded nodes found; check the load marker.")
    force_per_node = np.asarray(total_force, dtype=np.float64) / n_loaded

    A = assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = assemble_vector(L)

    # Add the point loads into the right-hand side (serial assumption).
    with b.localForm() as b_local:
        arr = b_local.array
        for blk in load_blocks:
            for comp in range(3):
                arr[blk * bs + comp] += force_per_node[comp]

    apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
    )
    set_bc(b, [bc])

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("gamg")
    ksp.setTolerances(rtol=1e-8)

    uh = fem.Function(V)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    reason = ksp.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"Linear solver did not converge (reason={reason}).")

    disp = uh.x.array.reshape(-1, 3)
    abs_max = np.abs(disp).max(axis=0)
    info = {
        "n_dofs": int(n_dofs),
        "n_fixed": int(len(fixed_dofs)),
        "n_loaded": int(n_loaded),
        "ksp_iterations": int(ksp.getIterationNumber()),
        "max_disp": float(np.linalg.norm(disp, axis=1).max()),
        "max_abs_u": float(abs_max[0]),
        "max_abs_v": float(abs_max[1]),
        "max_abs_w": float(abs_max[2]),
    }
    return uh, V, info


def save_displacement_components(
    V, uh, out_prefix, warp_factor=1.0, azimuth=25.0, elevation=15.0
):
    """
    Render the three displacement components (u, v, w) as separate contour PNGs
    on the UNDEFORMED mesh, viewed roughly along the z-axis (x horizontal, y
    vertical) but tilted slightly so the z-thickness is visible.

    ``azimuth`` / ``elevation`` (degrees) set the tilt away from the pure
    xy-plane view; set both to 0 for a face-on xy view.

    Saves ``<out_prefix>_u.png``, ``<out_prefix>_v.png``, ``<out_prefix>_w.png``
    and returns the list of file paths.
    """
    import pyvista as pv
    from dolfinx.plot import vtk_mesh

    pv.OFF_SCREEN = True

    disp = uh.x.array.reshape(-1, 3)

    cells, cell_types, geometry = vtk_mesh(V)
    grid = pv.UnstructuredGrid(cells, cell_types, geometry)
    grid["u"] = disp
    grid["u_x"] = disp[:, 0]
    grid["u_y"] = disp[:, 1]
    grid["u_z"] = disp[:, 2]

    # Plot on the undeformed configuration (no warp); colour by component.
    components = [("u", "u_x"), ("v", "u_y"), ("w", "u_z")]
    out_paths = []
    for label, array_name in components:
        out_png = f"{out_prefix}_{label}.png"
        plotter = pv.Plotter(off_screen=True, window_size=(900, 1000))
        plotter.add_mesh(
            grid, scalars=array_name, cmap=cmap_contours, show_edges=False,
            scalar_bar_args={"title": label},
        )
        plotter.add_text(os.path.basename(out_png), font_size=10)
        # Start from the xy-plane view, then tilt to expose the z-direction.
        plotter.view_xy()
        plotter.camera.azimuth = azimuth
        plotter.camera.elevation = elevation
        plotter.reset_camera()
        plotter.screenshot(out_png)
        plotter.close()
        out_paths.append(out_png)

    return out_paths


def verify_meshes(
    folder_name,
    msh_glob,
    E,
    nu,
    load_magnitude,
    warp_factor=1.0,
):
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    msh_files = sorted(glob.glob(os.path.join(folder_name, msh_glob)))
    if len(msh_files) == 0:
        raise FileNotFoundError(
            f"No .msh files matching {msh_glob!r} in {folder_name}. "
            "Run Step2_generate_mesh_3D.py first."
        )

    total_force = np.array([0.0, 0.0, float(load_magnitude)])  # +z load

    results = []
    for msh_path in msh_files:
        stem = os.path.splitext(os.path.basename(msh_path))[0]
        print(f"[read] {msh_path}")

        mesh_data = gmshio.read_from_msh(msh_path, MPI.COMM_WORLD, rank=0, gdim=3)
        domain = mesh_data.mesh

        uh, V, info = solve_linear_elasticity(
            domain=domain,
            E=E,
            nu=nu,
            total_force=total_force,
            fixed_marker=l_beam_fixed_marker,
            load_marker=l_beam_load_marker,
        )

        out_prefix = os.path.join(folder_name, f"{stem}_zload")
        images = save_displacement_components(
            V, uh, out_prefix, warp_factor=warp_factor
        )

        info["cell_type"] = domain.topology.cell_name()
        info["images"] = images
        results.append((stem, info))
        print(
            f"[done] {stem} | cell={info['cell_type']} dofs={info['n_dofs']} "
            f"fixed={info['n_fixed']} loaded={info['n_loaded']} "
            f"ksp_its={info['ksp_iterations']} "
            f"max|u|={info['max_abs_u']:.4f} max|v|={info['max_abs_v']:.4f} "
            f"max|w|={info['max_abs_w']:.4f}"
        )

    return results


if __name__ == "__main__":
    example = "L_beam"
    sampling_size = 3.0
    element_type = "hex"   # "tet" -> *_tet3d_*.msh, "hex" -> *_hex3d_*.msh

    # Material (linear elastic). Not specified by the problem; adjust as needed.
    E = 7.0
    nu = 0.25

    load_magnitude = 0.2   # total out-of-plane (+z) load magnitude
    warp_factor = 1.0      # deformation scale used only for the contour image

    folder_name = f"{example}_sampling_size{sampling_size}"
    msh_glob = "*_tet3d_*.msh" if element_type == "tet" else "*_hex3d_*.msh"

    results = verify_meshes(
        folder_name=folder_name,
        msh_glob=msh_glob,
        E=E,
        nu=nu,
        load_magnitude=load_magnitude,
        warp_factor=warp_factor,
    )

    print(f"\nVerified {len(results)} mesh(es) in {folder_name}:")
    for stem, info in results:
        print(
            f"  {stem}: max(|u|,|v|,|w|)="
            f"({info['max_abs_u']:.4f}, {info['max_abs_v']:.4f}, "
            f"{info['max_abs_w']:.4f}) -> {len(info['images'])} images"
        )
