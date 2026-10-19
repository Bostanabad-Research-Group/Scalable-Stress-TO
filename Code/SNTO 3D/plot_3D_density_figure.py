"""
plot_3D_density_L_beam.py
=========================
Publication-style 3D density visualization (isosurface + ambient-occlusion
shadowing) for the NN-TO results -- now BATCHED over multiple examples and reading
the CACHED density for speed.

It reads the caches written by preprocess_cache_3D_density.py:
    <example>/Run_<r>/density_field_<tag>.npz   (points, conn, rho, + BC/material)
and renders one figure per (example x run x selection). NO torch here -- pyvista only
(so it is fast and avoids the torch+VTK segfault). Run preprocess_cache_3D_density.py
first (TO env) to build the caches.

Run inside the FEM/plot env (needs: pyvista):
    python plot_3D_density_L_beam.py
"""

import os
import glob
import numpy as np

# =========================================================================
#                              PARAMETERS
# =========================================================================
# --- BATCH SCOPE (mirror preprocess_cache_3D_density.py) ------------------
EXAMPLES = [
    'Results/L_beam_3D/C2_L_beam_TO10_Sophia',
    'Results/L_beam_3D/C2_L_beam_TO11_Sophia',
    'Results/Portal_frame_half_3D/C2_Portal_frame_half_TO10_Sophia',
    'Results/Portal_frame_half_3D/C2_Portal_frame_half_TO11_Sophia',
]
RUNS       = [1, 2, 3, 4, 5, 6]
RUNS       = [1,2,3]
SELECTIONS = ['iter:29999', 'best:1', 'best:2', 'best:3',
              'gray:1', 'gray:2', 'gray:3', 'gray:4', 'gray:5']
FORCE      = False            # True -> re-render even if the output image exists

# --- isosurface / surface extraction --------------------------------------
RHO_ISO       = 0.5         # density cutoff (paper: 0.5)
SURFACE_METHOD = 'threshold'  # 'threshold' -> outer skin of solid cells (CLOSED body,
                              #   includes domain-boundary faces; recommended).
                              # 'contour'   -> marching cubes (smoother, OPEN at boundary)

# --- surface smoothing ----------------------------------------------------
SMOOTH_ENABLE   = True
SMOOTH_METHOD   = 'taubin'   # 'taubin' (volume-preserving) | 'laplacian' | 'none'
SMOOTH_ITER     = 30
SMOOTH_PASSBAND = 0.05       # taubin only, (0,2): LOWER = smoother
SMOOTH_RELAX    = 0.1        # laplacian only

# --- surface appearance ---------------------------------------------------
SURF_COLOR      = '#e10600'  # solid density color
SMOOTH_SHADING  = True
SPECULAR        = 0.35
SPECULAR_POWER  = 25.0
AMBIENT         = 0.35
DIFFUSE         = 0.85
OPACITY         = 1.0

# --- "shadowing" / depth --------------------------------------------------
SSAO            = False      # OFF: SSAO shades the GPU-tessellated sub-triangle edges of
                             #   the (smoothed) isosurface -> faint vertical/horizontal line
                             #   artifacts. Directional lighting gives enough depth. Set True
                             #   only if you accept those lines for stronger ambient occlusion.
SSAO_RADIUS     = 8.0
SSAO_BIAS       = 0.01
SSAO_KERNEL     = 256
CAST_SHADOWS    = False

# --- design-domain wireframe ---------------------------------------------
SHOW_BOX        = True
BOX_STYLE       = 'domain'   # 'domain' (L-shape edges) | 'bbox' (rectangular box)
BOX_COLOR       = 'black'
BOX_WIDTH       = 2.0
BOX_FEATURE_ANG = 30.0

# --- scene / camera -------------------------------------------------------
BACKGROUND      = 'white'
WINDOW_SIZE     = (1600, 1200)
# EXACT viewpoint (paste a printed "[camera] ..." tuple). When set (not None), it
# OVERRIDES the preset + azimuth/elevation below.
CAMERA_POSITION = [(249.3, 115.1, 262.3),      # camera position
                   (50.0, 50.0, 5.0),          # focal point (domain center)
                   (-0.119, 0.981, -0.156)]    # view-up
CAMERA_VIEW     = 'iso'        # preset when CAMERA_POSITION is None: 'iso'|'xy'|'xz'|'yz'
CAM_AZIMUTH     = -45.0
CAM_ELEVATION   = 20.0
CAM_ROLL        = None
CAM_ZOOM        = 1.3
# --- FILL THE FRAME (make the part LARGE at full resolution, instead of cropping) ---
# Applies to BOTH the fixed-CAMERA_POSITION view and the preset view, and to the saved
# figure AND the interactive window. ZOOM_TO_FIT slides the camera along the view
# direction so the part fills the frame (keeps the exact viewing angle); FILL_ZOOM then
# magnifies further (>1 = bigger/closer, may clip; <1 = more margin).
ZOOM_TO_FIT     = True
FILL_ZOOM       = 1.1
PARALLEL_PROJ   = False        # False = 3D perspective; True = orthographic
VIEW_ANGLE      = 23.0         # perspective FOV (deg) when PARALLEL_PROJ=False
ANTIALIAS       = 'ssaa'

# --- output ---------------------------------------------------------------
# For a batch, OFFSCREEN=True saves images without opening windows. With one case and
# OFFSCREEN=False you get the interactive window (and the "[camera]" print to copy).
# Output is a high-res PDF: the (supersampled) render is captured as an image array and
# embedded in a vector PDF at DPI. IMAGE_SCALE supersamples the render (pixels), DPI sets
# the PDF's physical resolution (pixels/inch) -> page size = WINDOW_SIZE*IMAGE_SCALE / DPI.
OFFSCREEN       = True
SAVE_IMAGE      = True
IMAGE_SCALE     = 2
DPI             = 600         # PDF resolution for the saved density figure
PDF_AUTOCROP    = False       # trim the white background so the topology fills the PDF page
                              #   (OFF by default -> use ZOOM_TO_FIT/FILL_ZOOM to enlarge the
                              #    part at FULL resolution rather than shrinking the page)
PDF_PAD_PX      = 12          # margin (px) kept around the object when autocropping
PDF_CROP_THRESH = 245         # a pixel counts as "content" if any RGB channel < this (white=255)
PDF_EDGE_IGNORE = 4           # ignore this many px at each edge so a thin window/page frame
                              #   border can't anchor the crop box to the full frame

# --- hex node ordering (Abaqus C3D8 == VTK; set a permutation only if needed) ---
HEX_REORDER     = None
# =========================================================================
#                          END PARAMETERS
# =========================================================================


def resolve_npz(run_folder, sel):
    """(npz_path or None, tag). Mirrors preprocess_cache_3D_density.py tag naming."""
    kind, _, val = sel.partition(':')
    kind = kind.strip().lower(); val = val.strip()
    if kind == 'iter':
        tag = str(val)
        p = os.path.join(run_folder, f'density_field_{tag}.npz')
        return (p if os.path.isfile(p) else None), tag
    if kind == 'best':
        pat = os.path.join(run_folder, f'density_field_best{val}_*.npz'); stem = f'best{val}'
    elif kind == 'gray':
        pat = os.path.join(run_folder, f'density_field_gray{val}p_*.npz'); stem = f'gray{val}p'
    else:
        raise ValueError(f'unknown selection kind: {sel!r}')
    matches = sorted(glob.glob(pat),
                     key=lambda q: int(q.rsplit('_', 1)[1].split('.')[0]))
    if not matches:
        return None, stem
    p = matches[-1]
    tag = os.path.basename(p)[len('density_field_'):-len('.npz')]
    return p, tag


# --------------------------------------------------------------- pyvista side
def build_grid(points, conn, rho):
    conn = np.asarray(conn, dtype=np.int64)
    if HEX_REORDER is not None:
        conn = conn[:, HEX_REORDER]
    ne = conn.shape[0]
    cells = np.empty((ne, 9), dtype=np.int64)
    cells[:, 0] = 8
    cells[:, 1:] = conn
    celltypes = np.full(ne, pv.CellType.HEXAHEDRON, np.uint8)
    grid = pv.UnstructuredGrid(cells.ravel(), celltypes, np.asarray(points, float))
    rho = np.asarray(rho, float)
    grid.cell_data['rho'] = rho[:, 0] if rho.ndim > 1 else rho
    return grid


def make_isosurface(grid):
    if SURFACE_METHOD == 'threshold':
        solid = grid.threshold(RHO_ISO, scalars='rho')
        iso = solid.extract_surface()
    else:
        gp = grid.cell_data_to_point_data()
        iso = gp.contour(isosurfaces=[RHO_ISO], scalars='rho')
    if iso.n_points == 0:
        raise RuntimeError(
            f'Empty surface at rho={RHO_ISO}. rho range is '
            f'[{grid.cell_data["rho"].min():.3f}, {grid.cell_data["rho"].max():.3f}].')
    if SMOOTH_ENABLE and SMOOTH_ITER > 0 and SMOOTH_METHOD != 'none':
        if SMOOTH_METHOD == 'taubin':
            iso = iso.smooth_taubin(n_iter=SMOOTH_ITER, pass_band=SMOOTH_PASSBAND)
        else:
            iso = iso.smooth(n_iter=SMOOTH_ITER, relaxation_factor=SMOOTH_RELAX)
    try:
        return iso.compute_normals(auto_orient_normals=True)
    except TypeError:
        return iso.compute_normals()


def domain_wireframe(grid):
    if BOX_STYLE == 'bbox':
        return grid.outline()
    surf = grid.extract_surface()
    return surf.extract_feature_edges(
        feature_angle=BOX_FEATURE_ANG, boundary_edges=True,
        feature_edges=True, manifold_edges=False, non_manifold_edges=False)


def _save_image_pdf(img, out_path, dpi=DPI, pad_px=PDF_PAD_PX):
    """Save an RGB(A) pyvista screenshot array as a high-res PDF at `dpi`, embedding the
    shaded/SSAO render faithfully. The render is AUTOCROPPED to the object's bounding box
    (uniform near-white background trimmed, a small `pad_px` margin kept) so the topology
    FILLS the page instead of sitting tiny in a sea of white -- the camera frames the part
    with lots of empty space, and without cropping all of it lands in the PDF. matplotlib
    is imported lazily so the pyvista-only fast path stays matplotlib-free until a save."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a = np.asarray(img)
    if PDF_AUTOCROP:
        # background is white; the isosurface + black domain box are not -> keep those.
        content = np.any(a[..., :3].astype(np.int16) < PDF_CROP_THRESH, axis=-1)
        # blank out a thin edge frame so a 1-2px window/page border can't hold the crop
        # box open to the whole frame (the exact failure that left the object tiny).
        e = PDF_EDGE_IGNORE
        if e > 0 and content.shape[0] > 2 * e and content.shape[1] > 2 * e:
            content[:e, :] = content[-e:, :] = False
            content[:, :e] = content[:, -e:] = False
        if content.any():
            ys = np.where(content.any(axis=1))[0]; xs = np.where(content.any(axis=0))[0]
            y0 = max(int(ys[0]) - pad_px, 0); y1 = min(int(ys[-1]) + 1 + pad_px, a.shape[0])
            x0 = max(int(xs[0]) - pad_px, 0); x1 = min(int(xs[-1]) + 1 + pad_px, a.shape[1])
            a = a[y0:y1, x0:x1]
    h, w = a.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(a)
    fig.savefig(out_path, format='pdf', dpi=dpi)
    plt.close(fig)


def render(cache_path, out_path):
    """pyvista-only: load the cached density field and render one figure."""
    d = np.load(cache_path)
    grid = build_grid(d['points'], d['conn'], d['rho'])
    r = grid.cell_data['rho']
    print(f'   [grid] {grid.n_cells} cells, rho in [{r.min():.3f}, {r.max():.3f}]', flush=True)
    iso = make_isosurface(grid)
    print(f'   [iso ] {iso.n_points} pts, {iso.n_cells} faces at rho={RHO_ISO}', flush=True)

    pl = pv.Plotter(off_screen=OFFSCREEN, window_size=list(WINDOW_SIZE))
    pl.set_background(BACKGROUND)
    if ANTIALIAS:
        pl.enable_anti_aliasing(ANTIALIAS)
    pl.add_mesh(iso, color=SURF_COLOR, smooth_shading=SMOOTH_SHADING,
                specular=SPECULAR, specular_power=SPECULAR_POWER,
                ambient=AMBIENT, diffuse=DIFFUSE, opacity=OPACITY)
    if SHOW_BOX:
        pl.add_mesh(domain_wireframe(grid), color=BOX_COLOR, line_width=BOX_WIDTH)
    if SSAO:
        pl.enable_ssao(radius=SSAO_RADIUS, bias=SSAO_BIAS, kernel_size=SSAO_KERNEL)
    if CAST_SHADOWS:
        pl.enable_shadows()

    if PARALLEL_PROJ:
        pl.enable_parallel_projection()
    else:
        pl.disable_parallel_projection()
    if CAMERA_POSITION is not None:
        pl.camera_position = CAMERA_POSITION
        if not PARALLEL_PROJ:
            pl.camera.view_angle = VIEW_ANGLE
    else:
        pl.camera_position = CAMERA_VIEW
        if CAM_AZIMUTH:
            pl.camera.Azimuth(CAM_AZIMUTH)
        if CAM_ELEVATION:
            pl.camera.Elevation(CAM_ELEVATION)
        pl.camera.OrthogonalizeViewUp()
        if CAM_ROLL is not None:
            pl.camera.roll = CAM_ROLL
        if not PARALLEL_PROJ:
            pl.camera.view_angle = VIEW_ANGLE
        if CAM_ZOOM:
            pl.camera.zoom(CAM_ZOOM)

    # Fill the frame from the chosen viewpoint at FULL resolution (large high-res figure)
    # instead of cropping the page down. Keeps the view angle; only the zoom changes.
    if ZOOM_TO_FIT:
        pl.reset_camera()            # slide camera along the view dir to fit the part
    if FILL_ZOOM and FILL_ZOOM != 1.0:
        pl.camera.zoom(FILL_ZOOM)    # extra magnification (>1 bigger, may clip)

    if OFFSCREEN:
        if SAVE_IMAGE:
            img = pl.screenshot(return_img=True, scale=IMAGE_SCALE)   # supersampled RGB array
            _save_image_pdf(img, out_path, dpi=DPI)                   # -> 600-dpi PDF
            print(f'   [out] {out_path}', flush=True)
        pl.close()
    else:
        pl.show(auto_close=False)                                    # keep the window/render alive
        if SAVE_IMAGE:
            img = pl.screenshot(return_img=True, scale=IMAGE_SCALE)
            _save_image_pdf(img, out_path, dpi=DPI)
            print(f'   [out] {out_path}', flush=True)
        print(f'   [camera] CAMERA_POSITION = {[tuple(p) for p in pl.camera_position]}')
        print(f'   [camera] VIEW_ANGLE = {pl.camera.view_angle:.2f}')
        pl.close()


def main():
    global pv
    import pyvista as pv
    here = os.path.dirname(os.path.abspath(__file__))
    for case in EXAMPLES:
        for run in RUNS:
            run_folder = os.path.join(here, case, f'Run_{run}') + os.sep
            if not os.path.isdir(run_folder):
                print(f'[skip] no run folder: {run_folder}'); continue
            for sel in SELECTIONS:
                npz, tag = resolve_npz(run_folder, sel)
                if npz is None:
                    print(f'[skip] {case} Run_{run} {sel}: no density cache '
                          f'(run preprocess_cache_3D_density.py first)'); continue
                out_path = os.path.join(run_folder, f'density_iso{RHO_ISO}_{tag}.pdf')
                if os.path.isfile(out_path) and not FORCE:
                    print(f'[exists] {out_path}'); continue
                print(f'[render] {case} Run_{run} {sel}', flush=True)
                try:
                    render(npz, out_path)
                except Exception as e:
                    print(f'   [error] {e}')


if __name__ == '__main__':
    print('=== plot_3D_density (cached, batched) ===')
    main()
