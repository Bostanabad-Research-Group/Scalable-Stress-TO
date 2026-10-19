import json
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import math


def canonical_edge_key(p0, p1):
    a = (float(p0[0]), float(p0[1]))
    b = (float(p1[0]), float(p1[1]))
    return tuple(sorted((a, b)))


def build_polygon_edge_controls(
    vertices,    default_sampling_size,
    edge_overrides=None,
    default_transition_distance=None,
):
    """
    Build per-edge meshing controls for a polygon.

    edge_overrides keys are undirected edge tuples:
    ((x0, y0), (x1, y1))
    """
    if default_transition_distance is None:
        default_transition_distance = 4.0 * default_sampling_size

    edge_overrides = edge_overrides or {}
    controls = []
    n = len(vertices)

    for i in range(n):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % n]
        key = canonical_edge_key(p0, p1)
        override = edge_overrides.get(key, {})

        local_sampling_size = float(
            override.get("sampling_size", default_sampling_size)
        )
        transition_distance = override.get("transition_distance", None)

        if local_sampling_size < default_sampling_size and transition_distance is None:
            transition_distance = default_transition_distance

        controls.append(
            {
                "edge_index": i,
                "start_vertex": [float(p0[0]), float(p0[1])],
                "end_vertex": [float(p1[0]), float(p1[1])],
                "sampling_size": local_sampling_size,
                "transition_distance": (
                    None if transition_distance is None else float(transition_distance)
                ),
            }
        )

    return controls

def sample_segment(
    p0,
    p1,
    sampling_size,
    include_endpoint=False,
    jitter_ratio=0.0,
    seed=None,
):
    """
    Sample points on a straight segment from p0 to p1.

    - End vertices are kept fixed.
    - Number of subsegments is fixed by sampling_size.
    - Interior points are jittered ALONG THE EDGE only.
    - jitter_ratio is a fraction of the nominal spacing.
      Example: jitter_ratio=0.15 means each interior point can move
      by up to 15% of the nominal segment spacing along the edge.
    """
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)

    if length == 0:
        return [[float(x0), float(y0)]]

    nseg = max(1, int(round(length / sampling_size)))
    t_values = np.linspace(0.0, 1.0, nseg + 1)

    if jitter_ratio > 0.0 and nseg > 1:
        rng = np.random.default_rng(seed)

        dt = 1.0 / nseg
        t_inner = t_values[1:-1].copy()

        # jitter each interior point along the segment direction
        max_jitter = jitter_ratio * dt
        t_inner += rng.uniform(-max_jitter, max_jitter, size=len(t_inner))

        # keep strict ordering so points do not cross each other
        eps = 1e-10
        for i in range(len(t_inner)):
            lower = (i + 1) * eps if i == 0 else t_inner[i - 1] + eps
            upper = 1.0 - (len(t_inner) - i) * eps
            t_inner[i] = np.clip(t_inner[i], lower, upper)

        t_inner = np.sort(t_inner)
        t_values[1:-1] = t_inner

    t_values_use = t_values if include_endpoint else t_values[:-1]

    pts = []
    for t in t_values_use:
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        pts.append([float(x), float(y)])
    return pts

def polygon_to_contour_points(
    vertices,
    sampling_size,
    edge_controls=None,
    jitter_ratio=0.0,
    seed=None,
):
    """
    Build contour points for a polygon.

    - Vertices are preserved exactly.
    - Non-vertex points on each edge are jittered along that edge only.
    - Average spacing remains controlled by sampling_size.
    """
    if len(vertices) < 3:
        raise ValueError("A polygon needs at least 3 vertices.")

    contour = []
    edge_metadata = []
    n = len(vertices)
    rng = np.random.default_rng(seed)

    if edge_controls is None:
        edge_controls = build_polygon_edge_controls(
            vertices=vertices,
            default_sampling_size=sampling_size,
        )
    if len(edge_controls) != n:
        raise ValueError("edge_controls must have the same length as vertices.")

    for i in range(n):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % n]
        edge_control = edge_controls[i]
        local_sampling_size = float(edge_control.get("sampling_size", sampling_size))

        local_seed = None if seed is None else int(rng.integers(0, 10**9))

        edge_pts = sample_segment(
            p0,
            p1,
            local_sampling_size,
            include_endpoint=False,
            jitter_ratio=jitter_ratio,
            seed=local_seed,
        )
        start_point_index = len(contour)
        contour.extend(edge_pts)
        edge_metadata.append(
            {
                "edge_index": int(edge_control.get("edge_index", i)),
                "start_vertex": [float(p0[0]), float(p0[1])],
                "end_vertex": [float(p1[0]), float(p1[1])],
                "sampling_size": local_sampling_size,
                "transition_distance": edge_control.get("transition_distance", None),
                "start_point_index": start_point_index,
                "num_curve_segments": len(edge_pts),
            }
        )

    return contour, edge_metadata

def build_contour_from_polygon(
    vertices,
    sampling_size,
    outer_name="outer_0",
    edge_controls=None,
    default_transition_distance=None,
    jitter_ratio=0.0,
    seed=None,
):
    """
    Build JSON-style contour data for polygon examples.
    """
    contour, edge_metadata = polygon_to_contour_points(
        vertices=vertices,
        sampling_size=sampling_size,
        edge_controls=edge_controls,
        jitter_ratio=jitter_ratio,
        seed=seed,
    )
    points = build_points_from_vertices(vertices)

    data = {
        "outer": {
            outer_name: contour
        },
        "points": points,
        "mesh_controls": {
            "interior_size": float(sampling_size),
            "default_transition_distance": float(
                4.0 * sampling_size
                if default_transition_distance is None
                else default_transition_distance
            ),
            "edge_segments": {
                outer_name: edge_metadata
            },
        },
    }
    return data, points

def build_points_from_vertices(vertices):
    points = {}
    for i, v in enumerate(vertices):
        name = f"Point{chr(ord('A') + i)}"
        points[name] = [float(v[0]), float(v[1])]
    return points

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

def plot_contours_from_json(json_path, ax=None, outer_style=None, hole_style=None):
    if ax is None:
        fig, ax = plt.subplots()

    if outer_style is None:
        outer_style = {
            "color": "r",
            "linestyle": None,
            "marker": "o",
            "linewidth": 2.0,
            "markersize": 6,
        }

    if hole_style is None:
        hole_style = {
            "color": "b",
            "linestyle": None,
            "marker": "o",
            "linewidth": 2.0,
            "markersize": 6,
        }

    contours_outer, contours_holes = load_contours_from_json(json_path)

    for c in contours_outer:
        ax.plot(c[:, 0], c[:, 1], **outer_style)

    for c in contours_holes:
        ax.plot(c[:, 0], c[:, 1], **hole_style)

    ax.set_aspect("equal")
    return ax

def sample_circle(center, radius, sampling_size, jitter_ratio=0.0, seed=None):
    cx, cy = center
    circumference = 2.0 * math.pi * radius
    nseg = max(8, int(round(circumference / sampling_size)))

    # for a closed circle, use nseg angular samples in [0, 2pi)
    base = np.linspace(0.0, 1.0, nseg, endpoint=False)

    if jitter_ratio > 0.0 and nseg > 1:
        rng = np.random.default_rng(seed)
        dt = 1.0 / nseg
        jitter = jitter_ratio * dt
        t = base + rng.uniform(-jitter, jitter, size=nseg)
        t = np.mod(t, 1.0)
        t = np.sort(t)
    else:
        t = base

    theta = 2.0 * math.pi * t
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    return np.column_stack([x, y])

def angle_from_center(center, point):
    cx, cy = center
    x, y = point
    return math.atan2(y - cy, x - cx)

def sample_arc_counterclockwise(
    center,
    radius,
    point_start,
    point_end,
    sampling_size,
    jitter_ratio=0.0,
    seed=None,
):
    theta_start = angle_from_center(center, point_start) % (2.0 * math.pi)
    theta_end = angle_from_center(center, point_end) % (2.0 * math.pi)

    if theta_end <= theta_start:
        theta_end += 2.0 * math.pi

    dtheta = theta_end - theta_start
    arc_length = radius * dtheta
    nseg = max(3, int(round(arc_length / sampling_size)))

    t = jittered_parameter_values(nseg, jitter_ratio=jitter_ratio, seed=seed)
    theta = theta_start + dtheta * t

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.column_stack([x, y])

def sample_arc_clockwise(
    center,
    radius,
    point_start,
    point_end,
    sampling_size,
    jitter_ratio=0.0,
    seed=None,
):
    theta_start = angle_from_center(center, point_start) % (2.0 * math.pi)
    theta_end = angle_from_center(center, point_end) % (2.0 * math.pi)

    if theta_end >= theta_start:
        theta_end -= 2.0 * math.pi

    dtheta = theta_end - theta_start   # negative
    arc_length = radius * abs(dtheta)
    nseg = max(3, int(round(arc_length / sampling_size)))

    t = jittered_parameter_values(nseg, jitter_ratio=jitter_ratio, seed=seed)
    theta = theta_start + dtheta * t

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.column_stack([x, y])

def external_tangent_point(circle1_center, r1, circle2_center, r2, tol=1e-3):
    """
    Tangency point on circle 1 for two externally tangent circles.
    """
    c1 = np.asarray(circle1_center, dtype=float)
    c2 = np.asarray(circle2_center, dtype=float)

    dvec = c2 - c1
    d = np.linalg.norm(dvec)

    if abs(d - (r1 + r2)) > tol:
        raise ValueError(
            f"Circles are not externally tangent: d={d:.6f}, r1+r2={r1+r2:.6f}"
        )

    u = dvec / d
    p = c1 + r1 * u
    return tuple(p)

def internal_tangent_point(inner_center, inner_radius, outer_center, outer_radius, tol=1e-3):
    """
    Tangency point on the inner circle for two internally tangent circles.

    The outer circle contains the inner circle.
    The tangency point lies on the line from outer_center to inner_center,
    extended to the inner circle.
    """
    c_in = np.asarray(inner_center, dtype=float)
    c_out = np.asarray(outer_center, dtype=float)

    dvec = c_in - c_out
    d = np.linalg.norm(dvec)

    if abs(d - (outer_radius - inner_radius)) > tol:
        raise ValueError(
            f"Circles are not internally tangent: d={d:.6f}, r_out-r_in={outer_radius-inner_radius:.6f}"
        )

    u = dvec / d
    p = c_in + inner_radius * u
    return tuple(p)

def build_contour_hook(
    circles,
    radius,
    mesh_size,
    outer_name="outer_0",
    hole_name="hole_0",
    loading_ring_sampling_size=None,
    loading_ring_transition_distance=None,
    default_transition_distance=None,
    jitter_ratio=0.0,
    seed=None,
):
    points = {}
    if loading_ring_sampling_size is None:
        loading_ring_sampling_size = mesh_size
    if default_transition_distance is None:
        default_transition_distance = 4.0 * mesh_size

    rng = np.random.default_rng(seed)

    def local_seed():
        return None if seed is None else int(rng.integers(0, 10**9))

    hole_0 = sample_circle(
        center=circles["c4"],
        radius=radius["r1"],
        sampling_size=mesh_size,
        jitter_ratio=jitter_ratio,
        seed=local_seed(),
    )

    points["PointA"] = external_tangent_point(
        circle1_center=circles["c4"],
        r1=radius["r2"],
        circle2_center=circles["c2"],
        r2=radius["r3"],
    )

    points["PointB"] = internal_tangent_point(
        inner_center=circles["c4"],
        inner_radius=radius["r2"],
        outer_center=circles["c1"],
        outer_radius=radius["r4"],
    )

    arc_BA = sample_arc_clockwise(
        center=circles["c4"],
        radius=radius["r2"],
        point_start=points["PointB"],
        point_end=points["PointA"],
        sampling_size=mesh_size,
        jitter_ratio=jitter_ratio,
        seed=local_seed(),
    )

    points["PointC"] = external_tangent_point(
        circle1_center=circles["c2"],
        r1=radius["r3"],
        circle2_center=circles["c5"],
        r2=radius["r5"],
    )

    arc_AC = sample_arc_counterclockwise(
        center=circles["c2"],
        radius=radius["r3"],
        point_start=points["PointA"],
        point_end=points["PointC"],
        sampling_size=mesh_size,
        jitter_ratio=jitter_ratio,
        seed=local_seed(),
    )

    points["PointD"] = internal_tangent_point(
        inner_center=circles["c6"],
        inner_radius=radius["r6"],
        outer_center=circles["c5"],
        outer_radius=radius["r5"],
    )

    arc_CD = sample_arc_clockwise(
        center=circles["c5"],
        radius=radius["r5"],
        point_start=points["PointC"],
        point_end=points["PointD"],
        sampling_size=mesh_size,
        jitter_ratio=jitter_ratio,
        seed=local_seed(),
    )

    points["PointE"] = (-19.0, 0.0)

    arc_DE = sample_arc_clockwise(
        center=circles["c6"],
        radius=radius["r6"],
        point_start=points["PointD"],
        point_end=points["PointE"],
        sampling_size=mesh_size,
        jitter_ratio=jitter_ratio,
        seed=local_seed(),
    )

    points["PointF"] = external_tangent_point(
        circle1_center=circles["c7"],
        r1=radius["r7"],
        circle2_center=circles["c1"],
        r2=radius["r4"],
    )

    points["PointG"] = (19.0, 0.0)

    arc_EG = sample_arc_counterclockwise(
        center=circles["c7"],
        radius=radius["r7"],
        point_start=points["PointE"],
        point_end=points["PointG"],
        sampling_size=loading_ring_sampling_size,
        jitter_ratio=jitter_ratio,
        seed=local_seed(),
    )

    arc_GF = sample_arc_counterclockwise(
        center=circles["c7"],
        radius=radius["r7"],
        point_start=points["PointG"],
        point_end=points["PointF"],
        sampling_size=mesh_size,
        jitter_ratio=jitter_ratio,
        seed=local_seed(),
    )

    arc_FB = sample_arc_clockwise(
        center=circles["c1"],
        radius=radius["r4"],
        point_start=points["PointF"],
        point_end=points["PointB"],
        sampling_size=mesh_size,
        jitter_ratio=jitter_ratio,
        seed=local_seed(),
    )

    outer_segments = [
        ("arc_BA", arc_BA[:-1], mesh_size, None),
        ("arc_AC", arc_AC[:-1], mesh_size, None),
        ("arc_CD", arc_CD[:-1], mesh_size, None),
        ("arc_DE", arc_DE[:-1], mesh_size, None),
        (
            "arc_EG",
            arc_EG[:-1],
            loading_ring_sampling_size,
            loading_ring_transition_distance,
        ),
        ("arc_GF", arc_GF[:-1], mesh_size, None),
        ("arc_FB", arc_FB[:-1], mesh_size, None),
    ]
    outer_0 = np.vstack([segment_points for _, segment_points, _, _ in outer_segments])

    points_json = {key: [float(val[0]), float(val[1])] for key, val in points.items()}
    edge_metadata = []
    start_point_index = 0
    for edge_index, (_, segment_points, local_sampling_size, transition_distance) in enumerate(outer_segments):
        num_curve_segments = len(segment_points)
        edge_metadata.append(
            {
                "edge_index": edge_index,
                "sampling_size": float(local_sampling_size),
                "transition_distance": (
                    None if transition_distance is None else float(transition_distance)
                ),
                "start_point_index": start_point_index,
                "num_curve_segments": num_curve_segments,
            }
        )
        start_point_index += num_curve_segments

    data = {
        "outer": {
            outer_name: outer_0.tolist()
        },
        "holes": {
            hole_name: hole_0.tolist()
        },
        "points": points_json,
        "mesh_controls": {
            "interior_size": float(mesh_size),
            "default_transition_distance": float(default_transition_distance),
            "edge_segments": {
                outer_name: edge_metadata
            },
        },
    }

    return data, points

def jittered_parameter_values(nseg, jitter_ratio=0.0, seed=None):
    """
    Create parameter values from 0 to 1 with fixed endpoints and
    small jitter on interior points only.

    nseg: number of subsegments
    returns: array of length nseg+1 from 0 to 1
    """
    t_values = np.linspace(0.0, 1.0, nseg + 1)

    if jitter_ratio > 0.0 and nseg > 1:
        rng = np.random.default_rng(seed)
        dt = 1.0 / nseg
        t_inner = t_values[1:-1].copy()

        max_jitter = jitter_ratio * dt
        t_inner += rng.uniform(-max_jitter, max_jitter, size=len(t_inner))

        # keep ordering
        eps = 1e-10
        for i in range(len(t_inner)):
            lower = (i + 1) * eps if i == 0 else t_inner[i - 1] + eps
            upper = 1.0 - (len(t_inner) - i) * eps
            t_inner[i] = np.clip(t_inner[i], lower, upper)

        t_inner = np.sort(t_inner)
        t_values[1:-1] = t_inner

    return t_values

num_contours = 1
example = "Portal_frame_half"   # L_beam  Crack 1.6 Portal_frame_half Hook
jitter_ratio = 0.0
if example == 'L_beam':
    outer_name = "outer_0"
    vertices = [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 35.0),
        (100.0, 40.0),
        # (45.0, 40.0),
        (40.0, 40.0),
        # (40.0, 45.0),
        (40.0, 100.0),
        (0.0, 100.0),
    ]
    sampling_size = 0.1885
    edge_controls = build_polygon_edge_controls(
        vertices=vertices,
        default_sampling_size=sampling_size,
        edge_overrides={
            canonical_edge_key((100.0, 40.0), (40.0, 40.0)): {
                "sampling_size": sampling_size,
                "transition_distance": 15,
            },
            canonical_edge_key((40.0, 40.0), (40.0, 100.0)): {
                "sampling_size": sampling_size,
                "transition_distance": 15,
            },
        },
        default_transition_distance=15
    )
elif example == 'Crack':
    outer_name = "outer_0"
    vertices = [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 190.0),
        (100.0, 200.0),
        (0.0, 200.0),
        (0.0, 105.0),
        (0.0, 100.0),
        (0.0, 95.0),
    ]
    sampling_size = 1.5
    edge_controls = build_polygon_edge_controls(
        vertices=vertices,
        default_sampling_size=sampling_size,
        edge_overrides={
            canonical_edge_key((0.0, 105.0), (0.0, 100.0)): {
                "sampling_size": 0.1,
                "transition_distance": 15,
            },
            canonical_edge_key((0.0, 100.0), (0.0, 95.0)): {
                "sampling_size": 0.1,
                "transition_distance": 15,
            },
        },
        default_transition_distance=15
    )
elif example == 'Portal_frame_half':
    outer_name = "outer_0"
    sampling_size = 0.3
    vertices = [
        (0.0, 35.0),
        (54.5, 0.0),
        (60.0, 0.0),
        (60.0, 60.0),
        (5.0, 60.0),
        (0.0, 60.0),
    ]
    edge_controls = build_polygon_edge_controls(
        vertices=vertices,
        default_sampling_size=sampling_size,
        edge_overrides={
            canonical_edge_key((60.0, 0.0), (60.0, 60.0)): {
                "sampling_size": sampling_size,
                "transition_distance": 15,
            },
            canonical_edge_key((54.5, 0.0), (60.0, 0.0)): {
                "sampling_size": sampling_size,
                "transition_distance": 15,
            },
        },
        default_transition_distance=15
    )
elif example == 'Portal_frame':
    outer_name = "outer_0"
    sampling_size = 0.85
    vertices = [
        (-5.0, 31.788991),
        (0.0, 35.0),
        (5.0, 31.788991),
        (54.5, 0.0),
        (60.0, 0.0),
        (60.0, 60.0),
        (5.0, 60.0),
        (-5.0, 60.0),
        (-60.0, 60.0),
        (-60.0, 0.0),
        (-54.5, 0.0),
    ]
    edge_controls = build_polygon_edge_controls(
        vertices=vertices,
        default_sampling_size=sampling_size,
        edge_overrides={
            canonical_edge_key((-5.0, 31.788991), (0.0, 35.0)): {
                "sampling_size": 0.1,
                "transition_distance": 15,
            },
            canonical_edge_key((0.0, 35.0), (5.0, 31.788991)): {
                "sampling_size": 0.1,
                "transition_distance": 15,
            },
        },
        default_transition_distance=15
    )
elif example == 'Hook':
    outer_name = "outer_0"
    hole_name = 'hole_0'
    sampling_size = 0.85
    loading_ring_sampling_size = 0.84
    loading_ring_transition_distance = 15
    circles = {
        "c1": (59.9713, 78.7683),
        "c2": (54.8716, 76.8672),
        "c4": (0.0, 80.6226),
        "c5": (14.8842, 1.8605),
        "c6": (-27.037267626740196604, 0),
        "c7": (0.0, 0.0),
    }
    radius = {
        "r1": 10,
        "r2": 20,
        "r3": 35,
        "r4": 80,
        "r5": 50,
        "r6": 8.0372676267401966043,
        "r7": 19,
    }
else:
    raise ValueError(f"Unknown example: {example}")

folder_name = f"{example}_sampling_size{sampling_size}"
os.makedirs(folder_name, exist_ok=True)

for i in range(num_contours):
    seed = 123 + i

    if example == 'Hook':
        data, points = build_contour_hook(
            circles=circles,
            radius=radius,
            mesh_size=sampling_size,
            outer_name=outer_name,
            hole_name=hole_name,
            loading_ring_sampling_size=loading_ring_sampling_size,
            loading_ring_transition_distance=loading_ring_transition_distance,
            default_transition_distance=6.0 * sampling_size,
            jitter_ratio=jitter_ratio,
            seed=seed,
        )
    else:
        data, points = build_contour_from_polygon(
            vertices=vertices,
            sampling_size=sampling_size,
            outer_name=outer_name,
            edge_controls=edge_controls,
            default_transition_distance=6.0 * sampling_size,
            jitter_ratio=jitter_ratio,
            seed=seed,
        )

    filename = os.path.join(folder_name, f"contour_{i:02d}.json")
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved contour JSON to: {filename}")
    print(f"Number of contour points: {len(data['outer'][outer_name])}")

    figname = os.path.join(folder_name, f"contour_{i:02d}.pdf")
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_contours_from_json(filename, ax=ax)
    ax.set_title(f"contours {i:02d}")
    plt.savefig(figname, format='pdf', dpi=600)
    plt.close(fig)


