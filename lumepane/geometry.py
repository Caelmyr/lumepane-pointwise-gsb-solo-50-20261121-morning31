"""Outline validation and geometric projection to the flat window plane."""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .contracts import DeskError, MAX_WORK_EDGE, collection, number, point, polygon, region_box
from .imaging import Surface

CORNERS = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
RING_LIMIT = 10
PLANE_MIN_EDGE = 48
WARP_BLOCK = 112
GEOMETRIC_EPS = 1e-9


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def span_contains(a, x, b):
    return min(a, b) - GEOMETRIC_EPS <= x <= max(a, b) + GEOMETRIC_EPS


def on_edge(a, b, point):
    return abs(cross(a, b, point)) <= GEOMETRIC_EPS and all(
        span_contains(a[i], point[i], b[i]) for i in (0, 1)
    )


def clash(a, b, c, d):
    ab_c, ab_d, cd_a, cd_b = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    if ab_c * ab_d < -GEOMETRIC_EPS**2 and cd_a * cd_b < -GEOMETRIC_EPS**2:
        return True
    return on_edge(a, b, c) or on_edge(a, b, d) or on_edge(c, d, a) or on_edge(c, d, b)


def loop(ring):
    return list(zip(ring, ring[1:] + ring[:1]))


def simple_ring(ring):
    if len(set(map(tuple, ring))) != len(ring):
        return False
    sides = loop(ring)
    for i, (start, stop) in enumerate(sides):
        for j, (other_start, other_stop) in enumerate(sides[i + 1:], i + 1):
            if j - i in (1, len(sides) - 1):
                continue
            if clash(start, stop, other_start, other_stop):
                return False
    return abs(signed_area(ring)) > 0.000003


def signed_area(points):
    return sum(a[0] * b[1] - a[1] * b[0] for a, b in loop(list(points))) / 2


def inside_outline(position, ring):
    x, y = position
    inside = False
    for a, b in loop(ring):
        if on_edge(a, b, position):
            return True
        if (a[1] > y) != (b[1] > y):
            at_x = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < at_x:
                inside = not inside
    return inside


def nonintersecting(left, right):
    return not any(clash(a, b, c, d) for a, b in loop(left) for c, d in loop(right))


def outline_geometry(data):
    outer = polygon(data.get("points"), "outline")
    if not simple_ring(outer):
        raise DeskError("Use a simple polygon with a visible area.", field="points")
    holes = []
    for raw in collection(data.get("holes", []), "holes", 0, RING_LIMIT):
        hole = polygon(raw, "hole")
        if not simple_ring(hole):
            raise DeskError("One hole overlaps itself or is too small.", field="holes")
        if not all(inside_outline(v, outer) for v in hole) or not nonintersecting(outer, hole):
            raise DeskError("A hole must lie completely within the outer edge.", field="holes")
        for present in holes:
            if (not nonintersecting(present, hole)
                    or inside_outline(present[0], hole) or inside_outline(hole[0], present)):
                raise DeskError("Keep holes separate from each other.", field="holes")
        holes.append(hole)
    return outer, holes


def polygon_mask(extent, outer, holes=()):
    width, height = extent
    target = Image.new("1", extent)
    draw = ImageDraw.Draw(target)
    transformed = [(x * width - .5, y * height - .5) for x, y in outer]
    draw.polygon(transformed, fill=1)
    for ring in holes:
        transformed = [(x * width - .5, y * height - .5) for x, y in ring]
        draw.polygon(transformed, fill=0)
    return np.array(target, dtype=bool)


def mask_for_marks(extent, marks, choices):
    mask = np.zeros(extent[::-1], dtype=bool)
    for record in marks:
        if record["shape"] == "polygon" and record["category"] in choices:
            mask |= polygon_mask(extent, record["points"], record["holes"])
    return mask


def bounds_px(box, size):
    w, h = size
    if not box:
        return 0, 0, w, h
    return (
        max(0, min(w - 1, math.floor(box[0] * w))),
        max(0, min(h - 1, math.floor(box[1] * h))),
        min(w, math.ceil(box[2] * w)),
        min(h, math.ceil(box[3] * h)),
    )


def domain_mask(extent, box):
    x0, y0, x1, y1 = bounds_px(box, extent)
    mask = np.zeros(extent[::-1], dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def output_plane(window, hint):
    aspect = window["physical_mm"][0] / window["physical_mm"][1] if window["physical_mm"] else None
    defaults = window.get("working_extent") or hint
    width, height = defaults
    if aspect:
        if width / height > aspect:
            width = round(height * aspect)
        else:
            height = round(width / aspect)
    if max(width, height) > MAX_WORK_EDGE:
        ratio = MAX_WORK_EDGE / max(width, height)
        width, height = round(width * ratio), round(height * ratio)
    if min(width, height) < PLANE_MIN_EDGE:
        raise DeskError("The flat working plane is too narrow for a stable examination.")
    return int(width), int(height)


def normalized_quad(data):
    coords = [point(entry, "corner", low=-1.4, high=2.4) for entry in collection(data, "corners", 4, 4)]
    cyclic = [cross(coords[i - 2], coords[i - 1], coords[i]) for i in range(4)]
    if min(cyclic) <= 1e-5 or signed_area(coords) <= 0.008:
        raise DeskError("Mark the four corners clockwise from the upper left.", field="corners")
    return coords


def stable_system(coords):
    center = np.mean(coords, axis=0)
    mean_radius = np.mean(np.linalg.norm(coords - center, axis=1))
    if mean_radius < 1e-6:
        raise DeskError("Spread out the landmarks.")
    factor = math.sqrt(2) / mean_radius
    kernel = np.array([[factor, 0., -center[0] * factor], [0., factor, -center[1] * factor], [0., 0., 1.]])
    normalized = (coords - center) * factor
    return normalized, kernel


def fit_transform(domain, capture):
    domain = np.asarray(domain, dtype=np.float64)
    capture = np.asarray(capture, dtype=np.float64)
    source, ns = stable_system(domain)
    target, nt = stable_system(capture)
    problem = []
    for (u, v), (x, y) in zip(source, target, strict=True):
        problem.append([u, v, 1., 0., 0., 0., -x * u, -x * v, -x])
        problem.append([0., 0., 0., u, v, 1., -y * u, -y * v, -y])
    _, singular, right = np.linalg.svd(np.asarray(problem))
    if singular[7] < 1e-7 * singular[0]:
        raise DeskError("The position marks cannot resolve a plane.")
    matrix = np.linalg.solve(nt, right[-1].reshape(3, 3) @ ns)
    if abs(matrix[2, 2]) < GEOMETRIC_EPS:
        raise DeskError("The plane would cross the photo horizon.")
    matrix /= matrix[2, 2]
    if abs(np.linalg.det(matrix)) < 1e-9:
        raise DeskError("The selected plane would collapse.")
    denominators = np.column_stack((np.array(CORNERS), np.ones(4))) @ matrix[2]
    if np.min(denominators) <= 0.03:
        raise DeskError("The mapped photo would cross or approach its projection horizon.")
    return matrix


def transform_points(matrix, points):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    product = homogeneous @ matrix.T
    numerators = product[:, :2]
    denominators = product[:, 2:]
    if np.any(np.abs(denominators) < GEOMETRIC_EPS):
        raise DeskError("A marked position is outside the reachable plane.")
    return numerators / denominators


def native_crop_registration(corners, frame, plane):
    matrix = fit_transform(CORNERS, corners)
    inverse = np.linalg.inv(matrix)
    covered = transform_points(inverse, CORNERS)
    return {
        "matrix": matrix.tolist(),
        "capture_corners": corners,
        "photo_extent": list(frame["extent"]),
        "plane_extent": list(plane),
        "photo_footprint": covered.round(7).tolist(),
    }


@dataclass(frozen=True)
class WarpSample:
    colors: np.ndarray
    accepted: np.ndarray
    bright: np.ndarray
    dark: np.ndarray


def resample_lattice(surface, source_x, source_y):
    width, height = surface.extent
    in_frame = ((source_x >= -.5) & (source_y >= -.5)
                & (source_x <= width - .5) & (source_y <= height - .5))
    coords = (source_y, source_x)
    weight = ndimage.map_coordinates(surface.valid.astype(np.float32), coords, order=1, mode="nearest")
    accepted = in_frame & (weight >= 0.96)
    values = np.zeros((*source_x.shape, 3), dtype=np.float32)
    for channel in range(3):
        input_values = surface.linear[..., channel]
        estimated = ndimage.map_coordinates(input_values, coords, order=1, mode="nearest")
        np.divide(estimated, weight, out=values[..., channel], where=accepted)
    high = ndimage.map_coordinates(surface.saturated.astype(np.float32), coords, order=1, mode="nearest")
    low = ndimage.map_coordinates(surface.dark.astype(np.float32), coords, order=1, mode="nearest")
    return WarpSample(values, accepted, accepted & (high > .05), accepted & (low > .75))


def warp_surface(surface, matrix, plane):
    width, height = plane
    source_width, source_height = surface.extent
    output = np.zeros((height, width, 3), dtype=np.float32)
    accepted = np.zeros((height, width), dtype=bool)
    bright = accepted.copy()
    dark = accepted.copy()
    matrix = np.asarray(matrix)
    x = (np.arange(width)[None, :] + .5) / width
    for start in range(0, height, WARP_BLOCK):
        stop = min(height, start + WARP_BLOCK)
        y = ((np.arange(start, stop) + .5) / height)[:, None]
        denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
        sx = ((matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator) * source_width - .5
        sy = ((matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator) * source_height - .5
        grid = resample_lattice(surface, sx, sy)
        output[start:stop] = grid.colors
        accepted[start:stop] = grid.accepted
        bright[start:stop] = grid.bright
        dark[start:stop] = grid.dark
    return Surface(surface.frame, output, accepted, bright, dark)


def normalized_points_to_px(points, extent):
    return np.array(points, dtype=float) * np.asarray(extent) - .5


def arc_length(points, scale=None):
    steps = np.diff(np.array(points, dtype=np.float64), axis=0)
    if scale is not None:
        steps *= scale
    return float(np.linalg.norm(steps, axis=1).sum())


def mark_line(data):
    return polygon(data.get("points"), "trace", minimum=2)


def analysis_box(data):
    return region_box(data.get("box"), "analysis box")


def scale_for(window, extent):
    if not window["physical_mm"]:
        return None
    return np.asarray(window["physical_mm"], dtype=np.float64) / extent


def support_grid(mask, cell=45):
    width, height = mask.shape[::-1]
    squares = []
    for top in range(0, height, cell):
        for left in range(0, width, cell):
            bottom, right = min(top + cell, height), min(left + cell, width)
            patch = mask[top:bottom, left:right]
            squares.append({
                "bounds": [left, top, right, bottom],
                "coverage": round(float(patch.mean()), 4),
            })
    return squares


def normalized_sampling(points, extent, interval, cap):
    positions = normalized_points_to_px(points, extent)
    length = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(length)]
    distance = float(cumulative[-1])
    if distance < 2:
        raise DeskError("Use a trace at least two pixels long.")
    count = max(2, int(math.ceil(distance / interval)) + 1)
    if count > cap:
        raise DeskError("This trace requires too many stations at the chosen interval.")
    sample = np.linspace(0, distance, count)
    x = np.interp(sample, cumulative, positions[:, 0])
    y = np.interp(sample, cumulative, positions[:, 1])
    return np.column_stack((x, y)), sample


class AngleWindow:
    def __init__(self, value, maximum_error):
        self.angle = number(value, "angle", -180, 180)
        self.radius = number(maximum_error, "angular radius", 0.5, 90)

    def contains(self, values):
        residual = (values - self.angle + 90) % 180 - 90
        return np.abs(residual) <= self.radius
