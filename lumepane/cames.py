"""Sample came strips around a drawn run without assuming hidden continuity."""

from dataclasses import dataclass

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .contracts import DeskError, enum, identifiers, integer, number, obj
from .geometry import normalized_sampling, scale_for
from .imaging import canvas_picture, luminance, percentile
from .visuals import AQUA, GOLD, ROSE, legend_band, series_chart


@dataclass(frozen=True)
class StripSettings:
    half: float
    interval: float
    offset: float
    tolerance: float

    @classmethod
    def parse(cls, payload):
        data = obj(payload)
        return cls(
            number(data.get("half_width"), "strip half width", 2, 35, default=12),
            number(data.get("step"), "sample step", 1, 35, default=5),
            number(data.get("reference_gap"), "side reference gap", 0, 35, default=3),
            number(data.get("minimum_shadow"), "minimum contrast loss", .02, .7, default=.12),
        )

    def json(self):
        return {
            "half_width": self.half, "step": self.interval,
            "reference_gap": self.offset, "minimum_shadow": self.tolerance,
        }


def came_lines(ctx, request):
    marks = [mark for mark in ctx.marks if mark["category"] == "came" and mark["shape"] == "line"]
    supplied = request.get("marks")
    if supplied:
        accepted = set(identifiers(supplied, "came runs", high=20))
        marks = [item for item in marks if item["id"] in accepted]
        if len(marks) != len(accepted):
            raise DeskError("A selected came run is unavailable.")
    if not marks:
        raise DeskError("Draw an inspected came line before using this review.")
    if len(marks) > 20:
        raise DeskError("Choose at most 20 came lines.")
    return marks


def strip_lattice(points, interval, radius, extent):
    positions, length = normalized_sampling(points, extent, interval, cap=1700)
    tangent = np.gradient(positions, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    if np.any(norm < 1e-6):
        raise DeskError("Space the came trace points so they do not double back onto themselves.")
    tangent /= norm[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    offsets = np.arange(-radius, radius + 1)
    samples = positions[:, None, :] + normal[:, None, :] * offsets[None, :, None]
    return length, positions, normal, offsets, samples


def sample_strip(ctx, points, settings):
    radius = int(np.ceil(settings.half + settings.offset + 7))
    length, positions, normal, offsets, lattice = strip_lattice(points, settings.interval, radius, ctx.surface.extent)
    x, y = lattice[..., 0], lattice[..., 1]
    positions_rc = (y, x)
    data = luminance(ctx.surface.linear)
    safe = ctx.usable & ~ctx.surface.saturated
    present = ndimage.map_coordinates(safe.astype(np.float32), positions_rc, order=1, mode="constant") >= .97
    values = ndimage.map_coordinates(data, positions_rc, order=1, mode="constant")
    fits = [describe_station(s, v, offsets, settings) for s, v in zip(present, values, strict=True)]
    return length, positions, normal, fits


def describe_station(present, values, offsets, settings):
    half = settings.half
    edge_from = half + settings.offset
    outer_left = (offsets < -edge_from) & present
    outer_right = (offsets > edge_from) & present
    inside = np.abs(offsets) <= half
    strip = inside & present
    if np.count_nonzero(strip) < max(3, np.count_nonzero(inside) * .58):
        return {"class": "obscured", "dark_width_px": None, "contrast": None}
    if min(np.count_nonzero(outer_left), np.count_nonzero(outer_right)) < 3:
        return {"class": "unmeasured boundary", "dark_width_px": None, "contrast": None}
    baseline = min(np.median(values[outer_left]), np.median(values[outer_right]))
    middle = float(np.quantile(values[strip], .24))
    contrast = float((baseline - middle) / max(baseline, .04))
    center = int(np.argmin(np.abs(offsets)))
    darker = strip & (values <= baseline * (1 - settings.tolerance))
    if contrast < settings.tolerance or not darker[center]:
        return {"class": "weak or interrupted", "dark_width_px": None, "contrast": round(contrast, 5)}
    start = stop = center
    while start and darker[start - 1]:
        start -= 1
    while stop + 1 < len(darker) and darker[stop + 1]:
        stop += 1
    uncertain = not present[max(0, start - 1)] or not present[min(len(present) - 1, stop + 1)]
    if uncertain:
        return {"class": "width meets occlusion", "dark_width_px": None, "contrast": round(contrast, 5)}
    return {"class": "visible came", "dark_width_px": float(stop - start + 1), "contrast": round(contrast, 5)}


def physical_distances(centers, scale, px):
    if scale is None:
        return px, "px"
    physical = np.diff(centers, axis=0) * scale
    position = np.r_[0, np.cumsum(np.linalg.norm(physical, axis=1))]
    return position, "mm"


def condition_groups(stations):
    if not stations:
        return []
    pieces = []
    current = stations[0]
    first = current["distance"]
    for row in stations[1:]:
        if current["class"] != row["class"]:
            pieces.append({
                "start": first, "stop": current["distance"],
                "condition": current["class"],
            })
            first = row["distance"]
        current = row
    pieces.append({"start": first, "stop": current["distance"], "condition": current["class"]})
    return pieces


def run_readout(ctx, record, settings):
    length, positions, normals, measurements = sample_strip(ctx, record["points"], settings)
    scale = ctx.scale if ctx.alignment else None
    physical, unit = physical_distances(positions, scale, length)
    rows = []
    read_width = []
    for i, row in enumerate(measurements):
        width_px = row.get("dark_width_px")
        width_mm = None
        if width_px is not None:
            width_mm = float(np.linalg.norm(normals[i] * scale) * width_px) if scale is not None else None
            read_width.append(width_px if width_mm is None else width_mm)
        rows.append(row | {
            "station": i + 1, "distance": round(float(physical[i]), 5), "distance_px": round(float(length[i]), 5),
            "dark_width_mm": round(width_mm, 5) if width_mm is not None else None,
            "normalized_point": [round(float(c / ctx.surface.extent[a]), 6) for a, c in enumerate(positions[i])],
        })
    return {
        "title": record["title"], "mark": record["id"], "unit": unit, "stations": rows,
        "groups": condition_groups(rows),
        "width_p05_p50_p95": percentile(np.asarray(read_width)),
        "trace_length": round(float(physical[-1]), 5),
        "unmeasured_stations": len(rows) - len(read_width),
    }


def strip_decorations(ctx, rows):
    image = canvas_picture(ctx.surface)
    pen = ImageDraw.Draw(image)
    width, height = image.size
    colors = {
        "visible came": AQUA, "weak or interrupted": ROSE,
        "obscured": "#a7a6bd", "unmeasured boundary": "#a7a6bd", "width meets occlusion": GOLD,
    }
    for entry in rows:
        positions = entry["stations"]
        for row in positions:
            a, b = row["normalized_point"]
            x, y = a * width, b * height
            radius = max(2, image.width // 370)
            pen.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colors[row["class"]])
    return image


def came_report(ctx, raw):
    settings = StripSettings.parse(raw)
    marks = came_lines(ctx, raw)
    results = [run_readout(ctx, mark, settings) for mark in marks]
    image = strip_decorations(ctx, results)
    art = legend_band(image, "Came strip examination", [
        ("Measured", AQUA), ("Weak or interrupted", ROSE),
        ("Occluded or out of view", (167, 166, 189)), ("Width touches an occlusion", GOLD),
    ], lines=3)
    first = results[0]
    use_mm = first["unit"] == "mm"
    key = "dark_width_mm" if use_mm else "dark_width_px"
    readings = [(r["distance"], r[key]) for r in first["stations"]]
    chart = series_chart([{"samples": readings, "color": "#277579"}], f"Width along {first['title']}", first["unit"])
    return {
        "settings": settings.json(), "came_runs": results,
        "measured_runs": len(results),
        "note": "Only the visibly sampled came strip is measured; gaps remain unverified.",
    }, {"map": art, "chart": chart}
