"""Band-limited optical line profiles with real stations and observable gaps."""

from dataclasses import dataclass

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .contracts import DeskError, enum, integer, number, obj, polygon
from .geometry import analysis_box, normalized_sampling
from .glasscolor import rgb_to_lab
from .imaging import canvas_picture, luminance, percentile
from .visuals import AQUA, GOLD, ROSE, legend_band, series_chart


@dataclass(frozen=True)
class ProfileSettings:
    points: list
    width: int
    step: float
    min_coverage: float
    measure: str
    method: str
    threshold: float | None

    @classmethod
    def parse(cls, raw):
        wanted = obj(raw, "optical profile")
        threshold = wanted.get("threshold")
        return cls(
            polygon(wanted.get("points"), "profile line", minimum=2),
            integer(wanted.get("half_width"), "station half width", 0, 40, default=4),
            number(wanted.get("step"), "station spacing", .5, 35, default=5),
            number(wanted.get("minimum_coverage"), "minimum station support", .25, 1, default=.70),
            enum(wanted.get("measure"), "channel", ("luminance", "red", "green", "blue", "lightness"), default="luminance"),
            enum(wanted.get("method"), "aggregation", ("median", "mean"), default="median"),
            number(threshold, "crossing threshold", -1, 110) if threshold is not None else None,
        )

    def json(self):
        return {
            "points": self.points, "half_width": self.width, "step": self.step,
            "minimum_coverage": self.min_coverage, "measure": self.measure, "method": self.method,
            "threshold": self.threshold,
        }


def working_signal(values, name):
    if name == "luminance":
        return luminance(values)
    if name == "lightness":
        return rgb_to_lab(values)[..., 0]
    return values[..., {"red": 0, "green": 1, "blue": 2}[name]]


def profile_band(centers, width):
    if width == 0:
        return centers[:, None, :]
    direction = np.gradient(centers, axis=0)
    distance = np.hypot(direction[:, 0], direction[:, 1])
    if np.any(distance < .0001):
        raise DeskError("Separate repeated profile vertices so each band has a direction.")
    normal = np.column_stack((direction[:, 1], -direction[:, 0])) / distance[:, None]
    return centers[:, None] + normal[:, None] * np.arange(-width, width + 1)[None, :, None]


def profile_values(ctx, view, settings):
    centers, position = normalized_sampling(settings.points, ctx.surface.extent, settings.step, 2200)
    x, y = np.moveaxis(profile_band(centers, settings.width), -1, 0)
    samples = ndimage.map_coordinates(view, (y, x), order=1, mode="constant")
    valid = np.asarray(ctx.usable & ~ctx.surface.saturated, dtype=np.float32)
    mask = ndimage.map_coordinates(valid, (y, x), order=1, mode="constant") >= .96
    return centers, position, samples, mask


def metric_length(ctx, stations, pixel_position):
    if ctx.alignment and ctx.scale is not None:
        difference = np.diff(stations, axis=0) * ctx.scale
        segment = np.linalg.norm(difference, axis=1)
        return np.cumsum(np.r_[0, segment]), "mm"
    return pixel_position, "px"


def summarise_position(index, long_axis, pixel_axis, positions, sample_values, support, cfg):
    count = int(np.count_nonzero(support))
    accepted = sample_values[support]
    observed = count >= np.ceil(len(support) * cfg.min_coverage)
    median = None
    if observed:
        reducer = np.median if cfg.method == "median" else np.mean
        median = round(float(reducer(accepted)), 6)
    return {
        "index": index + 1,
        "position": round(float(long_axis), 5),
        "position_px": round(float(pixel_axis), 5),
        "plane_xy": np.round(positions, 5).tolist(),
        "available": count,
        "requested": int(len(support)),
        "value": median,
        "range_p05_p50_p95": percentile(accepted) if observed else None,
    }


def interval_summary(stations):
    first = 0
    spans = []
    has_value = stations[0]["value"] is not None
    for i, row in enumerate(stations[1:], 1):
        present = row["value"] is not None
        if present != has_value:
            spans.append({"from": stations[first]["position"], "to": stations[i - 1]["position"],
                          "measured": has_value, "stations": i - first})
            first = i
            has_value = present
    last = len(stations) - 1
    spans.append({"from": stations[first]["position"], "to": stations[last]["position"],
                  "measured": has_value, "stations": last - first + 1})
    return spans


def threshold_crossings(stations, threshold):
    output = []
    if threshold is None:
        return output
    for first, second in zip(stations[:-1], stations[1:]):
        a, b = first["value"], second["value"]
        if a is None or b is None or (a < threshold) == (b < threshold):
            continue
        fraction = (threshold - a) / (b - a)
        step = second["position"] - first["position"]
        point = first["position"] + fraction * step
        output.append({
            "position": round(point, 5), "increases": b > a,
            "station_pair": [first["index"], second["index"]],
        })
    return output


def paint_profile(ctx, coordinates, rows):
    image = canvas_picture(ctx.surface)
    draw = ImageDraw.Draw(image)
    if len(coordinates) > 1:
        outline = [tuple(p) for p in coordinates]
        draw.line(outline, fill=GOLD, width=max(1, image.width // 600))
    step = max(1, len(rows) // 120)
    for row, xyz in zip(rows[::step], coordinates[::step], strict=True):
        x, y = xyz
        color = AQUA if row["value"] is not None else ROSE
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    return legend_band(image, "Measured optical stations", [
        ("Readable station", AQUA), ("No usable station", ROSE), ("Profile trace", GOLD),
    ])


def line_profile(ctx, raw):
    settings = ProfileSettings.parse(raw)
    signal = working_signal(ctx.surface.linear, settings.measure)
    centers, positions, collected, sample_ok = profile_values(ctx, signal, settings)
    span, units = metric_length(ctx, centers, positions)
    readings = [
        summarise_position(i, place, positions[i], center, collected[i], sample_ok[i], settings)
        for i, (place, center) in enumerate(zip(span, centers, strict=True))
    ]
    pixels = [r["value"] for r in readings if r["value"] is not None]
    if not pixels:
        raise DeskError("No station has enough usable measurements in this band.")
    crossings = threshold_crossings(readings, settings.threshold)
    row = [(item["position"], item["value"]) for item in readings]
    series = [{"color": "#286259", "samples": row}]
    if settings.threshold is not None:
        series.append({
            "color": "#b66258",
            "samples": [(0.0, settings.threshold), (float(span[-1]), settings.threshold)],
        })
    chart = series_chart(series, "Profile through the pane", units)
    picture = paint_profile(ctx, centers, readings)
    stats = {
        "settings": settings.json(), "units": units, "length": round(float(span[-1]), 6),
        "stations": readings, "intervals": interval_summary(readings), "threshold_crossings": crossings,
        "observed_points": len(pixels), "missing_points": len(readings) - len(pixels),
        "distribution": percentile(np.array(pixels)),
        "note": "Crossings are interpolated only between adjacent measured stations. Gaps are visible in the map and chart.",
    }
    return stats, {"map": picture, "chart": chart}
