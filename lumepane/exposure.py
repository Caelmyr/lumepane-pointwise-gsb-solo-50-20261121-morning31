"""Exposure and detail qualification over photographed glass."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .contracts import DeskError, enum, integer, number, obj, support_limit
from .geometry import analysis_box, bounds_px, domain_mask
from .imaging import gradient, grid_bounds, label, luminance, percentile
from .visuals import PALETTE, draw_bars, legend_band, overlay_colors

CLASSES = (
    ("Outside examination", None),
    ("Unavailable", PALETTE["unknown"]),
    ("Masked", PALETTE["excluded"]),
    ("Dark", PALETTE["dark"]),
    ("Highlight clip", PALETTE["clip"]),
    ("Even signal", PALETTE["measured"]),
    ("Readable detail", PALETTE["bright"]),
)


@dataclass(frozen=True)
class ExposureSettings:
    lows: float
    highs: float
    contrast: float
    rows: int
    columns: int
    zone: list | None

    @classmethod
    def parse(cls, raw):
        data = obj(raw)
        low = number(data.get("dark_floor"), "dark floor", .0005, .3, default=.018)
        high = number(data.get("bright_ceiling"), "bright ceiling", .35, 1, default=.95)
        return cls(
            low, high,
            number(data.get("local_detail"), "readable contrast", .001, .15, default=.028),
            integer(data.get("rows"), "audit rows", 1, 16, default=3),
            integer(data.get("columns"), "audit columns", 1, 16, default=4),
            analysis_box(data),
        )

    def as_json(self):
        return {
            "dark_floor": self.lows, "bright_ceiling": self.highs, "local_detail": self.contrast,
            "rows": self.rows, "columns": self.columns, "box": self.zone,
        }


def local_contrast(values, valid):
    dx, dy, edge_support = gradient(values, valid)
    energy = np.square(dx) + np.square(dy)
    field = ndimage.gaussian_filter(np.where(edge_support, energy, 0), 1.8)
    weight = ndimage.gaussian_filter(edge_support.astype(np.float32), 1.8)
    output = np.divide(field, weight, out=np.zeros_like(field), where=weight > .6)
    return np.sqrt(output), edge_support & (weight >= .72)


def grade_classes(surface, values, requested, excluded, settings):
    missing = requested & ~surface.valid
    masked = requested & surface.valid & excluded
    possible = requested & surface.valid & ~excluded
    dark = possible & ((values < settings.lows) | surface.dark)
    clipped = possible & ((values > settings.highs) | surface.saturated)
    clean = possible & ~clipped & ~dark
    contrast, good = local_contrast(values, clean)
    strong = clean & good & (contrast >= settings.contrast)
    tags = np.zeros(values.shape, dtype=np.uint8)
    tags[missing] = 1
    tags[masked] = 2
    tags[dark] = 3
    tags[clipped] = 4
    tags[clean] = 5
    tags[strong] = 6
    return tags, contrast


def describe_patch(tags, values, energy):
    counts = np.bincount(tags.ravel(), minlength=len(CLASSES))
    measured = tags >= 5
    count = int(np.sum(counts[1:]))
    measured_n = int(measured.sum())
    summary = {
        "pixels": count,
        "class_counts": {name: int(counts[i]) for i, (name, _) in enumerate(CLASSES) if i},
    }
    if count:
        summary["usable_percent"] = round(100 * measured_n / count, 3)
        summary["unavailable_percent"] = round(100 * (counts[1] + counts[2]) / count, 3)
    if measured_n:
        summary["tone_p05_p50_p95"] = percentile(values[measured])
        summary["contrast_p05_p50_p95"] = percentile(energy[measured])
    summary["status"] = patch_state(counts)
    return summary


def patch_state(counts):
    active = int(np.sum(counts[1:]))
    if active < 10:
        return "Outside selection"
    if counts[1] + counts[2] > .35 * active:
        return "Obscured or cropped"
    if counts[4] > .24 * active:
        return "Highlight limited"
    if counts[3] > .32 * active:
        return "Shadow limited"
    if counts[5] > .7 * active:
        return "Little local detail"
    return "Readable"


def audit_cells(surface, settings, tags, values, energy):
    x0, y0, x1, y1 = bounds_px(settings.zone, surface.extent)
    rows = []
    for y, x, box in grid_bounds(x1 - x0, y1 - y0, settings.columns, settings.rows):
        left, top, right, bottom = box
        roi = np.s_[y0 + top:y0 + bottom, x0 + left:x0 + right]
        rows.append(describe_patch(tags[roi], values[roi], energy[roi]) | {
            "row": y + 1, "column": x + 1,
            "bounds": [x0 + left, y0 + top, x0 + right, y0 + bottom],
        })
    return rows


def annotate_cells(image, cells):
    plot = image.copy()
    pen = ImageDraw.Draw(plot)
    gray = "#f0e9cb"
    for cell in cells:
        u, v, x, y = cell["bounds"]
        pen.rectangle((u, v, x - 1, y - 1), outline=gray, width=1)
        if x - u > 30 and y - v > 25:
            label(pen, (u + 4, v + 3), f"{cell['row']}.{cell['column']}", size=11)
    return plot


def exposure_audit(ctx, raw):
    settings = ExposureSettings.parse(raw)
    surface = ctx.surface
    values = luminance(surface.linear)
    requested = domain_mask(surface.extent, settings.zone)
    count = np.count_nonzero(requested)
    visible = np.count_nonzero(requested & surface.valid)
    support_limit(count, visible, "Exposure audit", minimum=72)
    tags, contrast = grade_classes(surface, values, requested, ctx.excluded, settings)
    detail = describe_patch(tags, values, contrast)
    cells = audit_cells(surface, settings, tags, values, contrast)
    summary = {
        "settings": settings.as_json(),
        "cell_grid": [settings.columns, settings.rows],
        "whole": detail,
        "cells": cells,
        "cell_states": dict(Counter(cell["status"] for cell in cells)),
        "note": "Dark or clipped glass in this capture is marked separately from unsupported and masked regions.",
    }
    colors = [color for _, color in CLASSES]
    graph = annotate_cells(overlay_colors(surface, tags, colors), cells)
    picture = legend_band(graph, "Photographic support atlas", list(CLASSES[1:]), lines=3)
    pair = [(name, count, shade) for (name, shade), count in zip(CLASSES[1:], detail["class_counts"].values(), strict=True)]
    bars = draw_bars((580, 320), "Pixels by exposure class", [p[0] for p in pair], [p[1] for p in pair], [p[2] for p in pair])
    return summary, {"map": picture, "chart": bars}


def readout_probe(ctx, raw):
    data = obj(raw, "probe")
    u = number(data.get("x"), "horizontal position", 0., 1.)
    v = number(data.get("y"), "vertical position", 0., 1.)
    radius = integer(data.get("radius"), "probe radius", 0, 35, default=4)
    surface = ctx.surface
    width, height = surface.extent
    x, y = min(width - 1, int(u * width)), min(height - 1, int(v * height))
    x0, x1 = max(x - radius, 0), min(x + radius + 1, width)
    y0, y1 = max(y - radius, 0), min(y + radius + 1, height)
    section = np.s_[y0:y1, x0:x1]
    support = ctx.usable[section]
    clean = support & ~surface.dark[section] & ~surface.saturated[section]
    channels = surface.linear[section][clean]
    return {
        "position_px": [x, y],
        "bounds_px": [x0, y0, x1, y1],
        "position_mm": np.round(ctx.scale * [x, y], 4).tolist() if ctx.scale is not None and ctx.alignment else None,
        "supported": int(support.sum()),
        "usable": int(clean.sum()),
        "sampled": int(support.size),
        "rgb_median_linear": np.round(np.median(channels, axis=0), 6).tolist() if len(channels) else None,
        "luminance_p05_p50_p95": percentile(luminance(channels)),
    }
