"""Registered comparisons of visits, with stable-patch illumination fields and unknown regions."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .catalog import capture_context
from .contracts import DeskError, enum, identifier, identifiers, number, obj, support_limit
from .geometry import analysis_box, domain_mask, polygon_mask
from .glasscolor import rgb_to_lab
from .imaging import canvas_picture, grid_bounds, label, luminance, masked_gaussian, percentile
from .visuals import legend_band, overlay_colors, photo_board, signed_map

CHANGE_COLORS = (None, (108, 113, 139), (108, 157, 215), (242, 153, 104), (108, 166, 151), (236, 203, 115),
                 (181, 132, 190))
PATCH_MIN_SAMPLES = 60
INCOHERENT_LIMIT = 4.0
FIELD_MARGIN_PX = .012


@dataclass(frozen=True)
class CompareInputs:
    other: str
    method: str
    threshold: float
    rect: list | None
    references: tuple
    blur: float
    grid: int

    @classmethod
    def parse(cls, raw):
        item = obj(raw, "visit comparison")
        marks = item.get("reference_marks")
        if marks is None and item.get("reference_mark"):
            marks = [item.get("reference_mark")]
        return cls(
            identifier(item.get("comparison"), "later capture"),
            enum(item.get("difference_kind"), "difference mode", ("luminance", "color"), default="luminance"),
            number(item.get("threshold"), "comparison threshold", .002, 30, default=.08),
            analysis_box(item),
            tuple(identifiers(marks, "matched stable patch", low=1, high=12)) if marks else (),
            number(item.get("smooth"), "comparison smoothing", 0, 15, default=1.1),
            int(number(item.get("cells"), "grid divisions", 2, 14, default=4)),
        )

    def json(self):
        return {
            "comparison": self.other, "difference_kind": self.method, "threshold": self.threshold,
            "box": self.rect, "reference_mark": self.references[0] if len(self.references) == 1 else None,
            "reference_marks": list(self.references), "smooth": self.blur, "cells": self.grid,
        }


def get_pair(ctx, query, key):
    candidate = capture_context(query, key, registered=True)
    if not ctx.alignment:
        raise DeskError("Register both captures to the window plane.")
    if ctx.capture["id"] == candidate.capture["id"] or ctx.window["id"] != candidate.window["id"]:
        raise DeskError("Pick two different captures of the same window.")
    if ctx.surface.extent != candidate.surface.extent:
        raise DeskError("These two captures do not use the same window resolution.")
    return candidate


def patch_observations(ctx, keys, available, delta):
    marks = {record["id"]: record for record in ctx.marks if record["category"] == "reference"}
    observations, rejected, links = [], [], []
    for key in keys:
        mark = marks.get(key)
        if not mark:
            raise DeskError("Choose a recorded reference patch from this window plane.")
        sampled = polygon_mask(ctx.surface.extent, mark["points"], mark["holes"]) & available
        depth = int(np.count_nonzero(sampled))
        if depth < PATCH_MIN_SAMPLES:
            rejected.append({"patch": key, "title": mark["title"], "paired_samples": depth,
                             "status": "unpaired", "reason": "Fewer than 60 common pixels; only one capture covers it."})
            continue
        values = delta[sampled]
        middle = np.median(values, axis=0)
        spread = 1.4826 * np.median(np.abs(values - middle), axis=0)
        centroid = np.argwhere(sampled).mean(axis=0)[::-1] / np.asarray(ctx.surface.extent)
        observations.append({"key": key, "title": mark["title"], "shift": np.atleast_1d(middle),
                             "spread": np.atleast_1d(spread), "center": centroid, "depth": depth})
        links.append((mark, "stable window patch"))
    return observations, rejected, links


def hull_mask(extent, centers):
    width, height = extent
    points = [(float(x * width - .5), float(y * height - .5)) for x, y in centers]
    area = Image.new("L", extent)
    ordered = sorted(points, key=lambda p: (p[1], p[0]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    ImageDraw.Draw(area).polygon(lower[:-1] + upper[:-1], fill=255)
    margin = max(2, int(round(FIELD_MARGIN_PX * min(extent))))
    return ndimage.binary_dilation(np.asarray(area, dtype=bool), iterations=margin)


def fit_coefficients(observations, channels):
    centers = np.stack([item["center"] for item in observations])
    shifts = np.stack([item["shift"] for item in observations]).reshape(len(observations), channels)
    weights = 1.0 / np.maximum(np.stack([item["spread"] for item in observations]).reshape(len(observations), channels), 1e-6)
    rank = int(np.linalg.matrix_rank(np.column_stack((centers, np.ones(len(observations)))) * np.sqrt(weights.max(axis=1, keepdims=True))))
    basis = (0, 1) if len(observations) > 2 and rank >= 3 else \
        ((int(np.argmax(centers.std(axis=0))),) if len(observations) > 1 else ())
    design = np.column_stack([centers[:, axis] for axis in basis] + [np.ones(len(observations))])
    root_w = np.sqrt(weights)
    weighted_design = design[None, :, :] * root_w[:, :, None].swapaxes(0, 1)
    weighted_target = (shifts * root_w).T[:, :, None]
    gram = weighted_design.swapaxes(1, 2) @ weighted_design
    solved = np.linalg.solve(gram, weighted_design.swapaxes(1, 2) @ weighted_target)[:, :, 0]
    coefficients = np.zeros((channels, 3), dtype=np.float64)
    coefficients[:, basis + (2,)] = solved
    return coefficients, basis


def residual_scores(observations, coefficients):
    centers = np.stack([item["center"] for item in observations])
    shifts = np.stack([np.atleast_1d(item["shift"]) for item in observations])
    spreads = np.maximum(np.stack([np.atleast_1d(item["spread"]) for item in observations]), 1e-6)
    predicted = centers @ coefficients[:, :2].T + coefficients[:, 2]
    return np.max(np.abs(shifts - predicted) / spreads, axis=1).tolist()


def solve_field(observations, extent, channels):
    remaining, dropped = list(observations), []
    while True:
        coefficients, basis = fit_coefficients(remaining, channels)
        if len(remaining) == 1:
            break
        scores = residual_scores(remaining, coefficients)
        if max(scores) <= INCOHERENT_LIMIT:
            break
        dropped.append(remaining.pop(int(np.argmax(scores))))
    xx, yy = np.meshgrid(np.arange(extent[0]) / extent[0], np.arange(extent[1]) / extent[1])
    field = coefficients[:, 2][None, None] + coefficients[:, 0] * xx[..., None] + coefficients[:, 1] * yy[..., None]
    remaining[0]["field"] = field.reshape(*extent[::-1], channels).squeeze().astype(np.float32)
    return remaining, dropped, basis


def reference_policy(ctx, keys, available, delta):
    shape = delta.shape
    zero = 0. if delta.ndim == 2 else np.zeros(shape[-1], dtype=np.float32)
    if not keys:
        return zero, np.ones(shape[:2], dtype=bool), {"applied": False, "patches": [], "model": "none",
                                                       "coverage_fraction": 1.0}, [], []
    observations, rejected, links = patch_observations(ctx, keys, available, delta)
    if not observations:
        raise DeskError("None of the selected stable patches has enough common pixels in both captures.")
    if len(observations) == 1:
        only = observations[0]
        field = np.broadcast_to(only["shift"].astype(np.float32), shape).copy()
        trusted, dropped, basis = np.ones(shape[:2], dtype=bool), [], ()
    else:
        observations, dropped, basis = solve_field(observations, ctx.surface.extent, shape[-1] if delta.ndim == 3 else 1)
        field = observations[0]["field"]
        trusted = hull_mask(ctx.surface.extent, [item["center"] for item in observations])
    dropped_keys = {item["key"] for item in dropped}
    details = [
        {"patch": item["key"], "title": item["title"], "paired_samples": item["depth"],
         "shift": np.round(item["shift"], 6).tolist(), "robust_spread": np.round(item["spread"], 6).tolist(),
         "status": "excluded_inconsistent" if item["key"] in dropped_keys else "used",
         "centroid": [round(float(v), 5) for v in item["center"]]}
        for item in observations + dropped
    ]
    details.extend(rejected)
    used_count = len(observations)
    return field.astype(np.float32), trusted, {
        "applied": True, "patch": details[0]["patch"] if used_count == 1 and not dropped else None,
        "patches": details, "model": "constant" if used_count == 1 else ("ramp" if len(basis) <= 1 else "planar"),
        "coverage_fraction": round(float(trusted.mean()), 5),
    }, links, dropped


def blur_join(values, validity, amount):
    if amount <= .01:
        return values, validity
    if values.ndim == 2:
        return masked_gaussian(values, validity, amount)
    variants = []
    support = validity.copy()
    for index in range(values.shape[-1]):
        piece, clear = masked_gaussian(values[..., index], validity, amount)
        variants.append(piece)
        support &= clear
    return np.stack(variants, axis=-1), support


def diff_classes(delta, common, request, threshold, chromatic, trusted):
    tags = np.zeros(common.shape, dtype=np.uint8)
    tags[request & ~trusted] = 6
    tags[request & trusted & ~common] = 1
    judged = common & trusted
    if not chromatic:
        tags[judged & (delta <= -threshold)] = 2
        tags[judged & (delta >= threshold)] = 3
        tags[judged & (np.abs(delta) < threshold)] = 4
    else:
        tags[judged & (delta >= threshold)] = 5
        tags[judged & (delta < threshold)] = 4
    return tags


def matched_metrics(first, second, compare):
    first_image, second_image = first.surface, second.surface
    basic = first.usable & second.usable
    basic &= ~first_image.saturated & ~second_image.saturated & ~first_image.dark & ~second_image.dark
    new = second_image.linear
    old = first_image.linear
    if compare.method == "luminance":
        initial = luminance(new) - luminance(old)
    else:
        initial = rgb_to_lab(new) - rgb_to_lab(old)
    align, trusted, reference, links, dropped = reference_policy(first, compare.references, basic, initial)
    initial -= align
    clean, valid = blur_join(initial, basic, compare.blur)
    valid &= trusted
    if compare.method == "color":
        clean = np.linalg.norm(clean, axis=2)
    request = domain_mask(first_image.extent, compare.rect)
    visible = request & valid
    support_limit(int(request.sum()), int(visible.sum()), "Comparable visit area", minimum=80)
    tags = diff_classes(clean, valid, request, compare.threshold, compare.method == "color", trusted)
    return tags, clean, visible, request, reference, links, dropped, trusted


def comparative_cells(visible, request, delta, tags, extent, cells):
    entries = []
    for r, c, (x0, y0, x1, y1) in grid_bounds(*extent, cells, cells):
        region = np.s_[y0:y1, x0:x1]
        data = delta[region][visible[region]]
        count = np.bincount(tags[region].ravel(), minlength=7)
        entries.append({
            "row": r + 1, "column": c + 1, "bounds": [x0, y0, x1, y1],
            "mapped": int(request[region].sum()), "paired": len(data), "unmeasured": int(count[1]),
            "uncomparable": int(count[6]),
            "weakened": int(count[2]), "increased": int(count[3]), "steady": int(count[4]),
            "changed_color": int(count[5]), "p05_p50_p95": percentile(data),
        })
    return entries


def inspection_map(second, option, tags, values, visible, selected, trusted):
    unknown = (selected & ~visible) & trusted
    outside = selected & ~trusted
    if option.method == "color":
        photo = overlay_colors(second.surface, tags, CHANGE_COLORS)
    else:
        photo = signed_map(second.surface, values, visible, option.threshold * 2, unknown, outside, CHANGE_COLORS[6])
    return legend_band(photo, "Registered visit comparison", [
        ("Unknown (no image support)", CHANGE_COLORS[1]), ("Lower light", CHANGE_COLORS[2]),
        ("Higher light", CHANGE_COLORS[3]), ("Steady", CHANGE_COLORS[4]), ("Color shift", CHANGE_COLORS[5]),
        ("Uncomparable: outside stable-patch coverage", CHANGE_COLORS[6]),
    ])


def visit_comparison(ctx, query, raw):
    settings = CompareInputs.parse(raw)
    later = get_pair(ctx, query, settings.other)
    tags, values, visible, domain, ref, links, dropped, trusted = matched_metrics(ctx, later, settings)
    counts = np.bincount(tags.ravel(), minlength=7)
    notes = []
    if later.capture["role"] != ctx.capture["role"]:
        notes.append("Different photographed lighting roles. The difference need not be a material change.")
    if not ref["applied"]:
        notes.append("No invariant patch selected. Overall capture drift remains in these differences.")
    elif len(settings.references) > 1:
        notes.append("Only pixels inside the hull joining the stable patches are judged; the rest is uncomparable.")
    if any(patch.get("status") == "unpaired" for patch in ref["patches"]):
        notes.append("A selected stable patch is visible in only one capture and could not calibrate its area.")
    if dropped:
        names = ", ".join(patch["title"] for patch in dropped)
        notes.append(f"Stable patches inconsistent with the fitted illumination field were excluded: {names}.")
    if later.capture.get("synthetic") != ctx.capture.get("synthetic"):
        notes.append("This compares demonstration media with photographed material.")
    report = {
        "settings": settings.json(),
        "comparison": later.capture["id"], "first_visit": ctx.capture["visit"], "second_visit": later.capture["visit"],
        "roles": [ctx.capture["role"], later.capture["role"]],
        "reference": ref, "paired_pixels": int(visible.sum()), "selected_pixels": int(domain.sum()),
        "unmeasured_pixels": int(counts[1]), "uncomparable_pixels": int(counts[6]),
        "changed_pixels": int(counts[2] + counts[3] + counts[5]),
        "distributions": percentile(values[visible]),
        "cells": comparative_cells(visible, domain, values, tags, ctx.surface.extent, settings.grid),
        "caveats": notes,
    }
    art = inspection_map(later, settings, tags, values, visible, domain, trusted)
    companions = photo_board("Source visits on the window plane", [
        (ctx.capture["caption"], canvas_picture(ctx.surface)),
        (later.capture["caption"], canvas_picture(later.surface)),
    ])
    links.extend([(later.capture, "later visit"), (later.alignment, "later registration")])
    return report, {"map": art, "source": companions}, links


def comparator_wipe(ctx, query, raw):
    other = get_pair(ctx, query, identifier(raw.get("comparison"), "comparison capture"))
    mode = enum(raw.get("wipe"), "wipe", ("vertical", "horizontal", "checker"), default="vertical")
    boundary = number(raw.get("boundary"), "wipe location", .05, .95, default=.5)
    board_a = canvas_picture(ctx.surface)
    board_b = canvas_picture(other.surface)
    width, height = board_a.size
    if mode == "vertical":
        layout = (int(boundary * width), 0, width, height)
        change = Image.new("L", board_a.size)
        ImageDraw.Draw(change).rectangle(layout, fill=255)
    elif mode == "horizontal":
        layout = (0, int(boundary * height), width, height)
        change = Image.new("L", board_a.size)
        ImageDraw.Draw(change).rectangle(layout, fill=255)
    else:
        yy, xx = np.ogrid[:height, :width]
        edge = max(12, int(min(width, height) * boundary * .26))
        pattern = (((xx // edge) + (yy // edge)) % 2 * 255).astype(np.uint8)
        change = Image.fromarray(pattern, "L")
    output = Image.composite(board_b, board_a, change)
    painter = ImageDraw.Draw(output)
    if mode == "vertical":
        painter.line((layout[0], 0, layout[0], height), fill="#ffdfa5", width=2)
    elif mode == "horizontal":
        painter.line((0, layout[1], width, layout[1]), fill="#ffdfa5", width=2)
    label(painter, (10, 9), f"{ctx.capture['caption']}  /  {other.capture['caption']}")
    return output
