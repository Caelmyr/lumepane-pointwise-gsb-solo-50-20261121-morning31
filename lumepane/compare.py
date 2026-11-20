"""Registered comparisons of visits, with explicit stable baselines and unknown regions."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .catalog import capture_context
from .contracts import DeskError, enum, identifier, number, obj, support_limit
from .geometry import analysis_box, bounds_px, domain_mask, polygon_mask
from .glasscolor import lab_delta, rgb_to_lab
from .imaging import canvas_picture, font, grid_bounds, label, luminance, masked_gaussian, percentile
from .visuals import GOLD, ROSE, SKY, draw_bars, legend_band, overlay_colors, photo_board, signed_map

CHANGE_COLORS = (None, (108, 113, 139), (108, 157, 215), (242, 153, 104), (108, 166, 151), (236, 203, 115))


@dataclass(frozen=True)
class CompareInputs:
    other: str
    method: str
    threshold: float
    rect: list | None
    reference: str | None
    blur: float
    grid: int

    @classmethod
    def parse(cls, raw):
        item = obj(raw, "visit comparison")
        reference = item.get("reference_mark")
        return cls(
            identifier(item.get("comparison"), "later capture"),
            enum(item.get("difference_kind"), "difference mode", ("luminance", "color"), default="luminance"),
            number(item.get("threshold"), "comparison threshold", .002, 30, default=.08),
            analysis_box(item),
            identifier(reference, "matched stable patch") if reference else None,
            number(item.get("smooth"), "comparison smoothing", 0, 15, default=1.1),
            int(number(item.get("cells"), "grid divisions", 2, 14, default=4)),
        )

    def json(self):
        return {
            "comparison": self.other, "difference_kind": self.method, "threshold": self.threshold,
            "box": self.rect, "reference_mark": self.reference, "smooth": self.blur, "cells": self.grid,
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


def reference_policy(ctx, key, available, delta):
    shape = delta.shape
    zero = 0. if delta.ndim == 2 else np.zeros(shape[-1], dtype=np.float32)
    if key is None:
        return zero, {"patch": None, "applied": False}, []
    mark = next((record for record in ctx.marks if record["id"] == key and record["category"] == "reference"), None)
    if not mark:
        raise DeskError("Choose a recorded reference patch from this window plane.")
    mask = polygon_mask(ctx.surface.extent, mark["points"], mark["holes"])
    sampled = mask & available
    depth = int(np.count_nonzero(sampled))
    if depth < 60:
        raise DeskError("The two images have too few common pixels within the stable patch.")
    values = delta[sampled]
    middle = np.median(values, axis=0)
    spread = 1.4826 * np.median(np.abs(values - middle), axis=0)
    return middle, {
        "patch": mark["id"], "applied": True, "paired_samples": depth,
        "shift": np.round(middle, 6).tolist(), "robust_spread": np.round(spread, 6).tolist(),
    }, [(mark, "stable window patch")]


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


def diff_classes(delta, common, request, threshold, chromatic):
    tags = np.zeros(common.shape, dtype=np.uint8)
    tags[request & ~common] = 1
    if not chromatic:
        tags[common & (delta <= -threshold)] = 2
        tags[common & (delta >= threshold)] = 3
        tags[common & (np.abs(delta) < threshold)] = 4
    else:
        tags[common & (delta >= threshold)] = 5
        tags[common & (delta < threshold)] = 4
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
    align, reference, links = reference_policy(first, compare.reference, basic, initial)
    initial -= align
    clean, valid = blur_join(initial, basic, compare.blur)
    if compare.method == "color":
        clean = np.linalg.norm(clean, axis=2)
    request = domain_mask(first_image.extent, compare.rect)
    visible = request & valid
    support_limit(int(request.sum()), int(visible.sum()), "Comparable visit area", minimum=80)
    tags = diff_classes(clean, visible, request, compare.threshold, compare.method == "color")
    return tags, clean, visible, request, reference, links


def comparative_cells(visible, request, delta, tags, extent, cells):
    entries = []
    for r, c, (x0, y0, x1, y1) in grid_bounds(*extent, cells, cells):
        region = np.s_[y0:y1, x0:x1]
        data = delta[region][visible[region]]
        count = np.bincount(tags[region].ravel(), minlength=6)
        entries.append({
            "row": r + 1, "column": c + 1, "bounds": [x0, y0, x1, y1],
            "mapped": int(request[region].sum()), "paired": len(data), "unmeasured": int(count[1]),
            "weakened": int(count[2]), "increased": int(count[3]), "steady": int(count[4]),
            "changed_color": int(count[5]), "p05_p50_p95": percentile(data),
        })
    return entries


def inspection_map(first, second, option, tags, values, visible, selected):
    if option.method == "color":
        photo = overlay_colors(second.surface, tags, CHANGE_COLORS)
    else:
        photo = signed_map(second.surface, values, visible, option.threshold * 2, selected & ~visible)
    return legend_band(photo, "Registered visit comparison", [
        ("Unknown", CHANGE_COLORS[1]), ("Lower light", CHANGE_COLORS[2]), ("Higher light", CHANGE_COLORS[3]),
        ("Steady", CHANGE_COLORS[4]), ("Color shift", CHANGE_COLORS[5]),
    ])


def visit_comparison(ctx, query, raw):
    settings = CompareInputs.parse(raw)
    later = get_pair(ctx, query, settings.other)
    tags, values, visible, domain, ref, links = matched_metrics(ctx, later, settings)
    counts = np.bincount(tags.ravel(), minlength=6)
    notes = []
    if later.capture["role"] != ctx.capture["role"]:
        notes.append("Different photographed lighting roles. The difference need not be a material change.")
    if not ref["applied"]:
        notes.append("No invariant patch selected. Overall capture drift remains in these differences.")
    if later.capture.get("synthetic") != ctx.capture.get("synthetic"):
        notes.append("This compares demonstration media with photographed material.")
    report = {
        "settings": settings.json(),
        "comparison": later.capture["id"], "first_visit": ctx.capture["visit"], "second_visit": later.capture["visit"],
        "roles": [ctx.capture["role"], later.capture["role"]],
        "reference": ref, "paired_pixels": int(visible.sum()), "selected_pixels": int(domain.sum()),
        "unmeasured_pixels": int(counts[1]),
        "changed_pixels": int(counts[2] + counts[3] + counts[5]),
        "distributions": percentile(values[visible]),
        "cells": comparative_cells(visible, domain, values, tags, ctx.surface.extent, settings.grid),
        "caveats": notes,
    }
    art = inspection_map(ctx, later, settings, tags, values, visible, domain)
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
