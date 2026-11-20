"""Geometry and condition summaries for already drawn glass-piece boundaries."""

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .contracts import DeskError, enum, flag, identifiers, number, obj
from .geometry import arc_length, domain_mask, normalized_points_to_px, polygon_mask
from .imaging import canvas_picture, gradient, label, luminance, percentile
from .visuals import legend_band, overlay_colors


def selected_panes(marks, raw):
    request = obj(raw)
    keys = set(identifiers(request["panes"], "glass pieces", high=24)) if request.get("panes") else None
    pieces = [item for item in marks if item["category"] == "pane" and item["shape"] == "polygon"]
    if keys is not None:
        pieces = [item for item in pieces if item["id"] in keys]
    if not pieces:
        raise DeskError("Draw a glass-piece outline on this window first.", field="panes")
    if keys is not None and len(pieces) != len(keys):
        raise DeskError("A glass-piece outline is not present on this view.", field="panes")
    if len(pieces) > 24:
        raise DeskError("Choose up to 24 pieces for one coverage check.", field="panes")
    return pieces


def physical_area(pixels, scale):
    return round(float(pixels * np.prod(scale)), 4) if scale is not None else None


def polygon_limits(mask, extent):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    width, height = extent
    return [int(xs.min()) / width, int(ys.min()) / height, (int(xs.max()) + 1) / width, (int(ys.max()) + 1) / height]


def pane_stats(surface, territory, clipped, support, scale, outline, label_id, conflicts):
    image_count = int(territory.sum())
    known = territory & support
    visible = int(known.sum())
    photometric = known & ~clipped
    vals = luminance(surface.linear)[photometric]
    box = polygon_limits(territory, surface.extent)
    perimeter = normalized_points_to_px(outline["points"] + outline["points"][:1], surface.extent)
    return {
        "number": label_id,
        "id": outline["id"],
        "title": outline["title"],
        "capture_bounds": box,
        "requested_pixels": image_count,
        "visible_pixels": visible,
        "photometric_pixels": int(len(vals)),
        "competing_mark_pixels": int(np.count_nonzero(conflicts & territory)),
        "coverage_percent": round(100 * visible / image_count, 3) if image_count else 0.0,
        "plane_area_mm2": physical_area(image_count, scale),
        "visible_area_mm2": physical_area(visible, scale),
        "perimeter_px": round(arc_length(perimeter), 3),
        "perimeter_mm": round(arc_length(perimeter, scale), 3) if scale is not None else None,
        "luma_quantiles": percentile(vals),
        "basis": outline["coordinate_basis"],
        "status": outline["state"],
    }


def occupancy_for(extent, outlines):
    masks = [polygon_mask(extent, item["points"], item["holes"]) for item in outlines]
    claims = np.zeros(extent[::-1], dtype=np.uint8)
    for mask in masks:
        claims += mask
    return masks, claims


def draw_piece_index(image, rows):
    plate = image.copy()
    pen = ImageDraw.Draw(plate)
    width, height = image.size
    for entry in rows:
        bounds = entry["capture_bounds"]
        if not bounds:
            continue
        center = ((bounds[0] + bounds[2]) * width * .5, (bounds[1] + bounds[3]) * height * .5)
        label(pen, center, str(entry["number"]), fill="#ffe9a6", size=15)
    return plate


def pane_inventory(ctx, raw):
    pieces = selected_panes(ctx.marks, raw)
    surface = ctx.surface
    masks, claims = occupancy_for(surface.extent, pieces)
    valid = ctx.usable
    scale = ctx.scale if ctx.alignment else None
    stats = [
        pane_stats(surface, mask, surface.dark | surface.saturated, valid, scale, outline, i, claims >= 2)
        for i, (mask, outline) in enumerate(zip(masks, pieces, strict=True), 1)
    ]
    visible_union = (claims > 0) & valid
    classes = np.zeros(valid.shape, dtype=np.uint8)
    classes[visible_union] = 1
    classes[(claims > 0) & ~valid] = 2
    classes[claims >= 2] = 3
    colors = (None, (123, 187, 170), (149, 148, 177), (235, 136, 106))
    art = draw_piece_index(overlay_colors(surface, classes, colors), stats)
    rows = {
        "glass_pieces": stats,
        "requested_union_pixels": int(np.count_nonzero(claims)),
        "covered_union_pixels": int(visible_union.sum()),
        "ambiguous_union_pixels": int(np.count_nonzero(claims >= 2)),
        "covered_union_mm2": physical_area(int(visible_union.sum()), scale),
        "note": "Pieces remain individual observations. Duplicate outline coverage is counted once in the union and flagged.",
    }
    return rows, {"map": legend_band(art, "Glass inventory", [("Visible piece", colors[1]), ("Not visible", colors[2]), ("Overlap review", colors[3])])}


def distance_band(mask, amount):
    coords = ndimage.distance_transform_edt(mask)
    return mask & (coords > 0) & (coords <= amount)


def split_rim(mask, band, center):
    dy, dx = np.indices(mask.shape, dtype=np.float32)
    cx, cy = center
    angle = ((np.arctan2(dy - cy, dx - cx) * 180 / math.pi + 405) % 360 / 90).astype(int)
    return [(direction, band & (angle == i)) for i, direction in enumerate(("right", "bottom", "left", "top"))]


def rim_metrics(ctx, mask, which, width):
    radii = ndimage.distance_transform_edt(mask)
    rims = (radii > 0) & (radii <= width)
    core = radii > width * 2
    if not np.any(core & ctx.usable):
        raise DeskError("The selected piece has no clear interior at this rim width.")
    values = luminance(ctx.surface.linear)
    gx, gy, support = gradient(values, ctx.usable)
    magnitude = np.hypot(gx, gy)
    median_core = float(np.median(values[core & ctx.usable]))
    ys, xs = np.nonzero(mask)
    sides = []
    for side, band in split_rim(mask, rims, (float(np.mean(xs)), float(np.mean(ys)))):
        readable = band & ctx.usable & ~ctx.surface.saturated & ~ctx.surface.dark
        estimates = values[readable]
        gradient_values = magnitude[support & readable]
        sides.append({
            "side": side,
            "measurements": int(readable.sum()),
            "pixels": int(band.sum()),
            "median_offset": round(float(np.median(estimates) - median_core), 6) if len(estimates) else None,
            "strong_edge_p90": round(float(np.quantile(gradient_values, .9)), 6) if len(gradient_values) else None,
            "usable_fraction": round(float(readable.sum() / max(1, band.sum())), 5),
        })
    return {
        "piece_title": which["title"], "piece_id": which["id"], "width_px": width,
        "reference_interior": median_core, "sides": sides,
    }, rims, core


def rim_view(ctx, raw):
    wanted = obj(raw)
    keys = wanted.get("panes")
    pieces = selected_panes(ctx.marks, raw)
    if len(pieces) != 1:
        raise DeskError("Choose one glass-piece outline for the rim comparison.")
    width = number(wanted.get("rim_width"), "rim depth", 2, 80, default=10)
    selected = pieces[0]
    mask = polygon_mask(ctx.surface.extent, selected["points"], selected["holes"])
    report, rims, core = rim_metrics(ctx, mask, selected, width)
    classes = np.zeros(mask.shape, dtype=np.uint8)
    classes[core & ctx.usable] = 1
    classes[rims & ctx.usable] = 2
    classes[mask & ~ctx.usable] = 3
    colors = (None, (106, 167, 183), (240, 206, 137), (142, 128, 172))
    picture = overlay_colors(ctx.surface, classes, colors)
    return report, {"map": legend_band(picture, "Inside edge examination", [
        ("Interior sample", colors[1]), ("Rim sample", colors[2]), ("Unobserved here", colors[3])
    ])}


def overlap_attrib(ctx, raw):
    request = obj(raw)
    selected = selected_panes(ctx.marks, raw)
    width = number(request.get("distance"), "edge tolerance", 1, 24, default=4)
    masks, claims = occupancy_for(ctx.surface.extent, selected)
    references = []
    accepted = ctx.usable
    for i, left in enumerate(masks):
        edge = distance_band(left, width) & accepted
        expanded = ndimage.distance_transform_edt(~left) <= width
        for j, right in enumerate(masks[i + 1:], i + 1):
            mutual = expanded & right & accepted
            collision = left & right
            if not np.any(mutual) and not np.any(collision):
                continue
            references.append({
                "first": selected[i]["title"], "second": selected[j]["title"],
                "overlapping_pixels": int(collision.sum()),
                "adjacent_support_pixels": int(mutual.sum()),
                "distance_px": width,
                "left_visible_edge_pixels": int(edge.sum()),
            })
    classes = np.zeros(accepted.shape, dtype=np.uint8)
    near = np.zeros(accepted.shape, dtype=np.uint8)
    for mask in masks:
        near += (ndimage.distance_transform_edt(~mask) <= width).astype(np.uint8)
    classes[(near >= 2) & accepted] = 1
    classes[claims >= 2] = 2
    colors = (None, (144, 211, 180), (235, 127, 114))
    picture = overlay_colors(ctx.surface, classes, colors)
    return {
        "neighbour_pairs": references, "overlap_pixels": int(np.count_nonzero(claims >= 2)),
        "note": "Nearby boundaries show review contact, while overlaps reveal conflicting ownership on the same map.",
    }, {"map": legend_band(picture, "Marked edge review", [
        ("Near two marked edges", colors[1]), ("Competing outlines", colors[2])
    ])}
