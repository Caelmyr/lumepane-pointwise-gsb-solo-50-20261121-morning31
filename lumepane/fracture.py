"""Map fine line evidence, distinct from identifying or joining actual fractures."""

from dataclasses import dataclass
import math

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .contracts import DeskError, enum, integer, number, obj, support_limit
from .geometry import analysis_box, domain_mask, mask_for_marks, polygon_mask
from .imaging import canvas_picture, label, luminance, masked_gaussian, percentile
from .visuals import legend_band, overlay_colors

NEIGHBOURS = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class FractureSettings:
    box: list | None
    feature_width: int
    sensitivity: float
    minimum: int
    max_candidates: int
    polarity: str
    seam_margin: float

    @classmethod
    def parse(cls, raw):
        data = obj(raw)
        return cls(
            analysis_box(data),
            integer(data.get("feature_width"), "trace width", 1, 13, default=3),
            number(data.get("sensitivity"), "ridge sensitivity", .005, .35, default=.055),
            integer(data.get("minimum_area"), "minimum candidate area", 4, 1000, default=22),
            integer(data.get("max_candidates"), "candidate count", 1, 60, default=25),
            enum(data.get("polarity"), "trace side", ("dark", "bright", "either"), default="dark"),
            number(data.get("seam_margin"), "came clearance", 0, 35, default=5),
        )

    def json(self):
        return {
            "box": self.box, "feature_width": self.feature_width,
            "sensitivity": self.sensitivity, "minimum_area": self.minimum, "max_candidates": self.max_candidates,
            "polarity": self.polarity, "seam_margin": self.seam_margin,
        }


def line_coverage(extent, markings, width):
    from PIL import Image
    image = Image.new("1", extent)
    painter = ImageDraw.Draw(image)
    for mark in markings:
        if mark["shape"] == "line":
            line = [(int(x * extent[0]), int(y * extent[1])) for x, y in mark["points"]]
            painter.line(line, width=max(1, int(width * 2 + 1)), fill=1)
        else:
            item = polygon_mask(extent, mark["points"], mark["holes"])
            existing = np.asarray(image, dtype=bool)
            image = Image.fromarray(existing | item)
            painter = ImageDraw.Draw(image)
    return np.array(image, dtype=bool)


def lead_excluded(ctx, width):
    came = [item for item in ctx.marks if item["category"] == "came"]
    if not came:
        return np.zeros(ctx.surface.valid.shape, dtype=bool)
    covered = line_coverage(ctx.surface.extent, came, width)
    return (ndimage.distance_transform_edt(~covered) <= width) | covered


def hessian_response(detail, scale, polarity):
    dy, dx = np.gradient(detail)
    yy, yx = np.gradient(dy)
    xy, xx = np.gradient(dx)
    mixed = .5 * (yx + xy)
    total = .5 * (xx + yy)
    spread = np.hypot(.5 * (xx - yy), mixed)
    a, b = total - spread, total + spread
    primary = np.where(np.abs(a) >= np.abs(b), a, b)
    secondary = np.where(np.abs(a) >= np.abs(b), b, a)
    ratio = np.abs(secondary) / np.maximum(np.abs(primary), 1e-7)
    suppression = np.square(np.clip(1 - ratio, 0, 1))
    combined = np.abs(primary) * suppression * (scale**2)
    if polarity == "dark":
        combined[primary < 0] = 0
    elif polarity == "bright":
        combined[primary > 0] = 0
    return combined


def fracture_field(ctx, settings):
    surface = ctx.surface
    selection = domain_mask(surface.extent, settings.box)
    came = lead_excluded(ctx, settings.seam_margin)
    visible = ctx.usable & ~surface.saturated & ~surface.dark
    eligible = selection & ~came & visible
    support_limit(int(np.count_nonzero(selection)), int(eligible.sum()), "Fracture evidence", minimum=75)
    light = luminance(surface.linear)
    background, support = masked_gaussian(light, visible, max(4, settings.feature_width * 3))
    high = np.where(support, light - background, 0)
    signal, core = masked_gaussian(high, support, max(.6, settings.feature_width / 2.8))
    support = ndimage.binary_erosion(core & eligible, iterations=max(1, settings.feature_width // 2))
    response = hessian_response(signal, max(1, settings.feature_width), settings.polarity)
    response[~support] = 0
    samples = response[support]
    if not len(samples):
        raise DeskError("Occlusion and framing leave no stable ridge samples.")
    robust = 1.4826 * np.median(np.abs(samples - np.median(samples)))
    threshold = float(max(settings.sensitivity / 3, 2.8 * robust))
    active = support & (response >= threshold)
    return {
        "selection": selection, "eligible": eligible, "came": came,
        "response": response, "threshold": threshold, "ridge_support": support,
        "detections": active, "line_contrast": high,
        "signal": signal, "core": core, "visible": visible,
    }


def component_shape(points, scale):
    spread_cov = np.cov(points, rowvar=False)
    eigenval, eigenvec = np.linalg.eigh(spread_cov)
    direction = eigenvec[:, 1]
    centered = points - np.mean(points, axis=0)
    projection = centered @ direction
    length = float(projection.max() - projection.min())
    slender = math.sqrt(max(float(eigenval[1]), 0.) / max(float(eigenval[0]), 1e-5))
    angle = math.atan2(direction[1], direction[0]) * 180. / math.pi % 180
    return {
        "spread_length_px": round(length, 4),
        "spread_length_mm": round(length * np.linalg.norm(direction * scale), 4) if scale is not None else None,
        "aspect_of_spread": round(slender, 5),
        "main_axis_degrees": round(angle, 3),
    }


def neighbour_ends(mask, known):
    border = ndimage.binary_dilation(mask, iterations=2) & ~mask
    if np.count_nonzero(border) == 0:
        return False, 0
    unknown = int(np.count_nonzero(border & ~known))
    return unknown > 0, unknown


def observation_match(width, height, parts, marks):
    for mark in marks:
        coords = [(int(x * width), int(y * height)) for x, y in mark["points"]]
        if not coords:
            continue
        if mark["shape"] == "polygon":
            coverage = polygon_mask((width, height), mark["points"], mark["holes"])
        else:
            coverage = line_coverage((width, height), [mark], width=3)
        overlap = int(np.count_nonzero(parts & coverage))
        if overlap:
            yield {"observation": mark["id"], "title": mark["title"], "overlapping_pixels": overlap}


def candidates_for(ctx, signal, settings):
    labels, _ = ndimage.label(signal["detections"], structure=NEIGHBOURS)
    areas = np.bincount(labels.ravel())
    accepted = [index for index in np.flatnonzero(areas >= settings.minimum) if index != 0]
    accepted.sort(key=lambda i: -int(areas[i]))
    boxes = ndimage.find_objects(labels)
    listed = []
    map_mask = np.zeros(labels.shape, dtype=bool)
    index_image = np.zeros(labels.shape, dtype=np.int32)
    extent = ctx.surface.extent
    known = signal["ridge_support"]
    marked = [m for m in ctx.marks if m["category"] == "fracture"]
    for i in accepted[:settings.max_candidates]:
        local = boxes[i - 1]
        if local is None:
            continue
        zone = labels[local] == i
        y, x = np.nonzero(zone)
        if len(y) < 4:
            continue
        xs, ys = local[1], local[0]
        points = np.column_stack((x + xs.start, y + ys.start))
        result = component_shape(points, ctx.scale if ctx.alignment else None)
        if result["aspect_of_spread"] < 1.65:
            continue
        territory = labels == i
        ev_touch, unknown = neighbour_ends(territory, known)
        data = signal["response"][local][zone]
        original_change = signal["line_contrast"][local][zone]
        width, height = extent
        idx = len(listed) + 1
        listed.append(result | {
            "index": idx,
            "response_p05_p50_p95": percentile(data),
            "signed_center_contrast": round(float(np.median(original_change)), 6),
            "sample_pixels": int(len(x)),
            "bounds": [xs.start / width, ys.start / height, xs.stop / width, ys.stop / height],
            "centroid": [round(float(points[:, 0].mean() / width), 6), round(float(points[:, 1].mean() / height), 6)],
            "ends_near_unknown": ev_touch,
            "border_with_unknown_samples": unknown,
            "source_marks": list(observation_match(width, height, territory, marked)),
        })
        map_mask |= territory
        index_image[territory] = idx
    return listed, map_mask, len(accepted), index_image


def label_candidates(picture, found):
    pen = ImageDraw.Draw(picture)
    width, height = picture.size
    for r in found:
        box = [item * (width if i % 2 == 0 else height) for i, item in enumerate(r["bounds"])]
        pen.rectangle(tuple(box), outline="#f0b891", width=max(1, width // 640))
        label(pen, (box[0], max(0, box[1] - 14)), str(r["index"]), size=11, fill="#f5c998")
    return picture


def fracture_candidates(ctx, raw):
    settings = FractureSettings.parse(raw)
    field = fracture_field(ctx, settings)
    candidates, marked, raw_count, _ = candidates_for(ctx, field, settings)
    classes = np.zeros(marked.shape, dtype=np.uint8)
    classes[field["selection"] & ~field["eligible"]] = 1
    classes[field["selection"] & field["came"]] = 2
    classes[marked] = 3
    colors = (None, (125, 131, 157), (110, 165, 167), (242, 146, 113))
    image = overlay_colors(ctx.surface, classes, colors)
    annotated = label_candidates(image, candidates)
    readout = {
        "settings": settings.json(), "candidate_count": len(candidates), "raw_component_count": raw_count,
        "threshold": round(field["threshold"], 6), "ridge_support_pixels": int(field["ridge_support"].sum()),
        "came_exclusion_pixels": int(np.count_nonzero(field["selection"] & field["came"])),
        "review_candidates": candidates,
        "missing_pixels": int(np.count_nonzero(field["selection"] & ~field["eligible"])),
        "note": "Candidate ridges are visible image evidence. Where a fragment meets a mask, continuity remains unverified.",
    }
    return readout, {"map": legend_band(annotated, "Fine fracture evidence", [
        ("Weak or missing", colors[1]), ("Came exclusion", colors[2]), ("Candidate ridge", colors[3]),
    ])}


def manual_trace_summary(ctx, raw):
    request = obj(raw)
    selected = analysis_box(request)
    support = domain_mask(ctx.surface.extent, selected)
    relevant = [m for m in ctx.marks if m["category"] == "fracture"]
    if not relevant:
        raise DeskError("Draw or save an observed fracture first.")
    statistics = []
    for mark in relevant:
        drawn = line_coverage(ctx.surface.extent, [mark], width=2) & support
        if not np.any(drawn):
            continue
        photographed = int(np.count_nonzero(drawn & ctx.usable))
        total = int(np.count_nonzero(drawn))
        stats = mark["measure"]
        statistics.append({
            "title": mark["title"], "id": mark["id"], "trace_px": stats.get("trace_px"),
            "trace_mm": stats.get("trace_mm"), "visible_trace_fraction": round(photographed / total, 5),
            "visible_samples": photographed, "drawn_samples": total,
        })
    if not statistics:
        raise DeskError("No marked line runs through this window selection.")
    from .imaging import marks_display
    art = marks_display(canvas_picture(ctx.surface), relevant)
    return {"traces": statistics, "trace_count": len(statistics)}, {"map": art}
