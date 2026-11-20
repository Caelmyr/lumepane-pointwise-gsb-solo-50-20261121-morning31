"""Recorded color, swatches, and image-space difference under stated lighting."""

from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

from .contracts import DeskError, enum, identifier, integer, number, obj
from .geometry import analysis_box, domain_mask, polygon_mask
from .imaging import font, image_rgb, linear_to_srgb, luminance, percentile
from .visuals import map_colors, legend_band

D65 = np.array([.95047, 1., 1.08883], dtype=np.float32)
RGB_TO_XYZ = np.array([
    [.4124564, .3575761, .1804375],
    [.2126729, .7151522, .0721750],
    [.0193339, .1191920, .9503041],
], dtype=np.float32)
CUT = 6 / 29


def rgb_to_lab(image):
    xyz = (np.asarray(image, dtype=np.float32) @ RGB_TO_XYZ.T) / D65
    value = np.where(xyz > CUT**3, np.cbrt(xyz), xyz / (3 * CUT**2) + 4 / 29)
    x, y, z = np.moveaxis(value, -1, 0)
    return np.stack([116 * y - 16, 500 * (x - y), 200 * (y - z)], axis=-1)


def lab_delta(left, right):
    error = left - right
    return np.sqrt(np.sum(error * error, axis=-1))


def reference_mark(ctx, raw):
    key = raw.get("neutral_mark")
    if not key:
        return None
    key = identifier(key, "neutral reference")
    return next((mark for mark in ctx.marks if mark["id"] == key and mark["category"] == "reference"), None)


def neutral_scale(ctx, record):
    if record is None:
        return np.ones(3, dtype=np.float32), None
    mask = polygon_mask(ctx.surface.extent, record["points"], record["holes"])
    mask &= ctx.usable & ~ctx.surface.saturated & ~ctx.surface.dark
    if np.count_nonzero(mask) < 72:
        raise DeskError("The neutral patch needs 72 usable measured pixels.")
    rgb = np.median(ctx.surface.linear[mask], axis=0)
    intensity = float(luminance(rgb))
    if np.min(rgb) < .02:
        raise DeskError("The selected neutral patch has an unusably dark channel.")
    gains = intensity / rgb
    if min(gains) < .42 or max(gains) > 2.6:
        raise DeskError("The selected patch is too saturated to be a neutral reference.")
    return gains, {"mark": record["id"], "linear_rgb": np.round(rgb, 6).tolist(), "gains": np.round(gains, 6).tolist()}


def histogram_colours(values, rgb, valid, divisions, desired):
    data = values[valid]
    linear = rgb[valid]
    bounds = [(-120, 120), (-120, 120)]
    light = np.clip((data[:, 0] / 100. * divisions).astype(int), 0, divisions - 1)
    raw = [light]
    for i, (low, high) in enumerate(bounds, 1):
        channel = np.clip(((data[:, i] - low) / (high - low) * divisions).astype(int), 0, divisions - 1)
        raw.append(channel)
    packed = np.ravel_multi_index(raw, (divisions,) * 3)
    bins, counts = np.unique(packed, return_counts=True)
    largest = np.argsort(counts)[::-1][:desired]
    result = []
    for index in largest:
        select = packed == bins[index]
        observed = np.median(linear[select], axis=0)
        s = np.rint(np.clip(linear_to_srgb(observed), 0, 1) * 255).astype(int)
        result.append({
            "pixel_count": int(counts[index]),
            "coverage_percent": round(float(counts[index] / len(packed)) * 100, 4),
            "representative_lab": np.round(np.median(data[select], axis=0), 5).tolist(),
            "swatch": "#" + "".join(f"{value:02x}" for value in s),
        })
    return result


def palette_board(rows):
    width, height = 625, 96 + 44 * max(1, len(rows))
    chart = Image.new("RGB", (width, height), "#f5f2e8")
    pen = ImageDraw.Draw(chart)
    pen.text((20, 17), "Observed color groups", fill="#202939", font=font(23, bold=True))
    pen.text((20, 51), "Swatch  •  area share  •  representative L* / a* / b*", fill="#555e68", font=font(14))
    for index, item in enumerate(rows):
        top = 91 + index * 44
        pen.rounded_rectangle((22, top, 75, top + 27), radius=4, fill=item["swatch"], outline="#adaaa2")
        pen.text((95, top + 4), f"{item['coverage_percent']:.1f}%", fill="#2c3446", font=font(15, bold=True))
        row = " / ".join(f"{value:.2f}" for value in item["representative_lab"])
        pen.text((190, top + 4), row, fill="#3b4553", font=font(14))
        pen.text((490, top + 5), f"{item['pixel_count']:,} px", fill="#444c60", font=font(13))
    return chart


def measured_area(ctx, requested):
    surface = ctx.surface
    domain = domain_mask(surface.extent, requested)
    used = ctx.usable & domain & ~surface.saturated & ~surface.dark
    if np.count_nonzero(used) < 80:
        raise DeskError("The color study needs at least 80 readable pixels.")
    return domain, used


def color_summary(ctx, raw):
    settings = obj(raw)
    box = analysis_box(settings)
    neutral = reference_mark(ctx, settings)
    if settings.get("neutral_mark") and not neutral:
        raise DeskError("This selected mark is not a reference patch in the current view.")
    correction, card = neutral_scale(ctx, neutral)
    rgb = np.clip(ctx.surface.linear * correction, 0, 1)
    all_color = rgb_to_lab(rgb)
    domain, used = measured_area(ctx, box)
    variance = np.median(all_color[used], axis=0)
    error = lab_delta(all_color, variance)
    target = integer(settings.get("swatches"), "number of swatches", 2, 12, default=7)
    division = integer(settings.get("bins"), "color histogram scale", 4, 18, default=12)
    groups = histogram_colours(all_color, rgb, used, division, target)
    summary = {
        "box": box, "role": ctx.capture["role"], "neutral_patch": card,
        "color_space": "sRGB working photo converted to Lab D65", "measured_pixels": int(used.sum()),
        "median_lab": np.round(variance, 6).tolist(), "swatches": groups,
        "variation_delta_e_p05_p50_p95": percentile(error[used]), "unmeasured_pixels": int((domain & ~used).sum()),
        "calibration": "image color with local neutral balancing" if card else "image color without a neutral plane reference",
    }
    gain = max(8., float(np.quantile(error[used], .93)))
    art = map_colors(ctx.surface, error, used, (0, gain), color=(237, 145, 121), unknown=domain & ~used)
    links = [(neutral, "selected neutral patch")] if neutral else []
    return summary, {"map": legend_band(art, "Color variation around this sample", [
        ("Higher difference", (237, 145, 121)), ("Unmeasured", (127, 129, 156)),
    ]), "chart": palette_board(groups)}, links


def swatch_records(records):
    groups = defaultdict(list)
    for record in records:
        group = record["reading"]["role"]
        groups[group].append(record)
    return [{
        "lighting_role": role,
        "approved_samples": [item["id"] for item in samples],
        "swatch_count": sum(len(item["reading"]["swatches"]) for item in samples),
        "capture_count": len({item["capture_id"] for item in samples}),
    } for role, samples in sorted(groups.items())]
