"""Relative and calibrated transmitted-light maps from a photographed reference."""

from dataclasses import dataclass

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .catalog import capture_context
from .contracts import DeskError, enum, identifier, integer, number, obj, support_limit
from .geometry import analysis_box, bounds_px, domain_mask
from .imaging import gradient, grid_bounds, label, luminance, masked_gaussian, open_surface, percentile
from .visuals import draw_bars, legend_band, map_colors, overlay_colors


@dataclass(frozen=True)
class TransmissionSettings:
    method: str
    reference: str | None
    black: str | None
    floor: float
    window: list | None
    tie: list | None
    quality_limit: float
    columns: int
    rows: int

    @classmethod
    def parse(cls, raw):
        wanted = obj(raw)
        method = enum(wanted.get("method"), "reference method", ("patch", "uniform", "field"), default="patch")
        ref = wanted.get("reference")
        dark = wanted.get("dark")
        if method in ("uniform", "field") and not ref:
            raise DeskError("Choose an incident-light photograph for this mode.")
        from .contracts import region_box
        return cls(
            method,
            identifier(ref, "incident reference") if ref else None,
            identifier(dark, "dark reference") if dark else None,
            number(wanted.get("minimum_span"), "light range floor", .015, .65, default=.12),
            analysis_box(wanted), region_box(wanted.get("reference_box"), "reference patch"),
            number(wanted.get("uncertainty"), "maximum support variation", .008, .55, default=.085),
            integer(wanted.get("columns"), "columns", 1, 12, default=4),
            integer(wanted.get("rows"), "rows", 1, 12, default=3),
        )

    def json(self):
        return {
            "method": self.method, "reference": self.reference, "dark": self.black,
            "minimum_span": self.floor, "box": self.window, "reference_box": self.tie,
            "uncertainty": self.quality_limit, "columns": self.columns, "rows": self.rows,
        }


def sample_status(surface, patch):
    support = patch & surface.valid & ~surface.saturated & ~surface.dark
    return support, int(support.sum())


def imaged_reference(query, capture, expected):
    item = query.one(capture, "captures")
    if item["role"] != expected:
        raise DeskError(f"Use a photograph with the {expected} lighting role.")
    surface = open_surface(query, item)
    return surface


def robust_unit_samples(image, patch, *, darkness=False):
    requested = domain_mask(image.extent, patch)
    status = requested & image.valid & ~image.saturated
    if not darkness:
        status &= ~image.dark
    complete = int(np.count_nonzero(status))
    support_limit(int(requested.sum()), complete, "Light reference", minimum=40)
    signal = luminance(image.linear)[status]
    level = float(np.median(signal))
    noise = 1.4826 * float(np.median(np.abs(signal - level)))
    return level, noise, complete


def lerp_reference(source, extent, patch, *, darkness=False):
    width, height = extent
    x0, y0, x1, y1 = bounds_px(patch, source.extent)
    if min(x1 - x0, y1 - y0) < 8:
        raise DeskError("The field reference area is too small.")
    valid = source.valid[y0:y1, x0:x1] & ~source.saturated[y0:y1, x0:x1]
    if not darkness:
        valid &= ~source.dark[y0:y1, x0:x1]
    values = luminance(source.linear[y0:y1, x0:x1])
    out_y = np.clip(y0 + (np.arange(height) + .5) * ((y1 - y0) / height) - .5, y0, y1 - 1) - y0
    out_x = np.clip(x0 + (np.arange(width) + .5) * ((x1 - x0) / width) - .5, x0, x1 - 1) - x0
    rows, cols = np.meshgrid(out_y, out_x, indexing="ij")
    coords = (rows, cols)
    weight = ndimage.map_coordinates(valid.astype(np.float32), coords, order=1, mode="nearest")
    signal = ndimage.map_coordinates(values * valid, coords, order=1, mode="nearest")
    baseline = np.divide(signal, weight, out=np.zeros_like(signal), where=weight >= .97)
    return baseline, weight >= .97


def reference_values(ctx, query, options):
    links = []
    if options.reference:
        light = imaged_reference(query, options.reference, "incident")
        if light.frame["parent"] != ctx.window["id"]:
            raise DeskError("Choose an incident reference from this window.")
        links.append((light.frame, "incident-light record"))
    else:
        light = ctx.surface
    black = None
    if options.black:
        black = imaged_reference(query, options.black, "dark")
        if black.frame["parent"] != ctx.window["id"]:
            raise DeskError("Choose a dark reference from this window.")
        links.append((black.frame, "dark-field record"))
    light_level, light_scatter, count = robust_unit_samples(light, options.tie)
    dark_level, dark_scatter, dc = robust_unit_samples(black, None, darkness=True) if black else (0.0, 0.0, 0)
    if light_level - dark_level < options.floor:
        raise DeskError("The incident and dark references are too close in intensity.")
    uncertainty = (light_scatter, dark_scatter)
    details = {
        "incident_level": round(light_level, 6),
        "dark_level": round(dark_level, 6),
        "reference_count": count,
        "dark_count": dc,
        "reference_image": light.frame["id"],
        "dark_image": black.frame["id"] if black else None,
        "normalization": options.method,
        "incident_mad": round(light_scatter, 6),
        "dark_mad": round(dark_scatter, 6),
    }
    if options.method != "field":
        usable = np.ones(ctx.surface.valid.shape, dtype=bool)
        return light_level, dark_level, usable, uncertainty, details, links
    extent = ctx.surface.extent
    bright, bsupport = lerp_reference(light, extent, options.tie)
    if black:
        shadow, dsupport = lerp_reference(black, extent, None, darkness=True)
    else:
        shadow, dsupport = dark_level, np.ones_like(bsupport)
    return bright, shadow, bsupport & dsupport, uncertainty, details, links


def photometric_values(ctx, query, options):
    bright, black, ref_valid, noise, details, links = reference_values(ctx, query, options)
    accepted = ctx.usable & ~ctx.surface.saturated
    request = domain_mask(ctx.surface.extent, options.window)
    observed = luminance(ctx.surface.linear)
    delta = bright - black
    channel = np.asarray(np.broadcast_to(delta, observed.shape), dtype=np.float32)
    support_limit(int(request.sum()), int((accepted & request).sum()), "Transmitted light", minimum=80)
    accepted &= request & ref_valid & (channel >= options.floor)
    near_black = request & ref_valid & (observed < np.asarray(black))
    brightness = np.divide(observed - black, channel, out=np.zeros_like(observed), where=accepted)
    uncertainty = np.divide(
        np.hypot(float(noise[1]), float(noise[0]) * np.clip(brightness, 0, 1)),
        channel, out=np.zeros_like(observed), where=accepted,
    )
    uncertain = accepted & (uncertainty > options.quality_limit)
    accepted &= ~uncertain
    return {
        "brightness": brightness, "support": accepted,
        "unknown": request & ~accepted, "request": request,
        "over_reference": accepted & (brightness > 1.02),
        "below_dark": near_black & accepted, "uncertain": uncertain,
        "error": uncertainty, "reference_details": details, "links": links,
    }


def stat_strip(values, support, unit):
    out = {"support_pixels": int(np.count_nonzero(support)), "unit": unit}
    if out["support_pixels"]:
        out["distribution_p05_p50_p95"] = percentile(values[support])
        out["median"] = round(float(np.median(values[support])), 6)
    else:
        out["distribution_p05_p50_p95"] = [None, None, None]
        out["median"] = None
    return out


def transmission_cells(result, extent, options):
    rows = []
    for r, c, (x0, y0, x1, y1) in grid_bounds(*extent, options.columns, options.rows):
        pos = np.s_[y0:y1, x0:x1]
        sample = stat_strip(result["brightness"][pos], result["support"][pos], "fraction")
        wanted = int(np.count_nonzero(result["request"][pos]))
        sample.update(row=r + 1, column=c + 1, box=[x0, y0, x1, y1], requested=wanted)
        sample["unknown"] = wanted - sample["support_pixels"]
        rows.append(sample)
    return rows


def transmission_report(ctx, query, raw):
    if ctx.capture["role"] != "transmitted":
        raise DeskError("Choose a transmitted-light capture.")
    options = TransmissionSettings.parse(raw)
    result = photometric_values(ctx, query, options)
    valid = result["support"]
    if np.count_nonzero(valid) < 40:
        raise DeskError("No useful calibrated region remains.")
    values = result["brightness"]
    density = -np.log10(np.maximum(values, .001))
    cells = transmission_cells(result, ctx.surface.extent, options)
    report = {
        "settings": options.json(), "reference": result["reference_details"],
        "transmission": stat_strip(values, valid, "fraction of incident reading"),
        "density": stat_strip(density, valid, "optical density"),
        "unknown_pixels": int(result["unknown"].sum()),
        "photometric_uncertainty": percentile(result["error"][valid]),
        "above_reference_pixels": int(result["over_reference"].sum()),
        "below_dark_pixels": int(result["below_dark"].sum()),
        "cells": cells,
        "note": "Use the recorded lighting reference. Pixel fractions are relative image estimates, not laboratory transmittance.",
    }
    picture = map_colors(ctx.surface, values, valid, (.02, 1.), unknown=result["unknown"], highlight=result["over_reference"])
    return report, {"map": legend_band(picture, "Transmitted-light ratio", [
        ("Higher transmission", (48, 181, 165)), ("Insufficient reference", (127, 129, 156)),
        ("Above reference", (248, 150, 95)),
    ])}, result["links"]


def contrast_per_cell(ctx, other, grid, tolerance):
    primary, comparison = ctx.surface, other.surface
    common = ctx.usable & other.usable & ~primary.saturated & ~comparison.saturated
    a, b = luminance(primary.linear), luminance(comparison.linear)
    base_a, support_a = masked_gaussian(a, common, 4.0)
    base_b, support_b = masked_gaussian(b, common, 4.0)
    accepted = support_a & support_b
    ga, gb = np.abs(a - base_a), np.abs(b - base_b)
    ratio = (gb - ga) / np.maximum(.025, ga + gb)
    report = []
    for row, col, (x0, y0, x1, y1) in grid:
        pos = np.s_[y0:y1, x0:x1]
        allowance = accepted[pos]
        n = int(allowance.sum())
        strong = np.count_nonzero(np.abs(ratio[pos][allowance]) > tolerance)
        report.append({
            "row": row + 1, "column": col + 1, "bounds": [x0, y0, x1, y1],
            "paired": n,
            "near_surface_trend": round(float(np.median(ratio[pos][allowance])), 5) if n else None,
            "strong_difference_pixels": int(strong),
            "unmeasured": int(allowance.size - n),
        })
    return report, ratio, accepted


def paired_surface_detail(ctx, query, raw):
    if not ctx.alignment:
        raise DeskError("Align the light photographs before comparing detail.")
    other = capture_context(query, identifier(raw.get("comparison"), "second light photograph"), registered=True)
    if other.window["id"] != ctx.window["id"] or other.capture["id"] == ctx.capture["id"]:
        raise DeskError("Choose a second registered capture of this window.")
    if other.surface.extent != ctx.surface.extent:
        raise DeskError("The registered plans have different dimensions.")
    controls = obj(raw)
    pieces = integer(controls.get("grid"), "cell density", 2, 12, default=4)
    tolerance = number(controls.get("disagreement"), "detail disagreement", .02, .8, default=.37)
    rows, values, supported = contrast_per_cell(ctx, other, grid_bounds(*ctx.surface.extent, pieces, pieces), tolerance)
    support_limit(supported.size, np.count_nonzero(supported), "Paired lighting detail", minimum=80)
    from .visuals import signed_map
    art = signed_map(ctx.surface, values, supported, tolerance * 2, ~supported)
    report = {"paired_detail": rows, "strong_disagreement": tolerance, "second_role": other.capture["role"],
              "median_change": round(float(np.median(values[supported])), 6),
              "note": "Contrast across the two photographed lighting modes. Different contrast is not a material identity."}
    references = [(other.capture, "second lighting capture"), (other.alignment, "second lighting geometry")]
    return report, {"map": legend_band(art, "Surface-detail difference", [
        ("Stronger on second image", (244, 163, 102)), ("Stronger on first image", (92, 147, 212)),
        ("Insufficient common detail", (127, 129, 156)),
    ])}, references
