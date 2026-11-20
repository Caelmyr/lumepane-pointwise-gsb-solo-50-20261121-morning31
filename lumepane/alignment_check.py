"""Evaluate alignment on independently marked landmarks and documented image support."""

from collections import Counter
from dataclasses import dataclass
from io import StringIO
import csv

import numpy as np
from PIL import Image, ImageDraw

from .catalog import capture_context
from .contracts import DeskError, collection, integer, number, obj, point, word
from .geometry import bounds_px, transform_points
from .imaging import canvas_picture, encode, font, resize_rgba
from .visuals import INK, PALE, QUIET, draw_bars, legend_band

PASS = "#6eb69e"
REVIEW = "#e6ab72"
UNCERTAIN = "#97a1be"


@dataclass(frozen=True)
class CheckLocation:
    name: str
    expected: tuple
    observed: tuple

    @classmethod
    def parse(cls, raw, index):
        item = obj(raw, "alignment landmark")
        return cls(
            word(item.get("name"), "landmark name", default=f"Landmark {index + 1}", limit=80),
            tuple(point(item.get("plane"), "location in window plane")),
            tuple(point(item.get("capture"), "location on original photo")),
        )


@dataclass(frozen=True)
class CheckControls:
    locations: tuple
    threshold_px: float
    support_floor: float
    rows: int
    columns: int
    title: str

    @classmethod
    def parse(cls, raw):
        form = obj(raw, "alignment check")
        locations = collection(form.get("landmarks"), "held-out landmarks", 2, 24)
        markers = tuple(CheckLocation.parse(item, i) for i, item in enumerate(locations))
        names = [item.name.casefold() for item in markers]
        if len(set(names)) != len(names):
            raise DeskError("Name each check location independently.")
        positions = {item.expected for item in markers}
        if len(positions) != len(markers):
            raise DeskError("Give each held-out landmark a distinct position on the plane.")
        return cls(
            markers,
            number(form.get("threshold"), "landmark tolerance in pixels", .25, 100, default=4),
            number(form.get("support_floor"), "cell support floor", .05, 1, default=.80),
            integer(form.get("rows"), "support grid rows", 2, 14, default=3),
            integer(form.get("columns"), "support grid columns", 2, 14, default=4),
            word(form.get("title"), "check title", default="Alignment landmark check", limit=100),
        )


def extent_coords(point, extent):
    return np.asarray(point, dtype=np.float64) * np.asarray(extent) - .5


def acceptable(surface, coords):
    x, y = np.rint(coords).astype(int)
    return (
        0 <= x < surface.extent[0]
        and 0 <= y < surface.extent[1]
        and surface.valid[y, x]
    )


def distance_fields(delta, inverse_delta, scale):
    doc = {
        "dx_source_px": round(float(delta[0]), 4),
        "dy_source_px": round(float(delta[1]), 4),
        "source_pixel_error": round(float(np.linalg.norm(delta)), 5),
        "plane_pixel_error": round(float(np.linalg.norm(inverse_delta)), 5),
    }
    if scale is not None:
        doc["plane_mm_error"] = round(float(np.linalg.norm(inverse_delta * scale)), 5)
    return doc


def location_audit(details, plane, original):
    matrix = np.asarray(plane.alignment["matrix"])
    inv = np.linalg.inv(matrix)
    intended = [item.expected for item in details.locations]
    marked = [item.observed for item in details.locations]
    expected_source = transform_points(matrix, intended)
    expected_plane = transform_points(inv, marked)
    rows = []
    for idx, location in enumerate(details.locations):
        at_original = extent_coords(location.observed, original.surface.extent)
        at_plane = extent_coords(location.expected, plane.surface.extent)
        to_original = extent_coords(expected_source[idx], original.surface.extent)
        to_plane = extent_coords(expected_plane[idx], plane.surface.extent)
        fields = distance_fields(at_original - to_original, to_plane - at_plane, plane.scale)
        supported = acceptable(original.surface, at_original) and acceptable(original.surface, to_original)
        accepted = bool(supported and fields["source_pixel_error"] <= details.threshold_px)
        rows.append({
            "name": location.name, "window_nominal": list(location.expected), "capture_observed": list(location.observed),
            "capture_predicted": expected_source[idx].round(8).tolist(),
            "window_predicted": expected_plane[idx].round(8).tolist(),
            "supported": bool(supported),
            "classification": "within tolerance" if accepted else "review residual" if supported else "unobserved source",
            **fields,
        })
    return rows


def supports_for_grid(details, plane):
    usable = plane.surface.valid
    height, width = usable.shape
    row_borders = np.linspace(0, height, details.rows + 1).astype(int)
    col_borders = np.linspace(0, width, details.columns + 1).astype(int)
    rows = []
    for row, (y0, y1) in enumerate(zip(row_borders[:-1], row_borders[1:]), start=1):
        for col, (x0, x1) in enumerate(zip(col_borders[:-1], col_borders[1:]), start=1):
            interior = usable[y0:y1, x0:x1]
            valid = int(interior.sum())
            fraction = valid / interior.size
            rows.append({
                "row": row, "column": col, "window_box_px": [int(v) for v in (x0, y0, x1, y1)],
                "photographed_px": valid, "region_px": int(interior.size),
                "photographed_fraction": round(fraction, 5),
                "classification": "review support" if fraction < details.support_floor else "photographed",
            })
    return rows


def quantiles(locations, key):
    values = [item[key] for item in locations if item["supported"] and key in item]
    if not values:
        return {"maximum": None, "median": None, "rms": None}
    return {
        "maximum": round(max(values), 5), "median": round(float(np.median(values)), 5),
        "rms": round(float(np.sqrt(np.mean(np.square(values)))), 5),
    }


def coordinate_distribution(locations):
    zones = {(pos[0] >= .5, pos[1] >= .5) for pos in (item["window_nominal"] for item in locations if item["supported"])}
    arms = [name for (right, lower), name in [
        ((False, False), "upper left"), ((True, False), "upper right"),
        ((False, True), "lower left"), ((True, True), "lower right")
    ] if (right, lower) in zones]
    return {"checked_quadrants": arms, "quadrant_count": len(zones)}


def audit_summary(details, locations, support):
    groups = Counter(item["classification"] for item in locations)
    distribution = coordinate_distribution(locations)
    return {
        "checked_landmarks": len(locations),
        "supported_landmarks": groups["within tolerance"] + groups["review residual"],
        "within_tolerance": groups["within tolerance"],
        "exceeding_tolerance": groups["review residual"],
        "unobserved_landmarks": groups["unobserved source"],
        "poorly_observed_cells": sum(item["classification"] == "review support" for item in support),
        "threshold_source_px": details.threshold_px,
        "minimum_cell_support": details.support_floor,
        "grid_rows": details.rows,
        "grid_columns": details.columns,
        "source_pixel_error": quantiles(locations, "source_pixel_error"),
        "plane_pixel_error": quantiles(locations, "plane_pixel_error"),
        "plane_mm_error": quantiles(locations, "plane_mm_error"),
        **distribution,
    }


def map_landmarks(original, locations):
    base = resize_rgba(canvas_picture(original.surface), 1000).convert("RGB")
    overlay = ImageDraw.Draw(base)
    for index, spot in enumerate(locations, start=1):
        measured = extent_coords(spot["capture_observed"], base.size)
        predicted = extent_coords(spot["capture_predicted"], base.size)
        color = UNCERTAIN if not spot["supported"] else PASS if spot["classification"] == "within tolerance" else REVIEW
        overlay.line([tuple(measured), tuple(predicted)], fill=color, width=3)
        x, y = measured
        overlay.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="#1f2226", width=2)
        x, y = predicted
        overlay.line((x - 6, y, x + 6, y), fill="#eee9d4", width=2)
        overlay.line((x, y - 6, x, y + 6), fill="#eee9d4", width=2)
        tx, ty = measured + (7, -13)
        overlay.text((tx, ty), str(index), fill="#f7f1e3", font=font(13, bold=True), stroke_width=2, stroke_fill="#212222")
    return legend_band(base, "Held-out landmarks on original", [("Within tolerance", PASS), ("Review residual", REVIEW),
                                                            ("No usable source", UNCERTAIN)])


def support_grid_image(plane, rows):
    base = resize_rgba(canvas_picture(plane.surface), 1000).convert("RGB")
    overlay = ImageDraw.Draw(base)
    scale = np.asarray(base.size) / np.asarray(plane.surface.extent)
    for cell in rows:
        x0, y0, x1, y1 = np.asarray(cell["window_box_px"]) * np.tile(scale, 2)
        color = REVIEW if cell["classification"] == "review support" else PASS
        overlay.rectangle((x0, y0, x1, y1), outline=color, width=2)
        text = f'{cell["row"]}.{cell["column"]}    {cell["photographed_fraction"]:.0%}'
        overlay.text((x0 + 6, y0 + 6), text, font=font(11), fill="#f7f1e3", stroke_width=2, stroke_fill="#232423")
    return legend_band(base, "Registered photograph support", [("Below cell support floor", REVIEW), ("Photographed", PASS)])


def comparison_card(summary, details):
    image = Image.new("RGB", (630, 290), PALE)
    ink = ImageDraw.Draw(image)
    ink.text((20, 15), details.title[:65], font=font(20, bold=True), fill=INK)
    values = [
        ("Supported landmarks", summary["supported_landmarks"]),
        ("Within tolerance", summary["within_tolerance"]),
        ("Exceeding tolerance", summary["exceeding_tolerance"]),
        ("Original median error", summary["source_pixel_error"]["median"]),
        ("Plane median error", summary["plane_pixel_error"]["median"]),
        ("Poorly observed cells", summary["poorly_observed_cells"]),
        ("Covered quadrants", summary["quadrant_count"]),
    ]
    for i, (label, number) in enumerate(values):
        y = 56 + 28 * i
        ink.text((21, y), label, font=font(14), fill=INK)
        cell = "Unavailable" if number is None else str(number)
        ink.text((315, y), cell, font=font(14, bold=True), fill=QUIET)
    return image


def record_data(controls, plane, original):
    rows = location_audit(controls, plane, original)
    support = supports_for_grid(controls, plane)
    summary = audit_summary(controls, rows, support)
    return {
        "title": controls.title, "method": "held-out registration landmark check",
        "capture": original.capture["id"], "capture_version": original.capture["version"],
        "alignment": plane.alignment["id"], "alignment_version": plane.alignment["version"],
        "window_version": plane.window["version"], "original_sha256": original.capture["original_digest"],
        "synthetic": original.capture["synthetic"], "reading": summary, "landmarks": rows, "support": support,
        "interpretation": "Residuals evaluate the recorded alignment. The entered points do not move the registered plane.",
    }, {
        "map": map_landmarks(original, rows), "support": support_grid_image(plane, support),
        "chart": comparison_card(summary, controls),
    }


def alignment_check(library, key, raw, *, save):
    controls = CheckControls.parse(raw)
    with library.use() as db:
        plane = capture_context(db, key, registered=True)
        native = capture_context(db, key, registered=False)
        values, pictures = record_data(controls, plane, native)
    if not save:
        return pictures["map"]
    with library.use(write=True) as db:
        result = db.add("alignment_checks", values, parent=plane.window["id"], action="checked recorded alignment on held-out points")
        for content, role in [(plane.alignment, "checked alignment"), (native.capture, "checked original capture"), (plane.window, "recorded window")]:
            db.link(result["id"], content, role)
        for tag, picture in pictures.items():
            db.attach(result["id"], tag, encode(picture), "image/png")
        return db.public(result)


def alignment_history(library, window):
    with library.use() as db:
        db.one(window, "windows")
        values = db.list("alignment_checks", parent=window, limit=100)
    return {"checks": values}


def audit_csv(result):
    out = StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(("Landmark", "Classification", "Original x error (px)", "Original y error (px)",
                     "Original distance (px)", "Plane distance (px)", "Plane distance (mm)"))
    for item in result["landmarks"]:
        writer.writerow((item["name"], item["classification"], item["dx_source_px"], item["dy_source_px"],
                         item["source_pixel_error"], item["plane_pixel_error"], item.get("plane_mm_error")))
    writer.writerow(())
    writer.writerow(("Row", "Column", "Observability", "Support", "Total pixels"))
    for item in result["support"]:
        writer.writerow((item["row"], item["column"], item["classification"], item["photographed_px"], item["region_px"]))
    return out.getvalue().encode()
