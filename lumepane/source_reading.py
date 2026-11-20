"""Describe a single immutable photograph's decoding and recorded visibility."""

from collections import Counter
from io import StringIO
import csv

import numpy as np

from .catalog import capture_context
from .exposure import percentile
from .imaging import luminance
from .intake import spreadsheet_cell


def counted_support(surface):
    valid = surface.valid
    clip = surface.saturated & valid
    dark = surface.dark & valid & ~clip
    clean = valid & ~dark & ~clip
    total = int(valid.size)
    frame = {
        "photographed": int(valid.sum()),
        "nonphotographed": int(total - valid.sum()),
        "overexposed": int(clip.sum()),
        "near_dark": int(dark.sum()),
        "usable": int(clean.sum()),
        "total": total,
    }
    percentages = {label: round(value / total * 100, 5) for label, value in frame.items() if label != "total"}
    return frame, percentages, clean


def scan_strips(surface, clean):
    height, width = clean.shape
    lum = luminance(surface.linear)
    runs = []
    edges = np.rint(np.linspace(0, height, 9)).astype(int)
    for index, top in enumerate(edges[:-1]):
        bottom = edges[index + 1]
        band = clean[top:bottom]
        size = band.size
        if not size:
            continue
        raw = lum[top:bottom][band]
        dark = (surface.dark[top:bottom] & surface.valid[top:bottom]).sum()
        runs.append({
            "row": index + 1, "y_start": int(top), "y_end": int(bottom), "width": width,
            "usable_fraction": round(float(band.sum()) / size, 6),
            "near_dark_fraction": round(float(dark) / size, 6),
            "photographed_fraction": round(float(surface.valid[top:bottom].sum()) / size, 6),
            "luminance_p05_p50_p95": percentile(raw),
        })
    return runs


def camera_receipt(context):
    record = context.capture
    metadata = dict(record["camera"])
    display = []
    for field, label in (
        ("make", "Camera make"),
        ("model", "Camera model"),
        ("camera_time", "Camera timestamp"),
        ("orientation", "Source EXIF orientation"),
        ("seconds", "Exposure duration seconds"),
        ("aperture", "Lens f-number"),
        ("iso", "ISO setting"),
    ):
        if field in metadata:
            display.append({"field": label, "value": metadata[field]})
    return display


def size_change(record):
    original_w, original_h = record["source_size"]
    upright_w, upright_h = record["upright_size"]
    working_w, working_h = record["extent"]
    return {
        "stored_width_px": original_w, "stored_height_px": original_h,
        "upright_width_px": upright_w, "upright_height_px": upright_h,
        "working_width_px": working_w, "working_height_px": working_h,
        "analysis_to_upright_width": round(working_w / upright_w, 6),
        "analysis_to_upright_height": round(working_h / upright_h, 6),
        "working_area_to_upright_area": round((working_w * working_h) / (upright_w * upright_h), 6),
    }


def support_cards(context):
    counts, percentages, clean = counted_support(context.surface)
    levels = luminance(context.surface.linear)[clean]
    by_light = scan_strips(context.surface, clean)
    channels = []
    if levels.size:
        pixels = context.surface.linear[clean]
        for index, label in enumerate(("Red", "Green", "Blue")):
            channels.append({"channel": label, "linear_p05_p50_p95": percentile(pixels[:, index])})
    return {
        "totals": counts, "coverage_percentages": percentages,
        "usable_luminance_p05_p50_p95": percentile(levels),
        "bands": by_light, "channels": channels,
    }


def capture_receipt(library, key):
    with library.use() as db:
        ctx = capture_context(db, key, registered=False)
        record = ctx.capture
        source = db.asset(key, "original")
        analysis = db.asset(key, "working")
        bindings = db.audit(key)
    verified = {item["label"]: item["hash_matches"] for item in bindings["assets"]}
    identity = {
        "title": record["caption"], "source_file": record["file_name"],
        "page_number": record["page"], "page_count": record["page_count"],
        "source_mime": source["mime"], "source_bytes": source["bytes"], "source_sha256": source["digest"],
        "working_sha256": analysis["digest"], "source_revision": record["version"],
        "source_ok": verified["original"], "working_ok": verified["working"],
        "drawing_scope": "original photograph before registration", "capture": record["id"],
        "color_profile_handling": record["analysis_space"], "working_depth": record["working_depth"],
        "synthetic": record["synthetic"],
    }
    return {
        "identity": identity, "dimensions": size_change(record), "camera_metadata": camera_receipt(ctx),
        "decoding_notes": record["warnings"], "visibility": support_cards(ctx),
    }


def receipt_csv(result):
    buffer = StringIO(newline="")
    rows = csv.writer(buffer)
    for key in ("identity", "dimensions"):
        rows.writerow((key,))
        for field, value in result[key].items():
            rows.writerow((field, spreadsheet_cell(value) if isinstance(value, str) else value))
        rows.writerow(())
    rows.writerow(("Camera metadata",))
    for item in result["camera_metadata"]:
        rows.writerow((item["field"], spreadsheet_cell(item["value"])))
    rows.writerow(())
    rows.writerow(("Band", "Y start", "Y end", "Usable", "Near dark", "Photographed", "Luma fifth", "Luma median", "Luma 95th"))
    for item in result["visibility"]["bands"]:
        row = [item[name] for name in ("row", "y_start", "y_end", "usable_fraction", "near_dark_fraction", "photographed_fraction")]
        rows.writerow(row + (item["luminance_p05_p50_p95"] or [None, None, None]))
    return buffer.getvalue().encode()
