"""Composite overlapping window segments into one saved, provenance-aware overview.

Each photograph names the quadrilateral it occupies on the window plane. A clear
photographed pixel always wins over an occluded, clipped, near-dark, or unmapped
sample from another photograph, so an obstruction in one segment cannot erase
the same place seen clearly in another. Where two clear samples disagree, both
sources remain recorded and a numbered seam marker is drawn on the composite so
the disagreement can be inspected and traced back to its photographs. Places no
photograph reached stay empty.
"""

from dataclasses import dataclass
from io import StringIO
import csv

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .catalog import MAX_WINDOW_MARKS, capture_context
from .contracts import DeskError, collection, identifier, number, obj, word
from .geometry import (
    CORNERS, fit_transform, normalized_quad, output_plane, polygon_mask, transform_points, warp_surface,
)
from .imaging import encode, font, label, luminance
from .repository import encoded
from .visuals import legend_band

MAX_PANORAMA_CAPTURES = 16
MAX_SEAMS = 24
MIN_SEAM_PIXELS_PER_AREA = 0.00005
PHOTO_ROLES = {"transmitted", "reflected", "rake_left", "rake_right"}
CLEAR, OCCLUDED, CLIPPED, DARK, ABSENT = range(5)
SOURCE_SHADES = (
    (78, 190, 168), (236, 168, 96), (211, 124, 192), (128, 165, 238), (196, 205, 112),
    (238, 128, 128), (116, 214, 214), (186, 156, 238), (222, 194, 122), (134, 190, 134),
    (228, 142, 170), (146, 176, 214), (206, 168, 118), (158, 202, 166), (196, 148, 208),
    (222, 182, 100),
)
VOID = (16, 20, 28)
SEAM_COLOR = (255, 214, 128)


@dataclass(frozen=True)
class Placement:
    capture: str
    quad: list
    stack: int

    @classmethod
    def parse(cls, raw, index):
        data = obj(raw, "segment placement")
        return cls(
            identifier(data.get("capture"), "segment photograph"),
            normalized_quad(data.get("corners")) if data.get("corners") else [list(p) for p in CORNERS],
            index,
        )


def parse_request(raw):
    data = obj(raw, "overview")
    placements = [
        Placement.parse(item, index)
        for index, item in enumerate(collection(data.get("placements"), "segment placements", 1, MAX_PANORAMA_CAPTURES))
    ]
    if len({item.capture for item in placements}) != len(placements):
        raise DeskError("Place each photograph only once in an overview.", field="placements")
    requested = data.get("extent")
    extent = None
    if requested:
        width, height = (int(number(v, "overview size", 240, 2400)) for v in collection(requested, "overview size", 2, 2))
        extent = (width, height)
    return {
        "title": word(data.get("title"), "overview title", default="Whole-window overview", limit=100),
        "extent": extent,
        "grid": int(number(data.get("grid"), "overview grid", 2, 24, default=6)),
        "placements": placements,
        "tolerance": number(data.get("tolerance"), "disagreement tolerance", .004, .5, default=.05),
    }


def overview_extent(window, requested):
    """Use the window's own proportions so segments are not stretched."""
    if requested:
        return requested
    base = output_plane(window, window.get("working_extent") or (1400, 980))
    width, height = base
    scale = min(1., 2000 / max(width, height))
    if scale < 1:
        width, height = int(round(width * scale)), int(round(height * scale))
    return max(240, width), max(240, height)


def segment_inputs(query, window, placements):
    result = []
    for placement in placements:
        photo = query.one(placement.capture, "captures", parent=window["id"])
        if photo["role"] not in PHOTO_ROLES:
            raise DeskError("Only photographs made through the window can join an overview.", field="placements")
        result.append((placement, capture_context(query, photo["id"], registered=False)))
    groups = {(ctx.capture["role"], ctx.capture["visit"], ctx.capture["day"]) for _, ctx in result}
    if len(groups) > 1:
        roles = sorted({role for role, _, _ in groups})
        visits = sorted({visit or "(no inspection label)" for _, visit, _ in groups})
        days = sorted({day or "(no date)" for _, _, day in groups})
        raise DeskError(
            "A combined overview joins one lighting at one inspection visit. "
            f"These photographs are {', '.join(visits)} on {', '.join(days)} under {', '.join(roles)}.",
            field="placements",
        )
    return result


def grade_pixels(surface, excluded):
    """Classify each source pixel: clear support, or why it cannot contribute."""
    grade = np.full(surface.valid.shape, ABSENT, dtype=np.uint8)
    grade[surface.valid] = OCCLUDED
    grade[surface.valid & surface.saturated] = CLIPPED
    grade[surface.valid & ~surface.saturated & surface.dark] = DARK
    grade[surface.valid & ~surface.saturated & ~surface.dark & ~excluded] = CLEAR
    return grade


def occlusion_by_capture(query, window, extent):
    """Marks that name an obstruction in one photograph, mapped onto the window plane.

    A mark is applied only to the photograph it was drawn on. Another segment that
    sees the same place clearly still supplies it, which is the point of combining.
    """
    marks = [
        record for record in query.list("marks", parent=window["id"], limit=MAX_WINDOW_MARKS)
        if not record["retired"] and record["shape"] == "polygon"
        and record["category"] in ("occluded", "ignore") and record.get("source_capture")
    ]
    if not marks:
        return {}
    alignments = {}
    for record in query.list("alignments", parent=window["id"], limit=300):
        alignments.setdefault(record["capture_id"], record)
    masks = {}
    width, height = extent
    for record in marks:
        if record["coordinate_basis"] == "window":
            basis = tuple(record["plane_extent"])
            points, holes = record["points"], record["holes"]
        else:
            alignment = alignments.get(record["source_capture"])
            if not alignment:
                continue
            # The alignment maps window positions into capture positions; a mark drawn
            # on the photograph is brought back through the inverse.
            inverse = np.linalg.inv(np.asarray(alignment["matrix"], dtype=np.float64))
            points = [list(transform_points(inverse, [point])[0]) for point in record["points"]]
            holes = [[list(transform_points(inverse, [point])[0]) for point in hole] for hole in record["holes"]]
            basis = tuple(alignment["plane_extent"])
        scaled = [[x * width / basis[0], y * height / basis[1]] for x, y in points]
        scaled_holes = [[[x * width / basis[0], y * height / basis[1]] for x, y in hole] for hole in holes]
        mask = masks.get(record["source_capture"])
        if mask is None:
            mask = np.zeros((height, width), dtype=bool)
            masks[record["source_capture"]] = mask
        mask |= polygon_mask(extent, scaled, scaled_holes)
    return masks


def warp_grade(matrix, source_extent, extent, grade):
    """Resample the source grade map over the placed quad only; the rest stays empty."""
    width, height = extent
    x = (np.arange(width)[None, :] + .5) / width
    y = ((np.arange(height) + .5) / height)[:, None]
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    sx = ((matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator) * source_extent[0] - .5
    sy = ((matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator) * source_extent[1] - .5
    inside = (sx >= -.5) & (sy >= -.5) & (sx <= source_extent[0] - .5) & (sy <= source_extent[1] - .5)
    sampled = ndimage.map_coordinates(grade.astype(np.float32), (sy, sx), order=0, mode="nearest")
    result = np.full((height, width), ABSENT, dtype=np.uint8)
    result[inside] = np.rint(sampled[inside]).astype(np.uint8)
    return result


@dataclass
class Composite:
    colors: np.ndarray
    owner: np.ndarray
    grade: np.ndarray
    contributions: list
    conflicts: list

    @property
    def extent(self):
        return self.colors.shape[1], self.colors.shape[0]


def composite_segments(segments, extent, tolerance):
    """Stack placed segments; a clear sample claims its place once and is never erased."""
    width, height = extent
    colors = np.zeros((height, width, 3), dtype=np.float32)
    owner = np.full((height, width), -1, dtype=np.int16)
    grade = np.full((height, width), ABSENT, dtype=np.uint8)
    contributions = [{
        "capture": placement.capture, "stack": placement.stack,
        "placed_clear_pixels": int(np.count_nonzero(graded == CLEAR)),
        "contributed_pixels": 0, "skipped_for_earlier_clear": 0,
        "filled_fallback_pixels": 0, "conflict_pixels": 0,
    } for placement, _, graded in segments]
    conflicts = []
    for index, (placement, surface, graded) in enumerate(segments):
        clear = graded == CLEAR
        present = graded != ABSENT
        claimed = grade == CLEAR
        accepted = clear & ~claimed
        colors[accepted] = surface.linear[accepted]
        owner[accepted] = index
        grade[accepted] = CLEAR
        if np.any(claimed):
            overlap = clear & claimed
            delta = np.abs(luminance(surface.linear) - luminance(colors))
            # Smooth before thresholding so per-pixel sensor and mosaic noise does not
            # read as a disagreement; only a coherent region becomes a seam.
            smooth = ndimage.gaussian_filter(np.where(overlap, delta, 0), 2.2)
            weight = ndimage.gaussian_filter(overlap.astype(np.float32), 2.2)
            coherent = np.divide(smooth, weight, out=np.zeros_like(smooth), where=weight > .35)
            disagreement = overlap & (weight > .35) & (coherent > tolerance)
            count = int(np.count_nonzero(disagreement))
            contributions[index]["conflict_pixels"] = count
            contributions[index]["skipped_for_earlier_clear"] = int(np.count_nonzero(overlap & ~disagreement))
            if count:
                conflicts.append((index, np.argwhere(disagreement), coherent[disagreement]))
        # A less clear but photographed sample may fill remaining empty space; a later
        # clear sample still wins over it because only clear claims block a clear sample.
        fallback = present & ~clear & (grade == ABSENT)
        colors[fallback] = surface.linear[fallback]
        owner[fallback] = index
        grade[fallback] = graded[fallback]
        contributions[index]["filled_fallback_pixels"] = int(np.count_nonzero(fallback))
        contributions[index]["contributed_pixels"] = int(np.count_nonzero(accepted) + np.count_nonzero(fallback))
    return Composite(colors, owner, grade, contributions, conflicts)


def coverage_grid(grade, owner, sources, cells):
    height, width = grade.shape
    steps_x = np.rint(np.linspace(0, width, cells + 1)).astype(int)
    steps_y = np.rint(np.linspace(0, height, cells + 1)).astype(int)
    rows = []
    for r in range(cells):
        for c in range(cells):
            box = (int(steps_x[c]), int(steps_y[r]), int(steps_x[c + 1]), int(steps_y[r + 1]))
            x0, y0, x1, y1 = box
            piece = grade[y0:y1, x0:x1]
            total = int(piece.size)
            clear = int(np.count_nonzero(piece == CLEAR))
            filled = int(np.count_nonzero(piece != ABSENT))
            taken = owner[y0:y1, x0:x1]
            shares = [
                {"capture": sources[index]["capture"], "label": "ABCDEFGHIJKLMNOP"[index],
                 "clear_pixels": int(np.count_nonzero((taken == index) & (piece == CLEAR)))}
                for index in range(len(sources)) if np.any((taken == index) & (piece == CLEAR))
            ]
            rows.append({
                "row": r + 1, "column": c + 1, "bounds": list(box),
                "clear_pixels": clear, "filled_pixels": filled, "empty_pixels": int(total - filled),
                "clear_fraction": round(clear / total, 6) if total else None,
                "source_shares": shares,
                "state": "empty" if not filled else "partial" if clear / max(1, total) < .55 else "covered",
            })
    return rows


def seam_records(conflicts, owner, placements, extent):
    """Group disagreeing pixels into numbered regions kept in the saved record.

    Only regions large enough to inspect are numbered, so isolated noise between
    two photographs does not bury a genuine seam in hundreds of tiny markers.
    """
    area = int(extent[0]) * int(extent[1])
    floor = max(60, int(area * MIN_SEAM_PIXELS_PER_AREA))
    bridge = max(2, round(min(extent) / 110))
    records = []
    for later, pixels, deltas in conflicts:
        if not len(pixels):
            continue
        mask = np.zeros(owner.shape, dtype=bool)
        mask[pixels[:, 0], pixels[:, 1]] = True
        # Bridge the thin came and frame lines that cross a seam so one disagreement
        # is reported as one marker rather than several fragments.
        grouped, _ = ndimage.label(ndimage.binary_dilation(mask, iterations=bridge), structure=np.ones((3, 3)))
        regions = grouped[pixels[:, 0], pixels[:, 1]]
        eligible = [
            int(value) for value in np.unique(regions)
            if value and np.count_nonzero(regions == value) >= floor
        ]
        if len(eligible) > MAX_SEAMS:
            eligible = sorted(eligible, key=lambda value: np.count_nonzero(regions == value), reverse=True)[:MAX_SEAMS]
        for region in sorted(eligible):
            chosen = regions == region
            spots = pixels[chosen]
            earlier = int(owner[spots[0, 0], spots[0, 1]])
            records.append({
                "number": len(records) + 1, "later_segment": later,
                "later_capture": placements[later].capture,
                "earlier_capture": placements[earlier].capture if earlier >= 0 else None,
                "bounds": [int(spots[:, 1].min()), int(spots[:, 0].min()),
                           int(spots[:, 1].max()) + 1, int(spots[:, 0].max()) + 1],
                "pixels": int(len(spots)), "median_delta": round(float(np.median(deltas[chosen])), 6),
            })
    return records


def placement_frame(window):
    """A window-plane guide on which each segment's corners are marked."""
    width, height = overview_extent(window, None)
    image = Image.new("RGB", (width, height), (34, 40, 52))
    pen = ImageDraw.Draw(image)
    step = max(24, round(max(width, height) / 12))
    for x in range(step, width, step):
        pen.line((x, 0, x, height), fill=(66, 74, 88))
    for y in range(step, height, step):
        pen.line((0, y, width, y), fill=(66, 74, 88))
    pen.rectangle((1, 1, width - 2, height - 2), outline=(186, 158, 108), width=2)
    pen.text((12, 10), "Window plane · mark each photographic segment here", fill="#f0e6cf", font=font(15, bold=True))
    pen.text((12, height - 26), f"{width} × {height} overview pixels · corners clockwise from the upper left", fill="#ded7c6", font=font(12))
    return image


def placement_guide(library, window_id):
    with library.use() as db:
        window = db.one(window_id, "windows")
    return placement_frame(window)


def source_plan(result, sources):
    """Colour each clear area by the photograph that supplies it; label unusable areas."""
    palette = np.array([VOID, (58, 52, 40), (60, 38, 38), (30, 36, 58), VOID], dtype=np.float32)
    colors = np.zeros((*result.grade.shape, 3), dtype=np.float32)
    for index in range(len(sources)):
        colors[result.owner == index] = SOURCE_SHADES[index % len(SOURCE_SHADES)]
    for index in (OCCLUDED, CLIPPED, DARK, ABSENT):
        colors[result.grade == index] = palette[index]
    image = Image.fromarray(np.uint8(colors), "RGB")
    entries = [(source["caption"], tuple(int(v) for v in SOURCE_SHADES[index % len(SOURCE_SHADES)]))
               for index, source in enumerate(sources)]
    return legend_band(image, "Photograph supplying each clear area", entries + [
        ("Occluded only", tuple(int(v) for v in palette[OCCLUDED])),
        ("Clipped only", tuple(int(v) for v in palette[CLIPPED])),
        ("Dark only", tuple(int(v) for v in palette[DARK])),
        ("Not photographed", VOID),
    ])


def overview_picture(composite, seams, title, note, sources):
    """Paint clear sampling, mark unusable-only areas, number disagreements, and key sources."""
    base = np.uint8(np.clip(composite.colors, 0, 1) * 255)
    empty = composite.grade == ABSENT
    fallback = ~empty & (composite.grade != CLEAR)
    tinted = base.astype(np.float32)
    tinted[fallback] = tinted[fallback] * .38 + np.asarray((122, 116, 134), dtype=np.float32) * .62
    tinted[empty] = VOID
    image = Image.fromarray(np.uint8(np.clip(tinted, 0, 255)), "RGB")
    if np.any(fallback):
        stripes = Image.new("L", image.size, 0)
        pen = ImageDraw.Draw(stripes)
        for offset in range(-image.height, image.width, 11):
            pen.line((offset, 0, offset + image.height, image.height), fill=255, width=3)
        hatch = Image.fromarray((np.asarray(stripes) & (fallback * 255).astype(np.uint8)), "L")
        image.paste(Image.new("RGB", image.size, (176, 165, 184)), (0, 0), hatch.point(lambda v: 150 if v else 0))
    pen = ImageDraw.Draw(image)
    width, height = image.size
    for seam in seams:
        x0, y0, x1, y1 = seam["bounds"]
        pen.rectangle((max(0, x0 - 1), max(0, y0 - 1), min(width - 1, x1), min(height - 1, y1)),
                      outline=SEAM_COLOR, width=max(1, image.width // 480))
        label(pen, (min(width - 26, x0 + 2), max(1, y0 + 1)), str(seam["number"]), size=12, fill=SEAM_COLOR)
    rows = 3
    band = Image.new("RGB", (image.width, image.height + 46 + 19 * rows), (26, 33, 46))
    band.paste(image, (0, 46))
    pen = ImageDraw.Draw(band)
    pen.text((12, 9), title[:80], fill="#fcf4dc", font=font(17, bold=True))
    key = "  ·  ".join(f'{letter} {source["caption"][:22]}' for letter, source in zip("ABCDEFGHIJKLMNOP", sources))
    lines = [
        note[:120],
        "Key: " + key[:170],
        "Hatched glass is photographed only through an occlusion, clip, or shadow. Black was not photographed. Yellow frames are seam markers.",
    ]
    for index, text in enumerate(lines):
        pen.text((12, image.height + 52 + index * 19), text[:185], fill="#dce0e7", font=font(11))
    return band


def compose_overview(library, window_id, raw, *, preview=False):
    request = parse_request(raw)
    with library.use() as query:
        window = query.one(window_id, "windows")
        extent = overview_extent(window, request["extent"])
        loaded = segment_inputs(query, window, request["placements"])
        plane_occluded = occlusion_by_capture(query, window, extent)
        segments = []
        for placement, ctx in loaded:
            grade = grade_pixels(ctx.surface, ctx.excluded)
            matrix = fit_transform(placement.quad, list(CORNERS))
            surface = warp_surface(ctx.surface, matrix, extent)
            placed = warp_grade(matrix, ctx.surface.extent, extent, grade)
            placed[~surface.valid] = ABSENT
            own_marks = plane_occluded.get(ctx.capture["id"])
            if own_marks is not None:
                placed[own_marks & (placed == CLEAR)] = OCCLUDED
            segments.append((placement, surface, placed, ctx))
        built = composite_segments([(p, s, g) for p, s, g, _ in segments], extent, request["tolerance"])
        seams = seam_records(built.conflicts, built.owner, request["placements"], extent)
        sources = [{
            "capture": ctx.capture["id"], "caption": ctx.capture["caption"], "role": ctx.capture["role"],
            "visit": ctx.capture["visit"], "day": ctx.capture["day"],
            "capture_version": ctx.capture["version"], "source_sha256": ctx.capture["original_digest"],
            "stack": placement.stack, "corners": placement.quad,
            "contribution": built.contributions[index],
        } for index, (placement, _, _, ctx) in enumerate(segments)]
        fingers = {source["capture"]: "ABCDEFGHIJKLMNOP"[index] for index, source in enumerate(sources)}
        for seam in seams:
            seam["later_label"] = fingers[seam["later_capture"]]
            seam["earlier_label"] = fingers.get(seam["earlier_capture"], "?")
            seam["later_caption"] = sources[seam["later_segment"]]["caption"]
            seam["earlier_caption"] = next(
                (source["caption"] for source in sources if source["capture"] == seam["earlier_capture"]), "")
        record = {
            "title": request["title"], "extent": list(extent),
            "tolerance": request["tolerance"], "grid": request["grid"],
            "lighting": segments[0][3].capture["role"], "visit": segments[0][3].capture["visit"],
            "sources": sources, "seams": seams,
            "cells": coverage_grid(built.grade, built.owner, sources, request["grid"]),
            "reading": {
                "photograph_count": len(sources),
                "clear_pixels": int(np.count_nonzero(built.grade == CLEAR)),
                "empty_pixels": int(np.count_nonzero(built.grade == ABSENT)),
                "filled_percent": round(float(100 * (1 - np.count_nonzero(built.grade == ABSENT) / built.grade.size)), 3),
                "seam_count": len(seams), "seam_pixels": int(sum(item["pixels"] for item in seams)),
                "conflict_pixels": int(sum(item["conflict_pixels"] for item in built.contributions)),
                "photographs_contributing": sum(1 for item in built.contributions if item["contributed_pixels"]),
            },
            "notes": "Clear photographed samples are kept once. Empty and disagreement areas are marked, never filled in.",
        }
        plan = source_plan(built, sources)
        overview = overview_picture(
            built, seams, request["title"],
            f"{len(sources)} photographs · {len(seams)} seam marker(s) · "
            f"{int(np.count_nonzero(built.grade == ABSENT))} pixels left unphotographed",
            sources,
        )
    if preview:
        return record, {"overview": overview, "source": plan}
    with library.use(write=True) as query:
        saved = query.add("panoramas", record, parent=window_id, action="composited photographed window segments")
        key = saved["id"]
        for placement, ctx in loaded:
            query.link(key, ctx.capture, f"segment photograph {placement.stack + 1}")
        query.attach(key, "map", encode(overview), "image/png")
        query.attach(key, "source", encode(plan), "image/png")
        query.attach(key, "index", encoded(record).encode(), "application/json")
        return query.public(saved)


def overview_csv(record):
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("Overview", record["title"]))
    writer.writerow(("Extent", *record["extent"]))
    writer.writerow(("Lighting", record["lighting"], "Visit", record["visit"]))
    writer.writerow(("Tolerance", record["tolerance"]))
    writer.writerow(())
    writer.writerow(("Source", "Caption", "Lighting", "Visit", "Stack", "Placed clear", "Contributed",
                     "Skipped for earlier clear", "Filled fallback", "Conflicts"))
    for source in record["sources"]:
        part = source["contribution"]
        writer.writerow((source["capture"], source["caption"], source["role"], source["visit"] or "",
                         source["stack"] + 1, part["placed_clear_pixels"], part["contributed_pixels"],
                         part["skipped_for_earlier_clear"], part["filled_fallback_pixels"], part["conflict_pixels"]))
    writer.writerow(())
    writer.writerow(("Seam", "Later source", "Earlier source", "X0", "Y0", "X1", "Y1", "Pixels", "Median delta"))
    for seam in record["seams"]:
        writer.writerow((seam["number"], seam["later_capture"], seam["earlier_capture"] or "",
                         *seam["bounds"], seam["pixels"], seam["median_delta"]))
    writer.writerow(())
    writer.writerow(("Cell", "Row", "Column", "X0", "Y0", "X1", "Y1", "Clear", "Filled", "Empty", "State", "Supplying photograph"))
    for cell in record["cells"]:
        suppliers = "; ".join(
            f'{share["label"]}={share["clear_pixels"]}' for share in cell.get("source_shares", [])
        )
        writer.writerow((f'{cell["row"]}.{cell["column"]}', cell["row"], cell["column"], *cell["bounds"],
                         cell["clear_pixels"], cell["filled_pixels"], cell["empty_pixels"], cell["state"], suppliers))
    return output.getvalue().encode()
