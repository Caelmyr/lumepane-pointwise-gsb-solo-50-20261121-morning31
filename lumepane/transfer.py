"""Carry observations from an old framed photograph onto a re-framed window plane.

The outer frame was replaced, so the old and new photographs do not share
identical borders. An operator identifies the same glass landmarks in both
images; their paired coordinates fit a projective homography from the old
window plane to the new registered plane. Each old observation is projected
and classified against the new boundary, its photographed support, and any
occluded or excluded regions already recorded on the new plane.
"""

import csv
from dataclasses import dataclass
from io import StringIO

import numpy as np
from PIL import Image, ImageDraw

from .catalog import latest_alignment
from .contracts import (
    MARK_KINDS, OBSERVATION_STATES, DeskError, collection, flag, identifier,
    number, obj, point, revision, word,
)
from .geometry import fit_transform, polygon_mask, transform_points
from .imaging import canvas_picture, encode, font, resize_rgba
from .visuals import AQUA, GOLD, legend_band

READY = "ready"
REVIEW = "review"
DEFAULT_SUPPORT_FLOOR = 0.98
POINT_LIMIT = 2.4
MIN_LANDMARKS = 4
MAX_LANDMARKS = 12
MAX_REVIEW_NOTES = 1400
OVERLAP_FLOOR = 0.02

STATUS_COLOR = {READY: AQUA, REVIEW: GOLD}
REASON_TEXT = {
    "crosses_boundary": "Crosses the new frame boundary",
    "occluded": "Falls in an occluded region",
    "excluded": "Falls in an excluded region",
    "unsupported": "Outside the photographed support",
    "unreachable": "Projects beyond the reachable plane",
}
REVIEW_STATUSES = {"open", "resolved", "dismissed"}


@dataclass(frozen=True)
class LandmarkPair:
    name: str
    old: tuple
    new: tuple

    @classmethod
    def parse(cls, raw, index):
        item = obj(raw, "landmark pair")
        return cls(
            word(item.get("name"), "landmark name", default=f"Landmark {index + 1}", limit=80),
            tuple(point(item.get("old"), "old plane position", low=-POINT_LIMIT, high=POINT_LIMIT)),
            tuple(point(item.get("new"), "new plane position", low=-POINT_LIMIT, high=POINT_LIMIT)),
        )


@dataclass(frozen=True)
class TransferSetup:
    old_capture: str
    new_capture: str
    landmarks: tuple
    title: str
    include_retired: bool
    support_floor: float

    @classmethod
    def parse(cls, raw):
        form = obj(raw, "frame transfer")
        pairs = collection(form.get("landmarks"), "paired landmarks", MIN_LANDMARKS, MAX_LANDMARKS)
        markers = tuple(LandmarkPair.parse(entry, i) for i, entry in enumerate(pairs))
        names = [item.name.casefold() for item in markers]
        if len(set(names)) != len(names):
            raise DeskError("Name each shared landmark independently.", field="landmarks")
        for side in ("old", "new"):
            positions = [getattr(item, side) for item in markers]
            if len(set(positions)) != len(positions):
                raise DeskError(f"Give each {side} landmark a distinct position.", field="landmarks")
        return cls(
            identifier(form.get("old_capture"), "old photograph"),
            identifier(form.get("new_capture"), "new photograph"),
            markers,
            word(form.get("title"), "transfer title", default="Frame transfer preview", limit=110),
            flag(form.get("include_retired"), "include archived observations"),
            number(form.get("support_floor"), "support floor", .5, 1, default=DEFAULT_SUPPORT_FLOOR),
        )


def registered_capture(db, window_id, key, label):
    capture = db.one(key, "captures", parent=window_id)
    alignment = latest_alignment(db, window_id, key)
    if alignment is None:
        raise DeskError(f"Register the {label} photograph to the window plane first.")
    return capture, alignment


def surface_for(db, capture):
    from .catalog import capture_context
    return capture_context(db, capture["id"], registered=True)


def transfer_matrix(setup):
    old_positions = [item.old for item in setup.landmarks]
    new_positions = [item.new for item in setup.landmarks]
    return np.asarray(fit_transform(old_positions, new_positions))


def project_outline(matrix, points):
    try:
        mapped = transform_points(matrix, points)
    except DeskError:
        return None
    return np.clip(mapped, -POINT_LIMIT, POINT_LIMIT).round(7).tolist()


def point_support(ctx, positions):
    """Fraction of outline vertices landing on valid new-plane pixels."""
    if not positions:
        return 0.0
    width, height = ctx.surface.extent
    inside = 0
    for x, y in positions:
        px = int(round(x * width - .5))
        py = int(round(y * height - .5))
        if 0 <= px < width and 0 <= py < height and bool(ctx.surface.valid[py, px]):
            inside += 1
    return inside / len(positions)


def region_support(ctx, points, holes):
    mask = polygon_mask(ctx.surface.extent, points, holes)
    total = int(mask.sum())
    if not total:
        return 0.0, 0.0, 0
    photographed = int(np.count_nonzero(mask & ctx.surface.valid))
    usable = int(np.count_nonzero(mask & ctx.usable))
    return photographed / total, usable / total, total


def overlay_masks(ctx, categories):
    masks = [polygon_mask(ctx.surface.extent, item["points"], item.get("holes", []))
             for item in ctx.marks
             if item["shape"] == "polygon" and item["category"] in categories]
    if not masks:
        return None
    return np.logical_or.reduce(masks)


def region_overlaps(ctx, points, holes, categories):
    overlay = overlay_masks(ctx, categories)
    if overlay is None:
        return False
    target = polygon_mask(ctx.surface.extent, points, holes)
    overlap = int(np.count_nonzero(target & overlay))
    return overlap > max(2, int(target.sum()) * OVERLAP_FLOOR)


def trace_overlaps(ctx, points, categories):
    overlay = overlay_masks(ctx, categories)
    if overlay is None:
        return False
    width, height = ctx.surface.extent
    hits = 0
    for x, y in points:
        px, py = int(round(x * width - .5)), int(round(y * height - .5))
        if 0 <= px < width and 0 <= py < height and overlay[py, px]:
            hits += 1
    return hits >= max(1, len(points) // 3)


def base_row(mark, projected, holes, reasons, support):
    return {
        "mark_id": mark["id"], "title": mark["title"], "category": mark["category"],
        "kind_label": MARK_KINDS[mark["category"]], "shape": mark["shape"],
        "state": mark["state"], "state_label": OBSERVATION_STATES[mark["state"]],
        "retired": mark["retired"], "source_capture": mark["source_capture"],
        "source_version": mark["version"], "source_points": mark["points"],
        "projected_points": projected, "projected_holes": holes,
        "status": REVIEW if reasons else READY,
        "reasons": reasons, "reason_labels": [REASON_TEXT[item] for item in reasons],
        "support": support,
    }


def evaluate_mark(mark, matrix, new_ctx, support_floor):
    projected = project_outline(matrix, mark["points"])
    if projected is None:
        return base_row(mark, None, [], ["unreachable"],
                        {"shape": mark["shape"], "vertex_support": 0.0})
    holes = [project_outline(matrix, ring) for ring in mark.get("holes", [])]
    if any(ring is None for ring in holes):
        return base_row(mark, projected, [], ["unreachable"],
                        {"shape": mark["shape"], "vertex_support": 0.0})
    holes = [ring for ring in holes if ring]
    reasons = []
    if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in projected):
        reasons.append("crosses_boundary")
    if mark["shape"] == "polygon":
        fraction, usable_fraction, pixels = region_support(new_ctx, projected, holes)
        if region_overlaps(new_ctx, projected, holes, {"occluded"}):
            reasons.append("occluded")
        if region_overlaps(new_ctx, projected, holes, {"ignore"}):
            reasons.append("excluded")
        if fraction < 0.999:
            reasons.append("unsupported")
        if not reasons and usable_fraction < support_floor:
            reasons.append("unsupported")
        support = {
            "shape": "polygon", "region_pixels": pixels,
            "photographed_fraction": round(fraction, 6),
            "usable_fraction": round(usable_fraction, 6),
            "vertex_support": round(point_support(new_ctx, projected), 6),
        }
    else:
        vertex = point_support(new_ctx, projected)
        if vertex < 1.0:
            reasons.append("unsupported")
        if trace_overlaps(new_ctx, projected, {"occluded"}):
            reasons.append("occluded")
        if trace_overlaps(new_ctx, projected, {"ignore"}):
            reasons.append("excluded")
        support = {"shape": "trace", "vertex_support": round(vertex, 6)}
    return base_row(mark, projected, holes, reasons, support)


def landmark_residuals(matrix, setup):
    predicted = transform_points(matrix, [item.old for item in setup.landmarks])
    rows = []
    for item, guess in zip(setup.landmarks, predicted):
        delta = np.asarray(item.new) - guess
        rows.append({
            "name": item.name, "old": list(item.old), "new": list(item.new),
            "predicted": guess.round(7).tolist(),
            "residual": round(float(np.linalg.norm(delta)), 7),
        })
    return rows


def build_preview(db, window_id, setup):
    window = db.one(window_id, "windows")
    if setup.old_capture == setup.new_capture:
        raise DeskError("Choose two different photographs for the old and new frame.")
    old_capture, old_align = registered_capture(db, window_id, setup.old_capture, "old")
    new_capture, new_align = registered_capture(db, window_id, setup.new_capture, "new")
    if old_align["plane_extent"] != new_align["plane_extent"]:
        raise DeskError("The two registrations use different working plane dimensions.")
    matrix = transfer_matrix(setup)
    old_ctx = surface_for(db, old_capture)
    new_ctx = surface_for(db, new_capture)
    marks = [item for item in db.list("marks", parent=window_id, limit=200, oldest=True)
             if (not item["retired"] or setup.include_retired)
             and item["source_capture"] == setup.old_capture]
    rows = [evaluate_mark(item, matrix, new_ctx, setup.support_floor) for item in marks]
    ready = [item for item in rows if item["status"] == READY]
    review = [item for item in rows if item["status"] == REVIEW]
    payload = {
        "title": setup.title,
        "window": window_id,
        "old_capture": setup.old_capture,
        "new_capture": setup.new_capture,
        "old_capture_caption": old_capture["caption"],
        "new_capture_caption": new_capture["caption"],
        "old_alignment": old_align["id"],
        "new_alignment": new_align["id"],
        "matrix": matrix.tolist(),
        "landmarks": [
            {"name": pair.name, "old": list(pair.old), "new": list(pair.new)}
            for pair in setup.landmarks
        ],
        "landmark_residuals": landmark_residuals(matrix, setup),
        "support_floor": setup.support_floor,
        "plane_extent": list(new_align["plane_extent"]),
        "window_version": window["version"],
        "observations": rows,
        "totals": {
            "considered": len(rows),
            "ready": len(ready),
            "review": len(review),
            "archived_considered": sum(1 for item in rows if item["retired"]),
        },
    }
    return payload, new_ctx, old_ctx, matrix


def draw_outline(pen, points, image_size, color, shape):
    width, height = image_size
    mapped = [(x * width, y * height) for x, y in points]
    if shape == "polygon":
        mapped.append(mapped[0])
    pen.line(mapped, fill="#20231f", width=6, joint="curve")
    pen.line(mapped, fill=color, width=3, joint="curve")


def preview_picture(new_ctx, rows, title):
    base = resize_rgba(canvas_picture(new_ctx.surface), 1000).convert("RGB")
    pen = ImageDraw.Draw(base)
    for row in rows:
        if row["projected_points"] is None:
            continue
        draw_outline(pen, row["projected_points"], base.size,
                     STATUS_COLOR[row["status"]], row["shape"])
        x, y = row["projected_points"][0]
        pen.text((x * base.size[0] + 3, y * base.size[1] + 3),
                 row["title"][:24], font=font(12, bold=True), fill=STATUS_COLOR[row["status"]],
                 stroke_width=2, stroke_fill="#20231f")
    return legend_band(base, title, [("Ready to carry over", STATUS_COLOR[READY]),
                                     ("Hold for review", STATUS_COLOR[REVIEW])])


def landmark_picture(old_ctx, new_ctx, setup):
    boards = (
        ("Shared landmarks on the old frame", old_ctx, [item.old for item in setup.landmarks]),
        ("Same landmarks on the new frame", new_ctx, [item.new for item in setup.landmarks]),
    )
    pieces = []
    for heading, ctx, positions in boards:
        image = resize_rgba(canvas_picture(ctx.surface), 760).convert("RGB")
        pen = ImageDraw.Draw(image)
        for index, ((x, y), pair) in enumerate(zip(positions, setup.landmarks), start=1):
            px, py = x * image.size[0], y * image.size[1]
            pen.ellipse((px - 7, py - 7, px + 7, py + 7), fill=GOLD, outline="#20231f", width=2)
            pen.text((px + 9, py - 9), str(index), font=font(14, bold=True), fill="#f7f1e3",
                     stroke_width=2, stroke_fill="#20231f")
            pen.text((px + 9, py + 4), pair.name[:18], font=font(10), fill="#f7f1e3",
                     stroke_width=2, stroke_fill="#20231f")
        pieces.append((heading, image))
    width = sum(item.size[0] for _, item in pieces) + 42
    height = max(item.size[1] for _, item in pieces) + 101
    page = Image.new("RGB", (width, height), (30, 36, 49))
    pen = ImageDraw.Draw(page)
    pen.text((20, 14), "Landmark correspondence used for the transfer",
             font=font(20, bold=True), fill="#f9edcf")
    x = 14
    for heading, image in pieces:
        pen.text((x + 4, 42), heading, font=font(14, bold=True), fill="#dce0e7")
        page.paste(image, (x, 65))
        x += image.size[0] + 14
    return page


def transfer_preview(library, window_id, raw):
    setup = TransferSetup.parse(raw)
    with library.use() as db:
        payload, new_ctx, old_ctx, _ = build_preview(db, window_id, setup)
    return (payload,
            preview_picture(new_ctx, payload["observations"], setup.title),
            landmark_picture(old_ctx, new_ctx, setup))


def landmark_board(library, window_id, raw):
    setup = TransferSetup.parse(raw)
    with library.use() as db:
        _, new_ctx, old_ctx, _ = build_preview(db, window_id, setup)
    return landmark_picture(old_ctx, new_ctx, setup)


def chosen_rows(payload, requested):
    if requested is None:
        return [row for row in payload["observations"] if row["status"] == READY]
    wanted = set(identifier(item, "observation")
                 for item in collection(requested, "selected observations", 1, 200))
    rows = [row for row in payload["observations"] if row["mark_id"] in wanted]
    if len(rows) != len(wanted):
        raise DeskError("One selected observation is not part of this preview.")
    blocked = [row for row in rows if row["status"] != READY]
    if blocked:
        raise DeskError("Resolve the review flags before carrying over "
                        + ", ".join(item["title"] for item in blocked[:4]) + ".")
    return rows


def carry_measure(mark, projected, holes, ctx):
    if mark["shape"] != "polygon":
        return {}
    mask = polygon_mask(ctx.surface.extent, projected, holes)
    count = int(mask.sum())
    visible = int(np.count_nonzero(mask & ctx.surface.valid))
    measure = {
        "pixels": count, "visible_pixels": visible,
        "visible_percent": round(100 * visible / count, 2) if count else None,
    }
    if ctx.scale is not None and ctx.alignment:
        measure["square_mm"] = round(float(count * np.prod(ctx.scale)), 3)
    return measure


def commit_transfer(library, window_id, raw):
    setup = TransferSetup.parse(raw)
    with library.use(write=True) as db:
        payload, new_ctx, old_ctx, _ = build_preview(db, window_id, setup)
        rows = chosen_rows(payload, raw.get("selection"))
        session = db.add("frame_transfers", {
            "title": setup.title,
            "method": "landmark homography from old to new framed plane",
            "old_capture": setup.old_capture,
            "new_capture": setup.new_capture,
            "old_alignment": payload["old_alignment"],
            "new_alignment": payload["new_alignment"],
            "matrix": payload["matrix"],
            "landmarks": payload["landmarks"],
            "landmark_residuals": payload["landmark_residuals"],
            "plane_extent": payload["plane_extent"],
            "support_floor": setup.support_floor,
            "window_version": payload["window_version"],
            "totals": payload["totals"],
            "observations": payload["observations"],
            "carried": [],
        }, parent=window_id, action="projected old observations onto the re-framed plane")
        session_id = session["id"]
        new_capture = db.one(setup.new_capture, "captures")
        new_alignment = db.one(payload["new_alignment"], "alignments", parent=window_id)
        old_capture = db.one(setup.old_capture, "captures")
        already_transferred = {
            item.get("transferred_from")
            for item in db.list("marks", parent=window_id, limit=200, oldest=True)
            if item.get("transfer_session")
        }
        carried = []
        skipped = []
        for row in rows:
            original = db.one(row["mark_id"], "marks", parent=window_id)
            if original["id"] in already_transferred:
                skipped.append(original["title"])
                continue
            properties = {
                "title": original["title"], "category": original["category"],
                "shape": original["shape"], "points": row["projected_points"],
                "holes": row["projected_holes"], "state": original["state"],
                "notes": original["notes"], "coordinate_basis": "window",
                "source_capture": new_capture["id"],
                "alignment": new_alignment["id"],
                "plane_extent": list(new_alignment["plane_extent"]),
                "measure": carry_measure(original, row["projected_points"],
                                         row["projected_holes"], new_ctx),
                "retired": False,
                "transferred_from": original["id"],
                "transfer_session": session_id,
            }
            new_mark = db.add("marks", properties, parent=window_id,
                              action="carried an observation onto the re-framed plane")
            db.link(new_mark["id"], new_capture, "marked photograph")
            db.link(new_mark["id"], new_alignment, "marked plane")
            db.link(new_mark["id"], original, "transferred from observation")
            db.link(new_mark["id"], session, "frame transfer")
            carried.append({
                "source_mark": original["id"], "source_version": row["source_version"],
                "title": original["title"], "new_mark": new_mark["id"],
            })
        db.link(session_id, old_capture, "old framed photograph")
        db.link(session_id, new_capture, "new framed photograph")
        db.link(session_id, new_alignment, "new registered plane")
        db.link(session_id, db.one(window_id, "windows"), "recorded window")
        db.attach(session_id, "map",
                  encode(preview_picture(new_ctx, payload["observations"], setup.title)), "image/png")
        db.attach(session_id, "chart", encode(landmark_picture(old_ctx, new_ctx, setup)), "image/png")
        fresh = db.one(session_id, "frame_transfers")
        body = {k: v for k, v in fresh.items()
                if k not in {"id", "kind", "parent", "version", "created", "updated"}}
        body["carried"] = carried
        body["skipped_already_carried"] = skipped
        saved = db.revise(session_id, "frame_transfers", fresh["version"], body,
                          "recorded the new observations created by the transfer")
        return db.public(saved)


def transfer_history(library, window_id):
    with library.use() as db:
        db.one(window_id, "windows")
        sessions = db.list("frame_transfers", parent=window_id, limit=80)
    return {"transfers": sessions}


def review_form(raw):
    form = obj(raw, "transfer review")
    points = form.get("projected_points")
    if points is not None and not isinstance(points, list):
        raise DeskError("Projected points must be a list of coordinate pairs.", field="projected_points")
    return {
        "session": identifier(form["session"], "transfer session") if form.get("session") else None,
        "mark": identifier(form.get("mark"), "old observation"),
        "new_capture": identifier(form["new_capture"], "new photograph") if form.get("new_capture") else None,
        "reason": word(form.get("reason"), "review reason", limit=160),
        "notes": word(form.get("notes"), "review notes", default="",
                      limit=MAX_REVIEW_NOTES, multiline=True),
        "projected_points": points,
    }


def save_review(library, window_id, raw):
    details = review_form(raw)
    with library.use(write=True) as db:
        window = db.one(window_id, "windows")
        mark = db.one(details["mark"], "marks", parent=window_id)
        session = None
        if details["session"]:
            session = db.one(details["session"], "frame_transfers", parent=window_id)
        if details["new_capture"]:
            db.one(details["new_capture"], "captures", parent=window_id)
        record = {
            "title": f"Review hold · {mark['title']}"[:110],
            "source_mark": mark["id"], "source_version": mark["version"],
            "source_capture": mark["source_capture"],
            "transfer_session": session["id"] if session else None,
            "new_capture": details["new_capture"],
            "reason": details["reason"], "notes": details["notes"],
            "status": "open",
            "projected_points": details["projected_points"],
            "window_version": window["version"],
        }
        saved = db.add("transfer_reviews", record, parent=window_id,
                       action="held an observation for manual re-framing review")
        db.link(saved["id"], mark, "observation under review")
        if session:
            db.link(saved["id"], session, "frame transfer session")
        if details["new_capture"]:
            db.link(saved["id"], db.one(details["new_capture"], "captures", parent=window_id),
                    "new framed photograph")
        return db.public(saved)


def revise_review(library, key, raw):
    form = obj(raw)
    with library.use(write=True) as db:
        current = db.one(key, "transfer_reviews")
        body = {k: v for k, v in current.items()
                if k not in {"id", "kind", "parent", "version", "created", "updated"}}
        body["status"] = word(form.get("status", current["status"]), "review status",
                              limit=24, default=current["status"])
        if body["status"] not in REVIEW_STATUSES:
            raise DeskError("Choose open, resolved, or dismissed for the review status.")
        body["notes"] = word(form.get("notes"), "review notes", default=current["notes"],
                             limit=MAX_REVIEW_NOTES, multiline=True)
        return db.public(db.revise(key, "transfer_reviews", revision(form), body,
                                   "updated the re-framing review hold"))


def review_list(library, window_id):
    with library.use() as db:
        db.one(window_id, "windows")
        rows = db.list("transfer_reviews", parent=window_id, limit=120, oldest=True)
    return {"reviews": rows}


def transfer_csv(session):
    out = StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(("Old observation", "Kind", "Status", "Reasons",
                     "Photographed support", "Usable support"))
    for row in session.get("observations", []):
        support = row.get("support", {})
        writer.writerow((
            row["title"], row["kind_label"], row["status"],
            "; ".join(row["reason_labels"]),
            support.get("photographed_fraction", support.get("vertex_support", "")),
            support.get("usable_fraction", ""),
        ))
    return out.getvalue().encode()
