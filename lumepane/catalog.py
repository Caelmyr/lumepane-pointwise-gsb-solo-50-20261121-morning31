"""Windows, photo records, alignment, and markings visible in the shared plane."""

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .contracts import (
    Capacity, DeskError, MARK_KINDS, OBSERVATION_STATES, ROLES, dimensions,
    enum, flag, identifier, integer, iso_day, obj, one_page, revision,
    window_setting, word,
)
from .geometry import (
    CORNERS, arc_length, mark_line, mask_for_marks, native_crop_registration,
    normalized_points_to_px, normalized_quad, outline_geometry, output_plane,
    polygon_mask, scale_for, warp_surface,
)
from .imaging import (
    THUMB_EDGE, encode, marks_display, open_surface, photo_decode, read_upload,
    resize_rgba, srgb_display,
)

MAX_WINDOW_CAPTURES = 36
MAX_WINDOW_MARKS = 160
ALL_REGISTRATION_ROLES = set(ROLES) - {"incident", "dark"}


def window_fields(form, prior=None):
    data = obj(form, "window")
    previous = prior or {}
    pos = previous | dict(data)
    physical = pos.get("physical_mm")
    geometry = pos.get("working_extent")
    return {
        "title": word(pos.get("title"), "window name"),
        "code": word(pos.get("code"), "window reference", default="", limit=60),
        "location": word(pos.get("location"), "location", default="", limit=120),
        "notes": word(pos.get("notes"), "window notes", default="", limit=1800, multiline=True),
        "physical_mm": dimensions(physical, "physical frame", low=5) if physical else None,
        "working_extent": [
            integer(geometry[0], "plane width", 48, 1800),
            integer(geometry[1], "plane height", 48, 1800),
        ] if geometry and isinstance(geometry, list) and len(geometry) == 2 else None,
        "synthetic": previous.get("synthetic", False),
    }


def save_window(library, data, key=None):
    with library.use(write=True) as db:
        if key is None:
            return db.add("windows", window_fields(data), action="cataloged a window")
        old = db.one(key, "windows")
        version = revision(data)
        fields = window_fields(data, old)
        if db.list("alignments", parent=key, limit=1) and fields["working_extent"] != old["working_extent"]:
            raise DeskError("The working dimensions are fixed after the first alignment.", field="working_extent")
        return db.revise(key, "windows", version, fields, "revised window notes and dimensions")


def gallery(library):
    with library.use() as db:
        windows = db.list("windows", limit=80)
        captures = db.children_counts([tile["id"] for tile in windows], "captures")
        packets = db.children_counts([tile["id"] for tile in windows], "packets")
        for item in windows:
            key = item["id"]
            latest = db.list("captures", parent=key, limit=1)
            item.update(
                photo_count=captures.get(key, 0),
                archive_count=packets.get(key, 0),
                thumb=latest[0]["id"] if latest else None,
            )
    return windows


def photo_fields(form):
    requested = one_page({"page": form.get("page", type=int)})
    return {
        "caption": word(form.get("caption"), "capture caption", default="", limit=120),
        "role": enum(form.get("role"), "lighting role", ROLES, default="transmitted"),
        "visit": word(form.get("visit"), "visit label", default="", limit=70),
        "day": iso_day(form.get("day"), "capture day"),
        "page": requested,
        "notes": word(form.get("notes"), "capture notes", default="", limit=1200, multiline=True),
    }


def new_capture(db, window, photo, name, settings, *, synthetic=False):
    already = db.list("captures", parent=window["id"], limit=MAX_WINDOW_CAPTURES + 1)
    for item in already:
        if (
            item["original_digest"] == photo.digest
            and item["page"] == settings["page"]
            and item["role"] == settings["role"]
            and item["visit"] == settings["visit"]
        ):
            return db.public(item), True
    if len(already) >= MAX_WINDOW_CAPTURES:
        raise Capacity(f"Keep at most {MAX_WINDOW_CAPTURES} photographs in one window record.")
    report = photo.metadata | settings | {
        "file_name": name,
        "caption": settings["caption"] or name,
        "synthetic": synthetic,
        "original_digest": photo.digest,
        "warnings": list(photo.warnings),
    }
    capture = db.add("captures", report, parent=window["id"], action="imported original photograph")
    key = capture["id"]
    db.attach(key, "original", photo.raw, photo.media_type)
    db.attach(key, "working", encode(photo.picture), "image/png")
    db.attach(key, "thumbnail", encode(resize_rgba(photo.picture, THUMB_EDGE)), "image/png")
    return db.public(capture), False


def upload_capture(library, key, form, file):
    fields = photo_fields(form)
    raw, name = read_upload(file)
    photo = photo_decode(raw, fields["page"])
    with library.use(write=True) as db:
        window = db.one(key, "windows")
        item, duplicate = new_capture(db, window, photo, name, fields)
    return {"capture": item, "duplicate": duplicate}


def revise_capture(library, key, data):
    payload = obj(data)
    with library.use(write=True) as db:
        current = db.one(key, "captures")
        revised = {
            k: v for k, v in current.items()
            if k not in {"id", "parent", "version", "kind", "created", "updated"}
        }
        revised["caption"] = word(payload.get("caption"), "caption", limit=120, default=current["caption"])
        revised["visit"] = word(payload.get("visit"), "visit", limit=70, default=current["visit"])
        revised["day"] = iso_day(payload.get("day", current["day"]), "capture day")
        revised["notes"] = word(payload.get("notes"), "capture notes", limit=1200, multiline=True, default=current["notes"])
        record = db.revise(key, "captures", revision(data), revised, "revised capture context")
        return db.public(record)


def latest_alignment(db, window_id, capture_id):
    candidates = db.list("alignments", parent=window_id, limit=300)
    return next((entry for entry in candidates if entry["capture_id"] == capture_id), None)


def captured_alignment(db, window, photo, requested):
    native = requested is None
    corners = list(CORNERS) if native else normalized_quad(requested)
    previous = db.list("alignments", parent=window["id"], limit=1)
    extent = tuple(previous[0]["plane_extent"]) if previous else output_plane(window, photo["extent"])
    information = native_crop_registration(corners, photo, extent)
    raw = open_surface(db, photo)
    fitted = warp_surface(raw, information["matrix"], extent)
    support = float(np.mean(fitted.valid))
    if support < .12:
        raise DeskError("The marked corners leave too little of the window in this photograph.")
    return information | {
        "capture_id": photo["id"],
        "capture_version": photo["version"],
        "support_fraction": round(support, 6),
        "whole_photo_assumed": native,
    }, fitted


def align_capture(library, photo_key, data, *, persist):
    wanted = obj(data, "alignment")
    with library.use(write=persist) as db:
        photo = db.one(photo_key, "captures")
        owner = db.one(photo["parent"], "windows")
        if photo["role"] not in ALL_REGISTRATION_ROLES:
            raise DeskError("Reference photos are sampled without a panel alignment.")
        if wanted.get("capture_version") is not None:
            db.require_version(photo, integer(wanted["capture_version"], "capture version", 1, 2_000_000_000))
        if wanted.get("window_version") is not None:
            db.require_version(owner, integer(wanted["window_version"], "window version", 1, 2_000_000_000))
        information, fitted = captured_alignment(db, owner, photo, wanted.get("corners"))
        if not persist:
            return information, srgb_display(fitted, None)
        key = db.add("alignments", information, parent=owner["id"], action="approved capture in window plane")
        db.link(key["id"], photo, "registered photo")
        db.link(key["id"], owner, "window dimensions")
        db.attach(key["id"], "plane", encode(srgb_display(fitted, None)), "image/png")
        return db.public(key)


@dataclass
class CaptureView:
    window: dict
    capture: dict
    alignment: dict | None
    surface: object
    marks: list

    @cached_property
    def excluded(self):
        return mask_for_marks(self.surface.extent, self.marks, {"ignore", "occluded"})

    @cached_property
    def usable(self):
        return self.surface.valid & ~self.excluded

    @property
    def scale(self):
        return scale_for(self.window, self.surface.extent)


def capture_context(db, photo_key, *, registered=None, exact_alignment=None):
    frame = db.one(photo_key, "captures")
    window = db.one(frame["parent"], "windows")
    if exact_alignment:
        align = db.one(exact_alignment, "alignments", parent=window["id"])
        if align["capture_id"] != frame["id"]:
            raise DeskError("This alignment was made from a different photo.")
    else:
        align = latest_alignment(db, window["id"], frame["id"]) if registered is not False else None
    if registered is True and not align:
        raise DeskError("Align the selected photograph to the window before measuring it.")
    surface = open_surface(db, frame)
    if align:
        surface = warp_surface(surface, align["matrix"], tuple(align["plane_extent"]))
    marks = [
        r for r in db.list("marks", parent=window["id"], limit=MAX_WINDOW_MARKS, oldest=True)
        if not r["retired"] and (
            r["coordinate_basis"] == "window" and align
            or not align and r["source_capture"] == photo_key and r["coordinate_basis"] == "capture"
        )
    ]
    return CaptureView(window, frame, align, surface, marks)


def display_photo(library, key, settings):
    values = obj(settings, "display configuration")
    plane = flag(values.get("plane"), "registered plane", default=True)
    show = flag(values.get("marks"), "show marks", default=True)
    with library.use() as db:
        context = capture_context(db, key, registered=None if plane else False)
        image = srgb_display(context.surface, window_setting(values.get("display", {})))
        return marks_display(image, context.marks) if show else image


def mark_details(data, context, previous=None):
    payload = obj(data, "observation")
    previous = previous or {}
    fields = previous | dict(payload)
    shape = enum(fields.get("shape"), "shape", ("polygon", "line"), default="polygon")
    category = enum(fields.get("category"), "mark category", MARK_KINDS)
    if shape == "line" and category in {"pane", "ignore", "occluded", "reference"}:
        raise DeskError("Use a polygon for this kind of marked region.", field="shape")
    if shape == "polygon":
        points, holes = outline_geometry(fields)
    else:
        points, holes = mark_line(fields), []
    extent = context.surface.extent
    measure = {}
    if shape == "polygon":
        mask = polygon_mask(extent, points, holes)
        count = int(np.count_nonzero(mask))
        visible = int(np.count_nonzero(mask & context.surface.valid))
        measure.update(
            pixels=count, visible_pixels=visible,
            visible_percent=round(100 * visible / count, 2) if count else None,
            square_mm=round(float(count * np.prod(context.scale)), 3) if context.scale is not None and context.alignment else None,
        )
    else:
        pixels = normalized_points_to_px(points, extent)
        measure["trace_px"] = round(arc_length(pixels), 3)
        measure["trace_mm"] = round(arc_length(pixels, context.scale), 3) if context.scale is not None and context.alignment else None
    return {
        "title": word(fields.get("title"), "observation title", limit=100),
        "category": category,
        "shape": shape, "points": points, "holes": holes,
        "state": enum(fields.get("state"), "observation status", OBSERVATION_STATES, default="observed"),
        "notes": word(fields.get("notes"), "notes", default="", multiline=True, limit=1000),
        "coordinate_basis": "window" if context.alignment else "capture",
        "source_capture": context.capture["id"],
        "alignment": context.alignment["id"] if context.alignment else None,
        "plane_extent": list(extent),
        "measure": measure,
        "retired": flag(fields.get("retired"), "archive observation"),
    }


def save_mark(library, key, form, existing=None):
    with library.use(write=True) as db:
        previous = db.one(existing, "marks") if existing else None
        if previous:
            requested = previous["source_capture"]
        else:
            requested = identifier(form.get("capture"), "source photograph")
        registered = (False if flag(form.get("native"), "original photograph") else None) if not previous else previous["coordinate_basis"] == "window"
        context = capture_context(db, requested, registered=registered)
        if context.window["id"] != key or previous and previous["parent"] != key:
            raise DeskError("The observation does not belong to this window.")
        if not previous and len(db.list("marks", parent=key, limit=MAX_WINDOW_MARKS)) >= MAX_WINDOW_MARKS:
            raise Capacity("This window already has the maximum number of observations.")
        properties = mark_details(form, context, previous)
        if previous:
            return db.revise(existing, "marks", revision(form), properties, "revised an observation")
        result = db.add("marks", properties, parent=key, action="drew an examination observation")
        db.link(result["id"], context.capture, "marked photograph")
        if context.alignment:
            db.link(result["id"], context.alignment, "marked plane")
        return db.public(result)


def case_contents(library, key):
    with library.use() as db:
        window = db.one(key, "windows")
        photos = db.list("captures", parent=key, limit=MAX_WINDOW_CAPTURES, oldest=True)
        alignments = defaultdict(list)
        for alignment in db.list("alignments", parent=key, limit=300):
            alignments[alignment["capture_id"]].append(alignment)
        for photo in photos:
            photo["alignment"] = next(iter(alignments[photo["id"]]), None)
        return {
            "window": window,
            "captures": photos,
            "marks": db.list("marks", parent=key, limit=MAX_WINDOW_MARKS),
            "reports": db.list("reports", parent=key, limit=100),
            "references": db.list("references", parent=key, limit=60),
            "packets": db.list("packets", parent=key, limit=60),
            "visits": db.list("treatments", parent=key, limit=180),
            "alignment_checks": db.list("alignment_checks", parent=key, limit=100),
            "conditions": db.list("condition_sheets", parent=key, limit=80),
            "intake_sessions": db.list("intake_sessions", parent=key, limit=30),
            "resurvey_plans": db.list("resurvey_plans", parent=key, limit=40),
        }
