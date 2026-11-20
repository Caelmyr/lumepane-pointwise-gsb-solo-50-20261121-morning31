"""Report orchestration, certified photo selection, and replayable image provenance."""

from collections import Counter
from io import StringIO
import csv

from .catalog import capture_context
from .cames import came_report
from .compare import visit_comparison
from .contracts import DeskError, enum, flag, identifier, identifiers, obj, revision, word
from .exposure import exposure_audit
from .fracture import fracture_candidates, manual_trace_summary
from .glasscolor import color_summary, swatch_records
from .imaging import encode, resize_rgba, srgb_display
from .panes import overlap_attrib, pane_inventory, rim_view
from .profiles import line_profile
from .repository import encoded
from .texture import texture_report
from .transmission import paired_surface_detail, transmission_report

SINGLE = {
    "exposure": ("Exposure and readability", exposure_audit),
    "panes": ("Visible glass inventory", pane_inventory),
    "rim": ("Inside edge condition", rim_view),
    "neighbours": ("Adjacent pane marks", overlap_attrib),
    "came": ("Came strip examination", came_report),
    "fracture": ("Fine fracture candidates", fracture_candidates),
    "trace-log": ("Pinned fracture traces", manual_trace_summary),
    "profile": ("Optical line profile", line_profile),
    "grain": ("Glass texture spectrum", texture_report),
}
WITH_QUERY = {
    "transmission": ("Transmitted-light ratio", transmission_report),
    "paired_detail": ("Surface detail across lights", paired_surface_detail),
    "color": ("Photographed glass color", color_summary),
    "visit": ("Registered visit comparison", visit_comparison),
}
TITLE_FOR = {key: item[0] for option in (SINGLE, WITH_QUERY) for key, item in option.items()}


def report_basis(ctx):
    return {
        "window": ctx.window["id"], "window_version": ctx.window["version"],
        "capture": ctx.capture["id"], "capture_version": ctx.capture["version"],
        "illumination": ctx.capture["role"], "original_sha256": ctx.capture["original_digest"],
        "extent": list(ctx.surface.extent), "physical_mm": ctx.window["physical_mm"] if ctx.alignment else None,
        "alignment": ctx.alignment["id"] if ctx.alignment else None,
        "coordinate_basis": "window plane" if ctx.alignment else "original capture",
        "visible_percent": round(float(ctx.surface.valid.mean()) * 100, 4),
        "used_mark_versions": {mark["id"]: mark["version"] for mark in ctx.marks},
    }


def run_reading(query, request, ctx, method, options):
    if method in SINGLE:
        stats, pictures = SINGLE[method][1](ctx, options)
        extra = []
    elif method == "color":
        stats, pictures, extra = color_summary(ctx, options)
    else:
        stats, pictures, extra = WITH_QUERY[method][1](ctx, query, options)
    return stats, pictures, extra


def job_inputs(query, window_id, request):
    source = identifier(request.get("capture"), "source")
    native = flag(request.get("native"), "source coordinates")
    context = capture_context(query, source, registered=False if native else None)
    if context.window["id"] != window_id:
        raise DeskError("Select a capture in this window.")
    options = obj(request.get("options", {}), "measurement options")
    method = enum(request.get("tool"), "inspection method", TITLE_FOR)
    return context, method, options


def attach_report_assets(db, result, assets):
    for role, image in assets.items():
        db.attach(result["id"], role, encode(image), "image/png")


def build_report(library, window_id, raw, *, preview=False):
    request = obj(raw, "new examination")
    with library.use() as query:
        ctx, method, options = job_inputs(query, window_id, request)
        stats, images, extra = run_reading(query, request, ctx, method, options)
        record = {
            "title": word(request.get("title"), "examination title", limit=100, default=TITLE_FOR[method]),
            "tool": method,
            "basis": report_basis(ctx),
            "reading": stats,
            "synthetic": ctx.capture["synthetic"],
            "primary_source": ctx.capture["id"],
        }
        record["meaning"] = "Photographic observation at the recorded working scale and lighting."
    if preview:
        return record, images
    with library.use(write=True) as query:
        saved = query.add("reports", record, parent=window_id, action="saved an image examination")
        keys = [(ctx.capture, "main examination capture"), (ctx.window, "recorded window size")]
        if ctx.alignment:
            keys.append((ctx.alignment, "main examination alignment"))
        keys.extend((mark, "known observation at time of measurement") for mark in ctx.marks)
        keys.extend(extra)
        links = {(key["id"], role): key for key, role in keys}
        for (_, role), original in links.items():
            query.link(saved["id"], original, role)
        attach_report_assets(query, saved, images)
        return query.public(saved)


def approve_swatch(library, key, raw):
    payload = obj(raw)
    with library.use(write=True) as db:
        result = db.one(key, "reports")
        if result["tool"] != "color":
            raise DeskError("Choose a saved glass color examination.")
        if not result["reading"]["swatches"]:
            raise DeskError("A saved reference needs measured colors.")
        records = db.list("references", parent=result["parent"], limit=120)
        if len(records) >= 120:
            raise DeskError("Archive older references before adding more samples.")
        report_basis = result["basis"]
        source = db.one(report_basis["capture"], "captures")
        value = {
            "name": word(payload.get("title"), "approved glass sample", limit=100, default=result["title"]),
            "family": word(payload.get("family"), "glass series", default="", limit=90),
            "notes": word(payload.get("notes"), "notes", default="", limit=900, multiline=True),
            "result_id": key, "capture_id": source["id"], "source_revision": report_basis["capture_version"],
            "source_image_sha256": report_basis["original_sha256"], "reading": result["reading"],
            "synthetic": bool(result["synthetic"]), "state": "active",
        }
        item = db.add("references", value, parent=result["parent"], action="approved stored image colors")
        db.link(item["id"], result, "approved measurement")
        db.attach(item["id"], "map", db.media(key, "map"), "image/png")
        db.attach(item["id"], "chart", db.media(key, "chart"), "image/png")
        return db.public(item)


def library_view(library, window_id):
    with library.use() as db:
        db.one(window_id, "windows")
        records = db.list("references", parent=window_id, limit=120)
        approved = [item for item in records if item["state"] == "active"]
        return {
            "references": [db.public(item) for item in records],
            "role_summary": swatch_records(approved),
            "distinct_source_count": len({item["capture_id"] for item in approved}),
        }


def retire_reference(library, key, raw):
    form = obj(raw)
    with library.use(write=True) as db:
        old = db.one(key, "references")
        frozen = {k: v for k, v in old.items() if k not in {"id", "kind", "version", "created", "updated", "parent"}}
        frozen["state"] = enum(form.get("state"), "reference visibility", ("active", "retired"), default="retired")
        return db.public(db.revise(key, "references", revision(form), frozen, "changed reference library state"))


def report_csv(record):
    reading = record["reading"]
    head = ["Window basis", record["basis"]["coordinate_basis"], "Tool", record["tool"]]
    out = StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(head)
    scalar = {key: value for key, value in reading.items() if isinstance(value, (str, int, float, bool)) or value is None}
    writer.writerows([[key, value] for key, value in scalar.items()])
    for name, rows in reading.items():
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            continue
        headers = list(dict.fromkeys(key for row in rows for key in row))
        writer.writerow([])
        writer.writerow([name])
        writer.writerow(headers)
        for row in rows:
            writer.writerow([json_cell(row.get(key)) for key in headers])
    return out.getvalue().encode()


def json_cell(value):
    if isinstance(value, (list, dict, tuple)):
        return encoded(value)
    return value


def evidence_overview(library, window_id):
    with library.use() as query:
        window = query.one(window_id, "windows")
        photos = query.list("captures", parent=window_id, limit=36)
        reports = query.list("reports", parent=window_id, limit=120)
        marks = query.list("marks", parent=window_id, limit=160)
        cards = query.list("packets", parent=window_id, limit=60)
        align = query.list("alignments", parent=window_id, limit=300)
        exposures = Counter(photo["role"] for photo in photos)
        states = Counter(mark["state"] for mark in marks if not mark["retired"])
        active_align = {entry["capture_id"] for entry in align}
        visits = {}
        for item in photos:
            if item["visit"] not in visits:
                visits[item["visit"]] = []
            visits[item["visit"]].append({
                "capture": item["id"], "light": item["role"], "caption": item["caption"], "day": item["day"],
                "registered": item["id"] in active_align,
            })
        return {
            "title": window["title"], "window_version": window["version"], "photo_roles": dict(exposures),
            "examination_modes": dict(Counter(doc["tool"] for doc in reports)),
            "observation_statuses": dict(states), "visits": visits,
            "registered_photos": len(active_align), "packets": [query.public(card) for card in cards],
        }
