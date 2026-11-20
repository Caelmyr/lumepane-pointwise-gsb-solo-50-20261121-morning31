"""Freeze review sheets from pinned visit citations and recorded observation states."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageDraw

from .catalog import capture_context
from .contracts import MARK_KINDS, OBSERVATION_STATES, DeskError, flag, identifier, identifiers, iso_day, obj, word
from .geometry import polygon_mask
from .imaging import canvas_picture, encode, font, resize_rgba
from .intake import spreadsheet_cell
from .visitlog import VISIT_TYPES
from .visuals import AQUA, GOLD, INK, PALE, QUIET, ROSE, legend_band

STATE_COLOR = {"observed": AQUA, "review": GOLD, "resolved": ROSE, "unreviewed": "#a9a39c"}


@dataclass(frozen=True)
class ReviewRequest:
    title: str
    day_start: str
    day_end: str
    include_archived: bool
    visits: tuple
    display_photo: str | None

    @classmethod
    def parse(cls, raw):
        form = obj(raw, "condition review sheet")
        first = iso_day(form.get("start"), "earliest visit day") or ""
        last = iso_day(form.get("end"), "latest visit day") or ""
        if first and last and last < first:
            raise DeskError("The latest day must follow the earliest day.")
        return cls(
            word(form.get("title"), "review sheet title", default="Condition observations", limit=100),
            first, last, flag(form.get("include_archived"), "include archived visits"),
            tuple(identifiers(form["visits"], "visit log records", high=40)) if form.get("visits") else (),
            identifier(form["display_photo"], "display photograph") if form.get("display_photo") else None,
        )


def list_allowed_visits(db, window, request):
    if request.visits:
        prior = [db.one(key, "treatments", parent=window) for key in request.visits]
    else:
        prior = db.list("treatments", parent=window, limit=180)
    result = sorted(
        (item for item in prior if (request.include_archived or not item["archived"])
         and (not request.day_start or request.day_start <= item["day"])
         and (not request.day_end or request.day_end >= item["day"])),
        key=lambda row: (row["day"], row["created"], row["id"]),
    )
    if not result:
        raise DeskError("No visit log records meet the selected dates and archive filter.")
    if len(result) > 40:
        raise DeskError("Limit a condition sheet to 40 visits by date or selection.")
    return result


def exact_link(db, item):
    try:
        live = db.one(item["id"], item["type"])
    except DeskError:
        return None, None
    for entry in db.series(live["id"]):
        if entry["version"] == item["version"]:
            pinned = entry["record"] | {
                "version": entry["version"], "id": live["id"], "kind": live["kind"], "parent": live["parent"],
            }
            return pinned, live
    return None, live


def reviewed_marks(db, window, visits):
    groups = defaultdict(list)
    unavailable = []
    for visit in visits:
        for ref in visit["observed_evidence"]:
            if ref["type"] != "marks":
                continue
            mark, live = exact_link(db, ref)
            if mark is None or mark["parent"] != window:
                unavailable.append({
                    "visit": visit["id"], "mark": ref["id"], "pinned_version": ref["version"], "label": ref["label"],
                })
                continue
            groups[mark["id"]].append((visit, mark, live))
    return groups, unavailable


def visit_line(visit, mark):
    return {
        "visit": visit["id"], "visit_title": visit["title"], "visit_day": visit["day"],
        "visit_type": VISIT_TYPES[visit["visit_type"]], "mark_version": mark["version"],
        "classification": mark["state"], "label": mark["title"], "archived_at_reference": mark["retired"],
    }


def basis_compatibility(mark, display):
    if not display:
        return "no photograph chosen"
    if mark["coordinate_basis"] == "window":
        return "available" if display.alignment else "needs registered photo"
    if mark["source_capture"] != display.capture["id"]:
        return "marked on another source"
    return "needs original view" if display.alignment else "available"


def usable_stats(mark, ctx):
    outcome = basis_compatibility(mark, ctx)
    if outcome != "available":
        return {"mapping": outcome}
    if mark["shape"] == "line":
        return {"mapping": "available", "observation_kind": "trace"}
    mask = polygon_mask(ctx.surface.extent, mark["points"], mark["holes"])
    all_pixels = int(mask.sum())
    photographed = int(np.count_nonzero(mask & ctx.surface.valid))
    usable = int(np.count_nonzero(mask & ctx.usable))
    return {
        "mapping": "available",
        "observation_kind": "measured region",
        "region_pixels": all_pixels,
        "photographed_pixels": photographed,
        "usable_pixels": usable,
        "fraction_photographed": round(photographed / all_pixels, 6) if all_pixels else None,
        "fraction_usable": round(usable / all_pixels, 6) if all_pixels else None,
    }


def grouped_rows(groups, display):
    details = []
    for key, snapshots in groups.items():
        visit, mark, current = snapshots[-1]
        touches = [visit_line(entry, snapshot) for entry, snapshot, _ in snapshots]
        statuses = [item["classification"] for item in touches]
        transitions = [
            {"from": prior, "to": following, "visit_day": on["visit_day"], "visit_title": on["visit_title"]}
            for prior, following, on in zip(statuses, statuses[1:], touches[1:]) if prior != following
        ]
        tangible = {
            "id": key, "title": mark["title"], "kind": MARK_KINDS[mark["category"]],
            "category": mark["category"], "shape": mark["shape"], "state": states_label(mark["state"]),
            "state_key": mark["state"], "recorded_on": visit["day"], "used_observation_version": mark["version"],
            "current_observation_version": current["version"], "current_revision_changed": current["version"] != mark["version"],
            "history": touches, "changes": transitions, "photograph_support": usable_stats(mark, display),
        }
        details.append((tangible, mark))
    return sorted(details, key=lambda item: (item[0]["kind"], item[0]["title"].casefold(), item[0]["id"]))


def states_label(value):
    return OBSERVATION_STATES.get(value, "Unclassified")


def review_totals(rows, visits, missing):
    by_kind, by_state = Counter(), Counter()
    pictured = 0
    changed = 0
    for line, _ in rows:
        by_kind[line["category"]] += 1
        by_state[line["state_key"]] += 1
        pictured += line["photograph_support"]["mapping"] == "available"
        changed += line["current_revision_changed"]
    return {
        "selected_visits": len(visits), "cited_observations": len(rows),
        "cited_versions_missing": len(missing), "mapped_observations": pictured,
        "later_revisions_available": changed, "status_count": dict(by_state), "mark_type_count": dict(by_kind),
        "date_first": visits[0]["day"], "date_last": visits[-1]["day"],
    }


def condition_page(title, window, items, counts):
    canvas = Image.new("RGB", (920, 265 + 37 * max(1, min(14, len(items)))), PALE)
    ink = ImageDraw.Draw(canvas)
    ink.text((22, 16), title[:75], font=font(26, bold=True), fill=INK)
    ink.text((23, 52), window["title"][:96], font=font(16), fill=QUIET)
    pulse = f'{counts["cited_observations"]} cited observations · {counts["selected_visits"]} visits'
    ink.text((23, 82), pulse, font=font(14, bold=True), fill=INK)
    ink.text((23, 105), f'{counts["date_first"]} to {counts["date_last"]}', font=font(12), fill=QUIET)
    for i, (word, key) in enumerate((("Observed", "observed"), ("Needs review", "review"), ("Resolved", "resolved"))):
        x0 = 22 + 285 * i
        ink.rounded_rectangle((x0, 137, x0 + 264, 183), radius=6, fill="#eae4d8")
        ink.rectangle((x0 + 9, 151, x0 + 23, 165), fill=STATE_COLOR[key])
        ink.text((x0 + 30, 150), f'{word} · {counts["status_count"].get(key, 0)}', font=font(14, bold=True), fill=INK)
    ink.text((22, 205), "Cited observation", font=font(13, bold=True), fill=QUIET)
    ink.text((500, 205), "Recorded status", font=font(13, bold=True), fill=QUIET)
    ink.text((740, 205), "Last pinned visit", font=font(13, bold=True), fill=QUIET)
    for index, (row, _) in enumerate(items[:14]):
        y = 234 + index * 36
        ink.line((22, y - 6, 895, y - 6), fill="#c9c2b6")
        ink.text((22, y), row["title"][:60], font=font(14), fill=INK)
        ink.text((500, y), row["state"], font=font(14), fill=INK)
        ink.text((740, y), row["recorded_on"], font=font(14), fill=INK)
    return canvas


def contrast_masks(display, reviewed, resized):
    by_state = defaultdict(list)
    for row, mark in reviewed:
        if row["photograph_support"]["mapping"] == "available":
            by_state[row["state_key"]].append(mark)
    for state, marks in by_state.items():
        outlines = []
        for mark in marks:
            form = [(u * resized.width, v * resized.height) for u, v in mark["points"]]
            if mark["shape"] == "polygon":
                form = form + form[:1]
            outlines.append((form, mark["title"]))
        yield STATE_COLOR[state], outlines


def observation_image(ctx, groups):
    image = resize_rgba(canvas_picture(ctx.surface), 1200).convert("RGB")
    pen = ImageDraw.Draw(image)
    for color, marks in contrast_masks(ctx, groups, image):
        for points, name in marks:
            pen.line(points, fill="#272825", width=7)
            pen.line(points, fill=color, width=3)
            x, y = points[0]
            pen.text((x + 4, y + 4), name[:25], font=font(12), fill=color, stroke_width=2, stroke_fill="#202622")
    return legend_band(image, "Pinned photographed conditions",
                       [(states_label(key), value) for key, value in STATE_COLOR.items() if key != "unreviewed"])


def sheet_record(db, window, request, visits):
    source = capture_context(db, request.display_photo) if request.display_photo else None
    if source and source.window["id"] != window["id"]:
        raise DeskError("Choose a review photograph from this window.")
    selected, missing = reviewed_marks(db, window["id"], visits)
    rows = grouped_rows(selected, source)
    tally = review_totals(rows, visits, missing)
    links = [{
        "id": item["id"], "day": item["day"], "version": item["version"], "title": item["title"], "archived": item["archived"],
    } for item in visits]
    card = {
        "title": request.title, "window_version": window["version"], "reading": tally, "window_title": window["title"],
        "observations": [row for row, _ in rows], "unavailable_pinned_versions": missing, "visits": links,
        "synthetic": window["synthetic"],
        "coordinate_photo": source.capture["id"] if source else None,
        "coordinate_photo_original_sha256": source.capture["original_digest"] if source else None,
        "coordinate_photo_version": source.capture["version"] if source else None,
        "coordinate_alignment": source.alignment["id"] if source and source.alignment else None,
    }
    previews = {"chart": condition_page(request.title, window, rows, tally)}
    if source:
        previews["map"] = observation_image(source, rows)
    return card, previews, source


def freeze_conditions(library, window_id, raw, *, preview=False):
    req = ReviewRequest.parse(raw)
    with library.use() as db:
        window = db.one(window_id, "windows")
        selected = list_allowed_visits(db, window_id, req)
        values, images, photo = sheet_record(db, window, req, selected)
    if preview:
        return images.get("map", images["chart"])
    with library.use(write=True) as db:
        result = db.add("condition_sheets", values, parent=window_id, action="froze review of pinned visit observations")
        key = result["id"]
        db.link(key, window, "recorded window")
        for planned in selected:
            db.link(key, planned, "included visit citation")
        if photo:
            db.link(key, photo.capture, "review illustration")
            if photo.alignment:
                db.link(key, photo.alignment, "review illustration alignment")
        for role, rendered in images.items():
            db.attach(key, role, encode(rendered), "image/png")
        return db.public(result)


def condition_list(library, window):
    with library.use() as db:
        db.one(window, "windows")
        return {"conditions": db.list("condition_sheets", parent=window, limit=80)}


def condition_csv(sheet):
    buff = StringIO(newline="")
    writer = csv.writer(buff)
    writer.writerow(("Observation", "Kind", "State", "Pinned version", "Current version", "Last visit", "Photograph mapping"))
    for row in sheet["observations"]:
        writer.writerow((spreadsheet_cell(row["title"]), row["kind"], row["state"], row["used_observation_version"],
                         row["current_observation_version"], row["recorded_on"], row["photograph_support"]["mapping"]))
    writer.writerow(())
    writer.writerow(("Observation", "Visit", "Date", "Recorded version", "Classification"))
    for row in sheet["observations"]:
        for event in row["history"]:
            writer.writerow((spreadsheet_cell(row["title"]), spreadsheet_cell(event["visit_title"]),
                             event["visit_day"], event["mark_version"], states_label(event["classification"])))
    return buff.getvalue().encode()
