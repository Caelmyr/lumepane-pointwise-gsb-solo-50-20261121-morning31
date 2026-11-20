"""Observed treatment and inspection history linked to fixed image references."""

from collections import defaultdict
from datetime import datetime

from .contracts import DeskError, enum, flag, identifier, identifiers, iso_day, obj, revision, word

VISIT_TYPES = {
    "initial": "Initial examination",
    "review": "Condition review",
    "followup": "Follow-up photography",
    "treatment_record": "Recorded intervention",
    "handover": "Documentation handover",
}


def tags(raw):
    entries = raw.get("tags") or []
    if not isinstance(entries, list) or len(entries) > 8:
        raise DeskError("Use up to eight observation labels.")
    return list(dict.fromkeys(word(item, "observation label", limit=35) for item in entries))


def pin_window_records(query, window, raw):
    requested = raw.get("refs")
    if not requested:
        return [], []
    maps = identifiers(requested, "linked evidence", high=18)
    records = [query.one(key, parent=window) for key in maps]
    allowed = {"captures", "marks", "reports", "packets"}
    if any(item["kind"] not in allowed for item in records):
        raise DeskError("Link a capture, observation, saved examination, or evidence packet.")
    return records, [
        {"id": item["id"], "type": item["kind"], "version": item["version"],
         "label": item.get("title", item.get("caption", ""))}
        for item in records
    ]


def date_breakdown(stamp):
    parsed = datetime.fromisoformat(stamp)
    return [parsed.year, parsed.month, parsed.day]


def save_visit(library, window, payload):
    opts = obj(payload, "inspection log item")
    when = iso_day(opts.get("day"), "inspection day")
    if not when:
        raise DeskError("Enter the date observed on the report or visit.", field="day")
    kind = enum(opts.get("visit_type"), "visit type", VISIT_TYPES, default="review")
    with library.use(write=True) as db:
        old = db.one(window, "windows")
        linked, evidence = pin_window_records(db, window, opts)
        doc = {
            "title": word(opts.get("title"), "title", limit=100),
            "day": when, "notes": word(opts.get("notes"), "inspection description", default="", limit=3500, multiline=True),
            "visit_type": kind, "operator": word(opts.get("operator"), "operator", default="", limit=70),
            "source_window_version": old["version"],
            "labels": tags(opts),
            "observed_evidence": evidence,
            "synthetic": bool(old["synthetic"]),
            "archived": False,
        }
        saved = db.add("treatments", doc, parent=window, action="pinned a visit to photograph and observation versions")
        for item in linked:
            db.link(saved["id"], item, "cited visit evidence")
        return db.public(saved)


def visit_notes(library, key, payload):
    opts = obj(payload)
    with library.use(write=True) as db:
        now = db.one(key, "treatments")
        values = {k: v for k, v in now.items() if k not in {"id", "kind", "parent", "version", "updated", "created"}}
        values["archived"] = flag(opts.get("archived"), "archived", default=now["archived"])
        values["notes"] = word(opts.get("notes"), "inspection notes", default=now["notes"], limit=3500, multiline=True)
        return db.public(db.revise(key, "treatments", revision(opts), values, "revised notes on a pinned visit"))


def current_references(db, citations):
    result = []
    for record in citations:
        source = db.one(record["id"], record["type"])
        result.append(record | {
            "current_version": source["version"],
            "revision_outcome": "current" if source["version"] == record["version"] else "later revision available",
        })
    return result


def timeline(library, window_id, retired=False):
    with library.use() as db:
        window = db.one(window_id, "windows")
        records = db.list("treatments", parent=window_id, limit=180)
        known = sorted((doc for doc in records if retired or not doc["archived"]), key=lambda r: (r["day"], r["created"]), reverse=True)
        groups = defaultdict(list)
        for entry in known:
            calendar = date_breakdown(entry["day"])
            history = current_references(db, entry["observed_evidence"])
            groups[f"{calendar[0]}-{calendar[1]:02d}"].append(entry | {
                "reviewed_bindings": history,
                "category_label": VISIT_TYPES[entry["visit_type"]],
            })
        return {"window": window["title"], "periods": [{"month": stamp, "visits": rows} for stamp, rows in groups.items()]}
