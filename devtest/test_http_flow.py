#!/usr/bin/env python3
"""HTTP-level check: multi-patch visit comparison saves and reopens with provenance."""
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumepane.demo import demo_library
from lumepane.repository import Library
from lumepane.web import create_app


def add_mark(db, win_id, alignment, capture, title, points):
    mark = db.add("marks", {
        "title": title, "category": "reference", "shape": "polygon", "points": points, "holes": [],
        "state": "observed", "notes": "http test", "coordinate_basis": "window",
        "alignment": alignment["id"], "source_capture": capture["id"],
        "plane_extent": alignment["plane_extent"], "measure": {}, "retired": False,
    }, parent=win_id, action="saved test patch")
    db.link(mark["id"], capture, "marked photograph")
    db.link(mark["id"], alignment, "marked plane")
    return mark


def main():
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(Path(tmp))
        library = app.config["LIBRARY"] if "LIBRARY" in app.config else None
        # Reuse the same library the app created by seeding through the demo endpoint.
        client = app.test_client()
        created = client.post("/api/demonstration").get_json()["windows"]
        win_id = created[0]["id"]
        case = client.get(f"/api/windows/{win_id}").get_json()
        transmitted = [c for c in case["captures"] if c["role"] == "transmitted"]
        first_id, later_id = transmitted[0]["id"], transmitted[1]["id"]
        existing_refs = [m["id"] for m in case["marks"] if m["category"] == "reference"]

        # Seed two extra scattered patches directly in the backing library.
        from lumepane.repository import Library as _Lib  # noqa: F401
        import sqlite3
        db_path = next(Path(tmp).rglob("lumepane.sqlite3"))
        lib = Library(db_path.parent)
        with lib.use(write=True) as db:
            window = db.one(win_id, "windows")
            cap = db.one(first_id, "captures")
            align = next(a for a in db.list("alignments", parent=win_id) if a["capture_id"] == first_id)
            extra = [
                add_mark(db, win_id, align, cap, "HTTP left patch", [[.06, .10], [.16, .10], [.16, .22], [.06, .22]]),
                add_mark(db, win_id, align, cap, "HTTP right patch", [[.82, .10], [.93, .10], [.93, .22], [.82, .22]]),
            ]
        patch_ids = existing_refs + [m["id"] for m in extra]

        payload = {
            "tool": "visit", "capture": first_id, "title": "Multi-patch visit check",
            "options": {"comparison": later_id, "reference_marks": patch_ids, "difference_kind": "luminance"},
        }
        saved = client.post(f"/api/windows/{win_id}/reports", json=payload)
        assert saved.status_code == 201, saved.get_json()
        report = saved.get_json()
        reading = report["reading"]
        assert reading["reference"]["model"] in ("planar", "ramp")
        titles = {p["title"] for p in reading["reference"]["patches"]}
        assert "HTTP left patch" in titles and "HTTP right patch" in titles, titles
        assert reading["settings"]["reference_marks"] == patch_ids
        assert reading["uncomparable_pixels"] > 0
        assert "map" in {a["label"] for a in report["assets"]}

        # Reopen the saved record and confirm provenance survived.
        reopened = client.get(f"/api/records/{report['id']}").get_json()
        again = reopened["reading"]["reference"]["patches"]
        assert len(again) == len(patch_ids)
        roles = {link["role"] for link in reopened["inputs"]}
        assert "stable window patch" in roles, roles
        print("saved+reopened ok: patches=", [(p["title"], p["status"]) for p in again])
        print("uncomparable=", reopened["reading"]["uncomparable_pixels"],
              "changed=", reopened["reading"]["changed_pixels"])

        # Legacy single-patch payload still works over HTTP.
        legacy = client.post(f"/api/windows/{win_id}/reports", json={
            "tool": "visit", "capture": first_id, "title": "Legacy single patch",
            "options": {"comparison": later_id, "reference_mark": existing_refs[0]},
        })
        assert legacy.status_code == 201, legacy.get_json()
        legacy_reading = legacy.get_json()["reading"]
        assert legacy_reading["reference"]["model"] == "constant"
        assert legacy_reading["settings"]["reference_marks"] == [existing_refs[0]]
        print("legacy single-patch payload ok")

        # Preview endpoint returns a PNG.
        preview = client.post(f"/api/windows/{win_id}/reports/preview", json=payload)
        assert preview.status_code == 200 and preview.mimetype == "image/png"
        print("preview png ok, bytes=", len(preview.data))

        # CSV export includes the new cell column.
        csv_data = client.get(f"/api/reports/{report['id']}/data.csv").data.decode()
        assert "uncomparable" in csv_data
        print("csv contains uncomparable column ok")

    print("HTTP CHECKS PASSED")


if __name__ == "__main__":
    main()
