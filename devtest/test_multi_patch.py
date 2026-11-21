#!/usr/bin/env python3 -I
"""Standalone checks for multi-patch illumination correction in visit comparisons."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lumepane.catalog import capture_context
from lumepane.compare import visit_comparison, CompareInputs
from lumepane.contracts import DeskError
from lumepane.demo import demo_library
from lumepane.repository import Library


def add_reference_mark(db, window, alignment, capture, title, points):
    mark = db.add("marks", {
        "title": title, "category": "reference", "shape": "polygon", "points": points, "holes": [],
        "state": "observed", "notes": "test stable patch", "coordinate_basis": "window",
        "alignment": alignment["id"], "source_capture": capture["id"],
        "plane_extent": alignment["plane_extent"], "measure": {}, "retired": False,
    }, parent=window["id"], action="saved a test reference patch")
    db.link(mark["id"], capture, "marked photograph")
    db.link(mark["id"], alignment, "marked plane")
    return mark


def main():
    with tempfile.TemporaryDirectory() as tmp:
        library = Library(Path(tmp))
        windows = demo_library(library)
        win_id = windows[0]["id"]
        with library.use() as db:
            window = db.one(win_id, "windows")
            captures = db.list("captures", parent=win_id, limit=36, oldest=True)
            transmitted = [c for c in captures if c["role"] == "transmitted"]
            first_cap, later_cap = transmitted[0], transmitted[1]
            align = next(a for a in db.list("alignments", parent=win_id, limit=300)
                         if a["capture_id"] == first_cap["id"])
            existing = [m for m in db.list("marks", parent=win_id, limit=160) if m["category"] == "reference"]
            mark_left = add_reference_mark(db, window, align, first_cap, "Left stable corner",
                                           [[.06, .10], [.16, .10], [.16, .22], [.06, .22]])
            mark_right = add_reference_mark(db, window, align, first_cap, "Right stable corner",
                                            [[.82, .10], [.93, .10], [.93, .22], [.82, .22]])
            mark_low = add_reference_mark(db, window, align, first_cap, "Lower stable corner",
                                          [[.42, .78], [.56, .78], [.56, .90], [.42, .90]])
            ids = [m["id"] for m in existing + [mark_left, mark_right, mark_low]]
            title_by_id = {m["id"]: m["title"] for m in db.list("marks", parent=win_id, limit=160)}

        def run(options):
            with library.use() as query:
                ctx = capture_context(query, first_cap["id"], registered=True)
                report, images, links = visit_comparison(ctx, query, options)
                return report, images, links

        base = {"comparison": later_cap["id"], "difference_kind": "luminance", "threshold": .08}

        # 1. Uncorrected legacy path.
        report, images, _ = run(dict(base))
        assert report["reference"]["applied"] is False
        assert report["uncomparable_pixels"] == 0
        assert images["map"].size[0] > 0
        print("1 uncorrected ok: changed=%d uncomparable=%d" % (report["changed_pixels"], report["uncomparable_pixels"]))

        # 2. Single reference patch (legacy semantics).
        report, _, _ = run(dict(base, reference_mark=ids[0]))
        assert report["reference"]["applied"] is True
        assert report["reference"]["model"] == "constant"
        assert len(report["reference"]["patches"]) == 1
        assert report["uncomparable_pixels"] == 0
        assert report["settings"]["reference_mark"] == ids[0]
        print("2 single-patch ok: shift=%s coverage=%.2f" %
              (report["reference"]["patches"][0]["shift"], report["reference"]["coverage_fraction"]))

        # 3. Multiple scattered patches: planar field, hull coverage limits judged area.
        report, _, _ = run(dict(base, reference_marks=ids))
        ref = report["reference"]
        assert ref["applied"] and ref["model"] == "planar", ref["model"]
        statuses = {p["title"]: p["status"] for p in ref["patches"]}
        assert all(s == "used" for s in statuses.values()), statuses
        assert report["uncomparable_pixels"] > 0, "areas outside the hull must be uncomparable"
        assert report["paired_pixels"] < report["selected_pixels"]
        print("3 multi-patch ok: model=%s coverage=%.3f uncomparable=%d paired=%d" %
              (ref["model"], ref["coverage_fraction"], report["uncomparable_pixels"], report["paired_pixels"]))

        # 4. Patches clustered in one corner: most of the pane is outside coverage.
        clustered = [mark_left["id"], existing[0]["id"]]
        report, _, _ = run(dict(base, reference_marks=clustered))
        wide = run(dict(base, reference_marks=ids))[0]
        assert report["reference"]["coverage_fraction"] < wide["reference"]["coverage_fraction"]
        assert report["uncomparable_pixels"] > wide["uncomparable_pixels"]
        print("4 clustered ok: coverage=%.3f uncomparable=%d" %
              (report["reference"]["coverage_fraction"], report["uncomparable_pixels"]))

        # 5. Patch only common to one image (mask placed over the autumn hidden area).
        with library.use(write=True) as db:
            hidden = add_reference_mark(db, window, align, first_cap, "Hidden later patch",
                                        [[.90, .60], [.97, .60], [.97, .80], [.90, .80]])
        report, _, _ = run(dict(base, reference_marks=[mark_left["id"], mark_right["id"], hidden["id"]]))
        patch_states = {p["title"]: p["status"] for p in report["reference"]["patches"]}
        assert patch_states["Hidden later patch"] == "unpaired", patch_states
        assert any("only one capture" in n for n in report["caveats"])
        print("5 unpaired patch flagged ok:", patch_states)

        # 6. All patches unpaired -> hard error, nothing counted as change.
        try:
            run(dict(base, reference_marks=[hidden["id"]]))
        except DeskError as error:
            print("6 all-unpaired error ok:", error)
        else:
            raise AssertionError("expected DeskError when no patch has common pixels")

        # 7. Incoherent patches: a patch whose shift contradicts the fitted field is excluded.
        from lumepane.compare import solve_field
        extent = align["plane_extent"]

        def obs(key, center, shift, spread=.004):
            return {"key": key, "title": key, "shift": np.array([shift], dtype=np.float32),
                    "spread": np.array([spread], dtype=np.float32), "center": np.array(center), "depth": 500}

        coherent = [obs("a", (.1, .1), .01), obs("b", (.9, .1), .03), obs("c", (.5, .9), .02)]
        kept, dropped, basis = solve_field(coherent + [obs("bad", (.5, .5), .30)], extent, 1)
        assert {item["key"] for item in kept} == {"a", "b", "c"}
        assert {item["key"] for item in dropped} == {"bad"}
        assert kept[0]["field"].shape == (extent[1], extent[0])
        print("7a incoherent patch excluded ok:", [d["key"] for d in dropped], "basis", basis)

        # Three contradictory patches: the outlier is rejected, leaving a coherent field.
        trio = [obs("a", (.1, .1), .01), obs("b", (.9, .1), .012), obs("c", (.5, .85), .011)]
        kept, dropped, basis = solve_field(trio + [obs("bad", (.5, .4), .30)], extent, 1)
        assert {item["key"] for item in kept} == {"a", "b", "c"} and len(dropped) == 1
        print("7b outlier rejected from four patches ok:", [d["key"] for d in dropped])

        # When more patches disagree than a planar field can explain, reduction ends on one auditable patch.
        chaotic = [obs("a", (.1, .1), .01), obs("b", (.9, .2), .25), obs("c", (.2, .9), -.20),
                   obs("d", (.85, .85), .18)]
        kept, dropped, basis = solve_field(chaotic, extent, 1)
        assert len(kept) >= 1 and len(dropped) >= 1
        print("7c chaotic set reduced ok: kept", [o["key"] for o in kept],
              "dropped", [d["key"] for d in dropped])

        # A collinear set falls back to a one-axis ramp and still produces a field.
        line = [obs("a", (.1, .5), .01), obs("b", (.5, .5), .02), obs("c", (.9, .5), .03)]
        kept, dropped, basis = solve_field(line, extent, 1)
        assert not dropped and basis == (0,)
        print("7d collinear ramp ok, basis", basis)

        # Real-data smoke run with a patch over the autumn sheen; it stays usable here.
        with library.use(write=True) as db:
            bogus = add_reference_mark(db, window, align, first_cap, "Sheen patch",
                                       [[.62, .58], [.74, .58], [.74, .70], [.62, .70]])
        report, _, _ = run(dict(base, reference_marks=[mark_left["id"], mark_right["id"], mark_low["id"], bogus["id"]]))
        assert report["reference"]["applied"]
        print("7e real-data field ok: model=%s coverage=%.3f" %
              (report["reference"]["model"], report["reference"]["coverage_fraction"]))

        # 8. Unknown mark id is a validation error.
        try:
            run(dict(base, reference_marks=["0" * 24]))
        except DeskError as error:
            print("8 bad id error ok:", error)
        else:
            raise AssertionError("expected identifier validation error")

        # 9. Color distance mode also runs through the field.
        report, _, _ = run(dict(base, difference_kind="color", reference_marks=ids))
        assert report["reference"]["applied"]
        print("9 color mode ok: model=%s uncomparable=%d" %
              (report["reference"]["model"], report["uncomparable_pixels"]))

        # 10. Saved report round-trips the participating stable patches in settings and reading.
        options = CompareInputs.parse(dict(base, reference_marks=ids))
        assert options.json()["reference_marks"] == ids
        print("10 settings json ok:", options.json()["reference_marks"])

        # 11. Cell rows expose the uncomparable count, never folded into change.
        cell = report["cells"][0]
        assert "uncomparable" in cell
        row_total = cell["unmeasured"] + cell["uncomparable"] + cell["weakened"] + \
            cell["increased"] + cell["steady"] + cell["changed_color"]
        assert row_total <= cell["mapped"]
        print("11 cells ok: sample cell uncomparable=%d" % cell["uncomparable"])

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
