#!/usr/bin/env python3
"""Browser-level check of the multi-patch visit comparison UI."""
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playwright.sync_api import sync_playwright
from lumepane.repository import Library
from lumepane.web import create_app


def seed_patches(tmp):
    db_path = next(Path(tmp).rglob("lumepane.sqlite3"))
    lib = Library(db_path.parent)
    with lib.use() as db:
        win = db.list("windows", limit=1)[0]
        cap = next(c for c in db.list("captures", parent=win["id"], oldest=True) if c["role"] == "transmitted")
        align = next(a for a in db.list("alignments", parent=win["id"]) if a["capture_id"] == cap["id"])
        for title, points in [
            ("UI left stable", [[.06, .10], [.16, .10], [.16, .22], [.06, .22]]),
            ("UI right stable", [[.82, .10], [.93, .10], [.93, .22], [.82, .22]]),
        ]:
            mark = db.add("marks", {
                "title": title, "category": "reference", "shape": "polygon", "points": points, "holes": [],
                "state": "observed", "notes": "ui", "coordinate_basis": "window",
                "alignment": align["id"], "source_capture": cap["id"],
                "plane_extent": align["plane_extent"], "measure": {}, "retired": False,
            }, parent=win["id"], action="ui patch")
            db.link(mark["id"], cap, "marked photograph")
            db.link(mark["id"], align, "marked plane")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(Path(tmp))
        server = app.test_client()
        server.post("/api/demonstration")
        win = server.get("/api/windows").get_json()["windows"][0]["id"]
        seed_patches(tmp)

        import threading
        from werkzeug.serving import make_server
        httpd = make_server("127.0.0.1", 19961, app)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda exc: errors.append(str(exc)))
                page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
                page.goto(f"http://127.0.0.1:19961/#window/{win}")
                page.wait_for_selector(".tile, .workspace")

                # Go to Examinations and pick the visit tool.
                page.get_by_role("button", name="Examinations").first.click()
                page.select_option('select[name="tool"]', "visit")
                page.wait_for_selector("#stable-patches")
                comparison = page.locator('select[name="comparison"]')
                comparison.select_option(index=1)
                boxes = page.locator('#stable-patches input[name="reference_marks"]')
                assert boxes.count() >= 3, "expected the demo patch plus two seeded patches"
                # Select all stable patches.
                for i in range(boxes.count()):
                    boxes.nth(i).check()

                # Save the examination.
                page.fill('input[name="title"]', "Browser multi-patch run")
                page.click('form[data-form="report"] button[type="submit"]')
                page.wait_for_selector("dialog[open]")

                # The reopened modal must show the correction block and every selected patch.
                dialog = page.locator("dialog[open]")
                dialog.get_by_role("heading", name="Illumination correction").wait_for(timeout=5000)
                body = dialog.inner_text()
                for name in ("UI left stable", "UI right stable", "Stable center reference"):
                    assert name in body, name
                assert "Participated" in body
                assert "Uncomparable pixels" in body
                assert "Judged coverage" in body
                print("modal correction block ok")
                print("coverage line:", [ln for ln in body.splitlines() if "Judged coverage" in ln])

                # The raw provenance JSON is still available in the details (hidden until expanded).
                pre = dialog.locator("pre").text_content()
                assert "reference_marks" in pre and "uncomparable_pixels" in pre
                assert "stable window patch" in pre
                print("provenance JSON ok")
                page.keyboard.press("Escape")

                # Reopen the saved result from the examination list.
                page.get_by_role("button", name="Open result").first.click()
                page.wait_for_selector("dialog[open]")
                again = page.locator("dialog[open]").inner_text()
                assert "Illumination correction" in again and "UI left stable" in again
                print("reopen saved result ok")

                # Now the uncorrected path: the form re-rendered after save, so reselect the pair,
                # leave every stable patch unchecked, and preview.
                page.keyboard.press("Escape")
                page.select_option('select[name="comparison"]', index=1)
                unchecked = page.locator('#stable-patches input[name="reference_marks"]')
                for i in range(unchecked.count()):
                    unchecked.nth(i).uncheck()
                with page.expect_response(lambda r: "/reports/preview" in r.url and r.ok, timeout=10000) as info:
                    page.get_by_role("button", name="Preview map").click()
                assert info.value.headers.get("content-type") == "image/png"
                page.wait_for_timeout(400)
                print("uncorrected preview ok")

                assert not errors, errors
                browser.close()
        finally:
            httpd.shutdown()
    print("BROWSER CHECKS PASSED")


if __name__ == "__main__":
    main()
