"""Loopback HTTP interface for the examination desk."""

from io import BytesIO
import json
import logging
from pathlib import Path
from threading import BoundedSemaphore

from flask import Blueprint, Flask, jsonify, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from .catalog import (
    align_capture, capture_context, case_contents, display_photo, gallery,
    revise_capture, save_mark, save_window, upload_capture,
)
from .compare import comparator_wipe
from .contracts import DeskError, MAX_IMAGE_BYTES, identifier, obj
from .demo import demo_library, showcase
from .evidence import MAX_ARCHIVE_BYTES, freeze_packet, receive_archive, received_view, scale_print
from .exposure import readout_probe
from .imaging import encode
from .reports import TITLE_FOR, approve_swatch, build_report, evidence_overview, library_view, report_csv, retire_reference
from .repository import Library
from .resurvey import create_plan, plan_csv, plan_list, prepare_plan, revise_plan
from .visitlog import save_visit, timeline, visit_notes
from .workflows import workflow_api

WORKERS = 2


def readable_body():
    return obj(request.get_json(silent=True), "application request")


def json_dict_form(name, fallback=None):
    value = request.args.get(name, "")
    if not value:
        return fallback or {}
    if len(value) > 12_000:
        raise DeskError("The display options are too long.")
    try:
        return obj(json.loads(value), "display options")
    except ValueError as error:
        raise DeskError("Decode the display options as an object.") from error


def picture(raw):
    if not isinstance(raw, bytes):
        raw = encode(raw)
    response = send_file(BytesIO(raw), mimetype="image/png", max_age=0)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def public_media(library, key, tag):
    with library.use() as query:
        entry = query.one(key)
        path, asset = query.media_location(key, tag)
    suggestion = entry.get("file_name") or entry.get("title") or entry.get("caption") or "lumepane"
    extension = {
        "image/png": ".png", "image/jpeg": ".jpg", "image/tiff": ".tif",
        "application/pdf": ".pdf", "application/zip": ".zip", "application/json": ".json",
    }.get(asset["mime"], "")
    name = suggestion if tag == "original" else f"{suggestion}{extension}"
    download = tag == "original" or not asset["mime"].startswith("image/")
    result = send_file(path, mimetype=asset["mime"], download_name=name, as_attachment=download,
                       etag=asset["digest"], conditional=True)
    result.headers["X-Content-Type-Options"] = "nosniff"
    return result


def build_api(library):
    api = Blueprint("views", __name__, url_prefix="/api")
    heavy = BoundedSemaphore(WORKERS)

    @api.get("/health")
    def health():
        return {"status": "ready"}

    @api.get("/tools")
    def tools():
        return TITLE_FOR

    @api.get("/showcase")
    def hero():
        return picture(showcase())

    @api.post("/demonstration")
    def training():
        with heavy:
            return jsonify({"windows": demo_library(library)}), 201

    @api.get("/windows")
    def windows():
        return jsonify({"windows": gallery(library)})

    @api.post("/windows")
    def window_new():
        return jsonify(save_window(library, readable_body())), 201

    @api.get("/windows/<key>")
    def window_detail(key):
        return jsonify(case_contents(library, identifier(key)))

    @api.put("/windows/<key>")
    def window_edit(key):
        return jsonify(save_window(library, readable_body(), identifier(key)))

    @api.post("/windows/<key>/captures")
    def upload(key):
        with heavy:
            return jsonify(upload_capture(library, identifier(key), request.form, request.files.get("image"))), 201

    @api.patch("/captures/<key>")
    def capture_edit(key):
        return jsonify(revise_capture(library, identifier(key), readable_body()))

    @api.get("/captures/<key>/picture")
    def capture_view(key):
        with heavy:
            return picture(display_photo(library, identifier(key), json_dict_form("view")))

    @api.post("/captures/<key>/probe")
    def capture_probe(key):
        raw = readable_body()
        with heavy, library.use() as db:
            context = capture_context(db, identifier(key), registered=False if raw.get("native") else None)
            return jsonify(readout_probe(context, raw))

    @api.post("/captures/<key>/wipe")
    def wipe(key):
        with heavy, library.use() as db:
            context = capture_context(db, identifier(key), registered=True)
            return picture(comparator_wipe(context, db, readable_body()))

    @api.post("/captures/<key>/align/preview")
    def align_preview(key):
        with heavy:
            details, photo = align_capture(library, identifier(key), readable_body(), persist=False)
            result = picture(photo)
            result.headers["X-Support-Fraction"] = str(details["support_fraction"])
            return result

    @api.post("/captures/<key>/align")
    def capture_align(key):
        with heavy:
            return jsonify(align_capture(library, identifier(key), readable_body(), persist=True)), 201

    @api.post("/windows/<key>/marks")
    def mark_new(key):
        with heavy:
            return jsonify(save_mark(library, identifier(key), readable_body())), 201

    @api.patch("/windows/<key>/marks/<mark>")
    def mark_revise(key, mark):
        with heavy:
            return jsonify(save_mark(library, identifier(key), readable_body(), existing=identifier(mark)))

    @api.post("/windows/<key>/reports")
    def report(key):
        with heavy:
            return jsonify(build_report(library, identifier(key), readable_body())), 201

    @api.post("/windows/<key>/reports/preview")
    def preview_report(key):
        with heavy:
            reading, images = build_report(library, identifier(key), readable_body(), preview=True)
            return picture(images["map"])

    @api.get("/windows/<key>/references")
    def reference_list(key):
        return jsonify(library_view(library, identifier(key)))

    @api.post("/reports/<key>/approve")
    def approve(key):
        return jsonify(approve_swatch(library, identifier(key), readable_body())), 201

    @api.put("/references/<key>")
    def reference_edit(key):
        return jsonify(retire_reference(library, identifier(key), readable_body()))

    @api.get("/reports/<key>/data.csv")
    def report_data(key):
        with library.use() as db:
            entry = db.one(identifier(key), "reports")
        return send_file(BytesIO(report_csv(entry)), mimetype="text/csv", download_name="examination.csv", as_attachment=True)

    @api.get("/windows/<key>/evidence")
    def summary(key):
        return jsonify(evidence_overview(library, identifier(key)))

    @api.post("/windows/<key>/packets")
    def packet(key):
        with heavy:
            return jsonify(freeze_packet(library, identifier(key), readable_body(), bundle=True)), 201

    @api.post("/windows/<key>/contact")
    def contact(key):
        with heavy:
            return jsonify(freeze_packet(library, identifier(key), readable_body(), bundle=False)), 201

    @api.post("/windows/<key>/scaled-proof")
    def proof(key):
        with heavy:
            return jsonify(scale_print(library, identifier(key), readable_body())), 201

    @api.post("/receive")
    def receive():
        with heavy:
            return jsonify(receive_archive(library, request.files.get("packet"))), 201

    @api.get("/receive")
    def received():
        with library.use() as db:
            return jsonify({"packets": [db.public(item) for item in db.list("received", limit=110)]})

    @api.get("/receive/<key>/view/<int:index>")
    def received_picture(key, index):
        return picture(received_view(library, identifier(key), index))

    @api.post("/windows/<key>/visits")
    def visit_save(key):
        return jsonify(save_visit(library, identifier(key), readable_body())), 201

    @api.get("/windows/<key>/visits")
    def visits(key):
        return jsonify(timeline(library, identifier(key), retired=request.args.get("archived") == "1"))

    @api.get("/windows/<key>/resurvey-plans")
    def resurvey_plans(key):
        return jsonify(plan_list(library, identifier(key)))

    @api.post("/windows/<key>/resurvey-plans")
    def resurvey_plan_save(key):
        with heavy:
            return jsonify(create_plan(library, identifier(key), readable_body())), 201

    @api.post("/windows/<key>/resurvey-plans/preview")
    def resurvey_plan_preview(key):
        with heavy:
            _, map_image = prepare_plan(library, identifier(key), readable_body())
            return picture(map_image)

    @api.put("/resurvey-plans/<key>")
    def resurvey_plan_edit(key):
        with heavy:
            return jsonify(revise_plan(library, identifier(key), readable_body()))

    @api.get("/resurvey-plans/<key>/data.csv")
    def resurvey_plan_data(key):
        with library.use() as db:
            entry = db.one(identifier(key), "resurvey_plans")
        return send_file(BytesIO(plan_csv(entry)), mimetype="text/csv",
                         download_name="resurvey-plan.csv", as_attachment=True)

    @api.patch("/visits/<key>")
    def visit_edit(key):
        return jsonify(visit_notes(library, identifier(key), readable_body()))

    @api.get("/records/<key>")
    def saved_record(key):
        with library.use() as db:
            return jsonify(db.public(db.one(identifier(key))))

    @api.get("/records/<key>/history")
    def history(key):
        with library.use() as db:
            return jsonify({"revisions": db.series(identifier(key))})

    @api.get("/records/<key>/check")
    def verify(key):
        with library.use() as db:
            return jsonify(db.audit(identifier(key)))

    @api.get("/records/<key>/media/<label>")
    def media(key, label):
        return public_media(library, identifier(key), label)

    return api


def create_app(path):
    root = Path(__file__).resolve().parents[1]
    app = Flask(__name__, static_folder=str(root / "static"))
    app.config.update(MAX_CONTENT_LENGTH=max(MAX_IMAGE_BYTES, MAX_ARCHIVE_BYTES) + 524288,
                      JSON_SORT_KEYS=False)
    storage = Library(path)
    app.register_blueprint(build_api(storage))
    app.register_blueprint(workflow_api(storage, picture, readable_body, BoundedSemaphore(WORKERS)))

    @app.get("/")
    def home():
        return app.send_static_file("index.html")

    @app.errorhandler(DeskError)
    def request_problem(error):
        return jsonify({"error": str(error), "field": error.field, "status": error.status}), error.status

    @app.errorhandler(RequestEntityTooLarge)
    def large_input(error):
        return jsonify({"error": "The selected data is larger than the local upload allowance.", "status": 413}), 413

    @app.errorhandler(500)
    def unexpected(error):
        app.logger.error("Examination service error: %s", error)
        return jsonify({"error": "The examination service could not finish this request.", "status": 500}), 500

    @app.after_request
    def client_headers(response):
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    return app
