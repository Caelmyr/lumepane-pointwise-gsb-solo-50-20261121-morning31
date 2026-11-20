"""Image intake and independently reviewable examinations."""

from io import BytesIO

from flask import Blueprint, jsonify, request, send_file

from .alignment_check import alignment_check, alignment_history, audit_csv
from .conditions import condition_csv, condition_list, freeze_conditions
from .contracts import identifier
from .intake import intake_csv, intake_import, open_session, session_list, stage_upload, staged_page_picture
from .source_reading import capture_receipt, receipt_csv


def downloadable(data, name):
    return send_file(BytesIO(data), mimetype="text/csv", download_name=name, as_attachment=True, max_age=0)


def workflow_api(library, picture, readable_body, heavy):
    bench = Blueprint("workflows", __name__, url_prefix="/api")

    @bench.post("/captures/<key>/alignment-checks")
    def audit_alignment_save(key):
        with heavy:
            return jsonify(alignment_check(library, identifier(key), readable_body(), save=True)), 201

    @bench.post("/captures/<key>/alignment-checks/preview")
    def audit_alignment_preview(key):
        with heavy:
            return picture(alignment_check(library, identifier(key), readable_body(), save=False))

    @bench.get("/windows/<key>/alignment-checks")
    def audit_alignments(key):
        return jsonify(alignment_history(library, identifier(key)))

    @bench.get("/alignment-checks/<key>/data.csv")
    def audit_alignment_download(key):
        with library.use() as db:
            item = db.one(identifier(key), "alignment_checks")
        return downloadable(audit_csv(item), "alignment-check.csv")

    @bench.get("/windows/<key>/conditions")
    def conditions(key):
        return jsonify(condition_list(library, identifier(key)))

    @bench.post("/windows/<key>/conditions")
    def conditions_freeze(key):
        with heavy:
            return jsonify(freeze_conditions(library, identifier(key), readable_body())), 201

    @bench.post("/windows/<key>/conditions/preview")
    def conditions_dry_run(key):
        with heavy:
            return picture(freeze_conditions(library, identifier(key), readable_body(), preview=True))

    @bench.get("/condition-sheets/<key>/data.csv")
    def conditions_download(key):
        with library.use() as db:
            item = db.one(identifier(key), "condition_sheets")
        return downloadable(condition_csv(item), "reviewed-conditions.csv")

    @bench.get("/windows/<key>/intake")
    def intake_sessions(key):
        return jsonify(session_list(library, identifier(key)))

    @bench.post("/windows/<key>/intake")
    def intake_stage(key):
        with heavy:
            return jsonify(stage_upload(library, identifier(key), request.form, request.files.getlist("images"))), 201

    @bench.get("/intake/<key>")
    def staged_intake(key):
        return jsonify(open_session(library, identifier(key)))

    @bench.post("/intake/<key>/import")
    def import_intake(key):
        with heavy:
            return jsonify(intake_import(library, identifier(key), readable_body())), 201

    @bench.get("/intake/<key>/data.csv")
    def intake_download(key):
        return downloadable(intake_csv(open_session(library, identifier(key))), "photograph-intake.csv")

    @bench.get("/intake/files/<key>/<int:page>")
    def staged_page(key, page):
        with heavy:
            return picture(staged_page_picture(library, identifier(key), page))

    @bench.get("/captures/<key>/source-check")
    def check_source_capture(key):
        with heavy:
            return jsonify(capture_receipt(library, identifier(key)))

    @bench.get("/captures/<key>/source-check/data.csv")
    def download_source_receipt(key):
        with heavy:
            record = capture_receipt(library, identifier(key))
        return downloadable(receipt_csv(record), "original-photograph-check.csv")

    return bench
