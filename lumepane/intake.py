"""Stage independent local image files, inspect pages, and import selected frames."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO, StringIO
import csv
import warnings

from PIL import Image, UnidentifiedImageError

from .catalog import MAX_WINDOW_CAPTURES, new_capture
from .contracts import (
    Capacity, DeskError, MAX_FRAME_EDGE, MAX_IMAGE_BYTES, MAX_UPLOAD_PIXELS, ROLES,
    collection, enum, identifier, integer, iso_day, obj, revision, word,
)
from .imaging import encode, photo_decode, resize_rgba, safe_filename

MAX_FILES = 8
MAX_STAGED_PAGES = 18
MAX_SCAN_BYTES = 35 * 1024**2
FORMATS = {"PNG", "JPEG", "TIFF", "WEBP"}


@dataclass
class Scan:
    name: str
    data: bytes
    digest: str
    report: dict
    preview: bytes | None


def image_prescan(data):
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as image:
            fmt = image.format
            if fmt not in FORMATS:
                raise DeskError("The file uses an unsupported image format.")
            count = getattr(image, "n_frames", 1)
            result = []
            for index in range(min(count, MAX_STAGED_PAGES)):
                image.seek(index)
                width, height = image.size
                admissible = (
                    width >= 16 and height >= 16
                    and width <= MAX_FRAME_EDGE and height <= MAX_FRAME_EDGE
                    and width * height <= MAX_UPLOAD_PIXELS
                )
                result.append({
                    "page": index + 1, "original_extent": [width, height], "storage_mode": image.mode,
                    "suitable_dimensions": admissible,
                })
            return fmt, count, result


def decoded_page(page, decoded):
    working = decoded.metadata
    page.update(
        available=True,
        planned_analysis_extent=working["extent"],
        upright_size=working["upright_size"],
        opaque_fraction=working["usable_alpha_fraction"],
        color_handling=working["analysis_space"],
        camera=working["camera"], warnings=list(decoded.warnings),
        outcome="ready",
    )


def examined_binary(data, name):
    content_hash = sha256(data).hexdigest()
    info = {"file_name": name, "bytes": len(data), "sha256": content_hash, "pages": [], "ready": 0, "state": "unreadable"}
    preview = None
    try:
        codec, count, pages = image_prescan(data)
        info.update(encoding=codec, page_count=count, scan_limit=min(count, MAX_STAGED_PAGES))
        for item in pages:
            if not item["suitable_dimensions"]:
                item.update(available=False, outcome="Outside photograph size limits")
                continue
            try:
                decoded = photo_decode(data, item["page"])
            except DeskError as error:
                item.update(available=False, outcome=str(error))
                continue
            decoded_page(item, decoded)
            if preview is None:
                preview = encode(resize_rgba(decoded.picture, 250))
    except (UnidentifiedImageError, Image.DecompressionBombWarning, Image.DecompressionBombError,
            OSError, EOFError, SyntaxError, ValueError, DeskError) as error:
        info["explanation"] = str(error) if isinstance(error, DeskError) else "Image metadata could not be read."
    else:
        usable = sum(item["available"] for item in pages)
        info.update(pages=pages, ready=usable,
                    state="all inspected pages available" if usable == len(pages) and count == len(pages) else
                    "some pages available" if usable else "no available pages")
    return Scan(name, data, content_hash, info, preview)


def staged_streams(files):
    if not 1 <= len(files) <= MAX_FILES:
        raise DeskError(f"Choose between 1 and {MAX_FILES} local files.")
    current = 0
    prior = set()
    result = []
    for stream in files:
        name = safe_filename(stream.filename)
        if not stream.filename:
            raise DeskError("A selected file has no name.")
        remaining = min(MAX_IMAGE_BYTES, MAX_SCAN_BYTES - current)
        data = stream.stream.read(remaining + 1)
        if len(data) > remaining:
            raise Capacity("The selected files exceed the staging limit. Split them into smaller batches.")
        current += len(data)
        digest = sha256(data).hexdigest()
        scan = examined_binary(data, name)
        scan.report["same_content_as_earlier_row"] = digest in prior
        prior.add(digest)
        result.append(scan)
    return result


def session_totals(staged):
    good = sum(item.report["ready"] for item in staged)
    return {
        "files": len(staged), "available_pages": good, "distinct_originals": len({item.digest for item in staged}),
        "stored_bytes": sum(len(item.data) for item in staged if item.report["ready"]),
        "source_warnings": sum(bool(item.report.get("explanation")) for item in staged),
    }


def register_scan(db, session, stage, index):
    values = stage.report | {"sequence": index}
    stored = db.add("intake_files", values, parent=session["id"], action="scanned source pages for photographic use")
    if stage.preview:
        db.attach(stored["id"], "thumbnail", stage.preview, "image/png")
        media = Image.MIME.get(stage.report["encoding"], "application/octet-stream")
        db.attach(stored["id"], "original", stage.data, media)
    return stored


def stage_upload(library, key, form, files):
    label = word(form.get("title"), "batch title", default="Photo intake selection", limit=100)
    with library.use() as db:
        db.one(key, "windows")
        if len(db.list("intake_sessions", parent=key, limit=30)) >= 30:
            raise Capacity("This window already contains 30 intake sessions.")
    scanned = staged_streams(files)
    result = session_totals(scanned)
    with library.use(write=True) as db:
        window = db.one(key, "windows")
        if len(db.list("intake_sessions", parent=key, limit=30)) >= 30:
            raise Capacity("This window already contains 30 intake sessions.")
        session = db.add("intake_sessions", {
            "title": label, "state": "staged", "totals": result, "imported": [],
            "window_revision": window["version"], "synthetic": window["synthetic"],
        }, parent=key, action="staged candidate photographs")
        db.link(session["id"], window, "target window at image intake")
        for idx, source in enumerate(scanned, start=1):
            register_scan(db, session, source, idx)
        return session_view(db, session)


def session_view(db, session):
    rows = db.list("intake_files", parent=session["id"], limit=MAX_FILES, oldest=True)
    return db.public(session) | {"files": [db.public(row) for row in sorted(rows, key=lambda i: i["sequence"])]}


def session_list(library, key):
    with library.use() as db:
        db.one(key, "windows")
        items = db.list("intake_sessions", parent=key, limit=30)
        return {"sessions": items}


def open_session(library, key):
    with library.use() as db:
        return session_view(db, db.one(key, "intake_sessions"))


def permitted_page(data, page):
    return next((row for row in data["pages"] if row["page"] == page and row["available"]), None)


def intake_options(raw, fallback):
    item = obj(raw, "selected image page")
    return {
        "source": identifier(item.get("file"), "staged image file"),
        "page": integer(item.get("page"), "page", 1, MAX_STAGED_PAGES),
        "caption": word(item.get("caption"), "caption", default="", limit=120),
        "role": enum(item.get("role"), "light assignment", ROLES, default=fallback["role"]),
        "visit": word(item.get("visit"), "visit", default=fallback["visit"], limit=70),
        "day": iso_day(item.get("day", fallback["day"]), "photographed date"),
        "notes": word(item.get("notes"), "capture notes", limit=1200, multiline=True, default=""),
    }


def select_pages(raw):
    form = obj(raw, "photo intake")
    defaults = {
        "role": enum(form.get("role"), "light assignment", ROLES, default="transmitted"),
        "visit": word(form.get("visit"), "visit", default="", limit=70),
        "day": iso_day(form.get("day"), "photographed date"),
    }
    selected = [intake_options(item, defaults) for item in collection(form.get("pages"), "selected image pages", 1, MAX_STAGED_PAGES)]
    pairs = [(item["source"], item["page"], item["role"], item["visit"]) for item in selected]
    if len(set(pairs)) != len(pairs):
        raise DeskError("Select each page and lighting assignment only once.")
    return selected


def prepare_pages(db, session, selected):
    result = []
    originals = {}
    for item in selected:
        row = db.one(item["source"], "intake_files", parent=session["id"])
        if not permitted_page(row, item["page"]):
            raise DeskError(f'Page {item["page"]} from {row["file_name"]} was excluded from intake.')
        if row["id"] not in originals:
            binary = db.media(row["id"], "original")
            if sha256(binary).hexdigest() != row["sha256"]:
                raise DeskError("A staged source failed the original-content check.")
            originals[row["id"]] = binary
        source_name = row["file_name"]
        inferred_title = source_name if row["page_count"] == 1 else f'{source_name} · page {item["page"]}'
        fields = {key: value for key, value in item.items() if key != "source"}
        fields["caption"] = fields["caption"] or inferred_title[:120]
        result.append((photo_decode(originals[row["id"]], item["page"]), fields, row))
    return result


def uniqueness(capture, photo=None, labels=None):
    if photo is None:
        return capture["original_digest"], capture["page"], capture["role"], capture["visit"]
    return photo.digest, labels["page"], labels["role"], labels["visit"]


def available_slots(db, window, prepared):
    existing = db.list("captures", parent=window, limit=MAX_WINDOW_CAPTURES)
    saved = {uniqueness(item) for item in existing}
    candidates = {uniqueness(None, image, fields) for image, fields, _ in prepared}
    if len(existing) + len(candidates - saved) > MAX_WINDOW_CAPTURES:
        raise Capacity("Select fewer distinct pages to stay within the window photo limit.")


def record_import(db, window, session, prepared):
    admitted = []
    dependencies = {}
    for image, fields, source in prepared:
        photo, duplicate = new_capture(db, window, image, source["file_name"], fields, synthetic=False)
        admitted.append({
            "capture": photo["id"], "caption": photo["caption"], "version": photo["version"], "existing": duplicate,
            "source_sha256": image.digest, "page": fields["page"], "role": fields["role"], "visit": fields["visit"],
        })
        dependencies[photo["id"]] = photo
    for entry in dependencies.values():
        db.link(session["id"], entry, "photograph admitted by this intake")
    payload = {name: session[name] for name in ("title", "window_revision", "totals", "synthetic")}
    states = Counter("existing" if item["existing"] else "new" for item in admitted)
    payload.update(state="imported", imported=admitted, new_count=states["new"], reused_count=states["existing"])
    return db.revise(session["id"], "intake_sessions", session["version"], payload, "imported staged pages as photographs")


def intake_import(library, key, raw):
    form = obj(raw, "intake selection")
    selected = select_pages(form)
    expected = revision(form)
    with library.use() as db:
        session = db.one(key, "intake_sessions")
        db.require_version(session, expected)
        if session["state"] != "staged":
            raise DeskError("This intake session already has a recorded import.")
        prepared = prepare_pages(db, session, selected)
    with library.use(write=True) as db:
        live = db.one(key, "intake_sessions")
        db.require_version(live, expected)
        window = db.one(live["parent"], "windows")
        available_slots(db, window["id"], prepared)
        saved = record_import(db, window, live, prepared)
        return session_view(db, saved)


def staged_page_picture(library, key, page):
    checked_page = integer(page, "page", 1, MAX_STAGED_PAGES)
    with library.use() as db:
        source = db.one(key, "intake_files")
        if not permitted_page(source, checked_page):
            raise DeskError("Only readable staged pages have previews.")
        photograph = photo_decode(db.media(key, "original"), checked_page)
    return resize_rgba(photograph.picture, 980)


def spreadsheet_cell(value):
    data = str(value)
    return "'" + data if data.startswith(("=", "+", "-", "@", "\t", "\r")) else data


def intake_csv(session):
    data = StringIO(newline="")
    csvout = csv.writer(data)
    csvout.writerow(("File", "Page", "Encoding", "Width", "Height", "Usable", "Outcome", "Original SHA256"))
    for source in session["files"]:
        for row in source["pages"]:
            csvout.writerow((spreadsheet_cell(source["file_name"]), row["page"], source["encoding"],
                             *row["original_extent"], row["available"], row["outcome"], source["sha256"]))
        if not source["pages"]:
            csvout.writerow((spreadsheet_cell(source["file_name"]), "", "", "", "", False,
                             source.get("explanation", source["state"]), source["sha256"]))
    if session["imported"]:
        csvout.writerow(())
        csvout.writerow(("Capture", "Caption", "Page", "Lighting", "Reused", "Source SHA256"))
        for row in session["imported"]:
            csvout.writerow((row["capture"], spreadsheet_cell(row["caption"]), row["page"], row["role"], row["existing"], row["source_sha256"]))
    return data.getvalue().encode()
