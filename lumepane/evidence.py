"""Local, version-pinned evidence packs and independently checked recipient views."""

import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
import numpy as np
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from PIL import Image

from .archiveviews import Paper, contact_pages, physical_strip, print_pdf
from .catalog import capture_context
from .contracts import Capacity, DeskError, digest, enum, flag, identifier, identifiers, integer, obj, word
from .geometry import analysis_box, bounds_px
from .imaging import encode, marks_display, opened_picture, resize_rgba, srgb_display
from .repository import encoded

FORMAT = "lumepane-evidence-v1"
MAX_ARCHIVE_BYTES = 42 * 1024**2
MAX_INFLATED_BYTES = 92 * 1024**2
MAX_PACK_ITEMS = 12


def hash_bytes(raw):
    return sha256(raw).hexdigest()


def pack_source(db, capture, mode, *, include_marks):
    ctx = capture_context(db, capture["id"], registered=None if mode == "plane" else False)
    canvas = srgb_display(ctx.surface, None)
    chosen = ctx.marks if include_marks else []
    result = marks_display(canvas, chosen) if chosen else canvas
    picture = resize_rgba(result, 1500)
    travel = {
        "id": capture["id"],
        "caption": capture["caption"],
        "role": capture["role"], "visit": capture["visit"],
        "synthetic": bool(capture["synthetic"]),
        "capture_version": capture["version"],
        "source_sha256": capture["original_digest"],
        "alignment": ctx.alignment["id"] if ctx.alignment else None,
        "frame_extent": list(ctx.surface.extent),
        "display_extent": list(picture.size),
        "mark_versions": [{"id": item["id"], "version": item["version"]} for item in chosen],
    }
    links = [(capture, "included capture")]
    if ctx.alignment:
        links.append((ctx.alignment, "included alignment"))
    links.extend((item, "printed mark") for item in chosen)
    return travel, picture, links


def report_source(db, report):
    if report["kind"] != "reports":
        raise DeskError("Choose a stored examination.")
    preferred = next((x for x in ("map", "chart", "source") if x in {a["label"] for a in db.assets(report["id"])}), None)
    if not preferred:
        raise DeskError("This examination has no printable image.")
    picture = resize_rgba(opened_picture(db.media(report["id"], preferred)), 1400)
    description = {
        "id": report["id"], "caption": report["title"], "role": "examination",
        "visit": "", "synthetic": bool(report.get("synthetic")), "capture_version": report["version"],
        "source_sha256": db.asset(report["id"], preferred)["digest"], "alignment": report["basis"].get("alignment"),
        "frame_extent": report["basis"].get("extent") or list(picture.size),
        "display_extent": list(picture.size), "mark_versions": [],
    }
    return description, picture, [(report, "included examination")]


def select_entries(db, window, raw):
    payload = obj(raw)
    orientation = enum(payload.get("view"), "capture view", ("plane", "capture"), default="plane")
    tag = flag(payload.get("marks"), "include marked observations", default=True)
    chosen = identifiers(payload.get("records"), "evidence records", high=MAX_PACK_ITEMS)
    result = []
    references = {}
    for key in chosen:
        artifact = db.one(key, parent=window["id"])
        if artifact["kind"] == "captures":
            document, picture, watches = pack_source(db, artifact, orientation, include_marks=tag)
        elif artifact["kind"] == "reports":
            document, picture, watches = report_source(db, artifact)
        else:
            raise DeskError("Choose an existing capture or examination record.")
        note = document["role"].replace("_", " ")
        if document["visit"]:
            note += "  •  " + document["visit"]
        if document["synthetic"]:
            note += "  •  synthetic demonstration"
        result.append((document, picture, note))
        for item, purpose in watches:
            references[(item["id"], purpose)] = item
    links = [(item, p) for (_, p), item in references.items()]
    return result, links


def csv_index(rows):
    columns = ["caption", "visit", "role", "source_sha256", "alignment", "capture_version", "synthetic", "file"]
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(columns)
    writer.writerows([item.get(key, "") for key in columns] for item in rows)
    return output.getvalue().encode()


def contact_pdf(title, window, paper, results, row_count, columns):
    subtitle = window["title"]
    if window["synthetic"]:
        subtitle += "  •  synthetic demonstration"
    pages = contact_pages(paper, title, subtitle, results, row_count, columns)
    pdf, preview = print_pdf(pages, paper)
    if len(pdf) > MAX_ARCHIVE_BYTES:
        raise Capacity("Export fewer items or choose a smaller printing resolution.")
    return pdf, preview


def zip_media(title, window, paper, entries, pdf):
    names = {}
    documents = []
    for i, (doc, picture, note) in enumerate(entries, 1):
        filename = f"views/{i:02d}.png"
        image = encode(picture)
        names[filename] = image
        documents.append(doc | {"file": filename, "preview_sha256": hash_bytes(image)})
    names["contact.pdf"] = pdf
    names["sources.csv"] = csv_index(documents)
    manifest = {
        "format": FORMAT,
        "title": title,
        "window": {"label": window["title"], "code": window["code"],
                   "dimensions_mm": window["physical_mm"], "synthetic": window["synthetic"]},
        "paper": paper.manifest(),
        "views": documents,
        "sha256": {name: hash_bytes(binary) for name, binary in names.items()},
    }
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=4) as writer:
        for name, data in names.items():
            writer.writestr(name, data)
        writer.writestr("lumepane.json", encoded(manifest).encode())
    wire = output.getvalue()
    if len(wire) > MAX_ARCHIVE_BYTES:
        raise Capacity("The evidence file is too large. Export fewer selected views.")
    return wire, manifest


def freeze_packet(library, window_id, raw, *, bundle=True):
    args = obj(raw)
    paper = Paper.parse(args.get("format", {}))
    title = word(args.get("title"), "packet title", limit=90)
    columns = integer(args.get("columns"), "paper columns", 1, 4, default=2)
    row_count = integer(args.get("rows"), "paper rows", 1, 4, default=2)
    with library.use() as query:
        window = query.one(window_id, "windows")
        records, links = select_entries(query, window, args)
    pdf, preview = contact_pdf(title, window, paper, records, row_count, columns)
    if bundle:
        wire, manifest = zip_media(title, window, paper, records, pdf)
    else:
        wire, manifest = None, None
    document = {
        "title": title, "packet_kind": "frozen evidence" if bundle else "contact sheet",
        "format": paper.manifest(), "entries": [doc for doc, _, _ in records],
        "page_count": (len(records) + row_count * columns - 1) // (row_count * columns),
        "asset_count": len(records), "pdf_sha256": hash_bytes(pdf),
        "bundle_sha256": hash_bytes(wire) if wire else None,
        "versioned_sources": True,
        "synthetic": window["synthetic"],
    }
    with library.use(write=True) as store:
        created = store.add("packets", document, parent=window_id, action="froze inspection proof")
        key = created["id"]
        for source, role in links:
            store.link(key, source, role)
        store.attach(key, "pdf", pdf, "application/pdf")
        store.attach(key, "index", encoded(manifest or document).encode(), "application/json")
        store.attach(key, "map", encode(resize_rgba(preview, 1000)), "image/png")
        if wire:
            store.attach(key, "zip", wire, "application/zip")
        return store.public(created)


def scale_print(library, window_id, raw):
    opts = obj(raw)
    requested = identifier(opts.get("capture"), "capture")
    paper = Paper.parse(opts.get("format", {}))
    title = word(opts.get("title"), "print title", limit=85)
    with library.use() as db:
        ctx = capture_context(db, requested, registered=True)
        if ctx.window["id"] != window_id:
            raise DeskError("Select a capture from this window.")
        box = bounds_px(analysis_box(opts), ctx.surface.extent)
        view = ctx.surface.subset(box)
        if np.count_nonzero(view.valid) < 20:
            raise DeskError("The crop contains too little photographed image.")
        size = ctx.scale * view.extent if ctx.scale is not None else None
        output, position = physical_strip(ctx, srgb_display(view, None), paper, title, size)
    data, _ = print_pdf([output], paper)
    size = [round(float(v), 5) for v in size] if size is not None else None
    result = {
        "title": title, "packet_kind": "detail scale proof", "format": paper.manifest(),
        "window_box": list(box), "plot_box_on_paper_px": list(position), "physical_crop_mm": size,
        "pdf_sha256": hash_bytes(data), "page_count": 1, "asset_count": 1,
        "alignment": ctx.alignment["id"], "notes": "Print at 100% and verify the 20 mm line.",
    }
    with library.use(write=True) as db:
        created = db.add("packets", result, parent=window_id, action="froze a physical scale proof")
        key = created["id"]
        db.link(key, ctx.capture, "printed capture")
        db.link(key, ctx.alignment, "printed flat position")
        db.attach(key, "map", encode(resize_rgba(output, 1000)), "image/png")
        db.attach(key, "pdf", data, "application/pdf")
        db.attach(key, "index", encoded(result).encode(), "application/json")
        return db.public(created)


def acceptable_name(name):
    key = PurePosixPath(name)
    if (key.is_absolute() or ".." in key.parts or str(key) != name or "\\" in name
            or ":" in name or any(ord(char) < 32 for char in name)):
        raise DeskError("A packet contains an invalid member path.")
    return name


def list_sizes(archive):
    sizes = {}
    total = 0
    for item in archive.infolist():
        name = acceptable_name(item.filename)
        if name in sizes or item.is_dir():
            raise DeskError("A packet contains duplicate or unsupported member records.")
        sizes[name] = item.file_size
        total += item.file_size
        if item.file_size > MAX_ARCHIVE_BYTES or total > MAX_INFLATED_BYTES:
            raise Capacity("A packet expands beyond the examination allowance.")
    if not 4 <= len(sizes) <= MAX_PACK_ITEMS + 3:
        raise DeskError("Unexpected number of materials in the packet.")
    return sizes


def open_manifest(archive, sizes):
    if not 40 <= sizes.get("lumepane.json", 0) <= 900_000:
        raise DeskError("A LumePane manifest is missing or too large.")
    try:
        manifest = json.loads(archive.read("lumepane.json"))
    except (ValueError, UnicodeError) as error:
        raise DeskError("The LumePane manifest cannot be decoded.") from error
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise DeskError("This packet was not written in the supported LumePane format.")
    batch = manifest.get("views")
    if not isinstance(batch, list) or not 1 <= len(batch) <= MAX_PACK_ITEMS:
        raise DeskError("The manifest contains no supported photo list.")
    values = manifest.get("sha256")
    if not isinstance(values, dict) or set(values) != set(sizes) - {"lumepane.json"}:
        raise DeskError("The member list differs from the manifest.")
    return manifest


def verified_views(archive, manifest):
    checksum = manifest["sha256"]
    attached = {}
    paths = {"contact.pdf", "sources.csv"}
    for entry in manifest["views"]:
        if not isinstance(entry, dict):
            raise DeskError("A view entry is incomplete.")
        item = entry.get("file")
        if not isinstance(item, str) or not item.startswith("views/") or not item.endswith(".png") or item in paths:
            raise DeskError("A packet view path is not a unique PNG member.")
        paths.add(item)
        if digest(entry.get("preview_sha256"), "view hash") != checksum.get(item):
            raise DeskError("The declared picture hash does not match its view entry.")
        digest(entry.get("source_sha256"), "source hash")
        word(entry.get("caption"), "caption", limit=120)
        size = entry.get("display_extent")
        if not isinstance(size, list) or len(size) != 2 or not all(type(v) is int and 1 <= v <= 1500 for v in size):
            raise DeskError("A packet view has unsafe or missing image dimensions.")
    if paths != set(checksum):
        raise DeskError("The packet has undeclared materials. No views are imported.")
    for attr in paths:
        expected = digest(checksum[attr], "member hash")
        raw = archive.read(attr)
        if hash_bytes(raw) != expected:
            raise DeskError(f"A saved member does not match its hash ({attr}).")
        attached[attr] = raw
    if not attached["contact.pdf"].startswith(b"%PDF-"):
        raise DeskError("The packet has no recognizable contact PDF.")
    for entry in manifest["views"]:
        data = attached[entry["file"]]
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or list(image.size) != entry["display_extent"]:
                raise DeskError("A view does not have the declared PNG format and dimensions.")
            image.verify()
    return attached


def recipient_payload(raw):
    if len(raw) > MAX_ARCHIVE_BYTES:
        raise Capacity("Choose a local LumePane ZIP smaller than 42 MiB.")
    try:
        with ZipFile(BytesIO(raw)) as packet:
            sizes = list_sizes(packet)
            manifest = open_manifest(packet, sizes)
            attachments = verified_views(packet, manifest)
            title = word(manifest.get("title"), "received packet title", limit=90)
    except (BadZipFile, OSError, RuntimeError, KeyError) as error:
        raise DeskError("The packet cannot be opened as a complete archive.") from error
    return manifest, attachments, title


def receive_archive(library, file):
    if file is None:
        raise DeskError("Choose a local evidence ZIP.")
    raw = file.stream.read(MAX_ARCHIVE_BYTES + 1)
    manifest, files, title = recipient_payload(raw)
    checksum = hash_bytes(raw)
    document = {
        "title": title, "status": "verified local packet", "packet_sha256": checksum,
        "source_window": manifest["window"], "views": manifest["views"],
        "verified_members": len(files), "original_count": len(manifest["views"]),
        "note": "Stored views are frozen received evidence; they do not edit any local examination.",
    }
    with library.use(write=True) as db:
        for record in db.list("received", limit=250):
            if record["packet_sha256"] == checksum:
                return db.public(record)
        record = db.add("received", document, action="verified and saved a received packet")
        key = record["id"]
        first = files[manifest["views"][0]["file"]]
        db.attach(key, "zip", raw, "application/zip")
        db.attach(key, "pdf", files["contact.pdf"], "application/pdf")
        db.attach(key, "map", first, "image/png")
        db.attach(key, "index", encoded(manifest).encode(), "application/json")
        return db.public(record)


def received_view(library, key, item):
    with library.use() as db:
        doc = db.one(key, "received")
        if not 1 <= item <= len(doc["views"]):
            raise DeskError("The picture number is outside the received index.")
        path = doc["views"][item - 1]["file"]
        with ZipFile(BytesIO(db.media(key, "zip"))) as archive:
            return archive.read(path)
