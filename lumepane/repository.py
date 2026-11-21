"""SQLite manifest and immutable, hash-addressed media storage."""

from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from uuid import uuid4

from .contracts import DeskError, Missing, Stale, now

KINDS = frozenset({
    "windows", "captures", "marks", "alignments", "treatments",
    "reports", "references", "packets", "received", "alignment_checks", "condition_sheets", "intake_sessions", "intake_files",
    "panoramas",
})
MEDIA_LABELS = frozenset({
    "original", "working", "thumbnail", "map", "photo", "plane", "source",
    "support", "chart", "diff", "pdf", "index", "zip", "receipt", "changes",
})
SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS objects (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    parent TEXT REFERENCES objects(id),
    version INTEGER NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS objects_by_parent ON objects (kind, parent, created, id);
CREATE TABLE IF NOT EXISTS history (
    owner TEXT NOT NULL REFERENCES objects(id),
    version INTEGER NOT NULL,
    stamp TEXT NOT NULL,
    action TEXT NOT NULL,
    record TEXT NOT NULL,
    PRIMARY KEY (owner, version)
);
CREATE TABLE IF NOT EXISTS assets (
    owner TEXT NOT NULL REFERENCES objects(id),
    label TEXT NOT NULL,
    digest TEXT NOT NULL,
    mime TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    PRIMARY KEY (owner, label)
);
CREATE INDEX IF NOT EXISTS assets_by_digest ON assets (digest);
CREATE TABLE IF NOT EXISTS links (
    owner TEXT NOT NULL REFERENCES objects(id),
    source TEXT NOT NULL REFERENCES objects(id),
    source_version INTEGER NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY (owner, source, role)
);
CREATE INDEX IF NOT EXISTS links_by_source ON links (source, owner);
"""


def encoded(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def kind_name(value):
    if value not in KINDS:
        raise DeskError("That record type cannot be edited.")
    return value


class Blobs:
    """The database commits references after complete blob writes."""

    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def address(self, digest):
        return self.root / digest[:2] / digest[2:]

    def put(self, payload):
        digest = sha256(payload).hexdigest()
        path = self.address(digest)
        if path.exists():
            if path.stat().st_size != len(payload):
                raise OSError("A stored object differs from its digest.")
            return digest
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temp = Path(stream.name)
            try:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
        return digest

    def require(self, digest):
        path = self.address(digest)
        if not path.is_file():
            raise Missing("The saved media is unavailable.")
        return path

    def verify(self, digest):
        path = self.require(digest)
        grab = sha256()
        with path.open("rb") as stream:
            for part in iter(lambda: stream.read(131072), b""):
                grab.update(part)
        return grab.hexdigest() == digest


def hydrated(row):
    body = json.loads(row["record"])
    return body | {key: row[key] for key in ("id", "kind", "parent", "version", "created", "updated")}


class Library:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_file = self.root / "lumepane.sqlite3"
        self.blobs = Blobs(self.root / "media")
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_file, timeout=12, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def use(self, *, write=False):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            query = Query(db, self.blobs)
            try:
                yield query
                db.commit()
            except BaseException:
                db.rollback()
                raise


class Query:
    def __init__(self, db, blobs):
        self.db = db
        self.blobs = blobs

    def exists(self, key, kind=None):
        row = self.db.execute("SELECT kind FROM objects WHERE id = ?", (key,)).fetchone()
        return bool(row and (kind is None or row["kind"] == kind))

    def one(self, key, kind=None, *, parent=None):
        row = self.db.execute("SELECT * FROM objects WHERE id = ?", (key,)).fetchone()
        if row is None or kind is not None and row["kind"] != kind:
            raise Missing("This record is unavailable.")
        if parent is not None and row["parent"] != parent:
            raise Missing("This record belongs to a different window.")
        return hydrated(row)

    def many(self, keys, kind=None, *, parent=None):
        return [self.one(key, kind, parent=parent) for key in keys]

    def list(self, kind, *, parent=None, limit=120, oldest=False):
        query = "SELECT * FROM objects WHERE kind = ?"
        values = [kind_name(kind)]
        if parent is not None:
            query += " AND parent = ?"
            values.append(parent)
        query += " ORDER BY created ASC, id ASC" if oldest else " ORDER BY created DESC, id DESC"
        query += " LIMIT ?"
        values.append(limit)
        return [hydrated(row) for row in self.db.execute(query, values)]

    def children_counts(self, owners, kind):
        if not owners:
            return {}
        matches = ",".join("?" for _ in owners)
        sql = f"SELECT parent, count(*) AS n FROM objects WHERE kind = ? AND parent IN ({matches}) GROUP BY parent"
        return {
            row["parent"]: row["n"]
            for row in self.db.execute(sql, (kind_name(kind), *owners))
        }

    def parent_counts(self, parent):
        rows = self.db.execute(
            "SELECT kind, count(*) AS n FROM objects WHERE parent = ? GROUP BY kind", (parent,)
        )
        return dict((row["kind"], row["n"]) for row in rows)

    def add(self, kind, payload, *, parent=None, action="created"):
        kind_name(kind)
        key = uuid4().hex[:24]
        stamp = now()
        binary = encoded(payload)
        self.db.execute(
            "INSERT INTO objects (id,kind,parent,version,created,updated,record) VALUES (?,?,?,?,?,?,?)",
            (key, kind, parent, 1, stamp, stamp, binary),
        )
        self.db.execute("INSERT INTO history VALUES (?,?,?,?,?)", (key, 1, stamp, action, binary))
        return self.one(key, kind)

    def require_version(self, record, version):
        if record["version"] != version:
            raise Stale()

    def revise(self, key, kind, version, payload, action):
        current = self.one(key, kind)
        self.require_version(current, version)
        binary = encoded(payload)
        stamp = now()
        following = version + 1
        self.db.execute(
            "UPDATE objects SET record = ?, updated = ?, version = ? WHERE id = ?",
            (binary, stamp, following, key),
        )
        self.db.execute("INSERT INTO history VALUES (?,?,?,?,?)", (key, following, stamp, action, binary))
        return self.one(key, kind)

    def series(self, owner):
        self.one(owner)
        return [{
            "version": row["version"],
            "stamp": row["stamp"],
            "action": row["action"],
            "record": json.loads(row["record"]),
        } for row in self.db.execute("SELECT * FROM history WHERE owner=? ORDER BY version DESC", (owner,))]

    def attach(self, owner, label, payload, mime):
        if label not in MEDIA_LABELS:
            raise DeskError("Unsupported media name.")
        previous = self.db.execute(
            "SELECT label FROM assets WHERE owner = ? AND label = ?", (owner, label)
        ).fetchone()
        if previous:
            raise Stale("Saved media cannot be overwritten.")
        handle = self.blobs.put(payload)
        self.db.execute(
            "INSERT INTO assets VALUES (?,?,?,?,?)", (owner, label, handle, mime, len(payload)),
        )
        return self.asset(owner, label)

    def asset(self, owner, label):
        row = self.db.execute("SELECT * FROM assets WHERE owner=? AND label=?", (owner, label)).fetchone()
        if not row:
            raise Missing("This media is not part of the saved result.")
        return dict(row)

    def assets(self, owner):
        return [dict(row) for row in self.db.execute("SELECT * FROM assets WHERE owner=? ORDER BY label", (owner,))]

    def media(self, owner, label):
        item = self.asset(owner, label)
        return self.blobs.require(item["digest"]).read_bytes()

    def media_location(self, owner, label):
        item = self.asset(owner, label)
        return self.blobs.require(item["digest"]), item

    def link(self, owner, source, role):
        self.db.execute(
            "INSERT INTO links VALUES (?,?,?,?)", (owner, source["id"], source["version"], role),
        )

    def linked(self, owner):
        return [dict(row) for row in self.db.execute(
            "SELECT source,source_version,role FROM links WHERE owner=? ORDER BY role,source", (owner,)
        )]

    def public(self, document):
        key = document["id"]
        return document | {"assets": self.assets(key), "inputs": self.linked(key)}

    def audit(self, key):
        record = self.one(key)
        rows = []
        for item in self.assets(key):
            try:
                ok = self.blobs.verify(item["digest"])
            except Missing:
                ok = False
            rows.append(item | {"hash_matches": ok})
        links = self.linked(key)
        current_versions = Counter()
        for link in links:
            source = self.one(link["source"])
            current_versions["advanced" if source["version"] != link["source_version"] else "same"] += 1
        return {
            "id": key, "kind": record["kind"], "version": record["version"],
            "assets": rows, "history_entries": len(self.series(key)),
            "source_revisions": dict(current_versions),
        }
