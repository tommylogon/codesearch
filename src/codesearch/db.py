"""SQLite storage: files, nodes, edges, unresolved refs, FTS5 and the vector cache."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id       INTEGER PRIMARY KEY,
    rel_path TEXT NOT NULL UNIQUE,
    language TEXT NOT NULL,
    hash     TEXT NOT NULL,
    mtime    REAL NOT NULL,
    lines    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS nodes (
    id             INTEGER PRIMARY KEY,
    file_id        INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    kind           TEXT NOT NULL,
    name           TEXT NOT NULL DEFAULT '',
    qualified_name TEXT NOT NULL DEFAULT '',
    parent_id      INTEGER,
    signature      TEXT NOT NULL DEFAULT '',
    docstring      TEXT NOT NULL DEFAULT '',
    source         TEXT NOT NULL DEFAULT '',
    line_start     INTEGER NOT NULL DEFAULT 1,
    line_end       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file_id);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);
CREATE INDEX IF NOT EXISTS idx_nodes_qname ON nodes(qualified_name);

CREATE TABLE IF NOT EXISTS edges (
    id        INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    kind      TEXT NOT NULL,
    meta      TEXT NOT NULL DEFAULT '',
    UNIQUE (source_id, target_id, kind, meta)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(source_id, kind);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(target_id, kind);

-- Raw references emitted by the parsers, resolved into edges after every index run.
CREATE TABLE IF NOT EXISTS refs (
    id          INTEGER PRIMARY KEY,
    source_id   INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    target      TEXT NOT NULL,
    qualifier   TEXT NOT NULL DEFAULT '',
    meta        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_refs_src ON refs(source_id);

-- One row per (node, part): which text was embedded, by hash.
CREATE TABLE IF NOT EXISTS node_parts (
    node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    part    TEXT NOT NULL,
    hash    INTEGER NOT NULL,
    PRIMARY KEY (node_id, part)
);
CREATE INDEX IF NOT EXISTS idx_node_parts_hash ON node_parts(part, hash);

-- Vectors keyed by content hash, so re-indexing a file never forces a re-embed.
CREATE TABLE IF NOT EXISTS vec_cache (
    part  TEXT NOT NULL,
    hash  INTEGER NOT NULL,
    model TEXT NOT NULL,
    dim   INTEGER NOT NULL,
    vec   BLOB NOT NULL,
    PRIMARY KEY (part, hash, model)
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '');
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    name, words, signature, docstring, rel_path, source,
    tokenize='unicode61 remove_diacritics 2'
);
"""

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_WORD = re.compile(r"[^A-Za-z0-9]+")


def split_words(text: str) -> str:
    """Split identifiers into words: syncGraphData and area_presence both become
    separate lowercase tokens, so a plain-English query can hit them."""
    out: list[str] = []
    for chunk in _NON_WORD.split(text):
        if not chunk:
            continue
        out.extend(p.lower() for p in _CAMEL.split(chunk) if p)
    return " ".join(out)


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    db_path = Path(path) if path else config.DB_PATH
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    conn.executescript(FTS_SCHEMA)
    return conn


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO meta(key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


def get_meta(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def fts_row(node_id: int, name: str, qname: str, signature: str, docstring: str,
            rel_path: str, source: str) -> tuple:
    """Values for one nodes_fts row. `words` holds the split identifier form of the
    name, qualified name and path so camelCase and snake_case are searchable."""
    words = split_words(f"{qname} {name} {rel_path}")
    return (node_id, name, words, signature, docstring, rel_path, source)


def insert_fts(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    conn.executemany(
        "INSERT INTO nodes_fts(rowid, name, words, signature, docstring, rel_path, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)


def delete_file(conn: sqlite3.Connection, file_id: int) -> None:
    """Remove a file and everything derived from it (FTS rows are not cascaded)."""
    ids = [r[0] for r in conn.execute("SELECT id FROM nodes WHERE file_id = ?", (file_id,))]
    conn.executemany("DELETE FROM nodes_fts WHERE rowid = ?", [(i,) for i in ids])
    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))


def stats(conn: sqlite3.Connection) -> dict:
    out = {
        "files": conn.execute("SELECT COUNT(*) FROM files").fetchone()[0],
        "nodes": conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
        "edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        "nodes_by_kind": {r[0]: r[1] for r in conn.execute(
            "SELECT kind, COUNT(*) FROM nodes GROUP BY kind ORDER BY 2 DESC")},
        "edges_by_kind": {r[0]: r[1] for r in conn.execute(
            "SELECT kind, COUNT(*) FROM edges GROUP BY kind ORDER BY 2 DESC")},
        "files_by_language": {r[0]: r[1] for r in conn.execute(
            "SELECT language, COUNT(*) FROM files GROUP BY language ORDER BY 2 DESC")},
        "vectors_by_part": {r[0]: r[1] for r in conn.execute(
            "SELECT p.part, COUNT(*) FROM node_parts p "
            "JOIN vec_cache c ON c.part = p.part AND c.hash = p.hash "
            "GROUP BY p.part")},
        "cached_vectors": conn.execute("SELECT COUNT(*) FROM vec_cache").fetchone()[0],
        "indexed_at": get_meta(conn, "indexed_at"),
    }
    return out
