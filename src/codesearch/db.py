"""SQLite storage: files, nodes, edges, unresolved refs, FTS5 and the vector cache."""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

from . import config, diag

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
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);

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
CREATE INDEX IF NOT EXISTS idx_vec_cache_model ON vec_cache(model, part, hash);

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


_REQUIRED_OBJECTS = frozenset({
    "files", "nodes", "edges", "refs", "node_parts", "vec_cache", "meta",
    "nodes_fts", "idx_nodes_parent", "idx_vec_cache_model",
})


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    t0 = time.perf_counter()
    db_path = Path(path) if path else config.DB_PATH
    is_mem = str(db_path) == ":memory:"
    if not is_mem:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -65536")
    if not is_mem:
        mode = (conn.execute("PRAGMA journal_mode").fetchone() or [""])[0]
        if str(mode).lower() != "wal":
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    existing = {
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'index')").fetchall()
    }
    missing = sorted(_REQUIRED_OBJECTS - existing)
    if missing:
        log = diag.get_logger("db")
        sm_t0 = time.perf_counter()
        log.info("DB_SCHEMA_INIT path=%s missing=%s", db_path, missing)
        diag.flush()
        with diag.watch_slow(f"db_schema_init:{db_path}", logger=log):
            conn.executescript(SCHEMA)
            conn.executescript(FTS_SCHEMA)
        log.info("DB_SCHEMA_READY path=%s elapsed_ms=%.1f", db_path, (time.perf_counter() - sm_t0) * 1000.0)
        diag.flush()
    else:
        diag.get_logger("db").debug("DB_CONNECT path=%s elapsed_ms=%.2f", db_path, (time.perf_counter() - t0) * 1000.0)
    return conn


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO meta(key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


def get_meta(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def bump_vec_generation(conn: sqlite3.Connection) -> str:
    """Increment the O(1) vector-pack invalidation stamp in meta."""
    cur = get_meta(conn, "vec_generation", "0")
    try:
        nxt = str(int(cur) + 1)
    except ValueError:
        nxt = "1"
    set_meta(conn, "vec_generation", nxt)
    return nxt


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


def _active_vector_coverage(conn: sqlite3.Connection) -> tuple[dict[str, int], dict[str, int]]:
    """Compute active-model vector coverage per part, cached in meta by vec_generation."""
    import json

    gen = get_meta(conn, "vec_generation", "")
    vc_max = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM vec_cache").fetchone()[0]
    cache_key = f"{gen}:{vc_max}:{config.EMBED_MODEL}:{config.IMAGE_EMBED_MODEL}"
    raw_cached = get_meta(conn, "stats_vec_coverage", "")
    if raw_cached:
        try:
            parsed = json.loads(raw_cached)
            if parsed.get("key") == cache_key:
                return dict(parsed["active"]), dict(parsed["total"])
        except Exception:
            pass

    parts_total = {
        r[0]: int(r[1])
        for r in conn.execute("SELECT part, COUNT(*) FROM node_parts GROUP BY part")
    }
    active_by_part: dict[str, int] = {}
    for part, total in parts_total.items():
        if total == 0:
            continue
        model = config.model_for_part(part)
        cnt = conn.execute(
            "SELECT COUNT(*) FROM node_parts p "
            "JOIN vec_cache c ON c.part = p.part AND c.hash = p.hash AND c.model = ? "
            "WHERE p.part = ?",
            (model, part),
        ).fetchone()[0]
        if cnt:
            active_by_part[part] = int(cnt)

    try:
        set_meta(conn, "stats_vec_coverage", json.dumps({"key": cache_key, "active": active_by_part, "total": parts_total}))
        conn.commit()
    except sqlite3.Error:
        pass
    return active_by_part, parts_total


def prune_stale_vectors(conn: sqlite3.Connection) -> dict[str, int]:
    """Delete vec_cache rows belonging to inactive embedding models."""
    active_models = (config.EMBED_MODEL, config.IMAGE_EMBED_MODEL)
    stale = {
        r[0]: int(r[1])
        for r in conn.execute(
            "SELECT model, COUNT(*) FROM vec_cache WHERE model NOT IN (?, ?) GROUP BY model",
            active_models,
        )
    }
    if stale:
        conn.execute("DELETE FROM vec_cache WHERE model NOT IN (?, ?)", active_models)
        bump_vec_generation(conn)
        conn.commit()
    return stale


def stats(conn: sqlite3.Connection) -> dict:
    vectors_by_part, parts_total_by_part = _active_vector_coverage(conn)
    cached_by_model = {
        r[0]: int(r[1])
        for r in conn.execute("SELECT model, COUNT(*) FROM vec_cache GROUP BY model ORDER BY 2 DESC")
    }
    active_models = {config.EMBED_MODEL, config.IMAGE_EMBED_MODEL}
    stale_models = {m: c for m, c in cached_by_model.items() if m not in active_models}
    coverage_pct: dict[str, float] = {}
    for part, total in parts_total_by_part.items():
        if total > 0:
            coverage_pct[part] = round(100.0 * vectors_by_part.get(part, 0) / total, 1)

    warnings: list[str] = []
    if stale_models:
        stale_desc = ", ".join(f"{m}={c}" for m, c in stale_models.items())
        warnings.append(f"Orphaned vec_cache rows for inactive model(s): {stale_desc}")
    incomplete = [f"{p}:{pct}%" for p, pct in coverage_pct.items() if pct < 100.0]
    if incomplete and vectors_by_part:
        warnings.append(f"Partial active-model coverage ({config.EMBED_MODEL}): {', '.join(incomplete)}")

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
        "embed_model": config.EMBED_MODEL,
        "image_embed_model": config.IMAGE_EMBED_MODEL,
        "vectors_by_part": vectors_by_part,
        "parts_total_by_part": parts_total_by_part,
        "active_coverage_pct": coverage_pct,
        "cached_vectors": sum(cached_by_model.values()),
        "cached_vectors_by_model": cached_by_model,
        "stale_models": stale_models,
        "indexed_at": get_meta(conn, "indexed_at"),
    }
    if warnings:
        out["stale_model_warning"] = "; ".join(warnings)
    return out
