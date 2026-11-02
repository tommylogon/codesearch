"""Multi-part embedding: four vector indexes, one per question you can ask.

    filename       what file is this about            (path + file summary)
    function_name  what is this thing called          (qualified name, words split)
    function       what does it look like / do        (signature, decorators, summary)
    comment        what do the humans say about it    (docstrings, comments, doc prose)

Vectors live in the `vec_cache` table keyed by (part, content hash, model), so a file
that is re-indexed re-embeds only text that actually changed. `node_parts` maps each node
to the hash of its text per part. Vectors are L2-normalised: cosine == dot product.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from . import config
from .db import split_words


def text_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=True)


def _clip(text: str) -> str:
    """Bound the text sent to the embedding model (its context window). The stored
    text is never shortened."""
    return text[: config.EMBED_MAX_CHARS]


def part_texts(node: sqlite3.Row) -> dict[str, str]:
    """The text each part embeds for one node. Absent part == nothing worth embedding."""
    kind = node["kind"]
    out: dict[str, str] = {}
    name, qname, sig, doc = node["name"], node["qualified_name"], node["signature"], node["docstring"]
    if kind == "file":
        out["filename"] = f"{node['rel_path']}\n{doc}".strip()
        return out
    if kind == "comment":
        out["comment"] = doc
        return out
    if kind in ("function", "method", "class", "variable", "element"):
        local = qname.split("::", 1)[1] if "::" in qname else name
        out["function_name"] = f"{local} {split_words(local)}".strip()
    if kind in ("function", "method", "class"):
        out["function"] = f"{sig}\n{doc.splitlines()[0] if doc else ''}".strip()
    if kind == "section":
        local = qname.split("::", 1)[1] if "::" in qname else name
        out["function_name"] = local.replace("::", " > ")
        if doc:
            out["comment"] = f"{name}\n{doc}"
    elif doc:
        out["comment"] = doc
    if kind == "block" and node["source"]:
        out["function"] = f"{sig}\n{node['source']}"
    return {k: v for k, v in out.items() if v and v.strip()}


def _normalise(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


def embed_texts(texts: list[str], timeout: int | None = None) -> np.ndarray:
    url = f"{config.EMBED_ENDPOINT.rstrip('/')}/embeddings"
    clipped = [_clip(t) for t in texts]
    for attempt in range(3):
        resp = requests.post(url, json={"model": config.EMBED_MODEL, "input": clipped},
                             timeout=timeout or config.EMBED_TIMEOUT)
        if resp.status_code == 200:
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            return _normalise(np.array([d["embedding"] for d in data], dtype=np.float32))
        if resp.status_code >= 500 and len(texts) > 1:
            mid = len(texts) // 2
            return np.vstack([embed_texts(texts[:mid], timeout=timeout),
                              embed_texts(texts[mid:], timeout=timeout)])
        time.sleep(0.5 * (attempt + 1))
    resp.raise_for_status()
    return np.empty((0, 0), dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    text = f"Instruct: {config.EMBED_QUERY_INSTRUCTION}\nQuery: {query}"
    return embed_texts([text], timeout=10)[0]


def preflight() -> None:
    try:
        r = requests.get(f"{config.EMBED_ENDPOINT.rstrip('/')}/models", timeout=10)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"embedding endpoint {config.EMBED_ENDPOINT} unreachable: {exc}") from exc
    ids = [m["id"] for m in r.json().get("data", [])]
    if config.EMBED_MODEL not in ids:
        raise RuntimeError(f"model {config.EMBED_MODEL!r} not served at {config.EMBED_ENDPOINT}")


def _path_tier(rel_path: str) -> int:
    """Order embedding work so runtime code and architecture docs are embedded before
    test suites and historical task cards."""
    if rel_path.startswith(("engine/", "routes/", "static/", "templates/", "tools/codesearch/")):
        return 0
    if "/" not in rel_path:
        return 0
    if rel_path.startswith("docs/") and not rel_path.startswith("docs/virtualWorld/dev_tasks/"):
        return 1
    if rel_path.startswith("tools/"):
        return 1
    if rel_path.startswith(("tests/", "data/")):
        return 2
    return 3


def _default_progress(msg: str) -> None:
    print(msg, flush=True)


def build(conn: sqlite3.Connection, progress=_default_progress, limit: int | None = None,
          path_prefix: str | None = None) -> dict:
    """Compute node_parts for target nodes and embed any hash not already cached."""
    prefix = path_prefix.replace("\\", "/").rstrip("/") if path_prefix else None
    if prefix:
        rows = conn.execute(
            "SELECT n.id, n.kind, n.name, n.qualified_name, n.signature, n.docstring, n.source, f.rel_path "
            "FROM nodes n JOIN files f ON f.id = n.file_id "
            "WHERE f.rel_path = ? OR f.rel_path LIKE ?",
            (prefix, prefix + "/%")).fetchall()
    else:
        rows = conn.execute(
            "SELECT n.id, n.kind, n.name, n.qualified_name, n.signature, n.docstring, n.source, f.rel_path "
            "FROM nodes n JOIN files f ON f.id = n.file_id").fetchall()
    rows = sorted(rows, key=lambda r: (_path_tier(r["rel_path"]), r["rel_path"], r["id"]))
    pending: dict[tuple[str, int], str] = {}
    part_rows: list[tuple] = []
    for row in rows:
        for part, text in part_texts(row).items():
            h = text_hash(text)
            part_rows.append((row["id"], part, h))
            if (part, h) not in pending:
                pending[(part, h)] = text
    conn.executemany("INSERT OR REPLACE INTO node_parts(node_id, part, hash) VALUES (?,?,?)", part_rows)
    conn.commit()
    cached = {(r[0], r[1]) for r in conn.execute(
        "SELECT part, hash FROM vec_cache WHERE model = ?", (config.EMBED_MODEL,))}
    todo = [(k, t) for k, t in pending.items() if k not in cached]
    already = len(pending) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    progress(f"embedding: {len(pending)} distinct texts, {already} cached, {len(todo)} to embed"
             + (f" (limited from {len(pending) - already})" if limit is not None else ""))
    if todo:
        preflight()
    start = time.time()
    done = 0
    batches = [todo[i:i + config.EMBED_BATCH] for i in range(0, len(todo), config.EMBED_BATCH)]

    def run(batch):
        return batch, embed_texts([t for _, t in batch])

    # Parallel requests overlap tokenisation and transfer with GPU work. Results are
    # committed from this thread only, in completion order, so a kill costs a few batches.
    with ThreadPoolExecutor(max_workers=config.EMBED_WORKERS) as pool:
        for n, (batch, vecs) in enumerate(pool.map(run, batches), 1):
            conn.executemany(
                "INSERT OR REPLACE INTO vec_cache(part, hash, model, dim, vec) VALUES (?,?,?,?,?)",
                [(k[0], k[1], config.EMBED_MODEL, int(vecs.shape[1]), vecs[j].tobytes())
                 for j, (k, _) in enumerate(batch)])
            conn.commit()
            done += len(batch)
            if n % 5 == 0 or n == len(batches):
                rate = done / max(time.time() - start, 1e-6)
                progress(f"  {done}/{len(todo)} embedded ({rate:.1f}/s)")
    conn.commit()
    return {"texts": len(pending), "embedded": done, "cached": already}


class VectorIndex:
    """In-memory matrices per part, loaded from the database."""

    def __init__(self, conn: sqlite3.Connection):
        self.parts: dict[str, dict] = {}
        for part in config.PARTS:
            rows = conn.execute(
                "SELECT p.node_id, c.vec, c.dim FROM node_parts p "
                "JOIN vec_cache c ON c.part = p.part AND c.hash = p.hash AND c.model = ? "
                "WHERE p.part = ?", (config.EMBED_MODEL, part)).fetchall()
            if not rows:
                continue
            dim = rows[0]["dim"]
            ids = np.array([r["node_id"] for r in rows], dtype=np.int64)
            mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32, count=dim) for r in rows])
            self.parts[part] = {"ids": ids, "vectors": mat}

    def available(self) -> bool:
        return bool(self.parts)

    def top(self, part: str, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        p = self.parts.get(part)
        if p is None:
            return []
        scores = p["vectors"] @ qvec
        k = min(k, scores.shape[0])
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(p["ids"][i]), float(scores[i])) for i in idx]
