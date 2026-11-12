"""Multi-part embedding: five vector indexes, one per question you can ask.

    filename       what file is this about            (path + file summary)
    function_name  what is this thing called          (qualified name, words split)
    function       what does it look like / do        (signature, decorators, summary)
    comment        what do the humans say about it    (docstrings, comments, doc prose)
    image          what does the picture look like    (the pixels themselves)

Text parts go to the code-text model (CODESEARCH_EMBED_ENDPOINT). The image part goes to
a multimodal model at CODESEARCH_IMAGE_EMBED_ENDPOINT, because a text embedder cannot see
pixels. The two spaces are never compared to each other: `search.py` fuses their ranked
lists with RRF, which only needs ranks.

Vectors live in the `vec_cache` table keyed by (part, content hash, model), so a file
that is re-indexed re-embeds only content that actually changed. `node_parts` maps each
node to the hash of its text (or bytes) per part. Vectors are L2-normalised: cosine ==
dot product.
"""
from __future__ import annotations

import base64
import hashlib
import mimetypes
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from . import config
from .db import split_words


def text_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=True)


def bytes_hash(data: bytes) -> int:
    """Identity for a binary part (an image file), same storage shape as text_hash."""
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=True)


def hex_to_i64(hex_hash: str) -> int:
    """The files table stores a 16-byte hex content hash; node_parts.hash is a signed
    64-bit int. Truncating the digest is deterministic, so the same bytes always map to
    the same key -- and it lets the embedder reuse the hash the indexer already computed
    instead of re-reading every image from disk."""
    v = int(hex_hash[:16], 16)
    return v - (1 << 64) if v >= (1 << 63) else v


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
    if kind in ("record", "tag"):
        # Keyword-only kinds. A record's docstring is a bag of field names and a tag is a
        # one-word label: giving either a vector lets it outrank real prose.
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


def embed_texts(texts: list[str], timeout: int | None = None,
                endpoint: str | None = None, model: str | None = None) -> np.ndarray:
    url = f"{(endpoint or config.EMBED_ENDPOINT).rstrip('/')}/embeddings"
    payload_model = model or config.EMBED_MODEL
    clipped = [_clip(t) for t in texts]
    for attempt in range(3):
        resp = requests.post(url, json={"model": payload_model, "input": clipped},
                             timeout=timeout or config.EMBED_TIMEOUT)
        if resp.status_code == 200:
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            return _normalise(np.array([d["embedding"] for d in data], dtype=np.float32))
        if resp.status_code >= 500 and len(texts) > 1:
            mid = len(texts) // 2
            return np.vstack([embed_texts(texts[:mid], timeout=timeout, endpoint=endpoint, model=model),
                              embed_texts(texts[mid:], timeout=timeout, endpoint=endpoint, model=model)])
        time.sleep(0.5 * (attempt + 1))
    resp.raise_for_status()
    return np.empty((0, 0), dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    text = f"{query}"
    return embed_texts([text], timeout=300)[0]


# --------------------------------------------------------------------------- images
def _image_uri(path: Path | str) -> str:
    """An image as a data URI: what the multimodal sidecar accepts as an input item."""
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _is_placeholder(path: Path | str) -> bool:
    """A Google Drive streaming placeholder is a file of the right size filled with zeros:
    the photo is in the cloud but was never downloaded locally. PIL cannot decode it, and
    sending it wastes a round trip, so it is recognised here and skipped."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
    except OSError:
        return True
    if not head:
        return True
    return not any(head)


def _post_images(items: list, timeout: int | None = None) -> tuple[np.ndarray, list[int], list[dict]]:
    """Post a batch of items to the sidecar.

    Returns (vectors, original_indices, failures). Indices are the positions in `items`,
    so a partial success still lines up with the caller's file list.
    """
    url = f"{config.IMAGE_EMBED_ENDPOINT.rstrip('/')}/embeddings"
    last = ""
    for attempt in range(3):
        resp = requests.post(url, json={"model": config.IMAGE_EMBED_MODEL, "input": items},
                             timeout=timeout or config.IMAGE_TIMEOUT)
        if resp.status_code == 200:
            body = resp.json()
            data = sorted(body.get("data", []), key=lambda d: d["index"])
            failures = body.get("errors", [])
            if not data:
                return np.empty((0, 0), dtype=np.float32), [], failures
            vecs = _normalise(np.array([d["embedding"] for d in data], dtype=np.float32))
            return vecs, [d["index"] for d in data], failures
        last = resp.text[:200]
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"image embedding request failed: {last}")


def _embed_chunk(paths: list[Path]) -> tuple[list[np.ndarray | None], list[dict]]:
    """Embed one batch, isolating failures instead of losing the whole batch.

    If the sidecar rejects the batch outright, the batch is halved and retried, so a
    single bad file (or an out-of-memory batch) costs one image, not eight.
    """
    if not paths:
        return [], []
    try:
        vecs, indices, failures = _post_images(
            [{"type": "image_url", "image_url": {"url": _image_uri(p)}} for p in paths])
    except Exception as exc:
        if len(paths) == 1:
            return [None], [{"path": str(paths[0]), "error": str(exc)}]
        mid = len(paths) // 2
        left, lfail = _embed_chunk(paths[:mid])
        right, rfail = _embed_chunk(paths[mid:])
        return left + right, lfail + rfail

    out: list[np.ndarray | None] = [None] * len(paths)
    for vec, idx in zip(vecs, indices):
        out[idx] = vec
    errors = [{"path": str(paths[f["index"]]), "error": f.get("error", "unknown")}
              for f in failures if 0 <= f["index"] < len(paths)]
    # Items the sidecar silently dropped (no vector, no error) count as failures too.
    for i, v in enumerate(out):
        if v is None and not any(i == f["index"] for f in failures):
            errors.append({"path": str(paths[i]), "error": "no vector returned"})
    return out, errors


def embed_images(paths: list[Path | str], timeout: int | None = None) -> np.ndarray:
    """One vector per image, in batches small enough to keep the sidecar's GPU busy
    without blowing its context (each image costs ~280 vision tokens)."""
    paths = [Path(p) for p in paths]
    chunks = [paths[i:i + config.IMAGE_BATCH] for i in range(0, len(paths), config.IMAGE_BATCH)]

    def run(chunk):
        return _embed_chunk(chunk)

    # Base64-encoding and JSON-serialising a photo is CPU work; the GPU work is on the
    # other side of the socket. Overlapping several requests keeps both busy instead of
    # alternating between them.
    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool:
            results = list(pool.map(run, chunks))
    else:
        results = [run(chunks[0])] if chunks else []

    rows: list[np.ndarray] = []
    for vecs, _errors in results:
        rows.extend(v for v in vecs if v is not None)
    return np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)


def image_query(query: str) -> np.ndarray:
    """A text query in the image space. Same model, same 768d space, so this vector is
    directly comparable to the stored image vectors."""
    vecs, _idx, _fails = _post_images([query], timeout=300)
    return vecs[0] if len(vecs) else np.empty((0,), dtype=np.float32)


def preflight(endpoint: str | None = None, model: str | None = None) -> None:
    endpoint = endpoint or config.EMBED_ENDPOINT
    model = model or config.EMBED_MODEL
    try:
        r = requests.get(f"{endpoint.rstrip('/')}/models", timeout=300)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"embedding endpoint {endpoint} unreachable: {exc}") from exc
    ids = [m["id"] for m in r.json().get("data", [])]
    if model not in ids:
        raise RuntimeError(f"model {model!r} not served at {endpoint}")


def preflight_images() -> None:
    preflight(config.IMAGE_EMBED_ENDPOINT, config.IMAGE_EMBED_MODEL)


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
          path_prefix: str | None = None, images: bool = True) -> dict:
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
        try:
            preflight()
        except RuntimeError as exc:
            # The text model being down must not block the image index: they are separate
            # models in separate spaces, and the image sidecar can still be serving.
            progress(f"text embedding skipped: {exc}")
            todo = []
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
            if n % 2 == 0 or n == len(batches):
                elapsed = max(time.time() - start, 1e-6)
                rate = done / elapsed
                eta = (len(todo) - done) / max(rate, 1e-6)
                progress(f"  text {done}/{len(todo)} embedded ({rate:.1f}/s, eta {eta/60:.1f}m)")
    conn.commit()
    out = {"texts": len(pending), "embedded": done, "cached": already}
    if images:
        out.update(_build_images(conn, progress, limit, prefix))
    return out


def _build_images(conn: sqlite3.Connection, progress, limit: int | None,
                  prefix: str | None) -> dict:
    """Embed the pixels of every indexed image node. Failures are reported, never fatal:
    a missing sidecar must not stop the text indexes from being built."""
    sql = ("SELECT n.id, f.rel_path, f.hash FROM nodes n JOIN files f ON f.id = n.file_id "
           "WHERE n.kind = 'file' AND f.language = 'image'")
    params: list = []
    if prefix:
        sql += " AND (f.rel_path = ? OR f.rel_path LIKE ?)"
        params += [prefix, prefix + "/%"]
    rows = conn.execute(sql + " ORDER BY f.rel_path", params).fetchall()
    if not rows:
        return {"images": 0, "images_embedded": 0, "images_cached": 0}

    part_rows: list[tuple] = []
    pending: dict[int, str] = {}
    for row in rows:
        # The indexer already hashed the file bytes and stored them in files.hash.
        # Reuse that key: no second pass over tens of gigabytes of photos.
        h = hex_to_i64(row["hash"])
        part_rows.append((row["id"], config.IMAGE_PART, h))
        pending.setdefault(h, row["rel_path"])
    conn.executemany("INSERT OR REPLACE INTO node_parts(node_id, part, hash) VALUES (?,?,?)", part_rows)
    conn.commit()

    cached = {r[0] for r in conn.execute(
        "SELECT hash FROM vec_cache WHERE part = ? AND model = ?",
        (config.IMAGE_PART, config.IMAGE_EMBED_MODEL))}
    todo = [(h, rel) for h, rel in pending.items() if h not in cached]
    already = len(pending) - len(todo)
    if limit is not None:
        todo = todo[:limit]

    # Google Drive streaming placeholders are files of the right size filled with zeros.
    # They are not images; they cannot be embedded and they are not the user's fault.
    placeholders: list[str] = []
    real: list[tuple[int, str]] = []
    for h, rel in todo:
        if _is_placeholder(config.REPO_ROOT / rel):
            placeholders.append(rel)
        else:
            real.append((h, rel))

    progress(f"images: {len(pending)} distinct, {already} cached, {len(real)} to embed"
             + (f", {len(placeholders)} skipped (not downloaded from Drive)" if placeholders else ""))
    if not real:
        return {"images": len(pending), "images_embedded": 0, "images_cached": already,
                "images_skipped": len(placeholders)}
    try:
        preflight_images()
    except RuntimeError as exc:
        progress(f"image embedding skipped: {exc}")
        return {"images": len(pending), "images_embedded": 0, "images_cached": already,
                "images_skipped": len(placeholders)}

    start = time.time()
    done = 0
    failures: list[dict] = []
    chunks = [real[i:i + config.IMAGE_BATCH] for i in range(0, len(real), config.IMAGE_BATCH)]
    for n, chunk in enumerate(chunks, 1):
        paths = [config.REPO_ROOT / rel for _, rel in chunk]
        vecs, errors = _embed_chunk(paths)
        failures.extend(errors)
        store = [(config.IMAGE_PART, chunk[i][0], config.IMAGE_EMBED_MODEL, int(v.shape[0]), v.tobytes())
                 for i, v in enumerate(vecs) if v is not None]
        if store:
            conn.executemany(
                "INSERT OR REPLACE INTO vec_cache(part, hash, model, dim, vec) VALUES (?,?,?,?,?)", store)
            conn.commit()
        done += len(store)
        elapsed = max(time.time() - start, 1e-6)
        rate = done / elapsed
        eta = (len(real) - done) / max(rate, 1e-6)
        progress(f"  images {done}/{len(real)} embedded ({rate:.1f}/s, eta {eta/60:.1f}m)")
        if len(failures) >= 25:
            progress(f"  {len(failures)} files failed so far; continuing")
            failures = failures[-25:]
    if failures:
        progress(f"  {len(failures)} image(s) could not be embedded:")
        for f in failures[:10]:
            progress(f"    {f['path']}: {f['error'][:90]}")
    if placeholders:
        progress(f"  {len(placeholders)} file(s) skipped: they are Drive placeholders, "
                 f"not downloaded locally. Make them available offline in Drive to index them.")
    return {"images": len(pending), "images_embedded": done, "images_cached": already,
            "images_skipped": len(placeholders), "images_failed": len(failures)}


class VectorIndex:
    """In-memory matrices per part, loaded from the database.

    Each part is loaded only for the model that built it. Parts from different models are
    separate ranked lists, never a shared matrix: their dimensions and their spaces differ.
    """

    def __init__(self, conn: sqlite3.Connection, parts: tuple[str, ...] | None = None):
        self.parts: dict[str, dict] = {}
        target_parts = parts if parts is not None else config.ALL_PARTS
        for part in target_parts:
            model = config.PART_MODELS.get(part, config.EMBED_MODEL)
            rows = conn.execute(
                "SELECT p.node_id, c.vec, c.dim FROM node_parts p "
                "JOIN vec_cache c ON c.part = p.part AND c.hash = p.hash AND c.model = ? "
                "WHERE p.part = ?", (model, part)).fetchall()
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
