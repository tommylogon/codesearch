"""codesearch web UI: regex-tester style search workbench, interactive graph, anatomy inspector.

Local only (127.0.0.1). Endpoints (all accept optional `?index=<id_or_path>`, defaulting to REPO_ROOT):
    GET  /api/indexes              registered and discovered indexes + active default
    GET  /api/files?path=&limit=   indexed files with node/edge counts
    GET  /api/inspect?path=        all parsed nodes, FTS words, embedded parts, edges, refs
    GET  /api/search?q=&part=&kind=&lang=&path=&any=&strict=&limit=
    GET  /api/node/<id>            full node + relations
    GET  /api/file_graph?path=     document breakdown & relation graph for one file
    GET  /api/overview?scope=      codebase / docs overview graph
    GET  /api/graph/<id>?depth=&kinds=
    GET  /api/code?path=&start=&end=&context=
    POST /api/open  {path, line, index?}   runs `code --goto`
    GET  /api/stats
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "codesearch"

from flask import Flask, abort, g, jsonify, request, send_file  # noqa: E402

from . import config, db, diag, graph, registry, renderer        # noqa: E402
from .parsers import pdf_parser                                  # noqa: E402
from .search import Searcher, query_tokens                       # noqa: E402

app = Flask(__name__, static_folder=None)
TEMPLATE = Path(__file__).with_name("templates") / "index.html"
BUNDLED_VIS = Path(__file__).with_name("static") / "vis-network-9.1.9.min.js"
LEGACY_VIS = config.PACKAGE_DIR.parent.parent / "static" / "js" / "vendor" / "vis-network" / "vis-network-9.1.9.min.js"

_vector_caches: dict[str, dict] = {}
_vector_lock = threading.RLock()
_log = diag.get_logger("web")


@app.before_request
def _req_start() -> None:
    if request.path.startswith("/api/"):
        diag.configure("web")
        g._req_t0 = time.perf_counter()


@app.after_request
def _req_end(resp):
    t0 = getattr(g, "_req_t0", None)
    if t0 is not None:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        level = logging.WARNING if (dt_ms >= config.SLOW_QUERY_MS or resp.status_code >= 500) else logging.DEBUG
        _log.log(
            level,
            "HTTP %s %s status=%d ms=%.1f qs=%s",
            request.method,
            request.path,
            resp.status_code,
            dt_ms,
            request.query_string.decode("utf-8", "replace")[:200],
        )
        diag.flush()
    return resp


def _resolve_repo(index_param: str | None = None) -> tuple[Path, Path]:
    """Resolve optional `index` query parameter to `(repo_root, db_path)`."""
    raw = (index_param if index_param is not None else request.args.get("index", "")).strip()
    if not raw:
        return config.REPO_ROOT, config.DB_PATH
    if raw == "__current__":
        return config.REPO_ROOT, config.DB_PATH
    try:
        entry = registry.resolve_index(raw)
    except registry.IndexResolutionError as exc:
        abort(404, str(exc))
    if not entry.is_ready:
        abort(400, f"Index {entry.id!r} has no database yet (run: codesearch -r {entry.path} index)")
    return entry.path, entry.db_path


def _conn(db_path: Path | None = None) -> sqlite3.Connection:
    return db.connect(db_path)


def _open_context(index_param: str | None = None) -> tuple[Path, sqlite3.Connection]:
    repo_root, db_path = _resolve_repo(index_param)
    return repo_root, _conn(db_path)


def _searcher_for(conn: sqlite3.Connection, repo_root: Path | None = None) -> Searcher:
    """Create a request-local Searcher bound to this thread's connection while
    reusing cached vector matrices safely across concurrent Flask requests."""
    root = repo_root or config.REPO_ROOT
    key = str(root)
    with _vector_lock:
        cache = _vector_caches.setdefault(key, {})
    return Searcher(conn, repo_root=root, vector_cache=cache, vector_lock=_vector_lock)


def _safe_path(rel: str, repo_root: Path | None = None) -> Path:
    """Resolve a repo-relative path, refusing anything outside `repo_root`."""
    root = (repo_root or config.REPO_ROOT).resolve()
    target = (root / rel).resolve()
    if root != target and root not in target.parents:
        abort(400, "path outside repo")
    if not target.is_file():
        abort(404, "no such file")
    return target


@app.get("/")
def index():
    return send_file(TEMPLATE)


@app.get("/vendor/vis-network.js")
def vis():
    target = BUNDLED_VIS if BUNDLED_VIS.exists() else LEGACY_VIS
    if not target.exists():
        abort(404, "vis-network bundle not found in src/codesearch/static")
    return send_file(target, mimetype="text/javascript")


@app.get("/api/indexes")
def api_indexes():
    entries = registry.all_entries()
    active_root = config.REPO_ROOT.resolve()
    summary = registry.index_summary(entries)
    active_id = ""
    for row in summary:
        if Path(row["path"]).resolve() == active_root:
            active_id = row["id"]
            row["active"] = True
        else:
            row["active"] = False
    if not active_id:
        cur_entry = registry.IndexEntry(
            id="__current__",
            path=active_root,
            description="Current server root",
            source="active",
        )
        cur_meta = registry.index_summary({"__current__": cur_entry})[0]
        cur_meta["active"] = True
        summary.insert(0, cur_meta)
        active_id = "__current__"
    return jsonify({"active_id": active_id, "active_path": str(active_root), "indexes": summary})


@app.get("/api/files")
def api_files():
    _root, conn = _open_context()
    try:
        limit = max(1, min(int(request.args.get("limit", 300)), 2000))
        prefix = request.args.get("path", "").strip().replace("\\", "/")
        sql = (
            "SELECT f.id, f.rel_path, f.language, f.lines, "
            "(SELECT COUNT(*) FROM nodes n WHERE n.file_id = f.id) AS node_count "
            "FROM files f "
        )
        params: list = []
        if prefix:
            sql += "WHERE f.rel_path LIKE ? "
            params.append(f"{prefix}%")
        sql += "ORDER BY f.rel_path LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        return jsonify({"total": total, "files": [dict(r) for r in rows]})
    finally:
        conn.close()


@app.get("/api/inspect")
def api_inspect():
    _root, conn = _open_context()
    try:
        path = request.args.get("path", "").strip()
        if not path:
            first = conn.execute("SELECT rel_path FROM files ORDER BY rel_path LIMIT 1").fetchone()
            if first is None:
                return jsonify({"file": None, "nodes": [], "edges": [], "refs": []})
            path = first["rel_path"]
        files = graph.resolve_inspect_files(conn, path)
        if not files:
            abort(404, f"no indexed file matches {path!r}")
        limit = max(1, min(int(request.args.get("limit", 400)), 2000))
        offset = max(0, int(request.args.get("offset", 0)))
        return jsonify(graph.inspect_file_payload(conn, files[0], limit=limit, offset=offset))
    finally:
        conn.close()


@app.get("/api/file_graph")
def api_file_graph():
    _root, conn = _open_context()
    try:
        path = request.args.get("path", "").strip()
        if not path:
            first = conn.execute("SELECT rel_path FROM files ORDER BY rel_path LIMIT 1").fetchone()
            if first is None:
                return jsonify({"nodes": [], "edges": [], "truncated": False, "file": None})
            path = first["rel_path"]
        max_nodes = max(20, min(int(request.args.get("max", 240)), 600))
        return jsonify(graph.file_graph(conn, path, max_nodes=max_nodes))
    finally:
        conn.close()


@app.get("/api/stats")
def api_stats():
    repo_root, conn = _open_context()
    try:
        data = db.stats(conn)
        data["repo_root"] = str(repo_root)
        return jsonify(data)
    finally:
        conn.close()


@app.get("/api/search")
def api_search():
    repo_root, conn = _open_context()
    try:
        s = _searcher_for(conn, repo_root=repo_root)
        a = request.args
        q = a.get("q", "")
        try:
            with diag.watch_slow(f"web_search:{q[:40]}", logger=_log):
                hits = s.search(
                    q, limit=min(int(a.get("limit", 30)), 100),
                    kinds=a.getlist("kind") or None, languages=a.getlist("lang") or None,
                    path_prefix=a.get("path") or None, part=a.get("part") or None,
                    match_all=a.get("any") not in ("1", "true"), strict=a.get("strict") in ("1", "true"),
                    semantic=a.get("semantic", "1") != "0",
                    after=a.get("after") or None, before=a.get("before") or None)
        except ValueError as exc:
            abort(400, f"Invalid date filter: {exc}")
        return jsonify({
            "results": hits,
            "note": s.last_vector_error,
            "tokens": query_tokens(q),
            "total_matched": getattr(s, "last_total_matched", len(hits)),
            "timings": getattr(s, "last_timings", {}),
        })
    finally:
        conn.close()


@app.get("/api/node/<int:node_id>")
def api_node(node_id: int):
    _root, conn = _open_context()
    try:
        node = graph.get_node(conn, node_id)
        if node is None:
            abort(404)
        node["relations"] = graph.summary(conn, node_id, limit=50)
        return jsonify(node)
    finally:
        conn.close()


@app.get("/api/overview")
def api_overview():
    _root, conn = _open_context()
    try:
        scope = request.args.get("scope", "runtime")
        return jsonify(graph.overview(conn, scope=scope))
    finally:
        conn.close()


@app.get("/api/graph/<int:node_id>")
def api_graph(node_id: int):
    _root, conn = _open_context()
    try:
        kinds = tuple(k for k in request.args.get("kinds", "").split(",") if k) or None
        data = graph.neighborhood(conn, node_id, depth=min(int(request.args.get("depth", 1)), 3),
                                  kinds=kinds, max_nodes=int(request.args.get("max", 150)))
        return jsonify(data)
    finally:
        conn.close()


@app.get("/api/render/<int:node_id>")
def api_render(node_id: int):
    _root, conn = _open_context()
    try:
        kinds = tuple(k for k in request.args.get("kinds", "").split(",") if k) or None
        exp = [int(x) for x in request.args.get("expand", "").split(",") if x.strip().isdigit()]
        res = renderer.render_subgraph(conn, node_id, depth=min(int(request.args.get("depth", 1)), 2),
                                       expand_ids=exp, kinds=kinds)
        return send_file(res["image_path"], mimetype="image/png")
    finally:
        conn.close()


@app.get("/api/code")
def api_code():
    repo_root, db_path = _resolve_repo()
    path = request.args.get("path", "")
    start = max(1, int(request.args.get("start", 1)))
    end = int(request.args.get("end", start))
    ctx = int(request.args.get("context", 3))
    target = _safe_path(path, repo_root=repo_root)
    if target.suffix.lower() in config.IMAGE_LANGUAGES:
        return jsonify({"path": path, "image": True, "absolute": str(target)})
    if target.suffix.lower() == ".pdf":
        res = pdf_parser.read_pages(target, start, end, max_pages=8)
        if "error" in res:
            return jsonify({"path": path, "image": False, "pdf": True, "start": 1, "end": 1,
                            "hit_start": 1, "hit_end": 1, "total": 0,
                            "lines": [f"[error: {res['error']}]"], "absolute": str(target)})
        lines = res["lines"]
        return jsonify({"path": path, "image": False, "pdf": True, "unit": "pages",
                        "page_start": res["start"], "page_end": res["end"],
                        "start": 1, "end": len(lines), "hit_start": 1, "hit_end": len(lines),
                        "total": res["total_pages"], "lines": lines, "absolute": str(target)})
    lo = max(1, start - ctx)
    hi = max(lo, end + ctx)
    picked: list[str] = []
    last_ln = 0
    eof_reached = False
    with target.open("r", encoding="utf-8", errors="replace") as fh:
        for ln, raw_line in enumerate(fh, 1):
            last_ln = ln
            if ln > hi:
                break
            if ln >= lo:
                picked.append(raw_line.rstrip("\r\n"))
        else:
            eof_reached = True
    actual_end = (lo - 1 + len(picked)) if picked else 0
    if eof_reached:
        total_lines = last_ln
    else:
        conn = _conn(db_path)
        try:
            row = conn.execute(
                "SELECT lines FROM files WHERE rel_path = ?",
                (path.replace("\\", "/").lstrip("./"),),
            ).fetchone()
        finally:
            conn.close()
        total_lines = int(row["lines"]) if row and row["lines"] else max(actual_end, last_ln)
    return jsonify({"path": path, "image": False, "start": lo, "end": actual_end,
                    "hit_start": start, "hit_end": end, "total": total_lines, "lines": picked,
                    "absolute": str(target)})


@app.get("/api/image")
def api_image():
    repo_root, _db_path = _resolve_repo()
    path = request.args.get("path", "")
    target = _safe_path(path, repo_root=repo_root)
    return send_file(target)


@app.post("/api/open")
def api_open():
    body = request.get_json(force=True, silent=True) or {}
    repo_root, _db_path = _resolve_repo(str(body.get("index", "")).strip() or None)
    target = _safe_path(str(body.get("path", "")), repo_root=repo_root)
    line = max(1, int(body.get("line", 1)))
    exe = shutil.which("code") or shutil.which("code.cmd")
    if exe is None:
        return jsonify({"ok": False, "error": "`code` is not on PATH; use the vscode:// link instead"}), 501
    # argument list, no shell: the path cannot inject anything
    subprocess.Popen([exe, "--goto", f"{target}:{line}"], shell=False)
    return jsonify({"ok": True, "opened": f"{target}:{line}"})


def run(port: int = config.WEB_PORT) -> None:
    log = diag.configure("web")
    conn = _conn()
    try:
        diag.log_startup_banner(log, component="web", repo_root=config.REPO_ROOT, db_path=config.DB_PATH, conn=conn)
    finally:
        conn.close()
    print(f"codesearch UI on http://127.0.0.1:{port} (log: {diag.current_log_path()})")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run()
