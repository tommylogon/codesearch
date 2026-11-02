"""codesearch web UI: search, interactive graph, code inspector, open in VS Code.

Local only (127.0.0.1). Endpoints:
    GET  /api/search?q=&part=&kind=&lang=&path=&any=&strict=&limit=
    GET  /api/node/<id>            full node + relations
    GET  /api/graph/<id>?depth=&kinds=
    GET  /api/code?path=&start=&end=&context=
    POST /api/open  {path, line}   runs `code --goto`
    GET  /api/stats
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "codesearch"

from flask import Flask, abort, jsonify, request, send_file     # noqa: E402

from . import config, db, graph, renderer                        # noqa: E402
from .search import Searcher                                     # noqa: E402

app = Flask(__name__, static_folder=None)
TEMPLATE = Path(__file__).with_name("templates") / "index.html"
VIS = config.PACKAGE_DIR.parent.parent / "static" / "js" / "vendor" / "vis-network" / "vis-network-9.1.9.min.js"

_state: dict = {}


def _conn() -> sqlite3.Connection:
    conn = db.connect()
    return conn


def _searcher_for(conn) -> Searcher:
    """One shared Searcher keeps the vector matrices in memory across requests;
    each request points it at its own short-lived connection."""
    s = _state.get("searcher")
    if s is None:
        s = Searcher(conn)
        _state["searcher"] = s
    s.conn = conn
    return s


def _safe_path(rel: str) -> Path:
    """Resolve a repo-relative path, refusing anything outside the repo."""
    target = (config.REPO_ROOT / rel).resolve()
    if config.REPO_ROOT != target and config.REPO_ROOT not in target.parents:
        abort(400, "path outside repo")
    if not target.is_file():
        abort(404, "no such file")
    return target


@app.get("/")
def index():
    return send_file(TEMPLATE)


@app.get("/vendor/vis-network.js")
def vis():
    if not VIS.exists():
        abort(404, "vis-network not found in static/js/vendor")
    return send_file(VIS, mimetype="text/javascript")


@app.get("/api/stats")
def api_stats():
    conn = _conn()
    try:
        return jsonify(db.stats(conn))
    finally:
        conn.close()


@app.get("/api/search")
def api_search():
    conn = _conn()
    try:
        s = _searcher_for(conn)
        a = request.args
        hits = s.search(
            a.get("q", ""), limit=min(int(a.get("limit", 20)), 100),
            kinds=a.getlist("kind") or None, languages=a.getlist("lang") or None,
            path_prefix=a.get("path") or None, part=a.get("part") or None,
            match_all=a.get("any") not in ("1", "true"), strict=a.get("strict") in ("1", "true"),
            semantic=a.get("semantic", "1") != "0")
        return jsonify({"results": hits, "note": s.last_vector_error})
    finally:
        conn.close()


@app.get("/api/node/<int:node_id>")
def api_node(node_id: int):
    conn = _conn()
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
    conn = _conn()
    try:
        scope = request.args.get("scope", "runtime")
        return jsonify(graph.overview(conn, scope=scope))
    finally:
        conn.close()


@app.get("/api/graph/<int:node_id>")
def api_graph(node_id: int):
    conn = _conn()
    try:
        kinds = tuple(k for k in request.args.get("kinds", "").split(",") if k) or None
        data = graph.neighborhood(conn, node_id, depth=min(int(request.args.get("depth", 1)), 3),
                                  kinds=kinds, max_nodes=int(request.args.get("max", 150)))
        return jsonify(data)
    finally:
        conn.close()


@app.get("/api/render/<int:node_id>")
def api_render(node_id: int):
    conn = _conn()
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
    path = request.args.get("path", "")
    start = max(1, int(request.args.get("start", 1)))
    end = int(request.args.get("end", start))
    ctx = int(request.args.get("context", 3))
    target = _safe_path(path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    lo, hi = max(1, start - ctx), min(len(lines), end + ctx)
    return jsonify({"path": path, "start": lo, "end": hi, "hit_start": start, "hit_end": end,
                    "total": len(lines), "lines": lines[lo - 1:hi],
                    "absolute": str(target)})


@app.post("/api/open")
def api_open():
    body = request.get_json(force=True, silent=True) or {}
    target = _safe_path(str(body.get("path", "")))
    line = max(1, int(body.get("line", 1)))
    exe = shutil.which("code") or shutil.which("code.cmd")
    if exe is None:
        return jsonify({"ok": False, "error": "`code` is not on PATH; use the vscode:// link instead"}), 501
    # argument list, no shell: the path cannot inject anything
    subprocess.Popen([exe, "--goto", f"{target}:{line}"], shell=False)
    return jsonify({"ok": True, "opened": f"{target}:{line}"})


def run(port: int = config.WEB_PORT) -> None:
    print(f"codesearch UI on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run()
