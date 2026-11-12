"""Separate, isolated read-only public API server for codesearch.

Exposes only safe query endpoints to 0.0.0.0 (e.g. port 8085) for external / remote querying.
Strictly sandboxed to the target repository (default: F:\\AI\\viwo\\virtual-world).
Does NOT touch or affect the local codesearch web UI or MCP setup.

Endpoints:
    GET /                    API info and health
    GET /api/stats           Database statistics (node count, file count)
    GET /api/search          Hybrid keyword + semantic search (?q=...&limit=20)
    GET /api/code            Sandboxed code snippet viewer (?path=...&start=1&end=50)
    GET /api/node/<id>       AST symbol details and relations
    GET /api/graph/<id>      Neighborhood graph
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Ensure codesearch package on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flask import Flask, abort, jsonify, make_response, request

from codesearch import config, db, graph
from codesearch.search import Searcher

app = Flask("codesearch_public", static_folder=None)

_searcher_cache: dict = {}


def _conn() -> sqlite3.Connection:
    return db.connect()


def _get_searcher(conn: sqlite3.Connection) -> Searcher:
    s = _searcher_cache.get("searcher")
    if s is None:
        s = Searcher(conn)
        _searcher_cache["searcher"] = s
    s.conn = conn
    return s


def _safe_path(rel: str) -> Path:
    """Resolve a repo-relative path, refusing anything outside the repo."""
    target = (config.REPO_ROOT / rel).resolve()
    if config.REPO_ROOT != target and config.REPO_ROOT not in target.parents:
        abort(403, "Access denied: path is outside the allowed repository")
    if not target.is_file():
        abort(404, "File not found")
    return target


@app.after_request
def add_cors_headers(response):
    """Enable CORS so any web page, remote client, or browser agent can call this API."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.route("/<path:_any>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def handle_options(_any: str = ""):
    """Handle CORS preflight requests."""
    res = make_response()
    return res


@app.get("/")
def index():
    return jsonify({
        "name": "codesearch-public-api",
        "status": "online",
        "repository": str(config.REPO_ROOT.name),
        "endpoints": [
            "/api/stats",
            "/api/search?q=...",
            "/api/code?path=...&start=1&end=50",
            "/api/node/<id>",
            "/api/graph/<id>?depth=1",
        ],
    })


@app.get("/api/stats")
def api_stats():
    conn = _conn()
    try:
        data = db.stats(conn)
        data["repository_root"] = str(config.REPO_ROOT.name)
        return jsonify(data)
    finally:
        conn.close()


@app.get("/api/search")
def api_search():
    conn = _conn()
    try:
        s = _get_searcher(conn)
        a = request.args
        query_text = a.get("q", "").strip()
        if not query_text:
            return jsonify({"results": [], "count": 0, "query": ""})

        limit = min(int(a.get("limit", 20)), 100)
        kinds = a.getlist("kind") or None
        langs = a.getlist("lang") or None
        prefix = a.get("path") or None
        part = a.get("part") or None
        match_all = a.get("any") not in ("1", "true")
        strict = a.get("strict") in ("1", "true")
        semantic = a.get("semantic", "1") != "0"

        hits = s.search(
            query_text,
            limit=limit,
            kinds=kinds,
            languages=langs,
            path_prefix=prefix,
            part=part,
            match_all=match_all,
            strict=strict,
            semantic=semantic,
        )
        return jsonify({
            "results": hits,
            "count": len(hits),
            "query": query_text,
            "vector_note": s.last_vector_error,
        })
    finally:
        conn.close()


@app.get("/api/node/<int:node_id>")
def api_node(node_id: int):
    conn = _conn()
    try:
        node = graph.get_node(conn, node_id)
        if node is None:
            abort(404, "Node not found")
        node["relations"] = graph.summary(conn, node_id, limit=50)
        return jsonify(node)
    finally:
        conn.close()


@app.get("/api/graph/<int:node_id>")
def api_graph(node_id: int):
    conn = _conn()
    try:
        kinds = tuple(k for k in request.args.get("kinds", "").split(",") if k) or None
        depth = min(int(request.args.get("depth", 1)), 3)
        max_nodes = min(int(request.args.get("max", 100)), 300)
        data = graph.neighborhood(conn, node_id, depth=depth, kinds=kinds, max_nodes=max_nodes)
        return jsonify(data)
    finally:
        conn.close()


@app.get("/api/code")
def api_code():
    path = request.args.get("path", "")
    if not path:
        abort(400, "Missing required parameter 'path'")
    start = max(1, int(request.args.get("start", 1)))
    end = int(request.args.get("end", start))
    ctx = min(max(0, int(request.args.get("context", 3))), 20)

    target = _safe_path(path)
    if target.suffix.lower() in config.IMAGE_LANGUAGES:
        return jsonify({"path": path, "image": True, "error": "Binary image file"})

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    lo = max(1, start - ctx)
    hi = min(len(lines), end + ctx)
    return jsonify({
        "path": path,
        "start": lo,
        "end": hi,
        "hit_start": start,
        "hit_end": end,
        "total_lines": len(lines),
        "lines": lines[lo - 1:hi],
    })


def main():
    parser = argparse.ArgumentParser(description="codesearch public read-only API server")
    parser.add_argument("--root", default=r"F:\AI\viwo\virtual-world", help="Target repository directory to serve")
    parser.add_argument("--port", type=int, default=8085, help="Port to listen on (default: 8085)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to (default: 0.0.0.0)")

    args = parser.parse_args()

    target_path = Path(args.root).resolve()
    if not target_path.exists():
        print(f"Error: Target path does not exist: {target_path}", file=sys.stderr)
        sys.exit(1)

    db_path = target_path / ".codesearch" / "codesearch.db"
    if not db_path.exists():
        print(f"Error: codesearch.db not found at {db_path}", file=sys.stderr)
        print("Please run `codesearch index` inside the target directory first.", file=sys.stderr)
        sys.exit(1)

    config.set_root(target_path)
    print("============================================================")
    print(" codesearch Public Read-Only API")
    print(f" Serving Repository: {target_path}")
    print(f" Database:           {db_path}")
    print(f" Listening on:       http://{args.host}:{args.port}")
    print(" Endpoints:          /api/stats, /api/search, /api/code, /api/node, /api/graph")
    print(" Security:           Read-only, strictly locked to target repository")
    print("============================================================")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
