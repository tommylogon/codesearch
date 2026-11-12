"""Separate, isolated read-only public API server for codesearch.

Supports multiple named indexes (e.g. 'viwo' and 'hogwarts') with optional password protection.
Binds to 0.0.0.0 (port 8085) for external / remote AI agents to query.

Authentication (if enabled):
    Query parameter:  ?key=<secret> or ?password=<secret>
    HTTP header:      Authorization: Bearer <secret>
    HTTP header:      X-API-Key: <secret>

Endpoints:
    GET /                    API info and status
    GET /api/indexes         List registered and available indexes
    GET /api/stats           Database statistics (?index=...)
    GET /api/search          Hybrid keyword + vector search (?q=...&index=...)
    GET /api/code            Sandboxed code / prose snippet reader (?path=...&index=...)
    GET /api/node/<id>       AST / section symbol details (?index=...)
    GET /api/graph/<id>      Neighborhood graph (?index=...)
"""
from __future__ import annotations

import argparse
import os
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

# Registry of accessible workspaces
INDEX_REGISTRY: dict[str, dict] = {
    "viwo": {
        "name": "Virtual World Engine",
        "path": Path(r"F:\AI\viwo\virtual-world"),
        "description": "Virtual World Python and TypeScript simulation engine",
    },
    "hogwarts": {
        "name": "Hallowed Are the Path of Chaos",
        "path": Path(r"C:\Users\Tommy\Documents\Books\hallowed are the path of chaos"),
        "description": "Harry Potter 2006 RP World Bible, character dossiers, scenes, and wiki",
    },
}

DEFAULT_INDEX = "viwo"
API_KEY: str | None = None
_searchers: dict[str, Searcher] = {}


def _resolve_index_key(name: str | None) -> str:
    key = (name or DEFAULT_INDEX).strip().lower()
    if key not in INDEX_REGISTRY:
        abort(404, f"Unknown index '{key}'. Available indexes: {list(INDEX_REGISTRY.keys())}")
    return key


def _get_index_paths(index_key: str) -> tuple[Path, Path]:
    info = INDEX_REGISTRY[index_key]
    root = info["path"].resolve()
    db_path = root / ".codesearch" / "codesearch.db"
    if not db_path.exists():
        abort(500, f"Database not found for index '{index_key}' at {db_path}. Please index this workspace first.")
    return root, db_path


def _conn_for_index(index_key: str) -> sqlite3.Connection:
    _, db_path = _get_index_paths(index_key)
    return db.connect(db_path)


def _searcher_for_index(index_key: str, conn: sqlite3.Connection) -> Searcher:
    root, _ = _get_index_paths(index_key)
    # Configure codesearch globals for this search
    config.set_root(root)
    s = _searchers.get(index_key)
    if s is None:
        s = Searcher(conn)
        _searchers[index_key] = s
    s.conn = conn
    return s


def _safe_path_for_index(index_key: str, rel: str) -> Path:
    """Resolve a path strictly inside the target index root."""
    root, _ = _get_index_paths(index_key)
    target = (root / rel).resolve()
    if root != target and root not in target.parents:
        abort(403, f"Access denied: path is outside the allowed '{index_key}' repository")
    if not target.is_file():
        abort(404, f"File not found in index '{index_key}': {rel}")
    return target


@app.before_request
def check_auth():
    """Verify password / API key before processing any request."""
    if request.method == "OPTIONS":
        return
    if not API_KEY:
        return

    # 1. Check Authorization: Bearer <key>
    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token == API_KEY:
            return

    # 2. Check X-API-Key header
    if request.headers.get("X-API-Key", "").strip() == API_KEY:
        return

    # 3. Check query parameters: ?key=... or ?password=... or ?token=...
    param_key = (
        request.args.get("key")
        or request.args.get("password")
        or request.args.get("token")
        or ""
    ).strip()
    if param_key == API_KEY:
        return

    abort(
        401,
        "Unauthorized: invalid or missing password/key. "
        "Pass via query parameter (?key=...) or header (Authorization: Bearer <key>).",
    )


@app.after_request
def add_cors_headers(response):
    """Enable CORS so web browsers and remote agents can call this API."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
    return response


@app.route("/<path:_any>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def handle_options(_any: str = ""):
    """Handle CORS preflight requests."""
    return make_response()


@app.get("/")
def index():
    return jsonify({
        "name": "codesearch-public-api",
        "status": "online",
        "auth_required": API_KEY is not None,
        "default_index": DEFAULT_INDEX,
        "available_indexes": list(INDEX_REGISTRY.keys()),
        "endpoints": [
            "/api/indexes",
            "/api/stats?index=...",
            "/api/search?q=...&index=...&limit=20",
            "/api/code?path=...&index=...&start=1&end=50",
            "/api/node/<id>?index=...",
            "/api/graph/<id>?index=...&depth=1",
        ],
    })


@app.get("/api/indexes")
def api_indexes():
    """List all available indexes on this server."""
    out = []
    for k, info in INDEX_REGISTRY.items():
        exists = (info["path"] / ".codesearch" / "codesearch.db").exists()
        out.append({
            "id": k,
            "name": info["name"],
            "description": info["description"],
            "default": k == DEFAULT_INDEX,
            "ready": exists,
        })
    return jsonify({"indexes": out, "count": len(out)})


@app.get("/api/stats")
def api_stats():
    index_key = _resolve_index_key(request.args.get("index"))
    root, _ = _get_index_paths(index_key)
    config.set_root(root)
    conn = _conn_for_index(index_key)
    try:
        data = db.stats(conn)
        data["index_id"] = index_key
        data["index_name"] = INDEX_REGISTRY[index_key]["name"]
        return jsonify(data)
    finally:
        conn.close()


@app.get("/api/search")
def api_search():
    index_key = _resolve_index_key(request.args.get("index"))
    conn = _conn_for_index(index_key)
    try:
        s = _searcher_for_index(index_key, conn)
        a = request.args
        query_text = a.get("q", "").strip()
        if not query_text:
            return jsonify({"results": [], "count": 0, "query": "", "index": index_key})

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
            "index": index_key,
            "vector_note": s.last_vector_error,
        })
    finally:
        conn.close()


@app.get("/api/node/<int:node_id>")
def api_node(node_id: int):
    index_key = _resolve_index_key(request.args.get("index"))
    conn = _conn_for_index(index_key)
    try:
        node = graph.get_node(conn, node_id)
        if node is None:
            abort(404, f"Node {node_id} not found in index '{index_key}'")
        node["relations"] = graph.summary(conn, node_id, limit=50)
        node["index"] = index_key
        return jsonify(node)
    finally:
        conn.close()


@app.get("/api/graph/<int:node_id>")
def api_graph(node_id: int):
    index_key = _resolve_index_key(request.args.get("index"))
    conn = _conn_for_index(index_key)
    try:
        kinds = tuple(k for k in request.args.get("kinds", "").split(",") if k) or None
        depth = min(int(request.args.get("depth", 1)), 3)
        max_nodes = min(int(request.args.get("max", 100)), 300)
        data = graph.neighborhood(conn, node_id, depth=depth, kinds=kinds, max_nodes=max_nodes)
        data["index"] = index_key
        return jsonify(data)
    finally:
        conn.close()


@app.get("/api/code")
def api_code():
    index_key = _resolve_index_key(request.args.get("index"))
    path = request.args.get("path", "")
    if not path:
        abort(400, "Missing required parameter 'path'")
    start = max(1, int(request.args.get("start", 1)))
    end = int(request.args.get("end", start))
    ctx = min(max(0, int(request.args.get("context", 3))), 20)

    target = _safe_path_for_index(index_key, path)
    if target.suffix.lower() in config.IMAGE_LANGUAGES:
        return jsonify({"path": path, "index": index_key, "image": True, "error": "Binary image file"})

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    lo = max(1, start - ctx)
    hi = min(len(lines), end + ctx)
    return jsonify({
        "path": path,
        "index": index_key,
        "start": lo,
        "end": hi,
        "hit_start": start,
        "hit_end": end,
        "total_lines": len(lines),
        "lines": lines[lo - 1:hi],
    })


def main():
    global API_KEY
    parser = argparse.ArgumentParser(description="codesearch multi-index public read-only API server")
    parser.add_argument("--port", type=int, default=8085, help="Port to listen on (default: 8085)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("--key", default=os.environ.get("CODESEARCH_API_KEY"), help="Optional secret password / API key")
    parser.add_argument("--add-index", action="append", help="Add custom index in form key=path (repeatable)")

    args = parser.parse_args()

    if args.key:
        API_KEY = args.key.strip()

    if args.add_index:
        for item in args.add_index:
            if "=" in item:
                k, p = item.split("=", 1)
                INDEX_REGISTRY[k.strip().lower()] = {
                    "name": k.strip(),
                    "path": Path(p.strip()).resolve(),
                    "description": f"Custom index for {p.strip()}",
                }

    print("============================================================")
    print(" codesearch Multi-Index Public API")
    print(f" Listening on:       http://{args.host}:{args.port}")
    print(f" Password / Auth:    {'ENABLED' if API_KEY else 'DISABLED (Open)'}")
    print(f" Registered Indexes: {list(INDEX_REGISTRY.keys())}")
    for k, v in INDEX_REGISTRY.items():
        exists = (v['path'] / '.codesearch' / 'codesearch.db').exists()
        status = 'READY' if exists else 'NOT INDEXED'
        print(f"   [{k:8}] -> {v['path']} ({status})")
    print("============================================================")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
