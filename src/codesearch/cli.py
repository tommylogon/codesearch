"""codesearch command line.

    python tools/codesearch/cli.py index [--force]
    python tools/codesearch/cli.py embed [--limit N]
    python tools/codesearch/cli.py "turn queue initiative"          (search is the default)
    python tools/codesearch/cli.py search "..." --part comment --kind function --any
    python tools/codesearch/cli.py callers syncGraphData
    python tools/codesearch/cli.py callees syncGraphData
    python tools/codesearch/cli.py refs area_presence
    python tools/codesearch/cli.py outline static/js/agent/turn-queue.js
    python tools/codesearch/cli.py stats | serve | mcp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "codesearch"

from codesearch import config, db, graph, indexer, renderer  # noqa: E402
from codesearch.search import Searcher                      # noqa: E402

COMMANDS = {"index", "embed", "search", "callers", "callees", "refs", "outline", "render", "stats", "serve", "mcp"}
KIND_ALIASES = {"func": "function", "fn": "function", "var": "variable", "doc": "section",
                "file": "file", "method": "method", "class": "class", "comment": "comment",
                "section": "section", "block": "block", "function": "function", "variable": "variable",
                "element": "element"}


def _print_hit(i: int, h: dict, full: bool, preview_lines: int) -> None:
    where = f"{h['file']}:{h['line_start']}-{h['line_end']}" if h["line_end"] != h["line_start"] \
        else f"{h['file']}:{h['line_start']}"
    name = h["qualified_name"].split("::", 1)[-1] if "::" in h["qualified_name"] else h["name"]
    print(f"[{i}] {h['kind'].upper():8} {name or h['file']}   score {h['score']:.4f}  ({', '.join(h['matched'])})")
    print(f"     {where}")
    if h["signature"] and h["kind"] not in ("file", "comment"):
        print(f"     {h['signature']}")
    doc = (h.get("docstring") or "").strip()
    if doc:
        lines = doc.splitlines()
        shown = lines if full else lines[:preview_lines]
        for ln in shown:
            print(f"       | {ln}")
        if len(shown) < len(lines):
            print(f"       | ... {len(lines) - len(shown)} more lines (--full shows all)")
    for key, rel in h["relations"].items():
        if key in ("annotated_by",) and h["kind"] != "comment":
            pass
        extra = f" (+{rel['count'] - len(rel['items'])} more)" if rel["count"] > len(rel["items"]) else ""
        print(f"     {key.replace('_', ' ')}: " + "; ".join(rel["items"]) + extra)
    print()


def cmd_search(args, conn) -> int:
    searcher = Searcher(conn)
    kinds = [KIND_ALIASES.get(k, k) for k in args.kind] if args.kind else None
    t0 = time.time()
    hits = searcher.search(" ".join(args.query), limit=args.limit, kinds=kinds,
                           languages=args.lang, path_prefix=args.path, part=args.part,
                           match_all=not args.any, strict=args.strict,
                           semantic=not args.no_semantic, keyword=not args.no_keyword)
    if args.json:
        print(json.dumps({"results": hits, "vector_note": searcher.last_vector_error}, indent=2))
        return 0
    took = time.time() - t0
    print(f"{len(hits)} results in {took:.2f}s")
    if searcher.last_vector_error and not args.no_semantic:
        print(f"note: keyword only - {searcher.last_vector_error}")
    print()
    for i, h in enumerate(hits, 1):
        _print_hit(i, h, args.full, args.preview)
    return 0


def _resolve(conn, symbol: str, kinds=None) -> dict | None:
    found = graph.find_symbol(conn, symbol, kinds)
    if not found:
        print(f"no symbol matches {symbol!r}")
        return None
    if len(found) > 1:
        print(f"{len(found)} matches for {symbol!r}, using the first:")
        for f in found[:8]:
            print(f"  {f['kind']:8} {f['qualified_name']}:{f['line_start']}")
        print()
    return found[0]


def _print_nodes(title: str, rows: list[dict]) -> None:
    print(f"{title} ({len(rows)}):")
    for r in rows:
        mode = f" [{r['mode']}]" if "mode" in r else ""
        print(f"  {r['file']}:{r['line_start']:<5} {r['kind']:8} {r['qualified_name'].split('::', 1)[-1]}{mode}")


def cmd_graph(args, conn) -> int:
    node = _resolve(conn, args.symbol)
    if node is None:
        return 1
    print(f"{node['kind'].upper()} {node['qualified_name']}  {node['file']}:{node['line_start']}")
    if args.cmd == "callers":
        _print_nodes("called by", graph.callers(conn, node["id"]))
    elif args.cmd == "callees":
        _print_nodes("calls", graph.callees(conn, node["id"]))
    else:
        _print_nodes("referenced by", graph.references(conn, node["id"]))
        if node["kind"] != "variable":
            print("(note: refs is meant for variables; use callers for functions)")
    return 0


def cmd_outline(args, conn) -> int:
    out = graph.outline(conn, args.path.replace("\\", "/"))
    if out is None:
        print(f"{args.path} is not indexed")
        return 1
    print(f"{out['file']} ({out['language']}, {out['lines']} lines)")
    for it in out["items"]:
        if it["kind"] == "comment" and not args.comments:
            continue
        indent = "  " * it["qualified_name"].count("::")
        label = it["signature"] or it["name"] or it["qualified_name"]
        print(f"{it['lines'][0]:>5}  {indent}{it['kind']:8} {label}")
    return 0


def _resolve_id_or_symbol(conn, token: str) -> int | None:
    s = str(token).strip().lstrip("#")
    if s.isdigit():
        nid = int(s)
        if graph.get_node(conn, nid):
            return nid
    found = graph.find_symbol(conn, s, kinds=("file", "class", "function", "method", "variable", "section"))
    return found[0]["id"] if found else None


def cmd_render(args, conn) -> int:
    root_id = _resolve_id_or_symbol(conn, args.symbol)
    if root_id is None:
        print(f"no symbol or node id matches {args.symbol!r}")
        return 1
    expand_ids: list[int] = []
    for part in (args.expand or "").split(","):
        part = part.strip()
        if not part:
            continue
        eid = _resolve_id_or_symbol(conn, part)
        if eid is not None:
            expand_ids.append(eid)
    kinds = tuple(k.strip() for k in (args.kinds or "").split(",") if k.strip()) or None
    out_path = Path(args.out) if args.out else None
    res = renderer.render_subgraph(conn, root_id, depth=max(1, min(args.depth, 2)),
                                   expand_ids=expand_ids, kinds=kinds, out_path=out_path)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"rendered: {res['image_path']}")
        print(f"web url:  {res['url']}")
        print(f"subgraph: {res['node_count']} nodes, {res['edge_count']} edges (root #{res['root']})")
        for n in res["nodes"]:
            print(f"  {n['mark']:<8} {n['kind']:8} {n['qualified_name']} ({n['file']}:{n['lines'][0]})")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    # Allow leading --root / --db before an implicit search query
    idx = 0
    while idx < len(argv):
        if argv[idx] in ("-r", "--root", "--db") and idx + 1 < len(argv):
            idx += 2
        elif argv[idx].startswith(("--root=", "--db=")):
            idx += 1
        else:
            break
    if idx < len(argv) and argv[idx] not in COMMANDS and not argv[idx].startswith("-"):
        argv.insert(idx, "search")
    p = argparse.ArgumentParser(prog="codesearch", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-r", "--root", help="target directory to index/search (default: CODESEARCH_ROOT or repo root)")
    p.add_argument("--db", help="database path (default <root>/.codesearch/codesearch.db)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="hybrid keyword + semantic search")
    s.add_argument("query", nargs="+")
    s.add_argument("-n", "--limit", type=int, default=10)
    s.add_argument("--part", choices=config.PARTS, help="search one vector index only")
    s.add_argument("--kind", action="append", help="file|class|function|method|variable|comment|section|block (repeatable)")
    s.add_argument("--lang", action="append", help="python|typescript|javascript|markdown|html|text (repeatable)")
    s.add_argument("--path", help="repo-relative path prefix")
    s.add_argument("--any", action="store_true", help="keyword list matches ANY word instead of all")
    s.add_argument("--strict", action="store_true", help="only keyword matches; vectors just reorder")
    s.add_argument("--no-semantic", action="store_true")
    s.add_argument("--no-keyword", action="store_true")
    s.add_argument("--full", action="store_true", help="show whole docstrings")
    s.add_argument("--preview", type=int, default=4, help="docstring lines shown per result")
    s.add_argument("--json", action="store_true")

    i = sub.add_parser("index", help="scan the repo and update the graph")
    i.add_argument("--force", action="store_true", help="re-parse every file")
    e = sub.add_parser("embed", help="embed new or changed text into the four vector indexes")
    e.add_argument("--limit", type=int, help="embed at most N new texts")
    e.add_argument("--path", help="repo-relative path prefix to embed first")
    for name in ("callers", "callees", "refs"):
        c = sub.add_parser(name)
        c.add_argument("symbol")
    o = sub.add_parser("outline", help="skeleton of one file")
    o.add_argument("path")
    o.add_argument("--comments", action="store_true")
    rn = sub.add_parser("render", help="render a Set-of-Mark subgraph PNG for vision agents")
    rn.add_argument("symbol", help="symbol name or #node_id")
    rn.add_argument("--depth", type=int, default=1)
    rn.add_argument("--expand", default="", help="comma-separated #ids or symbols to expand in place")
    rn.add_argument("--kinds", default="", help="comma-separated edge kinds filter")
    rn.add_argument("--out", help="output PNG path")
    rn.add_argument("--json", action="store_true")
    sub.add_parser("stats")
    sv = sub.add_parser("serve", help="web UI")
    sv.add_argument("--port", type=int, default=config.WEB_PORT)
    sub.add_parser("mcp", help="MCP server over stdio")

    args = p.parse_args(argv)
    if args.root:
        config.set_root(args.root)
    if args.cmd == "mcp":
        from codesearch import mcp_server
        mcp_server.run()
        return 0
    if args.cmd == "serve":
        from codesearch import app
        app.run(port=args.port)
        return 0

    conn = db.connect(args.db)
    try:
        if args.cmd == "index":
            res = indexer.index(conn, force=args.force, progress=print)
            print(json.dumps({k: v for k, v in res.items() if k != "problems"}))
            for prob in res["problems"]:
                print("parse problem:", prob)
            return 0
        if args.cmd == "embed":
            from codesearch import embedder
            print(json.dumps(embedder.build(conn, limit=args.limit, path_prefix=args.path)), flush=True)
            return 0
        if args.cmd == "stats":
            print(json.dumps(db.stats(conn), indent=2))
            return 0
        if db.stats(conn)["nodes"] == 0:
            print("index is empty - run: python tools/codesearch/cli.py index")
            return 1
        if args.cmd == "search":
            return cmd_search(args, conn)
        if args.cmd in ("callers", "callees", "refs"):
            return cmd_graph(args, conn)
        if args.cmd == "outline":
            return cmd_outline(args, conn)
        if args.cmd == "render":
            return cmd_render(args, conn)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
