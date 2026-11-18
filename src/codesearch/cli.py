"""codesearch command line.

    python tools/codesearch/cli.py index [--force] [--no-embed] [--embed-limit N]
    python tools/codesearch/cli.py embed [--limit N]
    python tools/codesearch/cli.py "turn queue initiative"          (search is the default)
    python tools/codesearch/cli.py search "..." --part comment --kind function --any
    python tools/codesearch/cli.py callers syncGraphData
    python tools/codesearch/cli.py callees syncGraphData
    python tools/codesearch/cli.py refs area_presence
    python tools/codesearch/cli.py outline static/js/agent/turn-queue.js
    python tools/codesearch/cli.py inspect [path] [--json]
    python tools/codesearch/cli.py stats | serve | mcp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "codesearch"

from codesearch import config, db, embedder, graph, indexer, renderer  # noqa: E402
from codesearch.search import Searcher                                 # noqa: E402

COMMANDS = {
    "index", "embed", "pack", "registry", "search", "callers", "callees", "refs",
    "outline", "inspect", "render", "stats", "serve", "mcp", "image-server", "embed-server",
}
KIND_ALIASES = {"func": "function", "fn": "function", "var": "variable", "doc": "section",
                 "file": "file", "method": "method", "class": "class", "comment": "comment",
                 "section": "section", "block": "block", "function": "function", "variable": "variable",
                 "element": "element", "record": "record", "tag": "tag"}


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
    part = args.part or (config.IMAGE_PART if args.image else None)
    try:
        hits = searcher.search(" ".join(args.query), limit=args.limit, kinds=kinds,
                               languages=args.lang, path_prefix=args.path, part=part,
                               match_all=not args.any, strict=args.strict,
                               semantic=not args.no_semantic, keyword=not args.no_keyword,
                               rerank=args.rerank,
                               must_terms=args.must, should_terms=args.should, not_terms=args.not_,
                               after=args.after, before=args.before)
    except ValueError as exc:
        print(f"invalid date filter: {exc}", file=sys.stderr)
        return 2
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
        if node["kind"] in ("function", "method"):
            print("(note: refs is meant for variables and sections; use callers for functions)")
    return 0


def cmd_outline(args, conn) -> int:
    out = graph.outline(conn, args.path.replace("\\", "/"))
    if out is None:
        print(f"{args.path} is not indexed")
        return 1
    unit = "pages" if out["language"] == "pdf" else "lines"
    print(f"{out['file']} ({out['language']}, {out['lines']} {unit})")
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
    found = graph.find_symbol(conn, s, kinds=("file", "class", "function", "method", "variable", "section", "record"))
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


def cmd_inspect(args, conn) -> int:
    """Dump everything indexed for a file (or all files): nodes, parts, FTS, edges, refs."""
    files = graph.resolve_inspect_files(conn, getattr(args, "path", None))
    if not files:
        target = getattr(args, "path", None) or "<index>"
        print(f"no indexed file matches {target!r}")
        return 1
    payload = [graph.inspect_file_payload(conn, f) for f in files]
    if args.json:
        print(json.dumps(payload[0] if len(payload) == 1 and args.path else payload, indent=2))
        return 0
    for item in payload:
        f = item["file"]
        print(f"=== {f['rel_path']} ({f['language']}, {f['lines']} lines, {f['node_count']} nodes, hash={f['hash']}) ===")
        for n in item["nodes"]:
            parent_str = f" parent=#{n['parent_id']}" if n["parent_id"] is not None else ""
            print(f"  [#{n['id']}] {n['kind']:8} L{n['lines'][0]}-L{n['lines'][1]}{parent_str}  {n['qualified_name']}")
            if n["signature"]:
                print(f"      signature: {n['signature']}")
            if n["docstring"]:
                print(f"      docstring: {n['docstring']!r}")
            if n["source"]:
                print(f"      source:    {n['source']!r}")
            if n["fts_words"]:
                print(f"      fts_words: {n['fts_words']}")
            for p in n["parts"]:
                vec_desc = ", ".join(f"{v['model']}({v['dim']}d)" for v in p["vectors"]) or "NOT EMBEDDED"
                print(f"      part[{p['part']}] ({vec_desc}): {p['text']!r}")
        if item["edges"]:
            print("  -- edges --")
            for e in item["edges"]:
                print(f"    #{e['source_id']} ({e['source_qname']}) -[{e['kind']}]-> #{e['target_id']} ({e['target_qname']})")
        if item["refs"]:
            print("  -- raw refs --")
            for r in item["refs"]:
                qual = f" ({r['qualifier']})" if r["qualifier"] else ""
                print(f"    #{r['source_id']} ({r['source_qname']}) {r['kind']} -> {r['target']}{qual}")
    return 0


def cmd_registry(args) -> int:
    """Manage named codesearch indexes in the persistent local registry."""
    from codesearch import registry

    action = getattr(args, "reg_cmd", None) or "list"
    if action == "list":
        entries = registry.all_entries()
        print(json.dumps({"registry_file": str(registry.registry_path()),
                          "indexes": registry.index_summary(entries)}, indent=2))
        return 0
    if action == "add":
        entry = registry.upsert_entry(args.path, name=args.name, description=args.desc or "")
        print(json.dumps({"id": entry.id, "path": str(entry.path), "ready": entry.is_ready}, indent=2))
        return 0
    if action == "remove":
        removed = registry.remove_entry(args.name)
        print(json.dumps({"removed": removed, "target": args.name}))
        return 0 if removed else 1
    if action == "scan":
        roots = args.roots or [Path.cwd()]
        added = registry.scan_roots(roots, max_depth=max(0, int(args.depth)))
        print(json.dumps({
            "added": [{"id": e.id, "path": str(e.path)} for e in added],
            "total": len(registry.all_entries()),
        }, indent=2))
        return 0
    return 1


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
    s.add_argument("query", nargs="*", default=[])
    s.add_argument("-n", "--limit", type=int, default=10)
    s.add_argument("--part", choices=config.ALL_PARTS, help="search one vector index only")
    s.add_argument("--image", action="store_true",
                   help="search the image vector index only (needs the image sidecar running)")
    s.add_argument("--kind", action="append", help="file|class|function|method|variable|comment|section|block|record|tag (repeatable)")
    s.add_argument("--lang", action="append", help="python|typescript|javascript|markdown|json|html|text|pdf|image (repeatable)")
    s.add_argument("--path", help="repo-relative path prefix")
    s.add_argument("--must", action="append", help="required term (AND, repeatable)")
    s.add_argument("--should", action="append", help="optional boost term (OR, repeatable)")
    s.add_argument("--not", dest="not_", action="append", help="excluded term (NOT, repeatable)")
    s.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None,
                   help="enable/disable stage-2 reranker (default: CODESEARCH_RERANK_ENABLED)")
    s.add_argument("--after", help="filter files modified at/after ISO date (YYYY-MM-DD), ISO datetime, or Unix timestamp")
    s.add_argument("--before", help="filter files modified at/before ISO date (YYYY-MM-DD), ISO datetime, or Unix timestamp")
    s.add_argument("--any", action="store_true", help="keyword list matches ANY word instead of all")
    s.add_argument("--strict", action="store_true", help="only keyword matches; vectors just reorder")
    s.add_argument("--no-semantic", action="store_true")
    s.add_argument("--no-keyword", action="store_true")
    s.add_argument("--full", action="store_true", help="show whole docstrings")
    s.add_argument("--preview", type=int, default=4, help="docstring lines shown per result")
    s.add_argument("--model", help="override active text embedding model for vector lookup")
    s.add_argument("--endpoint", help="override OpenAI-compatible embedding endpoint URL")
    s.add_argument("--json", action="store_true")

    i = sub.add_parser("index", help="scan the repo, update the graph, then embed new text")
    i.add_argument("--force", action="store_true", help="re-parse every file")
    i.add_argument("--embed", action=argparse.BooleanOptionalAction, default=True,
                   help="embed new or changed text after indexing (default: on; use --no-embed to skip)")
    i.add_argument("--embed-limit", type=int,
                   help="embed at most N new texts after indexing (default: all)")
    i.add_argument("--images", action="store_true",
                   help="also embed indexed image files into their own vector index (needs image-server)")
    e = sub.add_parser("embed", help="embed new or changed text into the vector indexes")
    e.add_argument("--limit", type=int, help="embed at most N new texts")
    e.add_argument("--path", help="repo-relative path prefix to embed first")
    e.add_argument("--images", action="store_true",
                   help="embed indexed image files into their own vector index (needs image-server)")
    e.add_argument("--prune-stale", action="store_true",
                   help="remove orphaned vec_cache rows from inactive embedding models before embedding")
    for target in (i, e):
        target.add_argument("--llama", nargs="?", const="embeddinggemma-300m", default=None,
                            help="auto-launch temporary llama-server (preset name or .gguf path; default: embeddinggemma-300m)")
        target.add_argument("--parallel", type=int, help="parallel llama-server slots (-np) when using --llama")
        target.add_argument("--model", help="override embedding model ID")
        target.add_argument("--endpoint", help="override OpenAI-compatible embedding endpoint URL")
        target.add_argument("--workers", type=int, help="override parallel HTTP worker threads")
        target.add_argument("--batch", type=int, help="override texts per HTTP batch request")
    for name in ("callers", "callees", "refs"):
        c = sub.add_parser(name)
        c.add_argument("symbol")
    o = sub.add_parser("outline", help="skeleton of one file")
    o.add_argument("path")
    o.add_argument("--comments", action="store_true")
    ins = sub.add_parser("inspect", help="dump every indexed node, embedded part text, FTS words, edge, and ref")
    ins.add_argument("path", nargs="?", help="relative file path (or omit to inspect all indexed files)")
    ins.add_argument("--json", action="store_true")
    rn = sub.add_parser("render", help="render a Set-of-Mark subgraph PNG for vision agents")
    rn.add_argument("symbol", help="symbol name or #node_id")
    rn.add_argument("--depth", type=int, default=1)
    rn.add_argument("--expand", default="", help="comma-separated #ids or symbols to expand in place")
    rn.add_argument("--kinds", default="", help="comma-separated edge kinds filter")
    rn.add_argument("--out", help="output PNG path")
    rn.add_argument("--json", action="store_true")
    sub.add_parser("stats")
    pk = sub.add_parser("pack", help="build contiguous .npy vector packs beside the DB for fast cold loads")
    pk.add_argument("--part", choices=config.ALL_PARTS, action="append", help="pack specific part(s) only")
    pk.add_argument("--model", help="override active text embedding model for packing")
    rg = sub.add_parser("registry", help="manage named indexes for multi-index MCP queries")
    rg_sub = rg.add_subparsers(dest="reg_cmd")
    rg_sub.add_parser("list", help="show registered and active indexes")
    rg_add = rg_sub.add_parser("add", help="register a folder under an alias")
    rg_add.add_argument("path", help="repository root directory")
    rg_add.add_argument("--name", help="index alias (default: slugified folder name)")
    rg_add.add_argument("--desc", help="short description for MCP callers")
    rg_rm = rg_sub.add_parser("remove", help="remove a registered index by alias or path")
    rg_rm.add_argument("name", help="index alias or directory path")
    rg_sc = rg_sub.add_parser("scan", help="discover .codesearch/codesearch.db folders and register them")
    rg_sc.add_argument("roots", nargs="*", help="parent folder(s) to scan")
    rg_sc.add_argument("--depth", type=int, default=2, help="max directory depth to walk (default: 2)")
    sv = sub.add_parser("serve", help="web UI")
    sv.add_argument("--port", type=int, default=config.WEB_PORT)
    mcp_p = sub.add_parser("mcp", help="MCP server over stdio (requires explicit `index` per tool call)")
    mcp_p.add_argument("--index", action="append", help="register extra index as name=path (repeatable)")
    srv_img = sub.add_parser("image-server", help="start the multimodal image embedding sidecar (port 1235)")
    srv_img.add_argument("--port", type=int, default=1235)
    srv_emb = sub.add_parser("embed-server", help="start parallel llama-server text embedder")
    srv_emb.add_argument("--model", default="embeddinggemma-300m",
                         help="preset name (embeddinggemma-300m, embeddinggemma-2, qwen3-0.6b, nomic-embed-text-v1.5) or .gguf path")
    srv_emb.add_argument("--port", type=int, default=1236)
    srv_emb.add_argument("--host", default="127.0.0.1")
    srv_emb.add_argument("--parallel", type=int, help="parallel server slots (-np)")
    srv_emb.add_argument("--slot-ctx", type=int, help="context size per slot")
    srv_emb.add_argument("--batch", type=int, help="logical batch size (-b)")
    srv_emb.add_argument("--ubatch", type=int, help="physical micro-batch size (-ub)")
    srv_emb.add_argument("--gpu-layers", type=int, default=99)

    args = p.parse_args(argv)
    if args.root:
        config.set_root(args.root)
    if args.cmd == "registry":
        return cmd_registry(args)
    if args.cmd == "mcp":
        from codesearch import mcp_server
        mcp_server.run(index_specs=args.index)
        return 0
    if args.cmd == "serve":
        from codesearch import app
        app.run(port=args.port)
        return 0
    if args.cmd == "image-server":
        from codesearch.image_server import main as image_server_main
        image_server_main()
        return 0
    if args.cmd == "embed-server":
        from codesearch import embed_server
        return embed_server.serve_forever(
            args.model,
            port=args.port,
            host=args.host,
            gpu_layers=args.gpu_layers,
            parallel=args.parallel,
            slot_ctx=args.slot_ctx,
            batch=args.batch,
            ubatch=args.ubatch,
        )

    @contextmanager
    def _embed_runtime():
        llama_spec = getattr(args, "llama", None)
        saved = {
            "EMBED_ENDPOINT": config.EMBED_ENDPOINT,
            "EMBED_MODEL": config.EMBED_MODEL,
            "EMBED_WORKERS": config.EMBED_WORKERS,
            "EMBED_BATCH": config.EMBED_BATCH,
        }
        if llama_spec:
            from codesearch import embed_server
            spec = embed_server.resolve_gguf_spec(
                llama_spec,
                model_id_override=getattr(args, "model", None),
            )
            workers = getattr(args, "workers", None)
            parallel = getattr(args, "parallel", None)
            batch = getattr(args, "batch", None)
            ctx_mgr = embed_server.LlamaEmbedServer(
                spec,
                parallel=parallel,
                workers=workers,
                req_batch=batch if batch is not None else 32,
            )
        else:
            ctx_mgr = nullcontext()
        try:
            with ctx_mgr:
                if getattr(args, "endpoint", None):
                    config.EMBED_ENDPOINT = args.endpoint
                if getattr(args, "model", None):
                    config.EMBED_MODEL = args.model
                if getattr(args, "workers", None) is not None:
                    config.EMBED_WORKERS = max(1, int(args.workers))
                if getattr(args, "batch", None) is not None:
                    config.EMBED_BATCH = max(1, int(args.batch))
                yield
        finally:
            for key, val in saved.items():
                setattr(config, key, val)

    from codesearch import diag
    cli_log_dir = Path(args.db).expanduser().resolve().parent if args.db else config.DATA_DIR
    cli_log = diag.configure("cli", log_dir=cli_log_dir)
    cli_log.debug("CLI_START cmd=%s root=%s db=%s", args.cmd, config.REPO_ROOT, args.db or config.DB_PATH)

    conn = db.connect(args.db)
    try:
        if args.cmd == "index":
            def progress(msg: str) -> None:
                print(msg, flush=True)
            res = indexer.index(conn, force=args.force, progress=progress)
            print(json.dumps({k: v for k, v in res.items() if k != "problems"}), flush=True)
            for prob in res["problems"]:
                print("parse problem:", prob)
            if args.embed or args.images:
                from codesearch import embedder
                with _embed_runtime():
                    # Text and image embedding use different models in different spaces. One
                    # being down must not stop the other from building.
                    if args.embed:
                        try:
                            embedder.preflight()
                        except RuntimeError as exc:
                            print("text embedding skipped:", exc, file=sys.stderr, flush=True)
                    if args.images:
                        try:
                            embedder.preflight_images()
                        except RuntimeError as exc:
                            print("image embedding skipped:", exc, file=sys.stderr, flush=True)
                    print(json.dumps(embedder.build(conn, limit=args.embed_limit, images=args.images)), flush=True)
            if not args.db:
                from codesearch import registry
                registry.upsert_entry(config.REPO_ROOT)
            return 0
        if args.cmd == "embed":
            from codesearch import embedder
            with _embed_runtime():
                if getattr(args, "prune_stale", False):
                    pruned = db.prune_stale_vectors(conn)
                    if pruned:
                        print(json.dumps({"pruned_stale_models": pruned}), flush=True)
                print(json.dumps(embedder.build(conn, limit=args.limit, path_prefix=args.path, images=args.images)), flush=True)
            return 0
        if args.cmd == "stats":
            print(json.dumps(db.stats(conn), indent=2))
            return 0
        if args.cmd == "pack":
            from codesearch import embedder
            orig_model = config.EMBED_MODEL
            if getattr(args, "model", None):
                config.EMBED_MODEL = args.model
            try:
                parts = tuple(args.part) if getattr(args, "part", None) else None
                print(json.dumps(embedder.pack_vectors(conn, parts=parts), indent=2), flush=True)
            finally:
                config.EMBED_MODEL = orig_model
            return 0
        if conn.execute("SELECT 1 FROM nodes LIMIT 1").fetchone() is None:
            print("index is empty - run: python tools/codesearch/cli.py index")
            return 1
        if args.cmd == "search":
            with _embed_runtime():
                return cmd_search(args, conn)
        if args.cmd in ("callers", "callees", "refs"):
            return cmd_graph(args, conn)
        if args.cmd == "outline":
            return cmd_outline(args, conn)
        if args.cmd == "inspect":
            return cmd_inspect(args, conn)
        if args.cmd == "render":
            return cmd_render(args, conn)
    finally:
        diag.flush()
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
