"""codesearch MCP server (stdio). Multi-index, structure-dense tools for agents.

Every index-scoped tool REQUIRES an explicit `index` argument (registered alias
or indexed repository directory path). No default repository is silently queried.

Tools:
    list_indexes()
    search_code(index, query, part, kind, language, path, match_all, strict, limit, after, before)
    code_neighbors(index, symbol, direction, depth)
    find_references(index, symbol)
    file_outline(index, path)
    get_code(index, path, start, end)
    render_code_graph(index, symbol, depth, expand, edge_kinds, return_image)
    codesearch_stats(index)
    codesearch_logs(lines, index, component)
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "codesearch"

from fastmcp import FastMCP                                  # noqa: E402
from fastmcp.utilities.types import Image                    # noqa: E402

from . import config, db, diag, graph, registry, renderer    # noqa: E402
from .parsers import pdf_parser                              # noqa: E402
from .search import Searcher                                 # noqa: E402

MCP_INSTRUCTIONS = """\
Multi-index hybrid keyword + 4-part semantic search and relational graph server.

Best practices:
0. REQUIRED `index`: there is NO default workspace. Every tool call (except `list_indexes` and `codesearch_logs`) MUST pass `index` (alias from `list_indexes()` or repo path).
1. `search_code` recipes (pick the right pattern):
   - Concept / natural-language search: `search_code(index=..., query="how fear panic works", part="comment")`
     Note: `query` ANDs all its words in the keyword leg, so long sentences rely on semantic search unless you split keywords into `should_terms`.
   - Exact symbol / strict keyword search: `search_code(index=..., must_terms=["ThreatDetector"], strict=True)`
   - Boolean + hybrid search: `search_code(index=..., query="", must_terms=["fear"], should_terms=["panic", "threat"], not_terms=["goblin"])`
     Rule: `not_terms` ALWAYS excludes semantic hits; `must_terms` only blocks non-matching semantic hits when `strict=True`.
2. Filters on `search_code`:
   - `part`: `'comment'` (docstrings/prose), `'function_name'` or `'function'` (code), `'filename'` (paths/files).
   - `path`: repo-relative prefix with forward slashes ONLY (e.g. `'engine/'`, NEVER `'F:\\AI\\...'`).
   - `language`: `'python'`, `'typescript'`, `'javascript'`, `'markdown'`, `'json'`, `'csv'`, `'html'`, `'text'`, `'pdf'`, `'image'`.
   - `kind`: `'file'`, `'class'`, `'function'`, `'method'`, `'variable'`, `'comment'`, `'section'`, `'block'`, `'record'`, `'tag'`.
   - `after` / `before`: `'YYYY-MM-DD'`, ISO datetime, or Unix epoch.
   - `rerank=True`: optional stage-2 cross-encoder reranking (falls back cleanly to RRF if offline).
3. Every `search_code` hit already includes its 1-hop graph relations (`called_by`, `calls`, `referenced_by`, `reads`, `writes`).
4. Walk deeper with `code_neighbors(index=..., symbol=..., direction='both', depth=1)` or `find_references(index=..., symbol=...)` (`[r]` reads / `[w]` writes).
5. Inspect structure before reading: `file_outline(index=..., path=...)` -> `get_code(index=..., path=..., start=..., end=...)` (for `.pdf`, `start`/`end` are 1-based pages, max 20/call).
6. Vision agents: `render_code_graph(index=..., symbol=..., depth=1, expand='id1,id2')` renders a Set-of-Mark (`#id`) PNG.
"""

mcp = FastMCP("codesearch", instructions=MCP_INSTRUCTIONS)

_CLI_INDEX_SPECS: list[str] = []
_vector_caches: dict[str, dict] = {}
_vector_locks: dict[str, threading.RLock] = {}
_caches_lock = threading.Lock()
_banner_logged: set[str] = set()
_log = diag.get_logger("mcp")
_VALID_LOG_COMPONENTS = frozenset({"mcp", "cli", "web", "public_api"})


def _log_dir() -> Path:
    override = os.environ.get("CODESEARCH_MCP_LOG_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return registry.registry_path().parent


def _ensure_log() -> logging.Logger:
    return diag.configure("mcp", log_dir=_log_dir())


def clear_caches() -> None:
    """Clear per-index vector caches and startup banner flags (primarily for tests)."""
    with _caches_lock:
        _vector_caches.clear()
        _vector_locks.clear()
        _banner_logged.clear()


def _index_error_payload(
    exc: Exception,
    available: dict[str, registry.IndexEntry] | None = None,
) -> dict:
    entries = available if available is not None else registry.all_entries(extra_specs=_CLI_INDEX_SPECS)
    return {
        "error": str(exc),
        "hint": "Call list_indexes() to see available indexes, then pass `index` explicitly.",
        "available_indexes": registry.index_summary(entries),
    }


@contextmanager
def _open_index(index: str):
    log = _ensure_log()
    entry = registry.resolve_index(index, extra_specs=_CLI_INDEX_SPECS)
    if not entry.db_path.is_file():
        raise registry.IndexResolutionError(
            f"Index {entry.id!r} has no database at {entry.db_path}. "
            f"Run `codesearch -r \"{entry.path}\" index` first.",
            registry.all_entries(extra_specs=_CLI_INDEX_SPECS),
        )

    cache_key = str(entry.path).lower()
    with _caches_lock:
        vec_cache = _vector_caches.setdefault(cache_key, {})
        vec_lock = _vector_locks.setdefault(cache_key, threading.RLock())
        first_open = cache_key not in _banner_logged
        if first_open:
            _banner_logged.add(cache_key)

    t0 = time.perf_counter()
    conn = db.connect(entry.db_path)
    try:
        if first_open:
            diag.log_startup_banner(
                log,
                component=f"mcp:{entry.id}",
                repo_root=entry.path,
                db_path=entry.db_path,
                conn=conn,
            )
            log.info("MCP_DB_READY index=%s connect_ms=%.1f", entry.id, (time.perf_counter() - t0) * 1000.0)
            diag.flush()
        searcher = Searcher(
            conn,
            repo_root=entry.path,
            vector_cache=vec_cache,
            vector_lock=vec_lock,
            background_pack_build=True,
        )
        yield entry, conn, searcher
    finally:
        conn.close()


@contextmanager
def _trace_tool(tool_name: str, **fields):
    log = _ensure_log()
    t0 = time.perf_counter()
    arg_str = " ".join(f"{k}={v!r}" for k, v in fields.items() if v is not None and v != "")
    log.debug("MCP_CALL tool=%s %s", tool_name, arg_str)
    diag.flush()
    span: dict[str, object] = {}
    try:
        with diag.watch_slow(f"mcp:{tool_name}", logger=log):
            yield span
    except Exception:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        log.exception("MCP_ERR tool=%s ms=%.1f %s", tool_name, dt_ms, arg_str)
        diag.flush()
        raise
    dt_ms = (time.perf_counter() - t0) * 1000.0
    level = logging.WARNING if (dt_ms >= config.SLOW_QUERY_MS or "error" in span) else logging.DEBUG
    span_str = diag.format_timings(span)
    log.log(level, "MCP_DONE tool=%s ms=%.1f %s", tool_name, dt_ms, span_str)
    diag.flush()


def _call_index(tool_name: str, index: str, fn, **fields) -> dict | list:
    with _trace_tool(tool_name, index=index, **fields) as span:
        try:
            with _open_index(index) as (entry, conn, searcher):
                span["resolved"] = entry.id
                return fn(entry, conn, searcher, span)
        except registry.IndexResolutionError as exc:
            span["error"] = "invalid_index"
            return _index_error_payload(exc, exc.available)


def _compact(hit: dict) -> dict:
    """One result, trimmed of presentation noise. Docstrings are returned whole."""
    out = {
        "id": hit["id"],
        "kind": hit["kind"],
        "name": hit["qualified_name"].split("::", 1)[-1] if "::" in hit["qualified_name"] else hit["name"],
        "file": hit["file"],
        "lines": [hit["line_start"], hit["line_end"]],
        "score": hit["score"],
        "matched": hit["matched"],
    }
    if hit.get("sim") is not None:
        out["sim"] = hit["sim"]
    if hit["signature"]:
        out["signature"] = hit["signature"]
    if hit["docstring"]:
        out["doc"] = hit["docstring"]
    for key, rel in hit["relations"].items():
        out[key] = rel["items"] if rel["count"] == len(rel["items"]) else {"count": rel["count"], "first": rel["items"]}
    return out


@mcp.tool()
def list_indexes() -> dict:
    """List all registered/discovered codesearch indexes so callers know which `index` to pass."""
    with _trace_tool("list_indexes") as span:
        entries = registry.all_entries(extra_specs=_CLI_INDEX_SPECS)
        items = registry.index_summary(entries)
        span["count"] = len(items)
        return {
            "count": len(items),
            "usage": "Pass `index` (id or indexed directory path) on every search, graph, outline, code, and stats call.",
            "registry_file": str(registry.registry_path()),
            "indexes": items,
        }


@mcp.tool()
def search_code(
    index: str,
    query: str = "",
    part: str | None = None,
    kind: str | None = None,
    language: str | None = None,
    path: str | None = None,
    not_path: str | list[str] | None = None,
    strict: bool = False,
    rerank: bool | None = None,
    must_terms: list[str] | None = None,
    should_terms: list[str] | None = None,
    not_terms: list[str] | None = None,
    limit: int = 8,
    after: str | None = None,
    before: str | None = None,
) -> dict:
    """Hybrid search over a required `index` (FTS5 keyword + 4-part semantic fused by RRF).
    index: required alias (from list_indexes()) or repo path.
    query: concept/identifier (ANDs all words in keyword leg; use should_terms for OR bags).
    must_terms / should_terms / not_terms: explicit AND / OR / NOT lists. not_terms always excludes semantic hits; must_terms only excludes non-matching semantic hits when strict=true.
    part: filename|function_name|function|comment.
    kind: file|class|function|method|variable|comment|section|block|record|tag.
    language: python|typescript|javascript|markdown|json|csv|html|text|pdf|image.
    path / not_path: repo-relative forward-slash prefix (e.g. 'engine/', not_path='dev_tasks/'), never an absolute path.
    after/before: YYYY-MM-DD, ISO datetime, or Unix epoch.
    strict=true keeps only keyword matches. rerank=true runs optional stage-2 reranking.
    Each result includes sim (raw max cosine), 1-hop graph relations."""

    def _run(entry: registry.IndexEntry, _conn, searcher: Searcher, span: dict) -> dict:
        try:
            hits = searcher.search(
                query,
                limit=limit,
                kinds=[kind] if kind else None,
                languages=[language] if language else None,
                path_prefix=path,
                not_path=not_path,
                part=part,
                strict=strict,
                rerank=rerank,
                must_terms=must_terms,
                should_terms=should_terms,
                not_terms=not_terms,
                after=after,
                before=before,
            )
        except ValueError as exc:
            span["error"] = str(exc)
            return {"index": entry.id, "error": f"invalid date filter: {exc}"}
        out: dict = {"index": entry.id, "results": [_compact(h) for h in hits]}
        notes: list[str] = []
        if searcher.last_vector_error:
            notes.append("keyword only: " + searcher.last_vector_error)
            span["vec_err"] = searcher.last_vector_error
        if getattr(searcher, "last_rerank_error", None):
            notes.append("rerank fallback: " + str(searcher.last_rerank_error))
            span["rerank_err"] = searcher.last_rerank_error
        if notes:
            out["note"] = "; ".join(notes)
        span["hits"] = len(out["results"])
        if searcher.last_timings:
            span["breakdown"] = diag.format_timings(searcher.last_timings)
            if os.environ.get("CODESEARCH_DEBUG_TIMINGS", "").strip().lower() in ("1", "true", "yes"):
                out["timings_ms"] = searcher.last_timings
        return out

    return _call_index(
        "search_code",
        index,
        _run,
        q=query[:80],
        part=part,
        kind=kind,
        lang=language,
        path=path,
        limit=limit,
        strict=strict,
    )


@mcp.tool()
def code_neighbors(index: str, symbol: str, direction: str = "both", depth: int = 1) -> dict:
    """Callers, callees and variable references of a symbol in a required `index`
    (name, Class.method or path::name). direction: callers|callees|both. depth>1 walks transitively."""

    def _run(entry: registry.IndexEntry, conn, _searcher, span: dict) -> dict:
        found = graph.find_symbol(conn, symbol)
        if not found:
            span["error"] = "not_found"
            return {"index": entry.id, "error": f"no symbol matches {symbol!r}"}
        node = found[0]
        out: dict = {
            "index": entry.id,
            "symbol": node["qualified_name"],
            "file": node["file"],
            "line": node["line_start"],
            "other_matches": [f"{f['qualified_name']}:{f['line_start']}" for f in found[1:6]],
        }

        def fmt(rows):
            return [
                {
                    "name": r["qualified_name"].split("::", 1)[-1],
                    "file": r["file"],
                    "line": r["line_start"],
                    **({"mode": r["mode"]} if "mode" in r else {}),
                }
                for r in rows
            ]

        def walk(start: int, fn):
            seen, layer, acc = {start}, [start], []
            for _ in range(max(1, min(depth, 4))):
                nxt = []
                for nid in layer:
                    for r in fn(conn, nid):
                        if r["id"] not in seen:
                            seen.add(r["id"])
                            nxt.append(r["id"])
                            acc.append(r)
                layer = nxt
            return acc

        if direction in ("callers", "both"):
            out["callers"] = fmt(walk(node["id"], graph.callers))
        if direction in ("callees", "both"):
            out["callees"] = fmt(walk(node["id"], graph.callees))
        if node["kind"] == "variable":
            out["referenced_by"] = fmt(graph.references(conn, node["id"]))
        span["matches"] = len(found)
        return out

    return _call_index("code_neighbors", index, _run, symbol=symbol, direction=direction, depth=depth)


@mcp.tool()
def find_references(index: str, symbol: str) -> dict:
    """Everything that touches a symbol in a required `index`: callers (functions),
    readers/writers (variables), importers, subclasses, and wikilinks."""

    def _run(entry: registry.IndexEntry, conn, _searcher, span: dict) -> dict:
        found = graph.find_symbol(conn, symbol)
        if not found:
            span["error"] = "not_found"
            return {"index": entry.id, "error": f"no symbol matches {symbol!r}"}
        node = found[0]
        rel = graph.summary(conn, node["id"], limit=200)
        span["rel_keys"] = len(rel)
        return {
            "index": entry.id,
            "symbol": node["qualified_name"],
            "file": node["file"],
            "line": node["line_start"],
            "kind": node["kind"],
            **{
                k: v["items"]
                for k, v in rel.items()
                if k in ("called_by", "referenced_by", "imported_by", "inherited_by", "linked_from")
            },
        }

    return _call_index("find_references", index, _run, symbol=symbol)


@mcp.tool()
def file_outline(index: str, path: str) -> dict:
    """Skeleton of one file in a required `index` (classes, functions, variables, doc sections)
    with line numbers, instead of reading the whole file."""

    def _run(entry: registry.IndexEntry, conn, _searcher, span: dict) -> dict:
        out = graph.outline(conn, path.replace("\\", "/"))
        if out is None:
            span["error"] = "not_indexed"
            return {"index": entry.id, "error": f"{path} is not indexed in {entry.id}"}
        out["index"] = entry.id
        out["items"] = [it for it in out["items"] if it["kind"] != "comment"]
        span["items"] = len(out["items"])
        return out

    return _call_index("file_outline", index, _run, path=path)


@mcp.tool()
def get_code(index: str, path: str, start: int, end: int) -> dict:
    """Return source lines (or PDF pages, max 20/call) [start, end] (1-based, inclusive)
    of a file inside the required `index` repository."""

    def _run(entry: registry.IndexEntry, _conn, _searcher, span: dict) -> dict:
        root = entry.path.resolve()
        target = (root / path).resolve()
        if root not in target.parents or not target.is_file():
            span["error"] = "outside_or_missing"
            return {"index": entry.id, "error": f"path must be a file inside index {entry.id!r}"}
        if target.suffix.lower() == ".pdf":
            res = pdf_parser.read_pages(target, start, end, max_pages=20)
            if "error" in res:
                span["error"] = res["error"]
                return {"index": entry.id, "error": res["error"]}
            out = {
                "index": entry.id,
                "file": path,
                "unit": "pages",
                "start": res["start"],
                "end": res["end"],
                "total_pages": res["total_pages"],
                "code": "\n".join(res["lines"]),
            }
            if res["clamped"]:
                out["note"] = (
                    f"clamped to {res['end'] - res['start'] + 1} pages per call; "
                    f"call get_code again from page {res['end'] + 1}"
                )
            span["pages"] = res["end"] - res["start"] + 1
            return out
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        lo, hi = max(1, start), min(len(lines), end)
        span["lines"] = max(0, hi - lo + 1)
        return {
            "index": entry.id,
            "file": path,
            "start": lo,
            "end": hi,
            "total_lines": len(lines),
            "code": "\n".join(f"{n}: {lines[n - 1]}" for n in range(lo, hi + 1)),
        }

    return _call_index("get_code", index, _run, path=path, start=start, end=end)


@mcp.tool()
def codesearch_stats(index: str) -> dict:
    """Index health for a required `index`: file/node/edge counts, vector coverage, paths, and log file."""

    def _run(entry: registry.IndexEntry, conn, _searcher, span: dict) -> dict:
        st = db.stats(conn)
        st["index"] = entry.id
        st["repo_root"] = str(entry.path)
        st["db_path"] = str(entry.db_path)
        st["vecpacks"] = (entry.data_dir / "vecpacks").is_dir()
        st["embed_endpoint"] = config.EMBED_ENDPOINT
        st["embed_model"] = config.EMBED_MODEL
        st["log_file"] = str(diag.current_log_path() or "")
        span["files"] = st["files"]
        span["nodes"] = st["nodes"]
        return st

    return _call_index("codesearch_stats", index, _run)


@mcp.tool()
def codesearch_logs(lines: int = 80, index: str | None = None, component: str = "mcp") -> dict:
    """Return the tail of the MCP diagnostic log (timings, fallbacks, stall watchdogs),
    or pass `index` + `component` ('cli'|'web'|'public_api') to inspect an index-local log."""
    _ensure_log()
    comp = (component or "mcp").strip().lower()
    if comp not in _VALID_LOG_COMPONENTS:
        return {"error": f"invalid component {component!r}; choose from {sorted(_VALID_LOG_COMPONENTS)}"}

    target_index: str | None = None
    if index:
        try:
            entry = registry.resolve_index(index, extra_specs=_CLI_INDEX_SPECS)
        except registry.IndexResolutionError as exc:
            return _index_error_payload(exc, exc.available)
        target_index = entry.id
        candidate = entry.data_dir / f"{comp}.log"
        log_path = candidate if (candidate.is_file() or comp != "mcp") else diag.current_log_path()
    else:
        log_path = diag.current_log_path()

    tail = diag.tail_lines(log_path, limit=lines)
    return {
        "index": target_index,
        "component": comp,
        "log_file": str(log_path or ""),
        "count": len(tail),
        "lines": tail,
    }


def _resolve_node_id(conn, token: str | int) -> int | None:
    if isinstance(token, int):
        return token if graph.get_node(conn, token) else None
    s = str(token).strip().lstrip("#")
    if s.isdigit():
        nid = int(s)
        if graph.get_node(conn, nid):
            return nid
    found = graph.find_symbol(conn, s, kinds=("file", "class", "function", "method", "variable", "section", "record"))
    return found[0]["id"] if found else None


@mcp.tool()
def render_code_graph(
    index: str,
    symbol: str | int,
    depth: int = 1,
    expand: str = "",
    edge_kinds: str = "",
    return_image: bool = False,
) -> dict | list:
    """Render an interactive Set-of-Mark (SoM) code graph PNG for a required `index`.
    Every node card displays its numeric mark (#id), symbol name, kind, and file:line.
    Pass visible #id values into `expand` (comma-separated) to expand neighbors in place,
    or `return_image=True` to include an inline MCP Image block."""

    def _run(entry: registry.IndexEntry, conn, _searcher, span: dict) -> dict | list:
        root_id = _resolve_node_id(conn, symbol)
        if root_id is None:
            span["error"] = "not_found"
            return {"index": entry.id, "error": f"no symbol or node id matches {symbol!r} in {entry.id}"}
        expand_ids: list[int] = []
        for part in (expand or "").split(","):
            part = part.strip()
            if not part:
                continue
            eid = _resolve_node_id(conn, part)
            if eid is not None:
                expand_ids.append(eid)
        kinds = tuple(k.strip() for k in (edge_kinds or "").split(",") if k.strip()) or None
        res = renderer.render_subgraph(
            conn,
            root_id,
            depth=max(1, min(depth, 2)),
            expand_ids=expand_ids,
            kinds=kinds,
            renders_dir=entry.data_dir / "renders",
        )
        res["index"] = entry.id
        span["nodes"] = res.get("node_count")
        span["edges"] = res.get("edge_count")
        if return_image:
            return [res, Image(path=res["image_path"])]
        return res

    return _call_index(
        "render_code_graph",
        index,
        _run,
        symbol=symbol,
        depth=depth,
        expand=expand,
        edge_kinds=edge_kinds,
        return_image=return_image,
    )


def _build_instructions(entries: dict[str, registry.IndexEntry]) -> str:
    if not entries:
        return MCP_INSTRUCTIONS
    lines = [MCP_INSTRUCTIONS.rstrip(), "", "Registered indexes (pass one as `index`):"]
    for item in registry.index_summary(entries):
        desc = f" - {item['description']}" if item.get("description") else ""
        status = "ready" if item["ready"] else "not indexed"
        lines.append(f"- {item['id']}: {item['path']} ({status}){desc}")
    return "\n".join(lines) + "\n"


def run(index_specs: list[str] | None = None) -> None:
    global _CLI_INDEX_SPECS
    _CLI_INDEX_SPECS = list(index_specs or [])
    log = _ensure_log()
    entries = registry.all_entries(extra_specs=_CLI_INDEX_SPECS)
    instr = _build_instructions(entries)
    mcp.instructions = instr
    if getattr(mcp, "_mcp_server", None) is not None:
        mcp._mcp_server.instructions = instr
    log.info(
        "MCP_BOOT pid=%d registry=%s indexes=%s log=%s",
        os.getpid(),
        registry.registry_path(),
        sorted(entries),
        diag.current_log_path(),
    )
    diag.flush()
    mcp.run()


if __name__ == "__main__":
    run()
