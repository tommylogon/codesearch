"""codesearch MCP server (stdio). Compact, structure-dense results for agents.

Tools:
    search_code(query, part, kind, language, path, match_all, strict, limit)
    code_neighbors(symbol, direction, depth)
    find_references(symbol)
    file_outline(path)
    get_code(path, start, end)
    codesearch_stats()
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "codesearch"

from fastmcp import FastMCP                                  # noqa: E402
from fastmcp.utilities.types import Image                    # noqa: E402

from . import config, db, graph, renderer                    # noqa: E402
from .search import Searcher                                 # noqa: E402

MCP_INSTRUCTIONS = """\
Hybrid keyword + 4-part semantic search and relational graph server.

Best practices:
1. Start with `search_code(query=...)` for both exact identifiers and natural-language concepts.
   - Use `part='comment'` for docstrings, design rationales, and prose sections.
   - Use `part='function_name'` or `part='function'` for code definitions and signatures.
   - Use `part='filename'` to find files by path or module topic.
   - Set `match_all=False` when passing several alternative keywords, or `strict=True` to require exact keyword matches.
   - Filter by `language` ('python', 'typescript', 'javascript', 'markdown', 'json', 'html', 'text') or `kind` ('file', 'class', 'function', 'method', 'variable', 'comment', 'section', 'block', 'record', 'tag').
2. Every `search_code` hit already includes its 1-hop graph relations (`called_by`, `calls`, `referenced_by`, `reads`, `writes`).
3. Use `code_neighbors(symbol=..., direction='both', depth=1)` to walk callers/callees, or `find_references(symbol=...)` to see all read (`[r]`) and write (`[w]`) sites for a variable/attribute.
4. Use `file_outline(path=...)` to inspect a file's structure before reading lines with `get_code(path=..., start=..., end=...)`.
5. For vision-capable agents, call `render_code_graph(symbol=..., depth=1, expand='id1,id2')` to render a Set-of-Mark (`#id`) graph PNG and expand nodes turn-by-turn.
"""

mcp = FastMCP("codesearch", instructions=MCP_INSTRUCTIONS)

_conn = None
_searcher: Searcher | None = None


def _db():
    global _conn, _searcher
    if _conn is None:
        _conn = db.connect()
        _conn.row_factory = __import__("sqlite3").Row
        _searcher = Searcher(_conn)
    return _conn, _searcher


def _compact(hit: dict) -> dict:
    """One result, trimmed of presentation noise. Docstrings are returned whole."""
    out = {
        "id": hit["id"], "kind": hit["kind"],
        "name": hit["qualified_name"].split("::", 1)[-1] if "::" in hit["qualified_name"] else hit["name"],
        "file": hit["file"], "lines": [hit["line_start"], hit["line_end"]],
        "score": hit["score"], "matched": hit["matched"],
    }
    if hit["signature"]:
        out["signature"] = hit["signature"]
    if hit["docstring"]:
        out["doc"] = hit["docstring"]
    for key, rel in hit["relations"].items():
        out[key] = rel["items"] if rel["count"] == len(rel["items"]) else {"count": rel["count"], "first": rel["items"]}
    return out


@mcp.tool()
def search_code(query: str, part: str | None = None, kind: str | None = None,
                language: str | None = None, path: str | None = None,
                match_all: bool = True, strict: bool = False, limit: int = 8) -> dict:
    """Hybrid search over code, comments and docs: every-word keyword match OR semantic
    match of the whole sentence, fused by rank. part: filename|function_name|function|comment
    narrows to one vector index. kind: file|class|function|method|variable|comment|section|block|record|tag.
    language: python|typescript|javascript|markdown|html|text. path: repo-relative prefix.
    strict=true keeps only keyword matches. Each result lists callers/callees/references."""
    conn, searcher = _db()
    hits = searcher.search(query, limit=limit, kinds=[kind] if kind else None,
                           languages=[language] if language else None, path_prefix=path,
                           part=part, match_all=match_all, strict=strict)
    out: dict = {"results": [_compact(h) for h in hits]}
    if searcher.last_vector_error:
        out["note"] = "keyword only: " + searcher.last_vector_error
    return out


@mcp.tool()
def code_neighbors(symbol: str, direction: str = "both", depth: int = 1) -> dict:
    """Callers, callees and variable references of a symbol (name, Class.method or path::name).
    direction: callers|callees|both. depth>1 walks the call graph transitively."""
    conn, _ = _db()
    found = graph.find_symbol(conn, symbol)
    if not found:
        return {"error": f"no symbol matches {symbol!r}"}
    node = found[0]
    out: dict = {"symbol": node["qualified_name"], "file": node["file"], "line": node["line_start"],
                 "other_matches": [f"{f['qualified_name']}:{f['line_start']}" for f in found[1:6]]}

    def fmt(rows):
        return [{"name": r["qualified_name"].split("::", 1)[-1], "file": r["file"], "line": r["line_start"],
                 **({"mode": r["mode"]} if "mode" in r else {})} for r in rows]

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
    return out


@mcp.tool()
def find_references(symbol: str) -> dict:
    """Everything that touches a symbol: callers (functions), readers/writers (variables),
    importers and subclasses."""
    conn, _ = _db()
    found = graph.find_symbol(conn, symbol)
    if not found:
        return {"error": f"no symbol matches {symbol!r}"}
    node = found[0]
    rel = graph.summary(conn, node["id"], limit=200)
    return {"symbol": node["qualified_name"], "file": node["file"], "line": node["line_start"],
            "kind": node["kind"], **{k: v["items"] for k, v in rel.items()
                                     if k in ("called_by", "referenced_by", "imported_by", "inherited_by", "linked_from")}}


@mcp.tool()
def file_outline(path: str) -> dict:
    """Skeleton of one file (classes, functions, variables, doc sections) with line numbers,
    instead of reading the whole file."""
    conn, _ = _db()
    out = graph.outline(conn, path.replace("\\", "/"))
    if out is None:
        return {"error": f"{path} is not indexed"}
    out["items"] = [it for it in out["items"] if it["kind"] != "comment"]
    return out


@mcp.tool()
def get_code(path: str, start: int, end: int) -> dict:
    """Return source lines [start, end] (1-based, inclusive) of an indexed repo file."""
    target = (config.REPO_ROOT / path).resolve()
    if config.REPO_ROOT not in target.parents or not target.is_file():
        return {"error": "path must be a file inside the repo"}
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start, end = max(1, start), min(len(lines), end)
    return {"file": path, "start": start, "end": end, "total_lines": len(lines),
            "code": "\n".join(f"{n}: {lines[n - 1]}" for n in range(start, end + 1))}


@mcp.tool()
def codesearch_stats() -> dict:
    """Index health: file/node/edge counts, vector coverage, last index time."""
    conn, _ = _db()
    return db.stats(conn)


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
def render_code_graph(symbol: str | int, depth: int = 1, expand: str = "",
                      edge_kinds: str = "", return_image: bool = False) -> dict | list:
    """Render an interactive Set-of-Mark (SoM) code graph PNG for vision-capable agents.
    Every node card in the rendered PNG displays its numeric mark (#id), symbol name, kind,
    and file:line. Pass any visible #id into `expand` (comma-separated, e.g. '231789,234597')
    to expand those nodes in place on the next call, or pass `return_image=True` to receive
    an inline MCP Image block alongside the structured node/edge metadata."""
    conn, _ = _db()
    root_id = _resolve_node_id(conn, symbol)
    if root_id is None:
        return {"error": f"no symbol or node id matches {symbol!r}"}
    expand_ids: list[int] = []
    for part in (expand or "").split(","):
        part = part.strip()
        if not part:
            continue
        eid = _resolve_node_id(conn, part)
        if eid is not None:
            expand_ids.append(eid)
    kinds = tuple(k.strip() for k in (edge_kinds or "").split(",") if k.strip()) or None
    res = renderer.render_subgraph(conn, root_id, depth=max(1, min(depth, 2)),
                                   expand_ids=expand_ids, kinds=kinds)
    if return_image:
        return [res, Image(path=res["image_path"])]
    return res


def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
