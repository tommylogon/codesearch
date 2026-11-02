"""Graph queries over the nodes/edges tables: symbol lookup, callers, callees,
references, neighborhoods for the web graph, and file outlines."""
from __future__ import annotations

import sqlite3

NODE_COLS = ("n.id, n.kind, n.name, n.qualified_name, n.signature, n.docstring, n.line_start, "
             "n.line_end, f.rel_path, f.language")
NODE_FROM = "FROM nodes n JOIN files f ON f.id = n.file_id"

SYMBOL_KINDS = ("function", "method", "class", "variable")


def _brief(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "kind": row["kind"], "name": row["name"],
            "qualified_name": row["qualified_name"], "file": row["rel_path"],
            "language": row["language"], "line_start": row["line_start"],
            "line_end": row["line_end"], "signature": row["signature"]}


def get_node(conn: sqlite3.Connection, node_id: int) -> dict | None:
    row = conn.execute(f"SELECT {NODE_COLS}, n.source {NODE_FROM} WHERE n.id = ?", (node_id,)).fetchone()
    if row is None:
        return None
    d = _brief(row)
    d["docstring"] = row["docstring"]
    d["source"] = row["source"]
    return d


def find_symbol(conn: sqlite3.Connection, symbol: str, kinds: tuple[str, ...] | None = None,
                limit: int = 20) -> list[dict]:
    """Resolve a user-typed symbol to nodes. Accepts a bare name (`syncGraphData`), a
    qualified suffix (`GraphNetwork::syncGraphData`, `GraphNetwork.syncGraphData`) or a
    full `path::Class::name`. Exact name hits rank before suffix hits."""
    sym = symbol.strip().replace(".", "::")
    kinds = kinds or SYMBOL_KINDS
    ph = ",".join("?" * len(kinds))
    rows = conn.execute(
        f"SELECT {NODE_COLS} {NODE_FROM} WHERE n.kind IN ({ph}) AND "
        "(n.name = ? OR n.qualified_name = ? OR n.qualified_name LIKE ?) "
        "ORDER BY (n.qualified_name = ?) DESC, (n.name = ?) DESC, n.kind, f.rel_path LIMIT ?",
        (*kinds, sym, sym, f"%::{sym}", sym, sym, limit)).fetchall()
    return [_brief(r) for r in rows]


def _edge_rows(conn, node_id: int, kind: str, direction: str):
    if direction == "out":
        return conn.execute(
            f"SELECT {NODE_COLS}, e.meta AS meta FROM edges e JOIN nodes n ON n.id = e.target_id "
            "JOIN files f ON f.id = n.file_id WHERE e.source_id = ? AND e.kind = ? "
            "ORDER BY f.rel_path, n.line_start", (node_id, kind)).fetchall()
    return conn.execute(
        f"SELECT {NODE_COLS}, e.meta AS meta FROM edges e JOIN nodes n ON n.id = e.source_id "
        "JOIN files f ON f.id = n.file_id WHERE e.target_id = ? AND e.kind = ? "
        "ORDER BY f.rel_path, n.line_start", (node_id, kind)).fetchall()


def callers(conn, node_id: int) -> list[dict]:
    return [_brief(r) for r in _edge_rows(conn, node_id, "CALLS", "in")]


def callees(conn, node_id: int) -> list[dict]:
    return [_brief(r) for r in _edge_rows(conn, node_id, "CALLS", "out")]


def references(conn, node_id: int) -> list[dict]:
    """Who touches this variable, with read/write mode."""
    out = []
    for r in _edge_rows(conn, node_id, "REFERENCES", "in"):
        d = _brief(r)
        d["mode"] = {"r": "READS", "w": "WRITES"}.get(r["meta"], r["meta"])
        out.append(d)
    return out


def summary(conn, node_id: int, limit: int = 6) -> dict:
    """Compact relationship summary used by search results and MCP: names plus counts,
    never silently dropping the fact that more exist."""
    out: dict = {}
    for key, kind, direction in (("calls", "CALLS", "out"), ("called_by", "CALLS", "in"),
                                 ("inherits", "INHERITS", "out"), ("inherited_by", "INHERITS", "in"),
                                 ("imports", "IMPORTS", "out"), ("imported_by", "IMPORTS", "in"),
                                 ("links_to", "LINKS_TO", "out"), ("linked_from", "LINKS_TO", "in")):
        rows = _edge_rows(conn, node_id, kind, direction)
        if rows:
            out[key] = {"count": len(rows),
                        "items": [f"{r['qualified_name'].split('::', 1)[-1] or r['rel_path']} ({r['rel_path']}:{r['line_start']})"
                                  for r in rows[:limit]]}
    refs = _edge_rows(conn, node_id, "REFERENCES", "in")
    if refs:
        out["referenced_by"] = {"count": len(refs),
                                "items": [f"{r['qualified_name'].split('::', 1)[-1]} [{ {'r': 'READS', 'w': 'WRITES'}.get(r['meta'], r['meta']) }] ({r['rel_path']}:{r['line_start']})"
                                          for r in refs[:limit]]}
    refd = _edge_rows(conn, node_id, "REFERENCES", "out")
    if refd:
        out["references"] = {"count": len(refd),
                             "items": [f"{r['qualified_name'].split('::', 1)[-1]} [{ {'r': 'READS', 'w': 'WRITES'}.get(r['meta'], r['meta']) }]" for r in refd[:limit]]}
    ann = _edge_rows(conn, node_id, "ANNOTATES", "out")
    if ann:
        out["annotates"] = {"count": len(ann), "items": [f"{r['kind']} {r['qualified_name'].split('::', 1)[-1] or r['rel_path']}" for r in ann[:limit]]}
    annby = _edge_rows(conn, node_id, "ANNOTATES", "in")
    if annby:
        out["annotated_by"] = {"count": len(annby),
                               "items": [f"{r['rel_path']}:{r['line_start']}" for r in annby[:limit]]}
    return out


def neighborhood(conn, node_id: int, depth: int = 1, kinds: tuple[str, ...] | None = None,
                 max_nodes: int = 150) -> dict:
    """Nodes and edges within `depth` hops, both directions, for the web graph.
    `truncated` is True when max_nodes stopped the expansion."""
    seen: dict[int, dict] = {}
    edges: dict[int, dict] = {}
    frontier = [node_id]
    root = get_node(conn, node_id)
    if root is None:
        return {"nodes": [], "edges": [], "truncated": False}
    root.pop("source", None)
    seen[node_id] = root
    truncated = False
    for _ in range(max(1, depth)):
        nxt: list[int] = []
        for nid in frontier:
            q = ("SELECT e.id AS eid, e.source_id, e.target_id, e.kind, e.meta FROM edges e "
                 "WHERE (e.source_id = ? OR e.target_id = ?)")
            params: list = [nid, nid]
            if kinds:
                q += " AND e.kind IN (%s)" % ",".join("?" * len(kinds))
                params += list(kinds)
            for e in conn.execute(q, params).fetchall():
                other = e["target_id"] if e["source_id"] == nid else e["source_id"]
                if other not in seen:
                    if len(seen) >= max_nodes:
                        truncated = True
                        continue
                    node = get_node(conn, other)
                    if node is None:
                        continue
                    node.pop("source", None)
                    seen[other] = node
                    nxt.append(other)
                edges[e["eid"]] = {"id": e["eid"], "from": e["source_id"], "to": e["target_id"],
                                   "kind": e["kind"], "meta": e["meta"]}
        frontier = nxt
    live = {i for i in seen}
    return {"nodes": list(seen.values()),
            "edges": [e for e in edges.values() if e["from"] in live and e["to"] in live],
            "truncated": truncated, "root": node_id}


def merged_neighborhood(conn, root_id: int, depth: int = 1, expand_ids: list[int] | None = None,
                        kinds: tuple[str, ...] | None = None, max_nodes: int = 150) -> dict:
    """Build a neighborhood around `root_id` and merge 1-hop expansions for `expand_ids` in place."""
    base = neighborhood(conn, root_id, depth=depth, kinds=kinds, max_nodes=max_nodes)
    seen = {n["id"]: n for n in base["nodes"]}
    edges = {e["id"]: e for e in base["edges"]}
    truncated = bool(base["truncated"])
    expanded: list[int] = []
    for eid in (expand_ids or []):
        if eid in expanded or eid == root_id:
            continue
        sub = neighborhood(conn, eid, depth=1, kinds=kinds, max_nodes=max_nodes)
        if not sub["nodes"]:
            continue
        expanded.append(eid)
        for n in sub["nodes"]:
            if n["id"] not in seen:
                if len(seen) >= max_nodes:
                    truncated = True
                    continue
                seen[n["id"]] = n
        for e in sub["edges"]:
            if e["from"] in seen and e["to"] in seen:
                edges[e["id"]] = e
    return {"nodes": list(seen.values()), "edges": list(edges.values()),
            "truncated": truncated, "root": root_id, "expanded": expanded}


def outline(conn, rel_path: str) -> dict | None:
    f = conn.execute("SELECT id, rel_path, language, lines FROM files WHERE rel_path = ?", (rel_path,)).fetchone()
    if f is None:
        return None
    rows = conn.execute(
        "SELECT id, kind, name, qualified_name, signature, line_start, line_end, parent_id, "
        "substr(docstring, 1, 0) AS _x, docstring FROM nodes WHERE file_id = ? AND kind != 'file' "
        "ORDER BY line_start, id", (f["id"],)).fetchall()
    items = []
    for r in rows:
        first = (r["docstring"] or "").strip().splitlines()
        items.append({"id": r["id"], "kind": r["kind"], "name": r["name"],
                      "qualified_name": r["qualified_name"].split("::", 1)[-1],
                      "signature": r["signature"], "lines": [r["line_start"], r["line_end"]],
                      "summary": first[0] if first else ""})
    return {"file": f["rel_path"], "language": f["language"], "lines": f["lines"], "items": items}


def overview(conn: sqlite3.Connection, scope: str = "runtime") -> dict:
    """Build a file-level codebase graph for cold-start overview in the Web UI.

    Scopes:
      - 'runtime': Python/TS/JS/HTML files outside tests/ and tools/ (IMPORTS + cross-file CALLS >= 6)
      - 'code': all Python/TS/JS/HTML files including tools/ and tests/ (IMPORTS + cross-file CALLS >= 10)
      - 'docs': Markdown/text files outside dev_tasks/ (LINKS_TO rolled up to file level, or section tree for prose folders)
    """
    if scope not in ("runtime", "code", "docs"):
        scope = "runtime"
    if scope in ("runtime", "code"):
        extra = "AND f.rel_path NOT LIKE 'tests/%' AND f.rel_path NOT LIKE 'tools/%' " if scope == "runtime" else ""
        rows = conn.execute(
            f"SELECT {NODE_COLS}, n.file_id {NODE_FROM} WHERE n.kind = 'file' "
            f"AND f.language IN ('python', 'typescript', 'javascript', 'html') {extra}"
            "ORDER BY f.rel_path").fetchall()
        doc_count = conn.execute(
            "SELECT COUNT(*) FROM files WHERE language IN ('markdown', 'text', 'json')").fetchone()[0]
        if not rows or (len(rows) <= 5 and doc_count > len(rows) * 2):
            # Prose-primary folder (e.g. a story/notes repo with .md/.json and a few helper scripts): fall back to docs
            scope = "docs"
    if scope == "docs":
        rows = conn.execute(
            f"SELECT {NODE_COLS}, n.file_id {NODE_FROM} WHERE n.kind = 'file' "
            "AND f.language IN ('markdown', 'text', 'json') "
            "AND f.rel_path NOT LIKE 'docs/virtualWorld/dev_tasks/%' "
            "AND f.rel_path NOT LIKE 'data/library/items/%' "
            "AND f.rel_path NOT LIKE 'data/library/tags/%' "
            "AND f.rel_path NOT LIKE 'data/library/biomes/%' "
            "ORDER BY f.rel_path").fetchall()

    file_to_node = {r["file_id"]: r["id"] for r in rows}
    node_set = set(file_to_node.values())
    nodes = [_brief(r) for r in rows]
    edges: list[dict] = []
    seen_pairs: set[tuple[int, int, str]] = set()

    if scope in ("runtime", "code"):
        for r in conn.execute("SELECT id, source_id, target_id, kind, meta FROM edges WHERE kind = 'IMPORTS'"):
            s, t = r["source_id"], r["target_id"]
            if s in node_set and t in node_set and (s, t, "IMPORTS") not in seen_pairs:
                seen_pairs.add((s, t, "IMPORTS"))
                edges.append({"id": r["id"], "from": s, "to": t, "kind": "IMPORTS", "meta": r["meta"]})
        min_calls = 6 if scope == "runtime" else 10
        call_rows = conn.execute(
            "SELECT n1.file_id AS f1, n2.file_id AS f2, count(*) AS cnt "
            "FROM edges e JOIN nodes n1 ON n1.id = e.source_id JOIN nodes n2 ON n2.id = e.target_id "
            "WHERE e.kind = 'CALLS' AND n1.file_id != n2.file_id "
            "GROUP BY n1.file_id, n2.file_id HAVING count(*) >= ?", (min_calls,)).fetchall()
        for idx, r in enumerate(call_rows, 1):
            s = file_to_node.get(r["f1"])
            t = file_to_node.get(r["f2"])
            if s and t and (s, t, "CALLS") not in seen_pairs:
                seen_pairs.add((s, t, "CALLS"))
                edges.append({"id": f"ov_call_{idx}", "from": s, "to": t,
                              "kind": "CALLS", "meta": f"{r['cnt']} calls"})
    else:
        link_rows = conn.execute(
            "SELECT n1.file_id AS f1, n2.file_id AS f2, count(*) AS cnt "
            "FROM edges e JOIN nodes n1 ON n1.id = e.source_id JOIN nodes n2 ON n2.id = e.target_id "
            "WHERE e.kind = 'LINKS_TO' AND n1.file_id != n2.file_id "
            "GROUP BY n1.file_id, n2.file_id").fetchall()
        for idx, r in enumerate(link_rows, 1):
            s = file_to_node.get(r["f1"])
            t = file_to_node.get(r["f2"])
            if s and t and (s, t, "LINKS_TO") not in seen_pairs:
                seen_pairs.add((s, t, "LINKS_TO"))
                edges.append({"id": f"ov_link_{idx}", "from": s, "to": t,
                              "kind": "LINKS_TO", "meta": f"{r['cnt']} links"})
        # When a prose/story folder has few or no cross-file links, include top-level sections
        if len(edges) < 3 and len(nodes) <= 200:
            sec_rows = conn.execute(
                f"SELECT {NODE_COLS}, n.file_id {NODE_FROM} WHERE n.kind = 'section' "
                "ORDER BY f.rel_path, n.line_start LIMIT 200").fetchall()
            for sr in sec_rows:
                fnode = file_to_node.get(sr["file_id"])
                if fnode and sr["id"] not in node_set:
                    node_set.add(sr["id"])
                    nodes.append(_brief(sr))
                    edges.append({"id": f"ov_def_{sr['id']}", "from": fnode, "to": sr["id"],
                                  "kind": "DEFINES", "meta": ""})

    connected_ids = {e["from"] for e in edges} | {e["to"] for e in edges}
    if connected_ids and (scope != "docs" or len(nodes) > 200):
        nodes = [n for n in nodes if n["id"] in connected_ids]

    return {"scope": scope, "nodes": nodes, "edges": edges, "truncated": False}

