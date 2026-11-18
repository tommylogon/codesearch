"""Graph queries over the nodes/edges tables: symbol lookup, callers, callees,
references, neighborhoods for the web graph, and file outlines."""
from __future__ import annotations

import sqlite3

NODE_COLS = ("n.id, n.kind, n.name, n.qualified_name, n.signature, n.docstring, n.line_start, "
             "n.line_end, f.rel_path, f.language")
NODE_FROM = "FROM nodes n JOIN files f ON f.id = n.file_id"

SYMBOL_KINDS = ("function", "method", "class", "variable", "section", "record", "file")


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
        "ORDER BY (n.qualified_name = ?) DESC, (n.name = ?) DESC, "
        "(n.kind IN ('function', 'method', 'class', 'variable')) DESC, n.kind, f.rel_path LIMIT ?",
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
    """Who touches this variable, section, or file, with read/write or link mode."""
    out = []
    for r in _edge_rows(conn, node_id, "REFERENCES", "in"):
        d = _brief(r)
        d["mode"] = {"r": "READS", "w": "WRITES"}.get(r["meta"], r["meta"])
        out.append(d)
    for r in _edge_rows(conn, node_id, "LINKS_TO", "in"):
        d = _brief(r)
        d["mode"] = "LINKS_TO"
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
    parent_row = conn.execute(
        "SELECT p.kind, p.qualified_name, f.rel_path, p.line_start "
        "FROM nodes c JOIN nodes p ON p.id = c.parent_id JOIN files f ON f.id = p.file_id "
        "WHERE c.id = ? AND p.kind != 'file'",
        (node_id,),
    ).fetchone()
    if parent_row is not None:
        out["defined_in"] = {
            "count": 1,
            "items": [f"{parent_row['qualified_name'].split('::', 1)[-1]} ({parent_row['rel_path']}:{parent_row['line_start']})"],
        }
    child_rows = conn.execute(
        "SELECT n.kind, n.qualified_name, f.rel_path, n.line_start "
        "FROM nodes n JOIN files f ON f.id = n.file_id "
        "WHERE n.parent_id = ? ORDER BY n.line_start, n.id",
        (node_id,),
    ).fetchall()
    if child_rows:
        out["contains"] = {
            "count": len(child_rows),
            "items": [
                f"{r['qualified_name'].split('::', 1)[-1]} ({r['rel_path']}:{r['line_start']})"
                for r in child_rows[:limit]
            ],
        }
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
            "SELECT COUNT(*) FROM files WHERE language IN ('markdown', 'text', 'json', 'pdf')").fetchone()[0]
        if not rows or (len(rows) <= 5 and doc_count > len(rows) * 2):
            # Prose-primary folder (e.g. a story/notes repo with .md/.json and a few helper scripts): fall back to docs
            scope = "docs"
    if scope == "docs":
        rows = conn.execute(
            f"SELECT {NODE_COLS}, n.file_id {NODE_FROM} WHERE n.kind = 'file' "
            "AND f.language IN ('markdown', 'text', 'json', 'pdf') "
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
                "AND n.parent_id IS NULL "
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


def resolve_inspect_files(conn: sqlite3.Connection, path_arg: str | None) -> list:
    """Return file rows matching `path_arg`, or all indexed files when omitted."""
    cols = "id, rel_path, language, hash, mtime, lines"
    if not path_arg:
        return conn.execute(f"SELECT {cols} FROM files ORDER BY rel_path").fetchall()
    needle = path_arg.replace("\\", "/").lstrip("./")
    exact = conn.execute(f"SELECT {cols} FROM files WHERE rel_path = ?", (needle,)).fetchall()
    if exact:
        return exact
    return conn.execute(
        f"SELECT {cols} FROM files WHERE rel_path LIKE ? OR rel_path LIKE ? ORDER BY rel_path",
        (f"%/{needle}", f"%{needle}"),
    ).fetchall()


def inspect_file_payload(
    conn: sqlite3.Connection,
    file_row: sqlite3.Row,
    limit: int | None = None,
    offset: int = 0,
) -> dict:
    """Collect stored nodes (optionally paged), vector parts, FTS entries, edges, and refs for one file."""
    from . import embedder

    fid = file_row["id"]
    total_nodes = conn.execute("SELECT COUNT(*) FROM nodes WHERE file_id = ?", (fid,)).fetchone()[0]
    node_sql = (
        "SELECT n.*, f.rel_path FROM nodes n JOIN files f ON f.id = n.file_id "
        "WHERE n.file_id = ? ORDER BY n.line_start, n.id"
    )
    node_params: tuple = (fid,)
    if limit is not None:
        node_sql += " LIMIT ? OFFSET ?"
        node_params = (fid, max(1, int(limit)), max(0, int(offset)))
    nodes = conn.execute(node_sql, node_params).fetchall()
    node_ids = [n["id"] for n in nodes]
    fts_words: dict[int, str] = {}
    parts_by_node: dict[int, dict[str, dict]] = {}
    if node_ids:
        for i in range(0, len(node_ids), 800):
            batch = node_ids[i:i + 800]
            ph = ",".join("?" * len(batch))
            for r in conn.execute(f"SELECT rowid, words FROM nodes_fts WHERE rowid IN ({ph})", batch):
                fts_words[r["rowid"]] = r["words"]
            part_rows = conn.execute(
                f"SELECT p.node_id, p.part, p.hash, c.model, c.dim FROM node_parts p "
                f"LEFT JOIN vec_cache c ON c.part = p.part AND c.hash = p.hash "
                f"WHERE p.node_id IN ({ph}) ORDER BY p.part, c.model",
                batch,
            ).fetchall()
            for r in part_rows:
                n_parts = parts_by_node.setdefault(r["node_id"], {})
                entry = n_parts.setdefault(r["part"], {"part": r["part"], "hash": r["hash"], "vectors": []})
                if r["model"]:
                    entry["vectors"].append({"model": r["model"], "dim": r["dim"]})

    node_items = []
    for n in nodes:
        texts = embedder.part_texts(n)
        stored_parts = parts_by_node.get(n["id"], {})
        part_list = []
        for pname in dict.fromkeys(list(stored_parts.keys()) + list(texts.keys())):
            info = stored_parts.get(pname, {"part": pname, "hash": None, "vectors": []})
            part_list.append({
                "part": pname,
                "hash": info["hash"],
                "embedded": bool(info["vectors"]),
                "vectors": info["vectors"],
                "text": texts.get(pname, ""),
            })
        node_items.append({
            "id": n["id"],
            "parent_id": n["parent_id"],
            "kind": n["kind"],
            "name": n["name"],
            "qualified_name": n["qualified_name"],
            "lines": [n["line_start"], n["line_end"]],
            "signature": n["signature"],
            "docstring": n["docstring"],
            "source": n["source"],
            "fts_words": fts_words.get(n["id"], ""),
            "parts": part_list,
        })

    edge_sql = (
        "SELECT e.id, e.kind, e.source_id, sn.qualified_name AS source_qname, "
        "e.target_id, tn.qualified_name AS target_qname, e.meta "
        "FROM edges e JOIN nodes sn ON sn.id = e.source_id JOIN nodes tn ON tn.id = e.target_id "
        "WHERE sn.file_id = ? OR tn.file_id = ? ORDER BY e.id"
    )
    edge_params: tuple = (fid, fid)
    ref_sql = (
        "SELECT r.source_id, sn.qualified_name AS source_qname, r.kind, r.target, r.qualifier, r.meta "
        "FROM refs r JOIN nodes sn ON sn.id = r.source_id "
        "WHERE sn.file_id = ? ORDER BY r.id"
    )
    ref_params: tuple = (fid,)
    if limit is not None:
        rel_cap = max(1000, int(limit) * 4)
        edge_sql += " LIMIT ?"
        edge_params = (fid, fid, rel_cap)
        ref_sql += " LIMIT ?"
        ref_params = (fid, rel_cap)
    edge_rows = conn.execute(edge_sql, edge_params).fetchall()
    ref_rows = conn.execute(ref_sql, ref_params).fetchall()
    return {
        "file": {
            "id": fid,
            "rel_path": file_row["rel_path"],
            "language": file_row["language"],
            "lines": file_row["lines"],
            "hash": file_row["hash"],
            "node_count": total_nodes,
        },
        "nodes": node_items,
        "edges": [dict(r) for r in edge_rows],
        "refs": [dict(r) for r in ref_rows],
        "total_nodes": total_nodes,
        "offset": max(0, int(offset)),
        "truncated": (max(0, int(offset)) + len(node_items)) < total_nodes,
    }


def file_graph(conn: sqlite3.Connection, path_arg: str, max_nodes: int = 240) -> dict:
    """Build a structural + relational breakdown graph for one file (`DEFINES`, `CONNECTS_TO`, etc.)."""
    files = resolve_inspect_files(conn, path_arg)
    if not files:
        return {"nodes": [], "edges": [], "truncated": False, "file": None}
    file_row = files[0]
    fid = file_row["id"]
    total = conn.execute("SELECT COUNT(*) FROM nodes WHERE file_id = ?", (fid,)).fetchone()[0]
    # Keep structural nodes ahead of split sentence leaves so large files keep their spine under caps.
    rows = conn.execute(
        f"SELECT {NODE_COLS}, n.parent_id {NODE_FROM} WHERE n.file_id = ? "
        "ORDER BY CASE WHEN n.kind = 'file' THEN 0 "
        "WHEN n.kind IN ('record', 'class', 'function', 'method') THEN 1 "
        "WHEN n.kind = 'section' AND n.name NOT LIKE '% sentence %' THEN 2 "
        "WHEN n.kind IN ('variable', 'comment', 'block', 'element') THEN 3 "
        "WHEN n.kind = 'tag' THEN 4 ELSE 5 END, n.line_start, n.id LIMIT ?",
        (fid, max_nodes),
    ).fetchall()
    nodes_by_id: dict[int, dict] = {}
    root_id: int | None = None
    for r in rows:
        b = _brief(r)
        b["parent_id"] = r["parent_id"]
        nodes_by_id[b["id"]] = b
        if b["kind"] == "file" and root_id is None:
            root_id = b["id"]
    truncated = total > len(nodes_by_id)
    local_ids = list(nodes_by_id)
    edges_by_id: dict[int, dict] = {}
    if local_ids:
        ph = ",".join("?" * len(local_ids))
        edge_rows = conn.execute(
            f"SELECT id, source_id, target_id, kind, meta FROM edges "
            f"WHERE source_id IN ({ph}) OR target_id IN ({ph}) ORDER BY id",
            local_ids + local_ids,
        ).fetchall()
        for e in edge_rows:
            for endpoint in (e["source_id"], e["target_id"]):
                if endpoint not in nodes_by_id:
                    if len(nodes_by_id) >= max_nodes + 40:
                        truncated = True
                        continue
                    ext = get_node(conn, endpoint)
                    if ext:
                        ext.pop("source", None)
                        nodes_by_id[endpoint] = ext
            if e["source_id"] in nodes_by_id and e["target_id"] in nodes_by_id:
                edges_by_id[e["id"]] = {
                    "id": e["id"], "from": e["source_id"], "to": e["target_id"],
                    "kind": e["kind"], "meta": e["meta"],
                }
    return {
        "file": file_row["rel_path"],
        "root": root_id,
        "nodes": list(nodes_by_id.values()),
        "edges": list(edges_by_id.values()),
        "truncated": truncated,
    }

