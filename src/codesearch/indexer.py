"""Directory scanner and incremental indexer.

Pipeline: scan -> hash -> parse changed files -> insert nodes, DEFINES/ANNOTATES edges,
FTS rows and raw refs -> resolve every ref into graph edges (CALLS, REFERENCES, IMPORTS,
INHERITS, LINKS_TO). Unchanged files (same content hash) are never re-parsed.
"""
from __future__ import annotations

import hashlib
import os
import posixpath
import re
import sqlite3
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from pathlib import Path

from . import config, db, embedder
from .parsers import csv_parser, html_parser, js_bridge, json_parser, md_parser, pdf_parser, py_parser

RESOLVED_KINDS = ("CALLS", "REFERENCES", "IMPORTS", "INHERITS", "LINKS_TO", "CONNECTS_TO")
JS_BATCH = 150
# Hashing a photo library is disk-latency bound, not CPU bound. Threads overlap the I/O.
READ_WORKERS = 8
BUILTIN_ATTR_CALLS = {
    "get", "set", "has", "add", "delete", "clear", "keys", "values", "entries", "items",
    "push", "pop", "shift", "unshift", "slice", "splice", "map", "filter", "forEach",
    "reduce", "find", "some", "every", "includes", "indexOf", "join", "split", "trim",
    "replace", "startswith", "endswith", "lower", "upper", "strip", "lstrip", "rstrip",
    "append", "extend", "update", "copy", "toString", "valueOf", "then", "catch", "finally",
    "sort", "reverse", "popitem", "setdefault", "discard", "remove", "insert", "format",
}


def _lang_family(lang: str) -> str:
    if lang in ("typescript", "javascript", "html"):
        return "js"
    return lang


# --------------------------------------------------------------------------- scan
def scan(root: Path) -> tuple[list[dict], dict]:
    """Walk the repo. Returns (files, skipped) where skipped counts why files were left out."""
    files: list[dict] = []
    skipped = defaultdict(int)
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        dirnames[:] = [d for d in dirnames if d not in config.IGNORE_DIRS]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            lang = config.LANGUAGES.get(ext) or config.IMAGE_LANGUAGES.get(ext)
            if lang is None:
                continue
            rel = fn if rel_dir == "." else f"{rel_dir}/{fn}"
            if rel.startswith(config.IGNORE_PREFIXES) or fn in config.IGNORE_FILES:
                skipped["vendored"] += 1
                continue
            if any(fnmatch(fn, pat) for pat in config.IGNORE_FILE_PATTERNS):
                skipped["generated_index"] += 1
                continue
            if fn.endswith(".d.ts"):
                skipped["declaration"] += 1
                continue
            full = Path(dirpath) / fn
            try:
                st = full.stat()
            except OSError:
                continue
            max_bytes = (config.MAX_JSON_BYTES if lang == "json"
                         else config.IMAGE_MAX_BYTES if lang == "image"
                         else config.MAX_FILE_BYTES)
            if max_bytes is not None and st.st_size > max_bytes:
                skipped["too_large"] += 1
                continue
            if (config.SKIP_JS_WITH_TS_SIBLING and ext == ".js"
                    and full.with_suffix(".ts").exists()):
                skipped["compiled_js"] += 1
                continue
            files.append({"rel": rel, "path": full, "language": lang,
                          "mtime": st.st_mtime, "size": st.st_size})
    return files, dict(skipped)


def _read(path: Path) -> str | None:
    with path.open("rb") as fh:
        if b"\x00" in fh.read(4096):
            return None
    return path.read_text(encoding="utf-8", errors="replace")


def _hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _hash_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _image_dims(path: Path) -> str:
    """Pixel size, e.g. 1214x1295. It goes in the file node's docstring, which means it is
    also keyword-searchable and part of the filename embedding."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return f"{im.width}x{im.height}"
    except Exception:
        return ""


# --------------------------------------------------------------------------- parse
def _parse_files(changed: list[dict], progress=lambda msg: None) -> tuple[dict[str, dict], list[str]]:
    """Return {rel: parsed} and a list of 'rel: reason' parse problems."""
    parsed: dict[str, dict] = {}
    problems: list[str] = []
    js_jobs: list[dict] = []
    html_scripts: dict[str, list[dict]] = {}
    t0 = time.time()
    for i, f in enumerate(changed, 1):
        text, lang, rel = f["text"], f["language"], f["rel"]
        if lang == "python":
            parsed[rel] = py_parser.parse(text, rel)
        elif lang == "markdown":
            parsed[rel] = md_parser.parse(text, rel)
        elif lang == "text":
            parsed[rel] = md_parser.parse_text(text)
        elif lang == "json":
            parsed[rel] = json_parser.parse(text, rel)
        elif lang == "csv":
            parsed[rel] = csv_parser.parse(text, rel)
        elif lang == "html":
            res = html_parser.parse(text, rel)
            parsed[rel] = res
            for i2, s in enumerate(res.get("scripts", [])):
                js_jobs.append({"id": f"{rel}#{i2}", "text": s["text"], "ts": False, "offset": s["offset"]})
            html_scripts[rel] = res.get("scripts", [])
        elif lang in ("typescript", "javascript"):
            js_jobs.append({"id": rel, "text": text, "ts": lang == "typescript", "offset": 0})
        elif lang == "pdf":
            res = pdf_parser.parse(f["path"], rel, progress=progress)
            f["lines"] = res.get("page_count", 0)
            parsed[rel] = res
        elif lang == "image":
            parsed[rel] = {"file_doc": "", "nodes": [], "refs": []}
        if i % 500 == 0:
            rate = i / max(time.time() - t0, 1e-6)
            progress(f"  parsing: {i}/{len(changed)} files ({rate:.0f}/s)")
    results: dict[str, dict] = {}
    for i in range(0, len(js_jobs), JS_BATCH):
        results.update(js_bridge.parse_batch(js_jobs[i:i + JS_BATCH]))
    for f in changed:
        rel, lang = f["rel"], f["language"]
        if lang in ("typescript", "javascript"):
            res = results.get(rel, {"error": "no result"})
            if "error" in res:
                problems.append(f"{rel}: {res['error']}")
                parsed[rel] = {"file_doc": "", "nodes": [], "refs": []}
            else:
                parsed[rel] = res
        elif lang == "html":
            base = parsed[rel]
            for i, _ in enumerate(html_scripts.get(rel, [])):
                res = results.get(f"{rel}#{i}", {"error": "no result"})
                if "error" in res:
                    problems.append(f"{rel} <script> {i}: {res['error']}")
                    continue
                base["nodes"].extend(res["nodes"])
                base["refs"].extend(res["refs"])
    for rel, res in parsed.items():
        for msg in res.get("problems", []):
            problems.append(f"{rel}: {msg}")
    return parsed, problems


# --------------------------------------------------------------------------- insert
def _annotation_target(c: dict, others: list[dict]) -> int | None:
    """The node a comment describes: the one starting right after it, else the
    innermost node containing it. Returns an index into `others`."""
    best, best_key = None, None
    for i, n in enumerate(others):
        gap = n["line_start"] - c["line_end"]
        if 1 <= gap <= 2:
            key = (gap, -(n["line_end"] - n["line_start"]))
            if best_key is None or key < best_key:
                best, best_key = i, key
    if best is not None:
        return best
    best_span = None
    for i, n in enumerate(others):
        if n["line_start"] <= c["line_start"] and n["line_end"] >= c["line_end"]:
            span = n["line_end"] - n["line_start"]
            if best_span is None or span < best_span:
                best, best_span = i, span
    return best


_INSERT_CHUNK = 50_000


def _insert_file(conn: sqlite3.Connection, f: dict, parsed: dict, progress=lambda msg: None) -> int:
    rel = f["rel"]
    is_image = f["language"] == "image"
    is_pdf = f["language"] == "pdf"
    lines = 0 if is_image else int(f.get("lines", 0)) if is_pdf else f["text"].count("\n") + 1
    cur = conn.execute("INSERT INTO files(rel_path, language, hash, mtime, lines) VALUES (?,?,?,?,?)",
                       (rel, f["language"], f["hash"], f["mtime"], lines))
    file_id = cur.lastrowid

    next_id = (conn.execute("SELECT COALESCE(MAX(id), 0) FROM nodes").fetchone()[0] or 0) + 1
    file_node = next_id
    next_id += 1

    doc = parsed.get("file_doc", "") or (_image_dims(f["path"]) if is_image else "")
    conn.execute(
        "INSERT INTO nodes(id, file_id, kind, name, qualified_name, parent_id, signature, docstring, source,"
        " line_start, line_end) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (file_node, file_id, "file", posixpath.basename(rel), rel, None, f["language"], doc, "", 1, max(lines, 1)),
    )
    file_row_obj = {"kind": "file", "name": posixpath.basename(rel), "qualified_name": rel,
                    "signature": f["language"], "docstring": doc, "source": "", "rel_path": rel}
    file_parts = [(file_node, part, embedder.text_hash(text))
                  for part, text in embedder.part_texts(file_row_obj).items()]
    if file_parts:
        conn.executemany("INSERT OR REPLACE INTO node_parts(node_id, part, hash) VALUES (?,?,?)", file_parts)
    db.insert_fts(conn, [db.fts_row(file_node, posixpath.basename(rel), "", f["language"], doc, rel, "")])

    nodes_raw = parsed["nodes"]
    qnames: list[str] = []
    node_ids: list[int] = []
    ids: dict[str, int] = {}
    has_comments = False
    for n in nodes_raw:
        qn = n["qname"]
        if qn in ids:
            qn = f"{qn}@{n['line_start']}"
        nid = next_id
        next_id += 1
        ids[qn] = nid
        qnames.append(qn)
        node_ids.append(nid)
        if n["kind"] == "comment":
            has_comments = True

    total = len(nodes_raw)
    node_sql = (
        "INSERT INTO nodes(id, file_id, kind, name, qualified_name, parent_id, signature, docstring, source,"
        " line_start, line_end) VALUES (?,?,?,?,?,?,?,?,?,?,?)"
    )
    edge_sql = "INSERT OR IGNORE INTO edges(source_id, target_id, kind, meta) VALUES (?,?,?,?)"
    part_sql = "INSERT OR REPLACE INTO node_parts(node_id, part, hash) VALUES (?,?,?)"

    for start in range(0, total, _INSERT_CHUNK):
        end = min(start + _INSERT_CHUNK, total)
        node_rows: list[tuple] = []
        edge_rows: list[tuple] = []
        part_rows: list[tuple] = []
        fts_rows: list[tuple] = []
        for idx in range(start, end):
            n = nodes_raw[idx]
            nid = node_ids[idx]
            qn = qnames[idx]
            full_qname = f"{rel}::{qn}" if qn else rel
            parent = ids.get(n["parent"]) if n["parent"] else None
            node_rows.append((nid, file_id, n["kind"], n["name"], full_qname, parent,
                              n["signature"], n["docstring"], n["source"], n["line_start"], n["line_end"]))
            fts_rows.append(db.fts_row(nid, n["name"], qn, n["signature"], n["docstring"], rel, n["source"]))
            row_obj = {"kind": n["kind"], "name": n["name"], "qualified_name": full_qname,
                       "signature": n["signature"], "docstring": n["docstring"],
                       "source": n["source"], "rel_path": rel}
            for part, text in embedder.part_texts(row_obj).items():
                part_rows.append((nid, part, embedder.text_hash(text)))
            if n["kind"] != "comment":
                edge_rows.append((parent if parent is not None else file_node, nid, "DEFINES", ""))
        conn.executemany(node_sql, node_rows)
        if edge_rows:
            conn.executemany(edge_sql, edge_rows)
        if part_rows:
            conn.executemany(part_sql, part_rows)
        db.insert_fts(conn, fts_rows)
        if total >= 100_000:
            progress(f"  storing {rel}: {end}/{total} nodes")

    if has_comments:
        others = [(nodes_raw[i], node_ids[i]) for i in range(total) if nodes_raw[i]["kind"] != "comment"]
        other_dicts = [o[0] for o in others]
        ann_edges: list[tuple] = []
        for i, n in enumerate(nodes_raw):
            if n["kind"] != "comment":
                continue
            idx = _annotation_target(n, other_dicts)
            target = others[idx][1] if idx is not None else file_node
            ann_edges.append((node_ids[i], target, "ANNOTATES", ""))
        if ann_edges:
            conn.executemany(edge_sql, ann_edges)

    ref_rows = []
    for r in parsed["refs"]:
        src = file_node if r["source"] == "" else ids.get(r["source"])
        if src is None:
            continue
        ref_rows.append((src, r["kind"], r["target"], r.get("qualifier", ""), r.get("meta", "")))
    if ref_rows:
        conn.executemany("INSERT INTO refs(source_id, kind, target, qualifier, meta) VALUES (?,?,?,?,?)", ref_rows)
    return file_id


# --------------------------------------------------------------------------- resolve
def resolve(conn: sqlite3.Connection) -> dict:
    """Rebuild every resolved edge from the refs table."""
    conn.execute("DELETE FROM edges WHERE kind IN (%s)" % ",".join("?" * len(RESOLVED_KINDS)),
                 RESOLVED_KINDS)
    refs = conn.execute("SELECT source_id, kind, target, qualifier, meta FROM refs ORDER BY id").fetchall()
    if not refs:
        return {}

    rel_by_file: dict[int, str] = {}
    fam_by_file: dict[int, str] = {}
    for r in conn.execute("SELECT id, rel_path, language FROM files"):
        rel_by_file[r["id"]] = r["rel_path"]
        fam_by_file[r["id"]] = _lang_family(r["language"])
    file_by_rel = {v: k for k, v in rel_by_file.items()}
    file_node = {r["file_id"]: r["id"] for r in conn.execute("SELECT id, file_id FROM nodes WHERE kind='file'")}

    def _by_ids(cols: str, where_col: str, id_list: list[int], extra: str = ""):
        rows = []
        for off in range(0, len(id_list), 800):
            chunk = id_list[off:off + 800]
            ph = ",".join("?" * len(chunk))
            rows.extend(conn.execute(f"SELECT {cols} FROM nodes WHERE {where_col} IN ({ph}) {extra}", chunk))
        return rows

    node_file: dict[int, int] = {}
    node_parent: dict[int, int | None] = {}
    node_name: dict[int, str] = {}
    source_ids = sorted({r["source_id"] for r in refs})
    for r in _by_ids("id, file_id, name, parent_id", "id", source_ids):
        node_file[r["id"]] = r["file_id"]
        node_parent[r["id"]] = r["parent_id"]
        node_name[r["id"]] = r["name"]

    ref_kinds = {r["kind"] for r in refs}
    by_name: dict[str, list[tuple]] = defaultdict(list)
    by_qname: dict[str, int] = {}
    if ref_kinds & {"CALLS", "REFERENCES", "INHERITS"}:
        missing_parents: set[int] = set()
        for r in conn.execute(
            "SELECT id, file_id, kind, name, qualified_name, parent_id FROM nodes "
            "WHERE kind IN ('function', 'method', 'class', 'variable')"
        ):
            node_file[r["id"]] = r["file_id"]
            node_parent[r["id"]] = r["parent_id"]
            node_name[r["id"]] = r["name"]
            by_qname[r["qualified_name"]] = r["id"]
            by_name[r["name"]].append((r["id"], r["file_id"], r["kind"], r["parent_id"]))
            if r["parent_id"] is not None:
                missing_parents.add(r["parent_id"])
        need_parents = sorted(pid for pid in missing_parents if pid not in node_name)
        if need_parents:
            for r in _by_ids("id, name", "id", need_parents):
                node_name[r["id"]] = r["name"]
        class_ref_files = sorted({
            node_file[r["source_id"]] for r in refs
            if r["kind"] == "REFERENCES" and r["qualifier"].startswith("class:") and r["source_id"] in node_file
        })
        if class_ref_files:
            for r in _by_ids("id, qualified_name", "file_id", class_ref_files):
                by_qname[r["qualified_name"]] = r["id"]

    # JSON records are link targets: a `target`/`way_id`/`area_from` value names a record
    # in the same file. Index them by display name and by the last segment of their path.
    record_by_file: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    if "CONNECTS_TO" in ref_kinds:
        connect_files = sorted({
            node_file[r["source_id"]] for r in refs
            if r["kind"] == "CONNECTS_TO" and r["source_id"] in node_file
        })
        for r in _by_ids("id, file_id, name, qualified_name", "file_id", connect_files, "AND kind = 'record'"):
            keys = {r["name"].strip().lower()}
            tail = r["qualified_name"].split("::", 1)[-1].rsplit("::", 1)[-1]
            keys.add(tail.strip().lower())
            for k in keys:
                if k:
                    record_by_file[r["file_id"]][k].append(r["id"])

    # Markdown, structured text, and PDF sections can be linked by heading name or anchor.
    section_by_name: dict[str, list[int]] = defaultdict(list)
    section_by_file: dict[int, list[tuple[str, int]]] = defaultdict(list)
    if any(r["kind"] == "LINKS_TO" and r["qualifier"] in ("wiki", "section") for r in refs):
        for r in conn.execute("SELECT id, file_id, name, signature FROM nodes WHERE kind = 'section'"):
            sig = (r["signature"] or "").strip().lower()
            if sig.startswith("page "):
                continue
            k = re.sub(r"\s+", " ", r["name"]).strip().lower()
            if k and k not in ("(intro)", "(text)", "table of contents"):
                section_by_name[k].append(r["id"])
                section_by_file[r["file_id"]].append((k, r["id"]))

    out: list[tuple] = []
    counts: dict[str, int] = defaultdict(int)

    def emit(src, dst, kind, meta=""):
        if src != dst:
            out.append((src, dst, kind, meta))
            counts[kind] += 1

    # ---- module resolvers
    py_mod: dict[str, int] = {}
    for rel, fid in file_by_rel.items():
        if rel.endswith(".py"):
            stem = rel[:-3]
            if stem.endswith("/__init__"):
                stem = stem[: -len("/__init__")]
            py_mod[stem] = fid
    stem_to_files: dict[str, list[int]] = defaultdict(list)
    for rel, fid in file_by_rel.items():
        stem_to_files[posixpath.splitext(posixpath.basename(rel))[0].lower()].append(fid)

    def resolve_py(source_rel: str, module: str, level: int, names: list[str]) -> list[int]:
        found: list[int] = []
        if level > 0:
            base = posixpath.dirname(source_rel)
            for _ in range(level - 1):
                base = posixpath.dirname(base)
            parts = module.split(".") if module else []
            path = posixpath.join(base, *parts) if parts else base
            path = posixpath.normpath(path) if path else ""
            cands = [path] if module else []
            cands += [posixpath.join(path, n) for n in names]
        else:
            path = module.replace(".", "/")
            cands = [path] + [f"{path}/{n}" for n in names]
        for c in cands:
            c = c.lstrip("./") if level == 0 else c
            if c in py_mod:
                found.append(py_mod[c])
            elif level == 0:
                suffix = "/" + c
                hits = [fid for stem, fid in py_mod.items() if stem.endswith(suffix)]
                if len(hits) == 1:
                    found.append(hits[0])
        return list(dict.fromkeys(found))

    def resolve_path(source_rel: str, target: str) -> int | None:
        target = target.split("?")[0].split("#")[0]
        if not target or "://" in target:
            return None
        if target.startswith("/"):
            base = target.lstrip("/")
        elif target.startswith("."):
            base = posixpath.normpath(posixpath.join(posixpath.dirname(source_rel), target))
        else:
            sibling = posixpath.normpath(posixpath.join(posixpath.dirname(source_rel), target))
            if sibling in file_by_rel:
                return file_by_rel[sibling]
            return None
        variants = [base]
        stem, ext = posixpath.splitext(base)
        if ext in (".js", ".mjs", ".cjs"):
            variants.append(stem + ".ts")
        variants += [base + e for e in (".ts", ".js", ".mjs", ".py", ".md")]
        variants += [base + "/index.ts", base + "/index.js"]
        for v in variants:
            if v in file_by_rel:
                return file_by_rel[v]
        return None

    # ---- imports first: REFERENCES resolution needs the imported-file sets
    imported: dict[int, set[int]] = defaultdict(set)
    for r in refs:
        if r["kind"] != "IMPORTS":
            continue
        src_file = node_file[r["source_id"]]
        src_rel = rel_by_file[src_file]
        if src_rel.endswith(".py"):
            names = [n for n in r["meta"].split(",") if n and n != "*"]
            targets = resolve_py(src_rel, r["target"], int(r["qualifier"] or 0), names)
        else:
            t = resolve_path(src_rel, r["target"])
            targets = [t] if t is not None else []
        for t in targets:
            if t != src_file:
                imported[src_file].add(t)
                emit(file_node[src_file], file_node[t], "IMPORTS", r["meta"] if len(r["meta"]) < 200 else "")

    def pick(cands, src_id, src_file, src_parent, qualifier, allow_methods, target_name):
        src_fam = fam_by_file.get(src_file)
        pool = [c for c in cands
                if (allow_methods or c[2] != "method") and fam_by_file.get(c[1]) == src_fam]
        if qualifier == "self" and src_parent is not None:
            same = [c for c in pool if c[3] == src_parent]
            if same:
                return same
        if qualifier.startswith("recv:"):
            recv = qualifier[5:].lower()
            by_recv = [c for c in pool if c[3] is not None and (
                node_name.get(c[3], "").lower() in (recv, recv + "impl"))]
            if by_recv:
                return by_recv
        if qualifier not in ("", "self"):
            pool = [c for c in pool if c[0] != src_id]
            if target_name in BUILTIN_ATTR_CALLS:
                return []
        same_file = [c for c in pool if c[1] == src_file]
        if same_file:
            return same_file
        in_imports = [c for c in pool if c[1] in imported.get(src_file, ())]
        if in_imports:
            return in_imports[:config.MAX_AMBIGUOUS_TARGETS]
        if len(pool) <= config.MAX_AMBIGUOUS_TARGETS:
            return pool
        return []

    for r in refs:
        kind, src = r["kind"], r["source_id"]
        if kind == "IMPORTS":
            continue
        src_file = node_file[src]
        src_fam = fam_by_file.get(src_file)
        if kind == "CALLS":
            cands = [c for c in by_name.get(r["target"], ()) if c[2] in ("function", "method", "class")]
            allow_methods = r["qualifier"] != ""
            for c in pick(cands, src, src_file, node_parent.get(src), r["qualifier"], allow_methods, r["target"]):
                emit(src, c[0], "CALLS")
        elif kind == "REFERENCES":
            if r["qualifier"].startswith("class:"):
                qn = f"{rel_by_file[src_file]}::{r['qualifier'][6:]}::{r['target']}"
                tgt = by_qname.get(qn)
                if tgt is not None:
                    emit(src, tgt, "REFERENCES", r["meta"])
            elif r["qualifier"] == "attr":
                # obj.attr on some other object: the receiver type is unknown, so link only
                # when the attribute name is distinctive (few variables carry it in the same language).
                cands = [c for c in by_name.get(r["target"], ())
                         if c[2] == "variable" and fam_by_file.get(c[1]) == src_fam]
                if 0 < len(cands) <= config.MAX_AMBIGUOUS_TARGETS:
                    for c in cands:
                        emit(src, c[0], "REFERENCES", r["meta"])
            else:
                for c in by_name.get(r["target"], ()):
                    if c[2] == "variable" and c[3] is None and (
                            c[1] == src_file or c[1] in imported.get(src_file, ())):
                        emit(src, c[0], "REFERENCES", r["meta"])
        elif kind == "INHERITS":
            cands = [c for c in by_name.get(r["target"], ())
                     if c[2] == "class" and fam_by_file.get(c[1]) == src_fam]
            same = [c for c in cands if c[1] == src_file]
            chosen = same or (cands if len(cands) == 1 else [c for c in cands if c[1] in imported.get(src_file, ())])
            for c in chosen:
                emit(src, c[0], "INHERITS")
        elif kind == "LINKS_TO":
            src_rel = rel_by_file[src_file]
            if r["qualifier"] == "wiki":
                name = r["target"].strip()
                stem = posixpath.basename(name).lower()
                hits = stem_to_files.get(stem, [])
                if "/" in name:
                    narrowed = [h for h in hits if posixpath.splitext(rel_by_file[h])[0].lower().endswith(name.lower())]
                    hits = narrowed or hits
                for h in hits[:config.MAX_AMBIGUOUS_TARGETS]:
                    emit(src, file_node[h], "LINKS_TO")

                # Also link directly to matching markdown section headings
                target_sec = (r["meta"] or name).strip().lower()
                sec_hits = section_by_name.get(target_sec, [])
                if not sec_hits and not hits:
                    sec_hits = [sid for sname, sids in section_by_name.items() if target_sec in sname for sid in sids]
                for sid in sec_hits[:config.MAX_AMBIGUOUS_TARGETS]:
                    emit(src, sid, "LINKS_TO")
            elif r["qualifier"] == "section":
                target_sec = re.sub(r"\s+", " ", r["target"]).strip().lower()
                if len(target_sec) >= 2:
                    local_secs = section_by_file.get(src_file, ())
                    hits = [nid for sname, nid in local_secs if sname == target_sec]
                    if not hits:
                        hits = [
                            nid for sname, nid in local_secs
                            if sname.startswith(target_sec + ":")
                            or sname.startswith(target_sec + " ")
                            or target_sec in sname
                        ]
                    if not hits:
                        hits = section_by_name.get(target_sec, [])
                    if not hits:
                        hits = [
                            sid for sname, sids in section_by_name.items()
                            if sname.startswith(target_sec + ":") or sname.startswith(target_sec + " ")
                            for sid in sids
                        ]
                    for sid in hits[:config.MAX_AMBIGUOUS_TARGETS]:
                        emit(src, sid, "LINKS_TO", "section")
            else:
                t = resolve_path(src_rel, r["target"]) if r["target"].startswith(("/", ".")) else None
                if t is None:
                    sibling = posixpath.normpath(posixpath.join(posixpath.dirname(src_rel), r["target"]))
                    t = file_by_rel.get(sibling)
                if t is not None:
                    emit(src, file_node[t], "LINKS_TO")
        elif kind == "CONNECTS_TO":
            key = r["target"].strip().lower()
            for c in record_by_file.get(src_file, {}).get(key, [])[:config.MAX_AMBIGUOUS_TARGETS]:
                emit(src, c, "CONNECTS_TO", r["qualifier"])

    conn.executemany("INSERT OR IGNORE INTO edges(source_id, target_id, kind, meta) VALUES (?,?,?,?)", out)
    return dict(counts)


# --------------------------------------------------------------------------- driver
def index(conn: sqlite3.Connection, root: Path | None = None, force: bool = False,
          progress=lambda msg: None) -> dict:
    root = Path(root or config.REPO_ROOT)
    t0 = time.time()
    files, skipped = scan(root)
    existing = {r["rel_path"]: r for r in conn.execute("SELECT id, rel_path, hash, mtime FROM files")}
    on_disk = {f["rel"] for f in files}

    removed = [rel for rel in existing if rel not in on_disk]
    for rel in removed:
        db.delete_file(conn, existing[rel]["id"])

    changed: list[dict] = []
    unchanged = 0

    # Only files whose mtime moved are re-read. The read is I/O bound (a photo library is
    # tens of GB), so it runs on a thread pool: disk latency overlaps instead of stacking.
    todo_read: list[dict] = []
    for f in files:
        prev = existing.get(f["rel"])
        if prev and not force and prev["mtime"] == f["mtime"]:
            unchanged += 1
        else:
            todo_read.append(f)

    def _read_one(f: dict) -> tuple[dict, str | None]:
        """Return (file, None) or (file, skip-reason)."""
        if f["language"] in ("image", "pdf"):
            try:
                raw = f["path"].read_bytes()
            except OSError:
                return f, "unreadable"
            f["text"] = ""                 # never UTF-8 decoded: images are pixels, PDFs are parsed per page
            f["hash"] = _hash_bytes(raw)
            return f, None
        text = _read(f["path"])
        if text is None:
            return f, "binary"
        f["text"] = text
        f["hash"] = _hash(text)
        return f, None

    t_read = time.time()
    read = 0
    if todo_read:
        with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
            for f, skip in pool.map(_read_one, todo_read):
                read += 1
                if skip:
                    skipped[skip] = skipped.get(skip, 0) + 1
                    continue
                prev = existing.get(f["rel"])
                if prev and not force and prev["hash"] == f["hash"]:
                    conn.execute("UPDATE files SET mtime = ? WHERE id = ?", (f["mtime"], prev["id"]))
                    unchanged += 1
                    continue
                changed.append(f)
                if read % 250 == 0:
                    rate = read / max(time.time() - t_read, 1e-6)
                    progress(f"  reading: {read}/{len(todo_read)} files hashed ({rate:.0f}/s)")
    progress(f"scan: {len(files)} files, {len(changed)} changed, {unchanged} unchanged, "
             f"{len(removed)} removed, skipped {skipped}")

    problems: list[str] = []
    if changed:
        t_parse = time.time()
        parsed, problems = _parse_files(changed, progress)
        for i, f in enumerate(changed, 1):
            prev = existing.get(f["rel"])
            if prev:
                db.delete_file(conn, prev["id"])
            _insert_file(conn, f, parsed.pop(f["rel"]), progress=progress)
            f.pop("text", None)
            if i % 250 == 0:
                rate = i / max(time.time() - t_parse, 1e-6)
                progress(f"  storing: {i}/{len(changed)} files ({rate:.0f}/s)")
        progress(f"parsed and stored {len(changed)} files")

    counts = {}
    if changed or removed or force:
        counts = resolve(conn)
        progress(f"resolved edges: {counts}")
        db.bump_vec_generation(conn)
    db.set_meta(conn, "indexed_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    conn.commit()
    return {"files": len(files), "changed": len(changed), "unchanged": unchanged,
            "removed": len(removed), "skipped": skipped, "problems": problems,
            "edges": counts, "seconds": round(time.time() - t0, 2)}
