"""Directory scanner and incremental indexer.

Pipeline: scan -> hash -> parse changed files -> insert nodes, DEFINES/ANNOTATES edges,
FTS rows and raw refs -> resolve every ref into graph edges (CALLS, REFERENCES, IMPORTS,
INHERITS, LINKS_TO). Unchanged files (same content hash) are never re-parsed.
"""
from __future__ import annotations

import hashlib
import os
import posixpath
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from . import config, db, embedder
from .parsers import html_parser, js_bridge, json_parser, md_parser, py_parser

RESOLVED_KINDS = ("CALLS", "REFERENCES", "IMPORTS", "INHERITS", "LINKS_TO")
JS_BATCH = 150
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
            lang = config.LANGUAGES.get(ext)
            if lang is None:
                continue
            rel = fn if rel_dir == "." else f"{rel_dir}/{fn}"
            if rel.startswith(config.IGNORE_PREFIXES) or fn in config.IGNORE_FILES:
                skipped["vendored"] += 1
                continue
            if fn.endswith(".d.ts"):
                skipped["declaration"] += 1
                continue
            full = Path(dirpath) / fn
            try:
                st = full.stat()
            except OSError:
                continue
            max_bytes = config.MAX_JSON_BYTES if lang == "json" else config.MAX_FILE_BYTES
            if st.st_size > max_bytes:
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
    data = path.read_bytes()
    if b"\x00" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")


def _hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


# --------------------------------------------------------------------------- parse
def _parse_files(changed: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """Return {rel: parsed} and a list of 'rel: reason' parse problems."""
    parsed: dict[str, dict] = {}
    problems: list[str] = []
    js_jobs: list[dict] = []
    html_scripts: dict[str, list[dict]] = {}
    for f in changed:
        text, lang, rel = f["text"], f["language"], f["rel"]
        if lang == "python":
            parsed[rel] = py_parser.parse(text, rel)
        elif lang == "markdown":
            parsed[rel] = md_parser.parse(text, rel)
        elif lang == "text":
            parsed[rel] = md_parser.parse_text(text)
        elif lang == "json":
            parsed[rel] = json_parser.parse(text, rel)
        elif lang == "html":
            res = html_parser.parse(text, rel)
            parsed[rel] = res
            for i, s in enumerate(res.get("scripts", [])):
                js_jobs.append({"id": f"{rel}#{i}", "text": s["text"], "ts": False, "offset": s["offset"]})
            html_scripts[rel] = res.get("scripts", [])
        elif lang in ("typescript", "javascript"):
            js_jobs.append({"id": rel, "text": text, "ts": lang == "typescript", "offset": 0})
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


def _insert_file(conn: sqlite3.Connection, f: dict, parsed: dict) -> int:
    rel = f["rel"]
    cur = conn.execute("INSERT INTO files(rel_path, language, hash, mtime, lines) VALUES (?,?,?,?,?)",
                       (rel, f["language"], f["hash"], f["mtime"], f["text"].count("\n") + 1))
    file_id = cur.lastrowid

    def add(kind, name, qname, signature, doc, source, ls, le):
        full_qname = f"{rel}::{qname}" if qname else rel
        c = conn.execute(
            "INSERT INTO nodes(file_id, kind, name, qualified_name, signature, docstring, source,"
            " line_start, line_end) VALUES (?,?,?,?,?,?,?,?,?)",
            (file_id, kind, name, full_qname, signature, doc, source, ls, le))
        nid = c.lastrowid
        fts.append(db.fts_row(nid, name, qname, signature, doc, rel, source))
        row_obj = {"kind": kind, "name": name, "qualified_name": full_qname,
                   "signature": signature, "docstring": doc, "source": source, "rel_path": rel}
        for part, text in embedder.part_texts(row_obj).items():
            part_rows.append((nid, part, embedder.text_hash(text)))
        return nid

    fts: list[tuple] = []
    part_rows: list[tuple] = []
    file_node = add("file", posixpath.basename(rel), "", f["language"], parsed.get("file_doc", ""),
                    "", 1, f["text"].count("\n") + 1)

    ids: dict[str, int] = {}
    seen: set[str] = set()
    nodes = []
    for n in parsed["nodes"]:
        n = dict(n)
        if n["qname"] in seen:
            n["qname"] = f"{n['qname']}@{n['line_start']}"
        seen.add(n["qname"])
        nodes.append(n)
    for n in nodes:
        ids[n["qname"]] = add(n["kind"], n["name"], n["qname"], n["signature"], n["docstring"],
                              n["source"], n["line_start"], n["line_end"])
    edges: list[tuple] = []
    parent_updates: list[tuple] = []
    for n in nodes:
        nid = ids[n["qname"]]
        parent = ids.get(n["parent"]) if n["parent"] else None
        if parent is not None:
            parent_updates.append((parent, nid))
        if n["kind"] != "comment":
            edges.append((parent if parent is not None else file_node, nid, "DEFINES", ""))
    conn.executemany("UPDATE nodes SET parent_id = ? WHERE id = ?", parent_updates)

    others = [n for n in nodes if n["kind"] != "comment"]
    for n in nodes:
        if n["kind"] != "comment":
            continue
        idx = _annotation_target(n, others)
        target = ids[others[idx]["qname"]] if idx is not None else file_node
        edges.append((ids[n["qname"]], target, "ANNOTATES", ""))
    conn.executemany("INSERT OR IGNORE INTO edges(source_id, target_id, kind, meta) VALUES (?,?,?,?)", edges)

    ref_rows = []
    for r in parsed["refs"]:
        src = file_node if r["source"] == "" else ids.get(r["source"])
        if src is None:
            continue
        ref_rows.append((src, r["kind"], r["target"], r.get("qualifier", ""), r.get("meta", "")))
    conn.executemany("INSERT INTO refs(source_id, kind, target, qualifier, meta) VALUES (?,?,?,?,?)", ref_rows)
    conn.executemany("INSERT OR REPLACE INTO node_parts(node_id, part, hash) VALUES (?,?,?)", part_rows)
    db.insert_fts(conn, fts)
    return file_id


# --------------------------------------------------------------------------- resolve
def resolve(conn: sqlite3.Connection) -> dict:
    """Rebuild every resolved edge from the refs table."""
    conn.execute("DELETE FROM edges WHERE kind IN (%s)" % ",".join("?" * len(RESOLVED_KINDS)),
                 RESOLVED_KINDS)
    rel_by_file: dict[int, str] = {}
    fam_by_file: dict[int, str] = {}
    for r in conn.execute("SELECT id, rel_path, language FROM files"):
        rel_by_file[r["id"]] = r["rel_path"]
        fam_by_file[r["id"]] = _lang_family(r["language"])
    file_by_rel = {v: k for k, v in rel_by_file.items()}
    file_node = {r["file_id"]: r["id"] for r in conn.execute("SELECT id, file_id FROM nodes WHERE kind='file'")}
    node_file: dict[int, int] = {}
    node_parent: dict[int, int | None] = {}
    node_name: dict[int, str] = {}
    by_name: dict[str, list[tuple]] = defaultdict(list)
    by_qname: dict[str, int] = {}
    for r in conn.execute("SELECT id, file_id, kind, name, qualified_name, parent_id FROM nodes"):
        node_file[r["id"]] = r["file_id"]
        node_parent[r["id"]] = r["parent_id"]
        node_name[r["id"]] = r["name"]
        by_qname[r["qualified_name"]] = r["id"]
        if r["kind"] in ("function", "method", "class", "variable"):
            by_name[r["name"]].append((r["id"], r["file_id"], r["kind"], r["parent_id"]))

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
    refs = conn.execute("SELECT source_id, kind, target, qualifier, meta FROM refs ORDER BY id").fetchall()
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
            else:
                t = resolve_path(src_rel, r["target"]) if r["target"].startswith(("/", ".")) else None
                if t is None:
                    sibling = posixpath.normpath(posixpath.join(posixpath.dirname(src_rel), r["target"]))
                    t = file_by_rel.get(sibling)
                if t is not None:
                    emit(src, file_node[t], "LINKS_TO")

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
    for f in files:
        prev = existing.get(f["rel"])
        if prev and not force and prev["mtime"] == f["mtime"]:
            unchanged += 1
            continue
        text = _read(f["path"])
        if text is None:
            skipped["binary"] = skipped.get("binary", 0) + 1
            continue
        f["text"] = text
        f["hash"] = _hash(text)
        if prev and not force and prev["hash"] == f["hash"]:
            conn.execute("UPDATE files SET mtime = ? WHERE id = ?", (f["mtime"], prev["id"]))
            unchanged += 1
            continue
        changed.append(f)
    progress(f"scan: {len(files)} files, {len(changed)} changed, {unchanged} unchanged, "
             f"{len(removed)} removed, skipped {skipped}")

    problems: list[str] = []
    if changed:
        parsed, problems = _parse_files(changed)
        for f in changed:
            prev = existing.get(f["rel"])
            if prev:
                db.delete_file(conn, prev["id"])
            _insert_file(conn, f, parsed[f["rel"]])
        progress(f"parsed and stored {len(changed)} files")

    counts = {}
    if changed or removed or force:
        counts = resolve(conn)
        progress(f"resolved edges: {counts}")
    db.set_meta(conn, "indexed_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    conn.commit()
    return {"files": len(files), "changed": len(changed), "unchanged": unchanged,
            "removed": len(removed), "skipped": skipped, "problems": problems,
            "edges": counts, "seconds": round(time.time() - t0, 2)}
