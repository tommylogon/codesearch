"""Markdown and plain-text extractor.

Nodes: `section` (one per heading, holding the prose under it) and `block` (fenced
code). Links become LINKS_TO refs: `[text](path)` and Obsidian `[[Note]]` wikilinks.
Section text is stored whole; nothing is cut here.
"""
from __future__ import annotations

import re

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^(\s*)(```+|~~~+)\s*([\w+-]*)")
_LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_WIKI = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")


def _frontmatter(lines: list[str]) -> tuple[str, int]:
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return "\n".join(lines[1:i]).strip(), i + 1
    return "", 0


def parse_text(src: str) -> dict:
    """Plain .txt: short files are one section; longer files split into paragraph blocks
    so a 10,000-line log never acts as one giant bag-of-words document."""
    lines = src.splitlines()
    body = src.strip()
    if not body:
        return {"file_doc": "", "nodes": [], "refs": []}
    if len(lines) <= 80:
        return {"file_doc": "", "nodes": [{
            "qname": "(text)", "kind": "section", "name": "(text)", "parent": None,
            "signature": "", "docstring": body, "source": "",
            "line_start": 1, "line_end": max(1, len(lines)),
        }], "refs": []}
    nodes = []
    buf: list[str] = []
    start = 1
    for idx, ln in enumerate(lines, 1):
        if not ln.strip() and len(buf) >= 20:
            chunk = "\n".join(buf).strip()
            if chunk:
                first = chunk.splitlines()[0][:60]
                nodes.append({"qname": f"(text)@{start}", "kind": "section", "name": first,
                              "parent": None, "signature": "", "docstring": chunk, "source": "",
                              "line_start": start, "line_end": idx - 1})
            buf = []
            start = idx + 1
        else:
            if not buf and not ln.strip():
                start = idx + 1
                continue
            buf.append(ln)
    if buf:
        chunk = "\n".join(buf).strip()
        if chunk:
            first = chunk.splitlines()[0][:60]
            nodes.append({"qname": f"(text)@{start}", "kind": "section", "name": first,
                          "parent": None, "signature": "", "docstring": chunk, "source": "",
                          "line_start": start, "line_end": len(lines)})
    return {"file_doc": "", "nodes": nodes, "refs": []}


def parse(src: str, rel_path: str = "") -> dict:
    lines = src.splitlines()
    result = {"file_doc": "", "nodes": [], "refs": []}
    fm, start = _frontmatter(lines)
    result["file_doc"] = fm
    nodes, refs = result["nodes"], result["refs"]

    stack: list[tuple[int, str]] = []        # (level, qname)
    cur: dict | None = None
    prose: list[str] = []
    in_fence: dict | None = None

    def flush(end_line: int):
        nonlocal cur, prose
        if cur is not None:
            cur["docstring"] = "\n".join(prose).strip()
            cur["line_end"] = max(cur["line_start"], end_line)
        prose = []

    def link_refs(text: str, source: str):
        for m in _LINK.finditer(text):
            target = m.group(1)
            if re.match(r"^[a-z]+://", target) or target.startswith("#") or target.startswith("mailto:"):
                continue
            refs.append({"source": source, "kind": "LINKS_TO", "target": target.split("#")[0],
                         "qualifier": "path", "meta": ""})
        for m in _WIKI.finditer(text):
            refs.append({"source": source, "kind": "LINKS_TO", "target": m.group(1).strip(),
                         "qualifier": "wiki", "meta": ""})

    def ensure_intro(line_no: int):
        nonlocal cur
        if cur is None:
            cur = {"qname": "(intro)", "kind": "section", "name": "(intro)", "parent": None,
                   "signature": "", "docstring": "", "source": "",
                   "line_start": line_no, "line_end": line_no}
            nodes.append(cur)

    for idx in range(start, len(lines)):
        line_no = idx + 1
        line = lines[idx]
        if in_fence is not None:
            if re.match(r"^\s*" + re.escape(in_fence["marker"][:3]) + r"+\s*$", line):
                in_fence["line_end"] = line_no
                in_fence["source"] = "\n".join(in_fence.pop("buf"))
                in_fence["docstring"] = ""
                in_fence.pop("marker")
                in_fence = None
            else:
                in_fence["buf"].append(line)
            continue
        fence = _FENCE.match(line)
        if fence:
            ensure_intro(line_no)
            lang = fence.group(3) or "text"
            block = {"qname": f"{cur['qname']}::block@{line_no}", "kind": "block", "name": lang,
                     "parent": cur["qname"], "signature": lang, "docstring": "", "source": "",
                     "line_start": line_no, "line_end": line_no, "buf": [], "marker": fence.group(2)}
            nodes.append(block)
            in_fence = block
            continue
        heading = _HEADING.match(line)
        if heading:
            flush(line_no - 1)
            level, title = len(heading.group(1)), heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent = stack[-1][1] if stack else None
            qname = f"{parent}::{title}" if parent else title
            # repeated headings under one parent stay unique
            existing = {n["qname"] for n in nodes}
            if qname in existing:
                qname = f"{qname}@{line_no}"
            cur = {"qname": qname, "kind": "section", "name": title, "parent": parent,
                   "signature": "#" * level + " " + title, "docstring": "", "source": "",
                   "line_start": line_no, "line_end": line_no}
            nodes.append(cur)
            stack.append((level, qname))
            continue
        if line.strip():
            ensure_intro(line_no)
        if cur is not None:
            prose.append(line)
            link_refs(line, cur["qname"])
    if in_fence is not None:
        in_fence["line_end"] = len(lines)
        in_fence["source"] = "\n".join(in_fence.pop("buf"))
        in_fence.pop("marker", None)
    flush(len(lines))
    for n in nodes:
        n.pop("buf", None)
        n.pop("marker", None)
    return result
