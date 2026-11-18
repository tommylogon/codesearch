"""Markdown and plain-text extractor.

Nodes: `section` (one per heading, holding the prose under it) and `block` (fenced
code). Links become LINKS_TO refs: `[text](path)` and Obsidian `[[Note]]` wikilinks.
Section text is stored whole; nothing is cut here.
"""
from __future__ import annotations

import re

from codesearch.parsers import doc_structure

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^(\s*)(```+|~~~+)\s*([\w+-]*)")
_LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_WIKI = re.compile(r"\[\[(?:([^\]|#]+))?(?:#([^\]|]+))?(?:\|[^\]]*)?\]\]")


def _frontmatter(lines: list[str]) -> tuple[str, int]:
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return "\n".join(lines[1:i]).strip(), i + 1
    return "", 0


def parse_text(src: str) -> dict:
    """Plain .txt: extract hierarchical sections when headings exist, else split cleanly."""
    return doc_structure.parse_structured_text(src)


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
        doc_structure.extract_prose_refs(text, source, refs)

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
