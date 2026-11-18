"""Shared structural hierarchy builder and heading/reference extractors for prose and PDF documents.

Instead of splitting plain text or PDFs into flat, context-free chunks, this module builds
a nested section tree (Book/Part -> Chapter -> Section -> Page/Chunk) connected by DEFINES
and LINKS_TO edges, matching Markdown AST semantics.
"""
from __future__ import annotations

import re

_LINK_RE = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_WIKI_RE = re.compile(r"\[\[(?:([^\]|#]+))?(?:#([^\]|]+))?(?:\|[^\]]*)?\]\]")
_SEE_REF_RE = re.compile(
    r"\b(?:see|refer\s+to|cf\.?)\s+((?:Book|Volume|Part|Chapter|Section|Appendix)\s+[IVXLC\d]+(?:\.\d+)*)",
    re.IGNORECASE,
)
_TOC_DOTS_RE = re.compile(r"^\s*(?P<title>.+?)\s*(?:\.\s*){3,}\s*(?P<page>\d+)\s*$")

_ATX_RE = re.compile(r"^\s*(#{1,6})\s+(.*?)\s*#*\s*$")
_VOL_PART_RE = re.compile(
    r"^\s*((?:BOOK|VOLUME|PART)\s+(?:[IVXLC]+|\d+|The\s+\w+)(?:(?:\s*[:.\-]+\s*|\s+).+)?)$",
    re.IGNORECASE,
)
_CHAPTER_RE = re.compile(
    r"^\s*((?:CHAPTER|SECTION|APPENDIX)\s+(?:[IVXLC]+|\d+[A-Za-z]?)(?:(?:\s*[:.\-]+\s*|\s+).+)?)$",
    re.IGNORECASE,
)
_NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+){1,2})\s+([A-Z][^\n]{2,80})$")
_PAGE_MARKER_RE = re.compile(r"^\s*===\s*(Page\s+\d+(?:\s+of\s+\d+)?)\s*===\s*$", re.IGNORECASE)
_SETEXT_EQ_RE = re.compile(r"^\s*={4,}\s*$")
_SETEXT_DASH_RE = re.compile(r"^\s*-{4,}\s*$")
_SETEXT_RAW_HDR_RE = re.compile(r"^\s*\S.*\S\s*$")


def normalize_heading(title: str) -> str:
    """Collapse internal runs of spaces in a heading line."""
    return re.sub(r"\s+", " ", (title or "").strip())


def parse_toc_entry(line: str) -> tuple[str, int] | None:
    """Parse a dotted Table of Contents row ('Title (Author) .... 937') into (title, page)."""
    m = _TOC_DOTS_RE.match(line or "")
    if not m:
        return None
    title = normalize_heading(m.group("title"))
    if len(title) < 2:
        return None
    try:
        page = int(m.group("page"))
    except ValueError:
        return None
    return title, page


def strip_author_paren(title: str) -> str:
    """Strip trailing '(Author)' metadata from a TOC title for matching standalone headers."""
    return re.sub(r"\s*\([^)]{2,60}\)\s*$", "", title).strip()


def extract_prose_refs(text: str, source_qname: str, refs: list[dict]) -> None:
    """Extract path links, wikilinks, and explicit section cross-references."""
    for m in _LINK_RE.finditer(text):
        target = m.group(1)
        if re.match(r"^[a-z]+://", target) or target.startswith("#") or target.startswith("mailto:"):
            continue
        refs.append({"source": source_qname, "kind": "LINKS_TO", "target": target.split("#")[0],
                     "qualifier": "path", "meta": ""})
    for m in _WIKI_RE.finditer(text):
        page = (m.group(1) or "").strip()
        anchor = (m.group(2) or "").strip()
        target = page or anchor
        meta = anchor if page and anchor else ""
        if target:
            refs.append({"source": source_qname, "kind": "LINKS_TO", "target": target,
                         "qualifier": "wiki", "meta": meta})
    for m in _SEE_REF_RE.finditer(text):
        sec_ref = normalize_heading(m.group(1))
        if sec_ref:
            refs.append({"source": source_qname, "kind": "LINKS_TO", "target": sec_ref,
                         "qualifier": "section", "meta": ""})


def chunk_prose(text: str, max_chars: int = 2400) -> list[str]:
    """Split text along paragraph or line boundaries so no leaf node exceeds embedding context."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    units: list[str] = []
    for block in blocks:
        if len(block) <= max_chars:
            units.append(block)
            continue
        cur: list[str] = []
        cur_len = 0
        for ln in block.splitlines():
            while len(ln) > max_chars:
                if cur:
                    units.append("\n".join(cur).strip())
                    cur, cur_len = [], 0
                cut = ln.rfind(" ", 0, max_chars)
                if cut < max_chars // 2:
                    cut = max_chars
                units.append(ln[:cut].strip())
                ln = ln[cut:].lstrip()
            if cur and cur_len + 1 + len(ln) > max_chars:
                units.append("\n".join(cur).strip())
                cur, cur_len = [ln], len(ln)
            else:
                cur.append(ln)
                cur_len += len(ln) + (1 if cur_len else 0)
        if cur:
            units.append("\n".join(cur).strip())

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for unit in units:
        if buf and buf_len + 2 + len(unit) > max_chars:
            chunks.append("\n\n".join(buf))
            buf, buf_len = [unit], len(unit)
        else:
            buf.append(unit)
            buf_len += len(unit) + (2 if buf_len else 0)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks or [text[:max_chars]]


def detect_text_heading(line: str, next_line: str = "", prev_blank: bool = True,
                        *, known_titles: set[str] | None = None, allow_caps: bool = False) -> tuple[int, str, bool] | None:
    """Detect whether `line` starts a document heading.

    Returns `(level, title, consumed_next_line)` or None.
    """
    raw = line or ""
    stripped = raw.strip()
    if not stripped or parse_toc_entry(stripped):
        return None

    m_atx = _ATX_RE.match(stripped)
    if m_atx:
        return len(m_atx.group(1)), normalize_heading(m_atx.group(2)), False

    if next_line and len(stripped) <= 90 and not _SETEXT_EQ_RE.match(stripped) and not _SETEXT_DASH_RE.match(stripped):
        if _SETEXT_EQ_RE.match(next_line):
            return 1, normalize_heading(stripped), True
        if _SETEXT_DASH_RE.match(next_line) and not stripped.startswith("---"):
            return 2, normalize_heading(stripped), True

    m_pg = _PAGE_MARKER_RE.match(stripped)
    if m_pg:
        return 3, normalize_heading(m_pg.group(1)), False

    m_vol = _VOL_PART_RE.match(stripped)
    if m_vol and len(stripped) <= 90 and not stripped.endswith("."):
        return 1, normalize_heading(m_vol.group(1)), False

    m_chap = _CHAPTER_RE.match(stripped)
    if m_chap and len(stripped) <= 95 and not stripped.endswith((".", ",")):
        kw = stripped.split(None, 1)[0].upper()
        lvl = 3 if kw == "SECTION" else 2
        return lvl, normalize_heading(m_chap.group(1)), False

    m_num = _NUMBERED_RE.match(stripped)
    if m_num and not stripped.endswith((".", ",", ";")):
        depth = min(5, m_num.group(1).count(".") + 2)
        return depth, normalize_heading(stripped), False

    if known_titles and len(stripped) <= 95:
        norm = normalize_heading(stripped).lower()
        if norm in known_titles or strip_author_paren(norm) in known_titles:
            return 2, normalize_heading(stripped), False

    if allow_caps and prev_blank and 3 <= len(stripped) <= 72:
        letters = [ch for ch in stripped if ch.isalpha()]
        leading_spaces = len(raw) - len(raw.lstrip(" "))
        is_all_caps = len(letters) >= 4 and all(ch.isupper() for ch in letters)
        if is_all_caps and (leading_spaces >= 6 or len(stripped) <= 50) and not stripped.endswith((".", ",", ";", ":")):
            return 2, normalize_heading(stripped), False

    return None


class SectionTreeBuilder:
    """Builds a hierarchical list of section nodes + refs with parent_id and range tracking."""

    def __init__(self, *, max_chunk_chars: int = 2400):
        self.max_chunk_chars = max_chunk_chars
        self.nodes: list[dict] = []
        self.refs: list[dict] = []
        self._seen_qnames: set[str] = set()
        self._stack: list[tuple[int, dict]] = []
        self._buf: list[str] = []
        self._buf_start: int = 1
        self._buf_end: int = 1
        self._has_child: set[str] = set()
        self._single_block: dict[str, tuple[str, int, int]] = {}

    def _unique_qname(self, base: str, loc: int) -> str:
        if base not in self._seen_qnames:
            self._seen_qnames.add(base)
            return base
        cand = f"{base}@{loc}"
        n = 2
        while cand in self._seen_qnames:
            cand = f"{base}@{loc}_{n}"
            n += 1
        self._seen_qnames.add(cand)
        return cand

    @property
    def current(self) -> dict | None:
        return self._stack[-1][1] if self._stack else None

    def flush(self, end_loc: int | None = None, *, sub_label_prefix: str = "part") -> None:
        if self.current is None:
            self._buf.clear()
            return
        cur = self.current
        text = "\n".join(self._buf).strip()
        self._buf.clear()
        if end_loc is not None:
            self._extend_stack_end(end_loc)
        if not text:
            return

        cur_level = self._stack[-1][0]
        qcur = cur["qname"]
        # Attach short standalone prose straight onto leaf sections without children.
        if (
            not cur["docstring"]
            and qcur not in self._has_child
            and len(text) <= self.max_chunk_chars
        ):
            cur["docstring"] = text
            extract_prose_refs(text, qcur, self.refs)
            return

        chunks = chunk_prose(text, max_chars=self.max_chunk_chars)
        for idx, chunk in enumerate(chunks, 1):
            label = f"{sub_label_prefix} {idx}" if len(chunks) > 1 else sub_label_prefix
            child_q = self._unique_qname(f"{qcur}::{sub_label_prefix}-{idx}", self._buf_start)
            child_name = f"{cur['name']} ({label})" if label else cur["name"]
            child = {
                "qname": child_q,
                "kind": "section",
                "name": child_name,
                "parent": qcur,
                "signature": label,
                "docstring": chunk,
                "source": "",
                "line_start": self._buf_start,
                "line_end": max(self._buf_start, self._buf_end),
            }
            self.nodes.append(child)
            self._has_child.add(qcur)
            extract_prose_refs(chunk, child_q, self.refs)

    def _extend_stack_end(self, end_loc: int) -> None:
        for _lvl, n in self._stack:
            if end_loc > n["line_end"]:
                n["line_end"] = end_loc

    def _promote_single_block(self, qcur: str) -> None:
        meta = self._single_block.pop(qcur, None)
        if meta is None:
            return
        parent_node = next((n for _lvl, n in self._stack if n["qname"] == qcur), None)
        if parent_node is None:
            parent_node = next((n for n in reversed(self.nodes) if n["qname"] == qcur), None)
        if parent_node is None or not parent_node["docstring"]:
            return
        label, b_start, b_end = meta
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", label.lower()).strip("-") or "chunk"
        child_q = self._unique_qname(f"{qcur}::{slug}", b_start)
        self.nodes.append({
            "qname": child_q,
            "kind": "section",
            "name": f"{parent_node['name']} - {label}",
            "parent": qcur,
            "signature": label,
            "docstring": parent_node["docstring"],
            "source": "",
            "line_start": b_start,
            "line_end": max(b_start, b_end),
        })
        parent_node["docstring"] = ""
        self._has_child.add(qcur)
        for ref in self.refs:
            if ref["source"] == qcur:
                ref["source"] = child_q

    def open_section(self, level: int, title: str, start: int, end: int | None = None,
                     *, signature: str = "", sub_label_prefix: str = "part") -> dict:
        self.flush(start - 1 if start > 1 else start, sub_label_prefix=sub_label_prefix)
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        parent = self._stack[-1][1]["qname"] if self._stack else None
        if parent is not None and parent in self._single_block:
            self._promote_single_block(parent)
        clean_title = normalize_heading(title) or f"Section {start}"
        base_q = f"{parent}::{clean_title}" if parent else clean_title
        qname = self._unique_qname(base_q, start)
        node_end = max(start, end if end is not None else start)
        node = {
            "qname": qname,
            "kind": "section",
            "name": clean_title,
            "parent": parent,
            "signature": signature or clean_title,
            "docstring": "",
            "source": "",
            "line_start": start,
            "line_end": node_end,
        }
        self.nodes.append(node)
        if parent is not None:
            self._has_child.add(parent)
        self._stack.append((level, node))
        self._extend_stack_end(node_end)
        self._buf_start = start
        self._buf_end = start
        return node

    def add_line(self, line: str, loc: int, *, default_level: int = 1, default_title: str = "(intro)") -> None:
        if self.current is None:
            if not line.strip():
                return
            self.open_section(default_level, default_title, loc)
        if not self._buf:
            self._buf_start = loc
        self._buf.append(line)
        self._buf_end = loc
        self._extend_stack_end(loc)

    def add_block(self, text: str, start: int, end: int, *, label: str,
                  default_title: str = "(intro)", sub_label_prefix: str = "part") -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        if self.current is None:
            self.open_section(1, default_title, start, end)
        cur = self.current
        assert cur is not None
        qcur = cur["qname"]
        self._extend_stack_end(max(start, end))

        chunks = chunk_prose(cleaned, max_chars=self.max_chunk_chars)
        total = len(chunks)
        if (total > 1 or cur["docstring"] or qcur in self._has_child) and qcur in self._single_block:
            self._promote_single_block(qcur)
        for idx, chunk in enumerate(chunks, 1):
            suffix = f" ({idx}/{total})" if total > 1 else ""
            # Put single short blocks directly on a fresh leaf section; promote if more blocks arrive.
            if (
                total == 1
                and not self._buf
                and not cur["docstring"]
                and qcur not in self._has_child
                and len(chunk) <= self.max_chunk_chars
            ):
                cur["docstring"] = chunk
                self._single_block[qcur] = (label, start, max(start, end))
                extract_prose_refs(chunk, qcur, self.refs)
            else:
                tag = f"{label}{suffix}"
                slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", tag.lower()).strip("-") or "chunk"
                child_q = self._unique_qname(f"{qcur}::{slug}", start)
                child = {
                    "qname": child_q,
                    "kind": "section",
                    "name": f"{cur['name']} - {tag}",
                    "parent": qcur,
                    "signature": tag,
                    "docstring": chunk,
                    "source": "",
                    "line_start": start,
                    "line_end": max(start, end),
                }
                self.nodes.append(child)
                self._has_child.add(qcur)
                extract_prose_refs(chunk, child_q, self.refs)

    def finalize(self, total_end: int, *, sub_label_prefix: str = "part") -> dict:
        self.flush(total_end, sub_label_prefix=sub_label_prefix)
        # Ensure multi-child structural parents have a concise TOC summary when their body is empty.
        by_q = {n["qname"]: n for n in self.nodes}
        children_by_parent: dict[str, list[dict]] = {}
        for n in self.nodes:
            if n["parent"]:
                children_by_parent.setdefault(n["parent"], []).append(n)
        for q, kids in children_by_parent.items():
            parent_node = by_q.get(q)
            if parent_node is not None and not parent_node["docstring"] and kids:
                titles = [k["name"] for k in kids[:12]]
                more = f" (+{len(kids) - 12} more)" if len(kids) > 12 else ""
                parent_node["docstring"] = f"{parent_node['name']}: " + "; ".join(titles) + more
        return {"file_doc": "", "nodes": self.nodes, "refs": self.refs}


def parse_structured_text(src: str, *, max_chunk_chars: int = 2400) -> dict:
    """Parse plain .txt with structural hierarchy if headings exist, else split cleanly by paragraphs."""
    lines = src.splitlines()
    body = src.strip()
    if not body:
        return {"file_doc": "", "nodes": [], "refs": []}

    has_headings = any(
        detect_text_heading(
            lines[i],
            lines[i + 1] if i + 1 < len(lines) else "",
            i == 0 or not lines[i - 1].strip(),
            allow_caps=False,
        )
        is not None
        for i in range(len(lines))
    )

    if has_headings:
        builder = SectionTreeBuilder(max_chunk_chars=max_chunk_chars)
        i = 0
        while i < len(lines):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            prev_blank = i == 0 or not lines[i - 1].strip()
            hit = detect_text_heading(lines[i], nxt, prev_blank, allow_caps=False)
            if hit is not None:
                lvl, title, consumed_next = hit
                builder.open_section(lvl, title, i + 1)
                i += 2 if consumed_next else 1
                continue
            builder.add_line(lines[i], i + 1, default_level=1, default_title="(intro)")
            i += 1
        return builder.finalize(len(lines))

    # Fallback for plain unstructured text without headings: paragraph-bounded sections with refs.
    if len(lines) <= 80 and len(body) <= max_chunk_chars:
        refs: list[dict] = []
        extract_prose_refs(body, "(text)", refs)
        return {
            "file_doc": "",
            "nodes": [{
                "qname": "(text)", "kind": "section", "name": "(text)", "parent": None,
                "signature": "", "docstring": body, "source": "",
                "line_start": 1, "line_end": max(1, len(lines)),
            }],
            "refs": refs,
        }

    nodes: list[dict] = []
    refs = []
    buf: list[str] = []
    start = 1
    for idx, ln in enumerate(lines, 1):
        cur_chars = sum(len(x) + 1 for x in buf)
        if (not ln.strip() and len(buf) >= 20) or (buf and cur_chars + len(ln) > max_chunk_chars and not ln.strip()):
            chunk = "\n".join(buf).strip()
            if chunk:
                first = chunk.splitlines()[0][:60]
                q = f"(text)@{start}"
                nodes.append({"qname": q, "kind": "section", "name": first, "parent": None,
                              "signature": "", "docstring": chunk, "source": "",
                              "line_start": start, "line_end": idx - 1})
                extract_prose_refs(chunk, q, refs)
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
            q = f"(text)@{start}"
            nodes.append({"qname": q, "kind": "section", "name": first, "parent": None,
                          "signature": "", "docstring": chunk, "source": "",
                          "line_start": start, "line_end": len(lines)})
            extract_prose_refs(chunk, q, refs)
    return {"file_doc": "", "nodes": nodes, "refs": refs}
