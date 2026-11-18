"""Hierarchical PDF parser and bounded page reader for codesearch.

Extracts PDF structure into a nested section tree (Outline bookmarks, dotted Tables of
Contents, typographic font headings, Volume/Part/Chapter/Setext headings, and per-page
leaves bounded by character limit) rather than dumping entire PDFs as monolithic blobs.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from codesearch.parsers import doc_structure

_BARE_PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
_BULLET_START_RE = re.compile(r"^(?:[-*+\u2022]|\d+[.)])\s")
_PAGE_SCAN_LIMIT = 6
_TOC_SCAN_PAGES = 40


def _clean_text(raw: str) -> str:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _is_bold_font(font_name: str) -> bool:
    low = font_name.lower()
    return any(tok in low for tok in ("bold", "heavy", "black", "demi", "semibold"))


def _extract_page_lines(page, *, with_style: bool) -> list[tuple[str, float, bool]]:
    """Return `[(line_text, dominant_font_size, is_all_bold), ...]` in reading order."""
    if not with_style:
        text = (page.extract_text() or "").replace("\r\n", "\n").replace("\r", "\n")
        return [(ln.rstrip(), 0.0, False) for ln in text.splitlines()]

    lines: list[tuple[str, float, bool]] = []
    cur_text: list[str] = []
    cur_sizes: Counter[float] = Counter()
    seen_non_ws = False
    all_bold = True

    def flush_line() -> None:
        nonlocal seen_non_ws, all_bold
        joined = "".join(cur_text).rstrip()
        cur_text.clear()
        dom_size = cur_sizes.most_common(1)[0][0] if cur_sizes else 0.0
        cur_sizes.clear()
        line_bold = seen_non_ws and all_bold
        seen_non_ws = False
        all_bold = True
        lines.append((joined, dom_size, line_bold))

    def visitor(text, cm, tm, font_dict, font_size) -> None:
        nonlocal seen_non_ws, all_bold
        if not text:
            return
        scale = 1.0
        if tm and len(tm) > 3:
            try:
                tm_y = abs(float(tm[3]))
                if tm_y > 0:
                    scale = tm_y
            except (TypeError, ValueError):
                pass
        try:
            eff = round(abs(float(font_size or 0.0)) * scale, 1)
        except (TypeError, ValueError):
            eff = 0.0
        fname = str(font_dict.get("/BaseFont", "")) if font_dict else ""
        bold = _is_bold_font(fname)

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        parts = normalized.split("\n")
        for idx, part in enumerate(parts):
            if part:
                cur_text.append(part)
                non_ws = sum(1 for ch in part if not ch.isspace())
                if non_ws > 0:
                    if eff > 0:
                        cur_sizes[eff] += non_ws
                    seen_non_ws = True
                    if not bold:
                        all_bold = False
            if idx < len(parts) - 1:
                flush_line()

    try:
        page.extract_text(visitor_text=visitor)
        if cur_text or not lines:
            flush_line()
        return lines
    except Exception:
        text = (page.extract_text() or "").replace("\r\n", "\n").replace("\r", "\n")
        return [(ln.rstrip(), 0.0, False) for ln in text.splitlines()]


def _extract_outline(reader, total_pages: int) -> tuple[dict[int, list[tuple[int, str]]], list[str]]:
    """Walk PDF bookmark outline and group `(level, title)` entries by 1-based page."""
    by_page: dict[int, list[tuple[int, str]]] = {}
    toc: list[str] = []
    try:
        outline = getattr(reader, "outline", None) or []
    except Exception:
        return by_page, toc

    def walk(items, depth: int = 1) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, min(6, depth + 1))
                continue
            title = doc_structure.normalize_heading(str(getattr(item, "title", "") or ""))
            if not title:
                continue
            if depth == 1 and len(toc) < 20:
                toc.append(title[:80])
            try:
                page_idx = reader.get_destination_page_number(item)
            except Exception:
                page_idx = None
            if isinstance(page_idx, int) and 0 <= page_idx < total_pages:
                by_page.setdefault(page_idx + 1, []).append((depth, title[:95]))

    try:
        walk(outline)
    except Exception:
        return {}, []
    return by_page, toc


def _detect_typography(pages) -> tuple[bool, float, bool]:
    """Sample opening pages to detect modal body font size and whether font headings exist."""
    size_weights: Counter[float] = Counter()
    sample_lines: list[tuple[str, float, bool]] = []
    nonempty_seen = 0
    for page in pages[: min(len(pages), _PAGE_SCAN_LIMIT * 2)]:
        plines = _extract_page_lines(page, with_style=True)
        has_text = any(s.strip() for s, _, _ in plines)
        if not has_text:
            continue
        sample_lines.extend(plines)
        for s, sz, _ in plines:
            if sz >= 4.0:
                n = sum(1 for ch in s if not ch.isspace())
                if n > 0:
                    size_weights[sz] += n
        nonempty_seen += 1
        if nonempty_seen >= _PAGE_SCAN_LIMIT:
            break

    if not size_weights:
        return False, 0.0, False
    body_size = size_weights.most_common(1)[0][0]
    if body_size <= 0:
        return False, 0.0, False

    bold_chars = sum(len(s.strip()) for s, _, b in sample_lines if b and s.strip())
    total_chars = sum(len(s.strip()) for s, _, _ in sample_lines if s.strip())
    body_mostly_bold = total_chars > 0 and (bold_chars / total_chars) > 0.45

    has_font_headings = False
    for s, sz, b in sample_lines:
        stripped = s.strip()
        if not stripped or len(stripped) > 95 or doc_structure.parse_toc_entry(stripped):
            continue
        if sum(ch.isalpha() for ch in stripped) < 2:
            continue
        if sz >= body_size * 1.15 or (b and not body_mostly_bold and len(stripped) <= 72 and len(stripped.split()) <= 8):
            has_font_headings = True
            break

    return has_font_headings, body_size, body_mostly_bold


def _strip_running_headers_footers(
    page_lines: list[list[tuple[str, float, bool]]],
    total_pages: int,
) -> None:
    """Remove identical headers/footers repeated across many pages and bare page numbers."""
    if total_pages < 3:
        return

    first_counts: Counter[str] = Counter()
    last_counts: Counter[str] = Counter()
    first_idx: list[int | None] = []
    last_idx: list[int | None] = []

    for plines in page_lines:
        f_i = next((i for i, (s, _, _) in enumerate(plines) if s.strip()), None)
        l_i = next((i for i in range(len(plines) - 1, -1, -1) if plines[i][0].strip()), None)
        first_idx.append(f_i)
        last_idx.append(l_i)
        if f_i is not None:
            norm = doc_structure.normalize_heading(plines[f_i][0])
            if 2 <= len(norm) <= 90:
                first_counts[norm.lower()] += 1
        if l_i is not None and l_i != f_i:
            norm = doc_structure.normalize_heading(plines[l_i][0])
            if 2 <= len(norm) <= 90:
                last_counts[norm.lower()] += 1

    repeat_threshold = max(4, int(total_pages * 0.35)) if total_pages >= 6 else total_pages + 1
    bad_first = {k for k, c in first_counts.items() if c >= repeat_threshold}
    bad_last = {k for k, c in last_counts.items() if c >= repeat_threshold}

    for p_i, plines in enumerate(page_lines):
        f_i = first_idx[p_i]
        if f_i is not None:
            txt = doc_structure.normalize_heading(plines[f_i][0])
            if txt.lower() in bad_first or _BARE_PAGE_NUM_RE.match(txt):
                plines[f_i] = ("", 0.0, False)
        l_i = last_idx[p_i]
        if l_i is not None:
            txt = doc_structure.normalize_heading(plines[l_i][0])
            if txt.lower() in bad_last or _BARE_PAGE_NUM_RE.match(txt):
                plines[l_i] = ("", 0.0, False)


def _titles_match(candidate: str, section_title: str) -> bool:
    a = doc_structure.strip_author_paren(doc_structure.normalize_heading(candidate).lower())
    b = doc_structure.strip_author_paren(doc_structure.normalize_heading(section_title).lower())
    if not a or not b:
        return False
    if a == b:
        return True
    strip_prefix = lambda s: re.sub(r"^(?:chapter|section|part|volume|book|\d+(?:\.\d+)*)\s*[:.\-]?\s*", "", s).strip()
    sa, sb = strip_prefix(a), strip_prefix(b)
    return bool(sa and sb and sa == sb)


def _font_heading(
    stripped: str,
    font_size: float,
    all_bold: bool,
    prev_blank: bool,
    body_size: float,
    body_mostly_bold: bool,
) -> int | None:
    """Return relative heading depth `(1..3)` if typography signals a standalone heading."""
    if not stripped or len(stripped) > 95 or doc_structure.parse_toc_entry(stripped):
        return None
    if _BULLET_START_RE.match(stripped) or sum(ch.isalpha() for ch in stripped) < 2:
        return None
    if stripped.endswith((",", ";")):
        return None

    ratio = (font_size / body_size) if body_size > 0 and font_size > 0 else 0.0
    if ratio >= 1.55:
        return 1
    if ratio >= 1.30:
        return 2
    if ratio >= 1.15:
        if stripped.endswith(".") and len(stripped.split()) > 6:
            return None
        return 3
    if (
        all_bold
        and not body_mostly_bold
        and prev_blank
        and 3 <= len(stripped) <= 72
        and len(stripped.split()) <= 8
        and not stripped.endswith((".", ":", "?", "!"))
    ):
        return 3
    return None


def _open_toc_page(
    builder: doc_structure.SectionTreeBuilder,
    toc_rows: list[tuple[str, int]],
    page_text: str,
    page_num: int,
    base_level: int,
) -> None:
    """Attach a Table of Contents section/page node and emit section LINKS_TO cross-refs."""
    cur = builder.current
    if cur is None or cur["name"] != "Table of Contents":
        builder.open_section(max(1, base_level), "Table of Contents", page_num, page_num)
    node_before = len(builder.nodes)
    prev_refs = len(builder.refs)
    builder.add_block(page_text, page_num, page_num, label=f"Page {page_num}")
    # Determine the node representing this TOC page for outbound LINKS_TO edges.
    if len(builder.nodes) > node_before:
        src_qname = builder.nodes[-1]["qname"]
    elif builder.current is not None:
        src_qname = builder.current["qname"]
    else:
        return
    seen_targets: set[str] = set()
    for ref in builder.refs[prev_refs:]:
        seen_targets.add(ref["target"].lower())
    for raw_title, _printed_page in toc_rows:
        for cand in (raw_title, doc_structure.strip_author_paren(raw_title)):
            key = cand.strip()
            if key and key.lower() not in seen_targets:
                seen_targets.add(key.lower())
                builder.refs.append({
                    "source": src_qname,
                    "kind": "LINKS_TO",
                    "target": key,
                    "qualifier": "section",
                    "meta": "",
                })


def parse(path: str | Path, rel_path: str = "", *, chunk_chars: int = 2400, progress=None) -> dict:
    """Extract PDF into a hierarchical tree of outlines, headings, TOC links, and page leaves."""
    try:
        import pypdf
    except ImportError:
        return {
            "file_doc": "PDF (pypdf not installed)",
            "nodes": [],
            "refs": [],
            "page_count": 0,
            "problems": ["pypdf is not installed (install pypdf to index PDF text)"],
        }

    problems: list[str] = []
    try:
        reader = pypdf.PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            try:
                if not reader.decrypt(""):
                    return {
                        "file_doc": "PDF (encrypted)",
                        "nodes": [],
                        "refs": [],
                        "page_count": 0,
                        "problems": ["PDF is password-protected and cannot be read"],
                    }
            except Exception as exc:
                return {
                    "file_doc": "PDF (encrypted)",
                    "nodes": [],
                    "refs": [],
                    "page_count": 0,
                    "problems": [f"PDF decrypt failed: {exc}"],
                }

        pages = reader.pages
        total_pages = len(pages)
        meta = getattr(reader, "metadata", None) or {}
        bookmarks_by_page, toc = _extract_outline(reader, total_pages)
        has_outline = bool(bookmarks_by_page)

        has_font_headings, body_size, body_mostly_bold = _detect_typography(pages)
        raw_pages: list[list[tuple[str, float, bool]]] = []
        for idx, page in enumerate(pages, 1):
            try:
                raw_pages.append(_extract_page_lines(page, with_style=has_font_headings))
            except Exception as exc:
                problems.append(f"page {idx}: text extraction failed ({exc})")
                raw_pages.append([])
            if progress is not None and total_pages >= 200 and idx % 200 == 0:
                progress(f"  pdf: {rel_path or Path(path).name} page {idx}/{total_pages}")
    except Exception as exc:
        return {
            "file_doc": "PDF (unreadable)",
            "nodes": [],
            "refs": [],
            "page_count": 0,
            "problems": [f"PDF parse error: {exc}"],
        }

    _strip_running_headers_footers(raw_pages, total_pages)

    # Frontmatter pass: collect dotted TOC entries so matching chapter headers connect tightly.
    known_titles: set[str] = set()
    page_toc_rows: dict[int, list[tuple[str, int]]] = {}
    for p_idx in range(1, min(total_pages, _TOC_SCAN_PAGES) + 1):
        rows: list[tuple[str, int]] = []
        for ln, _, _ in raw_pages[p_idx - 1]:
            entry = doc_structure.parse_toc_entry(ln)
            if entry is not None:
                rows.append(entry)
        if len(rows) >= 2:
            page_toc_rows[p_idx] = rows
            for raw_title, _ in rows:
                t_norm = doc_structure.normalize_heading(raw_title).lower()
                known_titles.add(t_norm)
                stripped = doc_structure.strip_author_paren(t_norm)
                if len(stripped) >= 3:
                    known_titles.add(stripped)

    builder = doc_structure.SectionTreeBuilder(max_chunk_chars=chunk_chars)
    active_outline_level = 0
    last_nonempty_page = 0

    for p_num, plines in enumerate(raw_pages, 1):
        non_empty = any(s.strip() for s, _, _ in plines)
        if non_empty:
            last_nonempty_page = p_num

        page_bms = bookmarks_by_page.get(p_num, [])
        bm_queue = list(page_bms)

        # Open leading outline bookmark(s) on this page (stacking Part -> Chapter chains).
        pending_siblings: list[tuple[int, str]] = []
        if bm_queue:
            first_lvl, first_title = bm_queue.pop(0)
            builder.open_section(first_lvl, first_title, p_num, p_num)
            active_outline_level = first_lvl
            while bm_queue and bm_queue[0][0] > active_outline_level:
                sub_lvl, sub_title = bm_queue.pop(0)
                builder.open_section(sub_lvl, sub_title, p_num, p_num)
                active_outline_level = sub_lvl
            pending_siblings = bm_queue

        if p_num in page_toc_rows:
            page_txt = _clean_text("\n".join(s for s, _, _ in plines))
            if page_txt:
                _open_toc_page(
                    builder,
                    page_toc_rows[p_num],
                    page_txt,
                    p_num,
                    base_level=active_outline_level or 1,
                )
            for sib_lvl, sib_title in pending_siblings:
                builder.open_section(sib_lvl, sib_title, p_num, p_num)
                active_outline_level = sib_lvl
            continue

        if not non_empty:
            for sib_lvl, sib_title in pending_siblings:
                builder.open_section(sib_lvl, sib_title, p_num, p_num)
                active_outline_level = sib_lvl
            continue

        buf: list[str] = []
        i = 0
        n_lines = len(plines)
        while i < n_lines:
            raw_ln, sz, bold = plines[i]
            stripped = raw_ln.strip()
            next_raw = plines[i + 1][0] if i + 1 < n_lines else ""
            prev_blank = i == 0 or not plines[i - 1][0].strip()

            # Check whether a pending sibling outline bookmark starts right at this line.
            if pending_siblings and stripped and _titles_match(stripped, pending_siblings[0][1]):
                slice_txt = _clean_text("\n".join(buf))
                if slice_txt:
                    builder.add_block(slice_txt, p_num, p_num, label=f"Page {p_num}")
                buf.clear()
                sib_lvl, sib_title = pending_siblings.pop(0)
                builder.open_section(sib_lvl, sib_title, p_num, p_num)
                active_outline_level = sib_lvl
                while pending_siblings and pending_siblings[0][0] > active_outline_level:
                    sub_lvl, sub_title = pending_siblings.pop(0)
                    builder.open_section(sub_lvl, sub_title, p_num, p_num)
                    active_outline_level = sub_lvl
                i += 1
                continue

            # Skip a standalone page-top heading that repeats the active outline bookmark verbatim.
            if (
                not buf
                and builder.current is not None
                and has_outline
                and stripped
                and len(stripped) <= 100
                and _titles_match(stripped, builder.current["name"])
            ):
                i += 1
                continue

            # Check structural text heading first, then font-based typographic heading.
            hit = doc_structure.detect_text_heading(
                raw_ln,
                next_raw,
                prev_blank,
                known_titles=known_titles,
                allow_caps=not has_outline,
            )
            consumed_next = False
            heading_level: int | None = None
            heading_title = ""

            if hit is not None:
                heading_level, heading_title, consumed_next = hit
            elif has_font_headings and stripped:
                f_lvl = _font_heading(stripped, sz, bold, prev_blank, body_size, body_mostly_bold)
                if f_lvl is not None:
                    heading_level = f_lvl
                    heading_title = doc_structure.normalize_heading(stripped)

            if heading_level is not None and heading_title:
                # Deduplicate inline heading repeating the currently open section name.
                if not buf and builder.current is not None and _titles_match(heading_title, builder.current["name"]):
                    i += 2 if consumed_next else 1
                    continue

                slice_txt = _clean_text("\n".join(buf))
                if slice_txt:
                    builder.add_block(slice_txt, p_num, p_num, label=f"Page {p_num}")
                buf.clear()

                target_level = (active_outline_level + heading_level) if active_outline_level else heading_level
                target_level = min(6, max(1, target_level))
                builder.open_section(target_level, heading_title[:95], p_num, p_num)
                i += 2 if consumed_next else 1
                continue

            buf.append(raw_ln)
            i += 1

        slice_txt = _clean_text("\n".join(buf))
        if slice_txt:
            builder.add_block(slice_txt, p_num, p_num, label=f"Page {p_num}")
        for sib_lvl, sib_title in pending_siblings:
            builder.open_section(sib_lvl, sib_title, p_num, p_num)
            active_outline_level = sib_lvl

    res = builder.finalize(last_nonempty_page or total_pages)
    nodes = res["nodes"]
    refs = res["refs"]

    doc_parts = [f"PDF, {total_pages} pages"]
    for label, key in (("Title", "/Title"), ("Author", "/Author"), ("Subject", "/Subject")):
        val = str(meta.get(key) or "").strip() if hasattr(meta, "get") else ""
        if val:
            doc_parts.append(f"{label}: {val[:120]}")
    if toc:
        doc_parts.append("Outline: " + " | ".join(toc))
    elif nodes:
        top_secs = [n["name"] for n in nodes if n["parent"] is None][:12]
        if top_secs:
            doc_parts.append("Sections: " + " | ".join(top_secs))
    if total_pages > 0 and not nodes:
        doc_parts.append("No extractable text (scanned images?)")
        problems.append("PDF contains no extractable text (scanned images?)")

    return {
        "file_doc": "\n".join(doc_parts),
        "nodes": nodes,
        "refs": refs,
        "page_count": total_pages,
        "problems": problems,
    }


def read_pages(path: str | Path, start: int = 1, end: int | None = None,
               *, max_pages: int = 20) -> dict:
    """Read a bounded inclusive 1-based range of PDF pages as formatted lines."""
    try:
        import pypdf
    except ImportError as exc:
        return {"error": f"pypdf is not installed: {exc}"}

    try:
        reader = pypdf.PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            if not reader.decrypt(""):
                return {"error": "PDF is password-protected and cannot be read"}
        total = len(reader.pages)
    except Exception as exc:
        return {"error": f"failed to open PDF: {exc}"}

    if total == 0:
        return {"total_pages": 0, "start": 0, "end": 0, "clamped": False, "lines": []}

    lo = max(1, min(int(start), total))
    req_end = lo if end is None else max(lo, min(int(end), total))
    max_span = max(1, int(max_pages))
    clamped = (req_end - lo + 1) > max_span
    hi = min(total, lo + max_span - 1) if clamped else req_end

    lines: list[str] = []
    for page_num in range(lo, hi + 1):
        try:
            body = _clean_text(reader.pages[page_num - 1].extract_text() or "")
        except Exception as exc:
            body = f"[error extracting page {page_num}: {exc}]"
        lines.append(f"=== Page {page_num} of {total} ===")
        if body:
            lines.extend(body.splitlines())
        else:
            lines.append("[blank or scanned page with no extractable text]")
        if page_num < hi:
            lines.append("")

    return {
        "total_pages": total,
        "start": lo,
        "end": hi,
        "clamped": clamped,
        "lines": lines,
    }
