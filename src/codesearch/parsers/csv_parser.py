"""CSV (comma-separated, semicolon-separated, tab-separated) tabular extractor.

Rows become `record` nodes with exact line ranges (including multiline cells).
Prose fields (summaries, descriptions, notes) become `section` nodes embedded in vector search.
Categorical / tag fields become keyword-only `tag` nodes.
Numeric ratings and pure numbers are captured directly in the row's record docstring.
"""
from __future__ import annotations

import csv
import io
import re
from typing import Any

from .. import config
from .json_parser import _is_prose, _split_sentences

_ID_COL_CANDIDATES = (
    "dato", "date", "id", "name", "title", "categories", "category",
    "what feeling", "feeling", "protocols", "protocol", "event",
)

_CLEAN_QNAME = re.compile(r"[^A-Za-z0-9_\-.]+")


def _sanitize_qname(text: str) -> str:
    cleaned = _CLEAN_QNAME.sub("_", text).strip("_")
    return cleaned[:40] if cleaned else "item"


def _is_numeric(text: str) -> bool:
    cleaned = text.strip().replace(" ", "").replace(",", ".")
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def _detect_delimiter(first_line: str) -> str:
    if "\t" in first_line and "," not in first_line:
        return "\t"
    if ";" in first_line and "," not in first_line:
        return ";"
    return ","


def parse(src: str, rel_path: str = "") -> dict[str, Any]:
    lines = src.splitlines()
    if not lines:
        return {"file_doc": "", "nodes": [], "refs": [], "problems": []}

    first_line = lines[0]
    delimiter = _detect_delimiter(first_line)

    nodes: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    problems: list[str] = []

    try:
        reader = csv.DictReader(io.StringIO(src), delimiter=delimiter)
        raw_fieldnames = reader.fieldnames or []
    except Exception as exc:
        return {
            "file_doc": f"Unparseable CSV: {exc}",
            "nodes": [],
            "refs": [],
            "problems": [f"CSV parse error: {exc}"],
        }

    # Clean field names: remove empty or whitespace-only headers
    clean_fieldnames: list[str] = []
    seen_headers: set[str] = set()
    for idx, f in enumerate(raw_fieldnames):
        header = (f or "").strip()
        if not header:
            continue
        if header in seen_headers:
            header = f"{header}_{idx}"
        seen_headers.add(header)
        clean_fieldnames.append(header)

    id_col = None
    for candidate in _ID_COL_CANDIDATES:
        for f in clean_fieldnames:
            if f.lower() == candidate:
                id_col = f
                break
        if id_col:
            break

    # Line tracking: reader.line_num gives the line where each row ends
    prev_line = reader.line_num
    capped = False
    max_nodes = getattr(config, "MAX_JSON_NODES", 50000)

    for row_idx, row in enumerate(reader, start=1):
        line_start = prev_line + 1
        line_end = reader.line_num
        prev_line = line_end

        if len(nodes) >= max_nodes:
            if not capped:
                capped = True
                problems.append(
                    f"node cap reached ({max_nodes} nodes); the rest of this CSV was not indexed"
                )
            break

        # Pick row identity
        row_id_val = (row.get(id_col) or "").strip() if id_col else ""
        if not row_id_val:
            for f in clean_fieldnames:
                v = (row.get(f) or "").strip()
                if v and not _is_numeric(v) and len(v) <= 50:
                    row_id_val = v
                    break

        row_display_id = row_id_val if row_id_val else f"row_{row_idx:04d}"
        row_qname = f"r{row_idx:04d}_{_sanitize_qname(row_display_id)}"

        # Build row docstring containing all non-empty fields
        row_summary_parts: list[str] = []
        for f in clean_fieldnames:
            val = (row.get(f) or "").strip()
            if val:
                row_summary_parts.append(f"{f}: {val}")

        row_doc = "\n".join(row_summary_parts).strip()
        raw_source = "\n".join(lines[max(0, line_start - 1):line_end]) if line_start <= len(lines) else ""

        # 1. Row record node
        nodes.append({
            "qname": row_qname,
            "kind": "record",
            "name": f"[{row_display_id}]",
            "parent": None,
            "signature": f"CSV row {row_idx}: {row_display_id}",
            "docstring": row_doc,
            "source": raw_source,
            "line_start": line_start,
            "line_end": line_end,
        })

        # 2. Process column values
        for f in clean_fieldnames:
            val = (row.get(f) or "").strip()
            if not val:
                continue

            # Pure numbers remain in the row record docstring; no separate child tag needed
            if _is_numeric(val):
                continue

            field_qname = f"{row_qname}.{_sanitize_qname(f)}"
            signature = f"{f} of {row_display_id}"

            if _is_prose(val):
                nodes.append({
                    "qname": field_qname,
                    "kind": "section",
                    "name": f,
                    "parent": row_qname,
                    "signature": signature,
                    "docstring": val,
                    "source": "",
                    "line_start": line_start,
                    "line_end": line_end,
                })
                # Sentence splitting for longer prose
                if len(val) >= 120:
                    sentences = _split_sentences(val)
                    if len(sentences) > 1:
                        for s_idx, sentence in enumerate(sentences, 1):
                            nodes.append({
                                "qname": f"{field_qname}::s{s_idx}",
                                "kind": "section",
                                "name": f"{f} sentence {s_idx}",
                                "parent": field_qname,
                                "signature": signature,
                                "docstring": sentence,
                                "source": "",
                                "line_start": line_start,
                                "line_end": line_end,
                            })
            else:
                nodes.append({
                    "qname": field_qname,
                    "kind": "tag",
                    "name": f"{f}: {val}",
                    "parent": row_qname,
                    "signature": signature,
                    "docstring": val,
                    "source": "",
                    "line_start": line_start,
                    "line_end": line_end,
                })

    header_summary = ", ".join(clean_fieldnames[:8])
    if len(clean_fieldnames) > 8:
        header_summary += f", ... (+{len(clean_fieldnames) - 8} more)"

    file_doc = f"CSV dataset with {len(clean_fieldnames)} columns ({header_summary})"

    return {
        "file_doc": file_doc,
        "nodes": nodes,
        "refs": refs,
        "problems": problems,
    }
