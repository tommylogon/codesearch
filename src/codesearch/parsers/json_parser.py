"""JSON document, scenario, library, and AI Studio transcript extractor.

Extracts structured `section` nodes from JSON files so that:
- AI Studio / chat transcripts (`chunkedPrompt.chunks`) become per-turn `section` nodes
  carrying the full turn text (never truncated).
- VirtualWorld scenario files (`graph.nodes` + `players` + `world_lore`) expose each
  authored area/item/way/character node and player profile as a searchable `section`
  with its description, personality, tags, and goals.
- Library definitions (`data/library/**/*.json`, character arc lists, single objects)
  expose their `name`, `description`, `personality`, `tags`, and nested entries.
"""
from __future__ import annotations

import json
import re
from typing import Any


def _find_line(lines: list[str], needle: str, start_idx: int = 0) -> int:
    """Locate 1-based line number of `needle` at or after `start_idx` (0-based)."""
    if not needle:
        return max(1, start_idx + 1)
    for i in range(max(0, start_idx), len(lines)):
        if needle in lines[i]:
            return i + 1
    for i in range(0, min(len(lines), max(0, start_idx))):
        if needle in lines[i]:
            return i + 1
    return max(1, start_idx + 1)


def _summarize_dict(obj: dict[str, Any]) -> str:
    """Build a readable multi-line docstring from a JSON object without truncating prose."""
    parts: list[str] = []
    for key in ("title", "name", "id", "type", "kind", "species", "life_stage"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(f"{key}: {val.strip()}")
    for key in ("description", "base_description", "personality", "overview", "scent",
                "summary", "text", "content", "prompt", "G", "I", "M", "P",
                "actions", "success_message", "fail_message"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(f"{key}: {val.strip()}")
    props = obj.get("properties")
    if isinstance(props, dict):
        for key in ("description", "base_description", "personality", "actions"):
            val = props.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(f"{key}: {val.strip()}")
        tags = props.get("tags")
        if isinstance(tags, list) and tags:
            parts.append("tags: " + ", ".join(str(t) for t in tags))
    tags = obj.get("tags")
    if isinstance(tags, list) and tags:
        parts.append("tags: " + ", ".join(str(t) for t in tags))
    traits = obj.get("traits")
    if isinstance(traits, dict) and traits:
        parts.append("traits: " + ", ".join(str(k) for k, v in traits.items() if v))
    elif isinstance(traits, list) and traits:
        parts.append("traits: " + ", ".join(str(t) for t in traits))
    goals = obj.get("goals")
    if isinstance(goals, list) and goals:
        g_lines = []
        for g in goals[:20]:
            if isinstance(g, dict):
                t = g.get("title") or g.get("name") or g.get("G") or ""
                if t:
                    g_lines.append(f"- {t}")
            elif isinstance(g, str):
                g_lines.append(f"- {g}")
        if g_lines:
            parts.append("goals:\n" + "\n".join(g_lines))
    if not parts:
        # Fallback: include compact scalar fields
        for k, v in list(obj.items())[:25]:
            if isinstance(v, (str, int, float, bool)) and str(v).strip():
                parts.append(f"{k}: {v}")
    return "\n".join(parts).strip()


def _parse_chat_transcript(data: dict[str, Any], lines: list[str]) -> dict:
    """Parse an AI Studio / Gemini `chunkedPrompt` JSON export into turn sections."""
    nodes: list[dict] = []
    refs: list[dict] = []
    sys_inst = data.get("systemInstruction")
    cursor = 0
    if isinstance(sys_inst, dict):
        sys_text = sys_inst.get("text") or ""
        if not sys_text and isinstance(sys_inst.get("parts"), list):
            sys_text = "\n".join(
                p.get("text", "") for p in sys_inst["parts"] if isinstance(p, dict)
            ).strip()
        if sys_text:
            ls = _find_line(lines, '"systemInstruction"', 0)
            nodes.append({
                "qname": "systemInstruction",
                "kind": "section",
                "name": "systemInstruction",
                "parent": None,
                "signature": "systemInstruction",
                "docstring": sys_text,
                "source": "",
                "line_start": ls,
                "line_end": ls,
            })
            cursor = ls

    chunks = data.get("chunkedPrompt", {}).get("chunks", [])
    turn_idx = 0
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, dict):
            continue
        if ch.get("isThought"):
            continue
        text = (ch.get("text") or "").strip()
        if not text and isinstance(ch.get("parts"), list):
            text = "\n".join(
                p.get("text", "") for p in ch["parts"]
                if isinstance(p, dict) and not p.get("thought")
            ).strip()
        if not text:
            continue
        turn_idx += 1
        role = ch.get("role") or "turn"
        first_line = next((ln.strip().lstrip("#").strip() for ln in text.splitlines() if ln.strip()), "")
        snippet = first_line[:64] if first_line else f"turn {turn_idx}"
        qname = f"turn_{turn_idx:03d}_{role}"
        # Locate approximate line in the raw JSON
        probe = json.dumps(text[:36])[1:-1]
        ls = _find_line(lines, probe, cursor)
        if ls >= cursor:
            cursor = ls
        nodes.append({
            "qname": qname,
            "kind": "section",
            "name": f"[{role} #{turn_idx}] {snippet}",
            "parent": None,
            "signature": f"{role} turn {turn_idx}",
            "docstring": text,
            "source": "",
            "line_start": ls,
            "line_end": ls,
        })
    return {"file_doc": f"AI transcript ({turn_idx} turns)", "nodes": nodes, "refs": refs}


def _to_text(val: Any) -> str:
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        return "\n".join(str(x) for x in val if x is not None).strip()
    if isinstance(val, dict):
        return "\n".join(f"{k}: {v}" for k, v in val.items() if v is not None).strip()
    return str(val).strip() if val is not None else ""


def _parse_scenario(data: dict[str, Any], lines: list[str]) -> dict:
    """Parse a VirtualWorld scenario JSON (`graph.nodes` + `players` + `world_lore`)."""
    nodes: list[dict] = []
    refs: list[dict] = []
    scen_name = _to_text(data.get("_scenario_name"))
    lore = _to_text(data.get("world_lore"))
    scenes = _to_text(data.get("scene_context"))
    file_doc = "\n\n".join(x for x in (scen_name, lore, scenes) if x)

    cursor = 0
    players = data.get("players")
    if isinstance(players, dict):
        for pname, pdata in players.items():
            if not isinstance(pdata, dict):
                continue
            ls = _find_line(lines, f'"{pname}"', cursor)
            cursor = max(cursor, ls - 1)
            doc = _summarize_dict(pdata)
            qname = f"players::{pname}"
            nodes.append({
                "qname": qname,
                "kind": "section",
                "name": pname,
                "parent": None,
                "signature": f"player {pname}",
                "docstring": doc,
                "source": "",
                "line_start": ls,
                "line_end": ls,
            })

    g_nodes = data.get("graph", {}).get("nodes")
    items_iter = []
    if isinstance(g_nodes, dict):
        items_iter = list(g_nodes.items())
    elif isinstance(g_nodes, list):
        items_iter = [(n.get("id") or n.get("name") or f"node_{i}", n)
                      for i, n in enumerate(g_nodes) if isinstance(n, dict)]

    cursor = 0
    for nid, ndata in items_iter:
        if not isinstance(ndata, dict):
            continue
        ntype = ndata.get("type") or "node"
        nname = ndata.get("name") or str(nid)
        ls = _find_line(lines, f'"{nid}"', cursor)
        cursor = max(cursor, ls - 1)
        doc = _summarize_dict(ndata)
        qname = f"graph::{ntype}::{nid}"
        nodes.append({
            "qname": qname,
            "kind": "section",
            "name": nname,
            "parent": None,
            "signature": f"{ntype} {nid}",
            "docstring": doc,
            "source": "",
            "line_start": ls,
            "line_end": ls,
        })

    return {"file_doc": file_doc, "nodes": nodes, "refs": refs}


def _is_wordy(text: str) -> bool:
    """True for multi-word / prose-ish values worth a dedicated node."""
    return len(text.split()) >= 2 or len(text) >= 20


def _walk_json(value: Any, path: str, nodes: list[dict], lines: list[str],
               cursor: list[int], wordy_parts: list[str], depth: int = 0) -> None:
    """Recursively turn a JSON value into section nodes.

    - Multi-word string values become their own section (keyword + vector target).
    - Single-word scalars accumulate into their parent's doc (keyword-only, still
      searchable but not vector-polluted).
    - Containers get a summary section of their leaf text so topic queries can land
      on them even when every child is a short scalar (e.g. tag lists).
    - Keys are folded into qname ("file::data::description") so FTS catches them.
    """
    if len(nodes) >= 400 or depth > 8:
        return
    indent = "  " * depth + "- "

    if isinstance(value, dict):
        keys_doc = []
        for k, v in value.items():
            child_path = f"{path}::{k}" if path else str(k)
            if isinstance(v, (dict, list)):
                _walk_json(v, child_path, nodes, lines, cursor, wordy_parts, depth + 1)
            else:
                text = _to_text(v)
                if _is_wordy(text):
                    child_parts = []
                    _walk_json(text, child_path, nodes, lines, cursor, child_parts, depth + 1)
                elif text:
                    keys_doc.append(f"{k}: {text}")
                else:
                    keys_doc.append(f"{k}: (empty)")
        if keys_doc:
            ls = _find_line(lines, f'"{path.rsplit("::", 1)[-1]}"', cursor[0])
            cursor[0] = max(cursor[0], ls - 1)
            nodes.append({
                "qname": path or "(root)",
                "kind": "section",
                "name": path.rsplit("::", 1)[-1] or "(root)",
                "parent": None,
                "signature": f"json {path}",
                "docstring": "\n".join(keys_doc)[:4000],
                "source": "",
                "line_start": ls,
                "line_end": ls,
            })
        return

    if isinstance(value, list):
        items_doc = []
        for i, item in enumerate(value, 1):
            if isinstance(item, (dict, list)):
                _walk_json(item, f"{path}::{i}", nodes, lines, cursor, wordy_parts, depth + 1)
            else:
                text = _to_text(item)
                if _is_wordy(text):
                    _walk_json(text, f"{path}::{i}", nodes, lines, cursor, wordy_parts, depth + 1)
                elif text:
                    items_doc.append(f"[{i}] {text}")
        # Summary section so pure-short-item lists (tags, ids) are still one
        # searchable chunk instead of N one-word sections.
        if items_doc and len(items_doc) >= 2:
            ls = _find_line(lines, f'"{path.rsplit("::", 1)[-1]}"', cursor[0])
            cursor[0] = max(cursor[0], ls - 1)
            nodes.append({
                "qname": path or "(list)",
                "kind": "section",
                "name": path.rsplit("::", 1)[-1] or "list",
                "parent": None,
                "signature": f"json {path} ({len(items_doc)} items)",
                "docstring": "\n".join(items_doc)[:4000],
                "source": "",
                "line_start": ls,
                "line_end": ls,
            })
        return

    # Leaf string value reached only via recursion from dict/list branches.
    text = str(value).strip()
    if text:
        wordy_parts.append(text)
        ls = _find_line(lines, json.dumps(text[:36])[1:-1], cursor[0])
        cursor[0] = max(cursor[0], ls - 1)
        nodes.append({
            "qname": path,
            "kind": "section",
            "name": (path.rsplit("::", 1)[-1] + ": " + text.splitlines()[0][:48]).strip(),
            "parent": path.rsplit("::", 1)[0] if "::" in path else None,
            "signature": f"json {path}",
            "docstring": text,
            "source": "",
            "line_start": ls,
            "line_end": ls,
        })


def _parse_json_generic(data: dict[str, Any], lines: list[str], rel_path: str) -> dict:
    """Parse ANY json object/array into fine-grained, key-searchable sections."""
    stem = rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    nodes: list[dict] = []
    cursor = [0]
    wordy_parts: list[str] = []
    _walk_json(data, stem, nodes, lines, cursor, wordy_parts)
    # Prefer nested entity payloads (spec_v2 cards keep name/description under "data")
    # so the file-level embedding carries the entity name, not just scaffolding keys.
    file_doc_parts = ""
    target = data.get("data") if isinstance(data, dict) and isinstance(data.get("data"), dict) else data
    if isinstance(target, dict):
        file_doc_parts = _summarize_dict(target)
    if not file_doc_parts:
        file_doc_parts = _to_text(data) if not isinstance(data, (dict, list)) else _summarize_dict(data)
    return {"file_doc": file_doc_parts, "nodes": nodes, "refs": []}


def parse(src: str, rel_path: str = "") -> dict:
    """Parse a JSON file into searchable `section` nodes."""
    lines = src.splitlines()
    try:
        data = json.loads(src)
    except Exception as exc:
        return {"file_doc": f"invalid json: {exc}", "nodes": [], "refs": []}

    if isinstance(data, dict):
        if "chunkedPrompt" in data and isinstance(data["chunkedPrompt"], dict):
            return _parse_chat_transcript(data, lines)
        if "graph" in data and "players" in data and isinstance(data.get("graph"), dict):
            return _parse_scenario(data, lines)

        # Generic fallback: any dict / list JSON is split into fine-grained sections
        # (per key path and multi-word value) instead of one blob per file.
        return _parse_json_generic(data, lines, rel_path)

    return {"file_doc": "", "nodes": [], "refs": []}
