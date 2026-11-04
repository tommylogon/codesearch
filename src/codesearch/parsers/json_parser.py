"""JSON document, scenario, library, and AI Studio transcript extractor.

Every JSON key path becomes a searchable row. Nothing is truncated.

Three node kinds, each with one job:

- `section` - a prose value (a description, a prompt, a message). Embedded in the
  `function_name` and `comment` vector indexes. A long prose value also gets
  sentence-level child nodes, so a semantic query can land on the exact sentence
  instead of a diluted blob.
- `tag` - a short scalar value (a tag, a label, a number, a boolean). Keyword only:
  it is in the FTS index and carries no vector, so a one-word vector can never
  outrank real prose.
- `record` - a dict or a list. Its docstring is the *shape* of the record (child keys
  and their types, never their values), so it answers "which record has a `scent`
  field?" without duplicating the text of its children and without polluting the
  vector index with a bag of field names.

References (`target`, `way_id`, `area_from`, ...) become `CONNECTS_TO` refs, resolved
against the records of the same file.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .. import config

# Keys whose string value points at another entity in the same document.
REFERENCE_KEYS = {
    "target", "way_id", "return_dir", "area_from", "area_to", "to_area", "from_area",
    "destination", "source", "world_scope_id", "scope_id", "item_id", "character_id",
    "player_id", "parent_id", "owner_id",
}

# Abbreviations that end in a period but do not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "fig", "no", "inc", "ltd", "co", "al", "approx", "dept", "est",
}

_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*")
_TRAILING_WORD = re.compile(r"([A-Za-z]+)$")
_SENTENCE_TAIL = re.compile(r"[.!?][\"')\]]*$")

# A prose value longer than this gets sentence-level children as well.
SENTENCE_SPLIT_MIN_CHARS = 120


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


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentences.

    A boundary is sentence-ending punctuation followed by whitespace and a capital,
    digit or quote. Abbreviations, initials and decimals do not split.
    """
    stripped = text.strip()
    if not stripped:
        return []
    out: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(stripped):
        end = match.end()
        rest = stripped[end:]
        if not rest.strip():
            break
        nxt = rest.lstrip()
        if not (nxt[:1].isupper() or nxt[:1].isdigit() or nxt[:1] in "\"'([“‘"):
            continue
        before = stripped[start:match.start()]
        word_match = _TRAILING_WORD.search(before)
        if word_match:
            word = word_match.group(1)
            if word.lower() in _ABBREVIATIONS:
                continue
            if len(word) == 1 and word.isupper():
                continue
        chunk = stripped[start:end].strip()
        if chunk:
            out.append(chunk)
        start = end
    tail = stripped[start:].strip()
    if tail:
        out.append(tail)
    return out or [stripped]


def _summarize_dict(obj: dict[str, Any]) -> str:
    """Build a readable multi-line docstring from a JSON object. No value is cut."""
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
        parts.append("traits: " + ", ".join(f"{k}: {v}" for k, v in traits.items() if v))
    elif isinstance(traits, list) and traits:
        parts.append("traits: " + ", ".join(str(t) for t in traits))
    goals = obj.get("goals")
    if isinstance(goals, list) and goals:
        g_lines = []
        for g in goals:
            if isinstance(g, dict):
                t = g.get("title") or g.get("name") or g.get("G") or ""
                if t:
                    g_lines.append(f"- {t}")
            elif isinstance(g, str):
                g_lines.append(f"- {g}")
        if g_lines:
            parts.append("goals:\n" + "\n".join(g_lines))
    if not parts:
        # Fallback: every scalar field, in full.
        for k, v in obj.items():
            if isinstance(v, (str, int, float, bool)) and str(v).strip():
                parts.append(f"{k}: {v}")
    return "\n".join(parts).strip()


def _scalar_text(value: Any) -> str:
    """Text of a JSON scalar. `None` becomes the literal `null` so it stays visible."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "text"
    if isinstance(value, list):
        return "list"
    return "object"


def _is_prose(text: str) -> bool:
    """True for values worth a dedicated, embedded node.

    Prose is a sentence, a paragraph, or a phrase long enough to carry meaning.
    A label ("Left Arm"), a tag ("Love") or a number is not prose.
    """
    if _SENTENCE_TAIL.search(text):
        return True
    if len(text.split()) >= 5:
        return True
    return len(text) >= 40


def _identity(value: dict[str, Any], inherited: str) -> str:
    """The human name of a record: its own name/title/id, else the inherited one."""
    for key in ("name", "title", "id"):
        val = value.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return inherited


class _Walker:
    """Walks a JSON document, emitting one node per key path."""

    def __init__(self, lines: list[str], rel_path: str):
        self.lines = lines
        self.rel_path = rel_path
        self.nodes: list[dict] = []
        self.refs: list[dict] = []
        self.problems: list[str] = []
        self.cursor = 0
        self.capped = False

    # -- helpers
    def _line(self, probe: str) -> int:
        ls = _find_line(self.lines, probe, self.cursor)
        self.cursor = max(self.cursor, ls - 1)
        return ls

    def _add(self, kind: str, qname: str, name: str, signature: str, doc: str,
             parent: str | None, ls: int) -> None:
        self.nodes.append({
            "qname": qname,
            "kind": kind,
            "name": name,
            "parent": parent,
            "signature": signature,
            "docstring": doc,
            "source": "",
            "line_start": ls,
            "line_end": ls,
        })

    @staticmethod
    def _signature(key: str, owner: str) -> str:
        return f"{key} of {owner}" if owner else key

    def _check_cap(self) -> bool:
        if len(self.nodes) >= config.MAX_JSON_NODES:
            if not self.capped:
                self.capped = True
                self.problems.append(
                    f"node cap reached ({config.MAX_JSON_NODES} nodes); "
                    "the rest of this document was not indexed")
            return True
        return False

    # -- walk
    def walk(self, value: Any, path: str, key: str, owner: str,
             parent: str | None, depth: int) -> None:
        if self._check_cap():
            return
        if depth > config.MAX_JSON_DEPTH:
            self.problems.append(f"depth cap reached at {path}")
            return
        if isinstance(value, dict):
            self._walk_dict(value, path, key, owner, parent, depth)
        elif isinstance(value, list):
            self._walk_list(value, path, key, owner, parent, depth)
        else:
            self._leaf(value, path, key, owner, parent)

    def _walk_dict(self, value: dict[str, Any], path: str, key: str, owner: str,
                   parent: str | None, depth: int) -> None:
        shape: list[str] = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                unit = "items" if isinstance(v, list) else "fields"
                shape.append(f"{k}: {_type_name(v)} ({len(v)} {unit})")
            else:
                shape.append(f"{k}: {_type_name(v)}")
        ls = self._line(f'"{key}"')
        self._add("record", path, _identity(value, key), self._signature(key, owner),
                  "\n".join(shape), parent, ls)
        child_owner = _identity(value, owner)
        for k, v in value.items():
            child_path = f"{path}::{k}"
            if isinstance(v, (dict, list)):
                self.walk(v, child_path, k, child_owner, path, depth + 1)
            else:
                self._leaf(v, child_path, k, child_owner, path)
            if k in REFERENCE_KEYS and isinstance(v, str) and v.strip():
                self.refs.append({
                    "source": path,
                    "kind": "CONNECTS_TO",
                    "target": v.strip(),
                    "qualifier": k,
                    "meta": "",
                })

    def _walk_list(self, value: list[Any], path: str, key: str, owner: str,
                   parent: str | None, depth: int) -> None:
        shape = [f"{len(value)} items"]
        if value and isinstance(value[0], dict):
            shape.append("item fields: " + ", ".join(str(k) for k in value[0]))
        ls = self._line(f'"{key}"')
        self._add("record", path, key, self._signature(key, owner),
                  "\n".join(shape), parent, ls)
        for i, item in enumerate(value, 1):
            child_path = f"{path}::{i}"
            if isinstance(item, (dict, list)):
                self.walk(item, child_path, str(i), owner, path, depth + 1)
            else:
                self._leaf(item, child_path, str(i), owner, path)

    def _leaf(self, value: Any, path: str, key: str, owner: str,
              parent: str | None) -> None:
        text = _scalar_text(value)
        signature = self._signature(key, owner)
        if value is None:
            ls = self._line(f'"{key}"')
            self._add("tag", path, f"{key}: null", signature, "null", parent, ls)
            return
        if value == "":
            ls = self._line(f'"{key}"')
            self._add("tag", path, f"{key}: (empty)", signature, "", parent, ls)
            return
        if not _is_prose(text):
            ls = self._line(json.dumps(text[:60], ensure_ascii=False)[1:-1])
            self._add("tag", path, f"{key}: {text}", signature, text, parent, ls)
            return
        ls = self._line(json.dumps(text[:60], ensure_ascii=False)[1:-1])
        self._add("section", path, key, signature, text, parent, ls)
        if len(text) >= SENTENCE_SPLIT_MIN_CHARS:
            sentences = _split_sentences(text)
            if len(sentences) > 1:
                for i, sentence in enumerate(sentences, 1):
                    self._add("section", f"{path}::s{i}", f"{key} sentence {i}",
                              signature, sentence, path, ls)


def _file_doc(data: Any) -> str:
    """The file-level summary: what this document is about."""
    parts: list[str] = []
    if isinstance(data, dict):
        for key in ("_scenario_name", "name", "title"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(f"{key}: {val.strip()}")
        for key in ("world_lore", "scene_context"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(f"{key}: {val.strip()}")
            elif isinstance(val, list):
                joined = "\n".join(str(x) for x in val if x is not None).strip()
                if joined:
                    parts.append(f"{key}: {joined}")
        nested = data.get("data")
        target = nested if isinstance(nested, dict) else data
        summary = _summarize_dict(target)
        if summary:
            parts.append(summary)
        for collection in ("areas", "rooms", "players", "items", "ways", "characters"):
            val = data.get(collection)
            if isinstance(val, dict) and val:
                parts.append(f"{collection}: " + ", ".join(str(k) for k in val))
    elif isinstance(data, list):
        parts.append(f"{len(data)} items")
    return "\n\n".join(p for p in parts if p).strip()


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
    for ch in chunks:
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
        probe = json.dumps(text[:60], ensure_ascii=False)[1:-1]
        ls = _find_line(lines, probe, cursor)
        if ls >= cursor:
            cursor = ls
        nodes.append({
            "qname": f"turn_{turn_idx:03d}_{role}",
            "kind": "section",
            "name": f"[{role} #{turn_idx}]",
            "parent": None,
            "signature": f"{role} turn {turn_idx}",
            "docstring": text,
            "source": "",
            "line_start": ls,
            "line_end": ls,
        })
    return {"file_doc": f"AI transcript ({turn_idx} turns)", "nodes": nodes,
            "refs": refs, "problems": []}


def parse(src: str, rel_path: str = "") -> dict:
    """Parse a JSON file into searchable rows, one per key path."""
    lines = src.splitlines()
    try:
        data = json.loads(src)
    except Exception as exc:
        return {"file_doc": f"invalid json: {exc}", "nodes": [], "refs": [],
                "problems": [f"invalid json: {exc}"]}

    if isinstance(data, dict) and "chunkedPrompt" in data and isinstance(data["chunkedPrompt"], dict):
        return _parse_chat_transcript(data, lines)

    stem = rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0] if rel_path else "root"
    walker = _Walker(lines, rel_path)
    walker.walk(data, stem, stem, "", None, 0)
    return {"file_doc": _file_doc(data), "nodes": walker.nodes, "refs": walker.refs,
            "problems": walker.problems}
