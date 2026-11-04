"""Parser tests for tools/codesearch: each extractor on a small fixture.

Run: python -m pytest tests/test_codesearch_parsers.py
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from codesearch import embedder                                   # noqa: E402
from codesearch.parsers import html_parser, js_bridge, md_parser, py_parser   # noqa: E402
from codesearch.parsers import json_parser                       # noqa: E402

PY = '''"""Module doc."""
import os
from .sibling import helper

LIMIT: int = 3

# explains the ledger
# second line
ledger = {}


class Base:
    pass


class World(Base):
    """World doc."""

    def __init__(self):
        self.presence = {}

    def record(self, who, where="x"):
        """Record someone."""
        self.presence[who] = where        # mutation through subscript
        self.helper_call()
        helper(who)
        total = LIMIT                      # reads module variable
        local = 1
        return total

    def helper_call(self):
        return self.presence.get("a")


def free(x):
    global ledger
    ledger = {}
    LIMIT = 9
    return LIMIT
'''


def kinds(res):
    return {(n["kind"], n["qname"]) for n in res["nodes"]}


def refs(res, **kw):
    return [r for r in res["refs"] if all(r[k] == v for k, v in kw.items())]


def test_python_structure():
    res = py_parser.parse(PY)
    k = kinds(res)
    assert res["file_doc"] == "Module doc."
    assert ("class", "World") in k and ("method", "World::record") in k and ("function", "free") in k
    assert ("variable", "LIMIT") in k and ("variable", "ledger") in k
    assert ("variable", "World::presence") in k          # instance attribute from self.presence = ...
    rec = next(n for n in res["nodes"] if n["qname"] == "World::record")
    assert rec["docstring"] == "Record someone."
    assert rec["signature"] == "def record(self, who, where = 'x')"


def test_python_calls_refs_and_modes():
    res = py_parser.parse(PY)
    assert refs(res, source="World::record", kind="CALLS", target="helper_call", qualifier="self")
    assert refs(res, source="World::record", kind="CALLS", target="helper", qualifier="")
    assert refs(res, source="World", kind="INHERITS", target="Base")
    # subscript store marks a WRITE of the attribute, not a read
    assert refs(res, source="World::record", kind="REFERENCES", target="presence", meta="w")
    assert not refs(res, source="World::record", kind="REFERENCES", target="presence", meta="r")
    # module variable read; a local of the same name is NOT a reference to the module variable
    assert refs(res, source="World::record", kind="REFERENCES", target="LIMIT", meta="r", qualifier="module")
    assert not refs(res, source="free", kind="REFERENCES", target="LIMIT")
    # `global` makes a write count
    assert refs(res, source="free", kind="REFERENCES", target="ledger", meta="w")
    imp = refs(res, kind="IMPORTS")
    assert {(r["target"], r["qualifier"]) for r in imp} == {("os", "0"), ("sibling", "1")}


def test_python_comment_blocks():
    res = py_parser.parse(PY)
    comments = [n for n in res["nodes"] if n["kind"] == "comment"]
    block = next(c for c in comments if "explains" in c["docstring"])
    assert block["docstring"] == "explains the ledger\nsecond line"
    assert block["line_end"] - block["line_start"] == 1
    assert any("mutation through subscript" in c["docstring"] for c in comments)


def test_markdown_sections_blocks_links():
    src = ("---\ntitle: x\n---\n# Top\nintro [file](./a.py) and [[World Building/Graph System]]\n"
           "## Sub\ntext\n```python\nprint(1)\n```\n## Sub\nagain\n")
    res = md_parser.parse(src)
    assert res["file_doc"] == "title: x"
    secs = [n for n in res["nodes"] if n["kind"] == "section"]
    assert [s["qname"] for s in secs][:2] == ["Top", "Top::Sub"]
    assert len({s["qname"] for s in secs}) == len(secs)          # repeated heading stays unique
    assert any(n["kind"] == "block" and n["source"] == "print(1)" for n in res["nodes"])
    links = refs(res, kind="LINKS_TO")
    assert {(r["target"], r["qualifier"]) for r in links} == {("./a.py", "path"), ("World Building/Graph System", "wiki")}


def test_markdown_keeps_full_section_text():
    body = "\n".join(f"line {i}" for i in range(500))
    res = md_parser.parse("# H\n" + body)
    assert res["nodes"][0]["docstring"].count("\n") == 499      # nothing silently cut


def test_parse_text_splits_large_text_by_paragraphs():
    para1 = "\n".join([f"alpha paragraph line {i}" for i in range(50)])
    para2 = "\n".join([f"beta paragraph line {i}" for i in range(50)])
    res = md_parser.parse_text(para1 + "\n\n" + para2)
    assert len(res["nodes"]) == 2
    assert "alpha paragraph line 0" in res["nodes"][0]["docstring"]
    assert "beta paragraph line 0" in res["nodes"][1]["docstring"]


# ----------------------------------------------------------------- json
JSON_DOC = {
    "_scenario_name": "pines",
    "world_lore": "The valley keeps its own counsel.",
    "scene_context": "Rain on the tin roof of the apartment block.",
    "areas": {
        "apartment 1a": {
            "description": "A cramped flat above the laundromat. The window sticks.",
            "environment": {"scent": "clove cigarettes", "light": 55},
            "exits": {"east": {"target": "hallway 1", "way_id": "way_hall_1"}},
        },
        "hallway 1": {
            "description": "A corridor lit by a single failing bulb.",
            "environment": {"scent": "wet wool", "light": 30},
        },
    },
    "tags": ["Love", "Wholesome"],
    "empty": "",
    "nothing": None,
}


def json_res():
    return json_parser.parse(json.dumps(JSON_DOC, indent=2), "world.json")


def test_json_one_row_per_key_path():
    k = kinds(json_res())
    assert ("record", "world::areas") in k
    assert ("record", "world::areas::apartment 1a") in k
    assert ("section", "world::areas::apartment 1a::description") in k
    assert ("tag", "world::areas::apartment 1a::environment::scent") in k
    assert ("tag", "world::areas::apartment 1a::environment::light") in k
    assert ("record", "world::areas::apartment 1a::exits::east") in k
    assert ("tag", "world::tags::1") in k
    assert ("tag", "world::empty") in k
    assert ("tag", "world::nothing") in k


def test_json_record_docstring_is_shape_not_values():
    """A record says which fields it has. The values live in their own rows, so the
    record never becomes a bag of words that outranks prose."""
    rec = next(n for n in json_res()["nodes"] if n["qname"] == "world::areas::apartment 1a")
    assert "description: text" in rec["docstring"]
    assert "environment: object (2 fields)" in rec["docstring"]
    assert "A cramped flat" not in rec["docstring"]
    assert "clove cigarettes" not in rec["docstring"]


def test_json_short_values_are_tags_and_get_no_vector():
    res = json_res()
    scent = next(n for n in res["nodes"] if n["qname"].endswith("::scent"))
    assert scent["kind"] == "tag" and scent["docstring"] == "clove cigarettes"
    row = {"kind": scent["kind"], "name": scent["name"], "qualified_name": scent["qname"],
           "signature": scent["signature"], "docstring": scent["docstring"], "source": ""}
    assert embedder.part_texts(row) == {}          # keyword only: no vector pollution
    desc = next(n for n in res["nodes"] if n["qname"].endswith("::description"))
    row = {"kind": desc["kind"], "name": desc["name"], "qualified_name": desc["qname"],
           "signature": desc["signature"], "docstring": desc["docstring"], "source": ""}
    assert set(embedder.part_texts(row)) == {"function_name", "comment"}


def test_json_empty_and_null_values_stay_visible():
    res = json_res()
    empty = next(n for n in res["nodes"] if n["qname"] == "world::empty")
    nothing = next(n for n in res["nodes"] if n["qname"] == "world::nothing")
    assert empty["kind"] == "tag" and empty["name"] == "empty: (empty)"
    assert nothing["kind"] == "tag" and nothing["docstring"] == "null"


def test_json_reference_keys_become_connects_to_refs():
    res = json_res()
    got = {(r["target"], r["qualifier"]) for r in res["refs"] if r["kind"] == "CONNECTS_TO"}
    assert ("hallway 1", "target") in got
    assert ("way_hall_1", "way_id") in got
    assert all(r["source"] == "world::areas::apartment 1a::exits::east" for r in res["refs"])


def test_json_long_prose_gets_sentence_children():
    long_prose = ("The market district is a busy hub of trade. Vendors shout over one "
                   "another from every stall. The smell of bread and river mud mixes "
                   "with the smoke of the forge.")
    res = json_parser.parse(json.dumps({"lore": long_prose}), "world.json")
    parent = next(n for n in res["nodes"] if n["qname"] == "world::lore")
    assert parent["kind"] == "section" and parent["docstring"] == long_prose
    sentences = [n for n in res["nodes"] if n["qname"].startswith("world::lore::s")]
    assert len(sentences) == 3
    assert sentences[0]["docstring"].startswith("The market district")
    assert all(s["parent"] == "world::lore" for s in sentences)


def test_json_keeps_full_text():
    body = "\n".join(f"line {i} of the description" for i in range(200))
    res = json_parser.parse(json.dumps({"description": body}), "world.json")
    node = next(n for n in res["nodes"] if n["qname"] == "world::description")
    assert node["docstring"] == body          # nothing silently cut


def test_json_file_doc_carries_lore_and_scene():
    doc = json_res()["file_doc"]
    assert "world_lore: The valley keeps its own counsel." in doc
    assert "scene_context: Rain on the tin roof" in doc


def test_json_invalid_is_reported_not_raised():
    res = json_parser.parse("{not json", "world.json")
    assert res["nodes"] == [] and res["problems"]



def test_html_structure():
    src = ('<html><head><title>My Page</title><link rel="stylesheet" href="a.css"></head><body>'
           '<h1>Heading</h1><div id="panel" class="x"></div><script src="app.js"></script>'
           '<script>function boot() { return 1 }</script></body></html>')
    res = html_parser.parse(src)
    assert res["file_doc"] == "My Page"
    assert {r["target"] for r in refs(res, kind="IMPORTS")} == {"a.css", "app.js"}
    assert any(n["kind"] == "element" and n["name"] == "panel" for n in res["nodes"])
    assert any(n["kind"] == "section" and n["name"] == "Heading" for n in res["nodes"])
    assert len(res["scripts"]) == 1


needs_node = pytest.mark.skipif(not js_bridge.available(), reason="node not available")

TS = '''import { thing } from './thing.js';

/** Registry of widgets. */
const registry: Map<string, number> = new Map();

// batches the redraw
// so it runs once
export function sync(items: string[], flag = false): void {
  registry.set('a', 1);
  thing(items);
  helper();
}

class Net extends Base {
  private cache = {};
  constructor(name: string) { super(); this.total = 0; }
  /** Update in place. */
  update(x: number): number {
    this.total += x;
    this.cache.items.push(x);
    this.update2();
    return this.total;
  }
}

const arrow = (a: number) => a + 1;
'''


@needs_node
def test_typescript_extraction():
    out = js_bridge.parse_batch([{"id": "t", "text": TS, "ts": True, "offset": 0}])["t"]
    assert "error" not in out, out
    k = kinds(out)
    assert ("function", "sync") in k and ("class", "Net") in k and ("method", "Net::update") in k
    assert ("function", "arrow") in k and ("variable", "registry") in k
    assert ("variable", "Net::total") in k                       # this.total = ... in a method
    sync_node = next(n for n in out["nodes"] if n["qname"] == "sync")
    assert sync_node["signature"] == "function sync(items: string[], flag = false): void"
    assert sync_node["line_start"] == 8        # the `export function` line: stripping keeps 1:1 numbering
    upd = next(n for n in out["nodes"] if n["qname"] == "Net::update")
    assert upd["docstring"] == "Update in place."
    assert refs(out, source="Net", kind="INHERITS", target="Base")
    assert refs(out, source="sync", kind="CALLS", target="thing")
    # registry.set(...) mutates the Map it is called on: a write, not a read
    assert refs(out, source="sync", kind="REFERENCES", target="registry", meta="w")
    assert not refs(out, source="sync", kind="REFERENCES", target="registry", meta="r")
    assert refs(out, source="Net::update", kind="CALLS", target="update2", qualifier="self")
    # x.items.push(...) is a write of `items`; this.total += is read+write
    assert refs(out, source="Net::update", kind="REFERENCES", target="items", qualifier="attr", meta="w")
    assert refs(out, source="Net::update", kind="REFERENCES", target="total", meta="w")
    assert refs(out, kind="IMPORTS", target="./thing.js")
    comments = [n for n in out["nodes"] if n["kind"] == "comment"]
    assert any(c["docstring"] == "batches the redraw\nso it runs once" for c in comments)
    # JSDoc became a docstring, so it is not duplicated as a comment node
    assert not any("Registry of widgets" in c["docstring"] for c in comments)
    assert next(n for n in out["nodes"] if n["qname"] == "registry")["docstring"] == "Registry of widgets."


@needs_node
def test_ts_parameter_property_is_reported_not_silently_reformatted():
    """Node's strip mode cannot handle parameter properties. The extractor must report an
    error rather than fall back to `transform`, which would shift every line number."""
    out = js_bridge.parse_batch([{"id": "p", "ts": True, "offset": 0,
                                  "text": "class A {\n  constructor(public x: number) {}\n}\n"}])["p"]
    assert "error" in out and "strip-only" in out["error"]


@needs_node
def test_js_inline_script_offset_and_errors():
    out = js_bridge.parse_batch([
        {"id": "ok", "text": "function boot() {}\n", "ts": False, "offset": 10},
        {"id": "bad", "text": "function (", "ts": False, "offset": 0},
    ])
    assert out["ok"]["nodes"][0]["line_start"] == 11
    assert "error" in out["bad"]


@needs_node
def test_ts_object_module_and_iife_extraction():
    src = """
(window as unknown as { GraphNetwork: any }).GraphNetwork = {
  syncGraphData(next: any) {
    GraphNetwork._applyNodeLabelDecorations(next);
  },
  _applyNodeLabelDecorations(rows: any) { return rows; }
};
window.TurnQueue = (() => {
  function initialize() { return advance(); }
  function advance() { return 1; }
  return { initialize, advance };
})();
"""
    out = js_bridge.parse_batch([{"id": "m", "text": src, "ts": True, "offset": 0}])["m"]
    assert "error" not in out, out
    k = kinds(out)
    assert ("class", "GraphNetwork") in k
    assert ("method", "GraphNetwork::syncGraphData") in k
    assert ("method", "GraphNetwork::_applyNodeLabelDecorations") in k
    assert ("class", "TurnQueue") in k
    assert ("method", "TurnQueue::initialize") in k
    assert ("method", "TurnQueue::advance") in k
    assert refs(out, source="GraphNetwork::syncGraphData", kind="CALLS",
                target="_applyNodeLabelDecorations", qualifier="self")
    assert refs(out, source="TurnQueue::initialize", kind="CALLS", target="advance")

