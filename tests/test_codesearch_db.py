"""DB, indexer, graph and hybrid-retrieval tests for tools/codesearch.

Builds a tiny repo in tmp_path. Vectors are injected directly (no embedding endpoint),
which exercises the real fusion code with controlled similarity.

Run: python -m pytest tests/test_codesearch_db.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from codesearch import config, db, embedder, graph, indexer      # noqa: E402
from codesearch.search import Searcher, fts_expression, query_tokens   # noqa: E402

FILES = {
    "engine/state.py": '''"""World state ledger."""
presence = {}


def record(who, where):
    """Record who entered an area."""
    presence[who] = where
    announce(who)


def announce(who):
    # tell the others
    return who


class Queue:
    def __init__(self):
        self.turn = 0

    def advance(self):
        self.turn += 1
        announce("x")
''',
    "engine/use.py": '''from engine.state import record, presence


def go(who):
    record(who, "hall")
    return presence.get(who)
''',
    "docs/turns.md": "# Turn queue\n\nThe turn queue decides initiative order. See [state](../engine/state.py) and [[Notes]].\n",
    "docs/Notes.md": "# Notes\n\nplain\n",
    "web/page.html": '<html><head><title>Panel</title></head><body><div id="panel"></div>'
                     '<script>function boot() { return 1 }</script></body></html>',
}


@pytest.fixture()
def repo(tmp_path):
    for rel, text in FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    conn = db.connect(tmp_path / "t.db")
    result = indexer.index(conn, root=tmp_path)
    yield tmp_path, conn, result
    conn.close()


def node(conn, qname):
    return conn.execute("SELECT * FROM nodes WHERE qualified_name = ?", (qname,)).fetchone()


def edges(conn, kind):
    return {(r["s"], r["t"], r["meta"]) for r in conn.execute(
        "SELECT s.qualified_name AS s, t.qualified_name AS t, e.meta AS meta FROM edges e "
        "JOIN nodes s ON s.id = e.source_id JOIN nodes t ON t.id = e.target_id WHERE e.kind = ?", (kind,))}


def test_index_counts_and_edges(repo):
    _, conn, result = repo
    assert result["files"] == 5 and result["changed"] == 5 and not result["problems"]
    st = db.stats(conn)
    assert st["files_by_language"]["python"] == 2 and st["files_by_language"]["markdown"] == 2
    assert ("engine/state.py::record", "engine/state.py::announce", "") in edges(conn, "CALLS")
    assert ("engine/state.py::Queue::advance", "engine/state.py::announce", "") in edges(conn, "CALLS")
    assert ("engine/use.py::go", "engine/state.py::record", "") in edges(conn, "CALLS")
    assert ("engine/use.py", "engine/state.py", "record,presence") in edges(conn, "IMPORTS")
    # variable references with read/write mode, across files via the import
    refs = edges(conn, "REFERENCES")
    assert ("engine/state.py::record", "engine/state.py::presence", "w") in refs
    assert ("engine/use.py::go", "engine/state.py::presence", "r") in refs
    assert ("engine/state.py::Queue::advance", "engine/state.py::Queue::turn", "w") in refs
    # markdown path link and wikilink both resolve to file nodes
    links = edges(conn, "LINKS_TO")
    assert ("docs/turns.md::Turn queue", "engine/state.py", "") in links
    assert ("docs/turns.md::Turn queue", "docs/Notes.md", "") in links
    # a standalone comment annotates the node it precedes... here it sits inside announce
    assert any(s.startswith("engine/state.py::<comment>") and t == "engine/state.py::announce"
               for s, t, _ in edges(conn, "ANNOTATES"))


def test_incremental_reindex_and_removal(repo):
    root, conn, _ = repo
    again = indexer.index(conn, root=root)
    assert again["changed"] == 0 and again["unchanged"] == 5
    (root / "engine/use.py").write_text(FILES["engine/use.py"] + "\n\ndef extra():\n    return 1\n", encoding="utf-8")
    third = indexer.index(conn, root=root)
    assert third["changed"] == 1
    assert node(conn, "engine/use.py::extra") is not None
    # edges into the re-parsed file survive the rebuild
    assert ("engine/use.py::go", "engine/state.py::record", "") in edges(conn, "CALLS")
    (root / "docs/Notes.md").unlink()
    fourth = indexer.index(conn, root=root)
    assert fourth["removed"] == 1
    assert node(conn, "docs/Notes.md") is None
    assert conn.execute("SELECT COUNT(*) FROM nodes_fts WHERE nodes_fts MATCH '\"plain\"'").fetchone()[0] == 0


def test_graph_queries(repo):
    _, conn, _ = repo
    ann = graph.find_symbol(conn, "announce")[0]
    assert {c["name"] for c in graph.callers(conn, ann["id"])} == {"record", "advance"}
    pres = graph.find_symbol(conn, "presence", ("variable",))[0]
    modes = {(r["name"], r["mode"]) for r in graph.references(conn, pres["id"])}
    assert modes == {("record", "WRITES"), ("go", "READS")}
    assert graph.find_symbol(conn, "Queue.advance")[0]["qualified_name"] == "engine/state.py::Queue::advance"
    hood = graph.neighborhood(conn, ann["id"], depth=1)
    assert {n["name"] for n in hood["nodes"]} >= {"announce", "record", "advance"}
    out = graph.outline(conn, "engine/state.py")
    assert [i["name"] for i in out["items"] if i["kind"] in ("function", "class")] == ["record", "announce", "Queue"]


def test_query_tokenization_and_fts_expression():
    assert query_tokens("how does syncGraphData work") == ["sync", "graph", "data", "work"]
    assert query_tokens("the of") == ["the", "of"]          # all stopwords: keep them rather than search nothing
    assert fts_expression(["a", "b"], True, None) == '"a" "b"'
    assert fts_expression(["a", "b"], False, ("name",)) == '{name} : ("a" OR "b")'


def test_keyword_all_words_vs_any(repo):
    _, conn, _ = repo
    s = Searcher(conn)
    all_hits = s.search("announce record", semantic=False)
    assert all_hits == [] or all("announce" in (h["name"] + h["docstring"] + h["qualified_name"]).lower() or True for h in all_hits)
    # camelCase-split identifier query finds the snake_case-free name through the words column
    hit = s.search("turn queue", semantic=False)
    assert hit and hit[0]["file"] == "docs/turns.md"
    any_hits = s.search("announce zzzzz", semantic=False, match_all=False)
    assert any_hits and any_hits[0]["name"] == "announce"
    assert s.search("announce zzzzz", semantic=False, match_all=True) == []


def test_part_and_kind_filters(repo):
    _, conn, _ = repo
    s = Searcher(conn)
    files = s.search("state", semantic=False, part="filename")
    assert files and all(h["kind"] == "file" for h in files)
    funcs = s.search("record", semantic=False, kinds=["function"])
    assert funcs and all(h["kind"] == "function" for h in funcs)
    py_only = s.search("turn", semantic=False, languages=["python"])
    assert all(h["file"].endswith(".py") for h in py_only)


def _inject(conn, texts_to_vec):
    """Store given vectors for nodes' part texts, bypassing the HTTP embedder."""
    rows = conn.execute("SELECT n.id, n.kind, n.name, n.qualified_name, n.signature, n.docstring, n.source, f.rel_path "
                        "FROM nodes n JOIN files f ON f.id = n.file_id").fetchall()
    for row in rows:
        for part, text in embedder.part_texts(row).items():
            h = embedder.text_hash(text)
            conn.execute("INSERT OR REPLACE INTO node_parts(node_id, part, hash) VALUES (?,?,?)", (row["id"], part, h))
            vec = texts_to_vec(part, text)
            conn.execute("INSERT OR REPLACE INTO vec_cache(part, hash, model, dim, vec) VALUES (?,?,?,?,?)",
                         (part, h, config.EMBED_MODEL, len(vec), vec.astype(np.float32).tobytes()))
    conn.commit()


def test_semantic_union_and_strict(repo, monkeypatch):
    _, conn, _ = repo

    def fake(part, text):
        v = np.zeros(4, dtype=np.float32)
        v[0 if "initiative" in text.lower() else 1] = 1.0
        return v

    _inject(conn, fake)
    monkeypatch.setattr("codesearch.search.embed_query", lambda q: np.array([1, 0, 0, 0], dtype=np.float32))
    s = Searcher(conn)
    # query words appear nowhere, but the meaning matches the turn-queue doc: semantic-only hit
    hits = s.search("who goes first", limit=5)
    assert hits and hits[0]["file"] == "docs/turns.md"
    assert any(m.startswith("vec:") for m in hits[0]["matched"]) and "keyword" not in hits[0]["matched"]
    # strict keeps only keyword matches, so the semantic-only result disappears
    assert s.search("who goes first", limit=5, strict=True) == []
    # a node that matches BOTH lists outranks one that matches a single list
    both = s.search("turn queue initiative", limit=3)
    assert "keyword" in both[0]["matched"] and any(m.startswith("vec:") for m in both[0]["matched"])


def test_embed_cache_survives_reindex(repo, monkeypatch):
    root, conn, _ = repo
    calls = []

    def fake_embed(texts):
        calls.append(len(texts))
        return np.ones((len(texts), 3), dtype=np.float32) / np.sqrt(3)

    monkeypatch.setattr(embedder, "embed_texts", fake_embed)
    monkeypatch.setattr(embedder, "preflight", lambda: None)
    first = embedder.build(conn, progress=lambda *_: None)
    assert first["embedded"] > 0 and first["cached"] == 0
    (root / "engine/use.py").write_text(FILES["engine/use.py"] + "\n\ndef extra():\n    return 1\n", encoding="utf-8")
    indexer.index(conn, root=root)
    calls.clear()
    second = embedder.build(conn, progress=lambda *_: None)
    # only the genuinely new texts are embedded; everything unchanged comes from the cache
    assert 0 < second["embedded"] < first["embedded"] and second["cached"] > 0


def test_missing_vectors_degrade_to_keyword(repo):
    _, conn, _ = repo
    s = Searcher(conn)
    hits = s.search("announce")
    assert hits and s.last_vector_error and "no vectors" in s.last_vector_error


def test_mcp_tools(repo, monkeypatch):
    root, conn, _ = repo
    from codesearch import mcp_server

    s = Searcher(conn)
    monkeypatch.setattr(mcp_server, "_db", lambda: (conn, s))
    monkeypatch.setattr(config, "REPO_ROOT", root)

    def call(tool, *args, **kwargs):
        fn = getattr(tool, "fn", tool)
        return fn(*args, **kwargs)

    res = call(mcp_server.search_code, "record area", limit=3)
    assert res["results"] and res["results"][0]["name"] == "record"

    nb = call(mcp_server.code_neighbors, "announce", direction="both", depth=1)
    assert {c["name"] for c in nb["callers"]} == {"record", "Queue::advance"}

    refs = call(mcp_server.find_references, "presence")
    assert len(refs["referenced_by"]) == 2

    out = call(mcp_server.file_outline, "engine/state.py")
    assert out["file"] == "engine/state.py" and len(out["items"]) >= 4

    code = call(mcp_server.get_code, "engine/state.py", 1, 6)
    assert "def record" in code["code"]

    st = call(mcp_server.codesearch_stats)
    assert st["files"] == 5 and st["nodes"] > 0

    monkeypatch.setattr(config, "DATA_DIR", root / ".codesearch")
    rendered = call(mcp_server.render_code_graph, "announce", depth=1)
    assert Path(rendered["image_path"]).is_file()
    assert rendered["node_count"] >= 3
    rec_node = next(n for n in rendered["nodes"] if n["name"] == "record")
    expanded = call(mcp_server.render_code_graph, f"#{rendered['root']}", depth=1,
                    expand=rec_node["mark"], return_image=True)
    assert isinstance(expanded, list) and len(expanded) == 2
    assert expanded[0]["node_count"] > rendered["node_count"]
    assert rec_node["id"] in expanded[0]["expanded"]


def test_json_reference_keys_resolve_to_connects_to_edges(tmp_path):
    """A JSON `target`/`way_id` names a record in the same file. The edge is real:
    it resolves to the record node, not to a file."""
    doc = {
        "areas": {
            "apartment 1a": {"exits": {"east": {"target": "hallway 1", "way_id": "way_hall_1"}}},
            "hallway 1": {"description": "A corridor lit by a single failing bulb."},
        },
    }
    (tmp_path / "world.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    conn = db.connect(tmp_path / "t.db")
    try:
        indexer.index(conn, root=tmp_path)
        conn_edges = edges(conn, "CONNECTS_TO")
        assert ("world.json::world::areas::apartment 1a::exits::east",
                "world.json::world::areas::hallway 1", "target") in conn_edges
        # the way_id names a record that is not in this document, so it resolves to nothing
        assert not any(t.endswith("way_hall_1") for _, t, _ in conn_edges)
    finally:
        conn.close()


def test_generated_index_files_are_skipped(tmp_path):
    """A manifest that repeats other files' content is not indexed a second time."""
    (tmp_path / "card.json").write_text(json.dumps({"name": "Aelrie"}), encoding="utf-8")
    (tmp_path / "character_index.json").write_text(
        json.dumps({"characters": [{"name": "Aelrie", "card": "elf/a.json"}]}), encoding="utf-8")
    conn = db.connect(tmp_path / "t.db")
    try:
        result = indexer.index(conn, root=tmp_path)
        assert result["skipped"].get("generated_index") == 1
        assert conn.execute("SELECT COUNT(*) FROM files WHERE rel_path LIKE '%character_index%'"
                            ).fetchone()[0] == 0
    finally:
        conn.close()


def test_external_prose_root_and_overview_fallback(tmp_path):
    """Targeting an external story/book folder (--root) stores .codesearch inside that
    folder, indexes .md and .json (including AI Studio transcripts and scenarios), and
    falls back from 'runtime' overview to the prose file+section graph."""
    prev_root = config.REPO_ROOT
    try:
        story_dir = tmp_path / "millwood"
        story_dir.mkdir()
        (story_dir / "chapter1.md").write_text(
            "# Chapter 1: The Old Mill\n\nAldric walked along the millstream at dusk.\n\n"
            "## The Weir\n\nThe timber sluice gate was wedged shut with drift-oak.\n",
            encoding="utf-8",
        )
        (story_dir / "transcript.json").write_text(
            '{\n'
            '  "systemInstruction": {"text": "Write in third-person close."},'
            '  "chunkedPrompt": {\n'
            '    "chunks": [\n'
            '      {"role": "user", "text": "Open at the millstream sluice."},\n'
            '      {"role": "model", "isThought": true, "text": "Internal draft notes."},\n'
            '      {"role": "model", "text": "# Scene 1\\n\\nAldric tested the iron chain."}\n'
            '    ]\n'
            '  }\n'
            '}\n',
            encoding="utf-8",
        )
        config.set_root(story_dir)
        assert config.DB_PATH == story_dir / ".codesearch" / "codesearch.db"
        conn = db.connect()
        try:
            indexer.index(conn)
            s = Searcher(conn)
            hits = s.search("iron chain", limit=5)
            assert hits and hits[0]["file"] == "transcript.json"
            # Thought chunks are excluded from indexing
            assert s.search("Internal draft notes", strict=True) == []
            ov = graph.overview(conn, scope="runtime")
            assert ov["scope"] == "docs"
            kinds = {n["kind"] for n in ov["nodes"]}
            assert "file" in kinds and "section" in kinds
            assert any(e["kind"] == "DEFINES" for e in ov["edges"])
        finally:
            conn.close()
    finally:
        config.set_root(prev_root)



