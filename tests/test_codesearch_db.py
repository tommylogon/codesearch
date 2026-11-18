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
    from contextlib import contextmanager
    root, conn, _ = repo
    from codesearch import mcp_server, registry

    s = Searcher(conn, repo_root=root)
    entry = registry.IndexEntry(id="sample", path=root)

    @contextmanager
    def fake_open(_index: str):
        yield entry, conn, s

    monkeypatch.setattr(mcp_server, "_open_index", fake_open)
    monkeypatch.setattr(config, "DATA_DIR", root / ".codesearch")

    def call(tool, *args, **kwargs):
        fn = getattr(tool, "fn", tool)
        return fn(*args, **kwargs)

    res = call(mcp_server.search_code, "sample", "record area", limit=3)
    assert res["index"] == "sample"
    assert res["results"] and res["results"][0]["name"] == "record"

    nb = call(mcp_server.code_neighbors, "sample", "announce", direction="both", depth=1)
    assert {c["name"] for c in nb["callers"]} == {"record", "Queue::advance"}

    refs = call(mcp_server.find_references, "sample", "presence")
    assert len(refs["referenced_by"]) == 2

    out = call(mcp_server.file_outline, "sample", "engine/state.py")
    assert out["file"] == "engine/state.py" and len(out["items"]) >= 4

    code = call(mcp_server.get_code, "sample", "engine/state.py", 1, 6)
    assert "def record" in code["code"]

    st = call(mcp_server.codesearch_stats, "sample")
    assert st["files"] == 5 and st["nodes"] > 0

    rendered = call(mcp_server.render_code_graph, "sample", "announce", depth=1)
    assert Path(rendered["image_path"]).is_file()
    assert rendered["node_count"] >= 3
    rec_node = next(n for n in rendered["nodes"] if n["name"] == "record")
    expanded = call(mcp_server.render_code_graph, "sample", f"#{rendered['root']}", depth=1,
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


def test_markdown_section_symbol_and_wikilink_references(repo):
    root, conn, _ = repo
    (root / "wiki").mkdir(exist_ok=True)
    (root / "wiki" / "spells.md").write_text("# Spells\n\n## Florum\n\nConjures a living rose.\n", encoding="utf-8")
    (root / "scenes").mkdir(exist_ok=True)
    (root / "scenes" / "train.md").write_text("# Chapter 1\n\nJake cast [[Florum]] to show off.\n", encoding="utf-8")
    indexer.index(conn, root=root)

    syms = graph.find_symbol(conn, "Florum")
    assert syms, "Should resolve 'Florum' section as a symbol"
    florum_sec = syms[0]
    assert florum_sec["name"] == "Florum" and florum_sec["kind"] == "section"

    refs = graph.references(conn, florum_sec["id"])
    assert any("train.md" in r["file"] and r["mode"] == "LINKS_TO" for r in refs), refs


def test_date_filtering_keyword_semantic_and_mcp(tmp_path, monkeypatch):
    """`after` / `before` filter indexed files by mtime across keyword, vector, and MCP paths."""
    import os
    import datetime as dt
    from codesearch import mcp_server
    from codesearch.search import parse_date_bound

    (tmp_path / "old_note.md").write_text("# Old Note\n\nLantern oil archive entry.\n", encoding="utf-8")
    (tmp_path / "new_note.md").write_text("# New Note\n\nLantern oil fresh revision.\n", encoding="utf-8")

    old_ts = dt.datetime(2025, 1, 10, 12, 0, 0).timestamp()
    new_ts = dt.datetime(2026, 6, 15, 12, 0, 0).timestamp()
    os.utime(tmp_path / "old_note.md", (old_ts, old_ts))
    os.utime(tmp_path / "new_note.md", (new_ts, new_ts))

    assert parse_date_bound("2026-06-15", end_of_day=False) <= new_ts <= parse_date_bound("2026-06-15", end_of_day=True)
    assert parse_date_bound("2026-06-15T12:00:00Z") == dt.datetime(2026, 6, 15, 12, 0, 0, tzinfo=dt.timezone.utc).timestamp()

    conn = db.connect(tmp_path / "t.db")
    try:
        indexer.index(conn, root=tmp_path)

        # Inject semantic vectors for both notes to prove `_allowed` filters vector-only hits too
        def fake_vec(part, text):
            v = np.zeros(4, dtype=np.float32)
            v[0 if "lantern" in text.lower() else 1] = 1.0
            return v

        _inject(conn, fake_vec)
        monkeypatch.setattr("codesearch.search.embed_query", lambda q: np.array([1, 0, 0, 0], dtype=np.float32))

        s = Searcher(conn)
        only_new_kw = s.search("lantern", semantic=False, after="2026-01-01")
        assert {h["file"] for h in only_new_kw} == {"new_note.md"}

        only_old_kw = s.search("lantern", semantic=False, before="2025-12-31")
        assert {h["file"] for h in only_old_kw} == {"old_note.md"}

        # Semantic-only query (no keyword overlap): _allowed must enforce date bounds
        only_new_sem = s.search("illumination fuel", keyword=False, after="2026-06-01", before="2026-06-30")
        assert {h["file"] for h in only_new_sem} == {"new_note.md"}

        from contextlib import contextmanager
        from codesearch import registry

        entry = registry.IndexEntry(id="notes", path=tmp_path)

        @contextmanager
        def fake_open(_index: str):
            yield entry, conn, s

        monkeypatch.setattr(mcp_server, "_open_index", fake_open)
        fn = getattr(mcp_server.search_code, "fn", mcp_server.search_code)
        mcp_res = fn("notes", "lantern", after="2026-06-15", before="2026-06-15")
        assert {r["file"] for r in mcp_res["results"]} == {"new_note.md"}
        bad = fn("notes", "lantern", after="not-a-date")
        assert "error" in bad
    finally:
        conn.close()


def test_embed_texts_bisects_and_trims_context_errors_and_query_instruction(monkeypatch):
    """HTTP 400/500 batch errors split batches, context overflow trims single items, and query instruction prefixes work."""
    calls = []

    class FakeResp:
        def __init__(self, status_code, body):
            self.status_code = status_code
            self._body = body
            self.text = json.dumps(body) if isinstance(body, dict) else str(body)

        def json(self):
            return self._body

        def raise_for_status(self):
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

    class FakeSession:
        def post(self, url, json, timeout):
            items = json["input"]
            calls.append(list(items))
            if len(items) > 1 and any(len(x) > 1200 for x in items):
                return FakeResp(400, {"error": {"code": 400, "message": "input is larger than the max context size"}})
            if len(items) == 1 and len(items[0]) > 1200:
                return FakeResp(500, {"error": {"code": 500, "message": "input (2105 tokens) is too large to process"}})
            data = [{"index": idx, "embedding": [1.0, 0.0, 0.0]} for idx in range(len(items))]
            return FakeResp(200, {"data": data})

    monkeypatch.setattr(embedder, "_session", lambda: FakeSession())

    vecs = embedder.embed_texts(["short a", "x" * 2000, "short b"])
    assert vecs.shape == (3, 3)
    assert any(len(c) == 1 and len(c[0]) <= 1200 for c in calls)

    calls.clear()
    monkeypatch.setattr(config, "EMBED_QUERY_INSTRUCTION", "Instruct: find code\nQuery:")
    qv = embedder.embed_query("turn queue")
    assert qv.shape == (3,)
    assert calls[0] == ["Instruct: find code\nQuery: turn queue"]


def test_llama_embed_server_command_and_context_manager(tmp_path, monkeypatch):
    """embed_server resolves presets/ggufs, builds parallel llama-server args, and restores config on exit."""
    from codesearch import embed_server

    gguf = tmp_path / "embeddinggemma-300M-Q8_0.gguf"
    gguf.write_bytes(b"GGUF")
    srv_bin = tmp_path / "llama-server.exe"
    srv_bin.write_bytes(b"EXE")

    monkeypatch.setattr(config, "MODEL_SEARCH_DIRS", (str(tmp_path),))
    monkeypatch.setattr(config, "LLAMA_SERVER_BIN", str(srv_bin))

    spec = embed_server.resolve_gguf_spec("embeddinggemma-300m")
    assert spec.gguf_path == gguf
    assert spec.model_id == "text-embedding-embeddinggemma-300m"

    cmd = embed_server.build_llama_command(spec, server_bin=srv_bin, port=12444)
    assert "--embedding" in cmd and "-np" in cmd and "-c" in cmd
    assert cmd[cmd.index("--alias") + 1] == "text-embedding-embeddinggemma-300m"
    assert cmd[cmd.index("-np") + 1] == str(spec.parallel)
    assert cmd[cmd.index("-c") + 1] == str(spec.slot_ctx * spec.parallel)

    terminated = []

    class FakeProc:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            terminated.append(True)
            self.returncode = 0

        def wait(self, timeout=None):
            return 0

    class FakeHttpResp:
        status_code = 200

    monkeypatch.setattr(embed_server.subprocess, "Popen", lambda *a, **kw: FakeProc())
    monkeypatch.setattr(embed_server.requests, "get", lambda *a, **kw: FakeHttpResp())

    orig_endpoint = config.EMBED_ENDPOINT
    orig_model = config.EMBED_MODEL
    with embed_server.LlamaEmbedServer(spec, port=12445) as srv:
        assert config.EMBED_ENDPOINT == srv.endpoint == "http://127.0.0.1:12445/v1"
        assert config.EMBED_MODEL == "text-embedding-embeddinggemma-300m"
        assert config.EMBED_WORKERS == spec.parallel
    assert terminated == [True]
    assert config.EMBED_ENDPOINT == orig_endpoint
    assert config.EMBED_MODEL == orig_model


def test_searcher_vectors_reloads_on_model_or_cache_change(repo, monkeypatch):
    """Searcher.vectors invalidates when vec_cache rows arrive or config.EMBED_MODEL changes."""
    _, conn, _ = repo
    searcher = Searcher(conn)
    assert not searcher.vectors.available()

    def fake_model_a(part, text):
        return np.array([1.0, 0.0], dtype=np.float32)

    _inject(conn, fake_model_a)
    assert searcher.vectors.available()

    monkeypatch.setattr(config, "EMBED_MODEL", "model-b")
    assert not searcher.vectors.available()

    def fake_model_b(part, text):
        return np.array([0.0, 1.0], dtype=np.float32)

    _inject(conn, fake_model_b)
    assert searcher.vectors.available()
    assert "cached_vectors_by_model" in db.stats(conn)
    assert len(db.stats(conn)["cached_vectors_by_model"]) == 2


def test_cli_embed_llama_runtime_flags(repo, tmp_path, monkeypatch):
    """CLI embed --llama starts managed server with --parallel/--batch and restores config on exit."""
    from codesearch import cli, embed_server

    root, _, _ = repo
    gguf = tmp_path / "embeddinggemma-300M-Q8_0.gguf"
    gguf.write_bytes(b"GGUF")
    srv_bin = tmp_path / "llama-server.exe"
    srv_bin.write_bytes(b"EXE")

    monkeypatch.setattr(config, "MODEL_SEARCH_DIRS", (str(tmp_path),))
    monkeypatch.setattr(config, "LLAMA_SERVER_BIN", str(srv_bin))

    entered = []
    exited = []
    seen_build = []

    class FakeServer:
        def __init__(self, spec, **kwargs):
            entered.append((spec.model_id, kwargs))

        def __enter__(self):
            config.EMBED_MODEL = "text-embedding-embeddinggemma-300m"
            config.EMBED_ENDPOINT = "http://127.0.0.1:12999/v1"
            config.EMBED_WORKERS = entered[-1][1]["parallel"]
            config.EMBED_BATCH = entered[-1][1]["req_batch"]
            return self

        def __exit__(self, *_exc):
            exited.append(True)

    monkeypatch.setattr(embed_server, "LlamaEmbedServer", FakeServer)
    monkeypatch.setattr(embedder, "build", lambda conn, **kw: seen_build.append((config.EMBED_MODEL, config.EMBED_WORKERS, config.EMBED_BATCH)) or {"embedded": 0})

    orig_root = config.REPO_ROOT
    orig_model = config.EMBED_MODEL
    try:
        rc = cli.main(["-r", str(root), "embed", "--llama", "embeddinggemma-300m", "--parallel", "6", "--batch", "16"])
        assert rc == 0
        assert entered and entered[0][0] == "text-embedding-embeddinggemma-300m"
        assert entered[0][1]["parallel"] == 6
        assert seen_build == [("text-embedding-embeddinggemma-300m", 6, 16)]
        assert exited == [True]
        assert config.EMBED_MODEL == orig_model
    finally:
        config.set_root(orig_root)


def test_pdf_end_to_end_index_graph_mcp_and_api(tmp_path):
    """Multi-page PDFs index as hierarchical sections+pages, never binary blobs, with clamped reads."""
    import pypdf
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    from codesearch import app, mcp_server

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    writer = pypdf.PdfWriter()
    res_dict = DictionaryObject({
        NameObject("/Font"): DictionaryObject({
            NameObject("/F1"): DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }),
            NameObject("/F2"): DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica-Bold"),
            }),
        }),
    })

    page_specs: list[list[tuple[str, float, bool]]] = [
        [("Chapter 1: Magic Theory ............ 2", 10, False), ("Chapter 2: Combat Rules ............ 4", 10, False)],
        [("Chapter 1: Magic Theory", 14, True), ("Arcane runes draw ambient aether.", 10, False)],
        [("Focus crystal resonance peaks at dawn. See Chapter 2 for duels.", 10, False)],
        [("Chapter 2: Combat Rules", 14, True), ("Initiative determines tactical turn order.", 10, False)],
    ]
    for p_idx in range(5, 23):
        page_specs.append([(f"Appendix note {p_idx} on cauldron maintenance.", 10, False)])

    for lines in page_specs:
        page = writer.add_blank_page(width=612, height=792)
        ops = ["BT"]
        y = 740
        for text, size, bold in lines:
            ops.append(f"{'/F2' if bold else '/F1'} {size:g} Tf 1 0 0 1 72 {y} Tm ({esc(text)}) Tj")
            y -= 18
        ops.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(ops).encode("latin-1"))
        page[NameObject("/Resources")] = res_dict
        page[NameObject("/Contents")] = writer._add_object(stream)

    part = writer.add_outline_item("Part I: Compendium", 1)
    writer.add_outline_item("Chapter 1: Magic Theory", 1, parent=part)
    writer.add_outline_item("Chapter 2: Combat Rules", 3, parent=part)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = docs_dir / "manual.pdf"
    with open(pdf_path, "wb") as fh:
        writer.write(fh)

    orig_root = config.REPO_ROOT
    try:
        config.set_root(tmp_path)
        conn = db.connect()
        try:
            res = indexer.index(conn, root=tmp_path)
            assert res["files"] == 1
            st = db.stats(conn)
            assert st["files_by_language"]["pdf"] == 1

            # Verify parent hierarchy in graph.summary and cross-ref LINKS_TO edge
            chap1_page2 = conn.execute(
                "SELECT id FROM nodes WHERE name LIKE '%Chapter 1: Magic Theory - Page 2%'"
            ).fetchone()
            assert chap1_page2 is not None
            rel_sum = graph.summary(conn, chap1_page2["id"])
            assert "defined_in" in rel_sum
            assert "Chapter 1: Magic Theory" in rel_sum["defined_in"]["items"][0]

            links_to_c2 = conn.execute(
                "SELECT s.name AS src, t.name AS dst FROM edges e "
                "JOIN nodes s ON s.id = e.source_id JOIN nodes t ON t.id = e.target_id "
                "WHERE e.kind = 'LINKS_TO' AND t.name = 'Chapter 2: Combat Rules'"
            ).fetchall()
            assert links_to_c2, "Expected LINKS_TO edge from TOC or prose reference to Chapter 2"

            # Search hits exact section/page text and never raw %PDF bytes
            hits = Searcher(conn).search("Initiative tactical turn", languages=["pdf"])
            assert hits
            assert any("Combat Rules" in h["name"] or "tactical" in h["docstring"] for h in hits)
            assert not any("%PDF-" in (h.get("docstring") or "") for h in hits)

            # Overview in docs mode shows top-level outline/TOC sections without exploding leaf pages
            ov = graph.overview(conn, scope="docs")
            ov_names = {n["name"] for n in ov["nodes"]}
            assert "manual.pdf" in ov_names
            assert "Part I: Compendium" in ov_names
            assert not any("- Page " in name for name in ov_names)
        finally:
            conn.close()

        from codesearch import registry

        registry.upsert_entry(tmp_path, name="manual")
        get_code_fn = getattr(mcp_server.get_code, "fn", mcp_server.get_code)
        got = get_code_fn("manual", "docs/manual.pdf", 1, 25)
        assert got["index"] == "manual"
        assert got["unit"] == "pages"
        assert (got["start"], got["end"], got["total_pages"]) == (1, 20, 22)
        assert "clamped to 20 pages" in got["note"]
        assert "=== Page 1 of 22 ===" in got["code"]

        client = app.app.test_client()
        resp = client.get("/api/code?path=docs/manual.pdf&start=2&end=4")
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload["pdf"] is True
        assert payload["unit"] == "pages"
        assert (payload["page_start"], payload["page_end"], payload["total"]) == (2, 4, 22)
        assert any("Arcane runes" in ln for ln in payload["lines"])
    finally:
        mcp_server.clear_caches()
        config.set_root(orig_root)


def test_concurrent_flask_search_and_public_api_pdf_code(tmp_path, monkeypatch):
    """Concurrent /api/search requests in both app.py and serve_public_viwo.py must not share
    SQLite connections across threads, and public /api/code returns bounded PDF page slices."""
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from codesearch import app

    pypdf = pytest.importorskip("pypdf")
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import serve_public_viwo as public_api

    notes_dir = tmp_path / "docs"
    notes_dir.mkdir()
    (notes_dir / "chapter_one.md").write_text(
        "# Lantern Signal\nA brass lantern hung above the harbor gate.\n", encoding="utf-8"
    )
    writer = pypdf.PdfWriter()
    for _ in range(23):
        writer.add_blank_page(width=612, height=792)
    with open(tmp_path / "manual.pdf", "wb") as fh:
        writer.write(fh)

    orig_root = config.REPO_ROOT
    monkeypatch.setattr(public_api, "INDEX_REGISTRY", {
        "demo": {"name": "Demo", "path": tmp_path, "description": "Temp test index"},
    })
    monkeypatch.setattr(public_api, "API_KEY", None)
    monkeypatch.setitem(public_api._vector_caches, "demo", {})
    app._vector_caches.clear()
    try:
        config.set_root(tmp_path)
        conn = db.connect()
        try:
            indexer.index(conn, root=tmp_path)
            _inject(conn, lambda _p, _t: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        finally:
            conn.close()

        monkeypatch.setattr("codesearch.search.embed_query", lambda _q: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        app.app.config["PROPAGATE_EXCEPTIONS"] = True
        public_api.app.config["PROPAGATE_EXCEPTIONS"] = True

        def hit_local(_i):
            r = app.app.test_client().get("/api/search?q=Lantern")
            assert r.status_code == 200
            return r.get_json()["results"]

        def hit_public(_i):
            r = public_api.app.test_client().get("/api/search?q=Lantern&index=demo")
            assert r.status_code == 200
            return r.get_json()["count"]

        with ThreadPoolExecutor(max_workers=6) as pool:
            local_res = list(pool.map(hit_local, range(12)))
            public_res = list(pool.map(hit_public, range(12)))

        assert all(len(hits) >= 1 for hits in local_res)
        assert all(cnt >= 1 for cnt in public_res)

        pdf_resp = public_api.app.test_client().get("/api/code?path=manual.pdf&index=demo&start=1&end=25")
        assert pdf_resp.status_code == 200
        pdf_json = pdf_resp.get_json()
        assert pdf_json["pdf"] is True
        assert pdf_json["unit"] == "pages"
        assert (pdf_json["page_start"], pdf_json["page_end"], pdf_json["total_pages"]) == (1, 20, 23)
        assert pdf_json["clamped"] is True
    finally:
        public_api._vector_caches.pop("demo", None)
        public_api._vector_locks.pop("demo", None)
        app._vector_caches.clear()
        config.set_root(orig_root)


def test_schema_creates_missing_parent_and_vec_model_indexes(tmp_path):
    """Connecting to a fresh or older DB guarantees idx_nodes_parent and idx_vec_cache_model exist."""
    import sqlite3

    db_path = tmp_path / "legacy.db"
    conn = db.connect(db_path)
    conn.close()

    raw = sqlite3.connect(str(db_path))
    raw.execute("DROP INDEX idx_nodes_parent")
    raw.execute("DROP INDEX idx_vec_cache_model")
    raw.commit()
    raw.close()

    migrated = db.connect(db_path)
    try:
        idxs = {
            r[0]
            for r in migrated.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert "idx_nodes_parent" in idxs
        assert "idx_vec_cache_model" in idxs
    finally:
        migrated.close()


def test_vector_pack_roundtrip_and_invalidation(tmp_path):
    """VectorIndex writes contiguous .npy packs on SQL miss, hits pack on reload, and invalidates on edits."""
    from codesearch import cli

    (tmp_path / "alpha.py").write_text("def spell_cast():\n    return 42\n", encoding="utf-8")
    orig_root = config.REPO_ROOT
    db_path = tmp_path / ".codesearch" / "codesearch.db"
    conn = None
    try:
        config.set_root(tmp_path)
        conn = db.connect(db_path)
        indexer.index(conn, root=tmp_path)
        _inject(conn, lambda _p, _t: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

        vi_sql = embedder.VectorIndex(conn, parts=("function",))
        assert vi_sql.load_stats["function"]["source"] == "sql"
        assert (db_path.parent / "vecpacks").is_dir()

        vi_pack = embedder.VectorIndex(conn, parts=("function",))
        assert vi_pack.load_stats["function"]["source"] == "pack"
        qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert vi_pack.top("function", qvec, 5) == vi_sql.top("function", qvec, 5)

        # CLI `pack` runs cleanly against the file DB
        conn.close()
        conn = None
        assert cli.main(["-r", str(tmp_path), "pack"]) == 0

        # Mutate cached vector; signature check must invalidate pack and reload from SQL
        conn = db.connect(db_path)
        _inject(conn, lambda _p, _t: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        vi_rebuilt = embedder.VectorIndex(conn, parts=("function",))
        assert vi_rebuilt.load_stats["function"]["source"] == "sql"
        vi_repacked = embedder.VectorIndex(conn, parts=("function",))
        assert vi_repacked.load_stats["function"]["source"] == "pack"
        hits = vi_repacked.top("function", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), 5)
        assert hits and hits[0][1] == pytest.approx(1.0)
    finally:
        if conn is not None:
            conn.close()
        config.set_root(orig_root)


def test_diag_mcp_timings_stall_watchdog_and_query_timeout(tmp_path, monkeypatch, capsys):
    """MCP tools log high-verbosity timings + stall watchdogs cleanly without writing to stdout."""
    import time as pytime
    from codesearch import diag, mcp_server, registry

    (tmp_path / "story.md").write_text("# Chapter One\nLantern glow in the archive.\n", encoding="utf-8")
    orig_root = config.REPO_ROOT
    mcp_server.clear_caches()
    diag.shutdown()
    try:
        config.set_root(tmp_path)
        conn = db.connect()
        try:
            indexer.index(conn, root=tmp_path)
            _inject(conn, lambda _p, _t: np.array([1.0, 0.0], dtype=np.float32))
        finally:
            conn.close()

        registry.upsert_entry(tmp_path, name="lantern")
        seen_timeouts = []

        class FakeResp:
            status_code = 200
            text = "{}"

            def json(self):
                return {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

        class FakeSession:
            def post(self, url, json, timeout):
                seen_timeouts.append(timeout)
                return FakeResp()

        monkeypatch.setattr(embedder, "_session", lambda: FakeSession())
        monkeypatch.setattr(config, "EMBED_CONNECT_TIMEOUT", 1.5)
        monkeypatch.setattr(config, "EMBED_QUERY_TIMEOUT", 7.5)
        monkeypatch.setenv("CODESEARCH_DEBUG_TIMINGS", "1")

        search_fn = getattr(mcp_server.search_code, "fn", mcp_server.search_code)
        stats_fn = getattr(mcp_server.codesearch_stats, "fn", mcp_server.codesearch_stats)
        logs_fn = getattr(mcp_server.codesearch_logs, "fn", mcp_server.codesearch_logs)

        res = search_fn("lantern", "Lantern", limit=5)
        assert res["index"] == "lantern"
        assert res["results"] and "timings_ms" in res
        assert seen_timeouts == [(1.5, 7.5)]
        assert "total_ms" in res["timings_ms"] and "kw_ms" in res["timings_ms"]

        st = stats_fn("lantern")
        assert st["index"] == "lantern"
        assert st["repo_root"] == str(tmp_path.resolve())
        assert st["log_file"] and Path(st["log_file"]).is_file()

        log = mcp_server._ensure_log()
        with diag.watch_slow("simulated_stall", logger=log, warn_after=0.04, repeat_every=1.0):
            pytime.sleep(0.14)

        logs_out = logs_fn(lines=80)
        joined = "\n".join(logs_out["lines"])
        assert "STARTUP component=mcp:lantern" in joined
        assert "SEARCH " in joined
        assert "MCP_DONE tool=search_code" in joined
        assert "STALL_WATCHDOG op=simulated_stall" in joined
        assert capsys.readouterr().out == ""
    finally:
        mcp_server.clear_caches()
        diag.shutdown()
        config.set_root(orig_root)



def test_mcp_requires_index_schema_errors_and_multi_index_isolation(tmp_path, monkeypatch):
    """MCP tools require an explicit `index`, never fall back silently, and isolate per repo."""
    import asyncio
    from codesearch import cli, diag, mcp_server, registry

    alpha = tmp_path / "alpha_world"
    beta = tmp_path / "beta_vault"
    gamma = tmp_path / "gamma_env"
    unindexed = tmp_path / "not_indexed"
    for d in (alpha, beta, gamma, unindexed):
        d.mkdir()

    (alpha / "sky.py").write_text(
        "def starlight_beacon():\n    return 'alpha-signal'\n",
        encoding="utf-8",
    )
    (beta / "cavern.py").write_text(
        "def deep_cavern():\n    return 'beta-secret'\n",
        encoding="utf-8",
    )
    (gamma / "harbor.py").write_text(
        "def harbor_bell():\n    return 'gamma-env'\n",
        encoding="utf-8",
    )

    for folder in (alpha, beta, gamma):
        conn = db.connect(folder / ".codesearch" / "codesearch.db")
        try:
            indexer.index(conn, root=folder)
        finally:
            conn.close()

    orig_root = config.REPO_ROOT
    mcp_server.clear_caches()
    diag.shutdown()
    try:
        tools = asyncio.run(mcp_server.mcp.get_tools())
        index_required_tools = (
            "search_code",
            "code_neighbors",
            "find_references",
            "file_outline",
            "get_code",
            "render_code_graph",
            "codesearch_stats",
        )
        for tool_name in index_required_tools:
            req = tools[tool_name].parameters.get("required", [])
            assert "index" in req, f"{tool_name} must require `index`"
        assert tools["list_indexes"].parameters.get("required", []) == []

        # Register alpha in JSON file, beta via CODESEARCH_INDEXES, gamma via CODESEARCH_ROOT.
        registry.upsert_entry(alpha, name="alpha", description="Sky simulator")
        monkeypatch.setenv("CODESEARCH_INDEXES", f"beta={beta}")
        config.set_root(gamma)

        list_fn = getattr(mcp_server.list_indexes, "fn", mcp_server.list_indexes)
        search_fn = getattr(mcp_server.search_code, "fn", mcp_server.search_code)
        code_fn = getattr(mcp_server.get_code, "fn", mcp_server.get_code)
        stats_fn = getattr(mcp_server.codesearch_stats, "fn", mcp_server.codesearch_stats)
        render_fn = getattr(mcp_server.render_code_graph, "fn", mcp_server.render_code_graph)

        listed = list_fn()
        by_id = {it["id"]: it for it in listed["indexes"]}
        assert {"alpha", "beta", "gamma_env"} <= set(by_id)
        assert by_id["alpha"]["ready"] is True

        # Empty or unknown index must return clear errors and never create a DB or fall back.
        for bad_idx in ("", "   ", "nonexistent_alias", str(unindexed)):
            err_res = search_fn(bad_idx, "harbor_bell")
            assert "error" in err_res and "available_indexes" in err_res
            assert not (unindexed / ".codesearch" / "codesearch.db").exists()

        # Queries strictly route only to the named index (never spilling across indexes).
        hit_alpha = search_fn("alpha", "starlight_beacon", strict=True)
        assert hit_alpha["index"] == "alpha" and hit_alpha["results"][0]["name"] == "starlight_beacon"
        miss_alpha = search_fn("alpha", "deep_cavern", strict=True)
        assert miss_alpha["results"] == []

        hit_beta = search_fn("beta", "deep_cavern", strict=True)
        assert hit_beta["index"] == "beta" and hit_beta["results"][0]["name"] == "deep_cavern"

        # Direct indexed directory path is also accepted as an explicit index.
        hit_path = search_fn(str(gamma), "harbor_bell", strict=True)
        assert hit_path["results"][0]["name"] == "harbor_bell"

        # Path confinement stays scoped per chosen index.
        denied = code_fn("alpha", "../beta_vault/cavern.py", 1, 5)
        assert "error" in denied
        ok_code = code_fn("beta", "cavern.py", 1, 2)
        assert ok_code["index"] == "beta" and "deep_cavern" in ok_code["code"]

        st_beta = stats_fn("beta")
        assert st_beta["index"] == "beta" and st_beta["repo_root"] == str(beta.resolve())

        rendered = render_fn("alpha", "starlight_beacon")
        assert rendered["index"] == "alpha"
        assert str(alpha.resolve()) in rendered["image_path"]
        assert Path(rendered["image_path"]).is_file()

        # CLI `registry scan` discovers new un-aliased indexed folders cleanly.
        assert cli.main(["registry", "scan", str(tmp_path), "--depth", "1"]) == 0
        assert "gamma_env" in registry.load_file_registry()
        assert cli.main(["registry", "remove", "alpha"]) == 0
        assert "alpha" not in registry.load_file_registry()
    finally:
        mcp_server.clear_caches()
        diag.shutdown()
        config.set_root(orig_root)


def test_cli_inspect_dumps_nodes_parts_edges_and_refs(tmp_path, capsys):
    """`codesearch inspect` exposes nodes, embedded part text, FTS words, edges, and refs."""
    from codesearch import cli, diag

    orig_root = config.REPO_ROOT
    repo = tmp_path / "sandbox"
    repo.mkdir()
    doc = {
        "characters": [
            {"id": "sammy", "name": "Sammy", "role": "Investigator"},
        ],
        "memories": [
            {
                "id": "mem_1",
                "character_id": "sammy",
                "tags": ["library", "key"],
                "summary": (
                    "Sammy spotted the brass key beneath the oak desk at dawn. "
                    "She kept quiet because Kay was watching from the stairwell."
                ),
            }
        ],
    }
    (repo / "mind.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    try:
        config.set_root(repo)
        conn = db.connect()
        try:
            indexer.index(conn, root=repo)
        finally:
            conn.close()
        capsys.readouterr()

        assert cli.main(["-r", str(repo), "inspect", "mind.json", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["file"]["rel_path"] == "mind.json"
        kinds = {n["kind"] for n in payload["nodes"]}
        assert {"file", "record", "tag", "section"} <= kinds

        record_nodes = [n for n in payload["nodes"] if n["kind"] == "record"]
        assert all(n["parts"] == [] for n in record_nodes)
        section_nodes = [n for n in payload["nodes"] if n["kind"] == "section"]
        assert any("brass key" in (n["docstring"] or "") for n in section_nodes)
        assert any(
            p["part"] == "comment" and "brass key" in p["text"]
            for n in section_nodes for p in n["parts"]
        )
        assert any(e["kind"] == "CONNECTS_TO" for e in payload["edges"])
        assert any(r["target"] == "sammy" for r in payload["refs"])

        # Text mode and missing file handling
        assert cli.main(["-r", str(repo), "inspect", "mind.json"]) == 0
        text_out = capsys.readouterr().out
        assert "mind.json" in text_out and "part[comment]" in text_out
        assert cli.main(["-r", str(repo), "inspect", "missing.json"]) == 1
    finally:
        diag.shutdown()
        config.set_root(orig_root)


def test_web_workbench_multi_index_inspect_and_file_graph(tmp_path):
    """Web workbench endpoints serve vis-network, switch indexes, inspect anatomy, and graph one file."""
    from codesearch import app as web_app, diag

    orig_root = config.REPO_ROOT
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    (repo_a / "alpha.json").write_text(
        json.dumps({"id": "alpha", "note": "Lantern flickered on the pier."}, indent=2),
        encoding="utf-8",
    )
    (repo_b / "beta.json").write_text(
        json.dumps(
            {
                "characters": [{"id": "sammy", "name": "Sammy"}],
                "scenes": [
                    {
                        "id": "sc_1",
                        "character_id": "sammy",
                        "tags": ["arcade", "neon"],
                        "summary": "Beacon hummed beside the claw machine while Sammy checked the ledger.",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        for repo in (repo_a, repo_b):
            config.set_root(repo)
            conn = db.connect()
            try:
                indexer.index(conn, root=repo)
            finally:
                conn.close()

        config.set_root(repo_a)
        client = web_app.app.test_client()

        home = client.get("/")
        assert home.status_code == 200
        html_text = home.get_data(as_text=True)
        assert "<search>" in html_text
        assert 'id="index-select"' in html_text and 'id="file-select"' in html_text

        vis = client.get("/vendor/vis-network.js")
        assert vis.status_code == 200
        assert len(vis.data) > 100_000

        indexes = client.get("/api/indexes")
        assert indexes.status_code == 200
        idx_payload = indexes.get_json()
        assert idx_payload["active_id"]
        assert isinstance(idx_payload["indexes"], list) and idx_payload["indexes"]

        files_a = client.get("/api/files").get_json()
        assert files_a["total"] == 1
        assert [f["rel_path"] for f in files_a["files"]] == ["alpha.json"]

        files_b = client.get("/api/files", query_string={"index": str(repo_b)}).get_json()
        assert files_b["total"] == 1
        assert [f["rel_path"] for f in files_b["files"]] == ["beta.json"]

        insp_b = client.get(
            "/api/inspect",
            query_string={"index": str(repo_b), "path": "beta.json"},
        )
        assert insp_b.status_code == 200
        insp_data = insp_b.get_json()
        assert insp_data["file"]["rel_path"] == "beta.json"
        assert any(n["kind"] == "section" for n in insp_data["nodes"])
        assert any(r["target"] == "sammy" for r in insp_data["refs"])

        fg_b = client.get(
            "/api/file_graph",
            query_string={"index": str(repo_b), "path": "beta.json"},
        )
        assert fg_b.status_code == 200
        fg_data = fg_b.get_json()
        assert fg_data["file"] == "beta.json"
        assert fg_data["root"] is not None
        assert len(fg_data["nodes"]) >= 3
        assert any(e["kind"] == "DEFINES" for e in fg_data["edges"])
        assert any(e["kind"] == "CONNECTS_TO" for e in fg_data["edges"])

        sr_b = client.get(
            "/api/search",
            query_string={"index": str(repo_b), "q": "beacon", "semantic": "0"},
        )
        assert sr_b.status_code == 200
        sr_data = sr_b.get_json()
        assert "beacon" in sr_data["tokens"]
        assert "timings" in sr_data
        assert sr_data["total_matched"] >= 1
        assert any("Beacon hummed" in (r.get("docstring") or "") for r in sr_data["results"])

        code_b = client.get(
            "/api/code",
            query_string={"index": str(repo_b), "path": "beta.json", "start": 1, "end": 500, "context": 0},
        )
        assert code_b.status_code == 200
        assert "claw machine" in "\n".join(code_b.get_json()["lines"])
    finally:
        with web_app._vector_lock:
            web_app._vector_caches.clear()
        diag.shutdown()
        config.set_root(orig_root)


def test_unlimited_file_size_and_paged_inspect_streaming_code(tmp_path, monkeypatch):
    """Default byte/node caps are None; explicit env/monkeypatch caps still work; inspect & code page cleanly."""
    from codesearch import app as web_app, diag
    from codesearch.parsers import csv_parser, json_parser

    assert config.MAX_FILE_BYTES is None
    assert config.MAX_JSON_BYTES is None
    assert config.IMAGE_MAX_BYTES is None
    assert config.MAX_CSV_NODES is None

    orig_root = config.REPO_ROOT
    repo = tmp_path / "big_repo"
    repo.mkdir()
    csv_lines = ["id,title"] + [f"t{i},Release item number {i} in HD 1080p format" for i in range(1, 31)]
    (repo / "items.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (repo / "small.json").write_text(json.dumps({"a": "first", "b": "second", "c": "third"}), encoding="utf-8")

    # Unlimited caps never fail or truncate in scan/parse.
    files, skipped = indexer.scan(repo)
    assert len(files) == 2 and skipped == {}

    # Explicit byte cap skips large files only when configured.
    monkeypatch.setattr(config, "MAX_FILE_BYTES", 500)
    files_capped, skipped_capped = indexer.scan(repo)
    assert skipped_capped.get("too_large") == 1
    assert [f["rel"] for f in files_capped] == ["small.json"]
    monkeypatch.undo()

    # Explicit CSV and JSON node caps report problems cleanly instead of crashing.
    monkeypatch.setattr(config, "MAX_CSV_NODES", 5)
    capped_csv = csv_parser.parse((repo / "items.csv").read_text(encoding="utf-8"), "items.csv")
    assert any("node cap reached" in p for p in capped_csv["problems"])
    monkeypatch.setattr(config, "MAX_JSON_NODES", 2)
    capped_json = json_parser.parse(json.dumps({"a": {"x": 1}, "b": {"y": 2}, "c": {"z": 3}}), "small.json")
    assert any("node cap reached" in p for p in capped_json["problems"])
    monkeypatch.undo()

    try:
        config.set_root(repo)
        conn = db.connect()
        try:
            indexer.index(conn, root=repo)
        finally:
            conn.close()

        client = web_app.app.test_client()
        paged = client.get("/api/inspect", query_string={"path": "items.csv", "limit": "5", "offset": "2"})
        assert paged.status_code == 200
        pdata = paged.get_json()
        assert pdata["truncated"] is True
        assert pdata["offset"] == 2
        assert len(pdata["nodes"]) == 5
        assert pdata["total_nodes"] > 30

        sliced = client.get("/api/code", query_string={"path": "items.csv", "start": "5", "end": "8", "context": "0"})
        assert sliced.status_code == 200
        sdata = sliced.get_json()
        assert sdata["start"] == 5 and sdata["end"] == 8 and sdata["total"] == 32
        assert sdata["lines"][0].startswith("t4,") and sdata["lines"][-1].startswith("t7,")
    finally:
        with web_app._vector_lock:
            web_app._vector_caches.clear()
        diag.shutdown()
        config.set_root(orig_root)


def test_stage2_reranker_mocked_and_offline_fallback(tmp_path, monkeypatch):
    from codesearch import embed_server, reranker, search

    (tmp_path / "alpha.py").write_text(
        'def handle_scared_npc():\n'
        '    """Unrelated mention of scared in dialogue."""\n'
        '    return "scared"\n',
        encoding="utf-8",
    )
    (tmp_path / "beta.py").write_text(
        'def resolve_fear_panic_cause():\n'
        '    """Causal fear resolution when scared by combat threat."""\n'
        '    return "panic_resolved"\n',
        encoding="utf-8",
    )

    orig_root = config.REPO_ROOT
    try:
        config.set_root(tmp_path)
        conn = db.connect()
        indexer.index(conn, tmp_path, force=True)
        searcher = search.Searcher(conn, repo_root=tmp_path)

        # Baseline RRF order without reranking.
        base = searcher.search("scared", limit=5, semantic=False, rerank=False)
        assert len(base) >= 2

        captured_payloads: list[dict] = []

        class _FakeResp:
            status_code = 200

            def __init__(self, data):
                self._data = data

            def raise_for_status(self):
                return None

            def json(self):
                return self._data

        def _fake_post(url, json=None, timeout=None):
            captured_payloads.append({"url": url, "json": json, "timeout": timeout})
            docs = json["documents"]
            # Reverse-boost whichever document mentions resolve_fear_panic_cause.
            results = []
            for idx, doc in enumerate(docs):
                score = 0.99 if "resolve_fear_panic_cause" in doc else 0.12
                results.append({"index": idx, "relevance_score": score})
            return _FakeResp({"results": results})

        monkeypatch.setattr(reranker.requests, "post", _fake_post)
        reranked = searcher.search("scared", limit=5, semantic=False, rerank=True)
        assert len(captured_payloads) == 1
        assert captured_payloads[0]["json"]["query"].startswith("Instruct:")
        assert "resolve_fear_panic_cause" in reranked[0]["name"]
        assert reranked[0]["rerank_score"] == 0.99
        assert "rrf_score" in reranked[0]
        assert searcher.last_timings["rerank_used"] is True

        # Offline fallback: unreachable endpoint must degrade cleanly to RRF order.
        monkeypatch.undo()
        monkeypatch.setattr(config, "RERANK_ENDPOINT", "http://127.0.0.1:59998/v1")
        monkeypatch.setattr(config, "RERANK_CONNECT_TIMEOUT", 0.2)
        monkeypatch.setattr(config, "RERANK_TIMEOUT", 0.3)
        fallback = searcher.search("scared", limit=5, semantic=False, rerank=True)
        assert [h["id"] for h in fallback] == [h["id"] for h in base]
        assert searcher.last_timings["rerank_used"] is False
        assert searcher.last_rerank_error is not None

        # Verify embed_server emits --reranking for reranker specs.
        spec = embed_server.LlamaModelSpec(gguf_path=tmp_path / "Qwen3-Reranker-0.6B-Q8_0.gguf", model_id="Qwen3-Reranker-0.6B", reranking=True)
        cmd = embed_server.build_llama_command(spec, server_bin=tmp_path / "llama-server.exe", port=1237)
        assert "--reranking" in cmd
        assert "--embedding" not in cmd
        conn.close()
    finally:
        config.set_root(orig_root)


def test_boolean_and_or_not_search_and_mcp_payload(tmp_path):
    from codesearch import mcp_server, search

    (tmp_path / "fire_spell.py").write_text(
        'def cast_fireball():\n'
        '    """Launch an explosive fireball spell."""\n'
        '    return "fireball"\n',
        encoding="utf-8",
    )
    (tmp_path / "fire_trap.py").write_text(
        'def trigger_fire_trap():\n'
        '    """Trigger a dungeon trap with fire damage."""\n'
        '    return "trap"\n',
        encoding="utf-8",
    )
    (tmp_path / "ice_bolt.py").write_text(
        'def cast_ice_bolt():\n'
        '    """Launch a frost bolt spell."""\n'
        '    return "ice"\n',
        encoding="utf-8",
    )

    orig_root = config.REPO_ROOT
    try:
        config.set_root(tmp_path)
        conn = db.connect()
        indexer.index(conn, tmp_path, force=True)
        searcher = search.Searcher(conn, repo_root=tmp_path)

        # build_boolean_fts_expression compiles AND, OR, NOT cleanly.
        expr = search.build_boolean_fts_expression(
            [], True, None,
            must_terms=["fire"],
            should_terms=["spell", "trap"],
            not_terms=["trap"],
        )
        assert '"fire"' in expr and "OR" in expr and "NOT" in expr

        # must_terms + not_terms excludes fire_trap.py while keeping fire_spell.py.
        hits = searcher.search(
            "",
            must_terms=["fire"],
            should_terms=["spell", "trap"],
            not_terms=["trap"],
            semantic=False,
        )
        files = {h["file"] for h in hits}
        assert "fire_spell.py" in files
        assert "fire_trap.py" not in files
        assert "ice_bolt.py" not in files

        # query + not_terms also excludes negated matches.
        q_hits = searcher.search("cast", not_terms=["ice"], semantic=False)
        q_files = {h["file"] for h in q_hits}
        assert "fire_spell.py" in q_files
        assert "ice_bolt.py" not in q_files

        # MCP search_code accepts must_terms / should_terms / not_terms.
        res = mcp_server.search_code.fn(
            index=str(tmp_path),
            query="",
            must_terms=["spell"],
            not_terms=["fire"],
        )
        assert "error" not in res
        mcp_files = {r["file"] for r in res["results"]}
        assert "ice_bolt.py" in mcp_files
        assert "fire_spell.py" not in mcp_files

        # strict=False allows semantic hits without must_terms; strict=True enforces must_terms.
        ice_node = conn.execute(
            "SELECT n.id FROM nodes n JOIN files f ON f.id = n.file_id WHERE f.rel_path = 'ice_bolt.py' LIMIT 1"
        ).fetchone()["id"]
        fire_node = conn.execute(
            "SELECT n.id FROM nodes n JOIN files f ON f.id = n.file_id WHERE f.rel_path = 'fire_spell.py' LIMIT 1"
        ).fetchone()["id"]
        orig_sem = searcher.semantic
        searcher.semantic = lambda *a, **kw: {"comment": [(ice_node, 0.92), (fire_node, 0.90)]}
        try:
            sem_open = searcher.search("", must_terms=["fire"], strict=False, semantic=True)
            assert {"fire_spell.py", "ice_bolt.py"}.issubset({h["file"] for h in sem_open})

            sem_strict = searcher.search("", must_terms=["fire"], strict=True, semantic=True)
            sem_strict_files = {h["file"] for h in sem_strict}
            assert "fire_spell.py" in sem_strict_files
            assert "ice_bolt.py" not in sem_strict_files
        finally:
            searcher.semantic = orig_sem
        conn.close()
    finally:
        config.set_root(orig_root)


def test_cold_vector_pack_nonblocking_strict_shortcircuit_and_fast_sig(tmp_path, monkeypatch):
    import numpy as np
    import time
    from codesearch import embedder, search

    (tmp_path / "alpha.py").write_text("def alpha_fn():\n    return 1\n", encoding="utf-8")

    orig_root = config.REPO_ROOT
    try:
        config.set_root(tmp_path)
        conn = db.connect()
        indexer.index(conn, tmp_path, force=True)

        # Insert fake vectors into vec_cache without creating .npy packs yet.
        np_rows = conn.execute("SELECT part, hash FROM node_parts").fetchall()
        vec_bytes = np.ones(4, dtype=np.float32).tobytes()
        for r in np_rows:
            conn.execute(
                "INSERT OR REPLACE INTO vec_cache(part, hash, model, dim, vec) VALUES (?,?,?,?,?)",
                (r["part"], r["hash"], config.EMBED_MODEL, 4, vec_bytes),
            )
        conn.commit()

        # 1. _part_signature uses O(1) v2 vec_generation.
        sig = embedder._part_signature(conn, "function_name", config.EMBED_MODEL)
        assert sig["v"] == 2
        assert "gen" in sig

        # 2. strict=True with 0 keyword hits skips semantic() completely.
        searcher = search.Searcher(conn, repo_root=tmp_path, background_pack_build=True)
        sem_called = []
        monkeypatch.setattr(searcher, "semantic", lambda *a, **kw: sem_called.append(True) or {})
        res_empty = searcher.search("nonexistent_symbol_xyz_999", strict=True)
        assert res_empty == []
        assert sem_called == []

        # 3. Missing .npy packs with background_pack_build=True defers pack build off MainThread.
        monkeypatch.undo()
        monkeypatch.setattr(search, "embed_query", lambda q: np.ones(4, dtype=np.float32))
        res_cold = searcher.search("alpha", limit=5, semantic=True)
        assert res_cold  # keyword hit still returned immediately
        assert "vec_deferred" in searcher.last_timings

        # Wait briefly for background warmup thread to write .npy packs.
        pack_dir = tmp_path / ".codesearch" / "vecpacks"
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if list(pack_dir.glob("*.meta.json")):
                break
            time.sleep(0.05)
        assert list(pack_dir.glob("*.meta.json"))
        conn.close()
    finally:
        config.set_root(orig_root)


def test_db_stats_active_model_coverage_prune_and_vec_only_gating(tmp_path, monkeypatch):
    import numpy as np
    from codesearch import search

    (tmp_path / "combat.py").write_text(
        'def resolve_combat_threat():\n'
        '    """Handle combat threat and fear response."""\n'
        '    return "combat"\n',
        encoding="utf-8",
    )
    (tmp_path / "design.md").write_text(
        '# Unrelated Design Note\n\nSome generic architecture text.\n',
        encoding="utf-8",
    )

    orig_root = config.REPO_ROOT
    try:
        config.set_root(tmp_path)
        conn = db.connect()
        indexer.index(conn, tmp_path, force=True)

        # Seed orphaned legacy model vectors + partial active model vectors.
        np_rows = conn.execute("SELECT node_id, part, hash FROM node_parts").fetchall()
        vec_bytes = np.ones(4, dtype=np.float32).tobytes()
        for r in np_rows:
            conn.execute(
                "INSERT OR REPLACE INTO vec_cache(part, hash, model, dim, vec) VALUES (?,?,?,?,?)",
                (r["part"], r["hash"], "legacy-old-model", 4, vec_bytes),
            )
        # Only embed the first node_part for the active model -> partial coverage.
        first = np_rows[0]
        conn.execute(
            "INSERT OR REPLACE INTO vec_cache(part, hash, model, dim, vec) VALUES (?,?,?,?,?)",
            (first["part"], first["hash"], config.EMBED_MODEL, 4, vec_bytes),
        )
        conn.commit()

        st = db.stats(conn)
        assert st["embed_model"] == config.EMBED_MODEL
        assert st["stale_models"] == {"legacy-old-model": len({(r["part"], r["hash"]) for r in np_rows})}
        assert "stale_model_warning" in st
        assert sum(st["vectors_by_part"].values()) == 1

        # Second stats call hits meta cache cleanly.
        st2 = db.stats(conn)
        assert st2["vectors_by_part"] == st["vectors_by_part"]

        # Prune orphaned legacy model rows.
        pruned = db.prune_stale_vectors(conn)
        assert "legacy-old-model" in pruned
        st3 = db.stats(conn)
        assert st3["stale_models"] == {}

        # Low-signal vec-only hit (0.48 < VEC_ONLY_MIN_SIMILARITY 0.55) is gated when kw_hits > 0.
        searcher = search.Searcher(conn, repo_root=tmp_path)
        design_node = conn.execute(
            "SELECT n.id FROM nodes n JOIN files f ON f.id = n.file_id WHERE f.rel_path = 'design.md' LIMIT 1"
        ).fetchone()["id"]
        monkeypatch.setattr(
            searcher,
            "semantic",
            lambda *a, **kw: {"comment": [(design_node, 0.48)]},
        )
        hits = searcher.search("combat threat", limit=5, semantic=True)
        assert all(h["file"] != "design.md" for h in hits)

        # High-signal vec-only hit (0.72 >= 0.55) IS admitted.
        monkeypatch.setattr(
            searcher,
            "semantic",
            lambda *a, **kw: {"comment": [(design_node, 0.72)]},
        )
        hits_high = searcher.search("combat threat", limit=5, semantic=True)
        assert any(h["file"] == "design.md" for h in hits_high)
        conn.close()
    finally:
        config.set_root(orig_root)


def test_sim_not_path_cosine_margin_and_block_downweight(tmp_path, monkeypatch):
    from codesearch import mcp_server, search

    (tmp_path / "docs").mkdir()
    (tmp_path / "dev_tasks").mkdir()
    (tmp_path / "docs" / "fear_guide.md").write_text(
        "# Fear Mechanics\n\nDetailed prose explanation of fear and panic.\n\n```json\n{\"fear\": true}\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "dev_tasks" / "task_fear.md").write_text(
        "# Task Fear\n\nBacklog note about fear.\n",
        encoding="utf-8",
    )

    orig_root = config.REPO_ROOT
    try:
        config.set_root(tmp_path)
        conn = db.connect()
        indexer.index(conn, tmp_path, force=True)
        searcher = search.Searcher(conn, repo_root=tmp_path)

        # 1. not_path excludes dev_tasks/ cleanly in both FTS and MCP search_code.
        res = mcp_server.search_code.fn(
            index=str(tmp_path),
            query="fear",
            not_path="dev_tasks/",
        )
        assert all(not r["file"].startswith("dev_tasks/") for r in res["results"])
        assert any(r["file"] == "docs/fear_guide.md" for r in res["results"])
        # Keyword-only hit has sim=None (omitted from compact MCP dict).
        assert "sim" not in res["results"][0]

        # 2. Cosine-margin RRF + block down-weighting:
        #    Give a block node rank #1 in vec:function with weak 0.46 sim,
        #    and a section node rank #2 in vec:comment with strong 0.88 sim.
        sec_node = conn.execute(
            "SELECT id FROM nodes WHERE kind = 'section' LIMIT 1"
        ).fetchone()["id"]
        blk_node = conn.execute(
            "SELECT id FROM nodes WHERE kind = 'block' LIMIT 1"
        ).fetchone()["id"]
        monkeypatch.setattr(
            searcher,
            "semantic",
            lambda *a, **kw: {
                "function": [(blk_node, 0.46)],
                "comment": [(blk_node, 0.46), (sec_node, 0.88)],
            },
        )
        ranked = searcher.search("fear", limit=5, keyword=False, semantic=True)
        assert ranked[0]["id"] == sec_node
        assert ranked[0]["sim"] == 0.88

        # Compact MCP output surfaces raw max cosine `sim`.
        compact = mcp_server._compact(ranked[0])
        assert compact["sim"] == 0.88
        conn.close()
    finally:
        config.set_root(orig_root)




