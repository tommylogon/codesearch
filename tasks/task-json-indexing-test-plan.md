# Task: JSON Indexing & Embedding Verification Test Plan

**Status:** Initial Matrix Implemented & Verified (`tests/test_codesearch_parsers.py::test_json_format_indexing_and_embedding_matrix`); Extended Corpus Planned
**Area:** Parsers & Embedder Routing (`src/codesearch/parsers/json_parser.py`, `src/codesearch/embedder.py`)
**Target:** Audit and verify exactly which nodes, FTS rows, graph edges, and vector embeddings are produced across every JSON and JSONL format encountered in codebases, game data, logs, and AI transcripts.

---

## 1. How `codesearch` Classifies & Embeds JSON Today

Unlike naive JSON text splitters, `json_parser.parse()` routes documents through three distinct code paths, and `embedder.part_texts()` applies strict filtering before queueing vectors:

| Node Kind Produced | Trigger Condition in `json_parser.py` | FTS5 Keyword Indexed? | Vector Embedded (`part_texts`)? | Notes / Caveats |
| :--- | :--- | :--- | :--- | :--- |
| **`file`** | Every valid/invalid JSON file | Yes (`rel_path` + `file_doc`) | Yes (`filename` part) | `_file_doc` summarizes title/name/lore/collection keys. |
| **`record`** | Any `dict` or `list` in recursive walker | Yes (shape docstring: child field names & types) | **No** (`part_texts` returns `{}`) | Answers "which record has field X?" without polluting cosine vectors with key names. |
| **`tag`** | Scalars failing `_is_prose()` (`< 40` chars, `< 5` words, no `[.!?]` ending), `null`, or `""` | Yes (`key: value`, split `words`) | **No** (`part_texts` returns `{}`) | Prevents 1-word tags/booleans/numbers from outranking full prose in vector search. |
| **`section`** | Prose scalars passing `_is_prose()`, plus AI Studio/JSONL turns | Yes | Yes (`function_name` + `comment`) | Values `>= 120` chars also emit sentence child nodes (`::s1`, `::s2`, ...). |
| **`CONNECTS_TO`** edge | String value under `REFERENCE_KEYS` (`target`, `way_id`, `owner_id`, etc.) | Reflected via `graph.summary` | Indirectly via `section` | Resolves only against records **within the same JSON file**. |
| **`CALLS`** edge | `tool_calls[].name` in Antigravity `.jsonl` transcripts | Reflected via `graph.summary` | Indirectly via turn `section` | Emitted only by `_parse_jsonl` when `type == "PLANNER_RESPONSE"`. |

---

## 2. Test Matrix Across JSON Archetypes

| # | JSON Archetype | Expected Parser Path | Key Assertions (Nodes & Embeddings) | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Scenario / World Map JSON** (`areas`, `ways`, `exits`, nested dicts) | `_Walker` + `_file_doc` | `CONNECTS_TO` edges resolve `target`/`area_from`; `_file_doc` includes `world_lore` and collection keys; `record` nodes stay keyword-only. | Verified (`test_json_reference_keys_become_connects_to_refs`, `test_json_file_doc_carries_lore_and_scene`) |
| **2** | **Entity / Item / Character JSON** (`tags`, `traits`, `goals`, `owner_id`, prose bio) | `_Walker` | Short tags -> `tag` (no vector); multi-sentence `personality` -> `section` (`function_name` + `comment` vectors) + `::s1`, `::s2` sentence nodes. | Verified (`test_json_format_indexing_and_embedding_matrix`) |
| **3** | **AI Studio / Gemini `chunkedPrompt` Export** | `_parse_chat_transcript` | Extracts `systemInstruction` and turns as `section`s; skips `isThought: true`; routes turns to `function_name` + `comment`. | Verified (`test_external_prose_root_and_overview_fallback` + matrix test) |
| **4** | **Antigravity Transcript `.jsonl`** (`USER_INPUT`, `PLANNER_RESPONSE`) | `_parse_jsonl` | Emits `turn_NNNN_user` / `turn_NNNN_agent` `section` nodes + `CALLS` refs for `tool_calls`. | Verified (`test_json_format_indexing_and_embedding_matrix`) |
| **5** | **Generic / OpenAI ChatML or Structured Log `.jsonl`** | `_parse_jsonl` (gap!) | **Known Gap Verified:** Currently produces **0 nodes** if records lack Antigravity's `type` enum (`USER_INPUT` / `PLANNER_RESPONSE` / `SYSTEM_MESSAGE` / `GENERIC`). | Documented & asserted in matrix test |
| **6** | **Config Manifests (`package.json`, `tsconfig.json`, schemas)** | `_Walker` | Produces `record` and `tag` nodes only; **zero** node-level vectors queued (only root `file` node gets `filename` vector). | Verified (`test_json_format_indexing_and_embedding_matrix`) |
| **7** | **Scale / Safety Boundaries** (`MAX_JSON_NODES`, `MAX_JSON_DEPTH`, malformed files) | `_Walker` / `parse` fallback | Emits explicit warnings in `problems[]` rather than silently truncating or crashing. | Partial (`test_json_invalid_is_reported_not_raised`); add node/depth cap tests next |

---

## 3. Next Planned Improvements From Test Findings

1. **Expand `_parse_jsonl` for OpenAI ChatML & Generic App Logs**:
   - Support `messages[]` format (fine-tuning / ChatML JSONL), `role` + `content`/`text` pairs, and arbitrary structured log records (fallback to flattened key/prose walk or message extraction) instead of dropping unknown schemas to 0 nodes.
2. **Add Node-Cap & Depth-Cap Unit Tests**:
   - Lower `config.MAX_JSON_NODES` / `config.MAX_JSON_DEPTH` via `monkeypatch` in `test_codesearch_parsers.py` to assert clean `problems[]` reporting at bounds.
3. **Add Vector Inspection CLI Flag (`codesearch outline <file.json> --parts`)**:
   - Show for any file which extracted nodes receive `filename` / `function_name` / `comment` embeddings vs. keyword-only (`record` / `tag`), making parser tuning transparent without SQLite inspection.
