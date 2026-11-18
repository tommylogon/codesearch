# Task: Smart LLM File Interpreter (Unstructured `.txt` to Structured JSON)

**Status:** Planned (Design Spec)
**Area:** Indexing & Parsers (`src/codesearch/indexer.py`, `src/codesearch/parsers/`)
**Target:** Convert flat, unstructured `.txt` files (notes, RP logs, raw transcripts, braindumps) into structured JSON representations so `codesearch` indexes entities, topics, dates, speakers, and relationships instead of blind line chunks.

---

## 1. Current Limitation in `.txt` Parsing

Right now `src/codesearch/indexer.py` routes `.txt`, `.srt`, and `.vtt` files through `md_parser.parse_text()`:
- Files `<= 80` lines become **one single `(text)` `section` node**.
- Files `> 80` lines split purely mechanically on blank lines once a buffer reaches `>= 20` lines (`qname = (text)@<line>`).
- Result: zero titles, zero speaker attribution, zero entity tags, zero date extraction, and zero graph edges (`LINKS_TO` / `CONNECTS_TO`). Long mixed-topic `.txt` notes dilute a single `comment` vector across unrelated thoughts.

---

## 2. Proposed Architecture

Introduce an opt-in offline preprocessing pipeline (`codesearch interpret` or `--interpret` during `index`) that uses a fast local LLM (LM Studio / OpenAI-compatible structured outputs) to distill unstructured `.txt` files into structured sidecar JSON documents parsed by `json_parser`.

```
raw note (.txt)  --[BLAKE2b content hash]--> unchanged? reuse sidecar cache
                       |
                 [hash changed / new]
                       v
        Local LLM Structured Interpreter
        (outputs validated JSON schema)
                       |
                       v
    Sidecar Cache: .codesearch/interpreted/<rel_path>.json
                       |
                       v
      Indexed alongside (never replacing!) raw .txt
```

### Non-Negotiable Safeguards

1. **Never overwrite the source `.txt` file:** User notes stay untouched on disk.
2. **Exact source line spans are mandatory:** Every extracted JSON section (`topics[]`, `events[]`, `characters[]`, `decisions[]`) must record `source_lines: [start_line, end_line]` matching the original `.txt`, so `get_code` and UI clicks jump directly to the real lines in the `.txt` file.
3. **Never drop literal keyword recall:** Keep raw `.txt` lines indexed in FTS5 (or keep raw chunks at lower vector weight) so an LLM paraphrase or omission never destroys exact-phrase search (`"strict=True"`).
4. **Content-hash caching & budget controls:** Key sidecars by the source file's BLAKE2b hash + interpreter prompt/schema version (`model + schema_v`), skip already-interpreted files, and support `--limit` / `--path` filters (mirroring `codesearch embed`).
5. **Strict schema validation:** If the local model returns invalid JSON or hallucinates out-of-range line numbers, fall back cleanly to `md_parser.parse_text()` and record a warning in `problems`.

---

## 3. Target Intermediate JSON Schema

```json
{
  "source_file": "notes/braindump_2026_11.txt",
  "source_hash": "blake2b...",
  "doc_type": "notes | transcript | story_draft | log",
  "summary": "High-level 2-sentence overview of the document.",
  "dates_mentioned": ["2026-11-12"],
  "entities": ["codesearch", "VirtualWorld", "Memory Intermediator"],
  "tags": ["architecture", "tasks-mcp", "embeddings"],
  "sections": [
    {
      "title": "Date Filtering Idea",
      "source_lines": [1, 18],
      "kind": "topic",
      "tags": ["codesearch", "date-filter"],
      "content": "Structured faithful summary + preserved key quotes/terms."
    }
  ]
}
```

Wait note on indexing: because `json_parser.py` routes short values (`tags`, `entities`, `dates_mentioned`) to **keyword-only `tag` nodes** and routes `content` / `summary` to **vector-embedded `section` nodes** (with sentence children above 120 chars), reusing `json_parser` gives rich multi-part search almost for free once line-number mapping points back to the original `.txt`!

---

## 4. Implementation Steps

1. Prototype `src/codesearch/interpreter.py` with schema validation and line-span verification against the input `.txt`.
2. Add CLI subcommand `codesearch interpret [--path ...] [--limit N] [--model ...]` using the OpenAI-compatible endpoint (`CODESEARCH_LLM_ENDPOINT` / LM Studio).
3. Update `indexer.py` to merge validated sidecar sections for matching hashes during `index()`, remapping `file_id` / `line_start` / `line_end` back to the `.txt` file row so `get_code` opens the raw `.txt`.
4. Benchmark retrieval accuracy vs. naive `md_parser.parse_text()` on real unstructured notes.
