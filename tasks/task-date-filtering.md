# Task: Date & Modification-Time Filtering (`after` / `before`)

**Status:** Phase 1 Implemented (File `mtime`); Phase 2 Planned (Extracted Document Dates)
**Area:** Search Engine (`src/codesearch/search.py`, `cli.py`, `mcp_server.py`, `app.py`, `serve_public_viwo.py`)
**Target:** Filter hybrid keyword and semantic search results by file modification time and, later, extracted in-document dates.

---

## 1. Current State & Honest Assessment

Before this update, `codesearch` indexed file modification timestamps (`files.mtime` stored as `REAL` Unix epoch seconds in SQLite `src/codesearch/db.py`), and `csv_parser.py` recognized `"dato"` / `"date"` columns only as row ID candidates. None of the search interfaces exposed any date or time filter:
- `Searcher._filters()` only filtered on `n.kind`, `f.language`, `f.rel_path`, and `PART_KINDS`.
- No CLI, MCP, or HTTP parameter existed for filtering by date.

### What Was Shipped (Phase 1: File `mtime`)

Phase 1 adds `after` and `before` filtering on `files.mtime` across the entire query stack:
- **`src/codesearch/search.py`**:
  - Added `parse_date_bound(value, *, end_of_day=False)` supporting `YYYY-MM-DD` (expanding `before` to `23:59:59.999999` local time so whole-day inclusive filters work naturally), full ISO-8601 datetimes (including `Z` UTC suffix under Python 3.10), and Unix epoch numbers/strings.
  - Updated `Searcher.search`, `Searcher.keyword`, `Searcher._allowed`, and `Searcher._filters` so both FTS5 keyword hits and dense vector hits respect `f.mtime >= after_ts` and `f.mtime <= before_ts`.
- **`src/codesearch/cli.py`**: Added `--after` and `--before` options on the `search` subcommand.
- **`src/codesearch/mcp_server.py`**: Added `after` and `before` parameters on `search_code`, returning `{"error": ...}` on invalid date strings.
- **`src/codesearch/app.py` and `serve_public_viwo.py`**: Added `?after=...&before=...` query parameters on `/api/search` (HTTP 400 on malformed dates).
- **Verified in `tests/test_codesearch_db.py::test_date_filtering_keyword_semantic_and_mcp`**: Verified keyword-only filtering, semantic-only vector `_allowed` gating, and MCP `search_code` validation.

---

## 2. Phase 2: Why `mtime` Alone Is Not Enough (Planned)

Filesystem `mtime` is fragile: `git clone`, file copies, batch formatting, or automated migrations bump `st_mtime` without changing when an event, journal entry, or document actually occurred.

Phase 2 should add semantic document/node dates without breaking existing zero-migration queries:
1. **Optional `node_dates` or `doc_date` column (`db.py` schema v2 / migration)**:
   - Extract normalized ISO timestamps (`YYYY-MM-DD` or epoch range) at parse time:
     - Markdown YAML frontmatter (`date`, `filed`, `created`, `updated`).
     - JSON / JSONL records (`timestamp`, `created_at`, `updated_at`, `filed`, `tick`).
     - CSV date columns (`date`, `dato`, `timestamp`).
     - Optional Git log fallback (`git log -1 --format=%ct -- <file>` for tracked files).
2. **Scope toggle (`--date-source mtime|doc|any`)**:
   - Let callers filter explicitly by filesystem `mtime` vs. extracted document/turn date (e.g., querying RP chat exports or patch notes written in October 2026 regardless of disk copy time).
3. **Hit metadata visibility**:
   - Return `mtime` and `doc_date` in materialized search results (`graph.get_node`) so agents see when a matched note/file was written or edited.
