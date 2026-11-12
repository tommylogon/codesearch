# Task: Smart Chat & Session Parser for codesearch

**Status:** Planned / Todo  
**Area:** Parsers & Ingestion  
**Target:** Ingest JSON/JSONL chat exports and SQLite agent session databases into `codesearch` without polluting code AST graphs.

---

## 1. Objective
Support indexing conversational AI data (ChatGPT exports, Claude/DeepSeek JSON exports, Google AI Studio JSON dumps, Kilo/OpenCode session SQLite databases, and transcript JSONLs) so users can semantically search past discussions, architecture decisions, and code prompts with `codesearch` at full vector speed.

---

## 2. The Core Problem (The Noise Tsunami)
Raw conversation logs and agent databases contain massive amounts of noisy terminal output, bash traces, npm logs, git diffs, and intermediate tool scratchpads (e.g. `kilo.db` has over 900,000 raw events and 270,000 parts). 

Dumping raw chat events directly into the code graph would:
- Drown real code search results in old conversation logs.
- Cause keywords like function names to match chat transcripts rather than source code definitions.

---

## 3. Architecture & Requirements

### A. New Node Kind & Language
- Add `kind = 'conversation'` (or `kind = 'turn'`) to `codesearch`.
- Add `language = 'chat'` or `language = 'json'` metadata.
- Exclude `kind = 'conversation'` from default code-only queries unless explicitly requested or when `--kind conversation` is passed (or via separate search mode).

### B. Supported Input Formats
1. **Generic JSON/JSONL Chat Exports:**
   - OpenAI / ChatGPT (`conversations.json`)
   - Claude / Anthropic exports
   - Google AI Studio prompt/session JSON
   - DeepSeek JSON exports
2. **Local Agent Session Databases:**
   - Kilo (`.local/share/kilo/kilo.db`)
   - OpenCode (`.local/share/opencode/opencode.db`)
   - Roo-Cline / Claude-Dev task JSON logs (`ui_messages.json`)

### C. Smart Ingestion & Filtering Rules
- **Extract only high-signal text:**
  - `role: user`: The user prompt / problem statement / instructions.
  - `role: assistant`: The final markdown synthesized response.
- **Strictly ignore:**
  - Tool execution dumps (bash command stdout, test traces, file diff chunks).
  - Base64 images and inline binary assets.
  - Intermediate thinking/monologue scratchpads unless flagged for inclusion.
- **Graph Linking:**
  - Link conversation nodes to any files or repo paths referenced in the chat via `REFERENCES` edges.

---

## 4. Implementation Steps
1. Create `src/codesearch/parsers/chat_parser.py`:
   - Inspect JSON/JSONL structure to detect schema (ChatGPT, AI Studio, DeepSeek, generic messages list).
   - Normalize into `{ "session_id", "title", "turns": [ {"role": "user"|"assistant", "text": "..."} ] }`.
2. Add SQLite extractor helper (`src/codesearch/extractors/agent_db.py`):
   - Read-only (`?mode=ro`) queries against `kilo.db` / `opencode.db` message tables.
   - Aggregate user questions and final assistant responses per session.
3. Update `config.py` and `search.py`:
   - Add `CONVERSATION` kind.
   - Default `search_code` to exclude `conversation` nodes unless `kind='conversation'` or `include_chat=True`.
4. Verification:
   - Ingest an exported chat JSON.
   - Run `codesearch search "pc upgrade paths" --kind conversation` and verify it returns the exact conversation turn in <3 seconds.
