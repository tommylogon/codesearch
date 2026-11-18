# codesearch: Agent & MCP Integration Guide

Portable connection and usage guide for connecting any LLM coding agent, IDE, or orchestrator (Claude Desktop, Cursor, Kilo, Cline, Roo Code, Windsurf, OpenCode, or Antigravity) to `codesearch`.

---

## 1. Connecting the MCP Server (`stdio`)

`codesearch` runs a local `FastMCP` server over `stdio` at `F:/AI/codesearch/src/codesearch/mcp_server.py` (or `F:/AI/codesearch/src/codesearch/cli.py mcp`).

Add one or more entries to your client's `mcpServers` JSON configuration:

### A. For `virtual-world` (default repo root)
```json
{
  "mcpServers": {
    "codesearch": {
      "command": "python",
      "args": [
        "F:/AI/codesearch/src/codesearch/cli.py",
        "mcp"
      ],
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "CODESEARCH_EMBED_ENDPOINT": "http://localhost:1234/v1",
        "CODESEARCH_EMBED_MODEL": "text-embedding-qwen3-embedding-0.6b"
      }
    }
  }
}
```

### B. Multi-index MCP Setup (Recommended)
Every index-scoped MCP tool requires an explicit `index` parameter (alias or indexed directory path). Register all your repositories once via CLI or pass them in `CODESEARCH_INDEXES` (`name=path;name2=path2` or JSON):

```bash
python -m codesearch registry scan /path/to/projects
python -m codesearch registry add my-docs /path/to/docs --name docs --desc "Project design notes"
```

Or configure multiple indexes directly in `env`:

```json
{
  "mcpServers": {
    "codesearch": {
      "command": "python",
      "args": [
        "-m",
        "codesearch",
        "mcp"
      ],
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "CODESEARCH_INDEXES": "app=/path/to/app;docs=/path/to/docs",
        "CODESEARCH_EMBED_ENDPOINT": "http://127.0.0.1:1234/v1",
        "CODESEARCH_EMBED_MODEL": "text-embedding-qwen3-embedding-0.6b"
      }
    }
  }
}
```

### Where each client stores `mcpServers`
- **Antigravity**: `C:\Users\Tommy\.gemini\antigravity\mcp_config.json`
- **Claude Desktop (Windows)**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Cursor**: `<workspace>\.cursor\mcp.json` or `~/.cursor/mcp.json`
- **Kilo / Roo Code / Cline**: `.kilo/mcp.json` or VS Code global Storage `mcp_settings.json`
- **Claude Code CLI**: `.mcp.json` in the workspace root

---

## 2. Drop-In System Prompt / Agent Rule Snippet

Copy the block below into any agent's system prompt, `AGENTS.md`, `.cursorrules`, `CLAUDE.md`, or `.gemini/rules/*.md` so the agent knows how to use `codesearch` via MCP or CLI:

```markdown
## Code & Lore Search (`codesearch`)

Prefer `codesearch` over blind `grep` loops or reading entire large files. `codesearch` combines SQLite FTS5 BM25 keyword search (with camelCase/snake_case splitting) and 4 separate semantic vector indexes (`filename`, `function_name`, `function`, `comment`) plus a resolved graph of `DEFINES`, `CALLS`, `REFERENCES` (`[r]`/`[w]`), `IMPORTS`, `INHERITS`, `ANNOTATES`, and `LINKS_TO` edges.

### MCP Tools (`codesearch`)
Every index-scoped tool REQUIRES `index` (alias from `list_indexes()` or indexed repository path). No default index is assumed.

- `list_indexes()`
  - List all registered/discovered indexes with paths, readiness, and vector-pack status. Call this first if unsure which `index` to pass.
- `search_code(index, query="", part=None, kind=None, language=None, path=None, strict=False, must_terms=None, should_terms=None, not_terms=None, rerank=None, limit=8, after=None, before=None)`
  - Hybrid keyword + semantic search. Every result already includes its 1-hop graph relations (`called_by`, `calls`, `referenced_by`, `reads`, `writes`).
  - `part`: `"filename"` | `"function_name"` | `"function"` | `"comment"` (use `"comment"` for docstrings, design comments, and Markdown/JSON prose).
  - `kind`: `"file"` | `"class"` | `"function"` | `"method"` | `"variable"` | `"comment"` | `"section"` | `"block"` | `"element"`.
  - `language`: `"python"` | `"typescript"` | `"javascript"` | `"markdown"` | `"json"` | `"csv"` | `"html"` | `"text"` | `"pdf"`.
  - `must_terms` / `should_terms` / `not_terms`: explicit AND / OR / NOT boolean term lists (`not_terms` also excludes semantic hits).
  - `strict=True`: require a keyword match (vectors only re-rank).
  - `rerank=True`: run optional stage-2 cross-encoder reranking (e.g. `Qwen3-Reranker-0.6B` via llama.cpp `/v1/rerank`).
- `code_neighbors(index, symbol, direction="both", depth=1)`
  - Walk `callers`, `callees`, or `both` up to `depth=4` hops for any function, method (`Class::method`), or variable.
- `find_references(index, symbol)`
  - List every caller, importer, subclass, wiki link, and `[r]` read / `[w]` write site for a symbol or attribute.
- `file_outline(index, path)`
  - Return the structural skeleton of a file (classes, functions, variables, Markdown headings, JSON sections, PDF hierarchy) with `[line_start, line_end]` spans (page numbers for PDFs) before reading content.
- `get_code(index, path, start, end)`
  - Read exact numbered lines `[start, end]` from disk (or 1-based pages, capped at 20 pages/call, for `.pdf` files).
- `render_code_graph(index, symbol, depth=1, expand="", edge_kinds="", return_image=False)`
  - For vision-capable agents: renders a Set-of-Mark (`#id`) dark-theme subgraph PNG where every node card shows `#id`, name, kind, and `file:line`. Pass any `#id` from the image into `expand="id1,id2"` on the next call to expand those nodes in place.
- `codesearch_stats(index)`
  - Check index health, file/node/edge counts, vector-pack state, and log path.
- `codesearch_logs(lines=80, index=None, component="mcp")`
  - Inspect rotating diagnostic logs (stage timings, embedder fallbacks, and slow-operation watchdogs).

### CLI Fallback (if MCP is unavailable in a shell-only agent)
Run `python F:/AI/codesearch/src/codesearch/cli.py` (or `python -m codesearch`, add `-r <folder>` to target another workspace):
- `python -m codesearch "<query>" [--part comment] [--kind method] [--lang python] [--path engine] [--any] [--strict] [--full]`
- `python -m codesearch callers "<symbol>"`
- `python -m codesearch callees "<symbol>"`
- `python -m codesearch refs "<symbol>"`
- `python -m codesearch outline <rel_path>`
- `python -m codesearch render "<symbol_or_#id>" --expand "<#id1,#id2>"`
- `python -m codesearch index` (incremental update after editing files)
- `python -m codesearch embed` (embed new/changed nodes via LM Studio)
```
