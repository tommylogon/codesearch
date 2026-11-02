# codesearch: Agent & MCP Integration Guide

Portable connection and usage guide for connecting any LLM coding agent, IDE, or orchestrator (Claude Desktop, Cursor, Kilo, Cline, Roo Code, Windsurf, OpenCode, or Antigravity) to `codesearch`.

---

## 1. Connecting the MCP Server (`stdio`)

`codesearch` runs a local `FastMCP` server over `stdio` at `F:/AI/viwo/virtual-world/tools/codesearch/mcp_server.py` (or `F:/AI/viwo/virtual-world/tools/codesearch.py mcp`).

Add one or more entries to your client's `mcpServers` JSON configuration:

### A. For `virtual-world` (default repo root)
```json
{
  "mcpServers": {
    "codesearch": {
      "command": "python",
      "args": [
        "F:/AI/viwo/virtual-world/tools/codesearch.py",
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

### B. For any other folder (e.g. `millwood` or another project)
Pass `--root <folder>` in `args` (or `CODESEARCH_ROOT` in `env`). Each folder uses its own isolated `<folder>/.codesearch/codesearch.db` database:

```json
{
  "mcpServers": {
    "millwood-search": {
      "command": "python",
      "args": [
        "F:/AI/viwo/virtual-world/tools/codesearch.py",
        "--root",
        "C:/Users/Tommy/Documents/Books/millwood",
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
- `search_code(query, part=None, kind=None, language=None, path=None, match_all=True, strict=False, limit=8)`
  - Hybrid keyword + semantic search. Every result already includes its 1-hop graph relations (`called_by`, `calls`, `referenced_by`, `reads`, `writes`).
  - `part`: `"filename"` | `"function_name"` | `"function"` | `"comment"` (use `"comment"` for docstrings, design comments, and Markdown/JSON prose).
  - `kind`: `"file"` | `"class"` | `"function"` | `"method"` | `"variable"` | `"comment"` | `"section"` | `"block"` | `"element"`.
  - `language`: `"python"` | `"typescript"` | `"javascript"` | `"markdown"` | `"json"` | `"html"` | `"text"`.
  - `match_all=False`: match ANY word in the keyword leg (`OR`) when searching a bag of synonyms.
  - `strict=True`: require a keyword match (vectors only re-rank).
- `code_neighbors(symbol, direction="both", depth=1)`
  - Walk `callers`, `callees`, or `both` up to `depth=4` hops for any function, method (`Class::method`), or variable.
- `find_references(symbol)`
  - List every caller, importer, subclass, wiki link, and `[r]` read / `[w]` write site for a symbol or attribute.
- `file_outline(path)`
  - Return the structural skeleton of a file (classes, functions, variables, Markdown headings, JSON sections) with `[line_start, line_end]` spans before reading code.
- `get_code(path, start, end)`
  - Read exact numbered lines `[start, end]` from disk.
- `render_code_graph(symbol, depth=1, expand="", edge_kinds="", return_image=False)`
  - For vision-capable agents: renders a Set-of-Mark (`#id`) dark-theme subgraph PNG where every node card shows `#id`, name, kind, and `file:line`. Pass any `#id` from the image into `expand="id1,id2"` on the next call to expand those nodes in place.
- `codesearch_stats()`
  - Check index health, file/node/edge counts, and vector coverage.

### CLI Fallback (if MCP is unavailable in a shell-only agent)
Run `python F:/AI/viwo/virtual-world/tools/codesearch.py` (add `-r <folder>` to target another workspace):
- `python tools/codesearch.py "<query>" [--part comment] [--kind method] [--lang python] [--path engine] [--any] [--strict] [--full]`
- `python tools/codesearch.py callers "<symbol>"`
- `python tools/codesearch.py callees "<symbol>"`
- `python tools/codesearch.py refs "<symbol>"`
- `python tools/codesearch.py outline <rel_path>`
- `python tools/codesearch.py render "<symbol_or_#id>" --expand "<#id1,#id2>"`
- `python tools/codesearch.py index` (incremental update after editing files)
- `python tools/codesearch.py embed` (embed new/changed nodes via LM Studio)
```
