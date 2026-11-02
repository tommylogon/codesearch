# codesearch-mcp

> A lightning-fast, local-first hybrid code and documentation search engine for AI agents with AST call-graphs, SQLite FTS5, and FastMCP.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-Protocol-purple.svg)](https://modelcontextprotocol.io/)

---

## Why codesearch-mcp?

Most code-search tools for AI coding assistants are either:
1. **Naive keyword greppers** that get blinded by line-by-line matches and have no understanding of code structure or AST hierarchies.
2. **Bloated cloud-dependent vector monsters** that require Docker containers, gigabytes of dependencies, or paid cloud databases that upload your private code.

`codesearch-mcp` takes a different approach:
- **100% Local and Private:** Embedded SQLite with FTS5 BM25 ranking. Zero telemetry, zero cloud uploads.
- **Fast Incremental Indexing:** Indexes thousands of files across Python, TypeScript, JavaScript, HTML, Markdown, and JSON in seconds, caching hashes in SQLite.
- **True AST Code Graphs:** Resolves definitions, calls, imports, references, and class inheritance trees.
- **Set-of-Mark Visual Graph (`render_code_graph`):** Renders visual architecture diagrams labeled with numeric badges (`#id`) so multimodal agents (Claude 3.7, GPT-4o, Gemini) can visually inspect call trees and expand subgraphs.
- **Hybrid Lexical + Vector Retrieval:** Exact identifier matching combined with local embedding models (via LM Studio, Ollama, or any OpenAI-compatible endpoint).
- **Batteries-Included MCP Server:** Ready to plug into Claude Desktop, Cursor, Cline, Windsurf, and Antigravity.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/your-username/codesearch.git
cd codesearch
pip install -e .
```

*(Optional for TypeScript/JavaScript AST parsing):*
```bash
npm install
```

### 2. Index a Repository

Navigate to any codebase or specify a target directory:

```bash
# Index current directory
codesearch index

# Or index a specific workspace
codesearch index --root /path/to/project
```

### 3. Run Searches from the Terminal

```bash
# Fuzzy hybrid search
codesearch search "authentication flow"

# Filter by node type or kind
codesearch search "connect" --kind function --part code

# Trace call trees
codesearch callers handle_request
codesearch callees handle_request

# Find symbol references
codesearch refs UserSession

# Inspect file outline
codesearch outline src/server.py
```

### 4. Launch the Web UI

```bash
codesearch serve --port 5003
```
Open `http://localhost:5003` to explore code visually with an interactive search bar and canvas graph.

---

## Connecting to AI Agents (MCP)

`codesearch-mcp` exposes 7 native FastMCP tools:
- `search_code`: Hybrid lexical BM25 + vector search with snippet previews.
- `code_neighbors`: Trace AST caller/callee graphs and structural neighbors up to N hops.
- `find_references`: Find all usages, imports, and calls to a given symbol name.
- `file_outline`: Return a structured tree of classes, functions, sections, and line ranges.
- `get_code`: Fetch exact source code for a symbol without reading the whole file.
- `render_code_graph`: Render an annotated Set-of-Mark visual diagram of call graphs for vision models.
- `codesearch_stats`: Report file counts, indexed nodes, edges, and vector dimensions.

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "codesearch": {
      "command": "codesearch",
      "args": ["mcp"],
      "env": {
        "CODESEARCH_ROOT": "/path/to/your/codebase"
      }
    }
  }
}
```

### Cursor

In Cursor Settings -> Features -> MCP Servers -> Add New MCP Server:
- **Name:** `codesearch`
- **Type:** `command`
- **Command:** `codesearch mcp --root /path/to/your/codebase`

### Cline / Roo Code

Add to your `cline_mcp_settings.json`:

```json
{
  "mcpServers": {
    "codesearch": {
      "command": "codesearch",
      "args": ["mcp", "--root", "/path/to/your/codebase"]
    }
  }
}
```

### Antigravity

Add to your `.gemini/antigravity.json` or workspace settings:

```json
{
  "mcp_servers": {
    "codesearch": {
      "command": "codesearch",
      "args": ["mcp", "--root", "${workspaceFolder}"]
    }
  }
}
```

---

## Local Embeddings (Optional)

By default, `codesearch-mcp` provides fast, accurate lexical BM25 ranking. To enable vector search:

1. Launch your local model runner (LM Studio or Ollama) with an embedding model (e.g. `text-embedding-nomic-embed-text-v1.5` or `bge-small-en-v1.5`).
2. Run the embedder:

```bash
# Embed up to 5,000 pending snippets using LM Studio (default: http://localhost:1234/v1)
codesearch embed --limit 5000

# Or point to Ollama / OpenAI
CODESEARCH_EMBED_URL="http://localhost:11434/v1/embeddings" CODESEARCH_EMBED_MODEL="nomic-embed-text" codesearch embed
```

---

## Architecture

```
codesearch-mcp
├── src/codesearch/
│   ├── config.py         # Tunables, file paths, embedding endpoint config
│   ├── db.py             # SQLite FTS5 schema, graph store, vector cache
│   ├── indexer.py        # Incremental file scanner and AST dispatcher
│   ├── search.py         # Multi-field BM25 + cosine vector fusion
│   ├── graph.py          # AST call/callee graph construction
│   ├── renderer.py       # Visual Set-of-Mark graph generator
│   ├── mcp_server.py     # FastMCP protocol interface
│   ├── app.py            # Local Flask web explorer
│   ├── cli.py            # Command line entry point
│   └── parsers/          # Language extractors (py, ts, js, html, md, json)
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
