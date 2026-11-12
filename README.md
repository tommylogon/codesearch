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
- **Multimodal Image Retrieval:** Indexes image files (.png, .jpg, .webp) with dual-space embeddings (filename text + image-server pixel embeddings).
- **Tabular CSV & JSON Parsing:** Fast extraction of tabular rows, column schemas, and structured documents.
- **Remote Web API for External Agents:** Expose a read-only, CORS-enabled HTTP interface so remote or web-based agents (e.g. Qwen, GLM, Claude web, or custom frontends) can explore your local codebase safely.
- **Worldbuilding & Lore Engine:** Treats sprawling fiction repos like stateful codebases. See [CREATIVE_WRITING.md](CREATIVE_WRITING.md) for how it tracks character dossiers, scene continuity, evidence inventories, and epistemic boundaries.
- **Batteries-Included MCP Server:** Ready to plug into Claude Desktop, Cursor, Cline, Windsurf, and Antigravity.

---

## Supported File Formats

`codesearch` indexes source code, tabular datasets, narrative prose, and visual media out of the box:

| Category | Extensions | Indexing & Extraction Depth |
| :--- | :--- | :--- |
| **Code** | `.py` | AST hierarchy (classes, methods, functions, calls, imports, docstrings). |
| | `.ts` | TypeScript AST via Acorn/Node.js bridge. |
| | `.js`, `.mjs`, `.cjs` | JavaScript AST (ES modules, CommonJS, functions, references). |
| | `.html` | HTML structure and embedded `<script>` AST extraction. |
| **Prose & Transcripts** | `.md` | Markdown sections, `# headings`, wikilinks (`[[...]]`), and prose text. |
| | `.txt` | Plain text documents and segmented notes. |
| | `.srt`, `.vtt` | Video subtitle and dialogue transcript timelines. |
| **Data & Tables** | `.json`, `.jsonl` | Nested key paths, schemas, and row-by-row JSONL records. |
| | `.csv` | Tabular data rows, column headers, and structured cell values. |
| **Images & Media** | `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp` | Multimodal dual-indexing: text embeddings for path/filename + visual pixel embeddings via CUDA sidecar. |

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/tommylogon/codesearch.git
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

## Local Embeddings (Recommended Setup)

By default, `codesearch-mcp` delivers fast lexical BM25 ranking. For semantic conceptual search, it supports local OpenAI-compatible embedding endpoints with automatic dimension detection and cosine vector fusion.

### Tested and Recommended Configuration

- **Model:** text-embedding-qwen3-embedding-0.6b
- **Dimensions:** 1024 vectors
- **Context Window:** 32k context length (allows indexing large functions, multi-turn transcripts, and full document sections without truncation)
- **Local Server:** LM Studio (port 1234) or Ollama (port 11434)

### Running the Embedder

Load your embedding model into LM Studio or Ollama, then run:

```bash
# Embed up to 5,000 pending snippets via LM Studio (default: http://localhost:1234/v1/embeddings)
codesearch embed --limit 5000

# Or point to Ollama with custom dimensions
CODESEARCH_EMBED_ENDPOINT="http://localhost:11434/v1/embeddings" \
CODESEARCH_EMBED_MODEL="text-embedding-qwen3-embedding-0.6b" \
codesearch embed --limit 5000
```

The embedder handles exponential backoff, batch splitting, and incremental vector storage directly in `.codesearch/codesearch.db`.

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

## Multimodal Image Embeddings

`codesearch` supports dual-space visual retrieval for UI screenshots, concept art, and diagrams:
1. File paths and names are embedded into standard text space.
2. Image pixels are encoded into vision vector space via a lightweight local sidecar.

```bash
# Start the visual embedding sidecar (default port 1235)
codesearch image-server --port 1235

# Index a repo including image assets
codesearch index --images

# Embed image pixels into the visual vector space
codesearch embed --images
```

---

## Remote Web API for Web Agents

Allow web-based AI assistants (such as GLM, Qwen, or custom browser agents) to query your local indexes over HTTP without granting shell access or cloud repository uploads.

The public runner provides a sandboxed, read-only interface:
- **Zero Shell Execution:** Strips execution hooks like `code --goto`.
- **Path Confinement:** Every query is strictly validated against the target repository root; attempts to navigate outside via `..` return `403 Forbidden`.
- **CORS Enabled:** Supports cross-origin browser `fetch()` requests and preflights.

### Running the API Server

```bash
# Run open (no password):
python serve_public_viwo.py --port 8085 --host 0.0.0.0

# Or protect with a password / key:
python serve_public_viwo.py --port 8085 --host 0.0.0.0 --key mysecretpassword
```

### Endpoints & Parameters

- `GET /api/indexes`: List all available workspace indexes (e.g. `viwo`, `hogwarts`).
- `GET /api/stats?index=hogwarts`: Node counts, file counts, and language breakdown for a specific workspace.
- `GET /api/search?q=<query>&index=hogwarts&limit=20`: Hybrid lexical + vector search across the chosen index.
- `GET /api/code?path=<relative_path>&index=hogwarts&start=1&end=50`: Sandboxed source line reader.
- `GET /api/node/<id>?index=hogwarts`: AST node details and graph relations.
- `GET /api/graph/<id>?index=hogwarts&depth=1`: Neighborhood call graph.

When password protection is enabled, pass your secret via query parameter (`?key=mysecretpassword` or `?password=...`) or via standard HTTP header (`Authorization: Bearer mysecretpassword` / `X-API-Key: ...`).

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
