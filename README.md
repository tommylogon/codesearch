# codesearch-mcp

> **Your files already know how they're structured. `codesearch` refuses to shred them, and lets any AI ask.**
>
> Parse every file at its natural grain (function, class, section, record, page, or image) instead of fixed-size token windows, wire intra-file and cross-file relationships into one local SQLite graph, and retrieve through exact keywords and semantic vectors fused together.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-Protocol-purple.svg)](https://modelcontextprotocol.io/)

---

## Core Vision & Pillars

1. **Natural grain over fixed windows:** The retrieval unit matches what the artifact actually is (a function, method, class, Markdown heading, JSON record, PDF section/page leaf, or image), never an arbitrary sliding window that slices code or narrative mid-thought.
2. **One graph, every medium:** Source code, documentation, configs, tabular datasets, multi-thousand-page PDFs, transcripts, and images live in the same SQLite graph with typed edges (`DEFINES`, `CALLS`, `REFERENCES`, `IMPORTS`, `INHERITS`, `ANNOTATES`, `CONNECTS_TO`, `LINKS_TO`). Design docs link straight to implementations; tables of contents link to chapters.
3. **Separate spaces, fused ranks:** Path/filename, symbol names, code signatures/bodies, prose/comments, and visual pixels each stay in their own vector space instead of blending into one muddled embedding. Native SQLite FTS5 BM25 and per-space cosine similarities meet through Reciprocal Rank Fusion (RRF).
4. **An index an agent can audit:** Search results return their receipts (`keyword` vs `vec:*` ranks, matching lines/pages, and 1-hop graph neighbors) so a model can inspect *why* something surfaced rather than trusting an opaque similarity score.
5. **Honest fidelity and no silent loss:** Each file format is decoded as deeply as its structure honestly supports. Stored node text is preserved without arbitrary slicing, parser depth/size bounds surface openly, and large documents like long PDFs expose bounded page windows instead of dumping megabytes into an agent context.
6. **Local, incremental, one file:** BLAKE2b content hashes, FTS5 tables, graph edges, and reusable vector caches live in one `.codesearch/codesearch.db` file per workspace. No Docker stack, no background daemon, no cloud upload.
7. **Agent-readable by default:** Built first for model consumers (CLI, `stdio` FastMCP, read-only CORS HTTP API, Set-of-Mark `#id` subgraph renders), paired with a local web UI for human inspection. Works across both codebases and sprawling multi-draft prose archives where continuity spans thousands of files.

### Scope & Honest Boundaries

- **What it is:** A single-file local structural graph and hybrid retrieval engine (plus an optional local embedder sidecar via LM Studio, Ollama, or standalone `llama-server`).
- **What it is not:** A full agent memory consolidation system or a distributed GraphRAG cluster. Cross-document prose links rely on explicit structure (PDF bookmarks/TOC entries, headings, anchors, and `[[wikilinks]]`) rather than free-form NLP inference over arbitrary sentences.

---

## Supported File Formats

> For a deep technical breakdown of how each extension is parsed, graphed, and routed across the 4+1 vector spaces, see [INDEXING_PIPELINE.md](INDEXING_PIPELINE.md).

`codesearch` extracts honest structure from source code, documents, structured data, transcripts, and images out of the box:

| Category | Extensions | Indexing & Extraction Depth |
| :--- | :--- | :--- |
| **Code** | `.py` | Native AST hierarchy (classes, methods, functions, variables, docstrings, calls, imports, `[r]`/`[w]` refs, inheritance). |
| | `.ts` | TypeScript AST via Acorn/Node.js strip-only transform (preserves exact source line numbers). |
| | `.js`, `.mjs`, `.cjs` | JavaScript AST (ES modules, CommonJS, arrow functions, object methods, references). |
| | `.html` | HTML element tree plus embedded `<script>` AST extraction wired into the JS call graph. |
| **Documents & Prose** | `.md` | Hierarchical `# Headings`, fenced code `block`s, anchors, and Obsidian-style `[[wikilinks]]` (`LINKS_TO` edges). |
| | `.pdf` | Hierarchical outline bookmarks, TOC cross-reference `LINKS_TO` edges, typographic heading detection, per-page leaf sections, and bounded page reading (capped per call). |
| | `.txt` | Heading/numbered-section structure when present, or bounded paragraph blocks without slicing sentences. |
| | `.srt`, `.vtt` | Timestamped subtitle cues and dialogue segments. |
| **Data & Tables** | `.json`, `.jsonl` | Recursive key paths (`record` shape, keyword-only `tag` scalars, prose `section` + sentence nodes) and relational `CONNECTS_TO` links. |
| | `.csv` | Column header schema summary and row-level records. |
| **Images & Media** | `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp` | Dual-space indexing: dimensions + path/filename text vector plus pixel embeddings via local vision sidecar. |

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
codesearch search "connect" --kind function --part function

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

# Prune orphaned vec_cache rows from inactive embedding models before embedding
codesearch embed --prune-stale --limit 5000

# Or point to Ollama with custom dimensions
CODESEARCH_EMBED_ENDPOINT="http://localhost:11434/v1/embeddings" \
CODESEARCH_EMBED_MODEL="text-embedding-qwen3-embedding-0.6b" \
codesearch embed --limit 5000
```

The embedder handles exponential backoff, batch splitting, and incremental vector storage directly in `.codesearch/codesearch.db`.

### Optional Stage-2 Reranking (`Qwen3-Reranker-0.6B`)

For higher precision after hybrid BM25 + vector RRF fusion, enable optional stage-2 cross-encoder reranking against a local `llama-server --reranking` endpoint (falls back cleanly to RRF order when offline):

```bash
CODESEARCH_RERANK_ENABLED=1 \
CODESEARCH_RERANK_ENDPOINT="http://127.0.0.1:1237/v1" \
CODESEARCH_RERANK_MODEL="Qwen3-Reranker-0.6B" \
codesearch search "causal fear resolution"
```

---

## Connecting to AI Agents (MCP)

> For full JSON request/response schemas, parameter types, and multi-workspace client configurations, see [MCP.md](MCP.md).

`codesearch-mcp` exposes FastMCP tools including:
- `search_code`: Hybrid lexical BM25 + vector search with explicit `must_terms` (AND), `should_terms` (OR), `not_terms` (NOT), `after`/`before` date filters, and optional `rerank` stage-2 scoring.
- `code_neighbors`: Trace AST caller/callee graphs and structural neighbors up to N hops.
- `find_references`: Find all usages, imports, and calls to a given symbol name.
- `file_outline`: Return a structured tree of classes, functions, sections, and line ranges.
- `get_code`: Fetch exact source code for a symbol without reading the whole file.
- `render_code_graph`: Render an annotated Set-of-Mark visual diagram of call graphs for vision models.
- `codesearch_stats`: Report file counts, indexed nodes, edges, active-model vector coverage, and stale model warnings.

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

Add to your MCP config (`~/.gemini/antigravity/mcp_config.json`) with `CODESEARCH_INDEXES` (or register folders via `codesearch registry scan` / `codesearch registry add`):

```json
{
  "mcpServers": {
    "codesearch": {
      "command": "python",
      "args": ["-m", "codesearch", "mcp"],
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "CODESEARCH_INDEXES": "app=/path/to/app;docs=/path/to/docs"
      }
    }
  }
}
```

Every MCP query/graph/code call requires an explicit `index` (call `list_indexes()` first to inspect registered indexes).

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
python serve_public_viwo.py --port 8085 --host 0.0.0.0 --add-index my-app=/path/to/codebase --add-index lore-repo=/path/to/docs

# Or protect with a password / key:
python serve_public_viwo.py --port 8085 --host 0.0.0.0 --key mysecretpassword --add-index my-app=/path/to/codebase
```

### Endpoints & Parameters

- `GET /api/indexes`: List all available workspace indexes (e.g. `my-app`, `lore-repo`).
- `GET /api/stats?index=my-app`: Node counts, file counts, and language breakdown for a specific workspace.
- `GET /api/search?q=<query>&index=my-app&limit=20`: Hybrid lexical + vector search across the chosen index.
- `GET /api/code?path=<relative_path>&index=my-app&start=1&end=50`: Sandboxed source line or bounded PDF page reader.
- `GET /api/node/<id>?index=my-app`: AST node details and graph relations.
- `GET /api/graph/<id>?index=my-app&depth=1`: Neighborhood call graph.

When password protection is enabled, pass your secret via query parameter (`?key=mysecretpassword` or `?password=...`) or via standard HTTP header (`Authorization: Bearer mysecretpassword` / `X-API-Key: ...`).

### Agent Compatibility & Sandboxing Notes

Not all web-based AI chat interfaces allow arbitrary outbound network access:

| Agent / Platform | Status | Network & Sandbox Behavior |
| :--- | :--- | :--- |
| **Qwen** | **Verified Working** | Tool-calling / bash environment can execute `curl` and `requests` against remote IP and port endpoints. |
| **GLM (chat.z.ai)** | **Verified Working** | Outbound tool execution works cleanly; successfully queries search, code, and graph endpoints over the web. |
| **ChatGPT (chatgpt.com)** | **Currently Restricted** | Web browsing blocks non-standard ports (such as 8085) and raw IP addresses for SSRF safety. Requires port 443 with a domain (via Cloudflare Tunnel or Tailscale Funnel), or an OpenAPI Action. |
| **Gemini (gemini.google.com)** | **Currently Restricted** | Cloud Python sandbox is airgapped with no outbound internet access. |
| **Claude (claude.ai)** | **Currently Restricted** | Standard web chat tools cannot reach external IP/ports unless running via client-side browser artifacts or extensions. |
| **DeepSeek (chat.deepseek.com)** | **Currently Restricted** | Standard web chat interface does not support arbitrary external network tools. |

---

## Architecture

```
codesearch-mcp
├── src/codesearch/
│   ├── config.py         # Tunables, file paths, embedding endpoint config
│   ├── db.py             # SQLite FTS5 schema, graph store, vector cache
│   ├── diag.py           # Rotating diagnostics and live stall watchdog
│   ├── indexer.py        # Incremental file scanner and AST/structure dispatcher
│   ├── embedder.py       # Multi-part text + multimodal image vector builder & packs
│   ├── embed_server.py   # Managed local llama-server embedding runner
│   ├── search.py         # Multi-field BM25 + cosine vector fusion (RRF)
│   ├── graph.py          # Relational callers/callees, outlines, and overview queries
│   ├── registry.py       # Persistent and env-based multi-index registry
│   ├── renderer.py       # Visual Set-of-Mark (#id) graph renderer
│   ├── mcp_server.py     # Multi-index FastMCP protocol interface
│   ├── app.py            # Local Flask web explorer
│   ├── cli.py            # Command line entry point
│   └── parsers/          # Extractors (py, ts/js, html, md, doc_structure, pdf, json, csv)
└── serve_public_viwo.py  # Sandboxed multi-index read-only CORS HTTP API server
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
