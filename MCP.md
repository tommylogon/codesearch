# codesearch MCP Protocol & Integration Guide

Complete reference for connecting any Model Context Protocol (MCP) compatible AI agent, IDE, or client to `codesearch`.

Includes universal configuration templates, parameter schemas, and full JSON request/response payloads for all 7 native tools.

---

## 1. Generic MCP Client Configuration

`codesearch` communicates via standard input/output (`stdio`) using the official FastMCP transport.

### Universal Client Configuration Schema

Add to your tool's MCP configuration file (e.g. `claude_desktop_config.json`, `.cursor/mcp.json`, `cline_mcp_settings.json`, or `antigravity.json`):

```json
{
  "mcpServers": {
    "codesearch": {
      "command": "python",
      "args": ["-m", "codesearch", "mcp"],
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "CODESEARCH_ROOT": "/path/to/your/workspace",
        "CODESEARCH_EMBED_ENDPOINT": "http://localhost:1234/v1",
        "CODESEARCH_EMBED_MODEL": "text-embedding-qwen3-embedding-0.6b"
      }
    }
  }
}
```

### Multi-Workspace Setup
To connect multiple distinct codebases or world bibles in the same client, register separate server instances pointing to different roots:

```json
{
  "mcpServers": {
    "codesearch-engine": {
      "command": "python",
      "args": ["-m", "codesearch", "-r", "F:/AI/viwo/virtual-world", "mcp"]
    },
    "codesearch-lore": {
      "command": "python",
      "args": ["-m", "codesearch", "-r", "C:/Users/Tommy/Documents/Books/hallowed are the path of chaos", "mcp"]
    }
  }
}
```

---

## 2. Tool Reference with Request & Response Payloads

### `search_code`
Hybrid BM25 keyword search fused with 4-part semantic vectors via Reciprocal Rank Fusion (RRF).

**Parameters:**
- `query` *(string, required)*: Natural language concept or exact identifier.
- `part` *(string, optional)*: Specific vector part (`"filename"`, `"function_name"`, `"function"`, `"comment"`).
- `kind` *(string, optional)*: Filter by node kind (`"file"`, `"class"`, `"function"`, `"method"`, `"variable"`, `"comment"`, `"section"`, `"block"`, `"record"`).
- `language` *(string, optional)*: Filter by language (`"python"`, `"typescript"`, `"javascript"`, `"markdown"`, `"html"`, `"json"`, `"text"`).
- `path` *(string, optional)*: Repo-relative path prefix.
- `match_all` *(boolean, optional, default: true)*: `true` for AND keyword logic, `false` for OR.
- `strict` *(boolean, optional, default: false)*: Only return exact keyword hits; vectors only re-rank.
- `limit` *(integer, optional, default: 8)*: Max results to return.

**Example Request:**
```json
{
  "name": "search_code",
  "arguments": {
    "query": "Florum rainbow rose",
    "kind": "section",
    "limit": 2
  }
}
```

**Example Response:**
```json
{
  "results": [
    {
      "id": 203714,
      "name": "4. Wands, The Living Rose (*Florum*), & The Viaduct Fireball",
      "qualified_name": "scenes/2006-09-01_the_hogwarts_express_expanded_journey.md::4. Wands, The Living Rose (*Florum*), & The Viaduct Fireball",
      "kind": "section",
      "file": "scenes/2006-09-01_the_hogwarts_express_expanded_journey.md",
      "language": "markdown",
      "lines": [259, 410],
      "signature": "",
      "doc": "Jake snapped the stem and handed the glowing iridescent flower to Saoirse...",
      "score": 0.0412,
      "matched": ["keyword", "vec:comment"],
      "ranks": {
        "keyword": {"rank": 1, "raw": 5.1204},
        "vec:comment": {"rank": 2, "raw": 0.8412}
      },
      "relations": {
        "links_to": {
          "count": 2,
          "items": ["wiki/Florum.md (wiki/Florum.md:1)", "wiki/Wyvern.md (wiki/Wyvern.md:1)"]
        },
        "linked_from": {
          "count": 1,
          "items": ["Act III: The Panic & The Cover-Up (scenes/2006-09-01_004_viaduct_blast.md:78)"]
        }
      }
    }
  ]
}
```

---

### `find_references`
Locates everything that references, imports, calls, or links to a symbol across the workspace.

**Parameters:**
- `symbol` *(string, required)*: Identifier, class name, method (`Class::method`), or section heading.

**Example Request:**
```json
{
  "name": "find_references",
  "arguments": {
    "symbol": "Florum"
  }
}
```

**Example Response:**
```json
{
  "symbol": "wiki/spells.md::Florum",
  "kind": "section",
  "file": "wiki/spells.md",
  "line": 42,
  "linked_from": {
    "count": 2,
    "items": [
      "scenes/2006-09-01_the_hogwarts_express_expanded_journey.md:259",
      "scenes/2006-09-01_003_wand_show_and_tell_and_florum.md:92"
    ]
  },
  "referenced_by": {
    "count": 0,
    "items": []
  }
}
```

---

### `code_neighbors`
Walks callers, callees, and variable references transitively up to N hops.

**Parameters:**
- `symbol` *(string, required)*: Function, class, method, or variable name.
- `direction` *(string, optional, default: "both")*: `"callers"`, `"callees"`, or `"both"`.
- `depth` *(integer, optional, default: 1)*: Traversal depth (1 to 4).

**Example Request:**
```json
{
  "name": "code_neighbors",
  "arguments": {
    "symbol": "PlayerManager::add_player",
    "direction": "callers",
    "depth": 1
  }
}
```

**Example Response:**
```json
{
  "symbol": "engine/player_manager.py::PlayerManager::add_player",
  "file": "engine/player_manager.py",
  "line": 232,
  "other_matches": [],
  "callers": [
    {
      "name": "WorldEngine::initialize_characters",
      "file": "engine/engine.py",
      "line": 145
    },
    {
      "name": "register_routes",
      "file": "routes/player_ops.py",
      "line": 34
    }
  ]
}
```

---

### `file_outline`
Returns the structural skeleton of a file (classes, methods, functions, Markdown sections) with exact line spans before reading code.

**Parameters:**
- `path` *(string, required)*: Repo-relative file path.

**Example Request:**
```json
{
  "name": "file_outline",
  "arguments": {
    "path": "engine/effect_handlers/memory.py"
  }
}
```

**Example Response:**
```json
{
  "file": "engine/effect_handlers/memory.py",
  "language": "python",
  "lines": 144,
  "items": [
    {"kind": "function", "name": "grant_memory", "lines": [22, 54], "signature": "def grant_memory(world, target_id, **kwargs)"},
    {"kind": "function", "name": "surface_memory", "lines": [56, 92], "signature": "def surface_memory(world, target_id, tags=None, keywords=None)"},
    {"kind": "function", "name": "suppress_memory", "lines": [94, 122], "signature": "def suppress_memory(world, target_id, duration=0)"},
    {"kind": "function", "name": "unblock_memory", "lines": [124, 138], "signature": "def unblock_memory(world, target_id)"}
  ]
}
```

---

### `get_code`
Reads exact numbered lines from disk with optional context padding.

**Parameters:**
- `path` *(string, required)*: Repo-relative file path.
- `start` *(integer, required)*: First line number to read (1-based).
- `end` *(integer, required)*: Last line number to read.

**Example Request:**
```json
{
  "name": "get_code",
  "arguments": {
    "path": "engine/effect_handlers/memory.py",
    "start": 22,
    "end": 35
  }
}
```

**Example Response:**
```json
{
  "path": "engine/effect_handlers/memory.py",
  "start": 22,
  "end": 35,
  "total": 144,
  "lines": [
    "def grant_memory(world, target_id, text, importance=5, memory_type='observation', tags=None, source='trigger'):",
    "    \"\"\"Adds a memory entry to a target player.\"\"\"",
    "    player = world.get_player(target_id)",
    "    if not player:",
    "        return False",
    "    if isinstance(tags, str):",
    "        tags = [t.strip() for t in tags.split(',') if t.strip()]",
    "    player.add_memory(text=text, tick=world.current_tick, importance=importance, memory_type=memory_type, tags=tags, source=source)",
    "    return True"
  ]
}
```

---

### `render_code_graph`
Renders an annotated Set-of-Mark (`#id`) dark-theme architecture diagram for multimodal vision models (Claude 3.7, GPT-4o, Gemini).

**Parameters:**
- `symbol` *(string, required)*: Symbol name or `#node_id`.
- `depth` *(integer, optional, default: 1)*: Call graph depth.
- `expand` *(string, optional)*: Comma-separated list of `#id` values to expand in place on the canvas.
- `kinds` *(string, optional)*: Comma-separated edge filter (e.g. `"CALLS,IMPORTS"`).
- `return_image` *(boolean, optional, default: false)*: If true, returns binary image payload; if false, returns local PNG file path.

**Example Request:**
```json
{
  "name": "render_code_graph",
  "arguments": {
    "symbol": "PlayerManager",
    "depth": 1
  }
}
```

**Example Response:**
```json
{
  "image_path": "C:/Users/Tommy/.gemini/antigravity/brain/.../graph_PlayerManager.png",
  "node_count": 8,
  "edge_count": 12,
  "roots": ["PlayerManager"],
  "tip": "Inspect the image to see visual badges (#1, #2). Call again with expand='1,2' to expand branches."
}
```

---

### `codesearch_stats`
Reports workspace index health, total files, AST nodes, relational edges, and vector dimensions.

**Example Request:**
```json
{
  "name": "codesearch_stats",
  "arguments": {}
}
```

**Example Response:**
```json
{
  "repository": "hallowed are the path of chaos",
  "files": 24955,
  "nodes": 202142,
  "edges": 48120,
  "files_by_language": {
    "markdown": 24910,
    "python": 35,
    "json": 10
  },
  "vector_dimensions": 1024,
  "status": "ready"
}
```
