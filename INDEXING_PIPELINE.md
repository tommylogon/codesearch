# Extension Indexing & Multi-Vector Architecture

A deep technical breakdown of how `codesearch` parses, graphs, and embeds different file extensions, and why its structural approach differs fundamentally from traditional RAG.

---

## 1. Why Traditional Code RAG is Broken

Most vector databases and AI code search engines (LangChain, typical RAG pipelines, generic embeddings) use **fixed-size sliding window chunking**:
1. They blindly cut files every 500 or 1,000 tokens with a 100-token overlap.
2. A function definition gets sliced in half across two chunks.
3. A docstring is separated from its implementation.
4. A dialogue exchange between characters gets mangled in the middle of a sentence.
5. All text is shoved into a single vector space, so a casual comment mentioning `"player"` scores the same as `class Player:`.

`codesearch` rejects fixed-window chunking entirely. Every file extension is handled by a **syntax-aware structural parser** that respects the AST (Abstract Syntax Tree) or document hierarchy, preserving relational edges and routing different parts of each node into specialized vector spaces.

---

## 2. The 4+1 Multi-Vector Architecture

Instead of blending code, comments, and file paths into one generic embedding, `codesearch` splits nodes into **4 distinct text vector spaces plus 1 visual vision space**:

| Vector Space | What Gets Embedded | Purpose & Retrieval Behavior |
| :--- | :--- | :--- |
| **`filename`** | Normalized path, stem, and directory context | Matches high-level module concepts and architectural locations. |
| **`function_name`** | Bare identifier, qualified name, camelCase/snake_case split | Precision lookup for function and class definitions without noise from the body. |
| **`function`** | Signature plus implementation code body | Semantic matching of logic patterns, algorithms, and execution behavior. |
| **`comment`** | Docstrings, Markdown prose, design rationales | High-level intent, character traits, narrative beats, and domain concepts. |
| **`image`** | Raw pixel visual embeddings via local sidecar | Multimodal visual similarity for UI designs, sprites, portraits, and diagrams. |

These spaces are never crudely averaged together. At query time, `codesearch` runs independent cosine similarity searches against each space, then combines them with SQLite FTS5 BM25 keyword rankings using **Reciprocal Rank Fusion (RRF)**:

$$\text{RRF Score} = \sum_{m \in \text{Matches}} \frac{\text{Weight}_m}{K + \text{Rank}_m}$$

---

## 3. Extension-by-Extension Indexing Matrix

| Extension | Language | Parser Used | Nodes Created | Graph Edges Formed | Vector Spaces Targeted |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`.py`** | Python | Native `ast` module | `file`, `class`, `function`, `method`, `variable`, `comment` | `CALLS`, `REFERENCES` (`[r]`/`[w]`), `IMPORTS`, `INHERITS`, `ANNOTATES` | `filename`, `function_name`, `function`, `comment` |
| **`.ts`** | TypeScript | Acorn / Babel AST bridge via Node.js | `file`, `class`, `function`, `method`, `variable`, `comment` | `CALLS`, `REFERENCES`, `IMPORTS`, `INHERITS` | `filename`, `function_name`, `function`, `comment` |
| **`.js`, `.mjs`, `.cjs`** | JavaScript | Acorn AST parser bridge | `file`, `class`, `function`, `method`, `variable`, `comment` | `CALLS`, `REFERENCES`, `IMPORTS`, `INHERITS` | `filename`, `function_name`, `function`, `comment` |
| **`.html`** | HTML | HTML tag parser + script extractor | `section`, `block`, plus JS AST nodes for inline scripts | Inline scripts connect to parent HTML and global JS call graph | `comment`, `function_name`, `function` |
| **`.md`** | Markdown | Structural markdown heading parser | `section` (one per heading with full prose), `block` (fenced code) | `LINKS_TO` (Obsidian `[[wikilinks]]`, section anchors, path links), `DEFINES` | `comment` (prose and meaning), `filename` |
| **`.txt`** | Text | Paragraph block chunker | `section` (short files: 1 node; long files: ~20 line paragraph blocks) | None | `comment` |
| **`.srt`, `.vtt`** | Subtitles | Subtitle block extractor | `section` with timestamps and dialogue lines | None | `comment` |
| **`.json`, `.jsonl`** | JSON / JSON Lines | Recursive schema & key-path parser | `section`, `record` (one per JSONL row or top-level key) | `CONNECTS_TO` (record-to-record references like `way_id`, `area_from`) | `function_name` (keys), `comment` (values) |
| **`.csv`** | Tabular Data | CSV row & column schema parser | `section` (schema summary), `record` (row entries) | Column header linkages | `comment` (row data and values) |
| **`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`** | Image | PIL dimension extractor + CUDA vision sidecar | `file` (with dimensions in docstring e.g. `1920x1080`) | None | Dual: `filename` (text space) + `image` (pixel vision space) |

---

## 4. Deep Dive: How Specific Formats Are Handled

### Python (`.py`)
- **AST Parsing:** Parses full classes, nested methods, standalone functions, global variables, and top-level docstrings.
- **Reference Tracking:** Distinguishes between reading a variable (`[r]`) and writing/assigning to it (`[w]`).
- **Call Graph:** Resolves method calls across imports, linking caller functions to their exact callee targets.

### TypeScript & JavaScript (`.ts`, `.js`, `.mjs`, `.cjs`)
- **Node.js Bridge:** Uses Acorn with strip-only TypeScript transforms so source line numbers are preserved with 100% precision.
- **Modern Syntax:** Handles ES modules (`import`/`export`), CommonJS (`require`), arrow functions, object methods, and IIFEs.

### Markdown (`.md`)
- **Hierarchical Sections:** Each `# Heading` creates a `section` node whose docstring contains all prose under that heading until the next heading of equal or higher rank.
- **Wikilinks & Anchors:** Automatically extracts `[[Note]]`, `[[Note#Section]]`, and `[[#Anchor]]`.
- **Cross-Document Graph:** Resolves wikilinks to target documents or specific `# Heading` nodes, emitting real `LINKS_TO` relational edges.

### JSON & JSON Lines (`.json`, `.jsonl`)
- **Row-Level Records:** Instead of turning a 50MB JSON file into one unsearchable text blob, `codesearch` indexes each JSONL row or object record as its own discrete node.
- **Game & Entity Relations:** Recognizes relational fields like `target`, `way_id`, or `area_from` and forms `CONNECTS_TO` edges to other records in the same workspace.

### Multimodal Images
- **Dual-Space Encoding:** Traditional search engines cannot search images by text unless someone manually writes an alt-tag. `codesearch` embeds the image file path into the text vector space (so `"goblins/archer.png"` matches queries for `"goblin ranger"`), while its pixel data is encoded by an image embedding model (such as `embeddinggemma-2`).
- **Pixel Similarity:** A visual query searches image feature vectors directly, allowing visual similarity discovery alongside lexical code search.

---

## 5. What `codesearch` Does That Most Tools Don't

1. **No Silent Truncation:** Prose, docstrings, and function bodies are indexed completely. Nothing is arbitrarily sliced mid-sentence.
2. **Unified Relational Graph:** Code classes, documentation pages, game data records, and narrative scenes all live in the **same SQLite graph**. A Markdown design document can point directly to the Python function that implements it.
3. **Keyword-Vector Separation with Fusion:** Many vector databases attempt "hybrid search" by throwing sparse BM25 tokens into a vector index. `codesearch` keeps native SQLite FTS5 BM25 and dense cosine matrices separate, combining them cleanly via Reciprocal Rank Fusion.
4. **Instant Incremental Hashing:** File mtimes and BLAKE2b content hashes are tracked in SQLite. When you re-index a 25,000-file repository, only the files you touched are re-parsed and re-embedded, finishing in seconds.
5. **Zero Cloud Lock-In:** 100% local, runs on an embedded SQLite database, and requires zero paid cloud infrastructure.
