# Task: Embedding Model Benchmarking & Comparison Harness

**Status:** Planned
**Area:** Embeddings & Evaluation (`src/codesearch/embedder.py`, `src/codesearch/config.py`, `tools/`)
**Target:** Systematically measure retrieval quality, score separation, latency, and storage tradeoffs across local and hosted embedding models.

---

## 1. Why Existing Architecture Already Supports This

`src/codesearch/db.py` keys `vec_cache` by `(part, hash, model)`:

```sql
CREATE TABLE IF NOT EXISTS vec_cache (
    part  TEXT NOT NULL,
    hash  INTEGER NOT NULL,
    model TEXT NOT NULL,
    dim   INTEGER NOT NULL,
    vec   BLOB NOT NULL,
    PRIMARY KEY (part, hash, model)
);
```

Because `model` is already part of the primary key, vectors from multiple candidate embedders can coexist cleanly in the same SQLite database without invalidating or overwriting baseline `text-embedding-qwen3-embedding-0.6b` vectors.

### Implementation Gap Found During Inspection

`src/codesearch/config.py` defines `EMBED_QUERY_INSTRUCTION = ""`, and `src/codesearch/embedder.py::embed_query()` currently sends bare `f"{query}"` without prepending `config.EMBED_QUERY_INSTRUCTION`. Models such as Qwen3-Embedding and BGE often show significantly different cosine distributions depending on whether an instruction prefix (`Instruct: ...\nQuery: ...`) is used on queries while omitting it on documents. The benchmark must test both symmetric (raw query) and asymmetric (instruct-prefixed query) modes and wire `EMBED_QUERY_INSTRUCTION` inside `embed_query()`.

---

## 2. Candidate Models to Evaluate

1. **`text-embedding-qwen3-embedding-0.6b`** (current default baseline).
2. **Larger Qwen3 tiers (`qwen3-embedding-4b` / `8b` quantized)** for dense narrative / lore vs. precision code retrieval.
3. **`nomic-embed-text-v1.5` / `nomic-embed-code`** (code-focused and Matryoshka dimension reduction tests).
4. **`bge-m3`** (multilingual Norwegian/English notes, longer context windows).
5. **Multimodal image space models** (evaluating `embeddinggemma-2` vs. SigLIP/CLIP variants against `IMAGE_MIN_SIMILARITY = 0.58`).

---

## 3. Golden Evaluation Suites

Mixtures of code and narrative behave differently, so benchmark suites must split results by corpus type:
1. **Codebase Suite (VirtualWorld Python + TypeScript)**:
   - Exact symbol / identifier paraphrase queries ("where are memory decay constants applied").
   - Cross-language front/back pairs ("TS call that posts player memory reflect").
2. **Game Data & Docs Suite (Markdown + JSON scenarios)**:
   - Finding room descriptions, trigger effects, and item properties without matching noisy record keys.
3. **Creative Writing & RP Transcripts Suite (Obsidian vault + AI Studio JSON)**:
   - Scene retrieval by mood, character subtext, or dialogue beat rather than literal keywords.

---

## 4. Metrics & Output Format

- **Ranking Quality**: Recall@1, Recall@5, MRR@10 (comparing semantic-only vs. hybrid RRF fusion).
- **Score Distribution & Threshold Calibration**: Mean true-positive cosine vs. 95th-percentile false-positive cosine (determines whether `config.MIN_SIMILARITY = 0.45` stays optimal per model, since different models have wildly different cosine baselines).
- **Speed & Footprint**: Texts/sec indexing throughput, single query latency (ms), DB bloat per 10k nodes (`dim * 4` bytes).
- **Deliverable**: `codesearch bench --models modelA,modelB --queries eval.json` writing a reproducible JSON + Markdown comparison report.
