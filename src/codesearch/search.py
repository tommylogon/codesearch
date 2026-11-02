"""Hybrid retrieval: FTS5 keyword match fused with four vector indexes by RRF.

Lists fused (each ranked, then `weight / (K + rank)` summed per node):
    keyword   every query word must appear (AND), or any (OR with match_all=False)
    vectors   cosine against filename / function_name / function / comment parts

Keyword hits and semantic hits are a UNION: a node that has all the words OR matches the
meaning of the whole sentence is a result. `strict=True` narrows it to keyword matches and
lets the vectors only reorder them.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from . import config, graph
from .db import split_words
from .embedder import VectorIndex, embed_query

STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were",
    "be", "how", "what", "where", "when", "which", "who", "why", "do", "does", "did", "it",
    "this", "that", "with", "by", "from", "as", "at", "i", "me", "my", "we", "can", "there",
}

# --part narrows which FTS columns and which nodes a part speaks for.
PART_FTS_COLUMNS = {
    "filename": ("rel_path", "words"),
    "function_name": ("name", "words"),
    "function": ("signature",),
    "comment": ("docstring",),
}
PART_KINDS = {"filename": ("file",)}


def query_tokens(query: str) -> list[str]:
    """Lowercased word tokens. camelCase and snake_case identifiers are split, so
    `syncGraphData` and `sync graph data` are the same query."""
    words = split_words(query).split()
    kept = [w for w in words if w not in STOPWORDS]
    return kept or words


def fts_expression(tokens: list[str], match_all: bool, columns: tuple[str, ...] | None) -> str:
    quoted = [f'"{t}"' for t in tokens]
    body = (" " if match_all else " OR ").join(quoted)
    if columns:
        return "{" + " ".join(columns) + "} : (" + body + ")"
    return body


@dataclass
class Hit:
    node_id: int
    score: float = 0.0
    lists: dict[str, tuple[int, float]] = field(default_factory=dict)   # list -> (rank, raw)


class Searcher:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._vectors: VectorIndex | None = None
        self._vec_count: int = -1
        self.last_vector_error: str | None = None

    @property
    def vectors(self) -> VectorIndex:
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM vec_cache WHERE model = ?", (config.EMBED_MODEL,)
        ).fetchone()[0]
        if self._vectors is None or cur != self._vec_count:
            self._vectors = VectorIndex(self.conn)
            self._vec_count = cur
        return self._vectors

    def reload_vectors(self) -> None:
        self._vectors = None
        self._vec_count = -1

    # ------------------------------------------------------------ keyword
    def keyword(self, tokens: list[str], match_all: bool, kinds, languages, path_prefix,
                part: str | None, limit: int) -> list[tuple[int, float]]:
        if not tokens:
            return []
        expr = fts_expression(tokens, match_all, PART_FTS_COLUMNS.get(part) if part else None)
        sql = ("SELECT nodes_fts.rowid AS id, bm25(nodes_fts, %s) AS rank FROM nodes_fts "
               "JOIN nodes n ON n.id = nodes_fts.rowid JOIN files f ON f.id = n.file_id "
               "WHERE nodes_fts MATCH ?" % ", ".join(str(w) for w in config.BM25_WEIGHTS))
        params: list = [expr]
        sql, params = self._filters(sql, params, kinds, languages, path_prefix, part)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r["id"], -float(r["rank"])) for r in rows]

    @staticmethod
    def _filters(sql, params, kinds, languages, path_prefix, part):
        eff_kinds = tuple(kinds) if kinds else None
        if part in PART_KINDS:
            allowed = PART_KINDS[part]
            eff_kinds = tuple(k for k in eff_kinds if k in allowed) if eff_kinds else allowed
        if eff_kinds:
            sql += " AND n.kind IN (%s)" % ",".join("?" * len(eff_kinds))
            params += list(eff_kinds)
        if languages:
            sql += " AND f.language IN (%s)" % ",".join("?" * len(languages))
            params += list(languages)
        if path_prefix:
            sql += " AND f.rel_path LIKE ?"
            params.append(path_prefix.replace("\\", "/").rstrip("%") + "%")
        return sql, params

    # ------------------------------------------------------------ semantic
    def semantic(self, query: str, parts: tuple[str, ...], limit: int) -> dict[str, list[tuple[int, float]]]:
        self.last_vector_error = None
        vi = self.vectors
        if not vi.available():
            self.last_vector_error = "no vectors built (run: codesearch embed)"
            return {}
        try:
            qvec = embed_query(query)
        except Exception as exc:        # endpoint down: degrade to keyword only, and say so
            self.last_vector_error = f"embedding endpoint unavailable: {exc}"
            return {}
        return {p: [(i, s) for i, s in vi.top(p, qvec, limit) if s >= config.MIN_SIMILARITY]
                for p in parts if p in vi.parts}

    # ------------------------------------------------------------ fusion
    def search(self, query: str, limit: int = 10, kinds=None, languages=None, path_prefix=None,
               part: str | None = None, match_all: bool = True, strict: bool = False,
               semantic: bool = True, keyword: bool = True) -> list[dict]:
        query = query.strip()
        if not query:
            return []
        tokens = query_tokens(query)
        hits: dict[int, Hit] = {}

        vec_contribs: dict[int, list[float]] = {}

        def add_kw(ranked: list[tuple[int, float]], weight: float):
            for rank, (nid, raw) in enumerate(ranked):
                h = hits.setdefault(nid, Hit(nid))
                h.score += weight / (config.RRF_K + rank + 1)
                h.lists["keyword"] = (rank + 1, raw)

        def add_vec(name: str, ranked: list[tuple[int, float]], weight: float):
            for rank, (nid, raw) in enumerate(ranked):
                h = hits.setdefault(nid, Hit(nid))
                vec_contribs.setdefault(nid, []).append(weight / (config.RRF_K + rank + 1))
                h.lists[name] = (rank + 1, raw)

        kw_ids: set[int] = set()
        if keyword:
            ranked = self.keyword(tokens, match_all, kinds, languages, path_prefix, part, config.CANDIDATES)
            kw_ids = {i for i, _ in ranked}
            add_kw(ranked, config.FTS_WEIGHT if match_all else 0.45)
        if semantic:
            parts = (part,) if part else config.PARTS
            sem = self.semantic(query, parts, config.CANDIDATES)
            allowed = self._allowed(set().union(*[{i for i, _ in v} for v in sem.values()]) if sem else set(),
                                    kinds, languages, path_prefix, part)
            for p, ranked in sem.items():
                add_vec(f"vec:{p}", [(i, s) for i, s in ranked if i in allowed], config.PART_WEIGHTS[p])
            for nid, scores in vec_contribs.items():
                scores.sort(reverse=True)
                hits[nid].score += scores[0] + config.MULTI_PART_DECAY * sum(scores[1:])
        if strict and keyword:
            hits = {i: h for i, h in hits.items() if i in kw_ids}
        ordered = sorted(hits.values(), key=lambda h: -h.score)[:limit]
        return [self._materialize(h) for h in ordered]

    def _allowed(self, ids: set[int], kinds, languages, path_prefix, part) -> set[int]:
        if not ids:
            return set()
        if not (kinds or languages or path_prefix or part in PART_KINDS):
            return ids
        out: set[int] = set()
        id_list = list(ids)
        for i in range(0, len(id_list), 500):
            chunk = id_list[i:i + 500]
            sql = ("SELECT n.id FROM nodes n JOIN files f ON f.id = n.file_id WHERE n.id IN (%s)"
                   % ",".join("?" * len(chunk)))
            params: list = list(chunk)
            sql, params = self._filters(sql, params, kinds, languages, path_prefix, part)
            out.update(r[0] for r in self.conn.execute(sql, params))
        return out

    def _materialize(self, h: Hit) -> dict:
        node = graph.get_node(self.conn, h.node_id)
        node["score"] = round(h.score, 5)
        node["matched"] = sorted(h.lists)
        node["ranks"] = {k: {"rank": r, "raw": round(raw, 4)} for k, (r, raw) in h.lists.items()}
        node["relations"] = graph.summary(self.conn, h.node_id)
        return node
