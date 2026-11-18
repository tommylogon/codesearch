"""Hybrid retrieval: FTS5 keyword match fused with vector lists by RRF.

Lists fused (each ranked, then `weight / (K + rank)` summed per node):
    keyword   every query word must appear (AND), or any (OR with match_all=False)
    vectors   cosine against filename / function_name / function / comment parts,
              plus an `image` list from a separate multimodal model for image files

Keyword hits and semantic hits are a UNION: a node that has all the words OR matches the
meaning of the whole sentence is a result. `strict=True` narrows it to keyword matches and
lets the vectors only reorder them.

The `image` list is never compared to the text lists as vectors — the two live in different
spaces with different dimensions. RRF only needs ranks, which is exactly why it is used.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config, db, diag, graph
from .db import split_words
from .embedder import VectorIndex, embed_query, image_query, schedule_pack_warmup

_log = diag.get_logger("search")

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
PART_KINDS = {"filename": ("file",), "image": ("file",)}


_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_date_bound(value: str | int | float | None, *, end_of_day: bool = False) -> float | None:
    """Parse an ISO date (`YYYY-MM-DD`), ISO datetime, or Unix timestamp to epoch seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    if _DATE_ONLY_RE.match(text):
        parsed_date = dt.date.fromisoformat(text)
        t = dt.time(23, 59, 59, 999999) if end_of_day else dt.time.min
        return dt.datetime.combine(parsed_date, t).timestamp()
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    return dt.datetime.fromisoformat(normalized).timestamp()


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


def _quote_term_fts(term: str) -> str:
    """Split and quote a boolean term safely for FTS5."""
    words = split_words(term).split()
    if not words:
        cleaned = term.replace('"', "").strip()
        return f'"{cleaned}"' if cleaned else ""
    if len(words) == 1:
        return f'"{words[0]}"'
    return "(" + " ".join(f'"{w}"' for w in words) + ")"


def build_boolean_fts_expression(
    tokens: list[str],
    match_all: bool,
    columns: tuple[str, ...] | None,
    *,
    must_terms: list[str] | None = None,
    should_terms: list[str] | None = None,
    not_terms: list[str] | None = None,
) -> str:
    """Compile query tokens plus explicit AND/OR/NOT term lists into an FTS5 MATCH expression."""
    pos_groups: list[str] = []
    if tokens:
        q_body = (" " if match_all else " OR ").join(f'"{t}"' for t in tokens)
        pos_groups.append(f"({q_body})" if len(tokens) > 1 and not match_all else q_body)

    must_q = [q for q in (_quote_term_fts(t) for t in (must_terms or [])) if q]
    if must_q:
        pos_groups.append(" AND ".join(must_q))

    should_q = [q for q in (_quote_term_fts(t) for t in (should_terms or [])) if q]
    if should_q:
        pos_groups.append("(" + " OR ".join(should_q) + ")" if len(should_q) > 1 else should_q[0])

    if not pos_groups:
        return ""

    pos_expr = " AND ".join(pos_groups) if len(pos_groups) > 1 else pos_groups[0]

    not_q = [q for q in (_quote_term_fts(t) for t in (not_terms or [])) if q]
    if not_q:
        neg_expr = " OR ".join(not_q)
        body = f"({pos_expr}) NOT ({neg_expr})"
    else:
        body = pos_expr

    if columns:
        return "{" + " ".join(columns) + "} : (" + body + ")"
    return body


@dataclass
class Hit:
    node_id: int
    score: float = 0.0
    lists: dict[str, tuple[int, float]] = field(default_factory=dict)   # list -> (rank, raw)


class Searcher:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        repo_root: Path | str | None = None,
        vector_cache: dict | None = None,
        vector_lock: threading.RLock | None = None,
        background_pack_build: bool = False,
    ):
        self.conn = conn
        self._repo_root = Path(repo_root).resolve() if repo_root else None
        self._vectors: VectorIndex | None = None
        self._vec_count: int | tuple = -1
        self._part_vector_cache: dict[tuple[str, ...], tuple[tuple, VectorIndex]] = (
            vector_cache if vector_cache is not None else {}
        )
        self._vec_lock = vector_lock if vector_lock is not None else threading.RLock()
        self._background_pack_build = background_pack_build
        self.last_vector_error: str | None = None
        self.last_rerank_error: str | None = None
        self.last_timings: dict[str, object] = {}

    def _vec_key(self) -> tuple:
        gen = db.get_meta(self.conn, "vec_generation", "")
        row = self.conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM vec_cache").fetchone()
        return (gen, int(row[0] or 0), config.EMBED_MODEL, config.IMAGE_EMBED_MODEL)

    @property
    def vectors(self) -> VectorIndex:
        # Count rows across every active model (text + image) so a vector rebuild is
        # triggered when vectors are added/updated for either active model underneath us.
        key = self._vec_key()
        with self._vec_lock:
            if self._vectors is None or key != self._vec_count:
                self._vectors = VectorIndex(self.conn)
                self._vec_count = key
            return self._vectors

    def reload_vectors(self) -> None:
        with self._vec_lock:
            self._vectors = None
            self._vec_count = -1
            self._part_vector_cache.clear()

    # ------------------------------------------------------------ keyword
    def keyword(self, tokens: list[str], match_all: bool, kinds, languages, path_prefix,
                part: str | None, limit: int,
                after_ts: float | None = None, before_ts: float | None = None,
                must_terms: list[str] | None = None,
                should_terms: list[str] | None = None,
                not_terms: list[str] | None = None,
                not_paths: list[str] | None = None) -> list[tuple[int, float]]:
        if not tokens and not must_terms and not should_terms:
            return []
        cols = PART_FTS_COLUMNS.get(part) if part else None
        if must_terms or should_terms or not_terms:
            expr = build_boolean_fts_expression(
                tokens, match_all, cols,
                must_terms=must_terms, should_terms=should_terms, not_terms=not_terms,
            )
        else:
            expr = fts_expression(tokens, match_all, cols)
        if not expr:
            return []
        bm25_weights = ", ".join(str(w) for w in config.BM25_WEIGHTS)
        has_filters = bool(
            kinds or languages or path_prefix or not_paths or (part in PART_KINDS)
            or after_ts is not None or before_ts is not None
        )
        try:
            if not has_filters:
                sql = f"SELECT rowid AS id, bm25(nodes_fts, {bm25_weights}) AS rank FROM nodes_fts WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?"
                rows = self.conn.execute(sql, [expr, limit]).fetchall()
                return [(r["id"], -float(r["rank"])) for r in rows]

            inner_limit = max(limit * 20, 2000)
            inner_sql = (
                f"SELECT fts.id AS id, fts.rank AS rank FROM ("
                f"SELECT rowid AS id, bm25(nodes_fts, {bm25_weights}) AS rank "
                f"FROM nodes_fts WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?"
                f") fts JOIN nodes n ON n.id = fts.id JOIN files f ON f.id = n.file_id WHERE 1=1"
            )
            params: list = [expr, inner_limit]
            inner_sql, params = self._filters(inner_sql, params, kinds, languages, path_prefix, part, after_ts, before_ts, not_paths=not_paths)
            inner_sql += " ORDER BY fts.rank LIMIT ?"
            params.append(limit)
            rows = self.conn.execute(inner_sql, params).fetchall()
            if len(rows) < limit:
                # If inner FTS window was saturated and selective filters pruned most rows, run full joined query.
                cnt_row = self.conn.execute(
                    "SELECT COUNT(*) FROM (SELECT 1 FROM nodes_fts WHERE nodes_fts MATCH ? LIMIT ?)",
                    [expr, inner_limit],
                ).fetchone()
                if cnt_row and int(cnt_row[0]) >= inner_limit:
                    full_sql = (
                        f"SELECT nodes_fts.rowid AS id, bm25(nodes_fts, {bm25_weights}) AS rank FROM nodes_fts "
                        "JOIN nodes n ON n.id = nodes_fts.rowid JOIN files f ON f.id = n.file_id "
                        "WHERE nodes_fts MATCH ?"
                    )
                    full_params: list = [expr]
                    full_sql, full_params = self._filters(full_sql, full_params, kinds, languages, path_prefix, part, after_ts, before_ts, not_paths=not_paths)
                    full_sql += " ORDER BY rank LIMIT ?"
                    full_params.append(limit)
                    rows = self.conn.execute(full_sql, full_params).fetchall()
        except sqlite3.OperationalError as exc:
            _log.debug("FTS_QUERY_ERR expr=%r err=%s", expr, exc)
            return []
        return [(r["id"], -float(r["rank"])) for r in rows]

    @staticmethod
    def _filters(sql, params, kinds, languages, path_prefix, part,
                 after_ts: float | None = None, before_ts: float | None = None,
                 not_paths: list[str] | None = None):
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
        for np_item in not_paths or []:
            np_clean = np_item.replace("\\", "/").strip().strip("%")
            if np_clean:
                sql += " AND f.rel_path NOT LIKE ? AND f.rel_path NOT LIKE ?"
                params.extend([f"{np_clean}%", f"%/{np_clean}%"])
        if after_ts is not None:
            sql += " AND f.mtime >= ?"
            params.append(after_ts)
        if before_ts is not None:
            sql += " AND f.mtime <= ?"
            params.append(before_ts)
        return sql, params

    # ------------------------------------------------------------ semantic
    def _available_parts(self, parts: tuple[str, ...]) -> tuple[str, ...]:
        avail: list[str] = []
        for part in parts:
            model = config.model_for_part(part)
            row = self.conn.execute(
                "SELECT 1 FROM node_parts p "
                "JOIN vec_cache c ON c.part = p.part AND c.hash = p.hash AND c.model = ? "
                "WHERE p.part = ? LIMIT 1",
                (model, part),
            ).fetchone()
            if row is not None:
                avail.append(part)
        return tuple(avail)

    def _part_vectors(self, parts: tuple[str, ...], stats_out: dict | None = None) -> VectorIndex:
        key_t0 = time.perf_counter()
        key = self._vec_key()
        key_ms = (time.perf_counter() - key_t0) * 1000.0
        lock_t0 = time.perf_counter()
        with self._vec_lock:
            lock_wait_ms = (time.perf_counter() - lock_t0) * 1000.0
            cached = self._part_vector_cache.get(parts)
            if cached is not None and cached[0] == key:
                if stats_out is not None:
                    stats_out["vec_key_ms"] = round(key_ms, 2)
                    stats_out["vec_lock_wait_ms"] = round(lock_wait_ms, 2)
                    stats_out["vec_cache_hit"] = True
                    stats_out["vec_load_ms"] = 0.0
                return cached[1]
            load_t0 = time.perf_counter()
            vi = VectorIndex(
                self.conn,
                parts=parts,
                allow_sql_build=not self._background_pack_build,
            )
            load_ms = (time.perf_counter() - load_t0) * 1000.0
            missing = getattr(vi, "missing_packs", [])
            if missing:
                db_row = self.conn.execute("PRAGMA database_list").fetchone()
                db_file = Path(db_row[2]) if db_row and db_row[2] else None
                if db_file and db_file.is_file():
                    schedule_pack_warmup(db_file, tuple(missing), on_done=self.reload_vectors)
                if stats_out is not None:
                    stats_out["vec_deferred"] = list(missing)
            else:
                self._part_vector_cache[parts] = (key, vi)
            if stats_out is not None:
                stats_out["vec_key_ms"] = round(key_ms, 2)
                stats_out["vec_lock_wait_ms"] = round(lock_wait_ms, 2)
                stats_out["vec_cache_hit"] = False
                stats_out["vec_load_ms"] = round(load_ms, 2)
                load_stats = getattr(vi, "load_stats", {})
                if load_stats:
                    stats_out["vec_parts"] = {
                        p: f"{st.get('source')}:{st.get('rows', 0)}({st.get('ms', 0)}ms)"
                        for p, st in load_stats.items()
                        if st.get("rows", 0) or st.get("source") != "empty"
                    }
            return vi

    def semantic(
        self,
        query: str,
        parts: tuple[str, ...],
        limit: int,
        stats_out: dict | None = None,
    ) -> dict[str, list[tuple[int, float]]]:
        self.last_vector_error = None
        t_avail = time.perf_counter()
        avail = self._available_parts(parts)
        if stats_out is not None:
            stats_out["sem_avail_ms"] = round((time.perf_counter() - t_avail) * 1000.0, 2)
            stats_out["avail_parts"] = avail
        if not avail:
            self.last_vector_error = "no vectors built (run: codesearch embed)"
            return {}
        result: dict[str, list[tuple[int, float]]] = {}
        try:
            # Query the embedder(s) BEFORE materializing numpy vector matrices so an
            # unreachable server falls back to keyword search in ~1-2s without loading RAM.
            img_qvec = None
            if config.IMAGE_PART in avail:
                t_img = time.perf_counter()
                img_qvec = image_query(query)
                if stats_out is not None:
                    stats_out["image_query_ms"] = round((time.perf_counter() - t_img) * 1000.0, 2)
            text_parts = [p for p in avail if p != config.IMAGE_PART]
            txt_qvec = None
            if text_parts:
                t_eq = time.perf_counter()
                txt_qvec = embed_query(query)
                if stats_out is not None:
                    stats_out["embed_query_ms"] = round((time.perf_counter() - t_eq) * 1000.0, 2)
            vi = self._part_vectors(avail, stats_out=stats_out)
            if not vi.available():
                if getattr(vi, "missing_packs", None):
                    self.last_vector_error = (
                        f"warming vector packs in background ({', '.join(vi.missing_packs)}); returning keyword results"
                    )
                else:
                    self.last_vector_error = "no vectors built (run: codesearch embed)"
                return {}
            # If the image part is requested, query the multimodal sidecar so vectors live in the
            # correct (image) space. The floor is stricter because text-vs-image cosines are
            # narrower (a car query scores ~0.51 against a goblin portrait; 0.45 floor would admit
            # noise). Images get their own floor of 0.58.
            t_dot = time.perf_counter()
            if img_qvec is not None:
                result[config.IMAGE_PART] = [
                    (i, s) for i, s in vi.top(config.IMAGE_PART, img_qvec, limit)
                    if s >= config.IMAGE_MIN_SIMILARITY
                ]
            # Text parts stay with the text embedder; if image was handled already, skip it.
            if txt_qvec is not None:
                for p in text_parts:
                    result[p] = [(i, s) for i, s in vi.top(p, txt_qvec, limit)
                                 if s >= config.MIN_SIMILARITY]
            if stats_out is not None:
                stats_out["vec_dot_ms"] = round((time.perf_counter() - t_dot) * 1000.0, 2)
        except Exception as exc:
            self.last_vector_error = f"embedding endpoint unavailable: {exc}"
            _log.warning("SEMANTIC_FALLBACK q=%r err=%s", query[:80], exc)
            return {}
        return result

    # ------------------------------------------------------------ fusion
    def _fts_matching_ids(self, ids: set[int], expr: str) -> set[int]:
        if not ids or not expr:
            return set()
        matched: set[int] = set()
        id_list = list(ids)
        for i in range(0, len(id_list), 500):
            chunk = id_list[i:i + 500]
            sql = (
                "SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH ? AND rowid IN (%s)"
                % ",".join("?" * len(chunk))
            )
            try:
                rows = self.conn.execute(sql, [expr, *chunk]).fetchall()
                matched.update(int(r[0]) for r in rows)
            except sqlite3.OperationalError:
                pass
        return matched

    def _excluded_by_not_terms(self, ids: set[int], not_terms: list[str] | None, part: str | None) -> set[int]:
        if not ids or not not_terms:
            return set()
        not_q = [q for q in (_quote_term_fts(t) for t in not_terms) if q]
        if not not_q:
            return set()
        neg_expr = " OR ".join(not_q)
        cols = PART_FTS_COLUMNS.get(part) if part else None
        if cols:
            neg_expr = "{" + " ".join(cols) + "} : (" + neg_expr + ")"
        return self._fts_matching_ids(ids, neg_expr)

    def search(self, query: str, limit: int = 10, offset: int = 0, kinds=None, languages=None, path_prefix=None,
               not_path: str | list[str] | None = None,
               part: str | None = None, match_all: bool = True, strict: bool = False,
               semantic: bool = True, keyword: bool = True, rerank: bool | None = None,
               must_terms: list[str] | None = None, should_terms: list[str] | None = None,
               not_terms: list[str] | None = None,
               after: str | int | float | None = None, before: str | int | float | None = None) -> list[dict]:
        search_t0 = time.perf_counter()
        self.last_rerank_error = None
        query = (query or "").strip()
        must_clean = [t.strip() for t in (must_terms or []) if str(t).strip()]
        should_clean = [t.strip() for t in (should_terms or []) if str(t).strip()]
        not_clean = [t.strip() for t in (not_terms or []) if str(t).strip()]
        if isinstance(not_path, str):
            not_paths = [not_path.strip()] if not_path.strip() else []
        else:
            not_paths = [str(p).strip() for p in (not_path or []) if str(p).strip()]
        if not query and not must_clean and not should_clean:
            self.last_total_matched = 0
            self.last_timings = {"total_ms": 0.0, "results": 0}
            return []
        after_ts = parse_date_bound(after, end_of_day=False)
        before_ts = parse_date_bound(before, end_of_day=True)
        tokens = query_tokens(query) if query else []
        sem_query = query or " ".join(must_clean + should_clean)
        hits: dict[int, Hit] = {}
        timings: dict[str, object] = {
            "q": (query or sem_query)[:80],
            "path": path_prefix,
            "part": part,
            "limit": limit,
        }

        vec_contribs: dict[int, list[float]] = {}

        def add_kw(ranked: list[tuple[int, float]], weight: float):
            for rank, (nid, raw) in enumerate(ranked):
                h = hits.setdefault(nid, Hit(nid))
                h.score += weight / (config.RRF_K + rank + 1)
                h.lists["keyword"] = (rank + 1, raw)

        def add_vec(name: str, ranked: list[tuple[int, float]], weight: float):
            floor = config.IMAGE_MIN_SIMILARITY if name == f"vec:{config.IMAGE_PART}" else config.MIN_SIMILARITY
            denom = max(1e-6, 1.0 - floor)
            for rank, (nid, raw) in enumerate(ranked):
                h = hits.setdefault(nid, Hit(nid))
                margin = max(0.05, (raw - floor) / denom)
                vec_contribs.setdefault(nid, []).append((weight * margin) / (config.RRF_K + rank + 1))
                h.lists[name] = (rank + 1, raw)

        kw_ids: set[int] = set()
        if keyword:
            t_kw = time.perf_counter()
            ranked = self.keyword(
                tokens, match_all, kinds, languages, path_prefix, part,
                config.CANDIDATES, after_ts=after_ts, before_ts=before_ts,
                must_terms=must_clean or None, should_terms=should_clean or None,
                not_terms=not_clean or None, not_paths=not_paths or None,
            )
            kw_ids = {i for i, _ in ranked}
            eff_and = match_all or bool(must_clean)
            add_kw(ranked, config.FTS_WEIGHT if eff_and else 0.45)
            timings["kw_ms"] = round((time.perf_counter() - t_kw) * 1000.0, 2)
            timings["kw_hits"] = len(ranked)
        if semantic and not (strict and keyword and len(kw_ids) == 0):
            parts = (part,) if part else config.PARTS
            sem_timings: dict[str, object] = {}
            sem = self.semantic(sem_query, parts, config.CANDIDATES, stats_out=sem_timings)
            timings.update(sem_timings)
            t_allow = time.perf_counter()
            raw_ids = set().union(*[{i for i, _ in v} for v in sem.values()]) if sem else set()
            allowed = self._allowed(raw_ids, kinds, languages, path_prefix, part,
                                    after_ts=after_ts, before_ts=before_ts, not_paths=not_paths or None)
            if not_clean and allowed:
                allowed -= self._excluded_by_not_terms(allowed, not_clean, part)
            timings["allowed_ms"] = round((time.perf_counter() - t_allow) * 1000.0, 2)
            timings["sem_raw"] = len(raw_ids)
            timings["sem_kept"] = len(allowed)
            t_rrf = time.perf_counter()
            vec_only_floor = config.VEC_ONLY_MIN_SIMILARITY if kw_ids else config.MIN_SIMILARITY
            for p, ranked in sem.items():
                floor = config.IMAGE_MIN_SIMILARITY if p == config.IMAGE_PART else vec_only_floor
                filtered = [
                    (i, s) for i, s in ranked
                    if i in allowed and (i in kw_ids or s >= floor)
                ]
                add_vec(f"vec:{p}", filtered, config.PART_WEIGHTS[p])
            for nid, scores in vec_contribs.items():
                scores.sort(reverse=True)
                hits[nid].score += scores[0] + config.MULTI_PART_DECAY * sum(scores[1:])
            timings["rrf_ms"] = round((time.perf_counter() - t_rrf) * 1000.0, 2)
        if strict and keyword:
            hits = {i: h for i, h in hits.items() if i in kw_ids}
        if not_clean and hits:
            excluded_final = self._excluded_by_not_terms(set(hits.keys()), not_clean, part)
            if excluded_final:
                hits = {i: h for i, h in hits.items() if i not in excluded_final}
        if hits and not (kinds and "block" in kinds) and config.BLOCK_UNREQUESTED_WEIGHT != 1.0:
            block_ids = self._block_node_ids(set(hits.keys()))
            for bid in block_ids:
                hits[bid].score *= config.BLOCK_UNREQUESTED_WEIGHT
        ordered = sorted(hits.values(), key=lambda h: -h.score)
        self.last_total_matched = len(ordered)

        use_rerank = config.RERANK_ENABLED if rerank is None else bool(rerank)
        if use_rerank and len(ordered) > 1:
            from .reranker import candidate_document_text, rerank_candidates

            pool_size = max(config.RERANK_TOP_K, offset + limit)
            pool_hits = ordered[:pool_size]
            t_mat = time.perf_counter()
            pool_nodes: list[dict] = []
            mat_max_ms = 0.0
            mat_slowest = ""
            for h in pool_hits:
                h_t0 = time.perf_counter()
                node = self._materialize(h)
                h_ms = (time.perf_counter() - h_t0) * 1000.0
                if h_ms > mat_max_ms:
                    mat_max_ms = h_ms
                    mat_slowest = f"{node.get('file', '')}#{h.node_id}"
                pool_nodes.append(node)
            docs = [candidate_document_text(n) for n in pool_nodes]
            ranked_pairs, rr_info = rerank_candidates(query, docs, top_n=len(pool_nodes))
            timings["rerank_ms"] = rr_info.get("ms", 0.0)
            timings["rerank_used"] = bool(rr_info.get("used"))
            if ranked_pairs:
                reordered: list[dict] = []
                seen_idx: set[int] = set()
                for idx, rr_score in ranked_pairs:
                    node = pool_nodes[idx]
                    node["rrf_score"] = node["score"]
                    node["rerank_score"] = round(rr_score, 5)
                    node["score"] = round(rr_score, 5)
                    if "rerank" not in node["matched"]:
                        node["matched"].append("rerank")
                    reordered.append(node)
                    seen_idx.add(idx)
                for idx, node in enumerate(pool_nodes):
                    if idx not in seen_idx:
                        reordered.append(node)
                out_hits = reordered[offset:offset + limit] if offset else reordered[:limit]
            else:
                if rr_info.get("error"):
                    self.last_rerank_error = str(rr_info["error"])
                    timings["rerank_err"] = self.last_rerank_error
                out_hits = pool_nodes[offset:offset + limit] if offset else pool_nodes[:limit]
        else:
            paged = ordered[offset:offset + limit] if offset else ordered[:limit]
            t_mat = time.perf_counter()
            out_hits = []
            mat_max_ms = 0.0
            mat_slowest = ""
            for h in paged:
                h_t0 = time.perf_counter()
                node = self._materialize(h)
                h_ms = (time.perf_counter() - h_t0) * 1000.0
                if h_ms > mat_max_ms:
                    mat_max_ms = h_ms
                    mat_slowest = f"{node.get('file', '')}#{h.node_id}"
                out_hits.append(node)
        timings["mat_ms"] = round((time.perf_counter() - t_mat) * 1000.0, 2)
        timings["mat_hits"] = len(out_hits)
        if out_hits:
            timings["mat_max_ms"] = round(mat_max_ms, 2)
            if mat_max_ms >= 100.0:
                timings["mat_slowest"] = mat_slowest
        if self.last_vector_error:
            timings["vec_err"] = self.last_vector_error
        total_ms = (time.perf_counter() - search_t0) * 1000.0
        timings["total_ms"] = round(total_ms, 2)
        self.last_timings = timings
        level = logging.WARNING if total_ms >= config.SLOW_QUERY_MS else logging.DEBUG
        _log.log(level, "SEARCH %s", diag.format_timings(timings))
        return out_hits

    def _block_node_ids(self, ids: set[int]) -> set[int]:
        if not ids:
            return set()
        out: set[int] = set()
        id_list = list(ids)
        for i in range(0, len(id_list), 500):
            chunk = id_list[i:i + 500]
            rows = self.conn.execute(
                "SELECT id FROM nodes WHERE kind = 'block' AND id IN (%s)" % ",".join("?" * len(chunk)),
                chunk,
            ).fetchall()
            out.update(int(r[0]) for r in rows)
        return out

    def _allowed(self, ids: set[int], kinds, languages, path_prefix, part,
                 after_ts: float | None = None, before_ts: float | None = None,
                 not_paths: list[str] | None = None) -> set[int]:
        if not ids:
            return set()
        if not (kinds or languages or path_prefix or not_paths or part in PART_KINDS
                or after_ts is not None or before_ts is not None):
            return ids
        out: set[int] = set()
        id_list = list(ids)
        for i in range(0, len(id_list), 500):
            chunk = id_list[i:i + 500]
            sql = ("SELECT n.id FROM nodes n JOIN files f ON f.id = n.file_id WHERE n.id IN (%s)"
                   % ",".join("?" * len(chunk)))
            params: list = list(chunk)
            sql, params = self._filters(sql, params, kinds, languages, path_prefix, part, after_ts, before_ts, not_paths=not_paths)
            out.update(r[0] for r in self.conn.execute(sql, params))
        return out

    def _materialize(self, h: Hit) -> dict:
        node = graph.get_node(self.conn, h.node_id)
        node["score"] = round(h.score, 5)
        vec_sims = [raw for k, (_r, raw) in h.lists.items() if k.startswith("vec:")]
        node["sim"] = round(max(vec_sims), 4) if vec_sims else None
        node["matched"] = sorted(h.lists)
        node["ranks"] = {k: {"rank": r, "raw": round(raw, 4)} for k, (r, raw) in h.lists.items()}
        node["relations"] = graph.summary(self.conn, h.node_id)
        # Inline head snippet for file-kind hits so callers don't need a separate /api/code hop
        if node["kind"] == "file" and not node.get("source") and node.get("language") not in ("image", "pdf"):
            try:
                root = self._repo_root or config.REPO_ROOT
                fp = (root / node["file"]).resolve()
                if fp.is_file():
                    head_lines = []
                    with open(fp, "r", encoding="utf-8", errors="replace") as f:
                        for _ in range(6):
                            line = f.readline()
                            if not line:
                                break
                            head_lines.append(line.rstrip("\r\n"))
                    node["source"] = "\n".join(head_lines)
            except Exception:
                pass
        return node
