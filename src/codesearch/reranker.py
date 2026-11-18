"""Optional stage-2 cross-encoder reranker client (llama.cpp /v1/rerank)."""
from __future__ import annotations

import time
from typing import Any

import requests

from . import config, diag

_log = diag.get_logger("reranker")


def format_qwen3_rerank_query(query: str, instruction: str = "") -> str:
    """Format query with optional Qwen3-Reranker instruction prefix."""
    instr = (instruction or "").strip()
    if not instr:
        return query
    return f"Instruct: {instr}\nQuery: {query}"


def candidate_document_text(node: dict[str, Any], max_chars: int | None = None) -> str:
    """Build a compact, high-signal text representation of a candidate node for reranking."""
    cap = config.RERANK_MAX_CHARS if max_chars is None else max_chars
    header_parts = [
        str(node.get("file") or ""),
        str(node.get("kind") or ""),
        str(node.get("qualified_name") or node.get("name") or ""),
    ]
    header = " | ".join(p for p in header_parts if p)
    body_parts = [
        str(node.get("signature") or "").strip(),
        str(node.get("doc") or "").strip(),
        str(node.get("source") or "").strip(),
    ]
    body = "\n".join(p for p in body_parts if p)
    text = f"{header}\n{body}".strip() if header else body
    if cap > 0 and len(text) > cap:
        return text[:cap]
    return text


def _post_rerank(
    endpoint: str,
    payload: dict[str, Any],
    *,
    connect_timeout: float,
    read_timeout: float,
) -> dict[str, Any]:
    base = endpoint.strip().rstrip("/")
    urls = [f"{base}/rerank"]
    if base.endswith("/v1"):
        urls.append(f"{base[:-3]}/rerank")
    last_exc: Exception | None = None
    for idx, url in enumerate(urls):
        try:
            resp = requests.post(url, json=payload, timeout=(connect_timeout, read_timeout))
            if resp.status_code == 404 and idx + 1 < len(urls):
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if idx + 1 < len(urls) and isinstance(exc, requests.HTTPError):
                continue
            break
    assert last_exc is not None
    raise last_exc


def rerank_candidates(
    query: str,
    documents: list[str],
    *,
    endpoint: str | None = None,
    model: str | None = None,
    instruction: str | None = None,
    top_n: int | None = None,
    timeout: float | None = None,
    connect_timeout: float | None = None,
) -> tuple[list[tuple[int, float]], dict[str, Any]]:
    """Rerank candidate documents via llama.cpp /v1/rerank (or /rerank).

    Returns ([(doc_index, relevance_score), ...], info_dict).
    Never raises on network/server failure; returns ([], {"used": False, "error": ...}).
    """
    t0 = time.perf_counter()
    if not documents:
        return [], {"used": False, "reason": "no documents", "ms": 0.0}

    ep = (endpoint if endpoint is not None else config.RERANK_ENDPOINT).strip().rstrip("/")
    if not ep:
        return [], {"used": False, "error": "no rerank endpoint configured", "ms": 0.0}

    mdl = model if model is not None else config.RERANK_MODEL
    instr = instruction if instruction is not None else config.RERANK_INSTRUCTION
    formatted_query = format_qwen3_rerank_query(query, instr)

    payload: dict[str, Any] = {
        "model": mdl,
        "query": formatted_query,
        "documents": documents,
    }
    if top_n is not None and top_n > 0:
        payload["top_n"] = int(top_n)

    c_to = config.RERANK_CONNECT_TIMEOUT if connect_timeout is None else float(connect_timeout)
    r_to = config.RERANK_TIMEOUT if timeout is None else float(timeout)

    try:
        data = _post_rerank(ep, payload, connect_timeout=max(0.1, c_to), read_timeout=max(0.2, r_to))
        raw_results = data.get("results") or []
        ranked: list[tuple[int, float]] = []
        seen: set[int] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if isinstance(idx, int) and 0 <= idx < len(documents) and idx not in seen and score is not None:
                seen.add(idx)
                ranked.append((idx, float(score)))
        ranked.sort(key=lambda pair: -pair[1])
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return ranked, {"used": True, "model": mdl, "ms": elapsed_ms, "returned": len(ranked)}
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        _log.warning("RERANK_FALLBACK q=%r err=%s ms=%.1f", query[:80], exc, elapsed_ms)
        return [], {"used": False, "error": str(exc), "ms": elapsed_ms}
