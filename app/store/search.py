"""BM25 keyword scoring and cosine-similarity fallback (stdlib only, no DB)."""

from __future__ import annotations

import math
import re
from typing import Any


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokeniser — splits on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_score(
    items: list[dict[str, Any]],
    query: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """Compute BM25 scores for *items* against *query*.

    Adds ``keyword_score`` and sets ``score = keyword_score`` on each item.
    Items with zero score are retained (caller filters if needed).

    BM25 parameters: k1=1.5 (term-frequency saturation), b=0.75 (length norm).
    """
    if not items:
        return items

    query_terms = set(tokenize(query))
    if not query_terms:
        for it in items:
            it["keyword_score"] = 0.0
            it["score"] = 0.0
        return items

    doc_tokens: list[list[str]] = [tokenize(it.get("raw_text", "")) for it in items]
    N = len(items)
    avg_dl = sum(len(t) for t in doc_tokens) / N if N else 1.0

    df: dict[str, int] = {}
    for tokens in doc_tokens:
        for term in query_terms:
            if term in tokens:
                df[term] = df.get(term, 0) + 1

    idf: dict[str, float] = {}
    for term in query_terms:
        n_t = df.get(term, 0)
        idf[term] = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1.0)

    for item, tokens in zip(items, doc_tokens):
        dl = len(tokens)
        tf_counts: dict[str, int] = {}
        for t in tokens:
            if t in query_terms:
                tf_counts[t] = tf_counts.get(t, 0) + 1

        bm25 = 0.0
        for term in query_terms:
            tf = tf_counts.get(term, 0)
            if tf == 0:
                continue
            norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            bm25 += idf[term] * norm_tf

        # Normalise to [0, 1]: max possible BM25 score per term ≈ idf * (k1+1)
        max_possible = sum(idf[t] * (k1 + 1) for t in query_terms) or 1.0
        norm_score = round(min(bm25 / max_possible, 1.0), 6)
        item["keyword_score"] = norm_score
        item["score"] = norm_score

    return items
