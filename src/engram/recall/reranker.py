"""Cross-encoder reranker for recall pipeline.

Lazy-loads BAAI/bge-reranker-v2-m3 on first call.
Reranks (query, candidate) pairs and filters by threshold.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from engram.config import RerankConfig
from engram.models import SearchResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger("engram")

_model_cache: CrossEncoder | None = None
_model_name_cache: str = ""
_model_lock = threading.Lock()


def _get_model(config: RerankConfig) -> CrossEncoder:
    """Lazy-load CrossEncoder model. Cached after first call. Thread-safe."""
    global _model_cache, _model_name_cache
    if _model_cache is not None and _model_name_cache == config.model_name:
        return _model_cache
    with _model_lock:
        # Double-checked locking
        if _model_cache is not None and _model_name_cache == config.model_name:
            return _model_cache
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker model: %s (device=%s)", config.model_name, config.device)
        _model_cache = CrossEncoder(config.model_name, max_length=config.max_length, device=config.device)
        _model_name_cache = config.model_name
        return _model_cache


def rerank(
    query: str,
    candidates: list[SearchResult],
    config: RerankConfig,
) -> list[SearchResult]:
    """Rerank candidates using cross-encoder.

    Returns filtered + sorted candidates. Falls back to original order
    if model loading fails or candidate count below minimum.
    """
    if not config.enabled or len(candidates) < config.min_candidates:
        return candidates

    try:
        model = _get_model(config)
    except Exception as e:
        logger.warning("Reranker model load failed, skipping: %s", e)
        return candidates

    # Build (query, doc) pairs
    pairs = [(query, c.content) for c in candidates]

    try:
        scores = model.predict(pairs, batch_size=config.batch_size)
    except Exception as e:
        logger.warning("Reranker predict failed, skipping: %s", e)
        return candidates

    # Attach reranker scores and sort descending
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # Filter by threshold
    above_threshold = [(c, s) for c, s in scored if s >= config.rerank_threshold]

    # Fallback: if too few pass threshold, take top fallback_top_k regardless
    if len(above_threshold) < config.fallback_top_k:
        return [c for c, _ in scored[:config.fallback_top_k]]

    return [c for c, _ in above_threshold[:config.rerank_top_k]]
