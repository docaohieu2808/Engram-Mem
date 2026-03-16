"""Local embedding via SentenceTransformer (BAAI/bge-m3, 1024d).

Lazy-loads model on first call. Thread-safe via module-level cache.
Used for episodic memory search/store when config.use_local_for_episodic=True.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger("engram")

_model_cache: SentenceTransformer | None = None
_model_name_cache: str = ""
_model_lock = threading.Lock()


def _get_model(model_name: str = "BAAI/bge-m3") -> SentenceTransformer:
    """Lazy-load SentenceTransformer. Cached after first call. Thread-safe."""
    global _model_cache, _model_name_cache
    if _model_cache is not None and _model_name_cache == model_name:
        return _model_cache
    with _model_lock:
        # Double-checked locking
        if _model_cache is not None and _model_name_cache == model_name:
            return _model_cache
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model: %s", model_name)
        _model_cache = SentenceTransformer(model_name)
        _model_name_cache = model_name
        return _model_cache


def embed(texts: list[str], model_name: str = "BAAI/bge-m3") -> list[list[float]]:
    """Embed texts using local SentenceTransformer. Returns list of 1024-d float vectors."""
    model = _get_model(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]


def embed_single(text: str, model_name: str = "BAAI/bge-m3") -> list[float]:
    """Embed a single text. Convenience wrapper."""
    return embed([text], model_name=model_name)[0]
