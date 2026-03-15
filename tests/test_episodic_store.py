"""Tests for EpisodicStore CRUD and search operations."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest

from engram.config import EmbeddingConfig, EpisodicConfig
from engram.episodic.store import EpisodicStore
from engram.models import MemoryType

_FIXED_EMBEDDING = [0.1] * 384

_EMBED_TARGETS = [
    "engram.episodic.embeddings._get_embeddings",
    "engram.episodic.store._get_embeddings",
    "engram.episodic.episodic_crud._get_embeddings",
    "engram.episodic.episodic_search._get_embeddings",
    "engram.episodic.batch_operations._get_embeddings",
]


def _fake_embeddings(_model, texts, _expected_dim=None):
    return [_FIXED_EMBEDDING for _ in texts]


def _patch_embeddings():
    """Patch _get_embeddings in all importing modules."""
    stack = ExitStack()
    for target in _EMBED_TARGETS:
        stack.enter_context(patch(target, side_effect=_fake_embeddings))
    return stack


@pytest.fixture
def store(tmp_path):
    cfg = EpisodicConfig(
        path=str(tmp_path / "episodic"),
        mode="embedded",
        dedup_enabled=False,
        fts_db_path=str(tmp_path / "fts.db"),
    )
    emb = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
    with _patch_embeddings():
        s = EpisodicStore(config=cfg, embedding_config=emb)
        s._embedding_dim = 384
        yield s


@pytest.mark.asyncio
async def test_remember_returns_id(store):
    """remember() returns a non-empty UUID string."""
    with _patch_embeddings():
        mem_id = await store.remember("Deploy failed on prod")
    assert isinstance(mem_id, str) and len(mem_id) > 0


@pytest.mark.asyncio
async def test_search_finds_stored(store):
    """Stored memory is returned by search query."""
    with _patch_embeddings():
        await store.remember("Database migration completed")
        results = await store.search("migration")
    assert len(results) >= 1
    assert any("migration" in r.content for r in results)


@pytest.mark.asyncio
async def test_search_with_filters(store):
    """Search with memory_type filter returns only matching type."""
    with _patch_embeddings():
        await store.remember("Decision: use PostgreSQL", memory_type=MemoryType.DECISION)
        await store.remember("Fact: server is down", memory_type=MemoryType.FACT)
        results = await store.search(
            "info", filters={"memory_type": {"$eq": "decision"}}
        )
    assert all(r.memory_type == MemoryType.DECISION for r in results)


@pytest.mark.asyncio
async def test_get_by_id(store):
    """get(id) retrieves the exact memory stored."""
    with _patch_embeddings():
        mem_id = await store.remember("Rollback deployed at 14:00")
        mem = await store.get(mem_id)
    assert mem is not None
    assert mem.id == mem_id
    assert "Rollback" in mem.content


@pytest.mark.asyncio
async def test_delete(store):
    """delete(id) returns True; subsequent get returns None."""
    with _patch_embeddings():
        mem_id = await store.remember("Temporary debug note")
        deleted = await store.delete(mem_id)
        mem = await store.get(mem_id)
    assert deleted is True
    assert mem is None


@pytest.mark.asyncio
async def test_stats(store):
    """stats() count increments after inserts."""
    with _patch_embeddings():
        await store.remember("First memory")
        await store.remember("Second memory")
        s = await store.stats()
    assert s["count"] == 2
