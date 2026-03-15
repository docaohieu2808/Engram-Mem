"""Shared pytest fixtures for engram test suite."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from engram.config import EmbeddingConfig, EpisodicConfig, SemanticConfig
from engram.episodic.store import EpisodicStore
from engram.models import EdgeDef, NodeDef, SchemaDefinition
from engram.semantic import create_graph
from engram.semantic.graph import SemanticGraph

# Fixed 384-dim vector for deterministic embedding tests
_FIXED_EMBEDDING = [0.1] * 384


@pytest.fixture(autouse=True)
def reset_resource_monitor():
    """Reset the global ResourceMonitor before each test to avoid tier contamination."""
    from engram.resource_tier import setup_resource_monitor
    setup_resource_monitor(failure_threshold=100, cooldown_seconds=0.0)
    yield


def _mock_embeddings(_model: str, texts: list[str], _expected_dim: int | None = None) -> list[list[float]]:
    return [_FIXED_EMBEDDING for _ in texts]


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Temporary directory for config files."""
    return tmp_path / "config"


@pytest.fixture
def mock_embeddings():
    """Patch _get_embeddings in all modules that import it directly."""
    with patch("engram.episodic.embeddings._get_embeddings", side_effect=_mock_embeddings), \
         patch("engram.episodic.store._get_embeddings", side_effect=_mock_embeddings), \
         patch("engram.episodic.episodic_crud._get_embeddings", side_effect=_mock_embeddings), \
         patch("engram.episodic.episodic_search._get_embeddings", side_effect=_mock_embeddings), \
         patch("engram.episodic.batch_operations._get_embeddings", side_effect=_mock_embeddings) as m:
        yield m


@pytest.fixture
def episodic_store(tmp_path, mock_embeddings):
    """EpisodicStore backed by tmp path with mocked embeddings (embedded Qdrant, no server needed)."""
    config = EpisodicConfig(
        path=str(tmp_path / "episodic"),
        mode="embedded",
        dedup_enabled=False,
        fts_db_path=str(tmp_path / "fts.db"),
    )
    embed_config = EmbeddingConfig(provider="test", model="all-MiniLM-L6-v2")
    store = EpisodicStore(config=config, embedding_config=embed_config)
    # Set embedding dim explicitly so embedded Qdrant creates collection with correct size
    store._embedding_dim = 384
    return store


@pytest.fixture
def semantic_graph(tmp_path):
    """SemanticGraph backed by tmp SQLite path via create_graph factory."""
    config = SemanticConfig(path=str(tmp_path / "semantic.db"))
    return create_graph(config)


@pytest.fixture
def sample_schema():
    """SchemaDefinition with 2 node types and 1 edge type."""
    return SchemaDefinition(
        nodes=[
            NodeDef(name="Service", description="A software service"),
            NodeDef(name="Team", description="An engineering team"),
        ],
        edges=[
            EdgeDef(name="owns", from_types=["Team"], to_types=["Service"]),
        ],
    )
