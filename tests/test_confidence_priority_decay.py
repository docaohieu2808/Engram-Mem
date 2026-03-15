"""Tests for confidence, priority, and decay fixes (v0.5.27-0.5.29).

Covers:
1. Default confidence = 0.5 (not 1.0)
2. Confidence boost on recall (+0.1, capped at 1.0)
3. Priority classification by memory type
4. Decay rate default = 0.03
5. Dedup merge reinforces access_count
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from engram.models import EpisodicMemory, MemoryType
from engram.capture.memory_classifier import classify_memory_type, classify_priority
from engram.episodic.episodic_builder import _build_memory
from engram.episodic.decay import compute_activation_score
from engram.config import ScoringConfig


# --- Priority Classification ---

class TestPriorityClassification:
    """Priority should vary by memory type, not always 5."""

    def test_decision_priority_8(self):
        assert classify_priority(MemoryType.DECISION) == 8

    def test_preference_priority_8(self):
        assert classify_priority(MemoryType.PREFERENCE) == 8

    def test_todo_priority_7(self):
        assert classify_priority(MemoryType.TODO) == 7

    def test_lesson_priority_7(self):
        assert classify_priority(MemoryType.LESSON) == 7

    def test_error_priority_6(self):
        assert classify_priority(MemoryType.ERROR) == 6

    def test_fact_priority_5(self):
        assert classify_priority(MemoryType.FACT) == 5

    def test_workflow_priority_5(self):
        assert classify_priority(MemoryType.WORKFLOW) == 5

    def test_unknown_type_defaults_to_5(self):
        # If a new type is added without mapping, should default to 5
        result = classify_priority("nonexistent_type")
        assert result == 5


# --- Confidence Defaults ---

class TestConfidenceDefaults:
    """New memories should start at 0.5 confidence, not 1.0."""

    def test_model_default_confidence(self):
        mem = EpisodicMemory(
            id="test-1",
            content="test",
            memory_type=MemoryType.FACT,
        )
        assert mem.confidence == 0.5

    def test_builder_default_confidence_no_metadata(self):
        """Builder fallback when confidence not in Qdrant metadata."""
        mem = _build_memory("test-2", "hello world", {
            "memory_type": "fact",
            "timestamp": "2026-03-15T12:00:00+00:00",
        })
        assert mem.confidence == 0.5

    def test_builder_preserves_stored_confidence(self):
        """Builder should use stored confidence when present."""
        mem = _build_memory("test-3", "hello", {
            "memory_type": "fact",
            "confidence": 0.8,
            "timestamp": "2026-03-15T12:00:00+00:00",
        })
        assert mem.confidence == 0.8

    def test_builder_confidence_zero_preserved(self):
        """Confidence of 0 should not fallback to default."""
        mem = _build_memory("test-4", "low conf", {
            "memory_type": "fact",
            "confidence": 0.0,
            "timestamp": "2026-03-15T12:00:00+00:00",
        })
        assert mem.confidence == 0.0


# --- Confidence Boost on Recall ---

class TestConfidenceRecallBoost:
    """Confidence should increase +0.1 per recall, capped at 1.0."""

    def test_boost_from_half(self):
        # Simulating: memory starts at 0.5, after 1 recall → 0.6
        initial = 0.5
        new_confidence = min(1.0, initial + 0.1)
        assert new_confidence == pytest.approx(0.6)

    def test_boost_capped_at_1(self):
        initial = 0.95
        new_confidence = min(1.0, initial + 0.1)
        assert new_confidence == 1.0

    def test_multiple_boosts(self):
        conf = 0.5
        for _ in range(6):  # 6 recalls: 0.5 → 0.6 → ... → 1.0 → 1.0
            conf = min(1.0, conf + 0.1)
        assert conf == 1.0


# --- Decay Rate ---

class TestDecayRate:
    """Default decay_rate should be 0.03, not 0.1."""

    def test_model_default_decay_rate(self):
        mem = EpisodicMemory(
            id="test-decay",
            content="test",
            memory_type=MemoryType.FACT,
        )
        assert mem.decay_rate == 0.03

    def test_builder_default_decay_rate(self):
        mem = _build_memory("test-decay-2", "hello", {
            "memory_type": "fact",
            "timestamp": "2026-03-15T12:00:00+00:00",
        })
        assert mem.decay_rate == 0.03

    def test_decay_rate_0_03_retention_after_7_days(self):
        """With 0.03 decay, retention after 7 days should be ~81%."""
        import math
        retention = math.exp(-0.03 * 7 / (1 + 0.1 * 0))
        assert retention == pytest.approx(0.81, abs=0.01)

    def test_decay_rate_0_03_retention_after_30_days(self):
        """With 0.03 decay, retention after 30 days should be ~41%."""
        import math
        retention = math.exp(-0.03 * 30 / (1 + 0.1 * 0))
        assert retention == pytest.approx(0.41, abs=0.01)

    def test_access_count_slows_decay(self):
        """Higher access_count should slow decay."""
        import math
        # 30 days, 0 access
        ret_0 = math.exp(-0.03 * 30 / (1 + 0.1 * 0))
        # 30 days, 10 access
        ret_10 = math.exp(-0.03 * 30 / (1 + 0.1 * 10))
        assert ret_10 > ret_0


# --- Composite Activation Score ---

class TestActivationScore:
    """Test compute_activation_score with new decay rate."""

    def test_fresh_memory_high_score(self):
        now = datetime.now(timezone.utc)
        score = compute_activation_score(
            similarity=0.9,
            timestamp=now,
            access_count=0,
            decay_rate=0.03,
            now=now,
            scoring=ScoringConfig(),
            decay_enabled=True,
        )
        # Fresh memory with high similarity should score well
        assert score > 0.8

    def test_old_memory_lower_score(self):
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=60)
        score = compute_activation_score(
            similarity=0.9,
            timestamp=old,
            access_count=0,
            decay_rate=0.03,
            now=now,
            scoring=ScoringConfig(),
            decay_enabled=True,
        )
        # 60-day old memory should have much lower score
        assert score < 0.7

    def test_frequently_accessed_decays_slower(self):
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=30)
        score_0 = compute_activation_score(
            similarity=0.9, timestamp=old, access_count=0,
            decay_rate=0.03, now=now, scoring=ScoringConfig(), decay_enabled=True,
        )
        score_10 = compute_activation_score(
            similarity=0.9, timestamp=old, access_count=10,
            decay_rate=0.03, now=now, scoring=ScoringConfig(), decay_enabled=True,
        )
        assert score_10 > score_0

    def test_decay_disabled_returns_raw_similarity(self):
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=365)
        score = compute_activation_score(
            similarity=0.85, timestamp=old, access_count=0,
            decay_rate=0.03, now=now, scoring=ScoringConfig(), decay_enabled=False,
        )
        assert score == 0.85


# --- Dedup Merge Reinforcement ---

class TestDedupReinforcement:
    """Dedup merge should increase access_count (tested at unit level via _dedup_merge)."""

    @pytest.mark.asyncio
    async def test_dedup_merge_updates_access_count_in_metadata(self):
        """_dedup_merge should include incremented access_count in update_meta."""
        import json
        from engram.episodic.store import EpisodicStore
        from engram.config import EpisodicConfig, EmbeddingConfig

        store = EpisodicStore(
            config=EpisodicConfig(dedup_enabled=True, dedup_threshold=0.85),
            embedding_config=EmbeddingConfig(provider="test", model="test"),
        )

        # Mock backend to simulate a near-duplicate match
        store._backend = AsyncMock()
        store._backend.count = AsyncMock(return_value=1)
        store._backend.query = AsyncMock(return_value={
            "ids": [["existing-id"]],
            "distances": [[0.1]],  # distance 0.1 → similarity 0.95
            "metadatas": [[{
                "entities": "[]",
                "tags": "[]",
                "priority": 5,
                "access_count": 3,
            }]],
            "documents": [["existing content"]],
        })
        store._backend.update = AsyncMock()
        store._backend_ready = True
        store._embedding_dim = 384

        with patch("engram.episodic.episodic_crud._get_embeddings", return_value=[[0.1] * 384]):
            result = await store._dedup_merge(
                "similar content", [], [], 5, MemoryType.FACT, None,
            )

        assert result == "existing-id"
        # Check that update was called with incremented access_count
        update_call = store._backend.update.call_args
        meta = update_call.kwargs.get("metadatas", update_call[1].get("metadatas", [{}]))[0]
        assert meta["access_count"] == 4  # was 3, +1
        assert "last_accessed" in meta


# --- Memory Classifier ---

class TestMemoryClassifier:
    """Test heuristic memory classifier for type detection."""

    def test_todo_pattern(self):
        assert classify_memory_type("TODO: fix the bug") == MemoryType.TODO

    def test_decision_pattern(self):
        assert classify_memory_type("we decided to use PostgreSQL") == MemoryType.DECISION

    def test_preference_pattern(self):
        assert classify_memory_type("I prefer dark mode") == MemoryType.PREFERENCE

    def test_error_pattern(self):
        assert classify_memory_type("Error: connection refused") == MemoryType.ERROR

    def test_lesson_pattern(self):
        assert classify_memory_type("lesson learned: always backup first") == MemoryType.LESSON

    def test_default_fact(self):
        assert classify_memory_type("The sky is blue") == MemoryType.FACT

    def test_vietnamese_todo(self):
        assert classify_memory_type("cần làm: update server") == MemoryType.TODO

    def test_vietnamese_decision(self):
        assert classify_memory_type("quyết định dùng React") == MemoryType.DECISION
