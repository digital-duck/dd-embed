"""Tests for EmbeddingAdapter ABC and EmbeddingResult dataclass."""

import pytest
import numpy as np

from dd_embed.base import EmbeddingAdapter, EmbeddingResult


class TestEmbeddingResult:
    def test_creation_minimal(self):
        r = EmbeddingResult(
            embeddings=np.zeros((3, 4)),
            success=True,
            provider="test",
            model="m1",
        )
        assert r.success
        assert r.embeddings.shape == (3, 4)
        assert r.dimensions == 0  # default
        assert r.cost_usd is None
        assert r.error is None

    def test_creation_full(self):
        r = EmbeddingResult(
            embeddings=np.ones((5, 128)),
            success=True,
            provider="openai",
            model="text-embedding-3-small",
            dimensions=128,
            num_texts=5,
            latency_ms=42.0,
            cost_usd=0.001,
        )
        assert r.dimensions == 128
        assert r.num_texts == 5


class TestEmbeddingAdapterABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            EmbeddingAdapter()

    def test_concrete_implementation(self):
        class Dummy(EmbeddingAdapter):
            def embed(self, texts, **kwargs):
                arr = np.random.randn(len(texts), 8)
                return EmbeddingResult(
                    embeddings=arr, success=True, provider="dummy",
                    model="d", dimensions=8, num_texts=len(texts),
                )

        adapter = Dummy()
        result = adapter.embed(["hello", "world"])
        assert result.success
        assert result.embeddings.shape == (2, 8)

    def test_list_models_default(self):
        class Dummy(EmbeddingAdapter):
            def embed(self, texts, **kwargs):
                return EmbeddingResult(
                    embeddings=np.empty((0, 0)), success=True,
                    provider="d", model="d",
                )

        assert Dummy().list_models() == []

    def test_timing_helpers(self):
        class Dummy(EmbeddingAdapter):
            def embed(self, texts, **kwargs):
                return EmbeddingResult(
                    embeddings=np.empty((0, 0)), success=True,
                    provider="d", model="d",
                )

        adapter = Dummy()
        start = adapter._measure_time()
        elapsed = adapter._elapsed_ms(start)
        assert elapsed >= 0
