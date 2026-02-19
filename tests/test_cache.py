"""Tests for EmbeddingCache."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dd_embed.cache import EmbeddingCache


@pytest.fixture
def cache(tmp_path):
    return EmbeddingCache(cache_path=tmp_path / "test_cache.pkl")


class TestEmbeddingCache:
    def test_compute_and_cache(self, cache):
        def embed_fn(texts):
            return np.random.randn(len(texts), 8)

        embs, cached, computed = cache.get_embeddings(
            ["a", "b", "c"], model_name="m1", scope="en", embed_fn=embed_fn,
        )
        assert embs.shape == (3, 8)
        assert cached == 0
        assert computed == 3

    def test_incremental_cache(self, cache):
        call_count = 0

        def embed_fn(texts):
            nonlocal call_count
            call_count += 1
            return np.ones((len(texts), 4))

        cache.get_embeddings(["a", "b"], "m1", "en", embed_fn)
        assert call_count == 1

        embs, cached, computed = cache.get_embeddings(
            ["a", "b", "c"], "m1", "en", embed_fn,
        )
        assert call_count == 2
        assert cached == 2
        assert computed == 1
        assert embs.shape == (3, 4)

    def test_force_recompute(self, cache):
        def embed_fn(texts):
            return np.zeros((len(texts), 4))

        cache.get_embeddings(["a"], "m1", "en", embed_fn)
        embs, cached, computed = cache.get_embeddings(
            ["a"], "m1", "en", embed_fn, force_recompute=True,
        )
        assert cached == 0
        assert computed == 1

    def test_persistence(self, tmp_path):
        path = tmp_path / "persist.pkl"

        def embed_fn(texts):
            return np.ones((len(texts), 4))

        cache1 = EmbeddingCache(cache_path=path)
        cache1.get_embeddings(["x", "y"], "m1", "en", embed_fn)
        cache1.save()

        cache2 = EmbeddingCache(cache_path=path)
        embs, cached, computed = cache2.get_embeddings(
            ["x", "y"], "m1", "en", embed_fn,
        )
        assert cached == 2
        assert computed == 0

    def test_clear_all(self, cache):
        def embed_fn(texts):
            return np.ones((len(texts), 4))

        cache.get_embeddings(["a"], "m1", "en", embed_fn)
        cache.clear()
        assert cache.stats()["total_texts"] == 0

    def test_clear_model(self, cache):
        def embed_fn(texts):
            return np.ones((len(texts), 4))

        cache.get_embeddings(["a"], "m1", "en", embed_fn)
        cache.get_embeddings(["b"], "m2", "en", embed_fn)
        cache.clear(model_name="m1")
        assert cache.stats()["total_texts"] == 1

    def test_stats(self, cache):
        def embed_fn(texts):
            return np.ones((len(texts), 4))

        cache.get_embeddings(["a", "b"], "m1", "en", embed_fn)
        cache.get_embeddings(["c"], "m1", "fr", embed_fn)
        stats = cache.stats()
        assert stats["total_models"] == 1
        assert stats["total_texts"] == 3
        assert stats["models"]["m1"]["en"] == 2
        assert stats["models"]["m1"]["fr"] == 1

    def test_embed_fn_none_raises(self, cache):
        def embed_fn(texts):
            return None

        with pytest.raises(ValueError, match="returned None"):
            cache.get_embeddings(["a"], "m1", "en", embed_fn)

    def test_embed_fn_shape_mismatch_raises(self, cache):
        def embed_fn(texts):
            return np.ones((1, 4))  # wrong: should be (len(texts), 4)

        with pytest.raises(ValueError, match="requested"):
            cache.get_embeddings(["a", "b"], "m1", "en", embed_fn)
