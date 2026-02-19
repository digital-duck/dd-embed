"""Tests for the adapter registry."""

import pytest
import numpy as np

from dd_embed.base import EmbeddingAdapter, EmbeddingResult
from dd_embed.registry import (
    _ADAPTER_REGISTRY,
    register_adapter,
    get_adapter,
    list_adapters,
)


class _MockAdapter(EmbeddingAdapter):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def embed(self, texts, **kwargs):
        return EmbeddingResult(
            embeddings=np.zeros((len(texts), 4)),
            success=True, provider="mock", model="m",
        )


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(_ADAPTER_REGISTRY)
    yield
    _ADAPTER_REGISTRY.clear()
    _ADAPTER_REGISTRY.update(saved)


class TestRegisterAdapter:
    def test_register_class(self):
        register_adapter("mock_test", _MockAdapter)
        adapter = get_adapter("mock_test")
        assert isinstance(adapter, _MockAdapter)

    def test_register_factory(self):
        def factory(**kwargs):
            return _MockAdapter(tag="from_factory", **kwargs)

        register_adapter("factory_test", factory)
        adapter = get_adapter("factory_test")
        assert adapter.kwargs["tag"] == "from_factory"


class TestGetAdapter:
    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown adapter"):
            get_adapter("nonexistent_xyz")

    def test_passes_kwargs(self):
        register_adapter("kw_test", _MockAdapter)
        adapter = get_adapter("kw_test", foo="bar")
        assert adapter.kwargs["foo"] == "bar"


class TestListAdapters:
    def test_list_contains_builtins(self):
        names = list_adapters()
        assert "sentence_transformers" in names
        assert "openai" in names
        assert "ollama" in names

    def test_list_sorted(self):
        names = list_adapters()
        assert names == sorted(names)
