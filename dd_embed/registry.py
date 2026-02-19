"""Adapter registry and factory.

Same pattern as dd-llm: a global registry so any code (including end-users)
can register custom embedding adapters and retrieve them by name.
"""

from __future__ import annotations

from typing import Callable

from dd_embed.base import EmbeddingAdapter

_ADAPTER_REGISTRY: dict[str, type[EmbeddingAdapter] | Callable[..., EmbeddingAdapter]] = {}


def register_adapter(
    name: str, adapter_cls_or_factory: type[EmbeddingAdapter] | Callable[..., EmbeddingAdapter]
) -> None:
    """Register a provider by name."""
    _ADAPTER_REGISTRY[name] = adapter_cls_or_factory


def get_adapter(name: str, **kwargs) -> EmbeddingAdapter:
    """Get an adapter instance by name."""
    if name not in _ADAPTER_REGISTRY:
        available = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
        raise ValueError(f"Unknown adapter '{name}'. Available: {available}")
    return _ADAPTER_REGISTRY[name](**kwargs)


def list_adapters() -> list[str]:
    """List registered adapter names."""
    return sorted(_ADAPTER_REGISTRY.keys())
