"""dd-embed — Shared embedding model abstraction layer for Digital Duck projects.

Public API
----------
- ``EmbeddingAdapter``   — Abstract base class for providers
- ``EmbeddingResult``    — Structured result dataclass
- ``EmbeddingCache``     — Disk-persistent embedding cache
- ``register_adapter``   — Register a custom provider
- ``get_adapter``        — Get an adapter instance by name
- ``list_adapters``      — List registered adapter names
- ``embed``              — Simple convenience function (returns np.ndarray)
"""

from dd_embed.base import EmbeddingAdapter, EmbeddingResult
from dd_embed.registry import register_adapter, get_adapter, list_adapters
from dd_embed.cache import EmbeddingCache

# Trigger auto-registration of built-in adapters
import dd_embed._builtins  # noqa: F401

__all__ = [
    "EmbeddingAdapter",
    "EmbeddingResult",
    "EmbeddingCache",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "embed",
]


def embed(
    texts: list[str],
    *,
    provider: str = "sentence_transformers",
    **kwargs,
) -> "np.ndarray":
    """Simple embedding call — returns numpy array.

    Raises ``RuntimeError`` on failure.
    """
    import numpy as np

    adapter = get_adapter(provider, **{k: v for k, v in kwargs.items() if k in _ADAPTER_INIT_KEYS})
    call_kwargs = {k: v for k, v in kwargs.items() if k not in _ADAPTER_INIT_KEYS}
    result = adapter.embed(texts, **call_kwargs)
    if not result.success:
        raise RuntimeError(f"Embedding failed ({result.provider}): {result.error}")
    return result.embeddings


# Keys that are adapter __init__ params, not embed() params
_ADAPTER_INIT_KEYS = {
    "model_name", "model_path", "device", "local_files_only",
    "trust_remote_code", "api_key", "base_url", "default_model",
    "provider_name", "batch_size", "host",
}
