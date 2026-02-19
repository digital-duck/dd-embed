"""Embedding cache — disk-persistent, per-word granular.

Ported from semanscope's EmbeddingCache, stripped of Streamlit dependencies.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import numpy as np


class EmbeddingCache:
    """Disk-persistent embedding cache with per-word granularity.

    Structure: ``{model_name: {scope: {text: embedding_vector}}}``

    Parameters
    ----------
    cache_path :
        Path to the pickle cache file.  Parent directories are created
        automatically.
    """

    def __init__(self, cache_path: str | Path | None = None):
        if cache_path is None:
            cache_path = Path.home() / "projects" / "embedding_cache" / "dd_embed" / "master.pkl"
        self._path = Path(cache_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict = self._load()
        self._dirty = False

    # -- persistence ---------------------------------------------------------

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def save(self) -> None:
        """Flush dirty cache to disk."""
        if self._dirty:
            with open(self._path, "wb") as f:
                pickle.dump(self._cache, f)
            self._dirty = False

    # -- main API ------------------------------------------------------------

    def get_embeddings(
        self,
        texts: list[str],
        model_name: str,
        scope: str,
        embed_fn: Callable[[list[str]], np.ndarray],
        *,
        force_recompute: bool = False,
    ) -> tuple[np.ndarray, int, int]:
        """Get embeddings, computing only uncached texts.

        Parameters
        ----------
        texts :
            Texts to embed.
        model_name :
            Model identifier (cache key level 1).
        scope :
            Scope/language/dataset key (cache key level 2).
        embed_fn :
            ``fn(texts) -> np.ndarray`` called for uncached texts.
        force_recompute :
            Ignore cache and recompute all.

        Returns
        -------
        (embeddings, cached_count, computed_count)
        """
        self._cache.setdefault(model_name, {}).setdefault(scope, {})
        scoped = self._cache[model_name][scope]

        results: list[np.ndarray | None] = []
        to_compute: list[str] = []

        for text in texts:
            if not force_recompute and text in scoped:
                results.append(scoped[text])
            else:
                to_compute.append(text)
                results.append(None)

        cached_count = len(texts) - len(to_compute)
        computed_count = 0

        if to_compute:
            new_embs = embed_fn(to_compute)
            if new_embs is None:
                raise ValueError(f"embed_fn returned None for {len(to_compute)} texts")
            if not isinstance(new_embs, np.ndarray):
                new_embs = np.array(new_embs)
            if new_embs.ndim == 1:
                new_embs = new_embs.reshape(1, -1)
            if new_embs.shape[0] != len(to_compute):
                raise ValueError(
                    f"embed_fn returned {new_embs.shape[0]} embeddings "
                    f"but {len(to_compute)} texts were requested"
                )
            idx = 0
            for i, text in enumerate(texts):
                if results[i] is None:
                    results[i] = new_embs[idx]
                    scoped[text] = new_embs[idx]
                    idx += 1
                    computed_count += 1
                    self._dirty = True

        return np.array(results), cached_count, computed_count

    # -- housekeeping --------------------------------------------------------

    def clear(
        self, model_name: str | None = None, scope: str | None = None
    ) -> None:
        """Clear cache (optionally filtered by model/scope)."""
        if model_name is None:
            self._cache.clear()
        elif scope is None:
            self._cache.pop(model_name, None)
        else:
            self._cache.get(model_name, {}).pop(scope, None)
        self._dirty = True

    def stats(self) -> dict:
        """Return cache statistics."""
        total_texts = sum(
            len(scope_dict)
            for model_dict in self._cache.values()
            for scope_dict in model_dict.values()
        )
        return {
            "total_models": len(self._cache),
            "total_texts": total_texts,
            "models": {
                model: {
                    scope: len(scope_dict)
                    for scope, scope_dict in model_dict.items()
                }
                for model, model_dict in self._cache.items()
            },
        }
