"""Embedding adapter abstract base class and response dataclass."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EmbeddingResult:
    """Structured embedding result with metadata.

    Attributes
    ----------
    embeddings : np.ndarray
        2-D array of shape ``(n_texts, embedding_dim)``.
    success : bool
        Whether the embedding call succeeded.
    provider : str
        Name of the provider that produced the embeddings.
    model : str
        Model identifier used.
    dimensions : int
        Embedding dimensionality.
    num_texts : int
        Number of texts that were embedded.
    latency_ms : float
        Wall-clock time in milliseconds.
    cost_usd : float or None
        Estimated cost, if known.
    error : str or None
        Error message on failure.
    """

    embeddings: np.ndarray
    success: bool
    provider: str
    model: str
    dimensions: int = 0
    num_texts: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None
    error: str | None = None


class EmbeddingAdapter(ABC):
    """Abstract interface for embedding providers.

    All dd-embed backends must implement the synchronous ``embed()`` method.
    """

    @abstractmethod
    def embed(self, texts: list[str], **kwargs) -> EmbeddingResult:
        """Embed a batch of texts.

        Parameters
        ----------
        texts :
            List of strings to embed.
        **kwargs :
            Provider-specific options (e.g. ``model``, ``batch_size``).

        Returns
        -------
        EmbeddingResult
            Always returned, even on failure (check ``.success``).
        """
        ...

    def list_models(self) -> list[str]:
        """List available models for this adapter."""
        return []

    def _measure_time(self) -> float:
        return time.perf_counter()

    def _elapsed_ms(self, start: float) -> float:
        return (time.perf_counter() - start) * 1000
