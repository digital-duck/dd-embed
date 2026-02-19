"""Voyage AI embedding adapter.

Ported from semanscope's VoyageModel, stripped of Streamlit deps.
"""

from __future__ import annotations

import os

import numpy as np

from dd_embed.base import EmbeddingAdapter, EmbeddingResult


class VoyageEmbedAdapter(EmbeddingAdapter):
    """Embedding adapter for the Voyage AI API.

    Parameters
    ----------
    api_key :
        Voyage AI API key. Falls back to ``VOYAGE_API_KEY`` env var.
    default_model :
        Default Voyage model.
    batch_size :
        Max texts per API call (Voyage supports up to 128).
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "voyage-3",
        batch_size: int = 128,
    ):
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        self.default_model = default_model
        self.batch_size = batch_size
        self._client = None

    def _get_client(self):
        if self._client is None:
            import voyageai

            self._client = voyageai.Client(api_key=self.api_key)
        return self._client

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResult:
        start = self._measure_time()
        model = kwargs.pop("model", None) or self.default_model

        try:
            client = self._get_client()
            all_embeddings: list = []

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                result = client.embed(
                    batch,
                    model=model,
                    input_type="document",
                    **kwargs,
                )
                if not result or not hasattr(result, "embeddings"):
                    return EmbeddingResult(
                        embeddings=np.empty((0, 0)),
                        success=False,
                        provider="voyage",
                        model=model,
                        error="Response missing embeddings",
                        latency_ms=self._elapsed_ms(start),
                    )
                all_embeddings.extend(result.embeddings)

            arr = np.array(all_embeddings)
            return EmbeddingResult(
                embeddings=arr,
                success=True,
                provider="voyage",
                model=model,
                dimensions=arr.shape[1] if arr.ndim == 2 else 0,
                num_texts=arr.shape[0],
                latency_ms=self._elapsed_ms(start),
            )
        except Exception as exc:
            return EmbeddingResult(
                embeddings=np.empty((0, 0)),
                success=False,
                provider="voyage",
                model=model,
                error=str(exc),
                latency_ms=self._elapsed_ms(start),
            )

    def list_models(self) -> list[str]:
        return [self.default_model]
