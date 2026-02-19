"""OpenAI embeddings adapter — also covers OpenRouter via base_url.

Ported from semanscope's OpenRouterModel, generalized.
"""

from __future__ import annotations

import os

import numpy as np

from dd_embed.base import EmbeddingAdapter, EmbeddingResult


class OpenAIEmbedAdapter(EmbeddingAdapter):
    """Embedding adapter for OpenAI-compatible embedding APIs.

    Covers OpenAI, OpenRouter, and any OpenAI-compatible endpoint.

    Parameters
    ----------
    api_key :
        API key. Falls back to ``OPENAI_API_KEY`` env var.
    base_url :
        Custom base URL (e.g. ``https://openrouter.ai/api/v1``).
    default_model :
        Default embedding model.
    provider_name :
        Name in EmbeddingResult.provider.
    batch_size :
        Max texts per API call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "text-embedding-3-small",
        provider_name: str = "openai",
        batch_size: int = 100,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.default_model = default_model
        self.provider_name = provider_name
        self.batch_size = batch_size
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            kwargs: dict = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResult:
        start = self._measure_time()
        model = kwargs.pop("model", None) or self.default_model

        try:
            client = self._get_client()
            all_embeddings: list = []

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                resp = client.embeddings.create(
                    model=model,
                    input=batch,
                    encoding_format="float",
                    **kwargs,
                )
                for item in resp.data:
                    all_embeddings.append(item.embedding)

            result = np.array(all_embeddings)
            return EmbeddingResult(
                embeddings=result,
                success=True,
                provider=self.provider_name,
                model=model,
                dimensions=result.shape[1],
                num_texts=result.shape[0],
                latency_ms=self._elapsed_ms(start),
            )
        except Exception as exc:
            return EmbeddingResult(
                embeddings=np.empty((0, 0)),
                success=False,
                provider=self.provider_name,
                model=model,
                error=str(exc),
                latency_ms=self._elapsed_ms(start),
            )

    def list_models(self) -> list[str]:
        return [self.default_model]
