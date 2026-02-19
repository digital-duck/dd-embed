"""Google Gemini embedding adapter.

Ported from semanscope's GeminiModel, stripped of Streamlit deps.
"""

from __future__ import annotations

import os

import numpy as np

from dd_embed.base import EmbeddingAdapter, EmbeddingResult


class GeminiEmbedAdapter(EmbeddingAdapter):
    """Embedding adapter for the Google Gemini API.

    Parameters
    ----------
    api_key :
        Google AI API key. Falls back to ``GEMINI_API_KEY`` env var.
    default_model :
        Default embedding model path.
    batch_size :
        Max texts per API call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gemini-embedding-001",
        batch_size: int = 100,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.default_model = default_model
        self.batch_size = batch_size
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._client = genai
        return self._client

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResult:
        start = self._measure_time()
        model = kwargs.pop("model", None) or self.default_model

        try:
            client = self._get_client()
            all_embeddings: list = []

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                resp = client.embed_content(
                    model=model,
                    content=batch,
                    task_type="retrieval_document",
                    **kwargs,
                )
                if hasattr(resp, "embedding"):
                    all_embeddings.append(resp.embedding)
                elif hasattr(resp, "embeddings"):
                    all_embeddings.extend(resp.embeddings)
                else:
                    return EmbeddingResult(
                        embeddings=np.empty((0, 0)),
                        success=False,
                        provider="gemini",
                        model=model,
                        error="Unexpected response format from Gemini API",
                        latency_ms=self._elapsed_ms(start),
                    )

            result = np.array(all_embeddings)
            return EmbeddingResult(
                embeddings=result,
                success=True,
                provider="gemini",
                model=model,
                dimensions=result.shape[1] if result.ndim == 2 else 0,
                num_texts=result.shape[0],
                latency_ms=self._elapsed_ms(start),
            )
        except Exception as exc:
            return EmbeddingResult(
                embeddings=np.empty((0, 0)),
                success=False,
                provider="gemini",
                model=model,
                error=str(exc),
                latency_ms=self._elapsed_ms(start),
            )

    def list_models(self) -> list[str]:
        return [self.default_model]
