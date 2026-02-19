"""Ollama embedding adapter — HTTP to local Ollama server.

Ported from semanscope's OllamaModel, stripped of Streamlit deps.
"""

from __future__ import annotations

import os

import numpy as np

from dd_embed.base import EmbeddingAdapter, EmbeddingResult


class OllamaEmbedAdapter(EmbeddingAdapter):
    """Embedding adapter for local Ollama server.

    Parameters
    ----------
    model_name :
        Ollama model name (e.g. ``"bge-m3"``, ``"nomic-embed-text"``).
    host :
        Ollama server URL. Defaults to ``OLLAMA_HOST`` env var or
        ``http://localhost:11434``.
    """

    def __init__(
        self,
        model_name: str = "bge-m3",
        host: str | None = None,
    ):
        self.model_name = model_name
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResult:
        start = self._measure_time()
        url = f"{self.host}/api/embeddings"
        session = self._get_session()
        embeddings = []

        try:
            for text in texts:
                resp = session.post(url, json={"model": self.model_name, "prompt": text})
                if resp.status_code != 200:
                    return EmbeddingResult(
                        embeddings=np.empty((0, 0)),
                        success=False,
                        provider="ollama",
                        model=self.model_name,
                        error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                        latency_ms=self._elapsed_ms(start),
                    )
                emb = resp.json().get("embedding")
                if not emb:
                    return EmbeddingResult(
                        embeddings=np.empty((0, 0)),
                        success=False,
                        provider="ollama",
                        model=self.model_name,
                        error=f"No embedding returned for: {text[:50]}",
                        latency_ms=self._elapsed_ms(start),
                    )
                arr = np.array(emb)
                if np.isnan(arr).any() or np.allclose(arr, 0.0):
                    continue
                embeddings.append(arr)

            if not embeddings:
                return EmbeddingResult(
                    embeddings=np.empty((0, 0)),
                    success=False,
                    provider="ollama",
                    model=self.model_name,
                    error="No valid embeddings produced",
                    latency_ms=self._elapsed_ms(start),
                )

            result = np.vstack(embeddings)
            return EmbeddingResult(
                embeddings=result,
                success=True,
                provider="ollama",
                model=self.model_name,
                dimensions=result.shape[1],
                num_texts=result.shape[0],
                latency_ms=self._elapsed_ms(start),
                cost_usd=0.0,
            )
        except Exception as exc:
            return EmbeddingResult(
                embeddings=np.empty((0, 0)),
                success=False,
                provider="ollama",
                model=self.model_name,
                error=str(exc),
                latency_ms=self._elapsed_ms(start),
            )

    def list_models(self) -> list[str]:
        return [self.model_name]
