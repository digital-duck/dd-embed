"""SentenceTransformer adapter — wraps sentence-transformers library.

This is the simplest path for local embedding: one call to `model.encode()`.
Used by maniscope for document and query embeddings.
"""

from __future__ import annotations

import numpy as np

from dd_embed.base import EmbeddingAdapter, EmbeddingResult


class SentenceTransformerAdapter(EmbeddingAdapter):
    """Adapter for the ``sentence-transformers`` library.

    Parameters
    ----------
    model_name :
        HuggingFace model id (e.g. ``"all-MiniLM-L6-v2"``).
    device :
        ``"cpu"`` or ``"cuda"``.
    local_files_only :
        Avoid network calls to HuggingFace Hub.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=self.local_files_only,
            )
        return self._model

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResult:
        start = self._measure_time()
        show_progress = kwargs.pop("show_progress_bar", False)

        try:
            model = self._get_model()
            embeddings = model.encode(
                texts, show_progress_bar=show_progress, **kwargs
            )
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            return EmbeddingResult(
                embeddings=embeddings,
                success=True,
                provider="sentence_transformers",
                model=self.model_name,
                dimensions=embeddings.shape[1],
                num_texts=embeddings.shape[0],
                latency_ms=self._elapsed_ms(start),
                cost_usd=0.0,
            )
        except Exception as exc:
            return EmbeddingResult(
                embeddings=np.empty((0, 0)),
                success=False,
                provider="sentence_transformers",
                model=self.model_name,
                error=str(exc),
                latency_ms=self._elapsed_ms(start),
            )

    def list_models(self) -> list[str]:
        return [self.model_name]
