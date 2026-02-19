"""HuggingFace transformers adapter — AutoModel + mean pooling.

Ported from semanscope's HuggingFaceModel.  Handles E5 preprocessing,
Qwen trust_remote_code, L2 normalization, and NaN/Inf sanitization.
"""

from __future__ import annotations

import numpy as np

from dd_embed.base import EmbeddingAdapter, EmbeddingResult


class HuggingFaceAdapter(EmbeddingAdapter):
    """Adapter using ``transformers.AutoModel`` with mean pooling.

    Parameters
    ----------
    model_name :
        Display name (e.g. ``"Qwen3-Embedding-0.6B"``).
    model_path :
        HuggingFace model path (e.g. ``"Qwen/Qwen3-Embedding-0.6B"``).
    trust_remote_code :
        Required for Qwen models.
    """

    def __init__(
        self,
        model_name: str = "",
        model_path: str = "",
        trust_remote_code: bool | None = None,
    ):
        self.model_name = model_name or model_path
        self.model_path = model_path or model_name
        self._trust_remote_code = (
            trust_remote_code
            if trust_remote_code is not None
            else ("qwen" in self.model_name.lower())
        )
        self._tokenizer = None
        self._model = None

    def _lazy_load(self):
        if self._tokenizer is not None:
            return
        from transformers import AutoTokenizer, AutoModel

        kw = {"trust_remote_code": True} if self._trust_remote_code else {}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, **kw)
        self._model = AutoModel.from_pretrained(self.model_path, **kw)

    def embed(self, texts: list[str], **kwargs) -> EmbeddingResult:
        start = self._measure_time()

        try:
            import torch
            import torch.nn.functional as F

            self._lazy_load()
            embeddings_list = []

            for text in texts:
                if not text or not text.strip():
                    dummy = self._tokenizer("dummy", return_tensors="pt")
                    dummy_out = self._model(**dummy)
                    zero = np.zeros_like(
                        dummy_out.last_hidden_state.mean(dim=1).detach().numpy()
                    )
                    embeddings_list.append(zero)
                    continue

                # E5 preprocessing
                if "e5" in self.model_name.lower() and "instruct" in self.model_name.lower():
                    text = f"query: {text}"

                inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                outputs = self._model(**inputs)

                # Mean pooling with attention mask
                attention_mask = inputs["attention_mask"]
                token_embs = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
                sum_embs = (token_embs * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                mean_pooled = sum_embs / sum_mask

                # L2 normalize
                mean_pooled = F.normalize(mean_pooled, p=2, dim=1)

                arr = mean_pooled.detach().numpy()
                # Sanitize NaN/Inf
                arr = np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0)
                embeddings_list.append(arr)

            final = np.vstack(embeddings_list)
            return EmbeddingResult(
                embeddings=final,
                success=True,
                provider="huggingface",
                model=self.model_name,
                dimensions=final.shape[1],
                num_texts=final.shape[0],
                latency_ms=self._elapsed_ms(start),
                cost_usd=0.0,
            )
        except Exception as exc:
            return EmbeddingResult(
                embeddings=np.empty((0, 0)),
                success=False,
                provider="huggingface",
                model=self.model_name,
                error=str(exc),
                latency_ms=self._elapsed_ms(start),
            )

    def list_models(self) -> list[str]:
        return [self.model_name]
