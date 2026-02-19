"""Auto-register built-in adapters on import."""

from __future__ import annotations

import os

from dd_embed.registry import register_adapter
from dd_embed.adapters.sentence_transformer import SentenceTransformerAdapter
from dd_embed.adapters.huggingface import HuggingFaceAdapter
from dd_embed.adapters.ollama_embed import OllamaEmbedAdapter
from dd_embed.adapters.openai_embed import OpenAIEmbedAdapter
from dd_embed.adapters.gemini_embed import GeminiEmbedAdapter
from dd_embed.adapters.voyage_embed import VoyageEmbedAdapter


def _make_openrouter(**kwargs):
    """Factory for OpenRouter — OpenAI-compatible embedding endpoint."""
    return OpenAIEmbedAdapter(
        api_key=kwargs.get("api_key") or os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        default_model=kwargs.get("default_model", "openai/text-embedding-3-small"),
        provider_name="openrouter",
        batch_size=kwargs.get("batch_size", 100),
    )


register_adapter("sentence_transformers", SentenceTransformerAdapter)
register_adapter("huggingface", HuggingFaceAdapter)
register_adapter("ollama", OllamaEmbedAdapter)
register_adapter("openai", OpenAIEmbedAdapter)
register_adapter("openrouter", _make_openrouter)
register_adapter("gemini", GeminiEmbedAdapter)
register_adapter("voyage", VoyageEmbedAdapter)
