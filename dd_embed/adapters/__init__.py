"""Built-in embedding adapter implementations."""

from dd_embed.adapters.sentence_transformer import SentenceTransformerAdapter
from dd_embed.adapters.huggingface import HuggingFaceAdapter
from dd_embed.adapters.openai_embed import OpenAIEmbedAdapter
from dd_embed.adapters.ollama_embed import OllamaEmbedAdapter
from dd_embed.adapters.gemini_embed import GeminiEmbedAdapter
from dd_embed.adapters.voyage_embed import VoyageEmbedAdapter

__all__ = [
    "SentenceTransformerAdapter",
    "HuggingFaceAdapter",
    "OpenAIEmbedAdapter",
    "OllamaEmbedAdapter",
    "GeminiEmbedAdapter",
    "VoyageEmbedAdapter",
]
