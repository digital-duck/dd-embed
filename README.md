# dd-embed

Shared embedding model abstraction layer for Digital Duck projects.

Extracted from semanscope and maniscope. Zero heavy deps in core (only numpy).
Adapters lazy-import their SDKs only when used.

## Install

```bash
pip install dd-embed                          # numpy only
pip install "dd-embed[sentence-transformers]" # + sentence-transformers
pip install "dd-embed[openai]"                # + OpenAI SDK (also covers openrouter)
pip install "dd-embed[voyageai]"              # + Voyage AI SDK
pip install "dd-embed[gemini]"                # + Google GenAI SDK
pip install "dd-embed[all]"                   # all provider SDKs
```

## Quick Start

```python
from dd_embed import embed

# Using sentence-transformers (local, free)
embeddings = embed(["hello", "world"], provider="sentence_transformers",
                   model_name="all-MiniLM-L6-v2")
print(embeddings.shape)  # (2, 384)

# Using OpenAI
embeddings = embed(["hello"], provider="openai", api_key="sk-...")

# Using Ollama (local)
embeddings = embed(["hello"], provider="ollama", model_name="bge-m3")
```

## Built-in Adapters

| Name | Class | SDK | Notes |
|------|-------|-----|-------|
| `sentence_transformers` | `SentenceTransformerAdapter` | `sentence-transformers` | Local, free, used by maniscope |
| `huggingface` | `HuggingFaceAdapter` | `transformers` + `torch` | AutoModel + mean pooling, E5/Qwen support |
| `ollama` | `OllamaEmbedAdapter` | `requests` | Local Ollama server |
| `openai` | `OpenAIEmbedAdapter` | `openai` | OpenAI embeddings API |
| `openrouter` | `OpenAIEmbedAdapter` (configured) | `openai` | OpenAI-compat endpoint |
| `gemini` | `GeminiEmbedAdapter` | `google-generativeai` | Google Gemini embeddings |
| `voyage` | `VoyageEmbedAdapter` | `voyageai` | Voyage AI embeddings |

## Embedding Cache

Disk-persistent, per-word granular cache (ported from semanscope):

```python
from dd_embed import EmbeddingCache, get_adapter

cache = EmbeddingCache()  # default: ~/projects/embedding_cache/dd_embed/master.pkl
adapter = get_adapter("sentence_transformers", model_name="all-MiniLM-L6-v2")

embeddings, cached, computed = cache.get_embeddings(
    texts=["apple", "banana", "cherry"],
    model_name="all-MiniLM-L6-v2",
    scope="en",
    embed_fn=lambda texts: adapter.embed(texts).embeddings,
)
print(f"Cached: {cached}, Computed: {computed}")
cache.save()
```

## Custom Adapters

```python
from dd_embed import EmbeddingAdapter, EmbeddingResult, register_adapter, embed
import numpy as np

class MyAdapter(EmbeddingAdapter):
    def embed(self, texts, **kwargs):
        vecs = np.random.randn(len(texts), 128)  # your logic here
        return EmbeddingResult(
            embeddings=vecs, success=True, provider="my_api",
            model="v1", dimensions=128, num_texts=len(texts),
        )

register_adapter("my_api", MyAdapter)
result = embed(["hello"], provider="my_api")
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | -- |
| `OPENROUTER_API_KEY` | OpenRouter API key | -- |
| `GEMINI_API_KEY` | Google Gemini API key | -- |
| `VOYAGE_API_KEY` | Voyage AI API key | -- |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |

## License

MIT
