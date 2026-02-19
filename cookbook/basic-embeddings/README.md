# Basic Embeddings

Generate text embeddings using any dd-embed provider.

Ported from `pocoflow-tool-embeddings`, now provider-agnostic.

## Run It

```bash
pip install -r requirements.txt

# Local (free, no API key)
python main.py "Hello world"

# OpenAI
export OPENAI_API_KEY="your-key"
python main.py --provider openai "Hello world"

# Ollama (local server)
python main.py --provider ollama --model bge-m3 "Hello world"
```

## What It Shows

- **Provider-agnostic**: same code works with any dd-embed adapter
- **Adapter registry**: `list_adapters()` shows all available providers
- **EmbeddingResult**: structured response with dimensions, latency, cost
