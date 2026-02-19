# Embedding Cache Demo

Demonstrates `EmbeddingCache` — dd-embed's disk-persistent, per-word
granular cache (ported from semanscope).

## Run It

```bash
pip install -r requirements.txt

# Run 1: computes all 5 words, saves to disk
python main.py

# Run 2: loads all 5 from cache (0 computed)
python main.py

# Run 3: loads 3 cached, computes only 3 new words
python main.py --extra

# Clear cache
python main.py --clear
```

## What It Shows

- **Incremental caching**: only uncached texts hit the model
- **Disk persistence**: cache survives across Python sessions
- **Cache stats**: inspect what's cached by model and scope
- **Semanscope pattern**: this is exactly how semanscope caches
  per-word embeddings across multilingual datasets
