# Semantic Similarity Search

Embed a corpus of documents, then find the most similar ones to a query.

This is the core pattern used by **maniscope** (document reranking) and
**semanscope** (multilingual semantic analysis).

## Run It

```bash
pip install -r requirements.txt
python main.py
python main.py --query "pets and animals"
python main.py --query "artificial intelligence" --top-k 3
```

## What It Shows

- **Batch embedding**: embed an entire corpus in one call
- **Cosine similarity**: rank documents by semantic relevance
- **Latency tracking**: see embedding time for corpus vs query
