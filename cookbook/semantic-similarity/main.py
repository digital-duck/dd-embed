"""dd-embed Semantic Similarity — find the most similar text in a corpus.

Demonstrates the core use case for maniscope and semanscope: embed a corpus
of documents, then embed a query and rank by cosine similarity.

Usage:
    python main.py
    python main.py --query "machine learning algorithms"
    python main.py --provider ollama --model bge-m3
"""

import click
import numpy as np
from dd_embed import get_adapter


SAMPLE_CORPUS = [
    "The cat sat on the mat",
    "Dogs are loyal companions",
    "Python is a programming language",
    "Machine learning uses neural networks",
    "The weather is sunny today",
    "Deep learning transforms artificial intelligence",
    "Cats and dogs are popular pets",
    "Natural language processing understands text",
    "The stock market fluctuates daily",
    "Transformers revolutionized NLP research",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and a matrix of vectors."""
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (a_norm @ b_norm.T).squeeze()


@click.command()
@click.option("--query", default="How do neural networks learn?", help="Search query")
@click.option("--provider", default="sentence_transformers", help="Embedding provider")
@click.option("--model", default=None, help="Model name")
@click.option("--top-k", default=5, help="Number of results to show")
def main(query, provider, model, top_k):
    """Semantic similarity search over a sample corpus."""

    # Build adapter
    adapter_kwargs = {}
    if model:
        if provider in ("openai", "openrouter"):
            adapter_kwargs["default_model"] = model
        else:
            adapter_kwargs["model_name"] = model
    adapter = get_adapter(provider, **adapter_kwargs)

    # Embed corpus
    print(f"Embedding {len(SAMPLE_CORPUS)} documents with {provider}...")
    corpus_result = adapter.embed(SAMPLE_CORPUS)
    if not corpus_result.success:
        print(f"ERROR embedding corpus: {corpus_result.error}")
        return

    # Embed query
    query_result = adapter.embed([query])
    if not query_result.success:
        print(f"ERROR embedding query: {query_result.error}")
        return

    # Rank by cosine similarity
    similarities = cosine_similarity(query_result.embeddings, corpus_result.embeddings)
    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    print(f"\nQuery: \"{query}\"")
    print(f"Model: {corpus_result.model} ({corpus_result.dimensions}d)")
    print(f"Corpus latency: {corpus_result.latency_ms:.0f}ms | Query latency: {query_result.latency_ms:.0f}ms")
    print(f"\nTop {top_k} results:")
    print("-" * 60)
    for rank, idx in enumerate(ranked_indices, 1):
        print(f"  {rank}. [{similarities[idx]:.4f}] {SAMPLE_CORPUS[idx]}")


if __name__ == "__main__":
    main()
