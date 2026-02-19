"""dd-embed Embedding Cache Demo — incremental caching across runs.

Demonstrates EmbeddingCache: first run computes all embeddings; second run
loads from cache and only computes new ones. This is the pattern used by
semanscope for per-word embedding caching across multilingual datasets.

Usage:
    python main.py           # Run 1: computes all, saves cache
    python main.py           # Run 2: loads from cache (0 computed)
    python main.py --extra   # Run 3: loads cached + computes new words only
    python main.py --clear   # Clear the cache
"""

import click
import numpy as np
from dd_embed import EmbeddingCache, get_adapter


WORDS_BASE = ["apple", "banana", "cherry", "date", "elderberry"]
WORDS_EXTRA = ["apple", "banana", "cherry", "fig", "grape", "honeydew"]


@click.command()
@click.option("--provider", default="sentence_transformers", help="Embedding provider")
@click.option("--model", default=None, help="Model name")
@click.option("--extra", is_flag=True, help="Use extended word list (tests incremental caching)")
@click.option("--clear", is_flag=True, help="Clear cache and exit")
def main(provider, model, extra, clear):
    """Demonstrate dd-embed's disk-persistent embedding cache."""

    cache = EmbeddingCache()  # default: ~/projects/embedding_cache/dd_embed/master.pkl

    if clear:
        cache.clear()
        cache.save()
        print("Cache cleared.")
        return

    words = WORDS_EXTRA if extra else WORDS_BASE

    # Build adapter
    adapter_kwargs = {}
    if model:
        if provider in ("openai", "openrouter"):
            adapter_kwargs["default_model"] = model
        else:
            adapter_kwargs["model_name"] = model
    adapter = get_adapter(provider, **adapter_kwargs)

    model_name = model or adapter.list_models()[0] if adapter.list_models() else provider

    # Use cache — only computes uncached words
    def embed_fn(texts):
        result = adapter.embed(texts)
        if not result.success:
            raise RuntimeError(f"Embedding failed: {result.error}")
        return result.embeddings

    embeddings, cached_count, computed_count = cache.get_embeddings(
        texts=words,
        model_name=model_name,
        scope="demo",
        embed_fn=embed_fn,
    )

    print(f"Words:    {words}")
    print(f"Shape:    {embeddings.shape}")
    print(f"Cached:   {cached_count}")
    print(f"Computed: {computed_count}")
    print()

    # Show cache stats
    stats = cache.stats()
    print(f"Cache stats: {stats['total_models']} model(s), {stats['total_texts']} text(s) cached")
    for m, scopes in stats["models"].items():
        for scope, count in scopes.items():
            print(f"  {m} / {scope}: {count} texts")

    # Save to disk
    cache.save()
    print("\nCache saved to disk.")


if __name__ == "__main__":
    main()
