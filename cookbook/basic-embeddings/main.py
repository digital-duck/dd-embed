"""dd-embed Basic Embeddings — generate text embeddings via any provider.

Ported from pocoflow-tool-embeddings, now provider-agnostic via dd-embed.

Usage:
    python main.py                                          # default: sentence_transformers
    python main.py --provider openai --model text-embedding-3-small
    python main.py --provider ollama --model bge-m3
    python main.py --provider voyage --model voyage-3
    python main.py "Your custom text here"
"""

import click
from dd_embed import embed, get_adapter, list_adapters


@click.command()
@click.argument("text", default="What is the meaning of life?")
@click.option("--provider", default="sentence_transformers", help="Embedding provider name")
@click.option("--model", default=None, help="Model name (provider-specific)")
def main(text, provider, model):
    """Generate a text embedding using any dd-embed provider."""

    print(f"Available providers: {list_adapters()}")
    print(f"Using provider: {provider}")
    print()

    # Build adapter kwargs
    adapter_kwargs = {}
    if model:
        if provider in ("openai", "openrouter"):
            adapter_kwargs["default_model"] = model
        else:
            adapter_kwargs["model_name"] = model

    # Get adapter and embed
    adapter = get_adapter(provider, **adapter_kwargs)
    result = adapter.embed([text])

    if not result.success:
        print(f"ERROR: {result.error}")
        return

    embedding = result.embeddings[0]
    print(f"Text:       {text[:80]}...")
    print(f"Provider:   {result.provider}")
    print(f"Model:      {result.model}")
    print(f"Dimensions: {result.dimensions}")
    print(f"Latency:    {result.latency_ms:.1f} ms")
    print(f"First 5:    {embedding[:5].tolist()}")
    print(f"Norm:       {(embedding @ embedding) ** 0.5:.4f}")

    if result.cost_usd is not None:
        print(f"Cost:       ${result.cost_usd:.6f}")


if __name__ == "__main__":
    main()
