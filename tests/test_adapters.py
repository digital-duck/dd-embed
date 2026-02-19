"""Tests for built-in adapters (mocked, no API keys or models needed)."""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from dd_embed.adapters.ollama_embed import OllamaEmbedAdapter
from dd_embed.adapters.openai_embed import OpenAIEmbedAdapter
from dd_embed.adapters.sentence_transformer import SentenceTransformerAdapter


class TestOllamaEmbedAdapter:
    def test_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}

        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp

        adapter = OllamaEmbedAdapter(model_name="bge-m3")
        adapter._session = mock_session

        result = adapter.embed(["hello", "world"])
        assert result.success
        assert result.embeddings.shape == (2, 3)
        assert result.provider == "ollama"

    def test_http_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp

        adapter = OllamaEmbedAdapter()
        adapter._session = mock_session

        result = adapter.embed(["hello"])
        assert not result.success
        assert "500" in result.error

    def test_connection_error(self):
        mock_session = MagicMock()
        mock_session.post.side_effect = ConnectionError("refused")

        adapter = OllamaEmbedAdapter()
        adapter._session = mock_session

        result = adapter.embed(["hello"])
        assert not result.success
        assert "refused" in result.error


class TestOpenAIEmbedAdapter:
    def test_success(self):
        mock_item = MagicMock()
        mock_item.embedding = [0.1, 0.2, 0.3, 0.4]

        mock_resp = MagicMock()
        mock_resp.data = [mock_item, mock_item]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_resp

        adapter = OpenAIEmbedAdapter(api_key="test-key")
        adapter._client = mock_client

        result = adapter.embed(["hello", "world"])
        assert result.success
        assert result.embeddings.shape == (2, 4)
        assert result.provider == "openai"

    def test_batching(self):
        mock_item = MagicMock()
        mock_item.embedding = [0.1, 0.2]

        mock_resp = MagicMock()
        mock_resp.data = [mock_item]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_resp

        adapter = OpenAIEmbedAdapter(api_key="test-key", batch_size=1)
        adapter._client = mock_client

        result = adapter.embed(["a", "b", "c"])
        assert result.success
        assert mock_client.embeddings.create.call_count == 3


class TestSentenceTransformerAdapter:
    def test_success(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(2, 384)

        adapter = SentenceTransformerAdapter(model_name="all-MiniLM-L6-v2")
        adapter._model = mock_model

        result = adapter.embed(["hello", "world"])
        assert result.success
        assert result.embeddings.shape == (2, 384)
        assert result.provider == "sentence_transformers"
        assert result.cost_usd == 0.0

    def test_failure(self):
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("model error")

        adapter = SentenceTransformerAdapter()
        adapter._model = mock_model

        result = adapter.embed(["hello"])
        assert not result.success
        assert "model error" in result.error
