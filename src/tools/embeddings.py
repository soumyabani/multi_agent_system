from __future__ import annotations

from typing import Sequence

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.config import settings


class EmbeddingModel:
    """Thin abstraction over the configured embedding provider."""

    def __init__(self) -> None:
        config = settings.embedding_config
        self.provider = config.provider
        self.model_name = config.model_name
        self.dimension = config.dimension

        if self.provider == "openai":
            self._client = OpenAIEmbeddings(
                model=self.model_name,
                api_key=settings.secret_key or "EMPTY",
            )
        else:
            self._client = OllamaEmbeddings(
                model=self.model_name,
                base_url=settings.ollama_base_url,
            )

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text using the configured model."""
        if not text or not text.strip():
            raise ValueError("Embedding input text must be non-empty.")
        return self._client.embed_query(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a list of documents using the configured model."""
        cleaned = [item.strip() for item in texts if item and item.strip()]
        if not cleaned:
            return []
        return self._client.embed_documents(cleaned)


def get_embedding_model() -> EmbeddingModel:
    """Return the configured embedding model wrapper."""
    return EmbeddingModel()
