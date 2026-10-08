from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)


class LLMConfig(BaseModel):
    """Validated LLM settings for local model serving and routing."""

    provider: Literal["ollama", "vllm", "openai"] = Field(default="ollama")
    model_name: str = Field(default="llama3.1:8b-instruct-q4_0")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=32, le=32768)
    base_url: str = Field(default="http://localhost:11434")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("model_name must not be empty")
        return value.strip()


class EmbeddingConfig(BaseModel):
    """Embedding model configuration used by the Researcher agent."""

    provider: Literal["ollama", "openai", "vllm"] = Field(default="ollama")
    model_name: str = Field(default="nomic-embed-text")
    dimension: int = Field(default=768, ge=1)
    base_url: str = Field(default="http://localhost:11434")


class LangfuseConfig(BaseModel):
    """Langfuse tracing configuration."""

    public_key: str = Field(default="pk-lf-dev")
    secret_key: str = Field(default="sk-lf-dev")
    host: str = Field(default="http://localhost:3000")
    enabled: bool = Field(default=True)
    debug: bool = Field(default=False)


class QdrantConfig(BaseModel):
    """Qdrant vector database connection settings."""

    url: str = Field(default="http://localhost:6333")
    api_key: str | None = Field(default=None)
    collection: str = Field(default="incident_knowledge")


class AppSettings(BaseSettings):
    """Global service configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "multi-agent-system"
    environment: str = "development"
    log_level: str = "INFO"

    model_provider: Literal["ollama", "vllm", "openai"] = "ollama"
    model_name: str = "llama3.1:8b-instruct-q4_0"
    model_temperature: float = 0.2
    default_max_tokens: int = 2048

    embedding_model_name: str = "nomic-embed-text"
    embedding_dimension: int = 768
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.0
    max_agent_iterations: int = 4

    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8000/v1"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "incident_knowledge"

    langfuse_public_key: str = "pk-lf-dev"
    langfuse_secret_key: str = "sk-lf-dev"
    langfuse_host: str = "http://localhost:3000"
    langfuse_tracing_enabled: bool = True
    langfuse_debug: bool = False

    redis_url: str = "redis://localhost:6379/0"
    secret_key: str = "change-me-in-production"

    @property
    def llm_config(self) -> LLMConfig:
        provider = self.model_provider
        base_url = self.ollama_base_url if provider == "ollama" else self.vllm_base_url
        return LLMConfig(
            provider=provider,
            model_name=self.model_name,
            temperature=self.model_temperature,
            max_tokens=self.default_max_tokens,
            base_url=base_url,
        )

    @property
    def embedding_config(self) -> EmbeddingConfig:
        provider = "ollama" if self.model_provider == "ollama" else "vllm" if self.model_provider == "vllm" else "openai"
        return EmbeddingConfig(
            provider=provider,
            model_name=self.embedding_model_name,
            dimension=self.embedding_dimension,
            base_url=self.ollama_base_url if provider == "ollama" else self.vllm_base_url,
        )

    @property
    def langfuse_config(self) -> LangfuseConfig:
        return LangfuseConfig(
            public_key=self.langfuse_public_key,
            secret_key=self.langfuse_secret_key,
            host=self.langfuse_host,
            enabled=self.langfuse_tracing_enabled,
            debug=self.langfuse_debug,
        )

    @property
    def qdrant_config(self) -> QdrantConfig:
        return QdrantConfig(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection=self.qdrant_collection,
        )

    def get_chat_model(self):
        """Return a chat model appropriate for the configured provider."""
        if self.model_provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=self.model_name,
                base_url=self.ollama_base_url,
                temperature=self.model_temperature,
            )

        if self.model_provider == "vllm":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.model_name,
                temperature=self.model_temperature,
                api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
                base_url=self.vllm_base_url,
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.model_name,
            temperature=self.model_temperature,
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        )


settings = AppSettings()


SYSTEM_PROMPTS = {
    "researcher": """
You are the Researcher agent for a Deal Desk / Incident Response workspace.
Gather only the facts supported by local retrieval. Use the available evidence to identify relevant incident patterns, policies, and precedent material.
Do not invent missing details. Clearly separate retrieved facts from assumptions.
    """.strip(),
    "analyst": """
You are the Analyst agent. Use only the retrieved evidence to evaluate the user request.
Identify likely root cause candidates, confidence level, key risks, recommendations, and missing information.
Do not invent facts or cite unsupported claims. If the evidence is insufficient, clearly say so.
    """.strip(),
    "writer": """
You are the Writer agent. Produce a clear final answer in natural language using the user request, retrieved evidence, and analyst findings.
Use direct, professional writing. Distinguish evidence from inference, mention uncertainty when needed, and avoid chain-of-thought details.
    """.strip(),
    "supervisor": """
You are the workflow supervisor. Start with the Researcher, then the Analyst, then the Writer. Keep the flow deterministic and efficient.
    """.strip(),
}


def get_runtime_env() -> dict[str, str]:
    """Return a safe dictionary of runtime environment variables for downstream services."""
    return {
        "APP_NAME": settings.app_name,
        "ENVIRONMENT": settings.environment,
        "LOG_LEVEL": settings.log_level,
        "MODEL_PROVIDER": settings.model_provider,
        "MODEL_NAME": settings.model_name,
        "QDRANT_URL": settings.qdrant_url,
        "LANGFUSE_HOST": settings.langfuse_host,
        "LANGFUSE_TRACING_ENABLED": str(settings.langfuse_tracing_enabled).lower(),
        "REDIS_URL": settings.redis_url,
    }
