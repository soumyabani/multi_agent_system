"""Langfuse observability integration for tracing agent workflows."""

from __future__ import annotations

import logging
from typing import Any

from langfuse import Langfuse
from src.config import settings

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Wrapper for Langfuse client initialization and management."""

    _instance: LangfuseTracer | None = None
    _client: Langfuse | None = None

    def __new__(cls) -> LangfuseTracer:
        """Singleton pattern to ensure only one tracer instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize Langfuse client if tracing is enabled."""
        if self._client is not None:
            return

        langfuse_cfg = settings.langfuse_config

        if not langfuse_cfg.enabled:
            logger.info("Langfuse tracing is disabled in configuration")
            self._client = None
            return

        try:
            self._client = Langfuse(
                public_key=langfuse_cfg.public_key,
                secret_key=langfuse_cfg.secret_key,
                host=langfuse_cfg.host,
                debug=langfuse_cfg.debug,
            )
            logger.info(f"Langfuse tracing initialized at {langfuse_cfg.host}")
        except Exception as exc:
            logger.warning(f"Failed to initialize Langfuse: {exc}. Tracing will be disabled.")
            self._client = None

    def get_client(self) -> Langfuse | None:
        """Return the Langfuse client if available."""
        return self._client

    def flush(self) -> None:
        """Flush any pending traces to Langfuse."""
        if self._client is not None:
            try:
                self._client.flush()
                logger.debug("Langfuse traces flushed")
            except Exception as exc:
                logger.debug(f"Error flushing Langfuse: {exc}")


def get_langfuse_tracer() -> LangfuseTracer:
    """Get the singleton Langfuse tracer instance."""
    return LangfuseTracer()


def get_langfuse_handler() -> Any | None:
    """Return a LangChain-compatible callback handler for Langfuse."""
    langfuse_cfg = settings.langfuse_config

    if not langfuse_cfg.enabled:
        logger.info("Langfuse tracing is disabled in configuration.")
        return None

    try:
        try:
            from langfuse.langchain import CallbackHandler
        except ImportError:
            from langfuse.callback import CallbackHandler

        handler = CallbackHandler(
            public_key=langfuse_cfg.public_key,
            secret_key=langfuse_cfg.secret_key,
            host=langfuse_cfg.host,
        )
        logger.info(f"Langfuse callback handler initialized for host {langfuse_cfg.host}")
        return handler
    except Exception as exc:
        logger.warning(f"Failed to initialize Langfuse CallbackHandler: {exc}. Tracing disabled.")
        return None