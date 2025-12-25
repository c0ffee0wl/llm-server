"""Configuration for the LLM server."""

import atexit
import threading
from concurrent.futures import ThreadPoolExecutor
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Server configuration settings."""

    host: str = "127.0.0.1"
    port: int = 8777
    model_name: Optional[str] = None  # None = use llm library default (gpt-4o-mini)
    debug: bool = False
    pidfile: Optional[str] = None
    logfile: Optional[str] = None
    max_workers: int = 10  # Thread pool size for concurrent LLM operations

    class Config:
        env_prefix = "LLM_SERVER_"


settings = Settings()

# Lazy initialization for executor to respect runtime settings changes
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()


def get_executor() -> ThreadPoolExecutor:
    """Get or create the shared thread pool executor."""
    global _executor
    # Double-checked locking pattern for thread-safe lazy initialization
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(max_workers=settings.max_workers)
                # Register shutdown on process exit
                atexit.register(shutdown_executor)
    return _executor


def shutdown_executor():
    """Shutdown the executor gracefully."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None


# Backwards compatibility: executor property that uses lazy initialization
class _ExecutorProxy:
    """Proxy object that lazily initializes the executor on first access."""

    def submit(self, *args, **kwargs):
        return get_executor().submit(*args, **kwargs)

    def map(self, *args, **kwargs):
        return get_executor().map(*args, **kwargs)

    def shutdown(self, *args, **kwargs):
        return get_executor().shutdown(*args, **kwargs)


executor = _ExecutorProxy()


def is_gemini_model(model_name: str) -> bool:
    """Check if the model is a Gemini/Vertex model."""
    if not model_name:
        return False
    return model_name.startswith("gemini/") or model_name.startswith("vertex/")


# Model names to ignore when falling back (placeholder names from clients)
IGNORED_MODEL_NAMES = frozenset({"local-llm", "gpt-41-copilot"})


def get_model_with_fallback(
    requested_model: Optional[str] = None,
    debug: bool = False,
) -> tuple:
    """
    Get a model with fallback chain. Returns (model, model_name).

    Fallback order:
    1. llm library's configured default (via `llm models default`)
    2. Requested model name (if not in IGNORED_MODEL_NAMES)
    3. Settings model_name
    4. First available model

    Raises:
        ValueError: If no model is available
    """
    import logging
    import llm

    logger = logging.getLogger(__name__)

    # 1. Try llm library's default model first
    try:
        model = llm.get_model()
        if debug:
            logger.debug(f"Using llm default model: {model.model_id}")
        return model, model.model_id
    except Exception as e:
        if debug:
            logger.debug(f"No default model configured: {e}")

    # 2. Try requested model
    if requested_model and requested_model not in IGNORED_MODEL_NAMES:
        try:
            model = llm.get_model(requested_model)
            if debug:
                logger.debug(f"Using requested model: {requested_model}")
            return model, requested_model
        except Exception as e:
            if debug:
                logger.debug(f"Model '{requested_model}' not found: {e}")

    # 3. Try settings model
    if settings.model_name:
        try:
            model = llm.get_model(settings.model_name)
            if debug:
                logger.debug(f"Using settings model: {settings.model_name}")
            return model, settings.model_name
        except Exception as e:
            if debug:
                logger.debug(f"Settings model '{settings.model_name}' not found: {e}")

    # 4. First available model
    try:
        available = list(llm.get_models())
        if available:
            model = available[0]
            if debug:
                logger.debug(f"Using first available model: {model.model_id}")
            return model, model.model_id
    except Exception as e:
        logger.warning(f"Failed to enumerate models: {e}")

    raise ValueError("No LLM models available. Configure with `llm models default <model>`.")
