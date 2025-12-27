"""Configuration for the LLM server."""

import hashlib
import json
import threading
from pydantic_settings import BaseSettings
from typing import Optional

import llm
import sqlite_utils
from llm.migrations import migrate

# Model-specific context limits (input tokens)
# Based on provider documentation as of 2025-12
MODEL_CONTEXT_LIMITS = {
    # Azure OpenAI / OpenAI - GPT-4.1 series (1M context)
    "gpt-4.1": 1000000,
    "gpt-4.1-mini": 1000000,
    "gpt-4.1-nano": 1000000,

    # Azure OpenAI / OpenAI - GPT-4o series (128k context)
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,

    # Azure OpenAI / OpenAI - GPT-5 series (272k context)
    "gpt-5": 270000,
    "gpt-5-mini": 270000,
    "gpt-5-nano": 270000,
    "gpt-5-chat": 110000,
    "gpt-5.1": 270000,
    "gpt-5.1-chat": 110000,
    "gpt-5.1-codex": 270000,
    "gpt-5.1-codex-mini": 270000,
    "gpt-5.1-codex-max": 270000,
    "gpt-5.2": 270000,
    "gpt-5.2-chat": 110000,

    # Azure OpenAI / OpenAI - Reasoning models (o-series)
    "o1": 200000,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o3": 200000,
    "o3-mini": 200000,
    "o3-pro": 200000,
    "o4-mini": 200000,
    "codex-mini": 200000,
}

# Default limits by provider prefix (fallback when model not in explicit list)
# Gemini/Vertex models have 1M, Claude models have 200k
PROVIDER_DEFAULT_LIMITS = {
    "azure/": 200000,       # Conservative default for unknown Azure models
    "vertex/": 1000000,     # Vertex models have 1M
    "gemini-": 1000000,     # Gemini models have 1M
    "claude-": 200000,      # Claude models have 200k (1M beta requires special header)
    "openai/": 128000,      # Conservative for unknown OpenAI models
}

# Absolute fallback
DEFAULT_CONTEXT_LIMIT = 200000


class Settings(BaseSettings):
    """Server configuration settings."""

    host: str = "127.0.0.1"
    port: int = 8777
    model_name: Optional[str] = None  # None = use llm library default (gpt-4o-mini)
    debug: bool = False
    pidfile: Optional[str] = None
    logfile: Optional[str] = None
    no_log: bool = False  # Disable database logging of requests/responses
    request_timeout: float = 300.0  # Timeout in seconds for LLM requests

    class Config:
        env_prefix = "LLM_SERVER_"


settings = Settings()


class ConversationTracker:
    """Track conversations by hashing message history.

    This allows detecting when a new request continues a previous conversation,
    even though the OpenAI API is stateless.

    After each response, we store: hash(all_messages) -> conversation_id
    On new request, we compute: hash(messages_except_last) and look it up.
    """

    # Role normalization: numeric roles to string
    ROLE_MAP = {0: "system", 1: "user", 2: "assistant", 3: "tool"}
    MAX_CACHE_SIZE = 10000  # Limit memory usage

    def __init__(self):
        self._lock = threading.Lock()
        self._hash_to_conv_id: dict[str, str] = {}

    def _normalize_role(self, role) -> str:
        """Normalize role to string format."""
        if isinstance(role, bool):
            # bool is subclass of int, handle separately
            return "user"
        if isinstance(role, int):
            return self.ROLE_MAP.get(role, "user")
        return str(role) if role else "user"

    def _normalize_content(self, content) -> str:
        """Extract text content from various formats."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Multimodal content - just use text parts for hashing
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            return "\n".join(text_parts)
        return str(content)

    def _hash_messages(self, messages: list[dict]) -> Optional[str]:
        """Create a stable hash of a message sequence.

        Only considers user and assistant messages (ignores system, tool).
        This provides stable conversation tracking even with tool-using conversations.

        Returns None if there are no user/assistant messages to hash.
        """
        normalized = []
        for msg in messages:
            role = self._normalize_role(msg.get("role"))

            # Only hash user and assistant messages
            if role not in ("user", "assistant"):
                continue

            content = self._normalize_content(msg.get("content"))
            normalized.append({"role": role, "content": content})

        # No user/assistant messages - can't create meaningful hash
        if not normalized:
            return None

        # Create deterministic JSON and hash it
        json_str = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def find_conversation(self, messages: list[dict]) -> Optional[str]:
        """Look up conversation ID for a message sequence.

        Hashes all messages except the last (assumed to be the new user message)
        and looks up if we've seen this prefix before.

        Returns conversation_id if found, None otherwise.
        """
        if len(messages) <= 1:
            # First message in conversation, no prefix to look up
            return None

        prefix = messages[:-1]
        prefix_hash = self._hash_messages(prefix)

        # Empty hash means no user/assistant messages in prefix - can't match
        if prefix_hash is None:
            return None

        with self._lock:
            return self._hash_to_conv_id.get(prefix_hash)

    def store_conversation(self, messages: list[dict], response_text: str, conversation_id: str):
        """Store the conversation state after a response.

        Adds the assistant response to messages and stores the hash -> conversation_id
        mapping so future requests can find this conversation.
        """
        full_messages = list(messages) + [{"role": "assistant", "content": response_text}]
        full_hash = self._hash_messages(full_messages)

        # No meaningful hash (shouldn't happen since we just added assistant msg)
        if full_hash is None:
            return

        with self._lock:
            # Evict oldest entries if cache is too large
            if len(self._hash_to_conv_id) >= self.MAX_CACHE_SIZE:
                # Remove ~10% of entries (simple eviction, not LRU)
                keys_to_remove = list(self._hash_to_conv_id.keys())[:self.MAX_CACHE_SIZE // 10]
                for key in keys_to_remove:
                    del self._hash_to_conv_id[key]

            self._hash_to_conv_id[full_hash] = conversation_id


# Global conversation tracker instance
conversation_tracker = ConversationTracker()


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
    Ruft ein Modell mit Fallback-Kette ab. Gibt (Modell, Modellname, war_fallback) zurück.

    Fallback-Reihenfolge:
    1. Das über die `llm models default`-Konfiguration der llm-Bibliothek festgelegte Standardmodell
    2. Angefragter Modellname (falls nicht in IGNORED_MODEL_NAMES enthalten)
    3. Modellname aus den Einstellungen (`settings.model_name`)
    4. Erstes verfügbares Modell

    Gibt zurück:
        Ein Tupel aus (Modell, Modellname, war_fallback), wobei war_fallback True ist,
        wenn das zurückgegebene Modell vom angefragten Modell abweicht.

    Löst aus:
        ValueError: Wenn kein Modell verfügbar ist
    """
    import logging
    import llm

    logger = logging.getLogger(__name__)

    # 1. Try llm library's default model first
    try:
        model = llm.get_model()
        if debug:
            logger.debug(f"Using llm default model: {model.model_id}")
        # It's a fallback if the default differs from requested
        was_fallback = requested_model and requested_model not in IGNORED_MODEL_NAMES and model.model_id != requested_model
        return model, model.model_id, was_fallback
    except Exception as e:
        if debug:
            logger.debug(f"No default model configured: {e}")

    # 2. Try requested model
    if requested_model and requested_model not in IGNORED_MODEL_NAMES:
        try:
            model = llm.get_model(requested_model)
            if debug:
                logger.debug(f"Using requested model: {requested_model}")
            return model, requested_model, False  # Not a fallback - using requested model
        except Exception as e:
            if debug:
                logger.debug(f"Model '{requested_model}' not found: {e}")

    # 3. Try settings model
    if settings.model_name:
        try:
            model = llm.get_model(settings.model_name)
            if debug:
                logger.debug(f"Using settings model: {settings.model_name}")
            # It's a fallback if settings model differs from requested
            was_fallback = requested_model and requested_model not in IGNORED_MODEL_NAMES
            return model, settings.model_name, was_fallback
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
            # It's a fallback if first available differs from requested
            was_fallback = requested_model and requested_model not in IGNORED_MODEL_NAMES
            return model, model.model_id, was_fallback
    except Exception as e:
        logger.warning(f"Failed to enumerate models: {e}")

    raise ValueError("No LLM models available. Configure with `llm models default <model>`.")


def get_async_model_with_fallback(
    requested_model: Optional[str] = None,
    debug: bool = False,
) -> tuple:
    """
    Get an async model with fallback chain. Returns (model, model_name, was_fallback).

    Fallback order:
    1. llm library's default model (async version)
    2. Requested model name (if not in IGNORED_MODEL_NAMES)
    3. Settings model name
    4. First available async model

    Returns:
        Tuple of (AsyncModel, model_name, was_fallback) where was_fallback is True
        if the returned model differs from the requested model.

    Raises:
        ValueError: If no async model is available
    """
    import logging

    logger = logging.getLogger(__name__)

    # 1. Try llm library's default model first (async version)
    try:
        model = llm.get_async_model()
        if debug:
            logger.debug(f"Using llm default async model: {model.model_id}")
        was_fallback = requested_model and requested_model not in IGNORED_MODEL_NAMES and model.model_id != requested_model
        return model, model.model_id, was_fallback
    except Exception as e:
        if debug:
            logger.debug(f"No default async model configured: {e}")

    # 2. Try requested model (async version)
    if requested_model and requested_model not in IGNORED_MODEL_NAMES:
        try:
            model = llm.get_async_model(requested_model)
            if debug:
                logger.debug(f"Using requested async model: {requested_model}")
            return model, requested_model, False
        except Exception as e:
            if debug:
                logger.debug(f"Async model '{requested_model}' not found: {e}")

    # 3. Try settings model (async version)
    if settings.model_name:
        try:
            model = llm.get_async_model(settings.model_name)
            if debug:
                logger.debug(f"Using settings async model: {settings.model_name}")
            was_fallback = requested_model and requested_model not in IGNORED_MODEL_NAMES
            return model, settings.model_name, was_fallback
        except Exception as e:
            if debug:
                logger.debug(f"Settings async model '{settings.model_name}' not found: {e}")

    # 4. First available async model
    try:
        available = list(llm.get_async_models())
        if available:
            model = available[0]
            if debug:
                logger.debug(f"Using first available async model: {model.model_id}")
            was_fallback = requested_model and requested_model not in IGNORED_MODEL_NAMES
            return model, model.model_id, was_fallback
    except Exception as e:
        logger.warning(f"Failed to enumerate async models: {e}")

    raise ValueError("No async LLM models available. Configure with `llm models default <model>`.")


def find_model_by_query(queries: list[str]):
    """
    Find a model matching the given query terms.

    Args:
        queries: List of search terms to match against model names/aliases

    Returns:
        Model instance or None if no match found
    """
    import llm

    for model in llm.get_models():
        model_id_lower = model.model_id.lower()
        # Check if all query terms match the model id
        if all(q.lower() in model_id_lower for q in queries):
            return model
    return None


def get_model_context_limit(model_name: str) -> int:
    """
    Get the context limit (max input tokens) for a model.

    Checks in order:
    1. Explicit MODEL_CONTEXT_LIMITS mapping
    2. PROVIDER_DEFAULT_LIMITS by prefix
    3. DEFAULT_CONTEXT_LIMIT fallback

    Args:
        model_name: The model identifier (e.g., "gpt-4o", "gemini-1.5-pro")

    Returns:
        Maximum input token count for the model
    """
    if not model_name:
        return DEFAULT_CONTEXT_LIMIT

    # Check explicit model mapping first
    if model_name in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model_name]

    # Check provider prefix defaults
    for prefix, limit in PROVIDER_DEFAULT_LIMITS.items():
        if model_name.startswith(prefix):
            return limit

    return DEFAULT_CONTEXT_LIMIT


def log_response_to_db(response, messages: list[dict] = None):
    """Log an LLM response to the database with conversation tracking.

    Uses llm.user_dir() to find the config directory and logs to log-server.db.
    Calls migrate() and response.log_to_db() from the llm library.

    If messages are provided, tracks the conversation so future requests
    with the same message prefix will be grouped together.

    Respects the no_log setting. Errors are logged but don't propagate.

    Note: For AsyncResponse objects, the response must be "done" (fully consumed
    via async iteration or await response.text()) before calling this function.
    This function will convert AsyncResponse to sync Response for logging.
    """
    if settings.no_log:
        return
    import logging
    from llm.models import Conversation, Response, AsyncResponse
    logger = logging.getLogger(__name__)
    try:
        # Convert AsyncResponse to sync Response for logging
        # AsyncResponse doesn't have log_to_db(), but after streaming completes
        # we can build a sync Response from its data
        if isinstance(response, AsyncResponse):
            if not response._done:
                logger.warning("Cannot log AsyncResponse that hasn't completed")
                return
            # Build a sync Response from the AsyncResponse data
            sync_response = Response(
                response.prompt,
                response.model,  # AsyncModel, but Response accepts it for logging
                response.stream,
                conversation=None,
            )
            sync_response.id = response.id
            sync_response._chunks = list(response._chunks)
            sync_response._done = response._done
            sync_response._end = response._end
            sync_response._start = response._start
            sync_response._start_utcnow = response._start_utcnow
            sync_response.input_tokens = response.input_tokens
            sync_response.output_tokens = response.output_tokens
            sync_response.token_details = response.token_details
            sync_response._prompt_json = response._prompt_json
            sync_response.response_json = response.response_json
            sync_response._tool_calls = list(response._tool_calls)
            sync_response.attachments = list(response.attachments)
            sync_response.resolved_model = response.resolved_model
            response = sync_response

        # Look up or create conversation ID based on message history
        conv_id = None
        if messages:
            conv_id = conversation_tracker.find_conversation(messages)

        if not conv_id:
            # New conversation - generate ID from first message hash
            import uuid
            conv_id = str(uuid.uuid4())

        # Set conversation on response before logging
        if not response.conversation:
            response.conversation = Conversation(model=response.model, id=conv_id)

        # Log to database
        db = sqlite_utils.Database(llm.user_dir() / "log-server.db")
        migrate(db)
        response.log_to_db(db)

        # Store conversation state for future lookups
        if messages:
            try:
                # Use text_or_raise() for sync access after response is done
                response_text = response.text_or_raise() if response._done else ""
            except Exception as e:
                logger.warning(f"Failed to get response text for conversation tracking: {e}")
                response_text = ""
            conversation_tracker.store_conversation(messages, response_text, conv_id)

    except Exception as e:
        logger.warning(f"Failed to log response to database: {e}")
