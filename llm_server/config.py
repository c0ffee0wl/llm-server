"""Configuration for the LLM server."""

import atexit
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pydantic_settings import BaseSettings
from typing import Optional

import llm
import sqlite_utils
from llm.migrations import migrate


class Settings(BaseSettings):
    """Server configuration settings."""

    host: str = "127.0.0.1"
    port: int = 8777
    model_name: Optional[str] = None  # None = use llm library default (gpt-4o-mini)
    debug: bool = False
    pidfile: Optional[str] = None
    logfile: Optional[str] = None
    no_log: bool = False  # Disable database logging of requests/responses
    max_workers: int = 10  # Thread pool size for concurrent LLM operations
    request_timeout: float = 300.0  # Timeout in seconds for LLM requests
    cleanup_timeout: float = 5.0  # Seconds to wait for streaming thread cleanup

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


def log_response_to_db(response, messages: list[dict] = None):
    """Log an LLM response to the database with conversation tracking.

    Uses llm.user_dir() to find the config directory and logs to log-server.db.
    Calls migrate() and response.log_to_db() from the llm library.

    If messages are provided, tracks the conversation so future requests
    with the same message prefix will be grouped together.

    Respects the no_log setting. Errors are logged but don't propagate.
    """
    if settings.no_log:
        return
    import logging
    from llm.models import Conversation
    logger = logging.getLogger(__name__)
    try:
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
                response_text = response.text()
            except Exception:
                response_text = ""
            conversation_tracker.store_conversation(messages, response_text, conv_id)

    except Exception as e:
        logger.warning(f"Failed to log response to database: {e}")
