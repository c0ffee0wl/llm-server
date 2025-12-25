"""Model-specific adapters for handling differences between LLM providers."""

from typing import Any, Dict
from abc import ABC, abstractmethod

from ..config import is_gemini_model


class ModelAdapter(ABC):
    """Base adapter for model-specific behavior."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name for logging."""
        pass

    def prepare_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare options for the model (filter/translate unsupported options)."""
        return options


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI and OpenAI-compatible models."""

    @property
    def name(self) -> str:
        return "openai"


class GeminiAdapter(ModelAdapter):
    """Adapter for Google Gemini/Vertex AI models."""

    @property
    def name(self) -> str:
        return "gemini"

    def prepare_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Translate options for Gemini compatibility."""
        # Gemini uses max_output_tokens instead of max_tokens
        if "max_tokens" in options:
            options = dict(options)  # Don't mutate input
            options["max_output_tokens"] = options.pop("max_tokens")
        return options


class ClaudeAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models."""

    @property
    def name(self) -> str:
        return "claude"


class LocalModelAdapter(ModelAdapter):
    """Adapter for local models (llama.cpp, ollama, etc.)."""

    @property
    def name(self) -> str:
        return "local"


def get_adapter(model_name: str) -> ModelAdapter:
    """Get the appropriate adapter for a model."""
    if not model_name:
        return OpenAIAdapter()

    model_lower = model_name.lower()

    # Gemini/Vertex models - use centralized detection function
    if is_gemini_model(model_name):
        return GeminiAdapter()

    # Claude models
    if "claude" in model_lower:
        return ClaudeAdapter()

    # Local models - check for prefixes and exact matches
    local_prefixes = ["llama-", "llama2", "llama3", "mistral-", "mixtral", "ollama/", "local/"]
    local_exact = ["llama", "mistral"]
    if any(model_lower.startswith(prefix) for prefix in local_prefixes):
        return LocalModelAdapter()
    if model_lower in local_exact:
        return LocalModelAdapter()
    # Check for GGUF file extension (local quantized models)
    if model_lower.endswith(".gguf"):
        return LocalModelAdapter()

    # Default to OpenAI-compatible
    return OpenAIAdapter()
