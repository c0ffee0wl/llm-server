"""Model-specific adapters for handling differences between LLM providers."""

from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    """Base adapter for model-specific behavior."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name for logging."""
        pass

    def supports_tools(self) -> bool:
        """Whether this model supports tool calling."""
        return True

    def supports_streaming(self) -> bool:
        """Whether this model supports streaming responses."""
        return True

    def prepare_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare options for the model (filter unsupported options)."""
        return options

    def format_tool_results(self, results: List[Any]) -> List[Any]:
        """Format tool results for the model."""
        return results

    def get_max_tokens_param(self) -> Optional[str]:
        """Get the parameter name for max tokens (varies by model)."""
        return "max_tokens"


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI and OpenAI-compatible models."""

    @property
    def name(self) -> str:
        return "openai"

    def supports_tools(self) -> bool:
        return True

    def get_max_tokens_param(self) -> Optional[str]:
        return "max_tokens"


class GeminiAdapter(ModelAdapter):
    """Adapter for Google Gemini/Vertex AI models."""

    @property
    def name(self) -> str:
        return "gemini"

    def supports_tools(self) -> bool:
        return True

    def prepare_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out options not supported by Gemini."""
        # Gemini uses max_output_tokens internally, not max_tokens
        filtered = {k: v for k, v in options.items() if k != "max_tokens"}
        return filtered

    def format_tool_results(self, results: List[Any]) -> List[Any]:
        """Ensure all tool results have non-empty names (Gemini requirement)."""
        for result in results:
            if hasattr(result, 'name') and not result.name:
                result.name = "function_result"
        return results

    def get_max_tokens_param(self) -> Optional[str]:
        # Gemini doesn't support max_tokens through the llm library
        return None


class ClaudeAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models."""

    @property
    def name(self) -> str:
        return "claude"

    def supports_tools(self) -> bool:
        return True

    def get_max_tokens_param(self) -> Optional[str]:
        return "max_tokens"


class LocalModelAdapter(ModelAdapter):
    """Adapter for local models (llama.cpp, ollama, etc.)."""

    @property
    def name(self) -> str:
        return "local"

    def supports_tools(self) -> bool:
        # Many local models don't support tools
        return False

    def get_max_tokens_param(self) -> Optional[str]:
        return "max_tokens"


def get_adapter(model_name: str) -> ModelAdapter:
    """Get the appropriate adapter for a model."""
    if not model_name:
        return OpenAIAdapter()

    model_lower = model_name.lower()

    # Gemini/Vertex models
    if model_lower.startswith("gemini/") or model_lower.startswith("vertex/"):
        return GeminiAdapter()

    # Claude models
    if "claude" in model_lower:
        return ClaudeAdapter()

    # Local models (common patterns)
    local_patterns = ["llama", "mistral", "ollama/", "local/", "gguf"]
    if any(pattern in model_lower for pattern in local_patterns):
        return LocalModelAdapter()

    # Default to OpenAI-compatible
    return OpenAIAdapter()


def is_gemini_model(model_name: str) -> bool:
    """Check if the model is a Gemini/Vertex model."""
    if not model_name:
        return False
    return model_name.startswith("gemini/") or model_name.startswith("vertex/")
