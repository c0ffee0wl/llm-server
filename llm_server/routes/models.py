"""Models endpoint for listing available models."""

import logging
import time
from fastapi import APIRouter
from typing import Any, Dict, List, Optional, Tuple

import llm

logger = logging.getLogger(__name__)

router = APIRouter()

# Model list cache with TTL
_model_cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_CACHE_TTL = 3600.0  # Cache for 1 hour


def _get_model_capabilities(model: llm.Model) -> Dict[str, Any]:
    """Build capabilities dict for a model based on its properties."""
    supports_vision = bool(model.attachment_types)
    supports_tools = getattr(model, "supports_tools", False)
    can_stream = getattr(model, "can_stream", True)

    return {
        "type": "chat",
        "family": model.model_id.split("-")[0] if "-" in model.model_id else model.model_id,
        "tokenizer": "cl100k_base",
        "limits": {
            "max_prompt_tokens": 128000,
            "max_output_tokens": 16384,
            "max_context_window_tokens": 128000,
        },
        "supports": {
            "parallel_tool_calls": supports_tools,
            "tool_calls": supports_tools,
            "streaming": can_stream,
            "vision": supports_vision,
            "prediction": False,
            "thinking": False,
        },
    }


def _model_to_openai_format(model: llm.Model) -> Dict[str, Any]:
    """Convert an llm Model to OpenAI model format."""
    return {
        "id": model.model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "llm-library",
        "permission": [],
        "root": model.model_id,
        "parent": None,
        "capabilities": _get_model_capabilities(model),
    }


def get_model_list() -> List[Dict[str, Any]]:
    """Return the list of available models from the llm library (cached)."""
    global _model_cache

    # Check cache validity
    if _model_cache is not None:
        cache_time, cached_models = _model_cache
        if time.time() - cache_time < _CACHE_TTL:
            return cached_models

    try:
        models = llm.get_models()
        result = [_model_to_openai_format(m) for m in models]
        _model_cache = (time.time(), result)
        return result
    except Exception as e:
        logger.warning(f"Failed to enumerate models: {e}")
        # Fallback to a default model if llm fails
        return [
            {
                "id": "gpt-4o-mini",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llm-library",
                "permission": [],
                "root": "gpt-4o-mini",
                "parent": None,
                "capabilities": {
                    "type": "chat",
                    "family": "gpt-4",
                    "tokenizer": "cl100k_base",
                    "limits": {
                        "max_prompt_tokens": 128000,
                        "max_output_tokens": 16384,
                        "max_context_window_tokens": 128000,
                    },
                    "supports": {
                        "parallel_tool_calls": True,
                        "tool_calls": True,
                        "streaming": True,
                        "vision": True,
                        "prediction": False,
                        "thinking": False,
                    },
                },
            }
        ]


@router.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    """List available models from the llm library."""
    return {
        "object": "list",
        "data": get_model_list(),
    }


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> Dict[str, Any]:
    """Get a specific model by ID."""
    models = get_model_list()
    for model in models:
        if model["id"] == model_id:
            return model

    # Try to get the model directly from llm
    try:
        llm_model = llm.get_model(model_id)
        return _model_to_openai_format(llm_model)
    except Exception as e:
        logger.debug(f"Model '{model_id}' not found: {e}")

    # Return first available model as fallback
    if models:
        return models[0]

    # Ultimate fallback
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "llm-library",
    }
