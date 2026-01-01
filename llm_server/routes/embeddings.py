"""Embeddings endpoint for generating text embeddings."""

import asyncio
import base64
import logging
import struct
from typing import List, Optional, Union

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from ..config import settings, get_embedding_model_with_fallback

logger = logging.getLogger(__name__)

router = APIRouter()


class EmbeddingRequest(BaseModel):
    """Request body for embeddings - matches OpenAI format."""

    model: str = "text-embedding-ada-002"
    input: Union[str, List[str]]  # Single string or array of strings
    encoding_format: Optional[str] = "float"  # "float" or "base64"
    dimensions: Optional[int] = None
    user: Optional[str] = None  # Optional user identifier

    class Config:
        extra = "allow"  # Allow unknown fields for compatibility

    @field_validator('input')
    @classmethod
    def validate_input(cls, v):
        """Validate input is string or list of strings (not token arrays)."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("input cannot be empty")
            return v
        elif isinstance(v, list):
            if not v:
                raise ValueError("input list cannot be empty")
            for i, item in enumerate(v):
                if isinstance(item, int):
                    raise ValueError(
                        "Token arrays (list of integers) are not supported. "
                        "Please provide text strings instead."
                    )
                if isinstance(item, list):
                    raise ValueError(
                        "Nested token arrays are not supported. "
                        "Please provide text strings instead."
                    )
                if not isinstance(item, str):
                    raise ValueError(f"input[{i}] must be a string, got {type(item).__name__}")
                if not item.strip():
                    raise ValueError(f"input[{i}] cannot be empty")
            return v
        else:
            raise ValueError(f"input must be a string or list of strings, got {type(v).__name__}")


def _encode_embedding_base64(embedding: List[float]) -> str:
    """Encode embedding as base64 string (little-endian floats)."""
    packed = struct.pack("<" + "f" * len(embedding), *embedding)
    return base64.b64encode(packed).decode("ascii")


async def _create_embeddings_impl(request: EmbeddingRequest):
    """
    Shared implementation for embeddings.
    Both /v1/embeddings and /v1c/embeddings respect client's model choice.
    """
    if settings.debug:
        logger.debug("=== Embeddings request ===")
        logger.debug(f"Model: {request.model}")
        input_count = 1 if isinstance(request.input, str) else len(request.input)
        logger.debug(f"Input count: {input_count}")
        logger.debug(f"Encoding format: {request.encoding_format}")

    # Get the embedding model
    try:
        model, model_name, was_fallback = get_embedding_model_with_fallback(
            request.model, settings.debug
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }
        )

    # Build headers
    headers = {}
    if was_fallback:
        headers["X-Model-Fallback"] = "true"
        headers["X-Requested-Model"] = request.model

    # Log warning if dimensions requested but model has fixed dimensions
    if request.dimensions is not None:
        model_dimensions = getattr(model, 'dimensions', None)
        if model_dimensions is not None and model_dimensions != request.dimensions:
            logger.warning(
                f"Requested dimensions={request.dimensions} but model '{model_name}' "
                f"has fixed dimensions={model_dimensions}. Using model's dimensions."
            )

    # Normalize input to list
    inputs: List[str] = [request.input] if isinstance(request.input, str) else list(request.input)

    try:
        # Run synchronous embedding in thread pool to avoid blocking
        timeout = settings.request_timeout
        if len(inputs) == 1:
            embedding = await asyncio.wait_for(
                asyncio.to_thread(model.embed, inputs[0]),
                timeout=timeout
            )
            embeddings = [embedding]
        else:
            embeddings = await asyncio.wait_for(
                asyncio.to_thread(lambda: list(model.embed_multi(inputs))),
                timeout=timeout
            )

        # Build response in OpenAI format
        data = []
        total_tokens = 0
        use_base64 = request.encoding_format == "base64"

        for i, (text, embedding) in enumerate(zip(inputs, embeddings)):
            if use_base64:
                embedding_value = _encode_embedding_base64(embedding)
            else:
                embedding_value = embedding

            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding_value,
            })
            # Rough token estimate: ~4 chars per token
            total_tokens += max(1, len(text) // 4)

        response_content = {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            }
        }

        if settings.debug:
            logger.debug(f"Generated {len(data)} embeddings, total_tokens={total_tokens}")

        return JSONResponse(
            content=response_content,
            headers=headers if headers else None,
        )

    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": f"Request timed out after {timeout} seconds",
                    "type": "timeout_error",
                    "code": "timeout"
                }
            }
        )
    except Exception as e:
        if settings.debug:
            logger.exception(f"Embedding error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "internal_error"
                }
            }
        )


@router.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings for the input text(s).

    Both /v1 and /v1c respect the client's model choice for embeddings,
    as embedding models are typically task-specific.
    """
    return await _create_embeddings_impl(request)


@router.post("/v1c/embeddings")
async def create_embeddings_v1c(request: EmbeddingRequest):
    """
    Create embeddings for the input text(s) (client model choice).

    Same behavior as /v1/embeddings - both respect client's model choice.
    """
    return await _create_embeddings_impl(request)
