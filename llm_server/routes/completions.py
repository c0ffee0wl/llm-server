"""Completions endpoint for code completion (inline suggestions)."""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

import llm

from ..adapters.model_adapters import get_adapter
from ..config import settings, executor, get_model_with_fallback, log_response_to_db
from ..streaming.sse import stream_llm_response, format_sse_message

logger = logging.getLogger(__name__)

router = APIRouter()


def generate_completion_id() -> str:
    """Generate a unique completion ID."""
    return f"cmpl-{uuid.uuid4().hex[:24]}"


async def stream_with_echo_suffix(
    base_stream,
    prompt: str,
    suffix: Optional[str],
    echo: bool,
    response_id: str,
    model_id: str,
    created_time: int,
):
    """
    Wrap a streaming response to add echo/suffix support.

    Args:
        base_stream: The base SSE stream from stream_llm_response
        prompt: The original prompt text
        suffix: Text to append after completion (if any)
        echo: Whether to include prompt in response
        response_id: Unique response ID
        model_id: Model identifier
        created_time: Unix timestamp for the response
    """
    # Echo: send prompt as first chunk
    if echo:
        yield format_sse_message({
            "id": response_id,
            "object": "text_completion",
            "created": created_time,
            "model": model_id,
            "choices": [{
                "index": 0,
                "text": prompt,
                "logprobs": None,
                "finish_reason": None,
            }],
        })

    # Stream original response, buffering one chunk to inject suffix before finish_reason
    # We need to detect the final message (with finish_reason) and inject suffix before it
    pending_chunk = None
    async for chunk in base_stream:
        # Check if this chunk contains finish_reason (final content chunk)
        is_final_content = 'finish_reason' in chunk and '"stop"' in chunk

        if pending_chunk is not None:
            yield pending_chunk

        if is_final_content and suffix:
            # Inject suffix before the final message
            yield format_sse_message({
                "id": response_id,
                "object": "text_completion",
                "created": created_time,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "text": suffix,
                    "logprobs": None,
                    "finish_reason": None,
                }],
            })
            pending_chunk = None
            yield chunk
        elif chunk.strip() == "data: [DONE]":
            # Yield [DONE] directly
            pending_chunk = None
            yield chunk
        else:
            pending_chunk = chunk

    # Yield any remaining pending chunk
    if pending_chunk is not None:
        yield pending_chunk


class CompletionRequest(BaseModel):
    """Request body for completions (legacy format).

    Supported parameters:
    - suffix: Text to append after the completion
    - echo: If true, includes the prompt in the response

    Note: The following parameters are parsed for OpenAI API compatibility
    but are not currently implemented:
    - n: Number of completions (always returns 1)
    - stop: Stop sequences (not passed to llm library)
    """

    model: str = "gpt-4o-mini"
    prompt: Any  # Can be string or list of strings
    suffix: Optional[str] = None  # Text to append after the completion
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1  # Not implemented - always returns 1 completion
    stream: bool = False
    stop: Optional[Any] = None  # Not implemented - parsed for API compatibility
    echo: bool = False  # Include prompt in response

    class Config:
        extra = "allow"


@router.post("/v1/completions")
@router.post("/v1/engines/{engine_id}/completions")
async def create_completion(request: CompletionRequest, engine_id: Optional[str] = None):
    """
    Create a completion for code/text.

    This endpoint handles both:
    - /v1/completions (standard)
    - /v1/engines/{engine_id}/completions (legacy)
    """
    if settings.debug:
        logger.debug(f"=== Completion request ===")
        logger.debug(f"Engine: {engine_id}, Model: {request.model}")
        logger.debug(f"Stream: {request.stream}")

    response_id = generate_completion_id()

    # Get the model using shared fallback chain
    try:
        requested = engine_id or request.model
        model, actual_model_name, was_fallback = get_model_with_fallback(requested, settings.debug)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "invalid_request_error", "code": "model_not_found"}}
        )

    # Build base headers for responses
    base_headers = {}
    if was_fallback:
        base_headers["X-Model-Fallback"] = "true"
        base_headers["X-Requested-Model"] = requested

    # Extract prompt
    if isinstance(request.prompt, list):
        prompt = "\n".join(str(p) for p in request.prompt)
    else:
        prompt = str(request.prompt) if request.prompt else ""

    # Validate prompt is not empty
    if not prompt or not prompt.strip():
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Prompt cannot be empty", "type": "invalid_request_error", "code": "empty_prompt"}}
        )

    if settings.debug:
        logger.debug(f"Prompt length: {len(prompt)}")

    # Get adapter for model-specific behavior
    adapter = get_adapter(actual_model_name)

    # Build options
    options: Dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.max_tokens is not None:
        options["max_tokens"] = request.max_tokens

    # Filter options through adapter (removes unsupported options like max_tokens for Gemini)
    options = adapter.prepare_options(options)

    try:
        # Make the request
        response = model.prompt(
            prompt=prompt,
            stream=request.stream,
            **options,
        )

        if request.stream:
            # Streaming response using shared SSE utilities
            stream_headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Request-ID": response_id,
                **base_headers,
            }

            base_stream = stream_llm_response(
                response=response,
                executor=executor,
                model_id=actual_model_name,
                response_id=response_id,
                response_type="completion",
                debug=settings.debug,
                on_complete=log_response_to_db,
            )

            # Wrap stream if echo or suffix is requested
            if request.echo or request.suffix:
                created_time = int(time.time())
                stream = stream_with_echo_suffix(
                    base_stream=base_stream,
                    prompt=prompt,
                    suffix=request.suffix,
                    echo=request.echo,
                    response_id=response_id,
                    model_id=actual_model_name,
                    created_time=created_time,
                )
            else:
                stream = base_stream

            return StreamingResponse(
                stream,
                media_type="text/event-stream",
                headers=stream_headers,
            )
        else:
            # Non-streaming response - run blocking call in executor with timeout
            loop = asyncio.get_running_loop()
            timeout = settings.request_timeout
            try:
                full_text = await asyncio.wait_for(
                    loop.run_in_executor(executor, lambda: response.text()),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=504,
                    content={"error": {"message": f"Request timed out after {timeout} seconds", "type": "timeout_error", "code": "timeout"}}
                )

            # Apply echo: prepend prompt if requested
            if request.echo:
                full_text = prompt + full_text

            # Apply suffix: append if provided
            if request.suffix:
                full_text = full_text + request.suffix

            # Log response to database
            log_response_to_db(response)

            return JSONResponse(
                content={
                    "id": response_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": actual_model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": full_text,
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": response.input_tokens or 0,
                        "completion_tokens": response.output_tokens or 0,
                        "total_tokens": (response.input_tokens or 0) + (response.output_tokens or 0),
                    },
                },
                headers=base_headers if base_headers else None,
            )

    except ValueError as e:
        # Client errors (invalid input)
        if settings.debug:
            logger.debug(f"Completion validation error: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "invalid_request_error", "code": "invalid_request"}}
        )
    except Exception as e:
        # Server errors
        if settings.debug:
            logger.exception(f"Completion error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error", "code": "internal_error"}}
        )
