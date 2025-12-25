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

from ..config import settings, is_gemini_model, executor, get_model_with_fallback
from ..streaming.sse import stream_llm_response

logger = logging.getLogger(__name__)

router = APIRouter()


def generate_completion_id() -> str:
    """Generate a unique completion ID."""
    return f"cmpl-{uuid.uuid4().hex[:24]}"


class CompletionRequest(BaseModel):
    """Request body for completions (legacy format).

    Note: The following parameters are parsed for OpenAI API compatibility
    but are not currently implemented:
    - suffix: Text to append after completion (not supported by llm library)
    - n: Number of completions (always returns 1)
    - stop: Stop sequences (not passed to llm library)
    - echo: Include prompt in response (not implemented)
    """

    model: str = "gpt-4o-mini"
    prompt: Any  # Can be string or list of strings
    suffix: Optional[str] = None  # Not implemented - parsed for API compatibility
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1  # Not implemented - always returns 1 completion
    stream: bool = False
    stop: Optional[Any] = None  # Not implemented - parsed for API compatibility
    echo: bool = False  # Not implemented - parsed for API compatibility

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
        model, actual_model_name = get_model_with_fallback(requested, settings.debug)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "invalid_request_error"}}
        )

    # Extract prompt
    if isinstance(request.prompt, list):
        prompt = "\n".join(str(p) for p in request.prompt)
    else:
        prompt = str(request.prompt) if request.prompt else ""

    # Validate prompt is not empty
    if not prompt or not prompt.strip():
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Prompt cannot be empty", "type": "invalid_request_error"}}
        )

    if settings.debug:
        logger.debug(f"Prompt length: {len(prompt)}")

    # Build options
    options: Dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.top_p is not None:
        options["top_p"] = request.top_p

    # max_tokens handling (skip for Gemini)
    is_gemini = is_gemini_model(actual_model_name)
    if request.max_tokens is not None and not is_gemini:
        options["max_tokens"] = request.max_tokens

    try:
        # Make the request
        response = model.prompt(
            prompt=prompt,
            stream=request.stream,
            **options,
        )

        if request.stream:
            # Streaming response using shared SSE utilities
            return StreamingResponse(
                stream_llm_response(
                    response=response,
                    executor=executor,
                    model_id=actual_model_name,
                    response_id=response_id,
                    response_type="completion",
                    debug=settings.debug,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Request-ID": response_id,
                },
            )
        else:
            # Non-streaming response - run blocking call in executor with timeout
            loop = asyncio.get_running_loop()
            try:
                full_text = await asyncio.wait_for(
                    loop.run_in_executor(executor, lambda: response.text()),
                    timeout=300.0
                )
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=504,
                    content={"error": {"message": "Request timed out after 300 seconds", "type": "timeout_error"}}
                )

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
                }
            )

    except Exception as e:
        if settings.debug:
            logger.exception(f"Completion error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}}
        )
