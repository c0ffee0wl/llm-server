"""Completions endpoint for code completion (inline suggestions)."""

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

import llm

from ..config import settings, is_gemini_model

# Thread pool for blocking LLM operations
_executor = ThreadPoolExecutor(max_workers=4)

logger = logging.getLogger(__name__)

router = APIRouter()


def generate_completion_id() -> str:
    """Generate a unique completion ID."""
    return f"cmpl-{uuid.uuid4().hex[:24]}"


class CompletionRequest(BaseModel):
    """Request body for completions (legacy format)."""

    model: str = "gpt-4o-mini"
    prompt: Any  # Can be string or list of strings
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    stream: bool = False
    stop: Optional[Any] = None  # String or list of strings
    echo: bool = False

    class Config:
        extra = "allow"


def get_model(model_name: str) -> tuple[llm.Model, str]:
    """Get a model by name with fallbacks. Returns (model, model_name)."""
    model = None

    # 1. Try llm library's default model first
    try:
        model = llm.get_model()
        actual_name = model.model_id
        if settings.debug:
            logger.debug(f"Using llm default model: {actual_name}")
        return model, actual_name
    except Exception as e:
        if settings.debug:
            logger.debug(f"No default model configured: {e}")

    # 2. Try the requested model
    if model_name and model_name not in ("gpt-41-copilot", "local-llm"):
        try:
            model = llm.get_model(model_name)
            if settings.debug:
                logger.debug(f"Using requested model: {model_name}")
            return model, model_name
        except Exception as e:
            if settings.debug:
                logger.debug(f"Model '{model_name}' not found: {e}")

    # 3. Try settings model
    if settings.model_name:
        try:
            model = llm.get_model(settings.model_name)
            if settings.debug:
                logger.debug(f"Using settings model: {settings.model_name}")
            return model, settings.model_name
        except Exception as e:
            if settings.debug:
                logger.debug(f"Settings model '{settings.model_name}' not found: {e}")

    # 4. First available
    try:
        available = list(llm.get_models())
        if available:
            model = available[0]
            return model, model.model_id
    except Exception as e:
        logger.warning(f"Failed to enumerate models: {e}")

    raise ValueError("No LLM models available")


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

    # Get the model
    try:
        model_name = engine_id or request.model
        model, actual_model_name = get_model(model_name)
    except ValueError as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}}
        )

    # Extract prompt
    if isinstance(request.prompt, list):
        prompt = "\n".join(str(p) for p in request.prompt)
    else:
        prompt = str(request.prompt) if request.prompt else ""

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
            # Streaming response
            _model_name = actual_model_name
            _response_id = response_id
            _response = response
            _debug = settings.debug

            async def sse_generator():
                created_time = int(time.time())
                if _debug:
                    logger.debug("Starting completion SSE generator")

                # Use queue to bridge sync iteration to async
                queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_event_loop()

                def iterate_sync():
                    """Run sync iteration in thread, put chunks in queue."""
                    try:
                        for chunk in _response:
                            loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
                        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                    except Exception as e:
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", e))

                # Start sync iteration in thread pool
                _executor.submit(iterate_sync)

                try:
                    while True:
                        msg_type, data = await queue.get()

                        if msg_type == "error":
                            raise data

                        if msg_type == "done":
                            break

                        # msg_type == "chunk"
                        text_chunk = data
                        if text_chunk:
                            chunk_msg = {
                                "id": _response_id,
                                "object": "text_completion",
                                "created": created_time,
                                "model": _model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "text": text_chunk,
                                        "logprobs": None,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk_msg)}\n\n"

                    # Final message
                    final_msg = {
                        "id": _response_id,
                        "object": "text_completion",
                        "created": created_time,
                        "model": _model_name,
                        "choices": [
                            {
                                "index": 0,
                                "text": "",
                                "logprobs": None,
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_msg)}\n\n"
                    yield "data: [DONE]\n\n"

                    if _debug:
                        logger.debug("Completion SSE generator completed")

                except Exception as e:
                    if _debug:
                        logger.exception(f"Error in completion SSE: {e}")
                    error_msg = {"error": {"message": str(e), "type": "server_error"}}
                    yield f"data: {json.dumps(error_msg)}\n\n"

            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Request-ID": response_id,
                },
            )
        else:
            # Non-streaming response - run blocking call in executor
            loop = asyncio.get_event_loop()
            full_text = await loop.run_in_executor(_executor, response.text)

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
