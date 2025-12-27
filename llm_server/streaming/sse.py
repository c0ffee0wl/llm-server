"""Server-Sent Events utilities for streaming responses."""

import json
import logging
import time
from typing import Any, AsyncIterator, Dict, Optional

from llm import AsyncResponse


logger = logging.getLogger(__name__)

# Timeout for waiting on chunks (5 minutes to allow for slow models)
CHUNK_TIMEOUT = 300.0


def format_sse_message(data: Dict[str, Any]) -> str:
    """Format a dictionary as an SSE message."""
    return f"data: {json.dumps(data)}\n\n"


def format_sse_done() -> str:
    """Format the final SSE done message."""
    return "data: [DONE]\n\n"


async def stream_llm_response(
    response: AsyncResponse,
    model_id: str,
    response_id: str,
    response_type: str = "chat",
    debug: bool = False,
    on_complete: Optional[callable] = None,
) -> AsyncIterator[str]:
    """
    Stream an LLM response using Server-Sent Events.

    Uses native async iteration from the llm library's AsyncResponse,
    eliminating the need for thread pool bridging.

    Args:
        response: The llm library AsyncResponse object (async iterable)
        model_id: The model ID to include in response chunks
        response_id: Unique response ID for this stream
        response_type: Either "chat" for chat completions or "completion" for text completions
        debug: Whether to log debug information
        on_complete: Optional callback to run after streaming completes successfully.
                     Called with the response object. Used for logging to database.

    Yields:
        SSE formatted messages
    """
    created_time = int(time.time())
    if debug:
        logger.debug(f"Starting SSE generator for {response_type}")

    try:
        first_chunk = True

        # Native async iteration - no thread pool bridging needed
        async for text_chunk in response:
            if debug and text_chunk:
                preview = text_chunk[:50] if len(text_chunk) > 50 else text_chunk
                logger.debug(f"Got chunk: {preview}...")

            if response_type == "chat":
                # Chat completion chunk format
                delta_content: Dict[str, Any] = {}
                if text_chunk:
                    delta_content["content"] = text_chunk
                if first_chunk:
                    delta_content["role"] = "assistant"
                    first_chunk = False

                # Only yield if we have something to send
                if delta_content:
                    chunk_msg = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta_content,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield format_sse_message(chunk_msg)
            else:
                # Text completion chunk format - only yield non-empty chunks
                if text_chunk:
                    chunk_msg = {
                        "id": response_id,
                        "object": "text_completion",
                        "created": created_time,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "text": text_chunk,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield format_sse_message(chunk_msg)

        # Handle tool calls at the end (chat completions only)
        tool_calls = None
        if response_type == "chat":
            try:
                tool_calls = await response.tool_calls()
            except AttributeError:
                pass
            except Exception as e:
                if debug:
                    logger.debug(f"Error getting tool calls: {e}")

        if response_type == "chat" and tool_calls:
            from ..adapters.tool_adapter import format_streaming_tool_call_delta

            tool_delta = format_streaming_tool_call_delta(tool_calls, 0)
            if tool_delta:
                tool_msg = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": tool_delta},
                            "finish_reason": None,
                        }
                    ],
                }
                yield format_sse_message(tool_msg)

        # Final message with finish_reason
        if response_type == "chat":
            finish_reason = "tool_calls" if tool_calls else "stop"
            final_msg = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
        else:
            final_msg = {
                "id": response_id,
                "object": "text_completion",
                "created": created_time,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
            }

        yield format_sse_message(final_msg)
        yield format_sse_done()

        # Call completion callback
        if on_complete is not None:
            try:
                on_complete(response)
            except Exception as e:
                if debug:
                    logger.debug(f"Error in on_complete callback: {e}")

        if debug:
            logger.debug("SSE generator completed successfully")

    except Exception as e:
        if debug:
            logger.exception(f"Error in SSE generator: {e}")
        error_msg = {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }
        yield format_sse_message(error_msg)
        yield format_sse_done()
