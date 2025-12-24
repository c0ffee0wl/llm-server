"""Server-Sent Events utilities for streaming responses."""

import json
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
import asyncio


def format_sse_message(data: Dict[str, Any]) -> str:
    """Format a dictionary as an SSE message."""
    return f"data: {json.dumps(data)}\n\n"


def format_sse_done() -> str:
    """Format the final SSE done message."""
    return "data: [DONE]\n\n"


async def generate_streaming_response(
    response_iterator: Iterator[str],
    model_id: str,
    response_id: Optional[str] = None,
    tool_calls: Optional[List[Any]] = None,
) -> AsyncIterator[str]:
    """
    Generate SSE streaming response from llm library response iterator.

    Args:
        response_iterator: Iterator yielding text chunks from llm
        model_id: The model ID to include in response
        response_id: Unique response ID (UUID-based)
        tool_calls: Optional tool calls to include at the end

    Yields:
        SSE formatted messages
    """
    if response_id is None:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    chunk_index = 0

    # First chunk includes role
    first_chunk = True

    # Stream text content
    for text_chunk in response_iterator:
        if text_chunk:
            delta_content: Dict[str, Any] = {"content": text_chunk}
            if first_chunk:
                delta_content["role"] = "assistant"
                first_chunk = False

            delta: Dict[str, Any] = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta_content,
                        "finish_reason": None,
                    }
                ],
            }
            yield format_sse_message(delta)
            chunk_index += 1
            # Small delay to prevent overwhelming client
            await asyncio.sleep(0)

    # Handle tool calls at the end if present
    if tool_calls:
        from ..adapters.tool_adapter import format_streaming_tool_call_delta

        tool_delta = format_streaming_tool_call_delta(tool_calls, chunk_index)
        if tool_delta:
            delta = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": tool_delta,
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield format_sse_message(delta)
            chunk_index += 1

    # Final message with finish_reason
    finish_reason = "tool_calls" if tool_calls else "stop"
    final_delta = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }
    yield format_sse_message(final_delta)

    # Done marker
    yield format_sse_done()
