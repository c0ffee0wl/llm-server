"""Adapter for tool/function calling between OpenAI and llm library."""

import json
import time
import uuid
from typing import Any, Dict, List, Optional


def _generate_tool_call_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:24]}"


def _serialize_arguments(args: Any) -> str:
    """Serialize tool arguments to JSON string."""
    if args is None:
        return "{}"
    if isinstance(args, str):
        # Already a string, validate it's JSON
        try:
            json.loads(args)
            return args
        except json.JSONDecodeError:
            return json.dumps({"value": args})
    if isinstance(args, dict):
        return json.dumps(args)
    return json.dumps(args)


def format_tool_call_response(
    tool_calls: List[Any],
    model_id: str,
    response_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Format tool calls into OpenAI-compatible response format.

    This is used for non-streaming responses with tool calls.
    """
    if response_id is None:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    formatted_calls = []
    for tc in tool_calls:
        args = tc.arguments if hasattr(tc, "arguments") else {}
        args_str = _serialize_arguments(args)

        formatted_calls.append(
            {
                "id": tc.tool_call_id if hasattr(tc, "tool_call_id") and tc.tool_call_id else _generate_tool_call_id(),
                "type": "function",
                "function": {
                    "name": tc.name if hasattr(tc, "name") else "unknown",
                    "arguments": args_str,
                },
            }
        )

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": formatted_calls,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def format_streaming_tool_call_delta(
    tool_calls: List[Any],
    chunk_index: int,
) -> Optional[List[Dict[str, Any]]]:
    """
    Format tool calls for streaming SSE delta format.

    Returns the delta object for the tool_calls field.
    """
    if not tool_calls:
        return None

    formatted_calls = []
    for i, tc in enumerate(tool_calls):
        args = tc.arguments if hasattr(tc, "arguments") else {}
        args_str = _serialize_arguments(args)

        call_delta = {
            "index": i,
            "id": tc.tool_call_id if hasattr(tc, "tool_call_id") and tc.tool_call_id else _generate_tool_call_id(),
            "type": "function",
            "function": {
                "name": tc.name if hasattr(tc, "name") else "unknown",
                "arguments": args_str,
            },
        }
        formatted_calls.append(call_delta)

    return formatted_calls if formatted_calls else None
