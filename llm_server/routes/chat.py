"""Chat completions endpoint - main LLM interaction point."""

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

import llm

# Thread pool for blocking LLM operations
_executor = ThreadPoolExecutor(max_workers=4)

from ..adapters.openai_adapter import (
    parse_conversation,
    convert_tool_definitions,
    extract_tool_results,
)
from ..adapters.tool_adapter import format_tool_call_response
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


def _is_gemini_model(model_name: str) -> bool:
    """Check if the model is a Gemini/Vertex model."""
    if not model_name:
        return False
    return model_name.startswith("gemini/") or model_name.startswith("vertex/")


def find_model_by_query(queries: List[str]) -> Optional[llm.Model]:
    """
    Find model matching all query strings (like llm -q).

    Multiple query terms are ANDed together.
    Returns the model with the shortest ID among matches.
    """
    if not queries:
        return None

    matches = []
    for model in llm.get_models():
        model_id = model.model_id.lower()
        if all(q.lower() in model_id for q in queries):
            matches.append(model)

    if not matches:
        return None

    # Return model with shortest ID (most specific match)
    return min(matches, key=lambda m: len(m.model_id))


def generate_response_id() -> str:
    """Generate a unique response ID using UUID."""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


class ChatMessage(BaseModel):
    """A chat message in OpenAI format."""

    role: Any  # Can be string or int (some clients send 0=system, 1=user, 2=assistant)
    content: Any  # Can be string or list for multimodal
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def get_role_str(self) -> str:
        """Convert role to string format."""
        if isinstance(self.role, int):
            role_map = {0: "system", 1: "user", 2: "assistant", 3: "tool"}
            return role_map.get(self.role, "user")
        return str(self.role)

    def get_content_str(self) -> str:
        """Extract text content from various formats."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            # Handle numeric format: [{"type": 1, "text": "..."}]
            # or standard format: [{"type": "text", "text": "..."}]
            texts = []
            for part in self.content:
                if isinstance(part, dict):
                    if "text" in part:
                        texts.append(part["text"])
                    elif "value" in part:
                        texts.append(part["value"])
                elif isinstance(part, str):
                    texts.append(part)
            return "\n".join(texts)
        return str(self.content) if self.content else ""


class FunctionDef(BaseModel):
    """Function definition for tool calling."""

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDef(BaseModel):
    """Tool definition in OpenAI format."""

    type: str = "function"
    function: FunctionDef


class ChatCompletionRequest(BaseModel):
    """Request body for chat completions."""

    model: str = "gpt-4o-mini"
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: bool = False
    tools: Optional[List[ToolDef]] = None
    tool_choice: Optional[Any] = None

    class Config:
        extra = "allow"


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion using the llm library.

    Supports both streaming and non-streaming responses.
    """
    import traceback

    # Debug logging (only when --debug flag is set)
    if settings.debug:
        logger.debug("=== Chat request ===")
        logger.debug(f"Model: {request.model}")
        logger.debug(f"Stream: {request.stream}")
        logger.debug(f"Messages count: {len(request.messages)}")
        for i, msg in enumerate(request.messages):
            logger.debug(f"  [{i}] role={msg.role}, content_type={type(msg.content).__name__}, content_len={len(str(msg.content))}")

    response_id = generate_response_id()

    # Get the model - respect llm library's default first
    model = None
    model_name = None

    # 1. Try llm library's default model first (respects `llm models default`)
    try:
        model = llm.get_model()  # Gets configured default from ~/.config/io.datasette.llm/default_model.txt
        model_name = model.model_id
        if settings.debug:
            logger.debug(f"Using llm default model: {model_name}")
    except Exception as e:
        if settings.debug:
            logger.debug(f"No default model configured: {e}")

    # 2. If no default, try settings or request model
    if model is None:
        try_model = settings.model_name or request.model
        if try_model and try_model != "local-llm":
            try:
                model = llm.get_model(try_model)
                model_name = try_model
                if settings.debug:
                    logger.debug(f"Using requested/settings model: {model_name}")
            except Exception as e:
                if settings.debug:
                    logger.debug(f"Model '{try_model}' not found: {e}")

    # 3. Last resort: try to get any available model
    if model is None:
        try:
            available = list(llm.get_models())
            if available:
                model = available[0]
                model_name = model.model_id
                if settings.debug:
                    logger.debug(f"Using first available model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to enumerate models: {e}")

    if model is None:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "No LLM models available. Please configure the llm library with `llm models default <model>`.", "type": "server_error"}}
        )

    # Parse messages into structured conversation data
    try:
        messages_raw = [m.model_dump() for m in request.messages]
        if settings.debug:
            logger.debug(f"Parsed {len(messages_raw)} raw messages")
        conv_data = parse_conversation(messages_raw)
        if settings.debug:
            logger.debug(f"Conv data: system={bool(conv_data.system_prompt)}, history={len(conv_data.messages)}, prompt={bool(conv_data.current_prompt)}")
    except Exception as e:
        if settings.debug:
            logger.exception(f"Error parsing messages: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"Failed to parse messages: {e}", "type": "parse_error"}}
        )

    # Convert tools to llm format (for future tool-using requests)
    tools_raw = [t.model_dump() for t in request.tools] if request.tools else None
    llm_tools = convert_tool_definitions(tools_raw)

    # Extract tool results from messages (works for all models now - names are guaranteed non-empty)
    is_gemini = _is_gemini_model(model_name)
    tool_results = extract_tool_results(messages_raw)
    if settings.debug and tool_results:
        logger.debug(f"Extracted {len(tool_results)} tool results")

    # Build options - only include ones supported by the model
    options: Dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    # max_tokens is not supported by Gemini models (they use max_output_tokens internally)
    if request.max_tokens is not None and not is_gemini:
        options["max_tokens"] = request.max_tokens
    if request.top_p is not None:
        options["top_p"] = request.top_p

    try:
        # Build the full prompt including conversation history
        full_prompt = ""

        # Add conversation history as context (including tool interactions)
        if conv_data.messages:
            history_parts = []
            for msg in conv_data.messages:
                if msg.role == "user" and msg.content:
                    history_parts.append(f"User: {msg.content}")
                elif msg.role == "assistant":
                    if msg.content:
                        history_parts.append(f"Assistant: {msg.content}")
                    # Include tool calls in history for context
                    if msg.tool_calls:
                        tool_names = [tc.get("function", {}).get("name", "unknown") for tc in msg.tool_calls]
                        history_parts.append(f"Assistant called tools: {', '.join(tool_names)}")
                elif msg.role == "tool" and msg.content:
                    # Include tool results in history
                    tool_name = msg.name or "tool"
                    # Truncate long tool results for context
                    content_preview = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                    history_parts.append(f"Tool result ({tool_name}): {content_preview}")
            if history_parts:
                full_prompt = "\n\n".join(history_parts) + "\n\n"

        # Add current prompt
        if conv_data.current_prompt:
            full_prompt += f"User: {conv_data.current_prompt}"

        if settings.debug:
            logger.debug(f"Full prompt length: {len(full_prompt)}")

        # Make the request
        try:
            if settings.debug:
                logger.debug(f"Calling model.prompt with model={model_name}")
            response = model.prompt(
                prompt=full_prompt,
                system=conv_data.system_prompt,
                attachments=conv_data.current_attachments if conv_data.current_attachments else None,
                tools=llm_tools if llm_tools else None,
                tool_results=tool_results if tool_results else None,
                stream=True,
                **options,
            )
            if settings.debug:
                logger.debug(f"Got response object: {type(response)}")
        except Exception as e:
            if settings.debug:
                logger.exception(f"Error calling model.prompt: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": f"LLM error: {e}", "type": "llm_error"}}
            )

        if request.stream:
            # Streaming response - yield chunks as they arrive
            # Capture variables for closure
            _model_name = model_name
            _response_id = response_id
            _response = response
            _debug = settings.debug

            async def sse_generator():
                created_time = int(time.time())
                if _debug:
                    logger.debug("Starting SSE generator")

                # Use queue to bridge sync iteration to async
                queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_event_loop()

                def iterate_sync():
                    """Run sync iteration in thread, put chunks in queue."""
                    try:
                        for chunk in _response:
                            loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
                        # Signal completion with tool_calls if any
                        tool_calls = None
                        try:
                            tool_calls = _response.tool_calls()
                        except AttributeError:
                            pass
                        except Exception as e:
                            if _debug:
                                logger.debug(f"Error getting tool calls: {e}")
                        loop.call_soon_threadsafe(queue.put_nowait, ("done", tool_calls))
                    except Exception as e:
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", e))

                # Start sync iteration in thread pool
                _executor.submit(iterate_sync)

                try:
                    first_chunk = True

                    while True:
                        msg_type, data = await queue.get()

                        if msg_type == "error":
                            raise data

                        if msg_type == "done":
                            tool_calls = data
                            break

                        # msg_type == "chunk"
                        text_chunk = data
                        if _debug and text_chunk:
                            logger.debug(f"Got chunk: {text_chunk[:50] if len(text_chunk) > 50 else text_chunk}...")
                        if text_chunk:
                            delta_content: Dict[str, Any] = {"content": text_chunk}
                            if first_chunk:
                                delta_content["role"] = "assistant"
                                first_chunk = False

                            chunk_msg = {
                                "id": _response_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": _model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta_content,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk_msg)}\n\n"

                    # Stream tool calls if present
                    if tool_calls:
                        from ..adapters.tool_adapter import format_streaming_tool_call_delta
                        tool_delta = format_streaming_tool_call_delta(tool_calls, 0)
                        if tool_delta:
                            tool_msg = {
                                "id": _response_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": _model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"tool_calls": tool_delta},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(tool_msg)}\n\n"

                    # Final message with finish_reason
                    finish_reason = "tool_calls" if tool_calls else "stop"
                    final_msg = {
                        "id": _response_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": _model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_msg)}\n\n"
                    yield "data: [DONE]\n\n"
                    if _debug:
                        logger.debug("SSE generator completed successfully")

                except Exception as e:
                    if _debug:
                        logger.exception(f"Error in SSE generator: {e}")
                    error_msg = {
                        "error": {
                            "message": str(e),
                            "type": "server_error",
                        }
                    }
                    yield f"data: {json.dumps(error_msg)}\n\n"

            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Request-ID": _response_id,
                    "x-ms-client-request-id": _response_id,
                },
            )
        else:
            # Non-streaming response - run blocking call in executor
            loop = asyncio.get_event_loop()
            full_text = await loop.run_in_executor(_executor, response.text)

            # Check for tool calls
            tool_calls = None
            try:
                tool_calls = await loop.run_in_executor(_executor, response.tool_calls)
            except AttributeError:
                pass  # Model doesn't support tool calls
            except Exception as e:
                if settings.debug:
                    logger.debug(f"Error getting tool calls: {e}")

            if tool_calls:
                return JSONResponse(
                    content=format_tool_call_response(
                        tool_calls, model_name, response_id
                    )
                )

            # Regular text response
            return JSONResponse(
                content={
                    "id": response_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": full_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": response.input_tokens or 0,
                        "completion_tokens": response.output_tokens or 0,
                        "total_tokens": (response.input_tokens or 0)
                        + (response.output_tokens or 0),
                    },
                }
            )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "internal_error",
                }
            },
        )
