"""Chat completions endpoint - main LLM interaction point."""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, field_validator

import llm

from ..adapters.openai_adapter import (
    parse_conversation,
    convert_tool_definitions,
    extract_tool_results,
    ImageSizeError,
)
from ..adapters.tool_adapter import format_tool_call_response
from ..adapters.model_adapters import get_adapter
from ..config import settings, executor, get_model_with_fallback
from ..streaming.sse import stream_llm_response

logger = logging.getLogger(__name__)

router = APIRouter()


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

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages list cannot be empty")
        return v


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion using the llm library.

    Supports both streaming and non-streaming responses.
    """
    # Debug logging (only when --debug flag is set)
    if settings.debug:
        logger.debug("=== Chat request ===")
        logger.debug(f"Model: {request.model}")
        logger.debug(f"Stream: {request.stream}")
        logger.debug(f"Messages count: {len(request.messages)}")
        for i, msg in enumerate(request.messages):
            logger.debug(f"  [{i}] role={msg.role}, content_type={type(msg.content).__name__}, content_len={len(str(msg.content))}")

    response_id = generate_response_id()

    # Get the model using shared fallback chain
    try:
        model, model_name, was_fallback = get_model_with_fallback(request.model, settings.debug)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "invalid_request_error", "code": "model_not_found"}}
        )

    # Build base headers for responses
    base_headers = {}
    if was_fallback:
        base_headers["X-Model-Fallback"] = "true"
        base_headers["X-Requested-Model"] = request.model

    # Parse messages into structured conversation data
    try:
        messages_raw = [m.model_dump() for m in request.messages]
        if settings.debug:
            logger.debug(f"Parsed {len(messages_raw)} raw messages")
        conv_data = parse_conversation(messages_raw)
        if settings.debug:
            logger.debug(f"Conv data: system={bool(conv_data.system_prompt)}, history={len(conv_data.messages)}, prompt={bool(conv_data.current_prompt)}")
    except ImageSizeError as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "invalid_request_error", "code": "image_too_large"}}
        )
    except Exception as e:
        if settings.debug:
            logger.exception(f"Error parsing messages: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"Failed to parse messages: {e}", "type": "parse_error", "code": "parse_error"}}
        )

    # Convert tools to llm format (for future tool-using requests)
    tools_raw = [t.model_dump() for t in request.tools] if request.tools else None
    llm_tools = convert_tool_definitions(tools_raw)

    # Extract tool results from messages (works for all models now - names are guaranteed non-empty)
    tool_results = extract_tool_results(messages_raw)
    if settings.debug and tool_results:
        logger.debug(f"Extracted {len(tool_results)} tool results")

    # Get adapter for model-specific behavior
    adapter = get_adapter(model_name)

    # Build options - only include ones supported by the model
    options: Dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.max_tokens is not None:
        options["max_tokens"] = request.max_tokens
    if request.top_p is not None:
        options["top_p"] = request.top_p

    # Filter options through adapter (removes unsupported options like max_tokens for Gemini)
    options = adapter.prepare_options(options)

    try:
        # Build the full prompt including conversation history
        full_prompt = ""

        # Add conversation history as context
        # Note: Tool messages are excluded here since they're passed via the tool_results parameter
        if conv_data.messages:
            history_parts = []
            for msg in conv_data.messages:
                if msg.role == "user" and msg.content:
                    history_parts.append(f"User: {msg.content}")
                elif msg.role == "assistant":
                    if msg.content:
                        history_parts.append(f"Assistant: {msg.content}")
                    # Include tool calls in history for context (with arguments)
                    if msg.tool_calls:
                        tool_parts = []
                        for tc in msg.tool_calls:
                            func = tc.get("function", {})
                            name = func.get("name", "unknown")
                            args = func.get("arguments", "{}")
                            tool_parts.append(f"{name}({args})")
                        history_parts.append(f"Assistant called tools: {', '.join(tool_parts)}")
                # Skip tool messages - they're passed via the tool_results parameter
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
                stream=request.stream,
                **options,
            )
            if settings.debug:
                logger.debug(f"Got response object: {type(response)}")
        except Exception as e:
            if settings.debug:
                logger.exception(f"Error calling model.prompt: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": f"LLM error: {e}", "type": "llm_error", "code": "llm_error"}}
            )

        if request.stream:
            # Streaming response using shared SSE utilities
            stream_headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Request-ID": response_id,
                "x-ms-client-request-id": response_id,
                **base_headers,
            }
            return StreamingResponse(
                stream_llm_response(
                    response=response,
                    executor=executor,
                    model_id=model_name,
                    response_id=response_id,
                    response_type="chat",
                    debug=settings.debug,
                ),
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

            # Check for tool calls
            tool_calls = None
            tool_call_warning = None
            try:
                tool_calls = await asyncio.wait_for(
                    loop.run_in_executor(executor, lambda: response.tool_calls()),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                tool_call_warning = f"Timeout extracting tool calls after {timeout}s"
                logger.warning(tool_call_warning)
            except AttributeError:
                pass  # Model doesn't support tool calls
            except Exception as e:
                tool_call_warning = f"Error extracting tool calls: {type(e).__name__}"
                if settings.debug:
                    logger.debug(f"Error getting tool calls: {e}")

            # Add warning header if tool call extraction failed
            if tool_call_warning:
                base_headers["X-Tool-Call-Warning"] = tool_call_warning

            if tool_calls:
                return JSONResponse(
                    content=format_tool_call_response(
                        tool_calls, model_name, response_id
                    ),
                    headers=base_headers if base_headers else None,
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
                },
                headers=base_headers if base_headers else None,
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
