"""Adapter for converting between OpenAI chat format and llm library format."""

import base64
import binascii
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import llm

logger = logging.getLogger(__name__)

# Maximum image size (20MB) to prevent memory exhaustion attacks
MAX_IMAGE_SIZE = 20 * 1024 * 1024


@dataclass
class ParsedMessage:
    """A parsed message from OpenAI format."""
    role: str
    content: str
    attachments: List[llm.Attachment]
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ConversationData:
    """Structured conversation data for the llm library."""
    system_prompt: Optional[str]
    messages: List[ParsedMessage]
    current_prompt: Optional[str]
    current_attachments: List[llm.Attachment]


def extract_messages(
    messages: List[Dict[str, Any]]
) -> Tuple[Optional[str], str, List[llm.Attachment]]:
    """
    Extract system prompt, user prompt, and attachments from OpenAI messages.

    This is the legacy function for simple single-turn usage.

    Returns:
        Tuple of (system_prompt, user_prompt, attachments)
    """
    conv_data = parse_conversation(messages)

    # For backwards compatibility, combine all non-system messages
    all_parts = []
    all_attachments = list(conv_data.current_attachments)

    for msg in conv_data.messages:
        if msg.role == "user":
            if msg.content:
                all_parts.append(msg.content)
            all_attachments.extend(msg.attachments)
        elif msg.role == "assistant":
            if msg.content:
                all_parts.append(f"Assistant: {msg.content}")

    if conv_data.current_prompt:
        all_parts.append(conv_data.current_prompt)

    return conv_data.system_prompt, "\n\n".join(all_parts), all_attachments


def _normalize_role(role: Any) -> str:
    """Convert role to string format (handles numeric roles from some clients)."""
    if isinstance(role, int):
        # Some clients send: 0=system, 1=user, 2=assistant, 3=tool
        role_map = {0: "system", 1: "user", 2: "assistant", 3: "tool"}
        return role_map.get(role, "user")
    return str(role) if role else "user"


def parse_conversation(messages: List[Dict[str, Any]]) -> ConversationData:
    """
    Parse OpenAI messages into structured conversation data.

    Returns:
        ConversationData with system prompt, message history, and current prompt
    """
    system_prompt: Optional[str] = None
    parsed_messages: List[ParsedMessage] = []

    for msg in messages:
        role = _normalize_role(msg.get("role", ""))
        content = msg.get("content", "")

        if role == "system":
            system_prompt = _extract_text_content(content)

        elif role == "user":
            text, attachments = _extract_content_and_attachments(content)
            parsed_messages.append(ParsedMessage(
                role="user",
                content=text,
                attachments=attachments,
            ))

        elif role == "assistant":
            text = _extract_text_content(content) or ""
            tool_calls = msg.get("tool_calls")
            parsed_messages.append(ParsedMessage(
                role="assistant",
                content=text,
                attachments=[],
                tool_calls=tool_calls,
            ))

        elif role == "tool":
            # Tool result message
            parsed_messages.append(ParsedMessage(
                role="tool",
                content=_extract_text_content(content) or "",
                attachments=[],
                tool_call_id=msg.get("tool_call_id"),
                name=msg.get("name"),
            ))

    # The last user message is the current prompt
    # All other messages (including those after it) become history
    current_prompt: Optional[str] = None
    current_attachments: List[llm.Attachment] = []
    history_messages: List[ParsedMessage] = []
    last_user_idx: Optional[int] = None

    # Find last user message to use as current prompt
    for i in range(len(parsed_messages) - 1, -1, -1):
        if parsed_messages[i].role == "user":
            last_user_idx = i
            current_prompt = parsed_messages[i].content
            current_attachments = parsed_messages[i].attachments
            break

    if last_user_idx is not None:
        # History includes messages both BEFORE and AFTER the current user message
        # This preserves tool calling context (e.g., tool results that follow user messages)
        history_messages = parsed_messages[:last_user_idx] + parsed_messages[last_user_idx+1:]
    else:
        # No user message found, use all as history
        history_messages = parsed_messages

    return ConversationData(
        system_prompt=system_prompt,
        messages=history_messages,
        current_prompt=current_prompt,
        current_attachments=current_attachments,
    )


def _extract_text_content(content: Any) -> Optional[str]:
    """Extract text from content (string or array format)."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                # Handle both string "text" and numeric 1
                if part_type == "text" or part_type == 1:
                    text_parts.append(part.get("text", ""))
                elif part_type == "value" or "value" in part:
                    text_parts.append(part.get("value", ""))
            else:
                logger.debug(f"Skipping non-dict content part: {type(part)}")
        if not text_parts:
            logger.debug("No text content found in array")
        return "\n".join(text_parts) if text_parts else None
    if content is not None:
        logger.debug(f"Unexpected content type: {type(content)}")
    return None


def _extract_content_and_attachments(
    content: Any
) -> Tuple[str, List[llm.Attachment]]:
    """Extract text and attachments from content."""
    if isinstance(content, str):
        return content, []

    text_parts: List[str] = []
    attachments: List[llm.Attachment] = []

    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type", "")
                # Handle both string "text" and numeric 1
                if part_type == "text" or part_type == 1:
                    text_parts.append(part.get("text", ""))
                elif part_type == "image_url" or part_type == 2:
                    image_url = part.get("image_url", {})
                    url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                    attachment = _create_image_attachment(url)
                    if attachment:
                        attachments.append(attachment)

    return "\n".join(text_parts), attachments


class ImageSizeError(ValueError):
    """Raised when an image exceeds the size limit."""
    pass


def _create_image_attachment(url: str) -> Optional[llm.Attachment]:
    """Create an llm Attachment from an image URL or base64 data.

    Raises:
        ImageSizeError: If the image exceeds the size limit.
    """
    if not url:
        return None

    if url.startswith("data:"):
        # Base64 encoded image with data URL: data:image/png;base64,<data>
        try:
            header, data = url.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            content = base64.b64decode(data)
            # Validate image size to prevent memory exhaustion
            if len(content) > MAX_IMAGE_SIZE:
                raise ImageSizeError(
                    f"Image exceeds size limit: {len(content)} bytes (max {MAX_IMAGE_SIZE} bytes / {MAX_IMAGE_SIZE // 1024 // 1024}MB)"
                )
            return llm.Attachment(type=mime_type, content=content)
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse data URL: {e}")
            return None
    elif url.startswith("http://") or url.startswith("https://"):
        # URL-based image
        return llm.Attachment(url=url)
    else:
        # Try to detect raw base64 data (some clients send this without data URL prefix)
        # Common base64 prefixes for images:
        # - PNG: iVBORw0KGgo (starts with \x89PNG)
        # - JPEG: /9j/ (starts with \xff\xd8\xff)
        # - GIF: R0lGOD (starts with GIF)
        # - WebP: UklGR (starts with RIFF)
        try:
            # Attempt to decode as base64
            content = base64.b64decode(url)
            # Validate image size to prevent memory exhaustion
            if len(content) > MAX_IMAGE_SIZE:
                raise ImageSizeError(
                    f"Image exceeds size limit: {len(content)} bytes (max {MAX_IMAGE_SIZE} bytes / {MAX_IMAGE_SIZE // 1024 // 1024}MB)"
                )
            # Detect mime type from magic bytes
            mime_type = _detect_image_mime_type(content)
            if mime_type:
                return llm.Attachment(type=mime_type, content=content)
            else:
                logger.debug("Could not detect image MIME type from magic bytes")
        except (ValueError, binascii.Error) as e:
            logger.debug(f"Failed to decode raw base64 image: {e}")
    return None


def _detect_image_mime_type(data: bytes) -> Optional[str]:
    """Detect image MIME type from magic bytes."""
    if not data or len(data) < 8:
        logger.debug(f"Data too short for MIME detection: {len(data) if data else 0} bytes")
        return None

    # PNG: \x89PNG\r\n\x1a\n
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    # JPEG: \xff\xd8\xff
    if data[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    # GIF: GIF87a or GIF89a
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    # WebP: RIFF....WEBP
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return "image/webp"
    # BMP: BM
    if data[:2] == b'BM':
        return "image/bmp"

    logger.debug(f"Unrecognized image magic bytes: {data[:8].hex()}")
    return None


def convert_tool_definitions(tools: Optional[List[Dict[str, Any]]]) -> List[llm.Tool]:
    """Convert OpenAI tool definitions to llm Tool objects."""
    if not tools:
        return []

    llm_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            name = func.get("name", "")
            if not name:
                logger.warning("Skipping tool with empty name")
                continue
            llm_tools.append(
                llm.Tool(
                    name=name,
                    description=func.get("description", ""),
                    input_schema=func.get("parameters", {}),
                )
            )
    return llm_tools


def convert_tool_calls_to_openai(
    tool_calls: List[Any],
) -> List[Dict[str, Any]]:
    """Convert llm ToolCall objects to OpenAI format."""
    result = []
    for tc in tool_calls:
        func_name = tc.name if hasattr(tc, "name") else str(tc)
        # Use UUID for deterministic, unique IDs
        result.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": (
                        tc.arguments
                        if hasattr(tc, "arguments")
                        else "{}"
                    ),
                },
            }
        )
    return result


def extract_tool_results(
    messages: List[Dict[str, Any]]
) -> List[llm.ToolResult]:
    """
    Extract tool results from messages with role 'tool'.

    This function builds a mapping of tool_call_id → function_name from
    assistant messages, then uses that to ensure all tool results have
    valid names (required by Gemini).
    """
    # First pass: build mapping of tool_call_id → name from assistant tool_calls
    tool_call_names: Dict[str, str] = {}
    for msg in messages:
        role = _normalize_role(msg.get("role"))
        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    tc_id = tc.get("id", "")
                    func = tc.get("function", {})
                    func_name = func.get("name", "")
                    if tc_id and func_name:
                        tool_call_names[tc_id] = func_name

    # Second pass: extract tool results with guaranteed names
    results = []
    fallback_counter = 0
    for msg in messages:
        role = _normalize_role(msg.get("role"))
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")

            # Try to get name from multiple sources (in order of preference)
            name = msg.get("name", "")

            # 1. Look up from assistant's tool_calls mapping
            if not name and tool_call_id:
                name = tool_call_names.get(tool_call_id, "")

            # 2. Use tool_call_id itself as name
            if not name and tool_call_id:
                name = tool_call_id

            # 3. CRITICAL: Always provide a unique fallback name (Gemini requires non-empty)
            if not name:
                fallback_counter += 1
                name = f"tool_result_{fallback_counter}"

            results.append(
                llm.ToolResult(
                    name=name,
                    output=content,
                    tool_call_id=tool_call_id,
                )
            )
    return results
