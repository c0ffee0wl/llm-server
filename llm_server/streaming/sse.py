"""Server-Sent Events utilities for streaming responses."""

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import Executor, Future, TimeoutError as FuturesTimeoutError
from typing import Any, AsyncIterator, Dict, Optional

from ..config import settings

logger = logging.getLogger(__name__)

# Timeout for waiting on chunks (5 minutes to allow for slow models)
CHUNK_TIMEOUT = 300.0
# Timeout for queue operations in the background thread
QUEUE_PUT_TIMEOUT = 5.0


def format_sse_message(data: Dict[str, Any]) -> str:
    """Format a dictionary as an SSE message."""
    return f"data: {json.dumps(data)}\n\n"


def format_sse_done() -> str:
    """Format the final SSE done message."""
    return "data: [DONE]\n\n"


async def stream_llm_response(
    response: Any,
    executor: Executor,
    model_id: str,
    response_id: str,
    response_type: str = "chat",
    debug: bool = False,
    on_complete: Optional[callable] = None,
) -> AsyncIterator[str]:
    """
    Stream an LLM response using Server-Sent Events.

    Bridges the synchronous llm library iteration to async streaming
    using a queue and thread pool executor.

    Args:
        response: The llm library response object (iterable)
        executor: ThreadPoolExecutor for running sync operations
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

    # Use queue with maxsize to provide backpressure when client is slow
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    loop = asyncio.get_running_loop()

    # Cancellation signal for the background thread
    cancel_event = threading.Event()

    # Track the background task future
    task_future: Optional[Future] = None

    def iterate_sync():
        """Run sync iteration in thread, put chunks in queue."""
        def safe_queue_put(item):
            """Thread-safe queue put with cancellation and event loop checks."""
            while not cancel_event.is_set():
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        queue.put(item), loop
                    )
                    future.result(timeout=QUEUE_PUT_TIMEOUT)
                    return True
                except TimeoutError:
                    # Queue full, check cancellation and retry
                    continue
                except RuntimeError:
                    # Event loop is closed (client disconnected)
                    return False
                except Exception:
                    # Other error (loop closed, etc.)
                    return False
            return False

        try:
            for chunk in response:
                # Check cancellation and put chunk atomically
                if cancel_event.is_set():
                    if debug:
                        logger.debug("Background iteration cancelled")
                    return

                if not safe_queue_put(("chunk", chunk)):
                    return

            # Signal completion with tool_calls if any (chat completions only)
            tool_calls = None
            if response_type == "chat":
                try:
                    tool_calls = response.tool_calls()
                except AttributeError:
                    pass
                except Exception as e:
                    if debug:
                        logger.debug(f"Error getting tool calls: {e}")

            if not cancel_event.is_set():
                safe_queue_put(("done", tool_calls))

        except Exception as e:
            if not cancel_event.is_set():
                safe_queue_put(("error", e))

    # Start sync iteration in thread pool and track the future
    task_future = executor.submit(iterate_sync)

    try:
        first_chunk = True

        while True:
            try:
                msg_type, data = await asyncio.wait_for(
                    queue.get(), timeout=CHUNK_TIMEOUT
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"LLM response timed out after {CHUNK_TIMEOUT} seconds"
                )

            if msg_type == "error":
                raise data

            if msg_type == "done":
                tool_calls = data
                break

            # msg_type == "chunk"
            text_chunk = data
            if debug and text_chunk:
                preview = text_chunk[:50] if len(text_chunk) > 50 else text_chunk
                logger.debug(f"Got chunk: {preview}...")

            if response_type == "chat":
                # Chat completion chunk format
                # Always include role on first chunk, even if content is empty
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

        # Call completion callback (e.g., for database logging)
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

    finally:
        # Signal the background thread to stop
        cancel_event.set()

        # Wait for thread to finish with configurable timeout
        if task_future is not None:
            try:
                task_future.result(timeout=settings.cleanup_timeout)
            except FuturesTimeoutError:
                logger.warning(
                    f"Streaming thread did not finish within {settings.cleanup_timeout}s - "
                    "it will complete in background"
                )
            except Exception as e:
                if debug:
                    logger.debug(f"Thread cleanup exception: {e}")

        if debug:
            logger.debug("SSE generator cleanup complete")
