# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Server is an OpenAI-compatible HTTP API wrapper for the `llm` library. It allows any OpenAI API client to use local or alternative LLM models (Gemini, Vertex AI, Anthropic Claude, OpenRouter) through a unified interface.

## Commands

```bash
# Run the server (development)
llm-server --debug --host 127.0.0.1 --port 11435

# Run with a specific model
llm-server -m gemini-1.5-pro

# Find model by query substring
llm-server -q gemini -q flash

# Run as background daemon
llm-server --daemon --pidfile /var/run/llm-server.pid --logfile /var/log/llm-server.log

# Install as systemd socket-activated service (user-level)
llm-server --service

# Install as system-level service
llm-server --service --system

# Uninstall systemd service
llm-server --uninstall-service

# Run tests
pytest

# Configure VS Code for local LLM mode
configure-vscode --user
```

## Architecture

### Request Flow

```
OpenAI Client → FastAPI Endpoint → Adapter Layer → llm Library (async) → LLM Provider
                     ↓
              Streaming: Native async iteration with SSE
              Non-streaming: Native async with timeout
```

### Key Components

**Routes (`llm_server/routes/`):**
- `chat.py` - `/v1/chat/completions` and `/v2/chat/completions` endpoints with conversation history, tool calling, image attachments
- `completions.py` - `/v1/completions` and `/v2/completions` legacy text completion endpoints with echo/suffix support
- `models.py` - `/v1/models` and `/v2/models` endpoints with 1-hour TTL cache

### API Versions

The server provides two API versions with different model selection behavior:

| Endpoint | Model Selection |
|----------|-----------------|
| `/v1/*` | **Server default** - Uses the server's configured model (`llm models default` or `-m` flag), ignoring the client's `model` parameter |
| `/v2/*` | **Client choice** - Respects the client's requested `model` parameter, falls back to server default only if unavailable |

Use `/v1` when you want centralized model control (e.g., all clients use the same model).
Use `/v2` when clients need to select specific models (e.g., different models for different tasks).

**Adapters (`llm_server/adapters/`):**
- `openai_adapter.py` - Converts OpenAI message format to llm library format. Handles multimodal content, tool definitions, and tool results. Key function: `parse_conversation()`
- `model_adapters.py` - Provider-specific option translation (e.g., `max_tokens` → `max_output_tokens` for Gemini)
- `tool_adapter.py` - Formats tool call responses for OpenAI compatibility

**Streaming (`llm_server/streaming/sse.py`):**
- Uses native async iteration from the llm library's AsyncResponse
- Formats chunks as Server-Sent Events for OpenAI API compatibility
- Handles tool calls at stream completion

**Config (`llm_server/config.py`):**
- `Settings` class with `LLM_SERVER_*` environment variable prefix
- `get_async_model_with_fallback()` - For `/v1`: llm default → requested → settings → first available
- `get_async_model_client_choice()` - For `/v2`: requested → llm default → settings → first available
- `ConversationTracker` - Hash-based conversation grouping for database logging
- `log_response_to_db()` - Handles both sync Response and async AsyncResponse objects

### Model Detection

The `is_gemini_model()` function checks for `gemini/`, `gemini-`, and `vertex/` prefixes to select the appropriate adapter. This is important for option translation.

### Environment Variables

- `LLM_SERVER_HOST` / `LLM_SERVER_PORT` - Bind address (default: 127.0.0.1:11435)
- `LLM_SERVER_MODEL_NAME` - Default model
- `LLM_SERVER_DEBUG` - Enable debug logging
- `LLM_SERVER_NO_LOG` - Disable database logging
- `LLM_SERVER_REQUEST_TIMEOUT` - Request timeout in seconds (default: 300)

### Database Logging

Responses are logged to `~/.config/llm/log-server.db` using the llm library's migration system. Conversations are grouped by hashing message sequences. Disable with `--no-log` or `LLM_SERVER_NO_LOG=true`.
