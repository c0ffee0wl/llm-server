# llm-server

OpenAI-compatible HTTP API wrapper for the [llm](https://github.com/simonw/llm) library.

Use any OpenAI API client with local or alternative LLM providers (Gemini, Vertex AI, Anthropic Claude, OpenRouter) through a single unified interface.

## Features

- **OpenAI API compatibility** -- drop-in replacement for OpenAI endpoints
- **Streaming** -- Server-Sent Events for real-time token streaming
- **Tool calling** -- function/tool call support in OpenAI-compatible format
- **Multimodal** -- image attachments via base64, data URLs, or HTTP URLs
- **Embeddings** -- text embedding endpoint supporting multiple providers
- **Dual API versions** -- `/v1` (server-controlled model) and `/v1c` (client-choice model)
- **Conversation tracking** -- hash-based conversation grouping with database logging
- **Daemon mode** -- background operation with PID file management
- **Systemd integration** -- socket-activated service installation

## Installation

Requires Python 3.10+.

```bash
uv pip install -e ".[dev]"
```

The llm library and provider plugins (Gemini, Vertex AI, OpenRouter, Anthropic) are installed automatically as dependencies.

## Quick Start

```bash
# Start the server
llm-server --host 127.0.0.1 --port 11435

# With a specific model
llm-server -m gemini-2.0-flash

# Find model by query substring
llm-server -q gemini -q flash

# Debug mode with verbose logging
llm-server --debug
```

Then point any OpenAI-compatible client at `http://127.0.0.1:11435/v1`:

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.0-flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (server default model) |
| `POST` | `/v1c/chat/completions` | Chat completions (client's model choice) |
| `POST` | `/v1/completions` | Text completions (server default model) |
| `POST` | `/v1c/completions` | Text completions (client's model choice) |
| `POST` | `/v1/engines/{id}/completions` | Legacy engine-scoped completions |
| `POST` | `/v1c/engines/{id}/completions` | Legacy engine-scoped completions |
| `POST` | `/v1/embeddings` | Text embeddings |
| `POST` | `/v1c/embeddings` | Text embeddings |
| `GET`  | `/v1/models` | List available models |
| `GET`  | `/v1c/models` | List available models |
| `GET`  | `/v1/models/{id}` | Get model details |
| `GET`  | `/v1c/models/{id}` | Get model details |
| `GET`  | `/health` | Health check |
| `GET`  | `/` | API information |

### API Versions

The server provides two API prefixes with different model selection behavior:

| Prefix | Model Selection | Use Case |
|--------|----------------|----------|
| `/v1`  | **Server default** -- uses the server's configured model, ignoring the client's `model` parameter | Centralized model control (all clients use the same model) |
| `/v1c` | **Client choice** -- respects the client's `model` parameter, falls back to server default if unavailable | Per-request model selection (different models for different tasks) |

Embeddings endpoints always respect the client's model choice regardless of prefix, since embedding models are typically task-specific.

## Deployment

### Systemd Service (Socket-Activated)

```bash
# Install as user-level service
llm-server --service

# Install as system-level service (requires root)
llm-server --service --system

# With a specific model
llm-server --service -m gemini-2.0-flash

# Uninstall
llm-server --uninstall-service
```

### Daemon Mode

```bash
llm-server --daemon \
  --pidfile /var/run/llm-server.pid \
  --logfile /var/log/llm-server.log
```

## Configuration

All settings can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_SERVER_HOST` | `127.0.0.1` | Bind address |
| `LLM_SERVER_PORT` | `11435` | Bind port |
| `LLM_SERVER_MODEL_NAME` | *(llm default)* | Default model |
| `LLM_SERVER_DEBUG` | `false` | Enable debug logging |
| `LLM_SERVER_NO_LOG` | `false` | Disable database logging |
| `LLM_SERVER_REQUEST_TIMEOUT` | `300` | Request timeout in seconds |

### Database Logging

Responses are logged to `~/.config/llm/log-server.db` using the llm library's migration system. Conversations are automatically grouped by hashing message sequences. Disable with `--no-log` or `LLM_SERVER_NO_LOG=true`.

## VS Code Integration

Configure VS Code to use the local LLM server instead of cloud services:

```bash
configure-vscode --user       # Configure user-level settings
configure-vscode --workspace  # Configure workspace-level settings
configure-vscode --dry-run    # Preview changes without applying
configure-vscode --restore    # Restore default settings
```

## License

[GNU General Public License v3.0](LICENSE)
