# lmstudio-mcp

A single-binary [MCP](https://modelcontextprotocol.io) server that bridges Claude
(or any MCP client) to a local [LM Studio](https://lmstudio.ai/) instance —
inference **and** model management, in one Rust binary that runs natively on
macOS, Windows, and Linux.

This project merges the functionality of two earlier, separate MCP servers:

- **[LMStudio-MCP](../LMStudio-MCP)** (Python) — inference: chat/text
  completions, embeddings, and stateful conversations via LM Studio's
  OpenAI-compatible API.
- **[lm-studio-mcp-server](../lm-studio-mcp-server)** (TypeScript) — model
  management: list, load, unload, and inspect models via LM Studio's SDK.

Both capabilities are reimplemented here over LM Studio's **native REST API**
(`/api/v1/...`, added in recent LM Studio releases alongside the
OpenAI-compatible `/v1/...` surface), so there's no WebSocket SDK dependency
and no Python/Node runtime to install — just one executable.

## Tools

| Tool | From | Description |
|------|------|-------------|
| `health_check` | both | Verify LM Studio is reachable |
| `list_models` | TS | List all downloaded models in the local library |
| `list_loaded_models` | TS | List currently loaded model instances |
| `get_current_model` | Python | Identify the loaded model (replaces the Python original's hack of asking the model to name itself — this reads it straight from the API) |
| `get_model_info` | TS | Detailed info for one loaded instance |
| `load_model` | TS | Load a model into memory |
| `unload_model` | TS | Unload a model instance |
| `chat_completion` | Python | Chat-formatted completion |
| `text_completion` | Python | Raw/non-chat completion (faster, for code/continuation) |
| `generate_embeddings` | Python | Vector embeddings for RAG/semantic search |
| `create_response` | Python | Stateful response via `/v1/responses` |
| `start_conversation` | Python | Begin a multi-turn session with a locked-in system prompt |
| `continue_conversation` | Python | Continue a session started above |

Every tool returns the same envelope — `{ success, message, data?, error? }`
— so a client can handle success and failure uniformly (this convention is
carried over from the TypeScript project).

> **Note on `create_response` / `start_conversation` / `continue_conversation`:**
> the original Python bridge accepted `temperature` and `max_tokens`
> parameters on these three tools but never actually included them in the
> request body — a latent bug. This port fixes that: both are sent through
> (as `temperature` / `max_output_tokens`, matching the Responses API).

## Prerequisites

- [LM Studio](https://lmstudio.ai/) running locally with its local server
  enabled (LM Studio → Developer tab → Start Server), on a version that
  exposes the native `/api/v1` REST API.
- Rust 1.75+ if building from source (see [Building](#building)).

## Configuration

Read from the environment at connect time:

| Variable | Default | Description |
|----------|---------|--------------|
| `LMSTUDIO_HOST` | `127.0.0.1` | LM Studio host |
| `LMSTUDIO_PORT` | `1234` | LM Studio port |
| `LMSTUDIO_BASE_URL` | _(derived)_ | Full `scheme://host:port` override, e.g. `http://192.168.1.100:5678`. Takes precedence over host/port. |
| `LMSTUDIO_API_TOKEN` | _(none)_ | Bearer token, if LM Studio's server has API token auth enabled (Developer tab → Server Settings). Sent as `Authorization: Bearer <token>`. |

## Building

```bash
git clone <this-repo-url>
cd lmstudio-mcp
cargo build --release
```

The binary is produced at `target/release/lmstudio-mcp` (`.exe` on Windows).
Pre-built binaries for macOS (arm64/x86_64), Windows, and Linux are also
published as CI artifacts on every push — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## MCP Client Configuration

### Claude Code

```bash
claude mcp add lmstudio -- /path/to/lmstudio-mcp
```

Or add directly to `.mcp.json` / `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "lmstudio": {
      "command": "/path/to/lmstudio-mcp",
      "env": {
        "LMSTUDIO_HOST": "127.0.0.1",
        "LMSTUDIO_PORT": "1234"
      }
    }
  }
}
```

On Windows, point `command` at the `.exe`:

```json
{
  "mcpServers": {
    "lmstudio": {
      "command": "C:\\path\\to\\lmstudio-mcp.exe"
    }
  }
}
```

### Connecting to LM Studio on another machine

```json
{
  "mcpServers": {
    "lmstudio": {
      "command": "/path/to/lmstudio-mcp",
      "env": {
        "LMSTUDIO_BASE_URL": "http://192.168.1.100:1234"
      }
    }
  }
}
```

## Architecture

```
src/
├── main.rs       # Entry point: logging, config, stdio transport, shutdown
├── server.rs      # Wires each tools::* function to an #[tool]-annotated method
├── client.rs      # HTTP client over LM Studio's native + OpenAI-compatible APIs
├── config.rs       # Environment-variable configuration
├── types.rs         # Shared ToolResult<T> envelope, ErrorCode, ClientError
└── tools/
    ├── health_check.rs
    ├── models.rs       # list_models, list_loaded_models, get_current_model,
    │                   # get_model_info, load_model, unload_model
    ├── chat.rs         # chat_completion, text_completion
    ├── embeddings.rs   # generate_embeddings
    └── responses.rs    # create_response, start_conversation, continue_conversation
```

Built on the official [`rmcp`](https://github.com/modelcontextprotocol/rust-sdk)
Rust MCP SDK, `tokio` for async I/O, and `reqwest` with `rustls` (no OpenSSL
dependency, for easy cross-compilation and static-ish binaries on all three
platforms).

### A note on LM Studio's native REST API

LM Studio's `/api/v1` model-management endpoints are newer and less
exhaustively documented than the stable OpenAI-compatible surface. Response
structs in `src/client.rs` are typed against LM Studio's published API
reference, but every field is optional with `#[serde(default)]` and the
top-level model entry keeps a `#[serde(flatten)] extra` catch-all — so a
field LM Studio adds, renames, or omits in a future version degrades
gracefully (parses with that field missing/extra) rather than failing to
parse at all. If you hit a real mismatch against your LM Studio version,
it's isolated to `src/client.rs`.

## Development

```bash
cargo fmt --all           # format
cargo clippy --all-targets -- -D warnings   # lint
cargo test                 # unit tests
cargo build --release      # release binary
```

## License

MIT
