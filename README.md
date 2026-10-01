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
| `run_subagent` | new | Delegate a task to a local model acting as a sandboxed subagent |
| `feedback_create` / `_list` / `_get` / `_update` / `_delete` | new | Draft/manage local feedback entries about this server |
| `feedback_check_duplicates` / `feedback_submit` | new | Check and submit a feedback entry as a GitHub issue |

Every tool returns the same envelope — `{ success, message, data?, error? }`
— so a client can handle success and failure uniformly (this convention is
carried over from the TypeScript project).

> **Note on `create_response` / `start_conversation` / `continue_conversation`:**
> the original Python bridge accepted `temperature` and `max_tokens`
> parameters on these three tools but never actually included them in the
> request body — a latent bug. This port fixes that: both are sent through
> (as `temperature` / `max_output_tokens`, matching the Responses API).

### Generation is streamed internally

`chat_completion`, `text_completion`, and the `/v1/responses` tools always
request `"stream": true` from LM Studio internally and reassemble the full
response here — the MCP tool call itself is still a single request/response,
this is purely a transport-level choice. The reason: a non-streaming call
has to be timed out against its *total* duration, and no duration is both
short enough to catch a genuinely hung connection and long enough for a slow
reasoning model's legitimate output. Streaming turns that into an *idle*
timeout instead (reset on every chunk that actually arrives, 60s default) —
a model generating steadily for minutes is never killed, and a stalled
connection is still caught quickly. See `src/sse.rs`.

### Reasoning models

A reasoning/thinking model (e.g. Qwen3) can spend an entire `max_tokens`
budget on its internal deliberation and produce no final answer. Rather than
reporting that as a bare failure, affected tools return it as a *success*
with the reasoning trace included in `reasoning_content` and a message
explaining what happened — so you can see why and retry with a larger
budget instead of getting an opaque error.

## Subagents: `run_subagent`

Lets the calling MCP client (e.g. Claude) delegate a task to a local LM
Studio model acting as a subagent with its own tools — comparable to how
Claude Code spawns a subagent with a task-appropriate tool profile and gets
back a final report, not the full transcript.

```json
{
  "task": "Investigate why the build fails and report what you find.",
  "working_directory": "/path/to/project",
  "capability": "read_only"
}
```

The subagent runs its own tool-calling loop (via LM Studio's OpenAI-style
function calling) against a fixed, sandboxed tool set — every path it's
given is resolved against, and checked to stay within, `working_directory`
(rejects `../` escapes, absolute paths outside it, and symlinks that resolve
back out) — up to `max_turns` (default 15), then returns one final report:
the answer, a brief action log, and why it stopped.

**Capability tiers** (`capability`, defaults to `read_only` — grant only
what the task needs):

| Tier | Adds | 
|------|------|
| `read_only` | `read_file`, `list_directory`, `search_files` |
| `read_write` | + `write_file` |
| `read_write_shell` | + `run_command` (shell, `cwd` = working directory, 60s timeout, output capped) |

`run_command` is additionally gated by a **non-bypassable, non-configurable**
pattern guard (`src/subagent/guard.rs`) that blocks high-risk command shapes
regardless of what the subagent was asked to do or what its prompt says:
privilege escalation (`sudo`/`su`/`doas`), combined recursive+force delete
(`rm -rf` and equivalents), disk/partition tools, writing to a raw block
device, `shutdown`/`reboot`, fork bombs, piping a downloaded script into a
shell, and force-pushing git history (`--force-with-lease` is allowed). This
exists because a smaller, locally-run model acting autonomously across many
turns can't be relied on to follow prompt instructions alone — that's
necessary but not sufficient, so it's backed by a real code-level check. It's
pattern-matching on the command string, not a full shell parser or a
guarantee against every obfuscation — defense in depth alongside the
working-directory sandbox, not a substitute for choosing a capability tier
and working directory you're actually comfortable the subagent operating in.

## Feedback: `feedback_*`

Reports feedback about **this server** (bugs, friction, ideas) as a GitHub
issue — the same local-first, human-reviewed process as
[`uictl-mac-mcp`](../uictl-mac-mcp)'s `feedback` verb:

1. `feedback_create` drafts an entry (`issue` / `error` / `recommendation`,
   title, body) to `~/.lmstudio-mcp/feedback.json`. Nothing leaves this
   machine yet; list/get/update/delete it like any local record.
2. `feedback_submit` checks the title against this repo's existing GitHub
   issues first (`feedback_check_duplicates` runs this same check without
   submitting anything) — a likely match deletes the local draft instead of
   filing a duplicate.
3. If no duplicate: **never files the issue through the API.** When the
   connected MCP client supports elicitation, it asks a human to review
   (and optionally edit) the title/body first, then hands the client a
   pre-filled "new issue" page via a second, URL-mode elicitation rather
   than silently opening a browser tab. If the client doesn't support
   elicitation, it falls back to opening that page directly in this
   machine's default browser — a human still has to review and click
   "Create" there either way.

The duplicate check needs a GitHub token for a private repo: pass `token`,
set `GITHUB_TOKEN`, or have the `gh` CLI already authenticated — otherwise
it's skipped gracefully rather than blocking submission. `repo` on either
tool defaults to this server's own repo; pass `"owner/repo"` to target
another one.

## Prerequisites

- [LM Studio](https://lmstudio.ai/) running locally with its local server
  enabled (LM Studio → Developer tab → Start Server), on a version that
  exposes the native `/api/v1` REST API.
- Rust 1.80+ if building from source (see [Building](#building)) — the
  floor is `std::sync::LazyLock` (`src/subagent/guard.rs`), stabilized in
  1.80; CI builds against the latest stable toolchain rather than pinning
  or independently testing this exact minimum.

## Configuration

Read from the environment at connect time:

| Variable | Default | Description |
|----------|---------|--------------|
| `LMSTUDIO_HOST` | `127.0.0.1` | LM Studio host |
| `LMSTUDIO_PORT` | `1234` | LM Studio port |
| `LMSTUDIO_BASE_URL` | _(derived)_ | Full `scheme://host:port` override, e.g. `http://192.168.1.100:5678`. Takes precedence over host/port. |
| `LMSTUDIO_API_TOKEN` | _(none)_ | Bearer token, if LM Studio's server has API token auth enabled (Developer tab → Server Settings). Sent as `Authorization: Bearer <token>`. |
| `GITHUB_TOKEN` | _(none)_ | Used by `feedback_check_duplicates`/`feedback_submit` for a private repo, if no `token` argument is passed and `gh` isn't already authenticated. |

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
├── main.rs        # Entry point: logging, config, stdio transport, shutdown
├── server.rs      # Wires each tools::* function to an #[tool]-annotated method
├── client.rs      # HTTP client over LM Studio's native + OpenAI-compatible APIs
├── config.rs      # Environment-variable configuration
├── types.rs       # Shared ToolResult<T> envelope, ErrorCode, ClientError
├── sse.rs         # Server-Sent Events reader (idle-timeout streaming, see above)
├── feedback/
│   ├── store.rs   # Local JSON CRUD for feedback entries
│   └── github.rs  # Duplicate-issue check, token resolution
├── subagent/
│   ├── tools.rs   # Sandboxed read_file/list_directory/search_files/write_file/run_command
│   ├── guard.rs   # Non-bypassable high-risk command pattern guard
│   └── runner.rs  # The agentic tool-calling loop
└── tools/
    ├── health_check.rs
    ├── models.rs       # list_models, list_loaded_models, get_current_model,
    │                   # get_model_info, load_model, unload_model
    ├── chat.rs         # chat_completion, text_completion
    ├── embeddings.rs   # generate_embeddings
    ├── responses.rs    # create_response, start_conversation, continue_conversation
    ├── subagent.rs     # run_subagent (MCP wrapper around subagent::runner)
    └── feedback.rs     # feedback_* (MCP wrapper, incl. the elicitation flow)
```

Built on the official [`rmcp`](https://github.com/modelcontextprotocol/rust-sdk)
Rust MCP SDK (including its `elicitation` feature, for `feedback_submit`'s
human-review flow), `tokio` for async I/O, and `reqwest` with `rustls` (no
OpenSSL dependency, for easy cross-compilation and static-ish binaries on
all three platforms).

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
