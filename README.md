# lmstudio-rs-mcp

A single-binary [MCP](https://modelcontextprotocol.io) server that bridges Claude
(or any MCP client) to an LLM backend of your choice — [LM Studio](https://lmstudio.ai/)
or [Ollama](https://ollama.com/) running locally, or OpenAI/Anthropic in the
cloud — inference **and**, where the provider supports it, model management,
in one Rust binary that runs natively on macOS, Windows, and Linux.

Exactly one provider is active per server instance, selected with
`LLM_PROVIDER` (defaults to `lmstudio` — see [Configuration](#configuration)).
Every tool works the same way regardless of which one is configured;
inference tools (`chat_completion`, `run_subagent`, ...) work identically on
all four, while model-management tools and the stateful `/v1/responses`
tools are only meaningful for providers that actually have that concept (see
the [Tools](#tools) table) and return a clear "not supported by this
provider" result otherwise rather than erroring confusingly.

This project originally merged the functionality of two earlier, separate
LM-Studio-only MCP servers:

- **[LMStudio-MCP](../LMStudio-MCP)** (Python) — inference: chat/text
  completions, embeddings, and stateful conversations via LM Studio's
  OpenAI-compatible API.
- **[lm-studio-mcp-server](../lm-studio-mcp-server)** (TypeScript) — model
  management: list, load, unload, and inspect models via LM Studio's SDK.

Both capabilities were originally implemented over LM Studio's **native REST
API** (`/api/v1/...`) and OpenAI-compatible `/v1/...` surface — no WebSocket
SDK dependency and no Python/Node runtime to install, just one executable.
Ollama/OpenAI/Anthropic support (see below) was added on top of that same
foundation: Ollama's own native API (`/api/tags`, `/api/ps`, `/api/generate`)
is translated into the same shape LM Studio's native API returns, and
Anthropic's `/v1/messages` wire format is translated to/from the
OpenAI-compatible shape every tool already speaks — so none of the tools
below need to know which provider is actually active.

## Tools

| Tool | Works on | Description |
|------|----------|-------------|
| `health_check` | all | Verify the configured provider is reachable |
| `lmstudio_status` | LM Studio | Check whether LM Studio is installed (and its version), the `lms` CLI is available (and its build commit), and the REST API is reachable — each reported independently |
| `lms_cli` | LM Studio | Run an `lms` CLI command the REST API lacks: server start/stop/status, runtime engines, LM Link, `get` (download), `import`, `clone`, `whoami`/`logout` (see below) |
| `list_models` | all | List models available from the configured provider — the local library for LM Studio/Ollama, or every model the API key can use for OpenAI/Anthropic |
| `list_loaded_models` | LM Studio, Ollama | List currently loaded model instances |
| `get_current_model` | LM Studio, Ollama | Identify the loaded model |
| `get_model_info` | LM Studio, Ollama | Detailed info for one loaded instance |
| `load_model` | LM Studio, Ollama | Load a model into memory |
| `unload_model` | LM Studio, Ollama | Unload a model instance |
| `chat_completion` | all | Chat-formatted completion |
| `text_completion` | LM Studio, Ollama, OpenAI | Raw/non-chat completion (faster, for code/continuation) |
| `generate_embeddings` | all except Anthropic | Vector embeddings for RAG/semantic search |
| `create_response` | LM Studio, OpenAI | Stateful response via a `/v1/responses`-style endpoint |
| `start_conversation` | LM Studio, OpenAI | Begin a multi-turn session with a locked-in system prompt |
| `continue_conversation` | LM Studio, OpenAI | Continue a session started above |
| `run_subagent` | all | Delegate a task to a model acting as a sandboxed subagent |
| `feedback_create` / `_list` / `_get` / `_update` / `_delete` | all | Draft/manage local feedback entries about this server |
| `feedback_check_duplicates` / `feedback_submit` | all | Check and submit a feedback entry as a GitHub issue |

Model-management and `/v1/responses` tools on a provider that doesn't
support them return a normal `ToolResult` explaining that (success for a
read like `list_loaded_models`, a clear error for a mutation like
`load_model`) rather than a confusing raw HTTP error.

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
request server-side streaming internally and reassemble the full response
here — the MCP tool call itself is still a single request/response, this is
purely a transport-level choice. The reason: a non-streaming call has to be
timed out against its *total* duration, and no duration is both short enough
to catch a genuinely hung connection and long enough for a slow reasoning
model's legitimate output. Streaming turns that into an *idle* timeout
instead (reset on every chunk that actually arrives, 60s default) — a model
generating steadily for minutes is never killed, and a stalled connection is
still caught quickly. See `src/sse.rs`. This applies identically whether the
stream is OpenAI-shaped (LM Studio/Ollama/OpenAI) or Anthropic-shaped — both
are reassembled into the same unified result in `src/client.rs`.

### Reasoning models

A reasoning/thinking model (e.g. Qwen3) can spend an entire `max_tokens`
budget on its internal deliberation and produce no final answer. Rather than
reporting that as a bare failure, affected tools return it as a *success*
with the reasoning trace included in `reasoning_content` and a message
explaining what happened — so you can see why and retry with a larger
budget instead of getting an opaque error.

## LM Studio environment: `lmstudio_status` and `lms_cli`

`lmstudio_status` probes installation, the `lms` CLI, and the API
separately, so it still gives a useful answer when LM Studio is missing or
not running. The CLI has no semantic version, so its version is the build
commit; the desktop app version is read from the app bundle on macOS only.

`lms_cli` takes a `command` from a fixed allowlist plus optional `args`:

| `command` | Runs |
|-----------|------|
| `server_start` / `server_stop` / `server_status` | `lms server start` / `stop` / `status` |
| `runtime_ls` / `runtime_select` / `runtime_update` / `runtime_get` / `runtime_survey` | `lms runtime ...` |
| `link_status` / `link_enable` / `link_disable` / `link_set_device_name` / `link_set_preferred_device` | `lms link ...` |
| `get` / `import` / `clone` | `lms get -y` / `import -y` / `clone` |
| `whoami` / `logout` | `lms whoami` / `logout` |

Model listing, loading and unloading are not here — use `list_models`,
`load_model` and the rest. `runtime remove` is excluded as destructive, and `chat`, `log stream`,
`dev` and `push` are excluded too (interactive, unbounded, or publishing). The CLI is
spawned directly (no shell), with stdin closed so it can't wait on a prompt,
a timeout (`timeout_seconds`, default 120, max 3600 — raise it for
downloads), and output capped at 64 KiB per stream.

Commands that would otherwise open an interactive picker or fail
(`runtime_select`, `runtime_get`, `link_set_device_name`,
`link_set_preferred_device`, `get`, `import`, `clone`) require at least one
entry in `args` and are rejected up front without it — e.g.
`runtime_select` with `["--latest"]` or an alias from `runtime_ls`, and
`link_set_preferred_device` with a device identifier (e.g.
`["d18d35beec0f1736dfc5164d8d97b7e4"]`), and `runtime_update` with `["--all"]` to update every installed extension, not
just the selected ones.

The CLI is found via `LMS_PATH` (explicit override), then `PATH`, then
`~/.lmstudio/bin`.

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
   title, body) to `~/.lmstudio-rs-mcp/feedback.json`. Nothing leaves this
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

Depending on which provider you configure (see [Configuration](#configuration)):

- **LM Studio** (default) — [LM Studio](https://lmstudio.ai/) running
  locally with its local server enabled (LM Studio → Developer tab → Start
  Server), on a version that exposes the native `/api/v1` REST API.
- **Ollama** — [Ollama](https://ollama.com/) running locally (`ollama
  serve`, or already running as a background service).
- **OpenAI** / **Anthropic** — an API key with available credit/quota.
  Nothing else to install; these are plain HTTPS calls to the provider's API.
- Rust 1.80+ if building from source (see [Building](#building)) — the
  floor is `std::sync::LazyLock` (`src/subagent/guard.rs`), stabilized in
  1.80; CI builds against the latest stable toolchain rather than pinning
  or independently testing this exact minimum.

## Configuration

Read from the environment at connect time:

| Variable | Default | Description |
|----------|---------|--------------|
| `LLM_PROVIDER` | `lmstudio` | One of `lmstudio`, `ollama`, `openai`, `anthropic` (`claude` also accepted). Unknown values fall back to `lmstudio` with a warning. |
| `LLM_BASE_URL` | provider default | Full `scheme://host:port` override. Provider defaults: `http://127.0.0.1:1234` (lmstudio), `http://127.0.0.1:11434` (ollama), `https://api.openai.com` (openai), `https://api.anthropic.com` (anthropic). |
| `LLM_API_KEY` | _(none)_ | API key / bearer token. **Required** for `openai` and `anthropic` (the server refuses to start without one); optional for `lmstudio`/`ollama`, if you've turned on local API auth. |
| `LMS_PATH` | _(auto-detected)_ | Full path to the `lms` binary, for `lmstudio_status`/`lms_cli`. Otherwise searched on `PATH`, then `~/.lmstudio/bin`. |
| `GITHUB_TOKEN` | _(none)_ | Used by `feedback_check_duplicates`/`feedback_submit` for a private repo, if no `token` argument is passed and `gh` isn't already authenticated. |

Anthropic authenticates with `x-api-key` + a required `anthropic-version`
header rather than `Authorization: Bearer`; this is handled internally based
on `LLM_PROVIDER` — `LLM_API_KEY` is still just your plain Anthropic API key.

### Existing LM Studio configs keep working

If you were already using `LMSTUDIO_HOST` / `LMSTUDIO_PORT` /
`LMSTUDIO_BASE_URL` / `LMSTUDIO_API_TOKEN`, nothing changes — they're still
read exactly as before whenever `LLM_PROVIDER` is unset or `lmstudio`, and
take precedence over the new variables' defaults for that provider only.
There's no need to migrate an existing config; `LLM_PROVIDER`/`LLM_BASE_URL`/
`LLM_API_KEY` are additive, for selecting a *different* provider or using the
new generic names going forward.

### Examples

Ollama on the default local port:
```json
{ "env": { "LLM_PROVIDER": "ollama" } }
```

OpenAI:
```json
{ "env": { "LLM_PROVIDER": "openai", "LLM_API_KEY": "sk-..." } }
```

Anthropic:
```json
{ "env": { "LLM_PROVIDER": "anthropic", "LLM_API_KEY": "sk-ant-..." } }
```

## Building

```bash
git clone https://github.com/ByronScottJones/lmstudio-rs-mcp.git
cd lmstudio-rs-mcp
cargo build --release
```

The binary is produced at `target/release/lmstudio-rs-mcp` (`.exe` on Windows).
Pre-built binaries for macOS (arm64/x86_64), Windows, and Linux are also
published as CI artifacts on every push — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## MCP Client Configuration

### Claude Code

```bash
claude mcp add lmstudio -- /path/to/lmstudio-rs-mcp
```

Or add directly to `.mcp.json` / `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "lmstudio": {
      "command": "/path/to/lmstudio-rs-mcp",
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
      "command": "C:\\path\\to\\lmstudio-rs-mcp.exe"
    }
  }
}
```

### Connecting to a backend on another machine

```json
{
  "mcpServers": {
    "lmstudio": {
      "command": "/path/to/lmstudio-rs-mcp",
      "env": {
        "LLM_BASE_URL": "http://192.168.1.100:1234"
      }
    }
  }
}
```

(`LMSTUDIO_BASE_URL` still works identically for the default `lmstudio`
provider — see [Configuration](#configuration).)

## Architecture

```
src/
├── main.rs        # Entry point: logging, config, stdio transport, shutdown
├── server.rs      # Wires each tools::* function to an #[tool]-annotated method
├── client.rs      # HTTP client over every provider's API, with Ollama/Anthropic translation
├── providers.rs   # The Provider enum: capability matrix per backend
├── config.rs      # Environment-variable configuration (provider, base URL, API key)
├── types.rs       # Shared ToolResult<T> envelope, ErrorCode, ClientError
├── lms.rs         # Locating/running the `lms` CLI, install detection, command allowlist
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
    ├── lms.rs          # lmstudio_status, lms_cli
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

### A note on native/translated API shapes

LM Studio's `/api/v1` model-management endpoints are newer and less
exhaustively documented than the stable OpenAI-compatible surface. Response
structs in `src/client.rs` are typed against LM Studio's published API
reference, but every field is optional with `#[serde(default)]` and the
top-level model entry keeps a `#[serde(flatten)] extra` catch-all — so a
field LM Studio adds, renames, or omits in a future version degrades
gracefully (parses with that field missing/extra) rather than failing to
parse at all. Ollama's native API (`/api/tags`, `/api/ps`, `/api/generate`)
is translated into this same shape, and Anthropic's `/v1/messages`
request/response/streaming shape is translated to/from the OpenAI-compatible
one every tool speaks — both translations, and every provider-specific
quirk they exist to paper over, are isolated to `src/client.rs`.

## Development

```bash
cargo fmt --all           # format
cargo clippy --all-targets -- -D warnings   # lint
cargo test                 # unit tests
cargo build --release      # release binary
```

## License

MIT

## AI Notice

I am a software engineer with nearly 50 years of experience. I now use AI tools to assist me with development. For this project, I have used Claude.ai and Anthropic GPT 5.5. After each feature update, I reviewed the application to ensure that it was working as I intended. As someone with Frontotemporal Dementia, AI tools are literally saving my career, and allowing me to continue working longer in a field that I love. If you don't like AI, you are welcome to not use my fork of this application. If you do not like that I use AI, tough.

In the words of Bender:

![Bender](assets/futurama-bender-bite-my-shiny-metal-ass.gif "Bender - Bite My Shiny Metal Ass")
