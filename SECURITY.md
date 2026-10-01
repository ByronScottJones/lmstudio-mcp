# Security

## Scope and threat model

`lmstudio-rs-mcp` runs as a local MCP server with no network exposure by
itself (it talks stdio to its MCP client and HTTP to a local — or
explicitly configured remote — LM Studio instance). The main things worth
knowing:

- **`run_subagent` with `read_write_shell` capability executes real shell
  commands**, chosen autonomously by whichever local model is driving it.
  `src/subagent/guard.rs` blocks a fixed set of high-risk command patterns
  (privilege escalation, recursive force-delete, disk/partition tools,
  shutdown, piping a remote script into a shell, force-push), but this is
  pattern-matching on a command string, not a full shell parser or a
  sandbox — it is defense in depth, not a guarantee. Only grant this tier
  for a working directory and task you'd be comfortable with the underlying
  model acting in directly.
- **Filesystem tools are sandboxed to a working directory** you pass per
  call (path traversal, absolute-path escapes, and symlink escapes are all
  rejected — see `sandboxed_path` in `src/subagent/tools.rs`), but that
  sandbox is only as strong as the directory you choose.
- **`LMSTUDIO_API_TOKEN` / `GITHUB_TOKEN`** are read from the environment
  and sent as bearer tokens over HTTPS (GitHub) or whatever scheme
  `LMSTUDIO_BASE_URL` specifies (plain HTTP by default, for localhost).
  Don't point `LMSTUDIO_BASE_URL` at a remote host over plain HTTP if the
  token matters.
- **`feedback_submit` never files a GitHub issue through the API** —
  it always ends at a human reviewing and clicking "Create" on a
  pre-filled GitHub page, specifically so an agent calling this tool can't
  silently publish something.

## Reporting a vulnerability

Please use GitHub's [private vulnerability
reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
for this repo (Security tab → "Report a vulnerability") rather than a
public issue, so it can be addressed before details are public.
