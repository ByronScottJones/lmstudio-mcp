//! The tools a subagent (a local LM Studio model, driven by
//! `subagent::runner`) can call: filesystem access sandboxed to a working
//! directory, and optionally shell commands. Capability tiers, from
//! safest to most powerful:
//!
//! - `ReadOnly`: `read_file`, `list_directory`, `search_files`
//! - `ReadWrite`: adds `write_file`
//! - `ReadWriteShell`: adds `run_command`, gated additionally by
//!   [`super::guard`]
//!
//! Every filesystem path a subagent provides is resolved against, and
//! checked to stay within, the working directory — see [`sandboxed_path`].

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use tokio::io::AsyncReadExt;

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde::Deserialize,
    serde::Serialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
// The shared "Read" prefix is deliberate — these name an increasing tier
// (read-only, read+write, read+write+shell), not three unrelated things
// that happen to start the same way; splitting it would read worse.
#[allow(clippy::enum_variant_names)]
pub enum Capability {
    ReadOnly,
    ReadWrite,
    ReadWriteShell,
}

/// Largest file content a single `read_file`/`write_file` call will
/// transfer, and the cap on `run_command` output — generous for real work,
/// but bounded so one call can't blow up the subagent's own context budget
/// or this process's memory.
const MAX_IO_BYTES: usize = 200_000;
const COMMAND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

pub struct SubagentContext {
    /// Canonicalized, so every sandboxing check below has a stable root to
    /// compare against.
    pub working_directory: PathBuf,
    pub capability: Capability,
}

impl SubagentContext {
    pub fn new(working_directory: &Path, capability: Capability) -> std::io::Result<Self> {
        Ok(Self {
            working_directory: std::fs::canonicalize(working_directory)?,
            capability,
        })
    }
}

/// Resolve a subagent-supplied path against the working directory, lexically
/// normalizing `.`/`..` first, then re-checking containment against the
/// canonicalized form of whatever on-disk prefix of it actually exists —
/// which also catches a symlink inside the working directory pointing back
/// out of it. Rejects anything that still doesn't stay under the root.
fn sandboxed_path(ctx: &SubagentContext, rel: &str) -> Result<PathBuf, String> {
    let rel = rel.trim();
    if rel.is_empty() {
        return Err("path must not be empty".to_string());
    }
    let candidate = if Path::new(rel).is_absolute() {
        PathBuf::from(rel)
    } else {
        ctx.working_directory.join(rel)
    };
    let normalized = normalize_lexically(&candidate);
    if !normalized.starts_with(&ctx.working_directory) {
        return Err(format!(
            "path escapes the working directory ({}): {rel}",
            ctx.working_directory.display()
        ));
    }

    // Re-check against the canonical form of the nearest existing ancestor,
    // to catch a symlink that resolves outside the sandbox even though the
    // lexical path looked fine.
    let mut probe = normalized.clone();
    while !probe.exists() {
        match probe.parent() {
            Some(p) => probe = p.to_path_buf(),
            None => break,
        }
    }
    if let Ok(canonical_ancestor) = std::fs::canonicalize(&probe) {
        if !canonical_ancestor.starts_with(&ctx.working_directory) {
            return Err(format!(
                "path escapes the working directory via a symlink: {rel}"
            ));
        }
    }

    Ok(normalized)
}

fn normalize_lexically(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn truncate_for_output(mut s: String, label: &str) -> String {
    if s.len() > MAX_IO_BYTES {
        // `String::truncate` panics unless the cut point lands on a char
        // boundary; walk back from the byte limit to the nearest one
        // rather than assuming MAX_IO_BYTES itself lands cleanly.
        let mut cut = MAX_IO_BYTES;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push_str(&format!(
            "\n\n...[{label} truncated at {MAX_IO_BYTES} bytes]"
        ));
    }
    s
}

/// OpenAI-style `tools` array entries for whichever tools `capability`
/// grants — passed as-is in the subagent's chat-completions request body.
pub fn tool_definitions(capability: Capability) -> Vec<Value> {
    let mut defs = vec![
        json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file. Path is relative to the working directory (or an absolute path inside it).",
                "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List the entries (name, type, size in bytes) of a directory. Path is relative to the working directory (or an absolute path inside it); omit for the working directory root.",
                "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string" } }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Case-insensitive literal text search across files under a directory (like a simple grep). Returns matching file:line:text for up to 200 matches.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Literal text to search for." },
                        "path": { "type": "string", "description": "Directory to search under. Defaults to the working directory root." }
                    },
                    "required": ["query"]
                }
            }
        }),
    ];

    if capability >= Capability::ReadWrite {
        defs.push(json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a UTF-8 text file. Path is relative to the working directory (or an absolute path inside it); the parent directory must already exist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" }
                    },
                    "required": ["path", "content"]
                }
            }
        }));
    }

    if capability >= Capability::ReadWriteShell {
        defs.push(json!({
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command in the working directory. A fixed set of high-risk patterns (sudo, recursive force-delete, disk/partition tools, shutdown, piping a remote script into a shell, force-pushing git history, and similar) is always blocked regardless of what you're asked to do. Avoid them; use narrower, specific commands instead.",
                "parameters": {
                    "type": "object",
                    "properties": { "command": { "type": "string" } },
                    "required": ["command"]
                }
            }
        }));
    }

    defs
}

/// Execute one tool call by name. `Err` becomes the tool-result content
/// sent back to the subagent model (as a normal tool failure it can see and
/// adapt to), not a hard stop of the agentic loop — the same way a shell
/// command returning a non-zero exit is routine feedback, not a crash.
pub async fn execute(
    ctx: &SubagentContext,
    name: &str,
    arguments: &Value,
) -> Result<String, String> {
    match name {
        "read_file" => read_file(ctx, arguments).await,
        "list_directory" => list_directory(ctx, arguments).await,
        "search_files" => search_files(ctx, arguments).await,
        "write_file" if ctx.capability >= Capability::ReadWrite => write_file(ctx, arguments).await,
        "run_command" if ctx.capability >= Capability::ReadWriteShell => {
            run_command(ctx, arguments).await
        }
        "write_file" | "run_command" => Err(format!(
            "tool '{name}' is not available at this subagent's capability tier ({:?})",
            ctx.capability
        )),
        other => Err(format!("unknown tool '{other}'")),
    }
}

fn arg_str<'a>(arguments: &'a Value, key: &str) -> Result<&'a str, String> {
    arguments
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("missing or non-string '{key}' argument"))
}

async fn read_file(ctx: &SubagentContext, arguments: &Value) -> Result<String, String> {
    let path = sandboxed_path(ctx, arg_str(arguments, "path")?)?;

    // Cap bytes actually read from disk at MAX_IO_BYTES, rather than
    // reading the whole file and truncating the resulting String — a
    // large file (gigabytes) would otherwise be fully loaded into memory
    // first, defeating the point of the limit.
    let mut file = tokio::fs::File::open(&path)
        .await
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let file_size = file.metadata().await.map(|m| m.len()).unwrap_or(0);

    let mut buf = Vec::with_capacity((file_size as usize).min(MAX_IO_BYTES) + 1);
    (&mut file)
        .take(MAX_IO_BYTES as u64)
        .read_to_end(&mut buf)
        .await
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    // `from_utf8_lossy` rather than `String::from_utf8` — a cut made at an
    // arbitrary byte offset (when the file is larger than the limit) can
    // land mid-character; replace it with U+FFFD instead of failing the
    // whole read over the last few bytes.
    let content = String::from_utf8_lossy(&buf).into_owned();
    if file_size as usize > MAX_IO_BYTES {
        Ok(format!(
            "{content}\n\n...[file content truncated at {MAX_IO_BYTES} bytes (file is {file_size} bytes)]"
        ))
    } else {
        Ok(content)
    }
}

async fn list_directory(ctx: &SubagentContext, arguments: &Value) -> Result<String, String> {
    let rel = arguments
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");
    let path = sandboxed_path(ctx, rel)?;
    let mut entries = tokio::fs::read_dir(&path)
        .await
        .map_err(|e| format!("failed to list {}: {e}", path.display()))?;

    let mut lines = Vec::new();
    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|e| format!("failed to read directory entry: {e}"))?
    {
        let meta = entry.metadata().await.ok();
        let kind = match &meta {
            Some(m) if m.is_dir() => "dir",
            Some(m) if m.is_symlink() => "symlink",
            Some(_) => "file",
            None => "unknown",
        };
        let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
        lines.push(format!(
            "{:<5} {:>10}  {}",
            kind,
            size,
            entry.file_name().to_string_lossy()
        ));
    }
    lines.sort();
    if lines.is_empty() {
        Ok("(empty directory)".to_string())
    } else {
        Ok(lines.join("\n"))
    }
}

async fn search_files(ctx: &SubagentContext, arguments: &Value) -> Result<String, String> {
    let query = arg_str(arguments, "query")?.to_lowercase();
    let rel = arguments
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");
    let root = sandboxed_path(ctx, rel)?;

    const MAX_MATCHES: usize = 200;
    const MAX_FILES_SCANNED: usize = 5_000;
    let mut matches = Vec::new();
    let mut files_scanned = 0usize;
    let mut stack = vec![root.clone()];

    while let Some(dir) = stack.pop() {
        if matches.len() >= MAX_MATCHES || files_scanned >= MAX_FILES_SCANNED {
            break;
        }
        let Ok(mut entries) = tokio::fs::read_dir(&dir).await else {
            continue;
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            if matches.len() >= MAX_MATCHES || files_scanned >= MAX_FILES_SCANNED {
                break;
            }
            let path = entry.path();
            let Ok(meta) = entry.metadata().await else {
                continue;
            };
            // `DirEntry::metadata()` reports the entry itself (lstat-like —
            // it does not follow a final symlink), so this is reachable for
            // a symlink and both `is_dir()`/`is_file()` below are false for
            // one either way; skip explicitly rather than relying on that
            // as an accidental side effect, so a symlink can never be used
            // to read or recurse outside `working_directory`.
            if meta.file_type().is_symlink() {
                continue;
            }
            if meta.is_dir() {
                // Skip the usual noisy directories that are never worth scanning.
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if !matches!(name.as_ref(), ".git" | "node_modules" | "target" | ".venv") {
                    stack.push(path);
                }
                continue;
            }
            if !meta.is_file() || meta.len() as usize > MAX_IO_BYTES {
                continue;
            }
            files_scanned += 1;
            let Ok(content) = tokio::fs::read_to_string(&path).await else {
                continue; // binary or non-UTF8 — skip rather than fail the whole search
            };
            let rel_display = path.strip_prefix(&ctx.working_directory).unwrap_or(&path);
            for (i, line) in content.lines().enumerate() {
                if line.to_lowercase().contains(&query) {
                    matches.push(format!(
                        "{}:{}: {}",
                        rel_display.display(),
                        i + 1,
                        line.trim()
                    ));
                    if matches.len() >= MAX_MATCHES {
                        break;
                    }
                }
            }
        }
    }

    if matches.is_empty() {
        Ok("no matches".to_string())
    } else {
        Ok(matches.join("\n"))
    }
}

async fn write_file(ctx: &SubagentContext, arguments: &Value) -> Result<String, String> {
    let path = sandboxed_path(ctx, arg_str(arguments, "path")?)?;
    let content = arg_str(arguments, "content")?;
    if content.len() > MAX_IO_BYTES {
        return Err(format!(
            "content is {} bytes, over the {MAX_IO_BYTES}-byte limit for a single write_file call",
            content.len()
        ));
    }
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() && !parent.exists() => {
            return Err(format!(
                "parent directory does not exist: {} (this tool won't create directories — use run_command if you have shell access, or write_file a path whose parent already exists)",
                parent.display()
            ));
        }
        _ => {}
    }
    tokio::fs::write(&path, content)
        .await
        .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
    Ok(format!(
        "wrote {} bytes to {}",
        content.len(),
        path.display()
    ))
}

async fn run_command(ctx: &SubagentContext, arguments: &Value) -> Result<String, String> {
    let command = arg_str(arguments, "command")?;
    super::guard::check(command)?;
    run_shell_command(&ctx.working_directory, command, COMMAND_TIMEOUT).await
}

/// Runs `command` in a shell under `cwd`, bounded by `timeout`. Split out
/// from [`run_command`] (which always passes [`COMMAND_TIMEOUT`]) so tests
/// can exercise the timeout/kill behavior without waiting out the real
/// production timeout.
async fn run_shell_command(
    cwd: &Path,
    command: &str,
    timeout: std::time::Duration,
) -> Result<String, String> {
    let mut cmd = if cfg!(windows) {
        let mut c = tokio::process::Command::new("cmd");
        c.args(["/C", command]);
        c
    } else {
        let mut c = tokio::process::Command::new("sh");
        c.args(["-c", command]);
        c
    };
    cmd.current_dir(cwd);
    cmd.stdin(std::process::Stdio::null());
    // Without this, the `tokio::time::timeout` below only stops *waiting*
    // on the child when it fires — the process itself keeps running
    // unsupervised. This makes dropping the in-flight `cmd.output()`
    // future (which owns the spawned `Child`) actually kill it.
    cmd.kill_on_drop(true);

    let output = tokio::time::timeout(timeout, cmd.output())
        .await
        .map_err(|_| format!("command timed out after {timeout:?}"))?
        .map_err(|e| format!("failed to run command: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let status = output
        .status
        .code()
        .map(|c| c.to_string())
        .unwrap_or_else(|| "signal".to_string());

    let combined =
        format!("exit status: {status}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
    Ok(truncate_for_output(combined, "command output"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lmstudio-mcp-subagent-test-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn ctx(dir: &Path, cap: Capability) -> SubagentContext {
        SubagentContext::new(dir, cap).unwrap()
    }

    #[test]
    fn truncate_for_output_does_not_panic_on_a_multibyte_boundary() {
        // Regression: `String::truncate` panics unless the cut point is a
        // char boundary. "€" is 3 bytes/char, and MAX_IO_BYTES (200_000)
        // isn't a multiple of 3, so a naive cut at that byte offset is
        // guaranteed to land mid-character.
        assert_ne!(
            MAX_IO_BYTES % 3,
            0,
            "test assumption: cutoff must not be 3-byte-aligned"
        );
        let s = "€".repeat(MAX_IO_BYTES / 3 + 10);
        let out = truncate_for_output(s, "test"); // must not panic
        assert!(out.contains("truncated"));
    }

    #[test]
    fn capability_ordering() {
        assert!(Capability::ReadOnly < Capability::ReadWrite);
        assert!(Capability::ReadWrite < Capability::ReadWriteShell);
    }

    #[tokio::test]
    async fn rejects_path_traversal_outside_working_dir() {
        let dir = temp_dir("traversal");
        let c = ctx(&dir, Capability::ReadOnly);
        let err = sandboxed_path(&c, "../../etc/passwd").unwrap_err();
        assert!(err.contains("escapes the working directory"));
    }

    #[tokio::test]
    async fn rejects_absolute_path_outside_working_dir() {
        let dir = temp_dir("absolute");
        let c = ctx(&dir, Capability::ReadOnly);
        let err = sandboxed_path(&c, "/etc/passwd").unwrap_err();
        assert!(err.contains("escapes the working directory"));
    }

    #[tokio::test]
    async fn allows_path_inside_working_dir() {
        let dir = temp_dir("inside");
        std::fs::write(dir.join("a.txt"), "hello").unwrap();
        let c = ctx(&dir, Capability::ReadOnly);
        let resolved = sandboxed_path(&c, "a.txt").unwrap();
        assert_eq!(resolved, dir.canonicalize().unwrap().join("a.txt"));
    }

    #[tokio::test]
    async fn read_file_round_trips_write_file() {
        let dir = temp_dir("rw");
        let c = ctx(&dir, Capability::ReadWrite);
        let write_result = execute(
            &c,
            "write_file",
            &json!({"path": "note.txt", "content": "hi there"}),
        )
        .await
        .unwrap();
        assert!(write_result.contains("wrote"));
        let read_back = execute(&c, "read_file", &json!({"path": "note.txt"}))
            .await
            .unwrap();
        assert_eq!(read_back, "hi there");
    }

    #[tokio::test]
    async fn read_file_caps_output_at_max_io_bytes_for_an_oversized_file() {
        // Regression: this used to read the whole file into memory (via
        // `read_to_string`) before truncating the resulting String — a
        // large file would be fully buffered regardless of the limit.
        // `read_file` now caps the actual disk read itself (`.take(...)`),
        // so this also checks the byte count read back is bounded, not
        // just that the *output string* happens to look truncated.
        let dir = temp_dir("read-cap");
        let big = "x".repeat(MAX_IO_BYTES + 50_000);
        std::fs::write(dir.join("big.txt"), &big).unwrap();

        let c = ctx(&dir, Capability::ReadOnly);
        let out = execute(&c, "read_file", &json!({"path": "big.txt"}))
            .await
            .unwrap();

        assert!(out.contains("truncated"));
        assert!(out.contains(&format!("file is {} bytes", big.len())));
        // The content portion (before the appended truncation note) must
        // not exceed what was actually read from disk.
        let content_len = out.find("\n\n...[file content truncated").unwrap();
        assert!(content_len <= MAX_IO_BYTES);
    }

    #[tokio::test]
    async fn write_file_denied_at_read_only_tier() {
        let dir = temp_dir("readonly-write");
        let c = ctx(&dir, Capability::ReadOnly);
        let err = execute(&c, "write_file", &json!({"path": "x.txt", "content": "x"}))
            .await
            .unwrap_err();
        assert!(err.contains("not available"));
    }

    #[tokio::test]
    async fn run_command_denied_below_shell_tier() {
        let dir = temp_dir("no-shell");
        let c = ctx(&dir, Capability::ReadWrite);
        let err = execute(&c, "run_command", &json!({"command": "echo hi"}))
            .await
            .unwrap_err();
        assert!(err.contains("not available"));
    }

    #[tokio::test]
    async fn run_command_executes_and_captures_stdout() {
        let dir = temp_dir("shell-echo");
        let c = ctx(&dir, Capability::ReadWriteShell);
        let out = execute(
            &c,
            "run_command",
            &json!({"command": "echo hello-subagent"}),
        )
        .await
        .unwrap();
        assert!(out.contains("hello-subagent"));
        assert!(out.contains("exit status: 0"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timed_out_command_is_actually_killed_not_left_running() {
        // Regression: wrapping `cmd.output()` in `tokio::time::timeout`
        // without `kill_on_drop(true)` stops *waiting* on the child when
        // the timeout fires, but leaves the process itself running
        // unsupervised. Prove the fix by timing out a command that would,
        // if left alive, write a marker file after the test has already
        // moved on — and confirming that file never appears.
        let dir = temp_dir("kill-on-timeout");
        let marker = dir.join("marker.txt");
        let command = format!("sleep 2 && touch {}", marker.display());

        let result = run_shell_command(&dir, &command, std::time::Duration::from_millis(200)).await;
        assert!(result.is_err(), "expected a timeout error");

        // Give the (correctly killed) process more than enough time to have
        // written the marker if it were still alive, then confirm it never did.
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        assert!(
            !marker.exists(),
            "process kept running past the timeout and wrote its marker file"
        );
    }

    #[tokio::test]
    async fn run_command_blocks_guarded_patterns() {
        let dir = temp_dir("shell-guard");
        let c = ctx(&dir, Capability::ReadWriteShell);
        let err = execute(&c, "run_command", &json!({"command": "sudo rm -rf /"}))
            .await
            .unwrap_err();
        assert!(err.contains("guard"));
    }

    #[tokio::test]
    async fn list_directory_lists_entries() {
        let dir = temp_dir("list");
        std::fs::write(dir.join("a.txt"), "x").unwrap();
        std::fs::create_dir(dir.join("sub")).unwrap();
        let c = ctx(&dir, Capability::ReadOnly);
        let out = execute(&c, "list_directory", &json!({})).await.unwrap();
        assert!(out.contains("a.txt"));
        assert!(out.contains("sub"));
    }

    #[tokio::test]
    async fn search_files_finds_matches() {
        let dir = temp_dir("search");
        std::fs::write(dir.join("a.txt"), "hello world\nfoo bar\n").unwrap();
        std::fs::write(dir.join("b.txt"), "nothing interesting\n").unwrap();
        let c = ctx(&dir, Capability::ReadOnly);
        let out = execute(&c, "search_files", &json!({"query": "hello"}))
            .await
            .unwrap();
        assert!(out.contains("a.txt:1"));
        assert!(!out.contains("b.txt"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn search_files_does_not_follow_a_symlink_outside_the_working_directory() {
        let dir = temp_dir("search-symlink-in");
        let outside = temp_dir("search-symlink-out");
        std::fs::write(outside.join("secret.txt"), "the-secret-needle\n").unwrap();
        std::os::unix::fs::symlink(&outside, dir.join("escape")).unwrap();

        let c = ctx(&dir, Capability::ReadOnly);
        let out = execute(&c, "search_files", &json!({"query": "the-secret-needle"}))
            .await
            .unwrap();
        assert_eq!(
            out, "no matches",
            "search_files must not follow a symlink out of the sandbox"
        );
    }

    #[tokio::test]
    async fn write_file_refuses_missing_parent_directory() {
        let dir = temp_dir("missing-parent");
        let c = ctx(&dir, Capability::ReadWrite);
        let err = execute(
            &c,
            "write_file",
            &json!({"path": "nosuchdir/x.txt", "content": "x"}),
        )
        .await
        .unwrap_err();
        assert!(err.contains("does not exist"));
    }
}
