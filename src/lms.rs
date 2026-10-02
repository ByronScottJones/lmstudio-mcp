//! Locating and running the `lms` CLI that ships with LM Studio, plus
//! detection of the LM Studio installation itself.
//!
//! The CLI is only ever spawned directly with an argument vector — never
//! through a shell — so caller-supplied arguments can't be reinterpreted as
//! shell syntax. Which subcommands may be run at all is decided by
//! [`LmsCommand`], not by the caller.

use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::LazyLock;
use std::time::Duration;
use tokio::process::Command;

/// Per-stream cap on captured CLI output, so a chatty command (e.g. a model
/// download's progress output) can't flood the MCP response.
const MAX_OUTPUT_BYTES: usize = 64 * 1024;

static ANSI_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\x1b\[[0-9;?]*[ -/]*[@-~]").expect("static regex is valid"));
static CLI_COMMIT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"CLI commit:\s*([0-9a-fA-F]+)").expect("static regex is valid"));
static PLIST_VERSION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<key>CFBundleShortVersionString</key>\s*<string>([^<]+)</string>")
        .expect("static regex is valid")
});

/// The `lms` subcommands this server will run on a caller's behalf.
///
/// Deliberately limited to what the LM Studio REST API can't already do:
/// `ls`/`ps`/`load`/`unload` are covered by `list_models`,
/// `list_loaded_models`, `load_model` and `unload_model`, and `chat`,
/// `log stream`, `dev` and `push` are interactive, unbounded, or publish
/// outward, so they're not offered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LmsCommand {
    /// `lms server start` — start the local API server.
    ServerStart,
    /// `lms server stop` — stop the local API server.
    ServerStop,
    /// `lms server status` — whether the local API server is running, and its port.
    ServerStatus,
    /// `lms runtime ls` — list installed inference engines.
    RuntimeLs,
    /// `lms runtime select <alias>` or `--latest` — select an installed engine.
    RuntimeSelect,
    /// `lms runtime remove <engine>` — remove an installed engine.
    RuntimeRemove,
    /// `lms runtime update` — update selected runtime extensions (`--all` for every installed one).
    RuntimeUpdate,
    /// `lms runtime get <name>` — download a runtime extension (a name is required: without one the CLI opens an interactive picker).
    RuntimeGet,
    /// `lms runtime survey` — survey the GPU/CPU/RAM available to the selected engines.
    RuntimeSurvey,
    /// `lms link status` — LM Link status and discovered devices.
    LinkStatus,
    /// `lms link enable` — enable LM Link on this device.
    LinkEnable,
    /// `lms link disable` — disable LM Link on this device.
    LinkDisable,
    /// `lms link set-device-name <name>` — rename this LM Link device.
    LinkSetDeviceName,
    /// `lms link set-preferred-device <device>` — set the preferred LM Link device (without an argument the CLI opens an interactive picker).
    LinkSetPreferredDevice,
    /// `lms get <name>` — search for and download a model or Hub artifact (always passes `-y`).
    Get,
    /// `lms import <file-path>` — import a model file into LM Studio (always passes `-y`).
    Import,
    /// `lms clone <owner/name> [path]` — clone a Hub artifact to a local folder.
    Clone,
    /// `lms whoami` — current LM Studio authentication status.
    Whoami,
    /// `lms logout` — log out of LM Studio.
    Logout,
}

impl LmsCommand {
    /// The fixed leading arguments for this command.
    pub fn base_args(self) -> &'static [&'static str] {
        match self {
            Self::ServerStart => &["server", "start"],
            Self::ServerStop => &["server", "stop"],
            Self::ServerStatus => &["server", "status", "--json"],
            Self::RuntimeLs => &["runtime", "ls"],
            Self::RuntimeSelect => &["runtime", "select"],
            Self::RuntimeRemove => &["runtime", "remove"],
            Self::RuntimeUpdate => &["runtime", "update"],
            Self::RuntimeGet => &["runtime", "get"],
            Self::RuntimeSurvey => &["runtime", "survey"],
            Self::LinkStatus => &["link", "status"],
            Self::LinkEnable => &["link", "enable"],
            Self::LinkDisable => &["link", "disable"],
            Self::LinkSetDeviceName => &["link", "set-device-name"],
            Self::LinkSetPreferredDevice => &["link", "set-preferred-device"],
            Self::Get => &["get", "-y"],
            Self::Import => &["import", "-y"],
            Self::Clone => &["clone"],
            Self::Whoami => &["whoami"],
            Self::Logout => &["logout"],
        }
    }

    /// Whether the command changes state (as opposed to just reporting it).
    pub fn is_mutating(self) -> bool {
        !matches!(
            self,
            Self::ServerStatus
                | Self::RuntimeLs
                | Self::RuntimeSurvey
                | Self::LinkStatus
                | Self::Whoami
        )
    }

    /// How many caller arguments the command needs to run without
    /// prompting. Commands that would otherwise open an interactive picker
    /// or error out are rejected up front with a clear message instead.
    pub fn min_args(self) -> usize {
        match self {
            Self::RuntimeSelect
            | Self::RuntimeRemove
            | Self::RuntimeGet
            | Self::LinkSetDeviceName
            | Self::LinkSetPreferredDevice
            | Self::Get
            | Self::Import
            | Self::Clone => 1,
            _ => 0,
        }
    }

    /// Full argv (after the program name): the fixed arguments followed by
    /// the caller's extra arguments.
    pub fn argv(self, extra: &[String]) -> Vec<String> {
        self.base_args()
            .iter()
            .map(|s| (*s).to_string())
            .chain(extra.iter().cloned())
            .collect()
    }
}

/// Reject extra arguments that can't be passed through to a process
/// unchanged, with a message that says what to fix.
pub fn validate_extra_args(command: LmsCommand, extra: &[String]) -> Result<(), String> {
    const MAX_ARGS: usize = 32;
    if extra.len() < command.min_args() {
        return Err(format!(
            "`{}` needs at least {} argument(s) in `args`; without one the CLI would prompt interactively or fail",
            command.base_args().join(" "),
            command.min_args()
        ));
    }
    if extra.len() > MAX_ARGS {
        return Err(format!(
            "too many arguments ({}); at most {MAX_ARGS} are allowed",
            extra.len()
        ));
    }
    if extra.iter().any(|a| a.contains('\0')) {
        return Err("arguments must not contain NUL characters".to_string());
    }
    Ok(())
}

/// File name of the CLI binary on this platform.
pub fn exe_name() -> &'static str {
    if cfg!(windows) {
        "lms.exe"
    } else {
        "lms"
    }
}

/// Find the `lms` binary: an explicit `LMS_PATH` override first, then
/// `PATH`, then LM Studio's own install location (`~/.lmstudio/bin`), which
/// the installer adds to `PATH` for new shells but a GUI-launched MCP client
/// may not have inherited.
pub fn locate_lms() -> Option<PathBuf> {
    locate_lms_in(
        std::env::var_os("LMS_PATH"),
        std::env::var_os("PATH"),
        dirs::home_dir(),
    )
}

pub(crate) fn locate_lms_in(
    override_path: Option<OsString>,
    path_var: Option<OsString>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(p) = override_path.filter(|p| !p.is_empty()) {
        let p = PathBuf::from(p);
        // An explicit override that doesn't exist is a misconfiguration;
        // don't silently fall through to a different binary.
        return p.is_file().then_some(p);
    }
    let on_path = path_var.into_iter().flat_map(|v| {
        std::env::split_paths(&v)
            .map(|dir| dir.join(exe_name()))
            .collect::<Vec<_>>()
    });
    let in_home = home.map(|h| h.join(".lmstudio").join("bin").join(exe_name()));
    on_path.chain(in_home).find(|p| p.is_file())
}

/// Candidate locations of the LM Studio desktop app on this platform.
pub fn app_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if cfg!(target_os = "macos") {
        out.push(PathBuf::from("/Applications/LM Studio.app"));
        if let Some(h) = dirs::home_dir() {
            out.push(h.join("Applications").join("LM Studio.app"));
        }
    } else if cfg!(windows) {
        if let Some(d) = dirs::data_local_dir() {
            out.push(d.join("Programs").join("LM Studio"));
        }
    }
    out
}

/// LM Studio's per-user data directory, if present (all platforms).
pub fn data_dir() -> Option<PathBuf> {
    dirs::home_dir()
        .map(|h| h.join(".lmstudio"))
        .filter(|d| d.is_dir())
}

/// Best-effort desktop app version. Only macOS exposes this cheaply (the
/// app bundle's `Info.plist`); elsewhere, or if the plist is binary-format,
/// returns `None`.
pub fn read_app_version(app_path: &Path) -> Option<String> {
    let plist = std::fs::read_to_string(app_path.join("Contents").join("Info.plist")).ok()?;
    parse_plist_version(&plist)
}

pub(crate) fn parse_plist_version(plist: &str) -> Option<String> {
    PLIST_VERSION_RE
        .captures(plist)
        .map(|c| c[1].trim().to_string())
}

/// Remove ANSI color/cursor escape sequences from CLI output.
pub fn strip_ansi(s: &str) -> String {
    ANSI_RE.replace_all(s, "").into_owned()
}

/// Extract the `CLI commit: <hash>` line from `lms version`/`lms --help` output.
pub fn parse_cli_commit(output: &str) -> Option<String> {
    CLI_COMMIT_RE
        .captures(&strip_ansi(output))
        .map(|c| c[1].to_string())
}

/// Captured result of one CLI invocation.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct CliOutput {
    /// Process exit code; `None` if it was killed by a signal or timed out.
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    /// True if stdout or stderr was cut to fit the output cap.
    pub truncated: bool,
}

impl CliOutput {
    pub fn succeeded(&self) -> bool {
        !self.timed_out && self.exit_code == Some(0)
    }
}

/// Keep at most `MAX_OUTPUT_BYTES` of `s`, cutting on a char boundary.
fn cap(s: String) -> (String, bool) {
    if s.len() <= MAX_OUTPUT_BYTES {
        return (s, false);
    }
    let mut end = MAX_OUTPUT_BYTES;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    (s[..end].to_string(), true)
}

/// Run `program` with `args`, capturing ANSI-stripped output.
///
/// Stdin is closed so a command that wants to prompt fails immediately
/// instead of hanging until the timeout, and the child is killed if the
/// timeout elapses (or this future is dropped).
pub async fn run(program: &Path, args: &[String], timeout: Duration) -> std::io::Result<CliOutput> {
    let child = Command::new(program)
        .args(args)
        .env("NO_COLOR", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;

    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(result) => {
            let out = result?;
            let (stdout, t1) = cap(strip_ansi(&String::from_utf8_lossy(&out.stdout)));
            let (stderr, t2) = cap(strip_ansi(&String::from_utf8_lossy(&out.stderr)));
            Ok(CliOutput {
                exit_code: out.status.code(),
                stdout,
                stderr,
                timed_out: false,
                truncated: t1 || t2,
            })
        }
        Err(_) => Ok(CliOutput {
            timed_out: true,
            ..Default::default()
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("lmstudio-rs-mcp-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).expect("create scratch dir");
        d
    }

    #[test]
    fn argv_puts_fixed_args_before_caller_args() {
        let argv = LmsCommand::RuntimeSelect.argv(&["llama.cpp-mac@2.50.0".to_string()]);
        assert_eq!(argv, vec!["runtime", "select", "llama.cpp-mac@2.50.0"]);
    }

    #[test]
    fn get_and_import_are_always_non_interactive() {
        assert_eq!(
            LmsCommand::Get.argv(&["a/b".into()]),
            vec!["get", "-y", "a/b"]
        );
        assert_eq!(
            LmsCommand::Import.argv(&["/tmp/m.gguf".into()]),
            vec!["import", "-y", "/tmp/m.gguf"]
        );
    }

    #[test]
    fn server_status_asks_for_json() {
        assert_eq!(
            LmsCommand::ServerStatus.argv(&[]),
            vec!["server", "status", "--json"]
        );
    }

    #[test]
    fn read_only_commands_are_not_mutating() {
        assert!(!LmsCommand::RuntimeLs.is_mutating());
        assert!(!LmsCommand::Whoami.is_mutating());
        assert!(LmsCommand::ServerStop.is_mutating());
        assert!(LmsCommand::RuntimeRemove.is_mutating());
        assert!(LmsCommand::Logout.is_mutating());
    }

    #[test]
    fn command_names_are_snake_case_on_the_wire() {
        let v = serde_json::to_value(LmsCommand::LinkSetPreferredDevice).unwrap();
        assert_eq!(v, serde_json::json!("link_set_preferred_device"));
        let parsed: LmsCommand = serde_json::from_value(serde_json::json!("runtime_ls")).unwrap();
        assert_eq!(parsed, LmsCommand::RuntimeLs);
        // Anything outside the allowlist must not deserialize.
        assert!(serde_json::from_value::<LmsCommand>(serde_json::json!("chat")).is_err());
        assert!(serde_json::from_value::<LmsCommand>(serde_json::json!("push")).is_err());
    }

    #[test]
    fn validate_extra_args_rejects_nul_and_excess() {
        let c = LmsCommand::Whoami;
        assert!(validate_extra_args(c, &["ok".into()]).is_ok());
        assert!(validate_extra_args(c, &["bad\0arg".into()]).is_err());
        let many = vec!["x".to_string(); 33];
        assert!(validate_extra_args(c, &many).is_err());
    }

    #[test]
    fn commands_that_would_prompt_require_an_argument() {
        // Seen live: bare `runtime select` errors, and bare `runtime get` /
        // `link set-preferred-device` open interactive pickers.
        for c in [
            LmsCommand::RuntimeSelect,
            LmsCommand::RuntimeGet,
            LmsCommand::LinkSetDeviceName,
            LmsCommand::LinkSetPreferredDevice,
        ] {
            assert!(validate_extra_args(c, &[]).is_err(), "{c:?}");
            assert!(validate_extra_args(c, &["x".into()]).is_ok(), "{c:?}");
        }
        assert!(validate_extra_args(LmsCommand::RuntimeLs, &[]).is_ok());
        let msg = validate_extra_args(LmsCommand::RuntimeSelect, &[]).unwrap_err();
        assert!(msg.contains("runtime select"), "{msg}");
    }

    #[test]
    fn strips_ansi_and_parses_cli_commit() {
        let banner = "\x1b[38;5;166m  banner\x1b[0m\nCLI commit: 69d945a\n";
        assert_eq!(strip_ansi(banner), "  banner\nCLI commit: 69d945a\n");
        assert_eq!(parse_cli_commit(banner).as_deref(), Some("69d945a"));
        assert_eq!(parse_cli_commit("no commit here"), None);
    }

    #[test]
    fn parses_bundle_version_from_plist() {
        let plist =
            "<dict><key>CFBundleShortVersionString</key>\n\t<string>0.4.25+1</string></dict>";
        assert_eq!(parse_plist_version(plist).as_deref(), Some("0.4.25+1"));
        assert_eq!(parse_plist_version("<dict></dict>"), None);
    }

    #[test]
    fn locate_prefers_override_and_does_not_fall_through() {
        let dir = scratch_dir("locate-override");
        let bin = dir.join(exe_name());
        std::fs::write(&bin, "").unwrap();

        let found = locate_lms_in(Some(bin.clone().into_os_string()), None, None);
        assert_eq!(found, Some(bin));

        // A bad override is a misconfiguration, even if PATH has a real lms.
        let missing = dir.join("nope");
        let path_var = std::env::join_paths([&dir]).unwrap();
        assert_eq!(
            locate_lms_in(Some(missing.into_os_string()), Some(path_var), None),
            None
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn locate_searches_path_then_lmstudio_home() {
        let on_path = scratch_dir("locate-path");
        std::fs::write(on_path.join(exe_name()), "").unwrap();
        let path_var = std::env::join_paths([&on_path]).unwrap();
        assert_eq!(
            locate_lms_in(None, Some(path_var), None),
            Some(on_path.join(exe_name()))
        );

        let home = scratch_dir("locate-home");
        let bin_dir = home.join(".lmstudio").join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        std::fs::write(bin_dir.join(exe_name()), "").unwrap();
        assert_eq!(
            locate_lms_in(None, None, Some(home.clone())),
            Some(bin_dir.join(exe_name()))
        );

        assert_eq!(locate_lms_in(None, None, None), None);
        std::fs::remove_dir_all(on_path).ok();
        std::fs::remove_dir_all(home).ok();
    }

    #[test]
    fn cap_truncates_on_a_char_boundary() {
        let s = "é".repeat(MAX_OUTPUT_BYTES); // 2 bytes each
        let (out, truncated) = cap(s);
        assert!(truncated);
        assert!(out.len() <= MAX_OUTPUT_BYTES);
        let (small, truncated) = cap("short".to_string());
        assert_eq!((small.as_str(), truncated), ("short", false));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_captures_output_and_exit_code() {
        let out = run(
            Path::new("/bin/sh"),
            &["-c".into(), "echo out; echo err >&2; exit 3".into()],
            Duration::from_secs(10),
        )
        .await
        .unwrap();
        assert_eq!(out.exit_code, Some(3));
        assert_eq!(out.stdout.trim(), "out");
        assert_eq!(out.stderr.trim(), "err");
        assert!(!out.succeeded());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_kills_and_reports_a_timeout() {
        let out = run(
            Path::new("/bin/sh"),
            &["-c".into(), "sleep 30".into()],
            Duration::from_millis(100),
        )
        .await
        .unwrap();
        assert!(out.timed_out);
        assert!(!out.succeeded());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_gives_the_child_a_closed_stdin() {
        // `cat` with no input must see EOF immediately rather than hang.
        let out = run(Path::new("/bin/cat"), &[], Duration::from_secs(10))
            .await
            .unwrap();
        assert!(out.succeeded());
    }

    #[tokio::test]
    async fn run_reports_a_missing_binary_as_an_io_error() {
        let err = run(
            Path::new("/definitely/not/a/real/lms"),
            &[],
            Duration::from_secs(1),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }
}
