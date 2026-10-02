//! LM Studio environment tools: `lmstudio_status` (installation, version,
//! `lms` CLI and API availability) and `lms_cli` (the `lms` features the
//! REST API doesn't cover).

use crate::client::ApiClient;
use crate::lms::{self, CliOutput, LmsCommand};
use crate::providers::Provider;
use crate::types::{ErrorCode, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// How long `lms version` gets during a status probe.
const PROBE_TIMEOUT: Duration = Duration::from_secs(15);
const DEFAULT_CLI_TIMEOUT_SECS: u64 = 120;
const MAX_CLI_TIMEOUT_SECS: u64 = 3600;

// ---------------------------------------------------------------------------
// lmstudio_status
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct InstallationInfo {
    /// True if the desktop app or the `lms` CLI was found on this machine.
    pub installed: bool,
    /// Path of the LM Studio desktop app (macOS/Windows), if found.
    pub app_path: Option<String>,
    /// Desktop app version, when it can be read (currently macOS only).
    pub app_version: Option<String>,
    /// LM Studio's per-user data directory (`~/.lmstudio`), if present.
    pub data_dir: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct CliInfo {
    /// True if an `lms` binary was found and ran successfully.
    pub available: bool,
    pub path: Option<String>,
    /// Commit hash the CLI was built from (`lms` doesn't print a semver).
    pub commit: Option<String>,
    /// Why the CLI is unavailable, if it isn't.
    pub error: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct ApiInfo {
    /// False when the check was skipped (the configured provider isn't LM Studio).
    pub checked: bool,
    /// True if the LM Studio REST API answered at `base_url`.
    pub available: bool,
    pub base_url: String,
    pub models_available: Option<usize>,
    /// Why the API is unavailable or the check was skipped.
    pub detail: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct LmStudioStatusData {
    pub installation: InstallationInfo,
    pub cli: CliInfo,
    pub api: ApiInfo,
}

fn path_string(p: &Path) -> String {
    p.display().to_string()
}

fn check_installation(cli_found: bool) -> InstallationInfo {
    let app_path = lms::app_candidates().into_iter().find(|p| p.exists());
    let app_version = app_path.as_deref().and_then(lms::read_app_version);
    let data_dir = lms::data_dir();
    InstallationInfo {
        installed: app_path.is_some() || cli_found,
        app_path: app_path.as_deref().map(path_string),
        app_version,
        data_dir: data_dir.as_deref().map(path_string),
    }
}

async fn check_cli() -> CliInfo {
    let Some(path) = lms::locate_lms() else {
        return CliInfo {
            error: Some(
                "`lms` not found via LMS_PATH, PATH, or ~/.lmstudio/bin. Run LM Studio once \
                 (or `~/.lmstudio/bin/lms bootstrap`) to install it."
                    .to_string(),
            ),
            ..Default::default()
        };
    };
    let path_str = path_string(&path);
    match lms::run(&path, &["version".to_string()], PROBE_TIMEOUT).await {
        Ok(out) if out.succeeded() => CliInfo {
            available: true,
            path: Some(path_str),
            commit: lms::parse_cli_commit(&out.stdout),
            error: None,
        },
        Ok(out) => CliInfo {
            path: Some(path_str),
            error: Some(if out.timed_out {
                "`lms version` timed out".to_string()
            } else {
                format!(
                    "`lms version` exited with {:?}: {}",
                    out.exit_code,
                    out.stderr.trim()
                )
            }),
            ..Default::default()
        },
        Err(e) => CliInfo {
            path: Some(path_str),
            error: Some(format!("could not run `lms`: {e}")),
            ..Default::default()
        },
    }
}

async fn check_api(client: &ApiClient) -> ApiInfo {
    let base_url = client.base_url().to_string();
    if client.provider() != Provider::LmStudio {
        return ApiInfo {
            base_url,
            detail: Some(format!(
                "skipped: the configured provider is {}, not LM Studio (use health_check for it)",
                client.provider()
            )),
            ..Default::default()
        };
    }
    match client.health_check().await {
        Ok(resp) => ApiInfo {
            checked: true,
            available: true,
            base_url,
            models_available: Some(resp.models.len()),
            detail: None,
        },
        Err(e) => ApiInfo {
            checked: true,
            base_url,
            detail: Some(e.to_string()),
            ..Default::default()
        },
    }
}

fn status_summary(d: &LmStudioStatusData) -> String {
    let yn = |b: bool| if b { "yes" } else { "no" };
    let api = if d.api.checked {
        yn(d.api.available)
    } else {
        "not checked"
    };
    format!(
        "LM Studio installed: {}{}; lms CLI available: {}{}; API available: {api}",
        yn(d.installation.installed),
        d.installation
            .app_version
            .as_deref()
            .map(|v| format!(" (v{v})"))
            .unwrap_or_default(),
        yn(d.cli.available),
        d.cli
            .commit
            .as_deref()
            .map(|c| format!(" (commit {c})"))
            .unwrap_or_default(),
    )
}

/// Probe all four things independently — a missing CLI must not hide a
/// working API, or vice versa — so this always succeeds and reports each
/// finding in `data`.
pub async fn lmstudio_status(client: &ApiClient) -> ToolResult<LmStudioStatusData> {
    let (cli, api) = tokio::join!(check_cli(), check_api(client));
    let installation = check_installation(cli.path.is_some());
    let data = LmStudioStatusData {
        installation,
        cli,
        api,
    };
    ToolResult::ok(status_summary(&data), data)
}

// ---------------------------------------------------------------------------
// lms_cli
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct LmsCliInput {
    /// Which `lms` subcommand to run.
    pub command: LmsCommand,
    /// Extra arguments appended after the subcommand, e.g. `["qwen/qwen3.5-9b@q8_0"]`
    /// for `get`, or `["--help"]`. Passed to `lms` directly, not through a shell.
    #[serde(default)]
    pub args: Vec<String>,
    /// Kill the command after this many seconds (default 120, max 3600). Downloads
    /// (`get`, `runtime get`) may need far longer than the default.
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct LmsCliData {
    /// The command line that was run, for the record.
    pub command_line: String,
    #[serde(flatten)]
    pub output: CliOutput,
}

pub async fn lms_cli(input: LmsCliInput) -> ToolResult<LmsCliData> {
    if let Err(e) = lms::validate_extra_args(input.command, &input.args) {
        return ToolResult::err("Invalid arguments", ErrorCode::InvalidInput, e);
    }
    let Some(program) = lms::locate_lms() else {
        return ToolResult::err(
            "The `lms` CLI was not found",
            ErrorCode::CliNotFound,
            "not found via LMS_PATH, PATH, or ~/.lmstudio/bin; see lmstudio_status",
        );
    };

    let argv = input.command.argv(&input.args);
    let timeout = Duration::from_secs(
        input
            .timeout_seconds
            .unwrap_or(DEFAULT_CLI_TIMEOUT_SECS)
            .clamp(1, MAX_CLI_TIMEOUT_SECS),
    );
    let command_line = format!("lms {}", argv.join(" "));
    tracing::info!(%command_line, mutating = input.command.is_mutating(), "Running lms CLI command");

    match lms::run(&program, &argv, timeout).await {
        Ok(output) if output.succeeded() => ToolResult::ok(
            format!("`{command_line}` succeeded"),
            LmsCliData {
                command_line,
                output,
            },
        ),
        Ok(output) if output.timed_out => ToolResult::err(
            format!("`{command_line}` timed out after {}s", timeout.as_secs()),
            ErrorCode::Timeout,
            "the command was killed; raise timeout_seconds if it needs longer",
        ),
        Ok(output) => {
            let detail = if output.stderr.trim().is_empty() {
                output.stdout.trim().to_string()
            } else {
                output.stderr.trim().to_string()
            };
            ToolResult::err(
                format!("`{command_line}` exited with {:?}", output.exit_code),
                ErrorCode::Unknown,
                detail,
            )
        }
        Err(e) => ToolResult::err(
            format!("Failed to run `{command_line}`"),
            ErrorCode::Unknown,
            e.to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_reports_each_finding() {
        let data = LmStudioStatusData {
            installation: InstallationInfo {
                installed: true,
                app_version: Some("0.4.25+1".into()),
                ..Default::default()
            },
            cli: CliInfo {
                available: true,
                commit: Some("69d945a".into()),
                ..Default::default()
            },
            api: ApiInfo {
                checked: true,
                available: false,
                ..Default::default()
            },
        };
        assert_eq!(
            status_summary(&data),
            "LM Studio installed: yes (v0.4.25+1); lms CLI available: yes (commit 69d945a); API available: no"
        );
    }

    #[test]
    fn summary_marks_skipped_api_check() {
        let s = status_summary(&LmStudioStatusData::default());
        assert!(s.ends_with("API available: not checked"), "{s}");
    }

    #[tokio::test]
    async fn lms_cli_rejects_bad_args_before_running_anything() {
        let r = lms_cli(LmsCliInput {
            command: LmsCommand::Whoami,
            args: vec!["a\0b".into()],
            timeout_seconds: None,
        })
        .await;
        assert!(!r.success);
        assert_eq!(r.error.unwrap().code, ErrorCode::InvalidInput);
    }

    #[test]
    fn input_defaults_args_when_omitted() {
        let input: LmsCliInput =
            serde_json::from_value(serde_json::json!({ "command": "server_status" })).unwrap();
        assert!(input.args.is_empty());
        assert_eq!(input.command, LmsCommand::ServerStatus);
    }
}
