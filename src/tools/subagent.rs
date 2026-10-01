//! MCP-facing `run_subagent` tool: delegates a task to a local LM Studio
//! model acting as a subagent, with its own sandboxed tools. See
//! `src/subagent/` for the agentic loop, tool sandbox, and safety guard.

use crate::client::ApiClient;
use crate::subagent::runner::{self, SubagentConfig, SubagentReport};
use crate::subagent::tools::Capability;
use crate::types::{ErrorCode, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct RunSubagentInput {
    /// The task for the subagent to complete. Be specific about the goal
    /// and what "done" looks like — this model is smaller than you and
    /// benefits from a narrow, concrete task more than a broad one.
    pub task: String,
    /// Absolute path to the directory the subagent's tools are sandboxed
    /// to. It can read (and, depending on `capability`, write and run
    /// commands) anywhere under this directory, nowhere else.
    pub working_directory: String,
    /// Tool-access tier: `read_only` (investigate and report back, default),
    /// `read_write` (also write files), or `read_write_shell` (also run
    /// shell commands, subject to a non-configurable safety guard against
    /// the highest-risk command patterns). Grant only what the task needs.
    #[serde(default)]
    pub capability: Option<Capability>,
    /// Which model to use as the subagent. Auto-detected if exactly one
    /// model is currently loaded (LM Studio/Ollama); required for providers
    /// with no loaded-model concept (OpenAI/Anthropic).
    #[serde(default)]
    pub model: Option<String>,
    /// Override the subagent's default system prompt/persona.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Maximum tool-calling turns before giving up and returning whatever
    /// progress was made. Defaults to 15; clamped to 1-50.
    #[serde(default)]
    pub max_turns: Option<u32>,
    /// Sampling temperature. Defaults to 0.3 — lower than `chat_completion`,
    /// since a tool-calling loop benefits from more deterministic behavior.
    #[serde(default)]
    pub temperature: Option<f32>,
}

pub async fn run_subagent(
    client: &ApiClient,
    input: RunSubagentInput,
) -> ToolResult<SubagentReport> {
    let working_directory = PathBuf::from(&input.working_directory);
    if !working_directory.is_dir() {
        return ToolResult::err(
            format!(
                "working_directory does not exist or is not a directory: {}",
                input.working_directory
            ),
            ErrorCode::InvalidInput,
            "working_directory must be an existing, accessible directory",
        );
    }

    let model = match input.model {
        Some(m) => m,
        None => match super::models::auto_detect_model(client).await {
            Ok(m) => m,
            Err(e) => {
                return ToolResult::err(e.to_string(), e.code(), "model auto-detection failed")
            }
        },
    };

    let config = SubagentConfig {
        task: input.task,
        model,
        system_prompt: input.system_prompt,
        capability: input.capability.unwrap_or(Capability::ReadOnly),
        max_turns: input.max_turns.unwrap_or(15).clamp(1, 50),
        temperature: input.temperature.unwrap_or(0.3),
    };

    match runner::run(client, config, &working_directory).await {
        Ok(report) => {
            let message = format!(
                "Subagent finished ({}) after {} turn(s)",
                report.stop_reason, report.turns_used
            );
            ToolResult::ok(message, report)
        }
        Err(e) => ToolResult::err(format!("Subagent run failed: {e}"), e.code(), e.to_string()),
    }
}
