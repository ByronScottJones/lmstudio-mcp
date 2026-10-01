//! The agentic loop: send the task to a local model with a tool-calling
//! `tools` array, execute whatever it calls via [`super::tools`], feed the
//! results back, and repeat until it answers with plain text or the turn
//! budget runs out.
//!
//! Deliberately mirrors the shape of a Claude Code subagent dispatch: the
//! caller supplies a task and a tool-access tier, the subagent works
//! autonomously across as many tool calls as it needs, and the caller gets
//! back one final report — not the full back-and-forth transcript, though a
//! brief action log is included since that's cheap and useful context for
//! judging whether to trust the result.

use crate::client::ApiClient;
use crate::subagent::tools::{self, Capability, SubagentContext};
use serde_json::{json, Value};
use std::path::Path;

pub struct SubagentConfig {
    pub task: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub capability: Capability,
    pub max_turns: u32,
    pub temperature: f32,
}

#[derive(Debug, Clone, serde::Serialize, schemars::JsonSchema)]
pub struct ActionLogEntry {
    pub turn: u32,
    pub tool: String,
    /// A one-line summary of the call and its outcome — not the full
    /// argument/result payload, to keep the report compact.
    pub summary: String,
}

#[derive(Debug, Clone, serde::Serialize, schemars::JsonSchema)]
pub struct SubagentReport {
    /// The subagent's final answer.
    pub final_message: String,
    /// Why the loop ended: "completed", "max_turns_reached", or
    /// "model_stopped_without_answer" (returned a tool_calls-free, empty
    /// final turn — rare, but distinct from a genuine answer).
    pub stop_reason: String,
    pub turns_used: u32,
    pub actions: Vec<ActionLogEntry>,
}

#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    #[error("request failed: {0}")]
    Client(#[from] crate::types::ClientError),
    #[error("working directory \"{path}\" doesn't exist or isn't accessible: {source}")]
    WorkingDirectory {
        path: String,
        source: std::io::Error,
    },
}

impl RunnerError {
    pub fn code(&self) -> crate::types::ErrorCode {
        match self {
            RunnerError::Client(e) => e.code(),
            RunnerError::WorkingDirectory { .. } => crate::types::ErrorCode::InvalidInput,
        }
    }
}

const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a focused subagent completing one specific task, delegated to you by another AI assistant. \
Use the tools available to you to investigate and complete the task; call a tool whenever you need \
information you don't already have rather than guessing. When you are done, reply with plain text \
(no further tool calls) summarizing what you found or changed and any result the delegator needs. \
Be direct and concrete — name the files you looked at or changed, and quote exact error text or \
output where relevant. If the task cannot be completed, say clearly what you tried and why it \
didn't work, rather than claiming success.\n\n\
Safety: never attempt privilege escalation (sudo/su/doas), recursive force-deletes, disk/partition \
commands, system shutdown/reboot, piping a downloaded script into a shell, or force-pushing git \
history — these are blocked for you regardless, but don't waste turns attempting them. Stay within \
the working directory you've been given.";

pub async fn run(
    client: &ApiClient,
    config: SubagentConfig,
    working_directory: &Path,
) -> Result<SubagentReport, RunnerError> {
    let ctx = SubagentContext::new(working_directory, config.capability).map_err(|e| {
        RunnerError::WorkingDirectory {
            path: working_directory.display().to_string(),
            source: e,
        }
    })?;

    let tool_defs = tools::tool_definitions(config.capability);
    let system_prompt = config
        .system_prompt
        .clone()
        .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());

    let mut messages = vec![
        json!({ "role": "system", "content": system_prompt }),
        json!({ "role": "user", "content": config.task }),
    ];
    let mut actions = Vec::new();

    for turn in 1..=config.max_turns {
        let body = json!({
            "model": config.model,
            "messages": messages,
            "tools": tool_defs,
            "temperature": config.temperature,
        });
        let resp = client.chat_completion(body).await?;
        let message = resp
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("message"))
            .cloned()
            .unwrap_or_else(|| json!({}));

        let tool_calls = message
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        if tool_calls.is_empty() {
            let content = message
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            let stop_reason = if content.is_empty() {
                "model_stopped_without_answer"
            } else {
                "completed"
            };
            return Ok(SubagentReport {
                final_message: if content.is_empty() {
                    "The subagent stopped without making any tool calls or producing a final answer.".to_string()
                } else {
                    content
                },
                stop_reason: stop_reason.to_string(),
                turns_used: turn,
                actions,
            });
        }

        // The assistant's tool-call turn goes into history verbatim (the
        // API requires it precede the matching "tool" result messages),
        // then one "tool" message per call with its result.
        messages.push(json!({
            "role": "assistant",
            "content": message.get("content").cloned().unwrap_or(Value::Null),
            "tool_calls": tool_calls,
        }));

        for call in &tool_calls {
            let call_id = call
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let name = call
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let raw_args = call
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str())
                .unwrap_or("{}")
                .to_string();

            let args: Value = if raw_args.trim().is_empty() {
                json!({})
            } else {
                match serde_json::from_str(&raw_args) {
                    Ok(v) => v,
                    Err(_) => {
                        // Malformed tool-call arguments are a routine thing
                        // for a smaller local model to produce — feed it
                        // back as a tool error so it can retry, rather than
                        // aborting the whole run over one bad call.
                        let msg =
                            format!("your arguments for '{name}' were not valid JSON: {raw_args}");
                        actions.push(ActionLogEntry {
                            turn,
                            tool: name.clone(),
                            summary: format!("{name}(...) -> error: invalid JSON arguments"),
                        });
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": msg,
                        }));
                        continue;
                    }
                }
            };

            let result = tools::execute(&ctx, &name, &args).await;
            let (summary, content) = match &result {
                Ok(output) => (
                    format!("{name}({}) -> {} chars", compact_args(&args), output.len()),
                    output.clone(),
                ),
                Err(err) => (
                    format!("{name}({}) -> error: {err}", compact_args(&args)),
                    format!("error: {err}"),
                ),
            };
            actions.push(ActionLogEntry {
                turn,
                tool: name.clone(),
                summary,
            });
            messages.push(json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            }));
        }
    }

    // Ran out of turns. Give back whatever the model's last words were (if
    // any tool-free content snuck into the final turn) plus the action log,
    // rather than just an empty failure — partial progress is still useful
    // for the delegator to see.
    Ok(SubagentReport {
        final_message: format!(
            "The subagent used all {} turns without producing a final answer. See `actions` for what it did; consider raising `max_turns` or narrowing the task.",
            config.max_turns
        ),
        stop_reason: "max_turns_reached".to_string(),
        turns_used: config.max_turns,
        actions,
    })
}

/// A short, human-scannable rendering of a tool call's arguments for the
/// action log — not the full JSON, just enough to tell calls apart at a
/// glance (e.g. `path="src/main.rs"`).
fn compact_args(args: &Value) -> String {
    let Value::Object(map) = args else {
        return String::new();
    };
    map.iter()
        .map(|(k, v)| {
            let v_str = match v {
                // Truncate by character count, not byte offset — `s[..60]`
                // panics if byte 60 falls inside a multibyte character.
                Value::String(s) if s.chars().count() > 60 => {
                    format!("\"{}...\"", s.chars().take(60).collect::<String>())
                }
                Value::String(s) => format!("\"{s}\""),
                other => other.to_string(),
            };
            format!("{k}={v_str}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_args_formats_short_strings_inline() {
        let args = json!({"path": "a.txt"});
        assert_eq!(compact_args(&args), "path=\"a.txt\"");
    }

    #[test]
    fn compact_args_truncates_long_strings() {
        let long = "x".repeat(100);
        let args = json!({"content": long});
        let out = compact_args(&args);
        assert!(out.len() < 100);
        assert!(out.contains("..."));
    }

    #[test]
    fn compact_args_handles_non_object() {
        assert_eq!(compact_args(&Value::Null), "");
    }

    #[test]
    fn compact_args_truncates_multibyte_strings_without_panicking() {
        // Regression: byte-index slicing (`&s[..60]`) panics if byte 60
        // falls inside a multibyte UTF-8 character. 4-byte emoji repeated
        // past the old 60-byte cutoff reliably hits that case.
        let long = "🦀".repeat(80); // 320 bytes, 80 chars — over the 60-char threshold
        let args = json!({"content": long});
        let out = compact_args(&args); // must not panic
        assert!(out.contains("..."));
    }
}
