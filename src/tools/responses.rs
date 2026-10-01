//! Stateful conversation tools built on LM Studio's `/v1/responses` endpoint:
//! create_response, start_conversation, continue_conversation.
//!
//! Requires LM Studio v0.3.29+. Unlike `chat_completion`, this endpoint
//! maintains conversation context server-side via response IDs — no manual
//! message history management needed.
//!
//! Note: the original Python bridge this was ported from accepted
//! `temperature`/`max_tokens` parameters on these tools but never actually
//! included them in the request payload. This port fixes that — both are
//! sent through as `temperature` / `max_output_tokens`, matching the
//! Responses API.

use super::models::fetch_loaded;
use crate::client::LmStudioClient;
use crate::types::{ErrorCode, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct ResponseData {
    /// Pass this to `previous_response_id` (create_response) or `response_id`
    /// (continue_conversation) to continue this conversation.
    pub response_id: String,
    pub message: String,
    /// The model's internal "thinking" trace, for reasoning models that
    /// expose one. Populated whenever a `reasoning` output block is present,
    /// even alongside a non-empty `message`.
    pub reasoning_content: Option<String>,
    pub model: String,
}

/// Reasoning effort for models that support it.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    fn as_str(self) -> &'static str {
        match self {
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
        }
    }
}

async fn auto_detect_model(client: &LmStudioClient) -> Result<String, String> {
    match fetch_loaded(client).await {
        Ok(models) if models.len() == 1 => Ok(models[0].model_key.clone()),
        Ok(models) if models.is_empty() => Err(
            "No model is currently loaded in LM Studio. Load one first, or pass `model` explicitly."
                .to_string(),
        ),
        Ok(_) => Err(
            "Multiple models are loaded; pass `model` explicitly to pick one.".to_string(),
        ),
        Err(e) => Err(format!("Could not detect the currently loaded model: {e}")),
    }
}

/// The assistant's final text and, separately, any reasoning trace, pulled
/// out of a `/v1/responses` payload's `output[]` array.
struct ExtractedOutput {
    message: String,
    reasoning_content: Option<String>,
}

/// Walk `output[]` for a `message` block (then its `content[]` for an
/// `output_text` item) and, separately, for a `reasoning` block. A model
/// that runs out of `max_output_tokens` while thinking can legitimately
/// produce the latter without the former — reasoning models (e.g. Qwen3 in
/// thinking mode) expose their deliberation this way.
fn extract_output(data: &Value) -> ExtractedOutput {
    let mut message = String::new();
    let mut reasoning_parts = Vec::new();

    match data.get("output") {
        Some(Value::Array(arr)) => {
            for block in arr {
                match block.get("type").and_then(|t| t.as_str()) {
                    Some("message") => {
                        if let Some(content) = block.get("content").and_then(|c| c.as_array()) {
                            for item in content {
                                if item.get("type").and_then(|t| t.as_str()) == Some("output_text")
                                {
                                    if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                                        message.push_str(text);
                                    }
                                }
                            }
                        }
                    }
                    Some("reasoning") => {
                        if let Some(content) = block.get("content").and_then(|c| c.as_array()) {
                            for item in content {
                                if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                                    reasoning_parts.push(text.to_string());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Some(Value::String(s)) => message.push_str(s),
        _ => {}
    }

    ExtractedOutput {
        message,
        reasoning_content: if reasoning_parts.is_empty() {
            None
        } else {
            Some(reasoning_parts.join("\n"))
        },
    }
}

async fn send_responses_request(
    client: &LmStudioClient,
    mut body: Map<String, Value>,
    model: Option<String>,
) -> ToolResult<ResponseData> {
    let model = match model {
        Some(m) => m,
        None => match auto_detect_model(client).await {
            Ok(m) => m,
            Err(msg) => {
                return ToolResult::err(
                    msg,
                    ErrorCode::ModelNotLoaded,
                    "model auto-detection failed",
                )
            }
        },
    };
    body.insert("model".into(), Value::String(model.clone()));
    // `LmStudioClient::responses` always requests this with "stream": true
    // internally (see its doc comment) and reassembles the result, so
    // nothing needs setting here.

    match client.responses(Value::Object(body)).await {
        Ok(data) => {
            let ExtractedOutput {
                message,
                reasoning_content,
            } = extract_output(&data);
            let response_id = data
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let status = data.get("status").and_then(|v| v.as_str());
            let model = data
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or(&model)
                .to_string();

            let result_message = if message.is_empty() && reasoning_content.is_some() {
                format!(
                    "The model used its entire token budget on reasoning and produced no final answer \
                     (status: {}). See `reasoning_content`; retry with a higher `max_output_tokens`.",
                    status.unwrap_or("unknown")
                )
            } else {
                "Received response".to_string()
            };

            ToolResult::ok(
                result_message,
                ResponseData {
                    response_id,
                    message,
                    reasoning_content,
                    model,
                },
            )
        }
        Err(e) => ToolResult::err(
            format!("Request to LM Studio failed: {e}"),
            e.code(),
            e.to_string(),
        ),
    }
}

// ---------------------------------------------------------------------------
// create_response
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CreateResponseInput {
    /// The user's input text.
    pub input_text: String,
    /// ID from a previous response to continue the conversation.
    #[serde(default)]
    pub previous_response_id: Option<String>,
    /// Reasoning depth, for models that support it. Defaults to medium.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Controls randomness (0.0 to 2.0).
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Maximum number of output tokens to generate.
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    /// Model to use. If omitted, the currently loaded model is used
    /// (auto-detection fails if zero or more than one model is loaded).
    #[serde(default)]
    pub model: Option<String>,
}

pub async fn create_response(
    client: &LmStudioClient,
    input: CreateResponseInput,
) -> ToolResult<ResponseData> {
    let mut body = Map::new();
    body.insert("input".into(), Value::String(input.input_text));
    if let Some(prev) = input.previous_response_id {
        body.insert("previous_response_id".into(), Value::String(prev));
    }
    let effort = input.reasoning_effort.unwrap_or(ReasoningEffort::Medium);
    body.insert(
        "reasoning".into(),
        serde_json::json!({ "effort": effort.as_str() }),
    );
    if let Some(t) = input.temperature {
        body.insert("temperature".into(), Value::from(t));
    }
    if let Some(m) = input.max_output_tokens {
        body.insert("max_output_tokens".into(), Value::from(m));
    }

    send_responses_request(client, body, input.model).await
}

// ---------------------------------------------------------------------------
// start_conversation
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct StartConversationInput {
    /// The persona or instructions to apply for the whole conversation
    /// (e.g. "You are a friend at a bar, keep it casual and fun").
    pub system_prompt: String,
    /// The opening message to send to the model.
    pub first_message: String,
    /// Controls randomness (0.0 to 2.0). Defaults to 0.7.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Maximum number of output tokens per response. Defaults to 2048.
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    /// Model to use. Auto-detected if omitted.
    #[serde(default)]
    pub model: Option<String>,
}

pub async fn start_conversation(
    client: &LmStudioClient,
    input: StartConversationInput,
) -> ToolResult<ResponseData> {
    let mut body = Map::new();
    body.insert("input".into(), Value::String(input.first_message));
    body.insert("instructions".into(), Value::String(input.system_prompt));
    body.insert(
        "temperature".into(),
        Value::from(input.temperature.unwrap_or(0.7)),
    );
    body.insert(
        "max_output_tokens".into(),
        Value::from(input.max_output_tokens.unwrap_or(2048)),
    );

    send_responses_request(client, body, input.model).await
}

// ---------------------------------------------------------------------------
// continue_conversation
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ContinueConversationInput {
    /// The `response_id` returned by `start_conversation` or a previous
    /// `continue_conversation` call.
    pub response_id: String,
    /// Your next message in the conversation.
    pub message: String,
    /// Controls randomness (0.0 to 2.0). Defaults to 0.7.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Maximum number of output tokens per response. Defaults to 2048.
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    /// Model to use. Auto-detected if omitted.
    #[serde(default)]
    pub model: Option<String>,
}

pub async fn continue_conversation(
    client: &LmStudioClient,
    input: ContinueConversationInput,
) -> ToolResult<ResponseData> {
    let mut body = Map::new();
    body.insert("input".into(), Value::String(input.message));
    body.insert(
        "previous_response_id".into(),
        Value::String(input.response_id),
    );
    body.insert(
        "temperature".into(),
        Value::from(input.temperature.unwrap_or(0.7)),
    );
    body.insert(
        "max_output_tokens".into(),
        Value::from(input.max_output_tokens.unwrap_or(2048)),
    );

    send_responses_request(client, body, input.model).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_text_and_reasoning_from_a_realistic_responses_payload() {
        let payload = serde_json::json!({
            "id": "resp_abc123",
            "model": "openai/gpt-oss-20b",
            "output": [
                {
                    "type": "reasoning",
                    "content": [{ "type": "reasoning_text", "text": "thinking..." }]
                },
                {
                    "type": "message",
                    "content": [
                        { "type": "output_text", "text": "Hello there!" }
                    ]
                }
            ]
        });

        let out = extract_output(&payload);
        assert_eq!(out.message, "Hello there!");
        assert_eq!(out.reasoning_content.as_deref(), Some("thinking..."));
    }

    #[test]
    fn falls_back_to_plain_string_output() {
        let payload = serde_json::json!({ "output": "just a string" });
        assert_eq!(extract_output(&payload).message, "just a string");
    }

    #[test]
    fn surfaces_reasoning_when_no_message_block_present() {
        // A model that exhausts max_output_tokens while thinking: a
        // `reasoning` block but no `message` block.
        let payload = serde_json::json!({
            "output": [{ "type": "reasoning", "content": [{ "text": "still thinking..." }] }]
        });
        let out = extract_output(&payload);
        assert_eq!(out.message, "");
        assert_eq!(out.reasoning_content.as_deref(), Some("still thinking..."));
    }

    #[test]
    fn returns_empty_when_output_is_absent_entirely() {
        let payload = serde_json::json!({});
        let out = extract_output(&payload);
        assert_eq!(out.message, "");
        assert_eq!(out.reasoning_content, None);
    }

    #[test]
    fn reasoning_effort_serializes_lowercase() {
        assert_eq!(ReasoningEffort::Low.as_str(), "low");
        assert_eq!(ReasoningEffort::High.as_str(), "high");
    }
}
