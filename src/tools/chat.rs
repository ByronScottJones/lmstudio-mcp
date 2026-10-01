//! Inference tools over the OpenAI-compatible endpoints: chat_completion,
//! text_completion.

use crate::client::LmStudioClient;
use crate::types::{ErrorCode, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// ---------------------------------------------------------------------------
// chat_completion
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ChatCompletionInput {
    /// The user's prompt to send to the model.
    pub prompt: String,
    /// Optional system instructions for the model.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Controls randomness (0.0 to 2.0). Defaults to 0.7.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate. Defaults to 2048.
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct ChatCompletionData {
    pub content: String,
    /// The model's internal "thinking" trace, for reasoning models that
    /// expose one (e.g. Qwen3's thinking mode). Populated whenever LM
    /// Studio returns a `reasoning_content` field, even alongside a
    /// non-empty `content`.
    pub reasoning_content: Option<String>,
    pub model: Option<String>,
    pub finish_reason: Option<String>,
}

pub async fn chat_completion(
    client: &LmStudioClient,
    input: ChatCompletionInput,
) -> ToolResult<ChatCompletionData> {
    let mut messages = Vec::new();
    if let Some(sys) = &input.system_prompt {
        if !sys.is_empty() {
            messages.push(serde_json::json!({ "role": "system", "content": sys }));
        }
    }
    messages.push(serde_json::json!({ "role": "user", "content": input.prompt }));

    let body = serde_json::json!({
        "messages": messages,
        "temperature": input.temperature.unwrap_or(0.7),
        "max_tokens": input.max_tokens.unwrap_or(2048),
    });

    match client.chat_completion(body).await {
        Ok(resp) => extract_chat_result(resp),
        Err(e) => ToolResult::err(
            format!("Failed to generate chat completion: {e}"),
            e.code(),
            e.to_string(),
        ),
    }
}

fn extract_chat_result(resp: Value) -> ToolResult<ChatCompletionData> {
    let choice = resp
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first());
    let Some(choice) = choice else {
        return ToolResult::err(
            "LM Studio returned no completion choices",
            ErrorCode::Unknown,
            resp.to_string(),
        );
    };
    let message = choice.get("message");
    let content = message
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or_default()
        .to_string();
    // Reasoning models (e.g. Qwen3 in thinking mode) emit their internal
    // deliberation here, separately from `content`. If max_tokens runs out
    // mid-thought, `content` can legitimately be empty while this is not —
    // that's a budget problem for the caller to fix, not a failed request.
    let reasoning_content = message
        .and_then(|m| m.get("reasoning_content"))
        .and_then(|c| c.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from);
    let finish_reason = choice
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .map(String::from);
    let model = resp.get("model").and_then(|v| v.as_str()).map(String::from);

    if content.is_empty() {
        return if let Some(reasoning) = reasoning_content {
            ToolResult::ok(
                format!(
                    "The model used its entire token budget on reasoning and produced no final answer \
                     (finish_reason: {}). See `reasoning_content`; retry with a higher `max_tokens`.",
                    finish_reason.as_deref().unwrap_or("unknown")
                ),
                ChatCompletionData {
                    content,
                    reasoning_content: Some(reasoning),
                    model,
                    finish_reason,
                },
            )
        } else {
            ToolResult::err(
                "LM Studio returned an empty response",
                ErrorCode::Unknown,
                resp.to_string(),
            )
        };
    }

    ToolResult::ok(
        "Generated chat completion",
        ChatCompletionData {
            content,
            reasoning_content,
            model,
            finish_reason,
        },
    )
}

// ---------------------------------------------------------------------------
// text_completion
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct TextCompletionInput {
    /// The text prompt to complete.
    pub prompt: String,
    /// Controls randomness (0.0 to 2.0). Defaults to 0.7.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate. Defaults to 2048.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Sequences where generation should stop.
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct TextCompletionData {
    pub text: String,
    pub model: Option<String>,
    pub finish_reason: Option<String>,
}

pub async fn text_completion(
    client: &LmStudioClient,
    input: TextCompletionInput,
) -> ToolResult<TextCompletionData> {
    let mut body = Map::new();
    body.insert("prompt".into(), Value::String(input.prompt));
    body.insert(
        "temperature".into(),
        Value::from(input.temperature.unwrap_or(0.7)),
    );
    body.insert(
        "max_tokens".into(),
        Value::from(input.max_tokens.unwrap_or(2048)),
    );
    if let Some(stop) = input.stop_sequences {
        if !stop.is_empty() {
            body.insert("stop".into(), Value::from(stop));
        }
    }

    match client.text_completion(Value::Object(body)).await {
        Ok(resp) => extract_text_result(resp),
        Err(e) => ToolResult::err(
            format!("Failed to generate text completion: {e}"),
            e.code(),
            e.to_string(),
        ),
    }
}

fn extract_text_result(resp: Value) -> ToolResult<TextCompletionData> {
    let choice = resp
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first());
    let Some(choice) = choice else {
        return ToolResult::err(
            "LM Studio returned no completion choices",
            ErrorCode::Unknown,
            resp.to_string(),
        );
    };
    let text = choice
        .get("text")
        .and_then(|c| c.as_str())
        .unwrap_or_default()
        .to_string();
    if text.is_empty() {
        return ToolResult::err(
            "LM Studio returned an empty completion",
            ErrorCode::Unknown,
            resp.to_string(),
        );
    }
    let finish_reason = choice
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .map(String::from);
    let model = resp.get("model").and_then(|v| v.as_str()).map(String::from);
    ToolResult::ok(
        "Generated text completion",
        TextCompletionData {
            text,
            model,
            finish_reason,
        },
    )
}
