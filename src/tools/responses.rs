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

use super::models::auto_detect_model;
use crate::client::ApiClient;
use crate::types::{ErrorCode, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

/// Tracks the persona (`instructions`) `start_conversation` locks in, keyed
/// by the most recent `response_id` in that conversation's chain.
///
/// This exists because the Responses API's `instructions` field is **not**
/// carried forward by `previous_response_id` the way the message history
/// itself is — it applies only to the single turn it's sent on. Without
/// this, `start_conversation`'s documented "locked in for the whole
/// session, no need to resend it" guarantee would silently stop being true
/// after the first turn. `continue_conversation` looks the persona up by
/// the incoming `response_id` and resends it as `instructions`, then
/// re-keys it under the new `response_id` the call returns, so it
/// propagates indefinitely along the chain.
///
/// Bounded FIFO eviction (not a real LRU — good enough for "don't grow
/// unboundedly over a long-running server session" without pulling in a
/// dependency for it) rather than ever-growing, since nothing else ever
/// removes an entry once its conversation is abandoned.
pub struct PersonaCache {
    inner: Mutex<PersonaCacheInner>,
}

struct PersonaCacheInner {
    personas: HashMap<String, String>,
    insertion_order: VecDeque<String>,
}

const MAX_TRACKED_CONVERSATIONS: usize = 500;

impl Default for PersonaCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PersonaCache {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(PersonaCacheInner {
                personas: HashMap::new(),
                insertion_order: VecDeque::new(),
            }),
        }
    }

    fn get(&self, response_id: &str) -> Option<String> {
        self.inner
            .lock()
            .unwrap()
            .personas
            .get(response_id)
            .cloned()
    }

    fn set(&self, response_id: String, persona: String) {
        let mut inner = self.inner.lock().unwrap();
        if !inner.personas.contains_key(&response_id) {
            inner.insertion_order.push_back(response_id.clone());
        }
        inner.personas.insert(response_id, persona);
        while inner.insertion_order.len() > MAX_TRACKED_CONVERSATIONS {
            if let Some(oldest) = inner.insertion_order.pop_front() {
                inner.personas.remove(&oldest);
            }
        }
    }

    fn remove(&self, response_id: &str) {
        let mut inner = self.inner.lock().unwrap();
        inner.personas.remove(response_id);
        // Left in `insertion_order` (if present) as a harmless tombstone —
        // it'll just no-op out of `personas` when its turn to evict comes.
    }
}

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

/// If `status` marks this response object as failed, return a detail
/// string describing why (from an `error` field if present, else a
/// generic fallback). Returns `None` for any other status, including a
/// missing one — only an explicit "failed" is treated as failure, since
/// not every LM Studio version necessarily sets `status` on success.
fn failure_detail(data: &Value, status: Option<&str>) -> Option<String> {
    if status != Some("failed") {
        return None;
    }
    Some(
        data.get("error")
            .and_then(|e| e.get("message").and_then(|m| m.as_str()).or(e.as_str()))
            .unwrap_or("the provider reported this response as failed, with no further detail")
            .to_string(),
    )
}

async fn send_responses_request(
    client: &ApiClient,
    mut body: Map<String, Value>,
    model: Option<String>,
) -> ToolResult<ResponseData> {
    if !client.provider().supports_responses_api() {
        return ToolResult::err(
            format!(
                "{} has no /v1/responses endpoint — use chat_completion instead, which works on every provider",
                client.provider()
            ),
            ErrorCode::InvalidInput,
            "this provider does not support create_response/start_conversation/continue_conversation",
        );
    }
    let model = match model {
        Some(m) => m,
        None => match auto_detect_model(client).await {
            Ok(m) => m,
            // Preserves the real error code (connection/auth failure vs.
            // genuinely no/too many models loaded) instead of reporting
            // every auto-detection failure as ModelNotLoaded.
            Err(e) => {
                return ToolResult::err(e.to_string(), e.code(), "model auto-detection failed")
            }
        },
    };
    body.insert("model".into(), Value::String(model.clone()));
    // `ApiClient::responses` always requests this with "stream": true
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

            // A terminal `response.failed` event is merged the same way a
            // successful one is (see `merge_responses_stream`/
            // `is_terminal_response_event`) — it has to be, since the
            // stream reader only knows "this is a terminal event", not
            // "this terminal event means success". Check for it here,
            // where the actual meaning is known, rather than reporting an
            // LM Studio-side failure as a successful empty response.
            if let Some(detail) = failure_detail(&data, status) {
                return ToolResult::err(
                    format!("Request failed: {detail}"),
                    ErrorCode::Unknown,
                    detail,
                );
            }

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
        Err(e) => ToolResult::err(format!("Request failed: {e}"), e.code(), e.to_string()),
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
    client: &ApiClient,
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
    client: &ApiClient,
    personas: &PersonaCache,
    input: StartConversationInput,
) -> ToolResult<ResponseData> {
    let mut body = Map::new();
    body.insert("input".into(), Value::String(input.first_message));
    body.insert(
        "instructions".into(),
        Value::String(input.system_prompt.clone()),
    );
    body.insert(
        "temperature".into(),
        Value::from(input.temperature.unwrap_or(0.7)),
    );
    body.insert(
        "max_output_tokens".into(),
        Value::from(input.max_output_tokens.unwrap_or(2048)),
    );

    let result = send_responses_request(client, body, input.model).await;
    if let Some(data) = &result.data {
        if !data.response_id.is_empty() {
            personas.set(data.response_id.clone(), input.system_prompt);
        }
    }
    result
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
    client: &ApiClient,
    personas: &PersonaCache,
    input: ContinueConversationInput,
) -> ToolResult<ResponseData> {
    // The Responses API doesn't carry `instructions` forward via
    // `previous_response_id` on its own — see `PersonaCache`'s doc comment.
    // Re-send whatever `start_conversation` locked in, if we're still
    // tracking it.
    let persona = personas.get(&input.response_id);

    let mut body = Map::new();
    body.insert("input".into(), Value::String(input.message));
    body.insert(
        "previous_response_id".into(),
        Value::String(input.response_id.clone()),
    );
    if let Some(persona) = &persona {
        body.insert("instructions".into(), Value::String(persona.clone()));
    }
    body.insert(
        "temperature".into(),
        Value::from(input.temperature.unwrap_or(0.7)),
    );
    body.insert(
        "max_output_tokens".into(),
        Value::from(input.max_output_tokens.unwrap_or(2048)),
    );

    let result = send_responses_request(client, body, input.model).await;
    if let Some(persona) = persona {
        if let Some(data) = &result.data {
            if !data.response_id.is_empty() {
                // Propagate to the new response_id so the next continuation
                // in the chain can still find it.
                personas.set(data.response_id.clone(), persona);
            }
        }
        personas.remove(&input.response_id);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persona_cache_round_trips() {
        let cache = PersonaCache::new();
        assert_eq!(cache.get("resp_1"), None);
        cache.set("resp_1".to_string(), "be a pirate".to_string());
        assert_eq!(cache.get("resp_1"), Some("be a pirate".to_string()));
    }

    #[test]
    fn persona_cache_remove_forgets_the_entry() {
        let cache = PersonaCache::new();
        cache.set("resp_1".to_string(), "be a pirate".to_string());
        cache.remove("resp_1");
        assert_eq!(cache.get("resp_1"), None);
    }

    #[test]
    fn persona_cache_evicts_oldest_once_over_capacity() {
        let cache = PersonaCache::new();
        for i in 0..(MAX_TRACKED_CONVERSATIONS + 10) {
            cache.set(format!("resp_{i}"), "persona".to_string());
        }
        // The earliest entries should have been evicted...
        assert_eq!(cache.get("resp_0"), None);
        assert_eq!(cache.get("resp_9"), None);
        // ...but the most recent MAX_TRACKED_CONVERSATIONS are still there.
        assert_eq!(
            cache.get(&format!("resp_{}", MAX_TRACKED_CONVERSATIONS + 9)),
            Some("persona".to_string())
        );
    }

    #[test]
    fn failure_detail_none_for_completed_status() {
        let data = serde_json::json!({"status": "completed"});
        assert_eq!(failure_detail(&data, Some("completed")), None);
    }

    #[test]
    fn failure_detail_none_for_missing_status() {
        let data = serde_json::json!({});
        assert_eq!(failure_detail(&data, None), None);
    }

    #[test]
    fn failure_detail_extracts_error_message_when_failed() {
        let data = serde_json::json!({
            "status": "failed",
            "error": {"message": "the model crashed"}
        });
        assert_eq!(
            failure_detail(&data, Some("failed")),
            Some("the model crashed".to_string())
        );
    }

    #[test]
    fn failure_detail_falls_back_to_generic_message_when_no_error_field() {
        let data = serde_json::json!({"status": "failed"});
        let detail = failure_detail(&data, Some("failed")).unwrap();
        assert!(detail.contains("no further detail"));
    }

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
