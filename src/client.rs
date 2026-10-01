//! HTTP client wrapper around LM Studio's two REST surfaces:
//!
//! - The **native** API (`/api/v1/...`) for model library/management:
//!   listing downloaded models, loading, and unloading.
//! - The **OpenAI-compatible** API (`/v1/...`) for inference: chat
//!   completions, raw completions, embeddings, and the stateful
//!   `/v1/responses` endpoint.
//!
//! Field names for the native API are taken from LM Studio's published
//! REST API reference. Every response struct derives `Default` and marks
//! fields `#[serde(default)]`, and the top-level model entry keeps a
//! `#[serde(flatten)]` catch-all, so a field LM Studio adds, renames, or
//! omits in a future version degrades gracefully instead of failing to parse.

use crate::types::{
    with_timeout, ClientError, DEFAULT_TIMEOUT, STREAM_IDLE_TIMEOUT, STREAM_MAX_DURATION,
};
use reqwest::Client as HttpClient;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use std::time::Duration;

pub mod model {
    use serde::{Deserialize, Serialize};
    use serde_json::{Map, Value};

    /// Response envelope for `GET /api/v1/models`.
    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct ModelsListResponse {
        #[serde(default)]
        pub models: Vec<ModelEntry>,
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct Quantization {
        #[serde(default)]
        pub name: Option<String>,
        #[serde(default)]
        pub bits_per_weight: Option<f64>,
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct ReasoningCapability {
        #[serde(default)]
        pub allowed_options: Vec<String>,
        #[serde(default)]
        pub default: Option<String>,
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct Capabilities {
        #[serde(default)]
        pub vision: bool,
        #[serde(default)]
        pub trained_for_tool_use: bool,
        #[serde(default)]
        pub reasoning: Option<ReasoningCapability>,
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct LoadedInstanceConfig {
        #[serde(default)]
        pub context_length: Option<u64>,
        #[serde(default)]
        pub eval_batch_size: Option<u64>,
        #[serde(default)]
        pub parallel: Option<u64>,
        #[serde(default)]
        pub flash_attention: Option<bool>,
        #[serde(default)]
        pub num_experts: Option<u64>,
        #[serde(default)]
        pub offload_kv_cache_to_gpu: Option<bool>,
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct LoadedInstance {
        pub id: String,
        #[serde(default)]
        pub config: Option<LoadedInstanceConfig>,
    }

    /// One entry in the model library, as returned by `GET /api/v1/models`.
    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct ModelEntry {
        #[serde(rename = "type", default)]
        pub kind: Option<String>,
        #[serde(default)]
        pub publisher: Option<String>,
        /// Unique key used to reference this model when loading it.
        pub key: String,
        #[serde(default)]
        pub display_name: Option<String>,
        #[serde(default)]
        pub architecture: Option<String>,
        #[serde(default)]
        pub quantization: Option<Quantization>,
        #[serde(default)]
        pub size_bytes: Option<u64>,
        #[serde(default)]
        pub params_string: Option<String>,
        #[serde(default)]
        pub loaded_instances: Vec<LoadedInstance>,
        #[serde(default)]
        pub max_context_length: Option<u64>,
        #[serde(default)]
        pub format: Option<String>,
        #[serde(default)]
        pub capabilities: Option<Capabilities>,
        /// Anything LM Studio sends that the fields above don't account for yet.
        #[serde(flatten)]
        pub extra: Map<String, Value>,
    }

    /// Response from `POST /api/v1/models/load`.
    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct LoadModelResponse {
        #[serde(rename = "type", default)]
        pub kind: Option<String>,
        #[serde(default)]
        pub instance_id: Option<String>,
        #[serde(default)]
        pub load_time_seconds: Option<f64>,
        #[serde(default)]
        pub status: Option<String>,
        #[serde(default)]
        pub load_config: Option<LoadedInstanceConfig>,
        #[serde(flatten)]
        pub extra: Map<String, Value>,
    }

    /// Response from `POST /api/v1/models/unload`.
    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct UnloadModelResponse {
        #[serde(default)]
        pub instance_id: Option<String>,
        #[serde(flatten)]
        pub extra: Map<String, Value>,
    }
}

pub struct LmStudioClient {
    http: HttpClient,
    pub native_base_url: String,
    pub openai_base_url: String,
    api_token: Option<String>,
}

impl LmStudioClient {
    pub fn new(config: &crate::config::Config) -> Self {
        let http = HttpClient::builder()
            .build()
            .expect("failed to build HTTP client");
        Self {
            http,
            native_base_url: config.native_base_url.clone(),
            openai_base_url: config.openai_base_url.clone(),
            api_token: config.api_token.clone(),
        }
    }

    fn auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_token {
            Some(token) => builder.bearer_auth(token),
            None => builder,
        }
    }

    async fn get_json<T: DeserializeOwned>(
        &self,
        url: &str,
        timeout: Duration,
    ) -> Result<T, ClientError> {
        with_timeout(
            async {
                let resp = self.auth(self.http.get(url)).send().await.map_err(|e| {
                    if e.is_connect() {
                        ClientError::Connect(e)
                    } else {
                        ClientError::Request(e)
                    }
                })?;
                Self::decode(resp).await
            },
            timeout,
        )
        .await
    }

    async fn post_json<B: Serialize, T: DeserializeOwned>(
        &self,
        url: &str,
        body: &B,
        timeout: Duration,
    ) -> Result<T, ClientError> {
        with_timeout(
            async {
                let resp = self
                    .auth(self.http.post(url).json(body))
                    .send()
                    .await
                    .map_err(|e| {
                        if e.is_connect() {
                            ClientError::Connect(e)
                        } else {
                            ClientError::Request(e)
                        }
                    })?;
                Self::decode(resp).await
            },
            timeout,
        )
        .await
    }

    /// POST without decoding the body — for callers that need the raw
    /// streaming `reqwest::Response` (SSE) rather than a parsed JSON value.
    /// The connect/send phase still gets `DEFAULT_TIMEOUT`; reading the
    /// (potentially long-running) body is the caller's responsibility.
    async fn post_for_stream<B: Serialize>(
        &self,
        url: &str,
        body: &B,
    ) -> Result<reqwest::Response, ClientError> {
        with_timeout(
            async {
                self.auth(self.http.post(url).json(body))
                    .send()
                    .await
                    .map_err(|e| {
                        if e.is_connect() {
                            ClientError::Connect(e)
                        } else {
                            ClientError::Request(e)
                        }
                    })
            },
            DEFAULT_TIMEOUT,
        )
        .await
    }

    async fn decode<T: DeserializeOwned>(resp: reqwest::Response) -> Result<T, ClientError> {
        let status = resp.status();
        let body = resp.text().await.map_err(ClientError::Request)?;
        if !status.is_success() {
            return Err(ClientError::Status {
                status: status.as_u16(),
                body,
            });
        }
        serde_json::from_str(&body).map_err(ClientError::Decode)
    }

    /// Check connectivity by listing the model library.
    pub async fn health_check(&self) -> Result<model::ModelsListResponse, ClientError> {
        let url = format!("{}/models", self.native_base_url);
        self.get_json(&url, DEFAULT_TIMEOUT).await
    }

    /// List every model in the local library (downloaded, and possibly loaded).
    pub async fn list_models(&self) -> Result<model::ModelsListResponse, ClientError> {
        let url = format!("{}/models", self.native_base_url);
        self.get_json(&url, DEFAULT_TIMEOUT).await
    }

    /// Load a model into memory.
    pub async fn load_model(&self, body: Value) -> Result<model::LoadModelResponse, ClientError> {
        let url = format!("{}/models/load", self.native_base_url);
        self.post_json(&url, &body, crate::types::LOAD_MODEL_TIMEOUT)
            .await
    }

    /// Unload a loaded model instance from memory.
    pub async fn unload_model(
        &self,
        instance_id: &str,
    ) -> Result<model::UnloadModelResponse, ClientError> {
        let url = format!("{}/models/unload", self.native_base_url);
        let body = serde_json::json!({ "instance_id": instance_id });
        self.post_json(&url, &body, DEFAULT_TIMEOUT).await
    }

    /// POST a chat-completions request to the OpenAI-compatible endpoint.
    ///
    /// Always requested with `"stream": true` internally — regardless of
    /// what the caller's `body` says — so the (potentially long) generation
    /// is bounded by inter-chunk idle time rather than total duration. The
    /// MCP tool call itself still returns a single result: streamed deltas
    /// are reassembled here into the same shape a non-streaming response
    /// would have had, so callers above this layer don't need to know the
    /// difference.
    pub async fn chat_completion(&self, mut body: Value) -> Result<Value, ClientError> {
        let url = format!("{}/chat/completions", self.openai_base_url);
        body["stream"] = Value::Bool(true);
        let resp = self.post_for_stream(&url, &body).await?;
        let events =
            crate::sse::collect_events(resp, STREAM_IDLE_TIMEOUT, STREAM_MAX_DURATION, |_| false)
                .await?;
        Ok(merge_chat_stream(&events))
    }

    /// POST a legacy text-completions request to the OpenAI-compatible
    /// endpoint. Streamed internally; see [`Self::chat_completion`].
    pub async fn text_completion(&self, mut body: Value) -> Result<Value, ClientError> {
        let url = format!("{}/completions", self.openai_base_url);
        body["stream"] = Value::Bool(true);
        let resp = self.post_for_stream(&url, &body).await?;
        let events =
            crate::sse::collect_events(resp, STREAM_IDLE_TIMEOUT, STREAM_MAX_DURATION, |_| false)
                .await?;
        Ok(merge_text_stream(&events))
    }

    /// POST an embeddings request to the OpenAI-compatible endpoint.
    /// Embedding generation is a single fast forward pass, not open-ended
    /// token generation, so this stays non-streaming.
    pub async fn embeddings(&self, body: Value) -> Result<Value, ClientError> {
        let url = format!("{}/embeddings", self.openai_base_url);
        self.post_json(&url, &body, crate::types::INFERENCE_TIMEOUT)
            .await
    }

    /// POST a stateful responses request to the OpenAI-compatible endpoint.
    /// Streamed internally; see [`Self::chat_completion`].
    pub async fn responses(&self, mut body: Value) -> Result<Value, ClientError> {
        let url = format!("{}/responses", self.openai_base_url);
        let requested_model = body
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or_default()
            .to_string();
        body["stream"] = Value::Bool(true);
        let resp = self.post_for_stream(&url, &body).await?;
        let events = crate::sse::collect_events(
            resp,
            STREAM_IDLE_TIMEOUT,
            STREAM_MAX_DURATION,
            is_terminal_response_event,
        )
        .await?;
        Ok(merge_responses_stream(&events, &requested_model))
    }
}

/// Reassemble streamed chat-completion chunks into the same shape a
/// non-streaming `/v1/chat/completions` response has:
/// `{"model", "choices":[{"message":{"content","reasoning_content"},"finish_reason"}]}`.
/// Each chunk's `choices[0].delta.{content,reasoning_content}` is an
/// incremental piece, not the cumulative text, so these are concatenated.
/// Accumulates one streamed tool call across however many delta chunks it
/// arrives in. Per the OpenAI streaming shape, a tool call's `id`/`type`/
/// `function.name` typically arrive whole in the chunk that introduces it
/// (identified by `index`) and `function.arguments` arrives as successive
/// string fragments to concatenate — but fragments are appended rather than
/// overwritten for every field here, since that's correct either way.
#[derive(Default)]
struct ToolCallAccumulator {
    id: String,
    kind: String,
    name: String,
    arguments: String,
}

fn merge_chat_stream(events: &[Value]) -> Value {
    let mut content = String::new();
    let mut reasoning_content = String::new();
    let mut finish_reason: Option<String> = None;
    let mut model: Option<String> = None;
    let mut tool_calls: std::collections::BTreeMap<u64, ToolCallAccumulator> =
        std::collections::BTreeMap::new();

    for event in events {
        if model.is_none() {
            model = event
                .get("model")
                .and_then(|m| m.as_str())
                .map(String::from);
        }
        let Some(choice) = event
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
        else {
            continue;
        };
        if let Some(delta) = choice.get("delta") {
            if let Some(s) = delta.get("content").and_then(|c| c.as_str()) {
                content.push_str(s);
            }
            if let Some(s) = delta.get("reasoning_content").and_then(|c| c.as_str()) {
                reasoning_content.push_str(s);
            }
            if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tcs {
                    let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                    let entry = tool_calls.entry(index).or_default();
                    // `id`/`type`/`name` are atomic identifying fields that
                    // some servers (observed: LM Studio) resend unchanged on
                    // every delta for a call, not just its first — so these
                    // are "first write wins", never concatenated. Only
                    // `arguments` is genuinely streamed piecewise text.
                    if entry.id.is_empty() {
                        if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                            entry.id.push_str(id);
                        }
                    }
                    if entry.kind.is_empty() {
                        if let Some(kind) = tc.get("type").and_then(|v| v.as_str()) {
                            entry.kind.push_str(kind);
                        }
                    }
                    if let Some(func) = tc.get("function") {
                        if entry.name.is_empty() {
                            if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                                entry.name.push_str(name);
                            }
                        }
                        if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                            entry.arguments.push_str(args);
                        }
                    }
                }
            }
        }
        if let Some(fr) = choice.get("finish_reason").and_then(|f| f.as_str()) {
            finish_reason = Some(fr.to_string());
        }
    }

    let mut message = serde_json::json!({ "content": content });
    if !reasoning_content.is_empty() {
        message["reasoning_content"] = Value::String(reasoning_content);
    }
    if !tool_calls.is_empty() {
        let calls: Vec<Value> = tool_calls
            .into_values()
            .map(|acc| {
                serde_json::json!({
                    "id": acc.id,
                    "type": if acc.kind.is_empty() { "function".to_string() } else { acc.kind },
                    "function": { "name": acc.name, "arguments": acc.arguments },
                })
            })
            .collect();
        message["tool_calls"] = Value::Array(calls);
    }
    serde_json::json!({
        "model": model,
        "choices": [{ "message": message, "finish_reason": finish_reason }],
    })
}

/// Reassemble streamed legacy-completion chunks into the same shape a
/// non-streaming `/v1/completions` response has:
/// `{"model", "choices":[{"text","finish_reason"}]}`. Each chunk's
/// `choices[0].text` is an incremental piece, concatenated here.
fn merge_text_stream(events: &[Value]) -> Value {
    let mut text = String::new();
    let mut finish_reason: Option<String> = None;
    let mut model: Option<String> = None;

    for event in events {
        if model.is_none() {
            model = event
                .get("model")
                .and_then(|m| m.as_str())
                .map(String::from);
        }
        let Some(choice) = event
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
        else {
            continue;
        };
        if let Some(s) = choice.get("text").and_then(|t| t.as_str()) {
            text.push_str(s);
        }
        if let Some(fr) = choice.get("finish_reason").and_then(|f| f.as_str()) {
            finish_reason = Some(fr.to_string());
        }
    }

    serde_json::json!({
        "model": model,
        "choices": [{ "text": text, "finish_reason": finish_reason }],
    })
}

fn is_terminal_response_event(event: &Value) -> bool {
    matches!(
        event.get("type").and_then(|t| t.as_str()),
        Some(t) if t.starts_with("response.completed")
            || t.starts_with("response.incomplete")
            || t.starts_with("response.failed")
    )
}

/// Reassemble streamed `/v1/responses` events into the full response object
/// `extract_output` (in `tools::responses`) already knows how to read.
///
/// Prefers the `response` object carried by the terminal
/// `response.completed` / `.incomplete` / `.failed` event, which (per the
/// Responses API spec this endpoint mirrors) is the complete final object —
/// identical to what the non-streaming call would have returned. Falls back
/// to reconstructing one from `*.delta` events if no terminal event arrived
/// (defensive: LM Studio's native streaming shape for this newer endpoint
/// isn't exhaustively documented at the time of writing).
fn merge_responses_stream(events: &[Value], fallback_model: &str) -> Value {
    if let Some(resp) = events
        .iter()
        .rev()
        .find(|e| is_terminal_response_event(e))
        .and_then(|e| e.get("response"))
    {
        return resp.clone();
    }

    let mut message = String::new();
    let mut reasoning = String::new();
    let mut id = String::new();
    let mut model = String::new();

    for event in events {
        let kind = event.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if let Some(delta) = event.get("delta").and_then(|d| d.as_str()) {
            if kind.contains("reasoning") {
                reasoning.push_str(delta);
            } else if kind.contains("output_text") {
                message.push_str(delta);
            }
        }
        if id.is_empty() {
            if let Some(v) = event
                .get("response_id")
                .or_else(|| event.get("id"))
                .and_then(|v| v.as_str())
            {
                id = v.to_string();
            }
        }
        if model.is_empty() {
            if let Some(v) = event.get("model").and_then(|v| v.as_str()) {
                model = v.to_string();
            }
        }
    }
    if model.is_empty() {
        model = fallback_model.to_string();
    }

    let mut output = Vec::new();
    if !reasoning.is_empty() {
        output.push(serde_json::json!({
            "type": "reasoning",
            "content": [{ "type": "reasoning_text", "text": reasoning }],
        }));
    }
    if !message.is_empty() {
        output.push(serde_json::json!({
            "type": "message",
            "content": [{ "type": "output_text", "text": message }],
        }));
    }

    serde_json::json!({ "id": id, "model": model, "output": output })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_chat_stream_deltas_including_reasoning_content() {
        let events = vec![
            serde_json::json!({"model": "m", "choices": [{"delta": {"role": "assistant"}}]}),
            serde_json::json!({"choices": [{"delta": {"reasoning_content": "thinking "}}]}),
            serde_json::json!({"choices": [{"delta": {"reasoning_content": "more"}}]}),
            serde_json::json!({"choices": [{"delta": {"content": "Hello"}}]}),
            serde_json::json!({"choices": [{"delta": {"content": " world"}}]}),
            serde_json::json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ];

        let merged = merge_chat_stream(&events);
        assert_eq!(merged["model"], "m");
        assert_eq!(merged["choices"][0]["message"]["content"], "Hello world");
        assert_eq!(
            merged["choices"][0]["message"]["reasoning_content"],
            "thinking more"
        );
        assert_eq!(merged["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn merges_streamed_tool_call_fragments_by_index() {
        let events = vec![
            serde_json::json!({"model": "m", "choices": [{"delta": {
                "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": ""}}]
            }}]}),
            serde_json::json!({"choices": [{"delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": "{\"path\":"}}]
            }}]}),
            serde_json::json!({"choices": [{"delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": "\"a.txt\"}"}}]
            }}]}),
            serde_json::json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        ];

        let merged = merge_chat_stream(&events);
        let tool_calls = merged["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "call_1");
        assert_eq!(tool_calls[0]["type"], "function");
        assert_eq!(tool_calls[0]["function"]["name"], "read_file");
        assert_eq!(
            tool_calls[0]["function"]["arguments"],
            "{\"path\":\"a.txt\"}"
        );
        assert_eq!(merged["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn does_not_duplicate_id_and_type_when_a_server_resends_them_on_every_chunk() {
        // Regression test: observed live against LM Studio — unlike OpenAI's
        // reference streaming shape (id/type sent once, on the chunk that
        // introduces the call), LM Studio resends "id" and "type" unchanged
        // on every delta chunk for a tool call, not just its first. Naively
        // concatenating them (as if they were streamed text, like
        // `arguments` genuinely is) corrupted "function" into
        // "functionfunction" and broke every multi-turn subagent run.
        let events = vec![
            serde_json::json!({"choices": [{"delta": {
                "tool_calls": [{"index": 0, "id": "340306449", "type": "function", "function": {"name": "list_directory", "arguments": "{}"}}]
            }}]}),
            serde_json::json!({"choices": [{"delta": {
                "tool_calls": [{"index": 0, "id": "340306449", "type": "function", "function": {"arguments": ""}}]
            }}]}),
        ];
        let merged = merge_chat_stream(&events);
        let tool_calls = merged["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(tool_calls[0]["id"], "340306449");
        assert_eq!(tool_calls[0]["type"], "function");
        assert_eq!(tool_calls[0]["function"]["name"], "list_directory");
    }

    #[test]
    fn merges_multiple_parallel_tool_calls_by_distinct_index() {
        let events = vec![
            serde_json::json!({"choices": [{"delta": {
                "tool_calls": [{"index": 0, "id": "call_a", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
            }}]}),
            serde_json::json!({"choices": [{"delta": {
                "tool_calls": [{"index": 1, "id": "call_b", "type": "function", "function": {"name": "list_directory", "arguments": "{}"}}]
            }}]}),
        ];
        let merged = merge_chat_stream(&events);
        let tool_calls = merged["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0]["function"]["name"], "read_file");
        assert_eq!(tool_calls[1]["function"]["name"], "list_directory");
    }

    #[test]
    fn no_tool_calls_key_when_none_streamed() {
        let events = vec![serde_json::json!({"choices": [{"delta": {"content": "hi"}}]})];
        let merged = merge_chat_stream(&events);
        assert!(merged["choices"][0]["message"].get("tool_calls").is_none());
    }

    #[test]
    fn merges_chat_stream_without_reasoning_content_key_when_absent() {
        let events = vec![
            serde_json::json!({"model": "m", "choices": [{"delta": {"content": "hi"}}]}),
            serde_json::json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ];
        let merged = merge_chat_stream(&events);
        assert_eq!(merged["choices"][0]["message"]["content"], "hi");
        assert!(merged["choices"][0]["message"]
            .get("reasoning_content")
            .is_none());
    }

    #[test]
    fn merges_text_completion_stream_deltas() {
        let events = vec![
            serde_json::json!({"model": "m", "choices": [{"text": "Once "}]}),
            serde_json::json!({"choices": [{"text": "upon a time"}]}),
            serde_json::json!({"choices": [{"text": "", "finish_reason": "length"}]}),
        ];
        let merged = merge_text_stream(&events);
        assert_eq!(merged["model"], "m");
        assert_eq!(merged["choices"][0]["text"], "Once upon a time");
        assert_eq!(merged["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn prefers_the_terminal_event_full_response_object_when_present() {
        let full_response =
            serde_json::json!({"id": "resp_1", "model": "m", "output": [{"type": "message"}]});
        let events = vec![
            serde_json::json!({"type": "response.output_text.delta", "delta": "ignored"}),
            serde_json::json!({"type": "response.completed", "response": full_response.clone()}),
        ];
        let merged = merge_responses_stream(&events, "fallback-model");
        assert_eq!(merged, full_response);
    }

    #[test]
    fn falls_back_to_reconstructing_from_deltas_when_no_terminal_event() {
        let events = vec![
            serde_json::json!({"type": "response.reasoning_text.delta", "delta": "thinking"}),
            serde_json::json!({"type": "response.output_text.delta", "delta": "Hi there"}),
            serde_json::json!({"type": "response.output_text.delta", "delta": "!"}),
        ];
        let merged = merge_responses_stream(&events, "fallback-model");
        assert_eq!(merged["model"], "fallback-model");
        let out = extract_output_for_test(&merged);
        assert_eq!(out.0, "Hi there!");
        assert_eq!(out.1.as_deref(), Some("thinking"));
    }

    #[test]
    fn is_terminal_response_event_matches_all_three_terminal_types() {
        for t in [
            "response.completed",
            "response.incomplete",
            "response.failed",
        ] {
            assert!(is_terminal_response_event(&serde_json::json!({"type": t})));
        }
        assert!(!is_terminal_response_event(
            &serde_json::json!({"type": "response.output_text.delta"})
        ));
    }

    // Local re-implementation of the bit of tools::responses::extract_output
    // this test needs, to verify merge_responses_stream's fallback output is
    // actually shaped the way that function reads — without creating a
    // cross-module test dependency.
    fn extract_output_for_test(data: &Value) -> (String, Option<String>) {
        let mut message = String::new();
        let mut reasoning = None;
        if let Some(arr) = data.get("output").and_then(|o| o.as_array()) {
            for block in arr {
                match block.get("type").and_then(|t| t.as_str()) {
                    Some("message") => {
                        if let Some(text) = block
                            .get("content")
                            .and_then(|c| c.as_array())
                            .and_then(|a| a.first())
                            .and_then(|i| i.get("text"))
                            .and_then(|t| t.as_str())
                        {
                            message.push_str(text);
                        }
                    }
                    Some("reasoning") => {
                        if let Some(text) = block
                            .get("content")
                            .and_then(|c| c.as_array())
                            .and_then(|a| a.first())
                            .and_then(|i| i.get("text"))
                            .and_then(|t| t.as_str())
                        {
                            reasoning = Some(text.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
        (message, reasoning)
    }
}
