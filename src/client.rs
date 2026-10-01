//! HTTP client wrapper over whichever backend `Provider` this server is
//! configured for (see `crate::providers`, `crate::config`). Presents one
//! API shape to callers regardless of which provider is active:
//!
//! - `list_models` / `load_model` / `unload_model`: native model-management
//!   APIs. LM Studio uses its own `/api/v1/...` surface; Ollama's native API
//!   (`/api/tags`, `/api/ps`, `/api/generate`) is translated into the same
//!   `model::ModelsListResponse` shape LM Studio returns. OpenAI/Anthropic
//!   have no such concept — callers must check
//!   `Provider::supports_model_management` before calling these.
//! - `chat_completion`: OpenAI-compatible `/v1/chat/completions` for LM
//!   Studio/Ollama/OpenAI; translated to/from Anthropic's `/v1/messages`
//!   shape for Anthropic, so every caller above this layer sees the same
//!   `{"model", "choices":[{"message":{...},"finish_reason"}]}` shape either way.
//! - `text_completion` / `embeddings` / `responses`: OpenAI-compatible only;
//!   callers check `Provider::supports_embeddings` /
//!   `Provider::supports_responses_api` first.
//!
//! Field names for LM Studio's native API are taken from its published REST
//! API reference. Every response struct derives `Default` and marks fields
//! `#[serde(default)]`, and the top-level model entry keeps a
//! `#[serde(flatten)]` catch-all, so a field a backend adds, renames, or
//! omits degrades gracefully instead of failing to parse.

use crate::providers::{Provider, WireFormat};
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

    /// Response envelope for `GET /api/v1/models` (and the translated shape
    /// every other provider's model listing is normalized into).
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

    /// One entry in the model library, as returned by `GET /api/v1/models`
    /// (LM Studio) or translated from a provider's own listing shape.
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
        /// Anything a provider sends that the fields above don't account for yet.
        #[serde(flatten)]
        pub extra: Map<String, Value>,
    }

    /// Response from `POST /api/v1/models/load` (and the translated shape
    /// for Ollama's native load path).
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

    /// Response from `POST /api/v1/models/unload` (and the translated shape
    /// for Ollama's native unload path).
    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct UnloadModelResponse {
        #[serde(default)]
        pub instance_id: Option<String>,
        #[serde(flatten)]
        pub extra: Map<String, Value>,
    }
}

pub struct ApiClient {
    http: HttpClient,
    provider: Provider,
    /// `scheme://host:port`, no trailing slash, no `/v1` or `/api/...` suffix.
    base_url: String,
    api_key: Option<String>,
}

impl ApiClient {
    pub fn new(config: &crate::config::Config) -> Self {
        let http = HttpClient::builder()
            .build()
            .expect("failed to build HTTP client");
        Self {
            http,
            provider: config.provider,
            base_url: config.base_url.clone(),
            api_key: config.api_key.clone(),
        }
    }

    pub fn provider(&self) -> Provider {
        self.provider
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The `/v1`-rooted API every provider mounts inference under.
    fn v1_url(&self) -> String {
        format!("{}/v1", self.base_url)
    }

    /// The native model-management API. Only `LmStudio`/`Ollama` have one —
    /// callers must check `Provider::supports_model_management` before
    /// reaching any code path that calls this.
    fn native_url(&self) -> String {
        match self.provider {
            Provider::LmStudio => format!("{}/api/v1", self.base_url),
            Provider::Ollama => format!("{}/api", self.base_url),
            Provider::OpenAi | Provider::Anthropic => {
                unreachable!("{} has no native model-management API", self.provider)
            }
        }
    }

    /// Anthropic authenticates with `x-api-key` + a required
    /// `anthropic-version` header instead of `Authorization: Bearer`.
    fn auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match (self.provider, &self.api_key) {
            (Provider::Anthropic, Some(key)) => builder
                .header("x-api-key", key)
                .header("anthropic-version", "2023-06-01"),
            (_, Some(key)) => builder.bearer_auth(key),
            (_, None) => builder,
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
        self.list_models().await
    }

    /// List every model this provider makes available. For `LmStudio`/
    /// `Ollama` this is the local library (downloaded, and possibly
    /// loaded); for `OpenAi`/`Anthropic` it's every model the API key can
    /// use (always "available", nothing to load).
    pub async fn list_models(&self) -> Result<model::ModelsListResponse, ClientError> {
        match self.provider {
            Provider::LmStudio => {
                let url = format!("{}/models", self.native_url());
                self.get_json(&url, DEFAULT_TIMEOUT).await
            }
            Provider::Ollama => self.list_models_ollama().await,
            Provider::OpenAi | Provider::Anthropic => self.list_models_v1().await,
        }
    }

    async fn list_models_ollama(&self) -> Result<model::ModelsListResponse, ClientError> {
        #[derive(Debug, Default, serde::Deserialize)]
        struct OllamaDetails {
            #[serde(default)]
            family: Option<String>,
            #[serde(default)]
            parameter_size: Option<String>,
            #[serde(default)]
            quantization_level: Option<String>,
            #[serde(default)]
            format: Option<String>,
        }
        #[derive(Debug, Default, serde::Deserialize)]
        struct OllamaModel {
            name: String,
            #[serde(default)]
            size: Option<u64>,
            #[serde(default)]
            details: Option<OllamaDetails>,
        }
        #[derive(Debug, Default, serde::Deserialize)]
        struct OllamaList {
            #[serde(default)]
            models: Vec<OllamaModel>,
        }

        let tags_url = format!("{}/tags", self.native_url());
        let tags: OllamaList = self.get_json(&tags_url, DEFAULT_TIMEOUT).await?;

        // Best-effort: if `/api/ps` fails for some reason, still return the
        // library listing rather than failing the whole call — we just
        // won't know which of them are currently loaded.
        let ps_url = format!("{}/ps", self.native_url());
        let loaded_names: std::collections::HashSet<String> = self
            .get_json::<OllamaList>(&ps_url, DEFAULT_TIMEOUT)
            .await
            .map(|ps| ps.models.into_iter().map(|m| m.name).collect())
            .unwrap_or_default();

        let models = tags
            .models
            .into_iter()
            .map(|m| {
                let loaded_instances = if loaded_names.contains(&m.name) {
                    vec![model::LoadedInstance {
                        id: m.name.clone(),
                        config: None,
                    }]
                } else {
                    vec![]
                };
                model::ModelEntry {
                    key: m.name.clone(),
                    display_name: Some(m.name),
                    architecture: m.details.as_ref().and_then(|d| d.family.clone()),
                    params_string: m.details.as_ref().and_then(|d| d.parameter_size.clone()),
                    format: m.details.as_ref().and_then(|d| d.format.clone()),
                    quantization: m
                        .details
                        .as_ref()
                        .and_then(|d| d.quantization_level.clone())
                        .map(|name| model::Quantization {
                            name: Some(name),
                            bits_per_weight: None,
                        }),
                    size_bytes: m.size,
                    loaded_instances,
                    ..Default::default()
                }
            })
            .collect();

        Ok(model::ModelsListResponse { models })
    }

    async fn list_models_v1(&self) -> Result<model::ModelsListResponse, ClientError> {
        #[derive(Debug, Default, serde::Deserialize)]
        struct V1ModelEntry {
            id: String,
            #[serde(default)]
            display_name: Option<String>,
        }
        #[derive(Debug, Default, serde::Deserialize)]
        struct V1ModelsList {
            #[serde(default)]
            data: Vec<V1ModelEntry>,
        }

        let url = format!("{}/models", self.v1_url());
        let resp: V1ModelsList = self.get_json(&url, DEFAULT_TIMEOUT).await?;
        let models = resp
            .data
            .into_iter()
            .map(|m| model::ModelEntry {
                key: m.id.clone(),
                display_name: m.display_name.or(Some(m.id)),
                ..Default::default()
            })
            .collect();
        Ok(model::ModelsListResponse { models })
    }

    /// Load a model into memory. Only `LmStudio`/`Ollama` support this —
    /// callers must check `Provider::supports_model_management` first.
    pub async fn load_model(&self, body: Value) -> Result<model::LoadModelResponse, ClientError> {
        match self.provider {
            Provider::LmStudio => {
                let url = format!("{}/models/load", self.native_url());
                self.post_json(&url, &body, crate::types::LOAD_MODEL_TIMEOUT)
                    .await
            }
            Provider::Ollama => self.load_model_ollama(body).await,
            Provider::OpenAi | Provider::Anthropic => {
                unreachable!("{} has no model management API", self.provider)
            }
        }
    }

    /// Ollama has no explicit "load" call — sending an empty-prompt,
    /// non-streaming generate request loads the model into memory and
    /// returns once it's ready, without generating anything.
    async fn load_model_ollama(
        &self,
        body: Value,
    ) -> Result<model::LoadModelResponse, ClientError> {
        let model_name = body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let url = format!("{}/generate", self.native_url());
        let req = serde_json::json!({ "model": model_name, "prompt": "", "stream": false });

        #[derive(Debug, Default, serde::Deserialize)]
        struct OllamaGenerateResp {
            #[serde(default)]
            done: bool,
            #[serde(default)]
            total_duration: Option<u64>,
        }
        let resp: OllamaGenerateResp = self
            .post_json(&url, &req, crate::types::LOAD_MODEL_TIMEOUT)
            .await?;

        Ok(model::LoadModelResponse {
            instance_id: Some(model_name),
            status: Some(if resp.done {
                "loaded".to_string()
            } else {
                "unknown".to_string()
            }),
            // Ollama reports this in nanoseconds.
            load_time_seconds: resp.total_duration.map(|ns| ns as f64 / 1e9),
            ..Default::default()
        })
    }

    /// Unload a loaded model instance from memory. Only `LmStudio`/`Ollama`
    /// support this — callers must check `Provider::supports_model_management`
    /// first.
    pub async fn unload_model(
        &self,
        instance_id: &str,
    ) -> Result<model::UnloadModelResponse, ClientError> {
        match self.provider {
            Provider::LmStudio => {
                let url = format!("{}/models/unload", self.native_url());
                let body = serde_json::json!({ "instance_id": instance_id });
                self.post_json(&url, &body, DEFAULT_TIMEOUT).await
            }
            // Ollama unloads a model by asking it to generate with
            // `keep_alive: 0`, which evicts it from memory immediately
            // once the (empty, here) request completes.
            Provider::Ollama => {
                let url = format!("{}/generate", self.native_url());
                let body = serde_json::json!({
                    "model": instance_id,
                    "prompt": "",
                    "stream": false,
                    "keep_alive": 0,
                });
                let _resp: Value = self.post_json(&url, &body, DEFAULT_TIMEOUT).await?;
                Ok(model::UnloadModelResponse {
                    instance_id: Some(instance_id.to_string()),
                    ..Default::default()
                })
            }
            Provider::OpenAi | Provider::Anthropic => {
                unreachable!("{} has no model management API", self.provider)
            }
        }
    }

    /// POST a chat-completions request.
    ///
    /// Always requested with server-side streaming internally — regardless
    /// of what the caller's `body` says — so the (potentially long)
    /// generation is bounded by inter-chunk idle time rather than total
    /// duration. The MCP tool call itself still returns a single result:
    /// streamed deltas are reassembled here into the same shape a
    /// non-streaming response would have had, so callers above this layer
    /// don't need to know the difference — or which provider is active.
    pub async fn chat_completion(&self, body: Value) -> Result<Value, ClientError> {
        match self.provider.wire_format() {
            WireFormat::OpenAiCompatible => self.chat_completion_openai(body).await,
            WireFormat::Anthropic => self.chat_completion_anthropic(body).await,
        }
    }

    async fn chat_completion_openai(&self, mut body: Value) -> Result<Value, ClientError> {
        let url = format!("{}/chat/completions", self.v1_url());
        body["stream"] = Value::Bool(true);
        let resp = self.post_for_stream(&url, &body).await?;
        let events =
            crate::sse::collect_events(resp, STREAM_IDLE_TIMEOUT, STREAM_MAX_DURATION, |_| false)
                .await?;
        Ok(merge_chat_stream(&events))
    }

    async fn chat_completion_anthropic(&self, body: Value) -> Result<Value, ClientError> {
        let url = format!("{}/messages", self.v1_url());
        let translated = translate_openai_request_to_anthropic(body);
        let resp = self.post_for_stream(&url, &translated).await?;
        let events = crate::sse::collect_events(
            resp,
            STREAM_IDLE_TIMEOUT,
            STREAM_MAX_DURATION,
            is_anthropic_terminal_event,
        )
        .await?;
        Ok(merge_anthropic_stream(&events))
    }

    /// POST a legacy text-completions request to the OpenAI-compatible
    /// endpoint. Streamed internally; see [`Self::chat_completion`]. Not
    /// offered by Anthropic — callers should prefer `chat_completion`,
    /// which works on every provider.
    pub async fn text_completion(&self, mut body: Value) -> Result<Value, ClientError> {
        let url = format!("{}/completions", self.v1_url());
        body["stream"] = Value::Bool(true);
        let resp = self.post_for_stream(&url, &body).await?;
        let events =
            crate::sse::collect_events(resp, STREAM_IDLE_TIMEOUT, STREAM_MAX_DURATION, |_| false)
                .await?;
        Ok(merge_text_stream(&events))
    }

    /// POST an embeddings request to the OpenAI-compatible endpoint.
    /// Embedding generation is a single fast forward pass, not open-ended
    /// token generation, so this stays non-streaming. Callers must check
    /// `Provider::supports_embeddings` first — Anthropic has none.
    pub async fn embeddings(&self, body: Value) -> Result<Value, ClientError> {
        let url = format!("{}/embeddings", self.v1_url());
        self.post_json(&url, &body, crate::types::INFERENCE_TIMEOUT)
            .await
    }

    /// POST a stateful responses request to the OpenAI-compatible endpoint.
    /// Streamed internally; see [`Self::chat_completion`]. Callers must
    /// check `Provider::supports_responses_api` first — only LM Studio and
    /// OpenAI have this endpoint.
    pub async fn responses(&self, mut body: Value) -> Result<Value, ClientError> {
        let url = format!("{}/responses", self.v1_url());
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

// ---------------------------------------------------------------------------
// Anthropic request/response translation
// ---------------------------------------------------------------------------

/// Translate an OpenAI-shaped `/v1/chat/completions` request body into
/// Anthropic's `/v1/messages` shape: `system` becomes a top-level field
/// (extracted out of `messages`), `max_tokens` is made required (Anthropic
/// rejects a request without it), tool definitions move from
/// `function.parameters` to `input_schema`, and OpenAI's `tool`-role
/// messages / assistant `tool_calls` become Anthropic's `tool_result` /
/// `tool_use` content blocks.
fn translate_openai_request_to_anthropic(body: Value) -> Value {
    let obj = body.as_object().cloned().unwrap_or_default();
    let model = obj.get("model").cloned().unwrap_or(Value::Null);
    let max_tokens = obj.get("max_tokens").cloned().unwrap_or(Value::from(4096));
    let temperature = obj.get("temperature").cloned();

    let mut system = String::new();
    let mut messages = Vec::new();

    if let Some(Value::Array(msgs)) = obj.get("messages") {
        for m in msgs {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            match role {
                "system" => {
                    if let Some(s) = m.get("content").and_then(|c| c.as_str()) {
                        if !system.is_empty() {
                            system.push('\n');
                        }
                        system.push_str(s);
                    }
                }
                "tool" => {
                    let tool_use_id = m
                        .get("tool_call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let content = m
                        .get("content")
                        .and_then(|c| c.as_str())
                        .unwrap_or_default()
                        .to_string();
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}]
                    }));
                }
                "assistant" => {
                    let mut blocks = Vec::new();
                    if let Some(text) = m.get("content").and_then(|c| c.as_str()) {
                        if !text.is_empty() {
                            blocks.push(serde_json::json!({"type": "text", "text": text}));
                        }
                    }
                    if let Some(tcs) = m.get("tool_calls").and_then(|v| v.as_array()) {
                        for tc in tcs {
                            let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or_default();
                            let func = tc.get("function");
                            let name = func
                                .and_then(|f| f.get("name"))
                                .and_then(|v| v.as_str())
                                .unwrap_or_default();
                            let args_str = func
                                .and_then(|f| f.get("arguments"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("{}");
                            let input: Value = serde_json::from_str(args_str)
                                .unwrap_or_else(|_| serde_json::json!({}));
                            blocks.push(
                                serde_json::json!({"type": "tool_use", "id": id, "name": name, "input": input}),
                            );
                        }
                    }
                    messages.push(serde_json::json!({"role": "assistant", "content": blocks}));
                }
                _ => {
                    let content = m
                        .get("content")
                        .cloned()
                        .unwrap_or(Value::String(String::new()));
                    messages.push(serde_json::json!({"role": "user", "content": content}));
                }
            }
        }
    }

    let mut out = serde_json::json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "stream": true,
    });
    if !system.is_empty() {
        out["system"] = Value::String(system);
    }
    if let Some(t) = temperature {
        out["temperature"] = t;
    }
    if let Some(tools) = obj.get("tools").and_then(|v| v.as_array()) {
        let translated: Vec<Value> = tools
            .iter()
            .filter_map(|t| {
                let func = t.get("function")?;
                Some(serde_json::json!({
                    "name": func.get("name")?.as_str()?,
                    "description": func.get("description").and_then(|v| v.as_str()).unwrap_or(""),
                    "input_schema": func.get("parameters").cloned()
                        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
                }))
            })
            .collect();
        if !translated.is_empty() {
            out["tools"] = Value::Array(translated);
        }
    }
    if let Some(tc) = obj.get("tool_choice") {
        out["tool_choice"] = translate_tool_choice(tc);
    }
    out
}

fn translate_tool_choice(tc: &Value) -> Value {
    match tc {
        Value::String(s) if s == "auto" => serde_json::json!({"type": "auto"}),
        Value::String(s) if s == "none" => serde_json::json!({"type": "none"}),
        Value::String(s) if s == "required" => serde_json::json!({"type": "any"}),
        Value::Object(_) => match tc
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|v| v.as_str())
        {
            Some(name) => serde_json::json!({"type": "tool", "name": name}),
            None => serde_json::json!({"type": "auto"}),
        },
        _ => serde_json::json!({"type": "auto"}),
    }
}

fn is_anthropic_terminal_event(event: &Value) -> bool {
    event.get("type").and_then(|t| t.as_str()) == Some("message_stop")
}

fn map_anthropic_stop_reason(reason: &str) -> String {
    match reason {
        "end_turn" | "stop_sequence" => "stop",
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        other => other,
    }
    .to_string()
}

/// Reassemble a stream of Anthropic `/v1/messages` SSE events — see the
/// module-level research this was built from: `message_start` once, then
/// repeating `content_block_start`/`content_block_delta`(xN)/
/// `content_block_stop` per content block, then `message_delta`(xN), then
/// `message_stop` — into the same unified shape `merge_chat_stream`
/// produces for the OpenAI-compatible family, so nothing above this layer
/// needs to know which wire format was actually used.
///
/// A `tool_use` content block gets its `id`/`name` immediately in
/// `content_block_start` (unlike OpenAI, which can split them across
/// chunks) and only its `input` streams incrementally, as a JSON string
/// fragment, via `input_json_delta.partial_json` — which already matches
/// the string-concatenation shape `ToolCallAccumulator::arguments` expects,
/// so no JSON parsing is needed until the caller actually consumes it.
fn merge_anthropic_stream(events: &[Value]) -> Value {
    let mut model: Option<String> = None;
    let mut content = String::new();
    let mut reasoning_content = String::new();
    let mut finish_reason: Option<String> = None;
    let mut tool_calls: std::collections::BTreeMap<u64, ToolCallAccumulator> =
        std::collections::BTreeMap::new();

    for event in events {
        match event.get("type").and_then(|t| t.as_str()) {
            Some("message_start") => {
                if model.is_none() {
                    model = event
                        .get("message")
                        .and_then(|m| m.get("model"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                }
            }
            Some("content_block_start") => {
                let index = event.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                if let Some(block) = event.get("content_block") {
                    if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                        let entry = tool_calls.entry(index).or_default();
                        if let Some(id) = block.get("id").and_then(|v| v.as_str()) {
                            entry.id = id.to_string();
                        }
                        entry.kind = "function".to_string();
                        if let Some(name) = block.get("name").and_then(|v| v.as_str()) {
                            entry.name = name.to_string();
                        }
                    }
                }
            }
            Some("content_block_delta") => {
                let index = event.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                if let Some(delta) = event.get("delta") {
                    match delta.get("type").and_then(|t| t.as_str()) {
                        Some("text_delta") => {
                            if let Some(s) = delta.get("text").and_then(|v| v.as_str()) {
                                content.push_str(s);
                            }
                        }
                        Some("thinking_delta") => {
                            if let Some(s) = delta.get("thinking").and_then(|v| v.as_str()) {
                                reasoning_content.push_str(s);
                            }
                        }
                        Some("input_json_delta") => {
                            if let Some(s) = delta.get("partial_json").and_then(|v| v.as_str()) {
                                tool_calls.entry(index).or_default().arguments.push_str(s);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Some("message_delta") => {
                if let Some(sr) = event
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|v| v.as_str())
                {
                    finish_reason = Some(map_anthropic_stop_reason(sr));
                }
            }
            _ => {}
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
                let arguments = if acc.arguments.is_empty() {
                    "{}".to_string()
                } else {
                    acc.arguments
                };
                serde_json::json!({
                    "id": acc.id,
                    "type": if acc.kind.is_empty() { "function".to_string() } else { acc.kind },
                    "function": { "name": acc.name, "arguments": arguments },
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

    #[test]
    fn is_anthropic_terminal_event_only_matches_message_stop() {
        assert!(is_anthropic_terminal_event(
            &serde_json::json!({"type": "message_stop"})
        ));
        assert!(!is_anthropic_terminal_event(
            &serde_json::json!({"type": "content_block_stop"})
        ));
    }

    #[test]
    fn translates_system_message_and_required_max_tokens() {
        let body = serde_json::json!({
            "model": "claude-sonnet-5",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hi"},
            ],
        });
        let translated = translate_openai_request_to_anthropic(body);
        assert_eq!(translated["system"], "Be terse.");
        assert_eq!(translated["messages"].as_array().unwrap().len(), 1);
        assert_eq!(translated["messages"][0]["role"], "user");
        assert_eq!(translated["messages"][0]["content"], "hi");
        // Anthropic rejects a request with no max_tokens — must be filled
        // in even when the caller's body never set one.
        assert!(translated["max_tokens"].as_u64().unwrap() > 0);
    }

    #[test]
    fn translates_tool_definitions_and_tool_use_round_trip() {
        let body = serde_json::json!({
            "model": "claude-sonnet-5",
            "max_tokens": 100,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
                }
            }],
            "messages": [
                {"role": "user", "content": "read a.txt"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"a.txt\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
            ],
        });
        let translated = translate_openai_request_to_anthropic(body);

        assert_eq!(translated["tools"][0]["name"], "read_file");
        assert_eq!(translated["tools"][0]["input_schema"]["type"], "object");

        let messages = translated["messages"].as_array().unwrap();
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"][0]["type"], "tool_use");
        assert_eq!(messages[1]["content"][0]["id"], "call_1");
        assert_eq!(messages[1]["content"][0]["input"]["path"], "a.txt");

        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"][0]["type"], "tool_result");
        assert_eq!(messages[2]["content"][0]["tool_use_id"], "call_1");
        assert_eq!(messages[2]["content"][0]["content"], "file contents");
    }

    #[test]
    fn translates_tool_choice_variants() {
        assert_eq!(
            translate_tool_choice(&serde_json::json!("auto")),
            serde_json::json!({"type": "auto"})
        );
        assert_eq!(
            translate_tool_choice(&serde_json::json!("required")),
            serde_json::json!({"type": "any"})
        );
        assert_eq!(
            translate_tool_choice(
                &serde_json::json!({"type": "function", "function": {"name": "f"}})
            ),
            serde_json::json!({"type": "tool", "name": "f"})
        );
    }

    #[test]
    fn merges_anthropic_text_stream_into_the_shared_shape() {
        // Literal event sequence shape verified against Anthropic's docs:
        // message_start -> content_block_start/delta(xN)/stop -> message_delta -> message_stop.
        let events = vec![
            serde_json::json!({"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-5", "role": "assistant", "content": []}}),
            serde_json::json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
            serde_json::json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}}),
            serde_json::json!({"type": "message_stop"}),
        ];
        let merged = merge_anthropic_stream(&events);
        assert_eq!(merged["model"], "claude-sonnet-5");
        assert_eq!(merged["choices"][0]["message"]["content"], "Hello world");
        assert_eq!(merged["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn merges_anthropic_tool_use_stream_with_id_and_name_from_block_start() {
        // tool_use blocks get id+name immediately in content_block_start
        // (unlike OpenAI, which can split them across chunks); only
        // `input` streams incrementally as a JSON-string fragment.
        let events = vec![
            serde_json::json!({"type": "message_start", "message": {"model": "claude-sonnet-5"}}),
            serde_json::json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}}}),
            serde_json::json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"path\":"}}),
            serde_json::json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "\"a.txt\"}"}}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}}),
            serde_json::json!({"type": "message_stop"}),
        ];
        let merged = merge_anthropic_stream(&events);
        let tool_calls = merged["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "toolu_1");
        assert_eq!(tool_calls[0]["type"], "function");
        assert_eq!(tool_calls[0]["function"]["name"], "read_file");
        assert_eq!(
            tool_calls[0]["function"]["arguments"],
            "{\"path\":\"a.txt\"}"
        );
        assert_eq!(merged["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn merges_anthropic_thinking_delta_into_reasoning_content() {
        let events = vec![
            serde_json::json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}),
            serde_json::json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "pondering..."}}),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({"type": "message_stop"}),
        ];
        let merged = merge_anthropic_stream(&events);
        assert_eq!(
            merged["choices"][0]["message"]["reasoning_content"],
            "pondering..."
        );
    }
}
