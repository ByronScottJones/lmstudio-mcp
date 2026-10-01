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

use crate::types::{with_timeout, ClientError, DEFAULT_TIMEOUT};
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
    pub async fn chat_completion(&self, body: Value) -> Result<Value, ClientError> {
        let url = format!("{}/chat/completions", self.openai_base_url);
        self.post_json(&url, &body, crate::types::INFERENCE_TIMEOUT)
            .await
    }

    /// POST a legacy text-completions request to the OpenAI-compatible endpoint.
    pub async fn text_completion(&self, body: Value) -> Result<Value, ClientError> {
        let url = format!("{}/completions", self.openai_base_url);
        self.post_json(&url, &body, crate::types::INFERENCE_TIMEOUT)
            .await
    }

    /// POST an embeddings request to the OpenAI-compatible endpoint.
    pub async fn embeddings(&self, body: Value) -> Result<Value, ClientError> {
        let url = format!("{}/embeddings", self.openai_base_url);
        self.post_json(&url, &body, crate::types::INFERENCE_TIMEOUT)
            .await
    }

    /// POST a stateful responses request to the OpenAI-compatible endpoint.
    pub async fn responses(&self, body: Value) -> Result<Value, ClientError> {
        let url = format!("{}/responses", self.openai_base_url);
        self.post_json(&url, &body, crate::types::INFERENCE_TIMEOUT)
            .await
    }
}
