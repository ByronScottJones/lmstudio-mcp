//! Model library and model-management tools: list_models, list_loaded_models,
//! get_current_model, get_model_info, load_model, unload_model.

use crate::client::{model::ModelEntry, ApiClient};
use crate::types::{ClientError, ErrorCode, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// ---------------------------------------------------------------------------
// Shared view types
// ---------------------------------------------------------------------------

/// A downloaded model in the library, as returned by `list_models`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ModelSummary {
    /// Key used to reference this model in `load_model` and elsewhere.
    pub key: String,
    pub display_name: Option<String>,
    pub publisher: Option<String>,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    pub size_bytes: Option<u64>,
    pub params_string: Option<String>,
    pub max_context_length: Option<u64>,
    pub format: Option<String>,
    pub vision: bool,
    pub trained_for_tool_use: bool,
    /// Instance identifiers of this model that are currently loaded in memory (empty if none).
    pub loaded_instance_ids: Vec<String>,
}

impl From<&ModelEntry> for ModelSummary {
    fn from(m: &ModelEntry) -> Self {
        Self {
            key: m.key.clone(),
            display_name: m.display_name.clone(),
            publisher: m.publisher.clone(),
            architecture: m.architecture.clone(),
            quantization: m.quantization.as_ref().and_then(|q| q.name.clone()),
            size_bytes: m.size_bytes,
            params_string: m.params_string.clone(),
            max_context_length: m.max_context_length,
            format: m.format.clone(),
            vision: m.capabilities.as_ref().map(|c| c.vision).unwrap_or(false),
            trained_for_tool_use: m
                .capabilities
                .as_ref()
                .map(|c| c.trained_for_tool_use)
                .unwrap_or(false),
            loaded_instance_ids: m.loaded_instances.iter().map(|i| i.id.clone()).collect(),
        }
    }
}

/// A specific loaded model instance, as returned by `list_loaded_models` and `get_model_info`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct LoadedModelSummary {
    /// Instance identifier — pass this to `unload_model` / `get_model_info`.
    pub identifier: String,
    /// The model key this instance was loaded from.
    pub model_key: String,
    pub display_name: Option<String>,
    pub size_bytes: Option<u64>,
    pub context_length: Option<u64>,
    pub eval_batch_size: Option<u64>,
    pub flash_attention: Option<bool>,
    pub vision: bool,
    pub trained_for_tool_use: bool,
}

fn loaded_instances(models: &[ModelEntry]) -> Vec<LoadedModelSummary> {
    models
        .iter()
        .flat_map(|m| {
            m.loaded_instances
                .iter()
                .map(move |inst| LoadedModelSummary {
                    identifier: inst.id.clone(),
                    model_key: m.key.clone(),
                    display_name: m.display_name.clone(),
                    size_bytes: m.size_bytes,
                    context_length: inst.config.as_ref().and_then(|c| c.context_length),
                    eval_batch_size: inst.config.as_ref().and_then(|c| c.eval_batch_size),
                    flash_attention: inst.config.as_ref().and_then(|c| c.flash_attention),
                    vision: m.capabilities.as_ref().map(|c| c.vision).unwrap_or(false),
                    trained_for_tool_use: m
                        .capabilities
                        .as_ref()
                        .map(|c| c.trained_for_tool_use)
                        .unwrap_or(false),
                })
        })
        .collect()
}

/// Fetch the library and return only the currently-loaded instances.
/// Shared by `list_loaded_models`, `get_current_model`, and model
/// auto-detection in the `/v1/responses` tools.
pub(crate) async fn fetch_loaded(
    client: &ApiClient,
) -> Result<Vec<LoadedModelSummary>, ClientError> {
    let resp = client.list_models().await?;
    Ok(loaded_instances(&resp.models))
}

/// Why [`auto_detect_model`] couldn't pick a model, distinguishing "LM
/// Studio is reachable but there's nothing/too much loaded" (always
/// `ModelNotLoaded`) from a genuine client-level failure (connection
/// refused, unauthorized, timeout, ...), which keeps its own real
/// [`ErrorCode`] instead of being flattened into `ModelNotLoaded` too.
#[derive(Debug)]
pub(crate) enum AutoDetectError {
    NoneLoaded,
    Ambiguous(usize),
    Client(ClientError),
}

impl AutoDetectError {
    pub(crate) fn code(&self) -> ErrorCode {
        match self {
            AutoDetectError::NoneLoaded | AutoDetectError::Ambiguous(_) => {
                ErrorCode::ModelNotLoaded
            }
            AutoDetectError::Client(e) => e.code(),
        }
    }
}

impl std::fmt::Display for AutoDetectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AutoDetectError::NoneLoaded => write!(
                f,
                "No model is currently loaded (or, for a provider without model loading, none is marked as such). Pass `model` explicitly."
            ),
            AutoDetectError::Ambiguous(n) => write!(
                f,
                "{n} models are loaded; pass `model` explicitly to pick one."
            ),
            AutoDetectError::Client(e) => write!(f, "Could not detect the currently loaded model: {e}"),
        }
    }
}

/// Pick the model to use when the caller didn't name one explicitly: the
/// single loaded instance's *identifier* (not its library key — these
/// differ when the model was loaded with a custom `identifier` via
/// `load_model`, and the OpenAI-compatible endpoints route by identifier).
/// Shared by the responses and subagent tools, which both need this.
pub(crate) async fn auto_detect_model(client: &ApiClient) -> Result<String, AutoDetectError> {
    let models = fetch_loaded(client)
        .await
        .map_err(AutoDetectError::Client)?;
    pick_auto_detected_model(models)
}

/// The pure selection logic behind [`auto_detect_model`], split out so it's
/// testable without a live LM Studio connection.
fn pick_auto_detected_model(models: Vec<LoadedModelSummary>) -> Result<String, AutoDetectError> {
    match models.len() {
        1 => Ok(models.into_iter().next().unwrap().identifier),
        0 => Err(AutoDetectError::NoneLoaded),
        n => Err(AutoDetectError::Ambiguous(n)),
    }
}

fn tool_err<T>(prefix: &str, e: ClientError) -> ToolResult<T> {
    let code = e.code();
    let detail = e.to_string();
    ToolResult::err(format!("{prefix}: {e}"), code, detail)
}

// ---------------------------------------------------------------------------
// list_models
// ---------------------------------------------------------------------------

pub async fn list_models(client: &ApiClient) -> ToolResult<Vec<ModelSummary>> {
    match client.list_models().await {
        Ok(resp) => {
            let models: Vec<ModelSummary> = resp.models.iter().map(ModelSummary::from).collect();
            ToolResult::ok(
                format!("Found {} model(s) in the library", models.len()),
                models,
            )
        }
        Err(e) => tool_err("Failed to list models", e),
    }
}

// ---------------------------------------------------------------------------
// list_loaded_models
// ---------------------------------------------------------------------------

pub async fn list_loaded_models(client: &ApiClient) -> ToolResult<Vec<LoadedModelSummary>> {
    if !client.provider().supports_model_management() {
        return ToolResult::ok(
            format!(
                "{} has no model loading concept — every model from list_models is directly usable without loading",
                client.provider()
            ),
            Vec::new(),
        );
    }
    match fetch_loaded(client).await {
        Ok(models) => ToolResult::ok(format!("Found {} loaded model(s)", models.len()), models),
        Err(e) => tool_err("Failed to list loaded models", e),
    }
}

// ---------------------------------------------------------------------------
// get_current_model
// ---------------------------------------------------------------------------

pub async fn get_current_model(client: &ApiClient) -> ToolResult<Vec<LoadedModelSummary>> {
    if !client.provider().supports_model_management() {
        return ToolResult::ok(
            format!(
                "{} has no model loading concept — pass `model` explicitly to whichever tool needs one",
                client.provider()
            ),
            Vec::new(),
        );
    }
    match fetch_loaded(client).await {
        Ok(models) if models.is_empty() => {
            ToolResult::ok(format!("No model is currently loaded in {}", client.provider()), models)
        }
        Ok(models) if models.len() == 1 => {
            let name = models[0].display_name.clone().unwrap_or_else(|| models[0].model_key.clone());
            ToolResult::ok(format!("Currently loaded model: {name}"), models)
        }
        Ok(models) => ToolResult::ok(
            format!(
                "{} models are currently loaded; pass an explicit identifier to tools that need one",
                models.len()
            ),
            models,
        ),
        Err(e) => tool_err("Failed to identify the current model", e)
    }
}

// ---------------------------------------------------------------------------
// get_model_info
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct GetModelInfoInput {
    /// The loaded model instance identifier to get information about (see `list_loaded_models`).
    pub identifier: String,
}

pub async fn get_model_info(
    client: &ApiClient,
    input: GetModelInfoInput,
) -> ToolResult<LoadedModelSummary> {
    if !client.provider().supports_model_management() {
        return ToolResult::err(
            format!("{} has no model loading concept", client.provider()),
            ErrorCode::InvalidInput,
            "this provider does not support get_model_info",
        );
    }
    match fetch_loaded(client).await {
        Ok(models) => match models
            .into_iter()
            .find(|m| m.identifier == input.identifier)
        {
            Some(m) => ToolResult::ok(
                format!("Retrieved information for '{}'", input.identifier),
                m,
            ),
            None => ToolResult::err(
                format!("Model '{}' not found or not loaded", input.identifier),
                ErrorCode::ModelNotLoaded,
                "no loaded instance with that identifier",
            ),
        },
        Err(e) => tool_err("Failed to get model info", e),
    }
}

// ---------------------------------------------------------------------------
// load_model
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct LoadModelInput {
    /// The model key to load (e.g. `"llama-3.2-3b-instruct"`, see `list_models`).
    pub model: String,
    /// Context window size in tokens.
    pub context_length: Option<u32>,
    /// Enable flash attention, if the model/backend supports it.
    pub flash_attention: Option<bool>,
    /// Number of tokens evaluated per batch.
    pub eval_batch_size: Option<u32>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct LoadedModelData {
    pub instance_id: Option<String>,
    pub status: Option<String>,
    pub load_time_seconds: Option<f64>,
    pub context_length: Option<u64>,
}

pub async fn load_model(client: &ApiClient, input: LoadModelInput) -> ToolResult<LoadedModelData> {
    if !client.provider().supports_model_management() {
        return ToolResult::err(
            format!(
                "{} has no model loading concept — every model is directly usable without loading",
                client.provider()
            ),
            ErrorCode::InvalidInput,
            "this provider does not support load_model",
        );
    }
    let mut body = Map::new();
    body.insert("model".into(), Value::String(input.model.clone()));
    if let Some(v) = input.context_length {
        body.insert("context_length".into(), Value::from(v));
    }
    if let Some(v) = input.flash_attention {
        body.insert("flash_attention".into(), Value::from(v));
    }
    if let Some(v) = input.eval_batch_size {
        body.insert("eval_batch_size".into(), Value::from(v));
    }

    match client.load_model(Value::Object(body)).await {
        Ok(resp) => {
            let identifier = resp
                .instance_id
                .clone()
                .unwrap_or_else(|| input.model.clone());
            ToolResult::ok(
                format!(
                    "Model '{}' loaded successfully as '{identifier}'",
                    input.model
                ),
                LoadedModelData {
                    instance_id: resp.instance_id,
                    status: resp.status,
                    load_time_seconds: resp.load_time_seconds,
                    context_length: resp.load_config.and_then(|c| c.context_length),
                },
            )
        }
        Err(e) => ToolResult::err(
            format!("Failed to load model '{}': {e}", input.model),
            ErrorCode::LoadFailed,
            e.to_string(),
        ),
    }
}

// ---------------------------------------------------------------------------
// unload_model
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct UnloadModelInput {
    /// The loaded model instance identifier to unload (see `list_loaded_models`).
    pub identifier: String,
}

pub async fn unload_model(client: &ApiClient, input: UnloadModelInput) -> ToolResult<()> {
    if !client.provider().supports_model_management() {
        return ToolResult::err(
            format!("{} has no model loading concept", client.provider()),
            ErrorCode::InvalidInput,
            "this provider does not support unload_model",
        );
    }
    match client.unload_model(&input.identifier).await {
        Ok(_) => ToolResult::ok_empty(format!(
            "Model '{}' unloaded successfully",
            input.identifier
        )),
        Err(e) => {
            let code = match &e {
                ClientError::Status { status, .. } if *status == 404 => ErrorCode::ModelNotLoaded,
                other => other.code(),
            };
            let message = if code == ErrorCode::ModelNotLoaded {
                format!("Model '{}' is not currently loaded", input.identifier)
            } else {
                format!("Failed to unload model '{}': {e}", input.identifier)
            };
            ToolResult::err(message, code, e.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::model::{Capabilities, LoadedInstance, LoadedInstanceConfig};

    fn sample_entry(key: &str, instances: Vec<LoadedInstance>) -> ModelEntry {
        ModelEntry {
            key: key.to_string(),
            display_name: Some(format!("{key} display")),
            loaded_instances: instances,
            capabilities: Some(Capabilities {
                vision: true,
                trained_for_tool_use: false,
                reasoning: None,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn flattens_loaded_instances_across_multiple_models() {
        let models = vec![
            sample_entry(
                "model-a",
                vec![LoadedInstance {
                    id: "model-a".into(),
                    config: Some(LoadedInstanceConfig {
                        context_length: Some(4096),
                        ..Default::default()
                    }),
                }],
            ),
            sample_entry("model-b", vec![]), // downloaded but not loaded
        ];

        let loaded = loaded_instances(&models);

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].identifier, "model-a");
        assert_eq!(loaded[0].context_length, Some(4096));
        assert!(loaded[0].vision);
    }

    #[test]
    fn model_summary_reports_loaded_instance_ids() {
        let entry = sample_entry(
            "model-c",
            vec![LoadedInstance {
                id: "instance-1".into(),
                config: None,
            }],
        );

        let summary = ModelSummary::from(&entry);
        assert_eq!(summary.loaded_instance_ids, vec!["instance-1".to_string()]);
    }

    #[test]
    fn deserializes_confirmed_lm_studio_models_response_shape() {
        // Shape taken from LM Studio's published REST API reference for
        // GET /api/v1/models.
        let json = serde_json::json!({
            "models": [{
                "type": "llm",
                "publisher": "google",
                "key": "google/gemma-4-26b-a4b",
                "display_name": "Gemma 4 26B A4B",
                "architecture": "gemma4",
                "quantization": { "name": "Q4_K_M", "bits_per_weight": 4 },
                "size_bytes": 17990911801u64,
                "params_string": "26B-A4B",
                "loaded_instances": [{
                    "id": "google/gemma-4-26b-a4b",
                    "config": {
                        "context_length": 4096,
                        "eval_batch_size": 512,
                        "parallel": 4,
                        "flash_attention": true,
                        "num_experts": 8,
                        "offload_kv_cache_to_gpu": true
                    }
                }],
                "max_context_length": 262144,
                "format": "gguf",
                "capabilities": {
                    "vision": true,
                    "trained_for_tool_use": true,
                    "reasoning": { "allowed_options": ["off", "on"], "default": "on" }
                }
            }]
        });

        let parsed: crate::client::model::ModelsListResponse =
            serde_json::from_value(json).unwrap();
        assert_eq!(parsed.models.len(), 1);
        let m = &parsed.models[0];
        assert_eq!(m.key, "google/gemma-4-26b-a4b");
        assert_eq!(m.loaded_instances[0].id, "google/gemma-4-26b-a4b");
        assert_eq!(
            m.loaded_instances[0]
                .config
                .as_ref()
                .unwrap()
                .context_length,
            Some(4096)
        );
    }

    #[test]
    fn tolerates_unknown_or_missing_fields() {
        // A minimal / future-shaped response should still parse: `key` is
        // the only required field, and unrecognized fields are captured by
        // the `extra` catch-all instead of causing a hard failure.
        let json = serde_json::json!({
            "models": [{ "key": "some/model", "a_future_field": 123 }]
        });

        let parsed: crate::client::model::ModelsListResponse =
            serde_json::from_value(json).unwrap();
        assert_eq!(parsed.models[0].key, "some/model");
        assert_eq!(
            parsed.models[0].extra.get("a_future_field"),
            Some(&serde_json::json!(123))
        );
    }

    fn loaded(identifier: &str, model_key: &str) -> LoadedModelSummary {
        LoadedModelSummary {
            identifier: identifier.to_string(),
            model_key: model_key.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn auto_detect_picks_the_instance_identifier_not_the_model_key() {
        // Regression: these differ when the model was loaded with a custom
        // `identifier` (see `load_model`) — the OpenAI-compatible endpoints
        // route by identifier, so sending the library key would target the
        // wrong (or a nonexistent) instance.
        let models = vec![loaded("my-custom-name", "qwen/qwen3.6-35b-a3b")];
        let picked = pick_auto_detected_model(models).unwrap();
        assert_eq!(picked, "my-custom-name");
    }

    #[test]
    fn auto_detect_errors_when_nothing_is_loaded() {
        assert!(matches!(
            pick_auto_detected_model(vec![]),
            Err(AutoDetectError::NoneLoaded)
        ));
    }

    #[test]
    fn auto_detect_errors_when_multiple_models_are_loaded() {
        let models = vec![loaded("a", "a"), loaded("b", "b")];
        assert!(matches!(
            pick_auto_detected_model(models),
            Err(AutoDetectError::Ambiguous(2))
        ));
    }

    #[test]
    fn auto_detect_error_codes_distinguish_client_failures_from_loaded_state() {
        assert_eq!(
            AutoDetectError::NoneLoaded.code(),
            ErrorCode::ModelNotLoaded
        );
        assert_eq!(
            AutoDetectError::Ambiguous(3).code(),
            ErrorCode::ModelNotLoaded
        );
        // A connection/auth failure must NOT be flattened into
        // ModelNotLoaded — that was the bug: every auto-detection failure
        // (including "LM Studio is unreachable") used to be reported as
        // "no model loaded", which is actively misleading to debug.
        let client_err = ClientError::Status {
            status: 401,
            body: String::new(),
        };
        assert_eq!(
            AutoDetectError::Client(client_err).code(),
            ErrorCode::Unauthorized
        );
    }
}
