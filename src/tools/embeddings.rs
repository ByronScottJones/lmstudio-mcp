//! generate_embeddings tool, over the OpenAI-compatible `/v1/embeddings` endpoint.

use crate::client::ApiClient;
use crate::types::{ErrorCode, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Either a single string or a batch of strings to embed.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(untagged)]
pub enum EmbeddingText {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct GenerateEmbeddingsInput {
    /// Single text string or list of texts to embed.
    pub text: EmbeddingText,
    /// Embedding model to use. Omit to use the currently loaded model, but an
    /// embedding-specific model (e.g. `text-embedding-nomic-embed-text-v1.5`)
    /// is recommended — a regular chat model will not work with this endpoint.
    #[serde(default)]
    pub model: Option<String>,
}

/// The raw embeddings response from LM Studio, passed through as-is: vectors
/// and usage accounting are large/opaque and not worth re-typing by hand.
#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct EmbeddingsData {
    pub raw: Value,
}

pub async fn generate_embeddings(
    client: &ApiClient,
    input: GenerateEmbeddingsInput,
) -> ToolResult<EmbeddingsData> {
    if !client.provider().supports_embeddings() {
        return ToolResult::err(
            format!("{} has no embeddings endpoint", client.provider()),
            ErrorCode::InvalidInput,
            "this provider does not support generate_embeddings",
        );
    }
    let text_value = match input.text {
        EmbeddingText::Single(s) => Value::String(s),
        EmbeddingText::Batch(v) => Value::from(v),
    };
    let mut body = serde_json::json!({ "input": text_value });
    if let Some(model) = input.model {
        body["model"] = Value::String(model);
    }

    match client.embeddings(body).await {
        Ok(raw) => ToolResult::ok("Generated embeddings", EmbeddingsData { raw }),
        Err(e) => ToolResult::err(
            format!("Failed to generate embeddings: {e}"),
            e.code(),
            e.to_string(),
        ),
    }
}
