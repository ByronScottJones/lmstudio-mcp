//! health_check tool.

use crate::client::LmStudioClient;
use crate::types::ToolResult;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct HealthCheckData {
    pub connected: bool,
    pub native_base_url: String,
    pub openai_base_url: String,
    pub models_in_library: usize,
}

pub async fn health_check(client: &LmStudioClient) -> ToolResult<HealthCheckData> {
    match client.health_check().await {
        Ok(resp) => ToolResult::ok(
            format!("Connected to LM Studio at {}", client.native_base_url),
            HealthCheckData {
                connected: true,
                native_base_url: client.native_base_url.clone(),
                openai_base_url: client.openai_base_url.clone(),
                models_in_library: resp.models.len(),
            },
        ),
        Err(e) => ToolResult::err(
            format!(
                "Failed to connect to LM Studio at {}: {e}",
                client.native_base_url
            ),
            e.code(),
            e.to_string(),
        ),
    }
}
