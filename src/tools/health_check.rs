//! health_check tool.

use crate::client::ApiClient;
use crate::types::ToolResult;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct HealthCheckData {
    pub connected: bool,
    pub provider: String,
    pub base_url: String,
    pub models_available: usize,
}

pub async fn health_check(client: &ApiClient) -> ToolResult<HealthCheckData> {
    match client.health_check().await {
        Ok(resp) => ToolResult::ok(
            format!(
                "Connected to {} at {}",
                client.provider(),
                client.base_url()
            ),
            HealthCheckData {
                connected: true,
                provider: client.provider().to_string(),
                base_url: client.base_url().to_string(),
                models_available: resp.models.len(),
            },
        ),
        Err(e) => ToolResult::err(
            format!(
                "Failed to connect to {} at {}: {e}",
                client.provider(),
                client.base_url()
            ),
            e.code(),
            e.to_string(),
        ),
    }
}
