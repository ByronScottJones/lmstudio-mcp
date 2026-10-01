//! Runtime configuration, read from environment variables (set lazily, at
//! connect time, not at process startup, so a client that sets env vars via
//! its MCP server launch config always wins).

use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    /// Base URL for LM Studio's native REST API, e.g. `http://127.0.0.1:1234/api/v1`.
    pub native_base_url: String,
    /// Base URL for LM Studio's OpenAI-compatible REST API, e.g. `http://127.0.0.1:1234/v1`.
    pub openai_base_url: String,
    /// Optional bearer token, if the LM Studio server has API token auth enabled.
    pub api_token: Option<String>,
}

impl Config {
    /// Build configuration from the environment.
    ///
    /// - `LMSTUDIO_BASE_URL`: full `scheme://host:port` override (e.g.
    ///   `http://192.168.1.100:5678`). Takes precedence over host/port.
    /// - `LMSTUDIO_HOST` (default `127.0.0.1`), `LMSTUDIO_PORT` (default `1234`).
    /// - `LMSTUDIO_API_TOKEN`: optional bearer token for `Authorization` header.
    pub fn from_env() -> Self {
        let base = if let Ok(base) = env::var("LMSTUDIO_BASE_URL") {
            base.trim_end_matches('/').to_string()
        } else {
            let host = env::var("LMSTUDIO_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
            let port = env::var("LMSTUDIO_PORT").unwrap_or_else(|_| "1234".to_string());
            let port: u16 = port.parse().unwrap_or_else(|_| {
                tracing::warn!(
                    "Invalid LMSTUDIO_PORT '{port}', falling back to 1234",
                    port = port
                );
                1234
            });
            format!("http://{host}:{port}")
        };

        Self {
            native_base_url: format!("{base}/api/v1"),
            openai_base_url: format!("{base}/v1"),
            api_token: env::var("LMSTUDIO_API_TOKEN").ok(),
        }
    }
}
