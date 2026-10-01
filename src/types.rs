//! Shared result envelope and error types used by every tool.
//!
//! Every tool returns the same `ToolResult<T>` shape (mirroring the TypeScript
//! `lm-studio-mcp-server` project's convention) so MCP clients can handle
//! success and failure uniformly regardless of which tool was called.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Standard error codes for tool operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    ModelNotFound,
    ModelNotLoaded,
    ConnectionFailed,
    Unauthorized,
    InvalidInput,
    LoadFailed,
    UnloadFailed,
    Timeout,
    Unknown,
}

/// Standard error shape for tool failures.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolError {
    pub code: ErrorCode,
    pub message: String,
}

/// Standard result envelope for all tool operations. Every tool returns this
/// shape for consistency, whether the underlying call succeeded or failed.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolResult<T> {
    pub success: bool,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ToolError>,
}

impl<T> ToolResult<T> {
    pub fn ok(message: impl Into<String>, data: T) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: Some(data),
            error: None,
        }
    }

    /// Success with no payload (e.g. unload_model).
    pub fn ok_empty(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: None,
            error: None,
        }
    }

    pub fn err(message: impl Into<String>, code: ErrorCode, detail: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            success: false,
            message: message.clone(),
            data: None,
            error: Some(ToolError {
                code,
                message: detail.into(),
            }),
        }
    }
}

/// Errors that can occur while talking to the LM Studio HTTP API.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("request timed out after {0:?}")]
    Timeout(Duration),
    #[error("could not connect to LM Studio: {0}")]
    Connect(reqwest::Error),
    #[error("LM Studio returned HTTP {status}: {body}")]
    Status { status: u16, body: String },
    #[error("failed to parse LM Studio's response as JSON: {0}")]
    Decode(serde_json::Error),
    #[error("request error: {0}")]
    Request(reqwest::Error),
}

impl ClientError {
    /// Best-effort mapping of a client error to one of our standard error codes.
    pub fn code(&self) -> ErrorCode {
        match self {
            ClientError::Timeout(_) => ErrorCode::Timeout,
            ClientError::Connect(_) => ErrorCode::ConnectionFailed,
            ClientError::Status { status, .. } if *status == 401 || *status == 403 => {
                ErrorCode::Unauthorized
            }
            ClientError::Status { status, .. } if *status == 404 => ErrorCode::ModelNotFound,
            ClientError::Status { .. } => ErrorCode::Unknown,
            ClientError::Decode(_) => ErrorCode::Unknown,
            ClientError::Request(e) if e.is_connect() => ErrorCode::ConnectionFailed,
            ClientError::Request(e) if e.is_timeout() => ErrorCode::Timeout,
            ClientError::Request(_) => ErrorCode::Unknown,
        }
    }
}

/// Default timeout for ordinary SDK/API calls.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Extended timeout for model loading, which can take a while for large models.
pub const LOAD_MODEL_TIMEOUT: Duration = Duration::from_secs(300);

/// Wrap a future with a timeout, mapping elapsed time to [`ClientError::Timeout`].
pub async fn with_timeout<F, T>(fut: F, timeout: Duration) -> Result<T, ClientError>
where
    F: std::future::Future<Output = Result<T, ClientError>>,
{
    match tokio::time::timeout(timeout, fut).await {
        Ok(result) => result,
        Err(_) => Err(ClientError::Timeout(timeout)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_401_and_403_map_to_unauthorized() {
        let e = ClientError::Status {
            status: 401,
            body: String::new(),
        };
        assert_eq!(e.code(), ErrorCode::Unauthorized);
        let e = ClientError::Status {
            status: 403,
            body: String::new(),
        };
        assert_eq!(e.code(), ErrorCode::Unauthorized);
    }

    #[test]
    fn status_404_maps_to_model_not_found() {
        let e = ClientError::Status {
            status: 404,
            body: String::new(),
        };
        assert_eq!(e.code(), ErrorCode::ModelNotFound);
    }

    #[test]
    fn timeout_maps_to_timeout() {
        let e = ClientError::Timeout(Duration::from_secs(30));
        assert_eq!(e.code(), ErrorCode::Timeout);
    }

    #[test]
    fn tool_result_ok_serializes_without_error_field() {
        let r = ToolResult::ok("done", 42);
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["success"], true);
        assert_eq!(json["data"], 42);
        assert!(json.get("error").is_none());
    }

    #[test]
    fn tool_result_err_serializes_without_data_field() {
        let r: ToolResult<i32> = ToolResult::err("failed", ErrorCode::Unknown, "detail");
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["success"], false);
        assert!(json.get("data").is_none());
        assert_eq!(json["error"]["code"], "UNKNOWN");
    }
}
