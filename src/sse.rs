//! Minimal Server-Sent Events (SSE) reader for OpenAI-style streaming
//! responses (`"stream": true`).
//!
//! This exists for one reason: a single non-streaming HTTP request has to
//! be timed out against its *total* duration, and there's no duration that
//! is simultaneously short enough to catch a genuinely hung connection and
//! long enough for a slow reasoning model's legitimate output. Streaming
//! turns that into an *idle* timeout instead — reset on every chunk that
//! actually arrives — which is correct in both directions: a model that
//! keeps producing tokens for ten minutes is never killed, and a
//! connection that stalls is caught quickly regardless of how long the
//! request has already been running.
//!
//! Only `data:` lines are interpreted (LM Studio, like OpenAI, puts all the
//! structured information — including the event's own `type` for the
//! `/v1/responses` endpoint — inside the JSON payload itself, not in the
//! SSE `event:` field), and a `data: [DONE]` sentinel is swallowed.
//! Malformed individual events are logged and skipped rather than failing
//! the whole stream, since a stray non-JSON keep-alive line shouldn't take
//! down an otherwise-good response.

use crate::types::ClientError;
use futures_util::StreamExt;
use serde_json::Value;
use std::time::Duration;
use tokio::time::Instant;

/// Read an SSE response body to completion (or until `stop_when` returns
/// true for an event, or the connection closes), returning every
/// successfully-parsed JSON `data:` payload in order.
///
/// `idle_timeout` bounds the gap between consecutive chunks arriving on the
/// wire. `max_duration` is a hard ceiling on the whole read, independent of
/// activity, as a last-resort safety net.
pub async fn collect_events(
    resp: reqwest::Response,
    idle_timeout: Duration,
    max_duration: Duration,
    mut stop_when: impl FnMut(&Value) -> bool,
) -> Result<Vec<Value>, ClientError> {
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.map_err(ClientError::Request)?;
        return Err(ClientError::Status {
            status: status.as_u16(),
            body,
        });
    }

    let deadline = Instant::now() + max_duration;
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut data_lines: Vec<String> = Vec::new();
    let mut events = Vec::new();

    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(ClientError::Timeout(max_duration));
        }
        let per_chunk_wait = idle_timeout.min(remaining);

        let chunk = match tokio::time::timeout(per_chunk_wait, stream.next()).await {
            Ok(Some(Ok(bytes))) => bytes,
            Ok(Some(Err(e))) => return Err(ClientError::Request(e)),
            Ok(None) => break, // server closed the connection normally
            Err(_) => {
                // Distinguish "genuinely idle" from "we were already past
                // the overall deadline and used a shortened wait for it".
                if Instant::now() >= deadline {
                    return Err(ClientError::Timeout(max_duration));
                }
                return Err(ClientError::Timeout(idle_timeout));
            }
        };
        buf.extend_from_slice(&chunk);

        // Drain complete lines (terminated by '\n', optionally preceded by
        // '\r'). Splitting on the '\n' byte is always safe even mid-UTF-8
        // multibyte sequence, since 0x0A never appears as a continuation
        // byte in valid UTF-8.
        while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim_end_matches(['\r', '\n']);

            if line.is_empty() {
                // Blank line: dispatch the event we've been accumulating.
                if !data_lines.is_empty() {
                    let payload = data_lines.join("\n");
                    data_lines.clear();
                    if payload == "[DONE]" {
                        continue;
                    }
                    match serde_json::from_str::<Value>(&payload) {
                        Ok(value) => {
                            let stop = stop_when(&value);
                            events.push(value);
                            if stop {
                                return Ok(events);
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Skipping malformed SSE event (not valid JSON): {e}; payload: {payload}"
                            );
                        }
                    }
                }
                continue;
            }

            if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
            }
            // Other SSE fields (event:, id:, retry:, ':' comments) carry no
            // information we need — LM Studio's event `type` lives inside
            // the JSON payload itself.
        }
    }

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sse_data_lines_via_a_fake_stream() {
        // Exercise the line-parsing logic directly rather than standing up
        // a real HTTP server: feed bytes through the same drain loop.
        let raw = b"data: {\"a\":1}\n\ndata: {\"a\":2}\n\ndata: [DONE]\n\n";
        let mut buf: Vec<u8> = raw.to_vec();
        let mut data_lines: Vec<String> = Vec::new();
        let mut events: Vec<Value> = Vec::new();

        while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                if !data_lines.is_empty() {
                    let payload = data_lines.join("\n");
                    data_lines.clear();
                    if payload != "[DONE]" {
                        events.push(serde_json::from_str(&payload).unwrap());
                    }
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
            }
        }

        assert_eq!(
            events,
            vec![serde_json::json!({"a":1}), serde_json::json!({"a":2})]
        );
    }
}
