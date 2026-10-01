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
    // Set when a `[DONE]` sentinel is seen (the chat/text-completions
    // convention). Callers that instead signal completion via `stop_when`
    // (the /v1/responses endpoint's typed terminal events) never reach the
    // natural-EOF path below at all on success — they return early. So
    // either way, reaching natural EOF with this still false means the
    // connection closed before the response actually finished.
    let mut saw_completion_signal = false;

    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(ClientError::Timeout(max_duration));
        }
        let per_chunk_wait = idle_timeout.min(remaining);

        let chunk = match tokio::time::timeout(per_chunk_wait, stream.next()).await {
            Ok(Some(Ok(bytes))) => bytes,
            Ok(Some(Err(e))) => return Err(ClientError::Request(e)),
            Ok(None) => {
                // Server closed the connection. Only treat that as a
                // complete response if we actually saw a completion
                // signal — otherwise this is a truncated stream (dropped
                // connection, proxy timeout, crash mid-response) that
                // would otherwise silently look like a short-but-valid
                // answer.
                if !saw_completion_signal {
                    return Err(ClientError::IncompleteStream {
                        events_seen: events.len(),
                    });
                }
                break;
            }
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
                        saw_completion_signal = true;
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

    /// Starts a one-shot raw HTTP server on an OS-assigned local port: it
    /// accepts a single connection, writes `body` verbatim after minimal
    /// headers, then closes the socket — closing *before* the headers claim
    /// (no `Content-Length`, so the client can only tell the body ended by
    /// the connection closing, exactly like a real chunked SSE stream).
    /// Returns the base URL to fetch from.
    async fn serve_once_then_close(body: &'static [u8]) -> String {
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            // Drain the request so the client's write isn't left hanging.
            let mut discard = [0u8; 1024];
            let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut discard).await;
            let headers =
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
            let _ = socket.write_all(headers).await;
            let _ = socket.write_all(body).await;
            let _ = socket.shutdown().await;
            // Socket drops here, closing the connection.
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn errors_on_premature_eof_without_done_sentinel() {
        // Two well-formed events, then the connection closes without a
        // "[DONE]" sentinel — simulating a dropped connection mid-stream.
        let url = serve_once_then_close(b"data: {\"a\":1}\n\ndata: {\"a\":2}\n\n").await;
        let resp = reqwest::get(&url).await.unwrap();

        let result = collect_events(resp, Duration::from_secs(5), Duration::from_secs(5), |_| {
            false
        })
        .await;

        match result {
            Err(ClientError::IncompleteStream { events_seen }) => assert_eq!(events_seen, 2),
            other => panic!("expected IncompleteStream, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn succeeds_when_done_sentinel_is_present() {
        let url = serve_once_then_close(b"data: {\"a\":1}\n\ndata: [DONE]\n\n").await;
        let resp = reqwest::get(&url).await.unwrap();

        let events = collect_events(resp, Duration::from_secs(5), Duration::from_secs(5), |_| {
            false
        })
        .await
        .expect("a stream ending with [DONE] should succeed");
        assert_eq!(events, vec![serde_json::json!({"a": 1})]);
    }

    #[tokio::test]
    async fn succeeds_when_stop_when_fires_before_connection_closes() {
        // Simulates the /v1/responses style: no "[DONE]", completion is
        // signaled by a typed terminal event that `stop_when` recognizes.
        // The connection closing afterward (even abruptly) shouldn't matter
        // since collect_events already returned.
        let url = serve_once_then_close(b"data: {\"type\":\"response.completed\"}\n\n").await;
        let resp = reqwest::get(&url).await.unwrap();

        let events = collect_events(resp, Duration::from_secs(5), Duration::from_secs(5), |v| {
            v.get("type").and_then(|t| t.as_str()) == Some("response.completed")
        })
        .await
        .expect("stop_when firing should be treated as completion");
        assert_eq!(events.len(), 1);
    }
}
