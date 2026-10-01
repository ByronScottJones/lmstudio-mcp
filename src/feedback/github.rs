//! Duplicate-issue checking against GitHub's public REST API, and token
//! resolution. Mirrors `uictl-mac-mcp`'s `GitHubIssues.swift` /
//! `GitHubToken.swift`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GitHubIssueSummary {
    pub number: u64,
    pub title: String,
    pub url: String,
    pub state: String,
}

#[derive(Debug, thiserror::Error)]
pub enum GitHubIssuesError {
    #[error("{0}")]
    Unavailable(String),
}

/// Resolve a GitHub token for the (optional) duplicate-issue check.
/// Precedence: an explicit param, then `GITHUB_TOKEN`, then whatever the
/// `gh` CLI (if installed and already authenticated) has stored. Returns
/// `None` if none of those pan out, so callers can treat "couldn't check"
/// as "skip the check" rather than failing outright.
pub async fn resolve_token(explicit: Option<String>) -> Option<String> {
    if let Some(t) = explicit {
        if !t.is_empty() {
            return Some(t);
        }
    }
    if let Ok(t) = std::env::var("GITHUB_TOKEN") {
        if !t.is_empty() {
            return Some(t);
        }
    }
    from_gh_cli().await
}

async fn from_gh_cli() -> Option<String> {
    let output = tokio::process::Command::new("gh")
        .args(["auth", "token"])
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let token = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if token.is_empty() {
        None
    } else {
        Some(token)
    }
}

/// Fetch a repo's issues from the public GitHub REST API, to check a draft
/// feedback entry against before submitting it. Unauthenticated requests
/// work fine against public repos (just a much lower rate limit); a
/// private repo needs `token`. Every failure mode (missing token, rate
/// limit, network error) surfaces as `GitHubIssuesError::Unavailable` so
/// callers can treat "couldn't check" as "skip the check" rather than
/// blocking submission on it.
pub async fn fetch_all_issues(
    client: &reqwest::Client,
    repo: &str,
    token: Option<&str>,
) -> Result<Vec<GitHubIssueSummary>, GitHubIssuesError> {
    let mut all_issues = Vec::new();
    // A generous but finite cap — this is a duplicate-title scan, not a
    // full mirror; 1000 issues is far more than that needs.
    for page in 1..=10u32 {
        let url = format!("https://api.github.com/repos/{repo}/issues");
        let mut req = client
            .get(&url)
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "lmstudio-rs-mcp")
            .query(&[
                ("state", "all"),
                ("per_page", "100"),
                ("page", &page.to_string()),
            ]);
        if let Some(token) = token {
            req = req.bearer_auth(token);
        }

        let resp = req.send().await.map_err(|e| {
            GitHubIssuesError::Unavailable(format!("network error contacting GitHub: {e}"))
        })?;
        let status = resp.status();
        if !status.is_success() {
            let hint = if token.is_none() {
                "this repo may be private — set GITHUB_TOKEN, authenticate `gh`, or pass a token"
            } else {
                "check that the token has access to this repo"
            };
            return Err(GitHubIssuesError::Unavailable(format!(
                "GitHub returned HTTP {status} for {repo} ({hint})"
            )));
        }

        let raw: Vec<serde_json::Value> = resp.json().await.map_err(|e| {
            GitHubIssuesError::Unavailable(format!("failed to parse GitHub's response: {e}"))
        })?;
        if raw.is_empty() {
            break;
        }

        let page_len = raw.len();
        for item in raw {
            // The issues endpoint also returns pull requests.
            if item.get("pull_request").is_some() {
                continue;
            }
            let (Some(number), Some(title), Some(html_url), Some(state)) = (
                item.get("number").and_then(|v| v.as_u64()),
                item.get("title").and_then(|v| v.as_str()),
                item.get("html_url").and_then(|v| v.as_str()),
                item.get("state").and_then(|v| v.as_str()),
            ) else {
                continue;
            };
            all_issues.push(GitHubIssueSummary {
                number,
                title: title.to_string(),
                url: html_url.to_string(),
                state: state.to_string(),
            });
        }
        if page_len < 100 {
            break;
        }
    }
    Ok(all_issues)
}

/// A deliberately simple, transparent heuristic — case-insensitive equality
/// or substring containment either direction — meant to catch obvious
/// re-reports, not near-miss wording. Not fuzzy matching.
pub fn find_duplicates(title: &str, issues: &[GitHubIssueSummary]) -> Vec<GitHubIssueSummary> {
    let normalized = title.trim().to_lowercase();
    if normalized.is_empty() {
        return Vec::new();
    }
    issues
        .iter()
        .filter(|issue| {
            let other = issue.title.trim().to_lowercase();
            other == normalized || other.contains(&normalized) || normalized.contains(&other)
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn issue(number: u64, title: &str) -> GitHubIssueSummary {
        GitHubIssueSummary {
            number,
            title: title.to_string(),
            url: format!("https://github.com/o/r/issues/{number}"),
            state: "open".to_string(),
        }
    }

    #[test]
    fn finds_exact_case_insensitive_match() {
        let issues = vec![issue(1, "Server crashes on startup")];
        let dups = find_duplicates("server crashes on startup", &issues);
        assert_eq!(dups.len(), 1);
    }

    #[test]
    fn finds_substring_match_either_direction() {
        let issues = vec![issue(
            1,
            "Server crashes on startup when no model is loaded",
        )];
        let dups = find_duplicates("Server crashes on startup", &issues);
        assert_eq!(dups.len(), 1);

        let issues2 = vec![issue(2, "crash")];
        let dups2 = find_duplicates("Server crash on startup", &issues2);
        assert_eq!(dups2.len(), 1);
    }

    #[test]
    fn no_match_for_unrelated_titles() {
        let issues = vec![issue(1, "Add dark mode")];
        let dups = find_duplicates("Server crashes on startup", &issues);
        assert!(dups.is_empty());
    }

    #[test]
    fn empty_title_matches_nothing() {
        let issues = vec![issue(1, "Add dark mode")];
        assert!(find_duplicates("", &issues).is_empty());
        assert!(find_duplicates("   ", &issues).is_empty());
    }

    #[tokio::test]
    async fn resolve_token_prefers_explicit_over_env() {
        // SAFETY: test-only env mutation; this test crate runs single-threaded
        // per `cargo test`'s default per-binary execution unless overridden,
        // and this var is unique to this test's assertions.
        unsafe { std::env::set_var("GITHUB_TOKEN", "env-token") };
        let resolved = resolve_token(Some("explicit-token".to_string())).await;
        assert_eq!(resolved.as_deref(), Some("explicit-token"));
        unsafe { std::env::remove_var("GITHUB_TOKEN") };
    }
}
