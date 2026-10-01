//! MCP-facing feedback tools: create/list/get/update/delete a local draft,
//! check it against GitHub's existing issues, and submit it. Mirrors
//! `uictl-mac-mcp`'s `feedback` verb and its MCP elicitation flow — see
//! `src/feedback/` for the underlying store and GitHub logic.
//!
//! `feedback_submit` never files a GitHub issue through the API. It always
//! ends at a pre-filled "new issue" page a human still has to review and
//! click "Create" on — by MCP elicitation when the client supports it
//! (reviewing, and optionally editing, the title/body first; then a
//! URL-mode elicitation instead of silently opening a browser tab), or by
//! opening that page directly on this machine as a fallback when it
//! doesn't.

use crate::feedback::github;
use crate::feedback::store::{FeedbackCategory, FeedbackEntry, FeedbackStore};
use crate::types::{ErrorCode, ToolResult};
use rmcp::model::{ElicitRequestParams, ElicitationAction, ElicitationSchema};
use rmcp::service::ElicitationMode;
use rmcp::{Peer, RoleServer};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const ELICITATION_TIMEOUT: Duration = Duration::from_secs(120);

fn store_err<T>(prefix: &str, e: crate::feedback::store::FeedbackStoreError) -> ToolResult<T> {
    let code = match &e {
        crate::feedback::store::FeedbackStoreError::NotFound(_) => ErrorCode::ModelNotFound,
        _ => ErrorCode::Unknown,
    };
    ToolResult::err(format!("{prefix}: {e}"), code, e.to_string())
}

// ---------------------------------------------------------------------------
// feedback_create
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct FeedbackCreateInput {
    /// One of: issue, error, recommendation.
    pub category: FeedbackCategory,
    /// Short summary — becomes the GitHub issue title.
    pub title: String,
    /// Full description — becomes the GitHub issue body.
    pub body: String,
}

pub fn feedback_create(
    store: &FeedbackStore,
    input: FeedbackCreateInput,
) -> ToolResult<FeedbackEntry> {
    match store.create(input.category, input.title, input.body) {
        Ok(entry) => ToolResult::ok(format!("Drafted feedback entry {}", entry.id), entry),
        Err(e) => store_err("Failed to create feedback entry", e),
    }
}

// ---------------------------------------------------------------------------
// feedback_list
// ---------------------------------------------------------------------------

pub fn feedback_list(store: &FeedbackStore) -> ToolResult<Vec<FeedbackEntry>> {
    match store.list() {
        Ok(entries) => ToolResult::ok(format!("{} local feedback entries", entries.len()), entries),
        Err(e) => store_err("Failed to list feedback entries", e),
    }
}

// ---------------------------------------------------------------------------
// feedback_get
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct FeedbackIdInput {
    /// Feedback entry id (from `feedback_list`).
    pub id: u64,
}

pub fn feedback_get(store: &FeedbackStore, input: FeedbackIdInput) -> ToolResult<FeedbackEntry> {
    match store.get(input.id) {
        Ok(entry) => ToolResult::ok(format!("Retrieved feedback entry {}", input.id), entry),
        Err(e) => store_err("Failed to get feedback entry", e),
    }
}

// ---------------------------------------------------------------------------
// feedback_update
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct FeedbackUpdateInput {
    pub id: u64,
    pub category: Option<FeedbackCategory>,
    pub title: Option<String>,
    pub body: Option<String>,
}

pub fn feedback_update(
    store: &FeedbackStore,
    input: FeedbackUpdateInput,
) -> ToolResult<FeedbackEntry> {
    match store.update(input.id, input.category, input.title, input.body) {
        Ok(entry) => ToolResult::ok(format!("Updated feedback entry {}", input.id), entry),
        Err(e) => store_err("Failed to update feedback entry", e),
    }
}

// ---------------------------------------------------------------------------
// feedback_delete
// ---------------------------------------------------------------------------

pub fn feedback_delete(store: &FeedbackStore, input: FeedbackIdInput) -> ToolResult<()> {
    match store.delete(input.id) {
        Ok(()) => ToolResult::ok_empty(format!("Deleted feedback entry {}", input.id)),
        Err(e) => store_err("Failed to delete feedback entry", e),
    }
}

// ---------------------------------------------------------------------------
// feedback_check_duplicates
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CheckDuplicatesInput {
    pub id: u64,
    /// GitHub repo to check against, as "owner/repo". Defaults to this
    /// server's own repo.
    #[serde(default)]
    pub repo: Option<String>,
    /// GitHub token, if the repo needs one. Falls back to $GITHUB_TOKEN,
    /// then `gh auth token`.
    #[serde(default)]
    pub token: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct CheckDuplicatesOutput {
    pub checked: bool,
    pub used_token: bool,
    pub reason: Option<String>,
    pub duplicates: Vec<github::GitHubIssueSummary>,
}

pub async fn feedback_check_duplicates(
    store: &FeedbackStore,
    http: &reqwest::Client,
    default_repo: &str,
    input: CheckDuplicatesInput,
) -> ToolResult<CheckDuplicatesOutput> {
    let entry = match store.get(input.id) {
        Ok(e) => e,
        Err(e) => return store_err("Failed to check duplicates", e),
    };
    let repo = input.repo.unwrap_or_else(|| default_repo.to_string());
    let token = github::resolve_token(input.token).await;

    match github::fetch_all_issues(http, &repo, token.as_deref()).await {
        Ok(issues) => {
            let duplicates = github::find_duplicates(&entry.title, &issues);
            ToolResult::ok(
                format!(
                    "Checked against {repo}; found {} possible duplicate(s)",
                    duplicates.len()
                ),
                CheckDuplicatesOutput {
                    checked: true,
                    used_token: token.is_some(),
                    reason: None,
                    duplicates,
                },
            )
        }
        Err(e) => ToolResult::ok(
            format!("Couldn't check for duplicates against {repo}: {e}"),
            CheckDuplicatesOutput {
                checked: false,
                used_token: token.is_some(),
                reason: Some(e.to_string()),
                duplicates: Vec::new(),
            },
        ),
    }
}

// ---------------------------------------------------------------------------
// feedback_submit
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct FeedbackSubmitInput {
    pub id: u64,
    /// GitHub repo to submit to, as "owner/repo". Defaults to this server's
    /// own repo.
    #[serde(default)]
    pub repo: Option<String>,
    /// GitHub token, for the duplicate check against a private repo. Falls
    /// back to $GITHUB_TOKEN, then `gh auth token`.
    #[serde(default)]
    pub token: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct FeedbackSubmitOutput {
    pub submitted: bool,
    pub duplicate: bool,
    pub matched_issue: Option<github::GitHubIssueSummary>,
    pub url: Option<String>,
    pub duplicate_check: String,
}

pub async fn feedback_submit(
    store: &FeedbackStore,
    http: &reqwest::Client,
    default_repo: &str,
    input: FeedbackSubmitInput,
    peer: Peer<RoleServer>,
) -> ToolResult<FeedbackSubmitOutput> {
    let entry = match store.get(input.id) {
        Ok(e) => e,
        Err(e) => return store_err("Failed to submit feedback", e),
    };
    let repo = input.repo.unwrap_or_else(|| default_repo.to_string());
    let token = github::resolve_token(input.token).await;

    // Check for an existing GitHub issue before bothering a human with a
    // review prompt at all — if this already exists there, there's nothing
    // to review, just discard the local draft.
    let duplicate_check;
    match github::fetch_all_issues(http, &repo, token.as_deref()).await {
        Ok(issues) => {
            duplicate_check = "ok, no duplicate found".to_string();
            if let Some(dup) = github::find_duplicates(&entry.title, &issues)
                .into_iter()
                .next()
            {
                let _ = store.delete(entry.id);
                return ToolResult::ok(
                    format!("Found an existing matching issue (#{}); discarded the local draft without submitting", dup.number),
                    FeedbackSubmitOutput {
                        submitted: false,
                        duplicate: true,
                        matched_issue: Some(dup),
                        url: None,
                        duplicate_check,
                    },
                );
            }
        }
        Err(e) => duplicate_check = format!("skipped: {e}"),
    }

    let modes = peer.supported_elicitation_modes();

    // Step 1: form-mode elicitation — let the human review (and tweak) the
    // content before anything about it leaves this machine, since it was
    // an agent asking to send it, not them.
    let (title, body) = if modes.contains(&ElicitationMode::Form) {
        let schema = ElicitationSchema::builder()
            .required_string_with("title", |s| {
                s.title("Title").with_default(entry.title.clone())
            })
            .required_string_with("body", |s| {
                s.title("Description").with_default(entry.body.clone())
            })
            .title("Review feedback before submitting")
            .build()
            .expect("two required string properties is always a valid elicitation schema");
        let request = ElicitRequestParams::FormElicitationParams {
            meta: None,
            message: format!(
                "An AI agent wants to submit this feedback to {repo}'s GitHub Issues. Review it (edit either field if you like) before it's sent anywhere:"
            ),
            requested_schema: schema,
        };
        match peer
            .create_elicitation_with_timeout(request, Some(ELICITATION_TIMEOUT))
            .await
        {
            Ok(review) if review.action == ElicitationAction::Accept => {
                let content = review.content.unwrap_or_default();
                let t = content
                    .get("title")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or(&entry.title)
                    .to_string();
                let b = content
                    .get("body")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or(&entry.body)
                    .to_string();
                (t, b)
            }
            Ok(review) => {
                let outcome = match review.action {
                    ElicitationAction::Decline => "declined",
                    ElicitationAction::Cancel => "cancelled",
                    _ => "did not accept",
                };
                return ToolResult::ok(
                    format!("Not submitted: reviewer {outcome} the submission"),
                    FeedbackSubmitOutput {
                        submitted: false,
                        duplicate: false,
                        matched_issue: None,
                        url: None,
                        duplicate_check,
                    },
                );
            }
            // Client claimed form support but the call itself failed
            // (timeout, disconnect, ...) — fall back rather than fail outright.
            Err(_) => return fallback_submit(store, &entry, &repo, duplicate_check).await,
        }
    } else {
        // Client doesn't support elicitation at all — same fallback the
        // CLI-equivalent path takes.
        return fallback_submit(store, &entry, &repo, duplicate_check).await;
    };

    if title != entry.title || body != entry.body {
        let _ = store.update(entry.id, None, Some(title.clone()), Some(body.clone()));
    }
    let reviewed = FeedbackEntry {
        title,
        body,
        ..entry.clone()
    };
    let submit_url = FeedbackStore::submission_url(&reviewed, &repo);

    // Step 2: URL-mode elicitation — hand the prepopulated page to the
    // client rather than opening our own browser tab on the agent's say-so.
    if modes.contains(&ElicitationMode::Url) {
        let Ok(parsed_url) = url::Url::parse(&submit_url) else {
            return fallback_submit_url(store, entry.id, &submit_url, duplicate_check);
        };
        let elicitation_id = uuid::Uuid::new_v4().to_string();
        match peer
            .elicit_url_with_timeout(
                "Opening GitHub with this feedback pre-filled — review and click \"Create\" there to finish.",
                parsed_url,
                elicitation_id,
                Some(ELICITATION_TIMEOUT),
            )
            .await
        {
            // Only `Accept` means "the human actually opened/will open this
            // page" — `elicit_url` returns `Decline`/`Cancel` just as
            // legitimately as the form step does, and those used to be
            // treated identically to `Accept` here, marking an unreviewed
            // draft submitted.
            Ok(ElicitationAction::Accept) => {
                let _ = store.mark_submitted(entry.id, submit_url.clone());
                ToolResult::ok(
                    "Handed the pre-filled issue page to the client for review",
                    FeedbackSubmitOutput {
                        submitted: true,
                        duplicate: false,
                        matched_issue: None,
                        url: Some(submit_url),
                        duplicate_check,
                    },
                )
            }
            Ok(review_action) => {
                let outcome = match review_action {
                    ElicitationAction::Decline => "declined",
                    ElicitationAction::Cancel => "cancelled",
                    _ => "did not accept",
                };
                ToolResult::ok(
                    format!("Not submitted: reviewer {outcome} opening the page"),
                    FeedbackSubmitOutput {
                        submitted: false,
                        duplicate: false,
                        matched_issue: None,
                        url: None,
                        duplicate_check,
                    },
                )
            }
            Err(_) => fallback_submit_url(store, entry.id, &submit_url, duplicate_check),
        }
    } else {
        fallback_submit_url(store, entry.id, &submit_url, duplicate_check)
    }
}

/// The client couldn't (or didn't) handle elicitation at all — open the URL
/// directly on this machine instead of failing outright. Still gets the
/// feedback as far as a human's browser; a human still has to review and
/// click "Create" on the GitHub page itself.
async fn fallback_submit(
    store: &FeedbackStore,
    entry: &FeedbackEntry,
    repo: &str,
    duplicate_check: String,
) -> ToolResult<FeedbackSubmitOutput> {
    let url = FeedbackStore::submission_url(entry, repo);
    fallback_submit_url(store, entry.id, &url, duplicate_check)
}

fn fallback_submit_url(
    store: &FeedbackStore,
    id: u64,
    url: &str,
    duplicate_check: String,
) -> ToolResult<FeedbackSubmitOutput> {
    // Both steps' errors used to be silently discarded (`let _ =`), always
    // reporting `submitted: true` regardless — including on a headless
    // Linux host with no `xdg-open`, or a local store write failure. Check
    // both and only report success when the draft is actually marked
    // submitted; the URL is still returned either way so a human can open
    // it by hand if the automatic open failed.
    let open_result = open_url_in_browser(url);
    let mark_result = store.mark_submitted(id, url.to_string());

    match (&open_result, &mark_result) {
        (Ok(()), Ok(_)) => ToolResult::ok(
            "Client doesn't support MCP elicitation (or the review timed out) — opened the pre-filled issue page directly in your browser instead. A human still needs to review and click \"Create\" there.",
            FeedbackSubmitOutput {
                submitted: true,
                duplicate: false,
                matched_issue: None,
                url: Some(url.to_string()),
                duplicate_check,
            },
        ),
        _ => {
            let mut problems = Vec::new();
            if let Err(e) = &open_result {
                problems.push(format!("couldn't open a browser automatically ({e})"));
            }
            if let Err(e) = &mark_result {
                problems.push(format!("couldn't record it as submitted locally ({e})"));
            }
            ToolResult::ok(
                format!(
                    "Client doesn't support MCP elicitation (or the review timed out), and the fallback hit a problem: {}. Here's the pre-filled issue URL — open it yourself to finish: {url}",
                    problems.join("; ")
                ),
                FeedbackSubmitOutput {
                    submitted: false,
                    duplicate: false,
                    matched_issue: None,
                    url: Some(url.to_string()),
                    duplicate_check,
                },
            )
        }
    }
}

fn open_url_in_browser(url: &str) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut c = std::process::Command::new("open");
        c.arg(url);
        c
    };
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "start", "", url]);
        c
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = {
        let mut c = std::process::Command::new("xdg-open");
        c.arg(url);
        c
    };

    command.spawn().map(|_child| ()).map_err(|e| {
        format!(
            "{} failed to launch: {e}",
            command.get_program().to_string_lossy()
        )
    })
}
