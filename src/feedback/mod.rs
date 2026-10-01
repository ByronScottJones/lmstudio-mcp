//! Feedback-about-this-server, filed as GitHub issues — the same process as
//! `uictl-mac-mcp`'s `feedback` verb: draft locally first, check for an
//! existing duplicate issue, then hand a specific entry off to GitHub —
//! never by silently filing it through the API, always by opening (or, over
//! MCP, eliciting confirmation on) a pre-filled "new issue" page a human
//! still has to review and click "Create" on.

pub mod github;
pub mod store;
