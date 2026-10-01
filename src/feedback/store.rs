//! Local-first storage for feedback entries, before a separate `submit`
//! step hands a specific one off to GitHub. One JSON file
//! (`~/.lmstudio-mcp/feedback.json`), read-modify-written on every call —
//! this isn't a hot path, so there's no in-memory cache to keep coherent,
//! just a mutex guarding against two calls racing on the file.
//!
//! Mirrors `uictl-mac-mcp`'s `FeedbackStore.swift`.

use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum FeedbackCategory {
    Issue,
    Error,
    Recommendation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum FeedbackStatus {
    Draft,
    Submitted,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FeedbackEntry {
    pub id: u64,
    pub category: FeedbackCategory,
    pub title: String,
    pub body: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub status: FeedbackStatus,
    pub submitted_at: Option<DateTime<Utc>>,
    pub submitted_url: Option<String>,
}

#[derive(Debug, Error)]
pub enum FeedbackStoreError {
    #[error("no feedback entry with id {0}")]
    NotFound(u64),
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to write {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("couldn't determine the home directory to store feedback in")]
    NoHomeDir,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct FileContents {
    next_id: u64,
    entries: Vec<FeedbackEntry>,
}

pub struct FeedbackStore {
    path: PathBuf,
    lock: Mutex<()>,
}

impl FeedbackStore {
    /// `~/.lmstudio-mcp/feedback.json`, created lazily on first write.
    pub fn new() -> Result<Self, FeedbackStoreError> {
        let home = dirs::home_dir().ok_or(FeedbackStoreError::NoHomeDir)?;
        Ok(Self {
            path: home.join(".lmstudio-mcp").join("feedback.json"),
            lock: Mutex::new(()),
        })
    }

    pub fn create(
        &self,
        category: FeedbackCategory,
        title: String,
        body: String,
    ) -> Result<FeedbackEntry, FeedbackStoreError> {
        let _guard = self.lock.lock().unwrap();
        let mut contents = self.load()?;
        let now = Utc::now();
        let entry = FeedbackEntry {
            id: contents.next_id,
            category,
            title,
            body,
            created_at: now,
            updated_at: now,
            status: FeedbackStatus::Draft,
            submitted_at: None,
            submitted_url: None,
        };
        contents.entries.push(entry.clone());
        contents.next_id += 1;
        self.save(&contents)?;
        Ok(entry)
    }

    pub fn list(&self) -> Result<Vec<FeedbackEntry>, FeedbackStoreError> {
        let _guard = self.lock.lock().unwrap();
        Ok(self.load()?.entries)
    }

    pub fn get(&self, id: u64) -> Result<FeedbackEntry, FeedbackStoreError> {
        let _guard = self.lock.lock().unwrap();
        let contents = self.load()?;
        contents
            .entries
            .into_iter()
            .find(|e| e.id == id)
            .ok_or(FeedbackStoreError::NotFound(id))
    }

    pub fn update(
        &self,
        id: u64,
        category: Option<FeedbackCategory>,
        title: Option<String>,
        body: Option<String>,
    ) -> Result<FeedbackEntry, FeedbackStoreError> {
        let _guard = self.lock.lock().unwrap();
        let mut contents = self.load()?;
        let entry = contents
            .entries
            .iter_mut()
            .find(|e| e.id == id)
            .ok_or(FeedbackStoreError::NotFound(id))?;
        if let Some(c) = category {
            entry.category = c;
        }
        if let Some(t) = title {
            entry.title = t;
        }
        if let Some(b) = body {
            entry.body = b;
        }
        entry.updated_at = Utc::now();
        let updated = entry.clone();
        self.save(&contents)?;
        Ok(updated)
    }

    pub fn delete(&self, id: u64) -> Result<(), FeedbackStoreError> {
        let _guard = self.lock.lock().unwrap();
        let mut contents = self.load()?;
        let len_before = contents.entries.len();
        contents.entries.retain(|e| e.id != id);
        if contents.entries.len() == len_before {
            return Err(FeedbackStoreError::NotFound(id));
        }
        self.save(&contents)
    }

    /// Marks an entry submitted without opening anything — used once a URL
    /// has already been handed to (and presumably opened by) an MCP client
    /// via URL-mode elicitation, so this doesn't also open its own browser
    /// tab for the same submission.
    pub fn mark_submitted(
        &self,
        id: u64,
        url: String,
    ) -> Result<FeedbackEntry, FeedbackStoreError> {
        let _guard = self.lock.lock().unwrap();
        let mut contents = self.load()?;
        let entry = contents
            .entries
            .iter_mut()
            .find(|e| e.id == id)
            .ok_or(FeedbackStoreError::NotFound(id))?;
        entry.status = FeedbackStatus::Submitted;
        entry.submitted_at = Some(Utc::now());
        entry.submitted_url = Some(url);
        let updated = entry.clone();
        self.save(&contents)?;
        Ok(updated)
    }

    /// GitHub natively pre-fills a new issue's title/body from query params.
    /// Category has no matching field there, so it's folded into the body.
    pub fn submission_url(entry: &FeedbackEntry, repo: &str) -> String {
        let category = match entry.category {
            FeedbackCategory::Issue => "issue",
            FeedbackCategory::Error => "error",
            FeedbackCategory::Recommendation => "recommendation",
        };
        let body = format!("**Category:** {category}\n\n{}", entry.body);
        url::Url::parse_with_params(
            &format!("https://github.com/{repo}/issues/new"),
            &[("title", entry.title.as_str()), ("body", body.as_str())],
        )
        .map(|u| u.to_string())
        // `repo` is always one of our own constants or a caller-supplied
        // "owner/repo" string with no scheme/host to go wrong — this can't
        // actually fail in practice, but degrade instead of panicking if it
        // somehow did.
        .unwrap_or_else(|_| format!("https://github.com/{repo}/issues/new"))
    }

    fn load(&self) -> Result<FileContents, FeedbackStoreError> {
        if !self.path.exists() {
            return Ok(FileContents {
                next_id: 1,
                entries: Vec::new(),
            });
        }
        let data = std::fs::read_to_string(&self.path).map_err(|e| FeedbackStoreError::Read {
            path: self.path.clone(),
            source: e,
        })?;
        serde_json::from_str(&data).map_err(|e| FeedbackStoreError::Parse {
            path: self.path.clone(),
            source: e,
        })
    }

    fn save(&self, contents: &FileContents) -> Result<(), FeedbackStoreError> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| FeedbackStoreError::Write {
                path: self.path.clone(),
                source: e,
            })?;
        }
        let json = serde_json::to_string_pretty(contents).expect("FileContents always serializes");
        // Write to a temp file and rename into place, so a crash or kill
        // mid-write can't leave a half-written, corrupt feedback.json.
        let tmp_path = self.path.with_extension("json.tmp");
        std::fs::write(&tmp_path, json).map_err(|e| FeedbackStoreError::Write {
            path: tmp_path.clone(),
            source: e,
        })?;
        std::fs::rename(&tmp_path, &self.path).map_err(|e| FeedbackStoreError::Write {
            path: self.path.clone(),
            source: e,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> FeedbackStore {
        let dir = std::env::temp_dir().join(format!("lmstudio-mcp-test-{}", uuid_like()));
        FeedbackStore {
            path: dir.join("feedback.json"),
            lock: Mutex::new(()),
        }
    }

    // Cheap unique-enough suffix without pulling in a uuid crate just for tests.
    fn uuid_like() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        format!(
            "{}-{:?}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            std::thread::current().id()
        )
    }

    #[test]
    fn create_list_get_roundtrip() {
        let store = temp_store();
        let entry = store
            .create(FeedbackCategory::Error, "title".into(), "body".into())
            .unwrap();
        assert_eq!(entry.id, 1);
        assert_eq!(entry.status, FeedbackStatus::Draft);

        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);

        let fetched = store.get(1).unwrap();
        assert_eq!(fetched.title, "title");
    }

    #[test]
    fn ids_increment_across_entries() {
        let store = temp_store();
        let a = store
            .create(FeedbackCategory::Issue, "a".into(), "".into())
            .unwrap();
        let b = store
            .create(FeedbackCategory::Issue, "b".into(), "".into())
            .unwrap();
        assert_eq!(a.id, 1);
        assert_eq!(b.id, 2);
    }

    #[test]
    fn update_only_touches_provided_fields() {
        let store = temp_store();
        store
            .create(
                FeedbackCategory::Recommendation,
                "orig title".into(),
                "orig body".into(),
            )
            .unwrap();
        let updated = store
            .update(1, None, Some("new title".into()), None)
            .unwrap();
        assert_eq!(updated.title, "new title");
        assert_eq!(updated.body, "orig body");
    }

    #[test]
    fn delete_removes_entry() {
        let store = temp_store();
        store
            .create(FeedbackCategory::Issue, "x".into(), "".into())
            .unwrap();
        store.delete(1).unwrap();
        assert!(matches!(store.get(1), Err(FeedbackStoreError::NotFound(1))));
    }

    #[test]
    fn get_missing_id_errors() {
        let store = temp_store();
        assert!(matches!(
            store.get(999),
            Err(FeedbackStoreError::NotFound(999))
        ));
    }

    #[test]
    fn mark_submitted_sets_status_and_url() {
        let store = temp_store();
        store
            .create(FeedbackCategory::Issue, "x".into(), "".into())
            .unwrap();
        let updated = store
            .mark_submitted(1, "https://github.com/x/y/issues/new?title=x".into())
            .unwrap();
        assert_eq!(updated.status, FeedbackStatus::Submitted);
        assert!(updated.submitted_url.is_some());
    }

    #[test]
    fn submission_url_folds_category_into_body_and_encodes_params() {
        let entry = FeedbackEntry {
            id: 1,
            category: FeedbackCategory::Error,
            title: "A bug & a space".into(),
            body: "details".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            status: FeedbackStatus::Draft,
            submitted_at: None,
            submitted_url: None,
        };
        let url = FeedbackStore::submission_url(&entry, "owner/repo");
        assert!(url.starts_with("https://github.com/owner/repo/issues/new?"));
        assert!(
            url.contains("title=A+bug+%26+a+space")
                || url.contains("title=A%20bug%20%26%20a%20space")
        );
        assert!(url.contains("body=%2A%2ACategory%3A%2A%2A+Error") || url.contains("Category"));
    }
}
