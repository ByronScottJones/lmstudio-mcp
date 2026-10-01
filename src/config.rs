//! Runtime configuration, read from environment variables (set lazily, at
//! connect time, not at process startup, so a client that sets env vars via
//! its MCP server launch config always wins).

use crate::providers::Provider;
use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub provider: Provider,
    /// `scheme://host:port`, no trailing slash, no `/v1` or `/api/...` suffix.
    pub base_url: String,
    pub api_key: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error(
        "{provider} requires an API key — set LLM_API_KEY (or LMSTUDIO_API_TOKEN for lmstudio)"
    )]
    MissingApiKey { provider: Provider },
}

impl Config {
    /// Build configuration from the environment.
    ///
    /// - `LLM_PROVIDER`: one of `lmstudio` (default), `ollama`, `openai`,
    ///   `anthropic`.
    /// - `LLM_BASE_URL`: full `scheme://host:port` override. Falls back to
    ///   provider-specific defaults (and, for `lmstudio` only, the legacy
    ///   `LMSTUDIO_BASE_URL`/`LMSTUDIO_HOST`/`LMSTUDIO_PORT` variables, kept
    ///   working exactly as before for existing configs).
    /// - `LLM_API_KEY`: bearer/API key, if the provider needs one. Falls
    ///   back to `LMSTUDIO_API_TOKEN` for `lmstudio`. Required (hard error)
    ///   for `openai` and `anthropic`.
    pub fn from_env() -> Result<Self, ConfigError> {
        let provider = match env::var("LLM_PROVIDER") {
            Ok(raw) => Provider::parse(&raw).unwrap_or_else(|| {
                tracing::warn!("Unknown LLM_PROVIDER '{raw}', falling back to lmstudio");
                Provider::LmStudio
            }),
            Err(_) => Provider::LmStudio,
        };

        let base_url = env::var("LLM_BASE_URL")
            .ok()
            .or_else(|| {
                (provider == Provider::LmStudio)
                    .then(legacy_lmstudio_base_url)
                    .flatten()
            })
            .map(|s| s.trim_end_matches('/').to_string())
            .unwrap_or_else(|| provider.default_base_url().to_string());

        let api_key = env::var("LLM_API_KEY")
            .ok()
            .or_else(|| {
                (provider == Provider::LmStudio)
                    .then(|| env::var("LMSTUDIO_API_TOKEN").ok())
                    .flatten()
            })
            .filter(|s| !s.is_empty());

        if provider.requires_api_key() && api_key.is_none() {
            return Err(ConfigError::MissingApiKey { provider });
        }

        Ok(Self {
            provider,
            base_url,
            api_key,
        })
    }
}

/// Reproduces the pre-multi-provider resolution exactly, for `lmstudio`
/// only: `LMSTUDIO_BASE_URL` directly, else `LMSTUDIO_HOST`/`LMSTUDIO_PORT`
/// (defaults `127.0.0.1`/`1234`) composed into a URL. Returns `None` when
/// none of these are set, so the caller falls through to the provider
/// default.
fn legacy_lmstudio_base_url() -> Option<String> {
    if let Ok(base) = env::var("LMSTUDIO_BASE_URL") {
        return Some(base);
    }
    let host_set = env::var("LMSTUDIO_HOST").is_ok();
    let port_set = env::var("LMSTUDIO_PORT").is_ok();
    if !host_set && !port_set {
        return None;
    }
    let host = env::var("LMSTUDIO_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("LMSTUDIO_PORT").unwrap_or_else(|_| "1234".to_string());
    let port: u16 = port.parse().unwrap_or_else(|_| {
        tracing::warn!("Invalid LMSTUDIO_PORT '{port}', falling back to 1234");
        1234
    });
    Some(format!("http://{host}:{port}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Environment variables are process-global, so these tests serialize
    // against each other with a lock rather than relying on cargo test's
    // default parallelism to not interleave them.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    const VARS: &[&str] = &[
        "LLM_PROVIDER",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LMSTUDIO_BASE_URL",
        "LMSTUDIO_HOST",
        "LMSTUDIO_PORT",
        "LMSTUDIO_API_TOKEN",
    ];

    fn with_clean_env(f: impl FnOnce()) {
        let _guard = ENV_LOCK.lock().unwrap();
        let saved: Vec<(&str, Option<String>)> =
            VARS.iter().map(|v| (*v, env::var(v).ok())).collect();
        for v in VARS {
            unsafe { env::remove_var(v) };
        }
        f();
        for (k, v) in saved {
            match v {
                Some(v) => unsafe { env::set_var(k, v) },
                None => unsafe { env::remove_var(k) },
            }
        }
    }

    #[test]
    fn defaults_to_lmstudio_with_no_env_set() {
        with_clean_env(|| {
            let config = Config::from_env().unwrap();
            assert_eq!(config.provider, Provider::LmStudio);
            assert_eq!(config.base_url, "http://127.0.0.1:1234");
            assert_eq!(config.api_key, None);
        });
    }

    #[test]
    fn legacy_lmstudio_host_and_port_still_work() {
        with_clean_env(|| {
            unsafe {
                env::set_var("LMSTUDIO_HOST", "192.168.1.50");
                env::set_var("LMSTUDIO_PORT", "5678");
            }
            let config = Config::from_env().unwrap();
            assert_eq!(config.base_url, "http://192.168.1.50:5678");
        });
    }

    #[test]
    fn legacy_lmstudio_base_url_takes_precedence_over_host_port() {
        with_clean_env(|| {
            unsafe {
                env::set_var("LMSTUDIO_BASE_URL", "http://legacy:9999");
                env::set_var("LMSTUDIO_HOST", "ignored");
            }
            let config = Config::from_env().unwrap();
            assert_eq!(config.base_url, "http://legacy:9999");
        });
    }

    #[test]
    fn legacy_lmstudio_api_token_still_works() {
        with_clean_env(|| {
            unsafe { env::set_var("LMSTUDIO_API_TOKEN", "secret") };
            let config = Config::from_env().unwrap();
            assert_eq!(config.api_key, Some("secret".to_string()));
        });
    }

    #[test]
    fn new_generic_vars_select_ollama_with_its_own_default_port() {
        with_clean_env(|| {
            unsafe { env::set_var("LLM_PROVIDER", "ollama") };
            let config = Config::from_env().unwrap();
            assert_eq!(config.provider, Provider::Ollama);
            assert_eq!(config.base_url, "http://127.0.0.1:11434");
        });
    }

    #[test]
    fn openai_without_an_api_key_is_a_hard_error() {
        with_clean_env(|| {
            unsafe { env::set_var("LLM_PROVIDER", "openai") };
            assert!(matches!(
                Config::from_env(),
                Err(ConfigError::MissingApiKey {
                    provider: Provider::OpenAi
                })
            ));
        });
    }

    #[test]
    fn anthropic_with_llm_api_key_resolves_cleanly() {
        with_clean_env(|| {
            unsafe {
                env::set_var("LLM_PROVIDER", "anthropic");
                env::set_var("LLM_API_KEY", "sk-ant-test");
            }
            let config = Config::from_env().unwrap();
            assert_eq!(config.provider, Provider::Anthropic);
            assert_eq!(config.base_url, "https://api.anthropic.com");
            assert_eq!(config.api_key, Some("sk-ant-test".to_string()));
        });
    }

    #[test]
    fn llm_base_url_overrides_the_provider_default() {
        with_clean_env(|| {
            unsafe {
                env::set_var("LLM_PROVIDER", "ollama");
                env::set_var("LLM_BASE_URL", "http://remote-box:11434/");
            }
            let config = Config::from_env().unwrap();
            // Trailing slash stripped, same as the legacy behavior.
            assert_eq!(config.base_url, "http://remote-box:11434");
        });
    }

    #[test]
    fn unknown_provider_name_falls_back_to_lmstudio() {
        with_clean_env(|| {
            unsafe { env::set_var("LLM_PROVIDER", "bedrock") };
            let config = Config::from_env().unwrap();
            assert_eq!(config.provider, Provider::LmStudio);
        });
    }
}
