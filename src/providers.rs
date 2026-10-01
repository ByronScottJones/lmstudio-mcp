//! The set of backends this server can talk to, and what each one
//! supports. Exactly one is active per server instance (selected by
//! `LLM_PROVIDER`) — this isn't a multi-backend router.

use std::fmt;

/// Which wire protocol a provider's chat/completions endpoint speaks.
/// Determines which code path `ApiClient::chat_completion` takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireFormat {
    /// `/v1/chat/completions`, OpenAI's request/response/streaming shape.
    /// LM Studio, Ollama, and OpenAI itself all speak this.
    OpenAiCompatible,
    /// `/v1/messages`, Anthropic's own shape — different auth header,
    /// different request fields (`system`, required `max_tokens`,
    /// `input_schema` instead of `function.parameters`), different SSE
    /// event sequence. Translated to/from the OpenAI shape inside
    /// `ApiClient` so nothing above it needs to know the difference.
    Anthropic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    LmStudio,
    Ollama,
    OpenAi,
    Anthropic,
}

impl Provider {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "lmstudio" | "lm-studio" | "lm_studio" => Some(Provider::LmStudio),
            "ollama" => Some(Provider::Ollama),
            "openai" | "open-ai" => Some(Provider::OpenAi),
            "anthropic" | "claude" => Some(Provider::Anthropic),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Provider::LmStudio => "lmstudio",
            Provider::Ollama => "ollama",
            Provider::OpenAi => "openai",
            Provider::Anthropic => "anthropic",
        }
    }

    /// Used when neither `LLM_BASE_URL` nor (for `lmstudio`) the legacy
    /// `LMSTUDIO_*` variables are set.
    pub fn default_base_url(self) -> &'static str {
        match self {
            Provider::LmStudio => "http://127.0.0.1:1234",
            Provider::Ollama => "http://127.0.0.1:11434",
            Provider::OpenAi => "https://api.openai.com",
            Provider::Anthropic => "https://api.anthropic.com",
        }
    }

    /// Cloud providers can't function without a key; local ones
    /// (LM Studio, Ollama) only need one if the user turned on auth.
    pub fn requires_api_key(self) -> bool {
        matches!(self, Provider::OpenAi | Provider::Anthropic)
    }

    pub fn wire_format(self) -> WireFormat {
        match self {
            Provider::Anthropic => WireFormat::Anthropic,
            _ => WireFormat::OpenAiCompatible,
        }
    }

    /// Whether this provider has a concept of explicitly loading/unloading
    /// a model into memory, with its own native API for it — as opposed to
    /// a cloud provider, where every model is just always available and
    /// there's nothing to load. Gates `load_model`/`unload_model`/
    /// `get_model_info`/`get_current_model`/`list_loaded_models`.
    pub fn supports_model_management(self) -> bool {
        matches!(self, Provider::LmStudio | Provider::Ollama)
    }

    /// Whether this provider has a `/v1/responses`-style stateful
    /// endpoint with `previous_response_id` chaining. Gates
    /// `create_response`/`start_conversation`/`continue_conversation`.
    /// (Ollama added a non-stateful `/v1/responses` in v0.13.3, but without
    /// `previous_response_id` support it can't back these conversation
    /// tools, so it's still excluded here.)
    pub fn supports_responses_api(self) -> bool {
        matches!(self, Provider::LmStudio | Provider::OpenAi)
    }

    /// Whether this provider has any embeddings endpoint at all. Anthropic
    /// doesn't offer one.
    pub fn supports_embeddings(self) -> bool {
        !matches!(self, Provider::Anthropic)
    }

    /// Whether this provider has a legacy `/v1/completions` (non-chat)
    /// endpoint. Anthropic has no equivalent at all.
    pub fn supports_text_completion(self) -> bool {
        !matches!(self, Provider::Anthropic)
    }
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_names_case_insensitively() {
        assert_eq!(Provider::parse("LMStudio"), Some(Provider::LmStudio));
        assert_eq!(Provider::parse("ollama"), Some(Provider::Ollama));
        assert_eq!(Provider::parse("OpenAI"), Some(Provider::OpenAi));
        assert_eq!(Provider::parse("Claude"), Some(Provider::Anthropic));
    }

    #[test]
    fn rejects_unknown_names() {
        assert_eq!(Provider::parse("bedrock"), None);
    }

    #[test]
    fn only_cloud_providers_require_a_key() {
        assert!(!Provider::LmStudio.requires_api_key());
        assert!(!Provider::Ollama.requires_api_key());
        assert!(Provider::OpenAi.requires_api_key());
        assert!(Provider::Anthropic.requires_api_key());
    }

    #[test]
    fn only_anthropic_uses_the_anthropic_wire_format() {
        assert_eq!(
            Provider::LmStudio.wire_format(),
            WireFormat::OpenAiCompatible
        );
        assert_eq!(Provider::Ollama.wire_format(), WireFormat::OpenAiCompatible);
        assert_eq!(Provider::OpenAi.wire_format(), WireFormat::OpenAiCompatible);
        assert_eq!(Provider::Anthropic.wire_format(), WireFormat::Anthropic);
    }

    #[test]
    fn only_local_providers_support_model_management() {
        assert!(Provider::LmStudio.supports_model_management());
        assert!(Provider::Ollama.supports_model_management());
        assert!(!Provider::OpenAi.supports_model_management());
        assert!(!Provider::Anthropic.supports_model_management());
    }

    #[test]
    fn anthropic_is_the_only_one_without_embeddings() {
        assert!(Provider::LmStudio.supports_embeddings());
        assert!(Provider::Ollama.supports_embeddings());
        assert!(Provider::OpenAi.supports_embeddings());
        assert!(!Provider::Anthropic.supports_embeddings());
    }

    #[test]
    fn anthropic_is_the_only_one_without_text_completion() {
        // Ollama documents /v1/completions as supported (verified against
        // its OpenAI-compatibility docs), so it's not excluded here.
        assert!(Provider::LmStudio.supports_text_completion());
        assert!(Provider::Ollama.supports_text_completion());
        assert!(Provider::OpenAi.supports_text_completion());
        assert!(!Provider::Anthropic.supports_text_completion());
    }
}
