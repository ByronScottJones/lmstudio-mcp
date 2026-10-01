//! The MCP server: wires each tool implementation in `crate::tools` up to
//! the `rmcp` tool-call dispatch machinery.

use crate::client::ApiClient;
use crate::feedback::store::FeedbackStore;
use crate::tools::{chat, embeddings, feedback, health_check, models, responses, subagent};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::{tool, tool_handler, tool_router, Peer, RoleServer, ServerHandler};
use std::sync::Arc;

/// This server's own repo — the default target for `feedback_submit` /
/// `feedback_check_duplicates` when the caller doesn't pass `repo`.
const DEFAULT_FEEDBACK_REPO: &str = "ByronScottJones/lmstudio-mcp";

#[derive(Clone)]
pub struct LmStudioServer {
    client: Arc<ApiClient>,
    feedback_store: Arc<FeedbackStore>,
    // Separate from `client`'s HTTP client: this one talks to api.github.com,
    // not LM Studio, and carries no LM Studio base URL/auth token.
    github_http: reqwest::Client,
    // Shared (not just cloned per-call) so every clone of this server sees
    // the same conversations — see PersonaCache's doc comment for why this
    // exists at all.
    personas: Arc<responses::PersonaCache>,
    // Read by the code `#[tool_handler]` generates to dispatch `call_tool`
    // requests; the dead-code lint can't see through that macro expansion.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

impl LmStudioServer {
    pub fn new(client: ApiClient, feedback_store: FeedbackStore) -> Self {
        Self {
            client: Arc::new(client),
            feedback_store: Arc::new(feedback_store),
            github_http: reqwest::Client::new(),
            personas: Arc::new(responses::PersonaCache::new()),
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl LmStudioServer {
    #[tool(
        description = "Check connectivity to the configured LLM provider (LM Studio, Ollama, OpenAI, or Anthropic — see LLM_PROVIDER)"
    )]
    async fn health_check(&self) -> Json<crate::types::ToolResult<health_check::HealthCheckData>> {
        Json(health_check::health_check(&self.client).await)
    }

    #[tool(
        description = "List models available from the configured provider — the local library for LM Studio/Ollama, or every model the API key can use for OpenAI/Anthropic"
    )]
    async fn list_models(&self) -> Json<crate::types::ToolResult<Vec<models::ModelSummary>>> {
        Json(models::list_models(&self.client).await)
    }

    #[tool(
        description = "List all currently loaded model instances. Only meaningful for LM Studio/Ollama — cloud providers have no loading concept and return an empty list"
    )]
    async fn list_loaded_models(
        &self,
    ) -> Json<crate::types::ToolResult<Vec<models::LoadedModelSummary>>> {
        Json(models::list_loaded_models(&self.client).await)
    }

    #[tool(
        description = "Identify the currently loaded model (or models, if more than one is loaded). Only meaningful for LM Studio/Ollama"
    )]
    async fn get_current_model(
        &self,
    ) -> Json<crate::types::ToolResult<Vec<models::LoadedModelSummary>>> {
        Json(models::get_current_model(&self.client).await)
    }

    #[tool(
        description = "Get detailed information about a specific loaded model instance. Only meaningful for LM Studio/Ollama"
    )]
    async fn get_model_info(
        &self,
        Parameters(input): Parameters<models::GetModelInfoInput>,
    ) -> Json<crate::types::ToolResult<models::LoadedModelSummary>> {
        Json(models::get_model_info(&self.client, input).await)
    }

    #[tool(
        description = "Load a model into memory. Only supported by LM Studio/Ollama — cloud providers have nothing to load"
    )]
    async fn load_model(
        &self,
        Parameters(input): Parameters<models::LoadModelInput>,
    ) -> Json<crate::types::ToolResult<models::LoadedModelData>> {
        Json(models::load_model(&self.client, input).await)
    }

    #[tool(description = "Unload a model instance from memory. Only supported by LM Studio/Ollama")]
    async fn unload_model(
        &self,
        Parameters(input): Parameters<models::UnloadModelInput>,
    ) -> Json<crate::types::ToolResult<()>> {
        Json(models::unload_model(&self.client, input).await)
    }

    #[tool(
        description = "Generate a chat completion from the configured provider's current/named model"
    )]
    async fn chat_completion(
        &self,
        Parameters(input): Parameters<chat::ChatCompletionInput>,
    ) -> Json<crate::types::ToolResult<chat::ChatCompletionData>> {
        Json(chat::chat_completion(&self.client, input).await)
    }

    #[tool(
        description = "Generate a raw text completion (non-chat format) — simpler and faster than chat_completion for single-turn tasks like code completion"
    )]
    async fn text_completion(
        &self,
        Parameters(input): Parameters<chat::TextCompletionInput>,
    ) -> Json<crate::types::ToolResult<chat::TextCompletionData>> {
        Json(chat::text_completion(&self.client, input).await)
    }

    #[tool(
        description = "Generate vector embeddings for text, for semantic search, RAG, and similarity comparisons. Requires an embedding-specific model (or provider — Anthropic has no embeddings endpoint)"
    )]
    async fn generate_embeddings(
        &self,
        Parameters(input): Parameters<embeddings::GenerateEmbeddingsInput>,
    ) -> Json<crate::types::ToolResult<embeddings::EmbeddingsData>> {
        Json(embeddings::generate_embeddings(&self.client, input).await)
    }

    #[tool(
        description = "Create a stateful response via a /v1/responses-style endpoint — conversation context is tracked server-side by response ID, no manual message history needed. Only LM Studio (v0.3.29+) and OpenAI support this; other providers return an error directing you to chat_completion instead"
    )]
    async fn create_response(
        &self,
        Parameters(input): Parameters<responses::CreateResponseInput>,
    ) -> Json<crate::types::ToolResult<responses::ResponseData>> {
        Json(responses::create_response(&self.client, input).await)
    }

    #[tool(
        description = "Start a stateful multi-turn conversation with a persistent system prompt. Returns a response_id to pass to continue_conversation. Only LM Studio (v0.3.29+) and OpenAI support this"
    )]
    async fn start_conversation(
        &self,
        Parameters(input): Parameters<responses::StartConversationInput>,
    ) -> Json<crate::types::ToolResult<responses::ResponseData>> {
        Json(responses::start_conversation(&self.client, &self.personas, input).await)
    }

    #[tool(
        description = "Continue a stateful conversation started with start_conversation — the original system prompt stays in effect automatically"
    )]
    async fn continue_conversation(
        &self,
        Parameters(input): Parameters<responses::ContinueConversationInput>,
    ) -> Json<crate::types::ToolResult<responses::ResponseData>> {
        Json(responses::continue_conversation(&self.client, &self.personas, input).await)
    }

    #[tool(
        description = "Delegate a task to a model from the configured provider acting as a subagent, with its own sandboxed tools (read_file, list_directory, search_files, and — depending on `capability` — write_file, run_command). Comparable to how Claude Code spawns a subagent: it works autonomously across as many tool calls as it needs and you get back a final report, not the full transcript. Good for offloading well-scoped, lower-level work (investigate a directory, make a specific edit, run and interpret a build/test command) from a smaller/local model instead of doing it yourself."
    )]
    async fn run_subagent(
        &self,
        Parameters(input): Parameters<subagent::RunSubagentInput>,
    ) -> Json<crate::types::ToolResult<crate::subagent::runner::SubagentReport>> {
        Json(subagent::run_subagent(&self.client, input).await)
    }

    #[tool(
        description = "Draft a new local feedback entry (issue/error/recommendation) about this server. Stored locally only — nothing leaves this machine until feedback_submit is called."
    )]
    async fn feedback_create(
        &self,
        Parameters(input): Parameters<feedback::FeedbackCreateInput>,
    ) -> Json<crate::types::ToolResult<crate::feedback::store::FeedbackEntry>> {
        Json(feedback::feedback_create(&self.feedback_store, input))
    }

    #[tool(description = "List all local feedback entries (drafts and submitted).")]
    async fn feedback_list(
        &self,
    ) -> Json<crate::types::ToolResult<Vec<crate::feedback::store::FeedbackEntry>>> {
        Json(feedback::feedback_list(&self.feedback_store))
    }

    #[tool(description = "Show one local feedback entry in full.")]
    async fn feedback_get(
        &self,
        Parameters(input): Parameters<feedback::FeedbackIdInput>,
    ) -> Json<crate::types::ToolResult<crate::feedback::store::FeedbackEntry>> {
        Json(feedback::feedback_get(&self.feedback_store, input))
    }

    #[tool(description = "Edit a local feedback entry's category, title, and/or body.")]
    async fn feedback_update(
        &self,
        Parameters(input): Parameters<feedback::FeedbackUpdateInput>,
    ) -> Json<crate::types::ToolResult<crate::feedback::store::FeedbackEntry>> {
        Json(feedback::feedback_update(&self.feedback_store, input))
    }

    #[tool(description = "Delete a local feedback entry.")]
    async fn feedback_delete(
        &self,
        Parameters(input): Parameters<feedback::FeedbackIdInput>,
    ) -> Json<crate::types::ToolResult<()>> {
        Json(feedback::feedback_delete(&self.feedback_store, input))
    }

    #[tool(
        description = "Check a local feedback entry's title against this repo's existing GitHub issues, without submitting anything. feedback_submit runs this same check automatically."
    )]
    async fn feedback_check_duplicates(
        &self,
        Parameters(input): Parameters<feedback::CheckDuplicatesInput>,
    ) -> Json<crate::types::ToolResult<feedback::CheckDuplicatesOutput>> {
        Json(
            feedback::feedback_check_duplicates(
                &self.feedback_store,
                &self.github_http,
                DEFAULT_FEEDBACK_REPO,
                input,
            )
            .await,
        )
    }

    #[tool(
        description = "Submit a local feedback entry to GitHub. Never files the issue directly through the API: first checks for an existing duplicate issue (discarding the local draft without submitting if one matches), then asks the connected client to let a human review — and optionally edit — the title/body via MCP elicitation, then hands the client a pre-filled \"new issue\" page to open via a second elicitation. A human still has to click \"Create\" there. If the client doesn't support elicitation, falls back to opening that page directly in this machine's browser instead."
    )]
    async fn feedback_submit(
        &self,
        Parameters(input): Parameters<feedback::FeedbackSubmitInput>,
        peer: Peer<RoleServer>,
    ) -> Json<crate::types::ToolResult<feedback::FeedbackSubmitOutput>> {
        Json(
            feedback::feedback_submit(
                &self.feedback_store,
                &self.github_http,
                DEFAULT_FEEDBACK_REPO,
                input,
                peer,
            )
            .await,
        )
    }
}

#[tool_handler(
    name = "lmstudio-mcp",
    version = "0.1.0",
    instructions = "Bridge to an LLM provider — LM Studio or Ollama (local), or OpenAI/Anthropic (cloud, require an API key) — selected via LLM_PROVIDER: inference (chat, text completion, embeddings, stateful conversations where supported), model management on local providers (list, load, unload), delegating work to a model as a sandboxed subagent (run_subagent), and filing feedback about this server itself as GitHub issues (feedback_*). Run health_check first to confirm the configured provider is reachable."
)]
impl ServerHandler for LmStudioServer {}
