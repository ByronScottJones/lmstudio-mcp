//! The MCP server: wires each tool implementation in `crate::tools` up to
//! the `rmcp` tool-call dispatch machinery.

use crate::client::LmStudioClient;
use crate::tools::{chat, embeddings, health_check, models, responses};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use std::sync::Arc;

#[derive(Clone)]
pub struct LmStudioServer {
    client: Arc<LmStudioClient>,
    // Read by the code `#[tool_handler]` generates to dispatch `call_tool`
    // requests; the dead-code lint can't see through that macro expansion.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

impl LmStudioServer {
    pub fn new(client: LmStudioClient) -> Self {
        Self {
            client: Arc::new(client),
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl LmStudioServer {
    #[tool(description = "Check connectivity to the LM Studio server")]
    async fn health_check(&self) -> Json<crate::types::ToolResult<health_check::HealthCheckData>> {
        Json(health_check::health_check(&self.client).await)
    }

    #[tool(description = "List all downloaded models in LM Studio's local library")]
    async fn list_models(&self) -> Json<crate::types::ToolResult<Vec<models::ModelSummary>>> {
        Json(models::list_models(&self.client).await)
    }

    #[tool(description = "List all currently loaded model instances in LM Studio")]
    async fn list_loaded_models(
        &self,
    ) -> Json<crate::types::ToolResult<Vec<models::LoadedModelSummary>>> {
        Json(models::list_loaded_models(&self.client).await)
    }

    #[tool(
        description = "Identify the currently loaded model (or models, if more than one is loaded)"
    )]
    async fn get_current_model(
        &self,
    ) -> Json<crate::types::ToolResult<Vec<models::LoadedModelSummary>>> {
        Json(models::get_current_model(&self.client).await)
    }

    #[tool(description = "Get detailed information about a specific loaded model instance")]
    async fn get_model_info(
        &self,
        Parameters(input): Parameters<models::GetModelInfoInput>,
    ) -> Json<crate::types::ToolResult<models::LoadedModelSummary>> {
        Json(models::get_model_info(&self.client, input).await)
    }

    #[tool(description = "Load a model into memory in LM Studio")]
    async fn load_model(
        &self,
        Parameters(input): Parameters<models::LoadModelInput>,
    ) -> Json<crate::types::ToolResult<models::LoadedModelData>> {
        Json(models::load_model(&self.client, input).await)
    }

    #[tool(description = "Unload a model instance from memory in LM Studio")]
    async fn unload_model(
        &self,
        Parameters(input): Parameters<models::UnloadModelInput>,
    ) -> Json<crate::types::ToolResult<()>> {
        Json(models::unload_model(&self.client, input).await)
    }

    #[tool(description = "Generate a chat completion from the current LM Studio model")]
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
        description = "Generate vector embeddings for text, for semantic search, RAG, and similarity comparisons. Requires an embedding-specific model to be loaded"
    )]
    async fn generate_embeddings(
        &self,
        Parameters(input): Parameters<embeddings::GenerateEmbeddingsInput>,
    ) -> Json<crate::types::ToolResult<embeddings::EmbeddingsData>> {
        Json(embeddings::generate_embeddings(&self.client, input).await)
    }

    #[tool(
        description = "Create a stateful response via LM Studio's /v1/responses endpoint — conversation context is tracked server-side by response ID, no manual message history needed. Requires LM Studio v0.3.29+"
    )]
    async fn create_response(
        &self,
        Parameters(input): Parameters<responses::CreateResponseInput>,
    ) -> Json<crate::types::ToolResult<responses::ResponseData>> {
        Json(responses::create_response(&self.client, input).await)
    }

    #[tool(
        description = "Start a stateful multi-turn conversation with a persistent system prompt. Returns a response_id to pass to continue_conversation. Requires LM Studio v0.3.29+"
    )]
    async fn start_conversation(
        &self,
        Parameters(input): Parameters<responses::StartConversationInput>,
    ) -> Json<crate::types::ToolResult<responses::ResponseData>> {
        Json(responses::start_conversation(&self.client, input).await)
    }

    #[tool(
        description = "Continue a stateful conversation started with start_conversation — the original system prompt stays in effect automatically"
    )]
    async fn continue_conversation(
        &self,
        Parameters(input): Parameters<responses::ContinueConversationInput>,
    ) -> Json<crate::types::ToolResult<responses::ResponseData>> {
        Json(responses::continue_conversation(&self.client, input).await)
    }
}

#[tool_handler(
    name = "lmstudio-mcp",
    version = "0.1.0",
    instructions = "Bridge to a local LM Studio instance: inference (chat, text completion, embeddings, stateful conversations) and model management (list, load, unload). Run health_check first to confirm LM Studio is reachable."
)]
impl ServerHandler for LmStudioServer {}
