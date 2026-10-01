//! lmstudio-mcp: an MCP server bridging Claude (and other MCP clients) to a
//! local LM Studio instance. Cross-platform (macOS, Windows, Linux).

mod client;
mod config;
mod server;
mod tools;
mod types;

use client::LmStudioClient;
use rmcp::transport::stdio;
use rmcp::ServiceExt;
use server::LmStudioServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // All logging goes to stderr — stdout is reserved for the MCP stdio
    // protocol and must never carry anything but protocol messages.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = config::Config::from_env();
    tracing::info!(
        native_base_url = %config.native_base_url,
        openai_base_url = %config.openai_base_url,
        "Starting lmstudio-mcp"
    );

    let client = LmStudioClient::new(&config);
    let server = LmStudioServer::new(client);

    let running = server.serve(stdio()).await?;

    // Grab a (non-owning) cancellation handle before `running` is consumed by
    // `.waiting()` below, so a signal handler running concurrently can still
    // ask the service to shut down.
    let shutdown = running.cancellation_token();
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            let mut sigterm =
                match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!("Failed to install SIGTERM handler: {e}");
                        return;
                    }
                };
            tokio::select! {
                _ = tokio::signal::ctrl_c() => tracing::info!("Received Ctrl-C, shutting down"),
                _ = sigterm.recv() => tracing::info!("Received SIGTERM, shutting down"),
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("Received Ctrl-C, shutting down");
        }
        shutdown.cancel();
    });

    // Runs until the peer disconnects (normal MCP client shutdown) or the
    // cancellation token above fires.
    running.waiting().await?;
    Ok(())
}
