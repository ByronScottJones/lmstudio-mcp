//! Lets the calling MCP client delegate a task to a local LM Studio model,
//! acting as a subagent with its own (sandboxed) tools — comparable to how
//! Claude Code spawns a subagent with a task-appropriate tool profile and
//! gets back a final report, not the full transcript.

pub mod guard;
pub mod runner;
pub mod tools;
