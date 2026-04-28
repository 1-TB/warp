//! Routes agent inference to a user-configured OpenAI-compatible endpoint
//! (Ollama, LM Studio, llama.cpp server, vLLM, hosted gateways) instead of
//! Warp's GraphQL backend.
//!
//! # Scope
//!
//! This is the Stage 2 happy-path implementation. It supports:
//!
//!   * Plain text user queries (the most recent `UserQuery` in the request).
//!   * Conversation history reconstructed from `task_context.tasks` —
//!     `UserQuery` → `user` role, `AgentOutput` → `assistant` role.
//!   * A single non-streaming POST to `/chat/completions`. The whole
//!     assistant reply is buffered, then emitted as one `AddMessagesToTask`
//!     event followed by `StreamFinished(Done)`.
//!
//! It deliberately does **not** support tool calls, streaming text deltas,
//! attachments, or the ~24 specialised `ToolCall` variants in the proto
//! schema. Calling local mode while the agent has tools enabled will cause
//! the model to either ignore them or produce text — both are acceptable
//! degradation modes for v1.
//!
//! Streaming, tool-call translation, and attachment handling will land in
//! follow-up commits. The interception point in
//! [`crate::server::server_api::ServerApi::generate_multi_agent_output`]
//! does not need to change to add those — only this module does.

use std::sync::Arc;

use anyhow::{anyhow, Context};
use futures::stream::StreamExt;
use prost_types::Timestamp;
use uuid::Uuid;
use warp_multi_agent_api::{self as api, ResponseEvent};

use ai::local_provider::{
    LocalEndpointConfig, OpenAiChatMessage, OpenAiChatRequest,
};

use super::{AIApiError, AIOutputStream};

const ROLE_SYSTEM: &str = "system";
const ROLE_USER: &str = "user";
const ROLE_ASSISTANT: &str = "assistant";

/// Build the OpenAI chat-completion request that corresponds to a Warp
/// `Request`. Currently extracts the running text history (UserQuery /
/// AgentOutput pairs) plus the latest user input. Returns `None` if no
/// user-facing input could be extracted — callers should fall back to the
/// original GraphQL path in that case rather than send an empty request.
pub fn translate_request(
    request: &api::Request,
    config: &LocalEndpointConfig,
) -> Option<OpenAiChatRequest> {
    let mut messages = Vec::new();

    // Conversation history: walk all tasks in order, then all messages in
    // each task. Map UserQuery → user, AgentOutput → assistant. Everything
    // else (tool calls, server events, reasoning) is skipped — those will
    // be handled in a follow-up.
    if let Some(task_context) = request.task_context.as_ref() {
        for task in &task_context.tasks {
            for msg in &task.messages {
                if let Some(payload) = msg.message.as_ref() {
                    match payload {
                        api::message::Message::UserQuery(q) if !q.query.is_empty() => {
                            messages.push(OpenAiChatMessage {
                                role: ROLE_USER.into(),
                                content: Some(q.query.clone()),
                                tool_call_id: None,
                                tool_calls: None,
                            });
                        }
                        api::message::Message::AgentOutput(a) if !a.text.is_empty() => {
                            messages.push(OpenAiChatMessage {
                                role: ROLE_ASSISTANT.into(),
                                content: Some(a.text.clone()),
                                tool_call_id: None,
                                tool_calls: None,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // The new user input. Newer requests carry `UserInputs` (a batch);
    // older ones use the deprecated direct `UserQuery`. Handle both.
    let new_input = extract_latest_user_query(request);
    if let Some(query) = new_input {
        if !query.is_empty() {
            messages.push(OpenAiChatMessage {
                role: ROLE_USER.into(),
                content: Some(query),
                tool_call_id: None,
                tool_calls: None,
            });
        }
    }

    if messages.is_empty() {
        return None;
    }

    // A minimal system prompt so models that lack agent training behave
    // sensibly. Intentionally generic — the user's choice of model can
    // override behaviour.
    messages.insert(
        0,
        OpenAiChatMessage {
            role: ROLE_SYSTEM.into(),
            content: Some(
                "You are an assistant embedded in the Warp terminal. Reply in plain text. \
                 Tool-use is not currently supported in local mode."
                    .into(),
            ),
            tool_call_id: None,
            tool_calls: None,
        },
    );

    Some(OpenAiChatRequest {
        model: config.model.clone(),
        messages,
        temperature: None,
        stream: Some(false),
        tools: None,
    })
}

fn extract_latest_user_query(request: &api::Request) -> Option<String> {
    let input = request.input.as_ref()?;
    let kind = input.r#type.as_ref()?;
    use api::request::input::Type;
    match kind {
        Type::UserInputs(inputs) => inputs.inputs.iter().rev().find_map(|i| {
            i.input.as_ref().and_then(|inner| match inner {
                api::request::input::user_inputs::user_input::Input::UserQuery(q) => {
                    Some(q.query.clone())
                }
                _ => None,
            })
        }),
        // The directly-on-Type UserQuery variant is deprecated in newer
        // proto revisions but still in use by older clients; handle it.
        #[allow(deprecated)]
        Type::UserQuery(q) => Some(q.query.clone()),
        _ => None,
    }
}

/// Identifies which task in the request the synthesised assistant reply
/// should be appended to. Picks the last task in `task_context.tasks`. If
/// the request doesn't carry any tasks, returns `None` — the caller will
/// surface an `InternalError` in that case.
fn pick_target_task_id(request: &api::Request) -> Option<String> {
    request
        .task_context
        .as_ref()?
        .tasks
        .last()
        .map(|t| t.id.clone())
}

/// Build a `ResponseEvent::ClientActions` event containing a single action.
fn wrap_action(action: api::client_action::Action) -> ResponseEvent {
    ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(action),
                }],
            },
        )),
    }
}

/// Build the synthetic `StreamInit` event echoed at the start of the
/// reply. Reuses the request's request_id from metadata if present so
/// telemetry / display ids stay consistent.
fn build_init_event(request: &api::Request, request_id: &str) -> ResponseEvent {
    let conversation_id = request
        .metadata
        .as_ref()
        .map(|m| m.conversation_id.clone())
        .unwrap_or_default();
    ResponseEvent {
        r#type: Some(api::response_event::Type::Init(
            api::response_event::StreamInit {
                request_id: request_id.to_string(),
                conversation_id,
                run_id: String::new(),
            },
        )),
    }
}

fn build_finished_event(reason: api::response_event::stream_finished::Reason) -> ResponseEvent {
    ResponseEvent {
        r#type: Some(api::response_event::Type::Finished(
            api::response_event::StreamFinished {
                reason: Some(reason),
                conversation_usage_metadata: None,
                token_usage: vec![],
                should_refresh_model_config: false,
                request_cost: None,
            },
        )),
    }
}

/// Build the `Message` carrying the assistant's full text reply.
fn build_agent_output_message(
    text: String,
    task_id: String,
    request_id: String,
) -> api::Message {
    api::Message {
        id: Uuid::new_v4().to_string(),
        task_id,
        request_id,
        server_message_data: String::new(),
        citations: vec![],
        timestamp: Some(now_timestamp()),
        message: Some(api::message::Message::AgentOutput(
            api::message::AgentOutput { text },
        )),
    }
}

fn now_timestamp() -> Timestamp {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    Timestamp {
        seconds: now.as_secs() as i64,
        nanos: now.subsec_nanos() as i32,
    }
}

/// Top-level entry point. Sends the request to the user's local
/// OpenAI-compatible endpoint and returns a stream of synthesised
/// `ResponseEvent`s the rest of the agent UI can consume unmodified.
///
/// Errors during the HTTP exchange are surfaced as a single
/// `StreamFinished(InternalError)` event rather than as a stream error,
/// so the UI displays the failure inline (matching how the real backend
/// reports model errors).
pub async fn dispatch_local(
    client: &http_client::Client,
    request: &api::Request,
    config: LocalEndpointConfig,
) -> Result<AIOutputStream<ResponseEvent>, Arc<AIApiError>> {
    let chat_request = translate_request(request, &config).ok_or_else(|| {
        Arc::new(AIApiError::Other(anyhow!(
            "no user input found to send to local endpoint"
        )))
    })?;

    let task_id = pick_target_task_id(request).ok_or_else(|| {
        Arc::new(AIApiError::Other(anyhow!(
            "local mode requires at least one task in task_context"
        )))
    })?;

    // Synthesize a fresh request id for the StreamInit event. The proto
    // `Metadata` only carries `conversation_id`; the request id is normally
    // generated server-side, so we mint one here.
    let request_id = Uuid::new_v4().to_string();

    let init = build_init_event(request, &request_id);

    // Run the HTTP exchange to completion before constructing the output
    // stream. v1 of local mode is non-streaming (`stream: false`); the
    // entire reply is buffered, so awaiting up front keeps lifetimes simple
    // and lets the resulting stream be `'static`.
    let url = config.chat_completions_url();
    let mut builder = client.post(url).json(&chat_request);
    if let Some(token) = config.api_key.as_deref() {
        builder = builder.bearer_auth(token);
    }

    let exchange_result = async {
        let response = builder
            .send()
            .await
            .context("POST to local endpoint failed")?;
        let status = response.status();
        let body = response
            .text()
            .await
            .context("reading local endpoint response body")?;
        if !status.is_success() {
            return Err(anyhow!(
                "local endpoint returned status {status}: {}",
                truncate_for_log(&body, 512)
            ));
        }
        let parsed: OpenAiChatResponse = serde_json::from_str(&body).with_context(|| {
            format!(
                "decoding local endpoint response: {}",
                truncate_for_log(&body, 512)
            )
        })?;
        parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| anyhow!("local endpoint returned no choices"))
    }
    .await;

    let events: Vec<ResponseEvent> = match exchange_result {
        Ok(text) => {
            let message = build_agent_output_message(text, task_id, request_id);
            let add_msg = wrap_action(api::client_action::Action::AddMessagesToTask(
                api::client_action::AddMessagesToTask {
                    task_id: message.task_id.clone(),
                    messages: vec![message],
                },
            ));
            let finished = build_finished_event(
                api::response_event::stream_finished::Reason::Done(
                    api::response_event::stream_finished::Done {},
                ),
            );
            vec![init, add_msg, finished]
        }
        Err(err) => {
            let message = format!("local endpoint error: {err:#}");
            let finished = build_finished_event(
                api::response_event::stream_finished::Reason::InternalError(
                    api::response_event::stream_finished::InternalError { message },
                ),
            );
            vec![init, finished]
        }
    };

    let stream = futures::stream::iter(events)
        .map(|event| Ok::<_, Arc<AIApiError>>(event));

    cfg_if::cfg_if! {
        if #[cfg(target_family = "wasm")] {
            Ok(stream.boxed_local())
        } else {
            Ok(stream.boxed())
        }
    }
}

fn truncate_for_log(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        s.split_at(max).0
    }
}

// ---- OpenAI response parsing -----------------------------------------------

#[derive(Debug, serde::Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiResponseMessage {
    content: Option<String>,
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> LocalEndpointConfig {
        LocalEndpointConfig {
            base_url: "http://localhost:11434/v1".into(),
            model: "qwen2.5-coder:7b".into(),
            api_key: None,
        }
    }

    fn user_msg(text: &str) -> api::Message {
        api::Message {
            id: format!("user-{text}"),
            task_id: "task-1".into(),
            request_id: "req-1".into(),
            server_message_data: String::new(),
            citations: vec![],
            timestamp: None,
            message: Some(api::message::Message::UserQuery(api::message::UserQuery {
                query: text.into(),
                ..Default::default()
            })),
        }
    }

    fn agent_msg(text: &str) -> api::Message {
        api::Message {
            id: format!("agent-{text}"),
            task_id: "task-1".into(),
            request_id: "req-1".into(),
            server_message_data: String::new(),
            citations: vec![],
            timestamp: None,
            message: Some(api::message::Message::AgentOutput(
                api::message::AgentOutput { text: text.into() },
            )),
        }
    }

    fn user_inputs_request(history: Vec<api::Message>, latest: &str) -> api::Request {
        api::Request {
            task_context: Some(api::request::TaskContext {
                tasks: vec![api::Task {
                    id: "task-1".into(),
                    messages: history,
                    ..Default::default()
                }],
            }),
            input: Some(api::request::Input {
                context: None,
                r#type: Some(api::request::input::Type::UserInputs(
                    api::request::input::UserInputs {
                        inputs: vec![api::request::input::user_inputs::UserInput {
                            input: Some(
                                api::request::input::user_inputs::user_input::Input::UserQuery(
                                    api::request::input::UserQuery {
                                        query: latest.into(),
                                        ..Default::default()
                                    },
                                ),
                            ),
                        }],
                    },
                )),
            }),
            ..Default::default()
        }
    }

    #[test]
    fn translate_request_emits_system_prompt_and_history() {
        let request = user_inputs_request(
            vec![user_msg("hi"), agent_msg("hello")],
            "what time is it?",
        );
        let req = translate_request(&request, &make_config()).expect("should translate");
        assert_eq!(req.model, "qwen2.5-coder:7b");
        assert_eq!(req.stream, Some(false));
        let roles: Vec<_> = req.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "user"]);
        assert_eq!(req.messages.last().unwrap().content.as_deref(), Some("what time is it?"));
    }

    #[test]
    #[allow(deprecated)]
    fn translate_request_handles_deprecated_direct_user_query() {
        let request = api::Request {
            task_context: Some(api::request::TaskContext {
                tasks: vec![api::Task {
                    id: "task-1".into(),
                    ..Default::default()
                }],
            }),
            input: Some(api::request::Input {
                context: None,
                r#type: Some(api::request::input::Type::UserQuery(
                    api::request::input::UserQuery {
                        query: "deprecated path".into(),
                        ..Default::default()
                    },
                )),
            }),
            ..Default::default()
        };
        let req = translate_request(&request, &make_config()).expect("should translate");
        assert_eq!(req.messages.last().unwrap().content.as_deref(), Some("deprecated path"));
    }

    #[test]
    fn translate_request_returns_none_with_no_input() {
        let request = api::Request::default();
        assert!(translate_request(&request, &make_config()).is_none());
    }

    #[test]
    fn pick_target_task_id_returns_last_task() {
        let request = api::Request {
            task_context: Some(api::request::TaskContext {
                tasks: vec![
                    api::Task {
                        id: "first".into(),
                        ..Default::default()
                    },
                    api::Task {
                        id: "last".into(),
                        ..Default::default()
                    },
                ],
            }),
            ..Default::default()
        };
        assert_eq!(pick_target_task_id(&request).as_deref(), Some("last"));
    }

    #[test]
    fn pick_target_task_id_returns_none_when_no_tasks() {
        let request = api::Request::default();
        assert!(pick_target_task_id(&request).is_none());
    }

    #[test]
    fn build_agent_output_message_carries_text() {
        let m = build_agent_output_message("hello world".into(), "t".into(), "r".into());
        assert_eq!(m.task_id, "t");
        assert_eq!(m.request_id, "r");
        assert!(m.timestamp.is_some());
        match m.message {
            Some(api::message::Message::AgentOutput(out)) => assert_eq!(out.text, "hello world"),
            _ => panic!("expected agent output"),
        }
    }
}
