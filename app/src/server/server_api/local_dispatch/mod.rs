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

use anyhow::anyhow;
use futures::stream::StreamExt;
use prost_types::Timestamp;
use uuid::Uuid;
use warp_multi_agent_api::{self as api, ResponseEvent};

use ai::local_provider::{
    LocalEndpointConfig, OpenAiChatMessage, OpenAiChatRequest,
};

use super::{AIApiError, AIOutputStream};

mod tools;

const ROLE_SYSTEM: &str = "system";
const ROLE_USER: &str = "user";
const ROLE_ASSISTANT: &str = "assistant";
const ROLE_TOOL: &str = "tool";

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
    // each task. Each prior message becomes one OpenAI message:
    //
    //   * UserQuery               → user
    //   * AgentOutput             → assistant (text)
    //   * ToolCall                → assistant (with tool_calls field)
    //   * ToolCallResult          → tool (linked by tool_call_id)
    //
    // Other message types (server events, reasoning, summarisations,
    // etc.) carry no information the LLM needs to make its next move,
    // so we drop them. Tool calls for tools we don't advertise are also
    // dropped — the model wouldn't be able to interpret them.
    if let Some(task_context) = request.task_context.as_ref() {
        for task in &task_context.tasks {
            for msg in &task.messages {
                let Some(payload) = msg.message.as_ref() else {
                    continue;
                };
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
                    api::message::Message::ToolCall(tc) => {
                        if let Some(tool) = tc.tool.as_ref() {
                            if let Some(call) =
                                tools::warp_tool_to_openai_tool_call(tool, &tc.tool_call_id)
                            {
                                messages.push(OpenAiChatMessage {
                                    role: ROLE_ASSISTANT.into(),
                                    content: None,
                                    tool_call_id: None,
                                    tool_calls: Some(vec![call]),
                                });
                            }
                        }
                    }
                    api::message::Message::ToolCallResult(_) => {
                        // History tool-call results live on the *request* side
                        // (`request::input::tool_call_result`), not in the
                        // task's prior message stream. The proto carries a
                        // separate Message::ToolCallResult variant for some
                        // legacy paths, but treating it as opaque here is
                        // safe — the next request will resend the result on
                        // the input side.
                    }
                    _ => {}
                }
            }
        }
    }

    // The new user input. Newer requests carry `UserInputs` (a batch);
    // older ones use the deprecated direct `UserQuery`. Handle both, plus
    // ToolCallResult inputs (which is how the Warp client sends back a
    // tool result after the agent emitted a ToolCall message).
    if let Some(input) = request.input.as_ref() {
        match append_new_inputs(input, &mut messages) {
            InputKind::None => {}
            InputKind::UserQuery | InputKind::ToolResult => {
                // proceed
            }
        }
    }

    if messages.is_empty() {
        return None;
    }

    // System prompt: tells the model it can use the local-mode tool
    // subset and to reply in plain text otherwise.
    messages.insert(
        0,
        OpenAiChatMessage {
            role: ROLE_SYSTEM.into(),
            content: Some(SYSTEM_PROMPT.into()),
            tool_call_id: None,
            tool_calls: None,
        },
    );

    Some(OpenAiChatRequest {
        model: config.model.clone(),
        messages,
        temperature: None,
        stream: Some(true),
        tools: Some(tools::supported_tools()),
    })
}

const SYSTEM_PROMPT: &str = "You are an assistant embedded in the Warp \
terminal, running on the user's machine via a local model. Reply in plain \
text by default. You may call tools when they help: run_shell_command, \
read_files, grep, file_glob, apply_file_diffs. Prefer one tool call at a \
time and wait for the result before deciding the next step. Do not invent \
tools that are not listed.";

#[derive(Debug, PartialEq, Eq)]
enum InputKind {
    None,
    UserQuery,
    ToolResult,
}

/// Append OpenAI messages corresponding to the new request input. May
/// emit one user message (for a UserQuery) or one or more tool messages
/// (for ToolCallResults that close out a previous round).
fn append_new_inputs(
    input: &api::request::Input,
    messages: &mut Vec<OpenAiChatMessage>,
) -> InputKind {
    let Some(kind) = input.r#type.as_ref() else {
        return InputKind::None;
    };
    use api::request::input::Type;
    match kind {
        Type::UserInputs(inputs) => {
            let mut any = InputKind::None;
            for ui in &inputs.inputs {
                let Some(inner) = ui.input.as_ref() else {
                    continue;
                };
                use api::request::input::user_inputs::user_input::Input as UI;
                match inner {
                    UI::UserQuery(q) if !q.query.is_empty() => {
                        messages.push(OpenAiChatMessage {
                            role: ROLE_USER.into(),
                            content: Some(q.query.clone()),
                            tool_call_id: None,
                            tool_calls: None,
                        });
                        any = InputKind::UserQuery;
                    }
                    UI::ToolCallResult(r) => {
                        if let Some(result) = r.result.as_ref() {
                            if let Some(msg) = tools::warp_tool_result_to_openai_message(
                                r.tool_call_id.clone(),
                                result,
                            ) {
                                messages.push(msg);
                                if any == InputKind::None {
                                    any = InputKind::ToolResult;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            any
        }
        #[allow(deprecated)]
        Type::UserQuery(q) if !q.query.is_empty() => {
            messages.push(OpenAiChatMessage {
                role: ROLE_USER.into(),
                content: Some(q.query.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
            InputKind::UserQuery
        }
        #[allow(deprecated)]
        Type::ToolCallResult(r) => {
            if let Some(result) = r.result.as_ref() {
                if let Some(msg) =
                    tools::warp_tool_result_to_openai_message(r.tool_call_id.clone(), result)
                {
                    messages.push(msg);
                    return InputKind::ToolResult;
                }
            }
            InputKind::None
        }
        _ => InputKind::None,
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

    // Stream the response. Each OpenAI SSE chunk maps to 0+ Warp
    // ResponseEvents:
    //
    //   * The first chunk with content emits StreamInit + an
    //     AddMessagesToTask containing an empty AgentOutput placeholder
    //     (so the UI has a message to append into).
    //   * Each subsequent content delta emits an AppendToMessageContent
    //     that the existing client machinery uses to grow the rendered
    //     text live.
    //   * The first chunk carrying `finish_reason` (or the upstream
    //     [DONE] / EOF) emits StreamFinished(Done).
    //
    // Errors during the HTTP exchange become a single
    // StreamFinished(InternalError) event so the UI displays them inline.
    let url = config.chat_completions_url();
    let mut builder = client.post(url).json(&chat_request);
    if let Some(token) = config.api_key.as_deref() {
        builder = builder.bearer_auth(token);
    }
    let eventsource = builder.eventsource();

    let placeholder_message_id = Uuid::new_v4().to_string();
    let init_state = StreamingState {
        eventsource,
        task_id,
        request_id,
        message_id: placeholder_message_id,
        emitted_init: false,
        emitted_placeholder: false,
        finished: false,
        pending_init: Some(init),
        pending_tool_calls: std::collections::BTreeMap::new(),
    };

    let stream = futures::stream::unfold(Some(init_state), |state| async move {
        let mut state = state?;
        if state.finished {
            return None;
        }
        let events = state.next_events().await;
        // If next_events flagged us finished, the next poll will stop;
        // otherwise we recycle the same state.
        Some((events, Some(state)))
    })
    .flat_map(futures::stream::iter)
    .map(|event| Ok::<_, Arc<AIApiError>>(event));

    cfg_if::cfg_if! {
        if #[cfg(target_family = "wasm")] {
            Ok(stream.boxed_local())
        } else {
            Ok(stream.boxed())
        }
    }
}

/// State carried through the streaming pipeline. Held inside
/// `stream::unfold` between chunks of the OpenAI SSE stream.
struct StreamingState {
    eventsource: http_client::EventSourceStream,
    task_id: String,
    request_id: String,
    /// ID of the placeholder assistant message we emit on first content
    /// chunk; subsequent deltas append into it via FieldMask.
    message_id: String,
    emitted_init: bool,
    emitted_placeholder: bool,
    finished: bool,
    /// The pre-built StreamInit event, taken on first emission.
    pending_init: Option<ResponseEvent>,
    /// In-flight tool calls being assembled across SSE chunks. Indexed
    /// by the OpenAI delta `index`. Drained and emitted as Warp
    /// ToolCall messages when `finish_reason == "tool_calls"`.
    pending_tool_calls: std::collections::BTreeMap<usize, PendingToolCall>,
}

impl StreamingState {
    /// Pulls one SSE chunk and returns the Warp `ResponseEvent`s it
    /// produces (possibly empty, e.g. for `Event::Open` or empty deltas).
    async fn next_events(&mut self) -> Vec<ResponseEvent> {
        use futures::StreamExt as _;

        let mut events = Vec::new();
        let chunk = self.eventsource.next().await;
        match chunk {
            None => {
                // Upstream EOF without a finish_reason. Treat as Done.
                self.maybe_push_init(&mut events);
                if !self.finished {
                    events.push(self.done_event());
                    self.finished = true;
                }
            }
            Some(Ok(reqwest_eventsource::Event::Open)) => {
                // Connection established; nothing user-visible yet.
            }
            Some(Ok(reqwest_eventsource::Event::Message(message))) => {
                let data = message.data.trim();
                if data == "[DONE]" {
                    self.maybe_push_init(&mut events);
                    if !self.finished {
                        events.push(self.done_event());
                        self.finished = true;
                    }
                    return events;
                }
                match serde_json::from_str::<OpenAiStreamChunk>(data) {
                    Ok(chunk) => self.process_chunk(chunk, &mut events),
                    Err(e) => {
                        // Best-effort: drop malformed chunks but log so
                        // diagnosis is possible. A single bad frame
                        // shouldn't kill the whole stream.
                        log::warn!(
                            "local endpoint sent unparseable SSE frame: {e}; data={}",
                            truncate_for_log(data, 256)
                        );
                    }
                }
            }
            Some(Err(err)) => {
                self.maybe_push_init(&mut events);
                if !self.finished {
                    let message = format!("local endpoint stream error: {err}");
                    events.push(build_finished_event(
                        api::response_event::stream_finished::Reason::InternalError(
                            api::response_event::stream_finished::InternalError { message },
                        ),
                    ));
                    self.finished = true;
                }
            }
        }
        events
    }

    fn process_chunk(&mut self, chunk: OpenAiStreamChunk, events: &mut Vec<ResponseEvent>) {
        let Some(choice) = chunk.choices.into_iter().next() else {
            return;
        };
        let delta_content = choice.delta.content.unwrap_or_default();
        let has_content = !delta_content.is_empty();
        let finish = choice.finish_reason;

        // Accumulate any tool-call deltas. Emission happens on
        // finish_reason — only fully-assembled calls are useful since
        // their JSON arguments stream piece by piece.
        if let Some(deltas) = choice.delta.tool_calls {
            self.absorb_tool_call_deltas(deltas);
        }

        if has_content {
            self.maybe_push_init(events);
            self.maybe_push_placeholder(events);
            events.push(self.append_text_event(&delta_content));
        }

        if let Some(reason) = finish {
            self.maybe_push_init(events);
            let reason_kind = reason.as_str();

            // For tool_calls the model wants us to execute its tool
            // calls and round-trip the results — emit Warp ToolCall
            // messages so the existing client machinery handles them.
            // We deliberately do NOT emit the empty-text placeholder
            // when there's no content but tool calls exist, since
            // showing an empty assistant block above the tool calls is
            // visually noisy.
            if reason_kind == "tool_calls" || !self.pending_tool_calls.is_empty() {
                self.emit_pending_tool_calls(events);
            } else {
                // Plain text turn — keep an empty placeholder so the
                // assistant message renders even when the model emitted
                // zero content tokens (rare but possible).
                self.maybe_push_placeholder(events);
            }

            if !self.finished {
                events.push(match reason_kind {
                    "length" => build_finished_event(
                        api::response_event::stream_finished::Reason::MaxTokenLimit(
                            api::response_event::stream_finished::ReachedMaxTokenLimit {},
                        ),
                    ),
                    _ => self.done_event(),
                });
                self.finished = true;
            }
        }
    }

    /// Merge a batch of OpenAI tool-call deltas into in-flight pending
    /// calls keyed by index. Each delta may set the id, set the name, or
    /// extend the JSON arguments string.
    fn absorb_tool_call_deltas(&mut self, deltas: Vec<OpenAiStreamToolCallDelta>) {
        for d in deltas {
            let entry = self
                .pending_tool_calls
                .entry(d.index)
                .or_insert_with(PendingToolCall::default);
            if let Some(id) = d.id {
                if !id.is_empty() {
                    entry.id = id;
                }
            }
            if let Some(func) = d.function {
                if let Some(name) = func.name {
                    if !name.is_empty() {
                        entry.name = name;
                    }
                }
                if let Some(args) = func.arguments {
                    entry.arguments.push_str(&args);
                }
            }
        }
    }

    /// Drain `pending_tool_calls` and emit one `AddMessagesToTask` event
    /// containing a Warp `ToolCall` message per fully-assembled call.
    /// Calls with unknown names or unparseable args are emitted as
    /// AgentOutput error notes so the user can see what the model tried.
    fn emit_pending_tool_calls(&mut self, events: &mut Vec<ResponseEvent>) {
        if self.pending_tool_calls.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.pending_tool_calls);
        let mut messages = Vec::new();
        let mut errors = Vec::new();
        for (_, call) in pending {
            if call.name.is_empty() {
                errors.push("(model emitted a tool call with no function name)".to_string());
                continue;
            }
            // OpenAI guarantees the id; some local runtimes don't, so
            // mint one if missing — the Warp client only cares that
            // request and result use the same id.
            let tool_call_id = if call.id.is_empty() {
                Uuid::new_v4().to_string()
            } else {
                call.id.clone()
            };
            // Empty arg blob is valid JSON-equivalent for the no-arg
            // case ({}); some models emit `""` which we coerce.
            let args = if call.arguments.trim().is_empty() {
                "{}".to_string()
            } else {
                call.arguments.clone()
            };
            match tools::try_build_warp_tool(&call.name, &args) {
                Some(tool) => {
                    messages.push(api::Message {
                        id: Uuid::new_v4().to_string(),
                        task_id: self.task_id.clone(),
                        request_id: self.request_id.clone(),
                        server_message_data: String::new(),
                        citations: vec![],
                        timestamp: Some(now_timestamp()),
                        message: Some(api::message::Message::ToolCall(api::message::ToolCall {
                            tool_call_id,
                            tool: Some(tool),
                        })),
                    });
                }
                None => {
                    errors.push(format!(
                        "(model called unknown or malformed tool `{}` with args: {})",
                        call.name,
                        truncate_for_log(&args, 200)
                    ));
                }
            }
        }

        if !messages.is_empty() {
            events.push(wrap_action(
                api::client_action::Action::AddMessagesToTask(
                    api::client_action::AddMessagesToTask {
                        task_id: self.task_id.clone(),
                        messages,
                    },
                ),
            ));
        }
        // Surface tool-construction failures inline as an agent text
        // turn so the user knows the model misbehaved.
        if !errors.is_empty() {
            self.maybe_push_placeholder(events);
            events.push(self.append_text_event(&errors.join("\n")));
        }
    }

    fn maybe_push_init(&mut self, events: &mut Vec<ResponseEvent>) {
        if !self.emitted_init {
            if let Some(init) = self.pending_init.take() {
                events.push(init);
            }
            self.emitted_init = true;
        }
    }

    fn maybe_push_placeholder(&mut self, events: &mut Vec<ResponseEvent>) {
        if self.emitted_placeholder {
            return;
        }
        let placeholder = api::Message {
            id: self.message_id.clone(),
            task_id: self.task_id.clone(),
            request_id: self.request_id.clone(),
            server_message_data: String::new(),
            citations: vec![],
            timestamp: Some(now_timestamp()),
            message: Some(api::message::Message::AgentOutput(
                api::message::AgentOutput {
                    text: String::new(),
                },
            )),
        };
        events.push(wrap_action(
            api::client_action::Action::AddMessagesToTask(
                api::client_action::AddMessagesToTask {
                    task_id: self.task_id.clone(),
                    messages: vec![placeholder],
                },
            ),
        ));
        self.emitted_placeholder = true;
    }

    /// Builds an `AppendToMessageContent` that grows the placeholder
    /// `AgentOutput.text` by `delta`. The mask path matches the proto
    /// field path used elsewhere in the codebase for streamed text.
    fn append_text_event(&self, delta: &str) -> ResponseEvent {
        let delta_message = api::Message {
            id: self.message_id.clone(),
            task_id: self.task_id.clone(),
            request_id: self.request_id.clone(),
            server_message_data: String::new(),
            citations: vec![],
            timestamp: None,
            message: Some(api::message::Message::AgentOutput(
                api::message::AgentOutput {
                    text: delta.to_string(),
                },
            )),
        };
        wrap_action(api::client_action::Action::AppendToMessageContent(
            api::client_action::AppendToMessageContent {
                task_id: self.task_id.clone(),
                message: Some(delta_message),
                mask: Some(prost_types::FieldMask {
                    paths: vec!["message.agent_output.text".to_string()],
                }),
            },
        ))
    }

    fn done_event(&self) -> ResponseEvent {
        build_finished_event(api::response_event::stream_finished::Reason::Done(
            api::response_event::stream_finished::Done {},
        ))
    }
}

fn truncate_for_log(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        s.split_at(max).0
    }
}

// ---- OpenAI response parsing (streaming) -----------------------------------

/// Single SSE frame in OpenAI's `chat.completion.chunk` stream. We accept
/// only the fields we need; missing/unknown fields are silently ignored
/// (`#[serde(default)]`). This is intentionally permissive — local
/// runtimes vary in which optional fields they include.
#[derive(Debug, serde::Deserialize)]
struct OpenAiStreamChunk {
    #[serde(default)]
    choices: Vec<OpenAiStreamChoice>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiStreamChoice {
    #[serde(default)]
    delta: OpenAiStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct OpenAiStreamDelta {
    #[serde(default)]
    content: Option<String>,
    /// Tool-call deltas. Each chunk may carry partial information for
    /// any number of in-flight tool calls (identified by `index`). The
    /// `id`, `type`, and `function.name` fields appear on the first
    /// chunk for a given index; subsequent chunks for the same index
    /// only update `function.arguments` (which streams in pieces).
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiStreamToolCallDelta>>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiStreamToolCallDelta {
    /// Index identifying which tool call this delta belongs to. Multiple
    /// concurrent tool calls within one assistant turn use distinct
    /// indices.
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAiStreamFunctionDelta>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiStreamFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Incrementally assembled in-flight tool call. Multiple chunks contribute
/// to one of these — the `id` and `name` arrive in the first chunk, the
/// `arguments` JSON streams in pieces.
#[derive(Debug, Default)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
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
        assert_eq!(req.stream, Some(true));
        let roles: Vec<_> = req.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "user"]);
        assert_eq!(req.messages.last().unwrap().content.as_deref(), Some("what time is it?"));
        // Tools are now advertised so the model can call them.
        assert!(req.tools.is_some(), "tools should be advertised in local mode");
        let tools = req.tools.as_ref().unwrap();
        assert!(
            tools.iter().any(|t| t.function.name == "run_shell_command"),
            "run_shell_command should be in advertised tools"
        );
    }

    #[test]
    fn translate_request_walks_prior_tool_call_into_assistant_tool_calls() {
        // A history that contains a previous assistant tool call should
        // surface as an assistant message with a `tool_calls` entry, so
        // the model sees its own prior turn.
        let prior_tool_call = api::Message {
            id: "tc-1".into(),
            task_id: "task-1".into(),
            request_id: "req-prev".into(),
            server_message_data: String::new(),
            citations: vec![],
            timestamp: None,
            message: Some(api::message::Message::ToolCall(api::message::ToolCall {
                tool_call_id: "call_abc".into(),
                tool: Some(api::message::tool_call::Tool::RunShellCommand(
                    api::message::tool_call::RunShellCommand {
                        command: "ls".into(),
                        ..Default::default()
                    },
                )),
            })),
        };
        let request = user_inputs_request(vec![user_msg("hi"), prior_tool_call], "and now?");
        let req = translate_request(&request, &make_config()).expect("translate");
        let assistant_with_tool_calls = req
            .messages
            .iter()
            .find(|m| m.role == "assistant" && m.tool_calls.is_some())
            .expect("expected an assistant message with tool_calls");
        let calls = assistant_with_tool_calls.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "run_shell_command");
        assert!(calls[0].function.arguments.contains("ls"));
    }

    #[test]
    fn translate_request_routes_tool_call_result_input_to_tool_message() {
        // When the new input is a ToolCallResult, it should produce an
        // OpenAI `tool` role message keyed by tool_call_id.
        let result_input = api::request::Input {
            context: None,
            r#type: Some(api::request::input::Type::UserInputs(
                api::request::input::UserInputs {
                    inputs: vec![api::request::input::user_inputs::UserInput {
                        input: Some(
                            api::request::input::user_inputs::user_input::Input::ToolCallResult(
                                api::request::input::ToolCallResult {
                                    tool_call_id: "call_abc".into(),
                                    result: Some(
                                        api::request::input::tool_call_result::Result::RunShellCommand(
                                            api::RunShellCommandResult {
                                                command: "ls".into(),
                                                result: Some(
                                                    api::run_shell_command_result::Result::CommandFinished(
                                                        api::ShellCommandFinished {
                                                            exit_code: 0,
                                                            output: "Cargo.toml\n".into(),
                                                            ..Default::default()
                                                        },
                                                    ),
                                                ),
                                                ..Default::default()
                                            },
                                        ),
                                    ),
                                },
                            ),
                        ),
                    }],
                },
            )),
        };
        let request = api::Request {
            task_context: Some(api::request::TaskContext {
                tasks: vec![api::Task {
                    id: "task-1".into(),
                    messages: vec![user_msg("hi")],
                    ..Default::default()
                }],
            }),
            input: Some(result_input),
            ..Default::default()
        };
        let req = translate_request(&request, &make_config()).expect("translate");
        let tool_msg = req
            .messages
            .iter()
            .find(|m| m.role == "tool")
            .expect("expected a tool role message");
        assert_eq!(tool_msg.tool_call_id.as_deref(), Some("call_abc"));
        assert!(tool_msg.content.as_deref().unwrap().contains("Cargo.toml"));
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
    fn parse_openai_stream_chunk_with_content() {
        let raw = r#"{"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}"#;
        let chunk: OpenAiStreamChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hi"));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn parse_openai_stream_chunk_with_finish_reason() {
        let raw = r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
        let chunk: OpenAiStreamChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn parse_openai_stream_chunk_tolerates_unknown_fields() {
        // Some backends emit extra fields like `system_fingerprint`,
        // `usage` mid-stream, etc. We must not bail on those.
        let raw = r#"{"id":"x","object":"chat.completion.chunk","model":"m","system_fingerprint":"f","usage":{"prompt_tokens":1},"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null,"logprobs":null}]}"#;
        let chunk: OpenAiStreamChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("!"));
    }

    #[test]
    fn parse_openai_stream_chunk_empty_choices() {
        // First chunk from some backends has no choices (just role
        // metadata); we should accept and ignore.
        let raw = r#"{"id":"x","choices":[]}"#;
        let chunk: OpenAiStreamChunk = serde_json::from_str(raw).unwrap();
        assert!(chunk.choices.is_empty());
    }

    /// Build a `StreamingState` whose eventsource won't be polled — used
    /// only to exercise the synchronous helpers (`maybe_push_init`,
    /// `maybe_push_placeholder`, `append_text_event`, `process_chunk`)
    /// without needing a real HTTP fixture.
    fn fake_streaming_state() -> StreamingState {
        StreamingState {
            // An empty stream is fine — we never poll it in these tests.
            eventsource: futures::stream::empty().boxed(),
            task_id: "task-1".into(),
            request_id: "req-1".into(),
            message_id: "msg-1".into(),
            emitted_init: false,
            emitted_placeholder: false,
            finished: false,
            pending_init: Some(ResponseEvent {
                r#type: Some(api::response_event::Type::Init(
                    api::response_event::StreamInit {
                        request_id: "req-1".into(),
                        conversation_id: "conv".into(),
                        run_id: String::new(),
                    },
                )),
            }),
            pending_tool_calls: std::collections::BTreeMap::new(),
        }
    }

    #[test]
    fn process_chunk_emits_init_then_placeholder_then_append() {
        let mut state = fake_streaming_state();
        let chunk = OpenAiStreamChunk {
            choices: vec![OpenAiStreamChoice {
                delta: OpenAiStreamDelta {
                    content: Some("Hello".into()), tool_calls: None,
                },
                finish_reason: None,
            }],
        };
        let mut events = vec![];
        state.process_chunk(chunk, &mut events);
        assert_eq!(events.len(), 3, "init + placeholder + append");
        assert!(matches!(
            events[0].r#type,
            Some(api::response_event::Type::Init(_))
        ));
        // Second event is the placeholder AddMessagesToTask.
        let actions = match &events[1].r#type {
            Some(api::response_event::Type::ClientActions(a)) => &a.actions,
            _ => panic!("expected ClientActions"),
        };
        assert!(matches!(
            actions[0].action,
            Some(api::client_action::Action::AddMessagesToTask(_))
        ));
        // Third event is the AppendToMessageContent with our delta text.
        let actions = match &events[2].r#type {
            Some(api::response_event::Type::ClientActions(a)) => &a.actions,
            _ => panic!("expected ClientActions"),
        };
        match &actions[0].action {
            Some(api::client_action::Action::AppendToMessageContent(a)) => {
                let mask = a.mask.as_ref().unwrap();
                assert_eq!(mask.paths, vec!["message.agent_output.text".to_string()]);
                let msg = a.message.as_ref().unwrap();
                match &msg.message {
                    Some(api::message::Message::AgentOutput(out)) => {
                        assert_eq!(out.text, "Hello");
                    }
                    _ => panic!("expected agent output"),
                }
            }
            _ => panic!("expected AppendToMessageContent"),
        }
    }

    #[test]
    fn process_chunk_subsequent_deltas_skip_init_and_placeholder() {
        let mut state = fake_streaming_state();
        // First chunk: emits init + placeholder + append
        state.process_chunk(
            OpenAiStreamChunk {
                choices: vec![OpenAiStreamChoice {
                    delta: OpenAiStreamDelta {
                        content: Some("Hi".into()), tool_calls: None,
                    },
                    finish_reason: None,
                }],
            },
            &mut vec![],
        );
        // Second chunk: just the append delta.
        let mut events = vec![];
        state.process_chunk(
            OpenAiStreamChunk {
                choices: vec![OpenAiStreamChoice {
                    delta: OpenAiStreamDelta {
                        content: Some(" there".into()), tool_calls: None,
                    },
                    finish_reason: None,
                }],
            },
            &mut events,
        );
        assert_eq!(events.len(), 1, "only the append delta on subsequent chunk");
    }

    #[test]
    fn process_chunk_finish_reason_emits_done() {
        let mut state = fake_streaming_state();
        let mut events = vec![];
        state.process_chunk(
            OpenAiStreamChunk {
                choices: vec![OpenAiStreamChoice {
                    delta: OpenAiStreamDelta { content: None, tool_calls: None },
                    finish_reason: Some("stop".into()),
                }],
            },
            &mut events,
        );
        // Init + empty placeholder + StreamFinished(Done) — even with no
        // content, we keep the placeholder so the assistant turn renders.
        assert_eq!(events.len(), 3);
        assert!(matches!(
            events.last().unwrap().r#type,
            Some(api::response_event::Type::Finished(api::response_event::StreamFinished {
                reason: Some(api::response_event::stream_finished::Reason::Done(_)),
                ..
            }))
        ));
        assert!(state.finished);
    }

    #[test]
    fn process_chunk_finish_reason_length_emits_max_token_limit() {
        let mut state = fake_streaming_state();
        let mut events = vec![];
        state.process_chunk(
            OpenAiStreamChunk {
                choices: vec![OpenAiStreamChoice {
                    delta: OpenAiStreamDelta { content: None, tool_calls: None },
                    finish_reason: Some("length".into()),
                }],
            },
            &mut events,
        );
        assert!(matches!(
            events.last().unwrap().r#type,
            Some(api::response_event::Type::Finished(api::response_event::StreamFinished {
                reason: Some(api::response_event::stream_finished::Reason::MaxTokenLimit(_)),
                ..
            }))
        ));
    }

    fn tool_call_delta_chunk(
        index: usize,
        id: Option<&str>,
        name: Option<&str>,
        args_fragment: Option<&str>,
        finish_reason: Option<&str>,
    ) -> OpenAiStreamChunk {
        OpenAiStreamChunk {
            choices: vec![OpenAiStreamChoice {
                delta: OpenAiStreamDelta {
                    content: None,
                    tool_calls: Some(vec![OpenAiStreamToolCallDelta {
                        index,
                        id: id.map(String::from),
                        function: Some(OpenAiStreamFunctionDelta {
                            name: name.map(String::from),
                            arguments: args_fragment.map(String::from),
                        }),
                    }]),
                },
                finish_reason: finish_reason.map(String::from),
            }],
        }
    }

    #[test]
    fn streaming_tool_call_assembles_across_chunks() {
        let mut state = fake_streaming_state();
        let mut events = vec![];
        // OpenAI splits tool-call chunks: first carries id+name, subsequent
        // chunks extend `arguments` token-by-token.
        state.process_chunk(
            tool_call_delta_chunk(0, Some("call_42"), Some("run_shell_command"), Some(""), None),
            &mut events,
        );
        state.process_chunk(
            tool_call_delta_chunk(0, None, None, Some(r#"{"comm"#), None),
            &mut events,
        );
        state.process_chunk(
            tool_call_delta_chunk(0, None, None, Some(r#"and":"ls"}"#), None),
            &mut events,
        );
        // Mid-stream: nothing emitted yet (we wait for finish_reason).
        assert!(events.is_empty());
        // Final chunk with finish_reason: emit Init + AddMessagesToTask
        // with the ToolCall + StreamFinished.
        state.process_chunk(
            OpenAiStreamChunk {
                choices: vec![OpenAiStreamChoice {
                    delta: OpenAiStreamDelta {
                        content: None,
                        tool_calls: None,
                    },
                    finish_reason: Some("tool_calls".into()),
                }],
            },
            &mut events,
        );
        // Init + AddMessagesToTask + StreamFinished.
        assert_eq!(events.len(), 3);
        let actions = match &events[1].r#type {
            Some(api::response_event::Type::ClientActions(a)) => &a.actions,
            _ => panic!("expected ClientActions"),
        };
        match &actions[0].action {
            Some(api::client_action::Action::AddMessagesToTask(amt)) => {
                let msg = &amt.messages[0];
                match &msg.message {
                    Some(api::message::Message::ToolCall(tc)) => {
                        assert_eq!(tc.tool_call_id, "call_42");
                        match tc.tool.as_ref().unwrap() {
                            api::message::tool_call::Tool::RunShellCommand(c) => {
                                assert_eq!(c.command, "ls");
                            }
                            _ => panic!("expected RunShellCommand"),
                        }
                    }
                    _ => panic!("expected ToolCall message"),
                }
            }
            _ => panic!("expected AddMessagesToTask"),
        }
    }

    #[test]
    fn streaming_tool_call_with_unknown_name_falls_back_to_text() {
        let mut state = fake_streaming_state();
        let mut events = vec![];
        state.process_chunk(
            tool_call_delta_chunk(
                0,
                Some("call_x"),
                Some("nonexistent_tool"),
                Some("{}"),
                None,
            ),
            &mut events,
        );
        state.process_chunk(
            OpenAiStreamChunk {
                choices: vec![OpenAiStreamChoice {
                    delta: OpenAiStreamDelta {
                        content: None,
                        tool_calls: None,
                    },
                    finish_reason: Some("tool_calls".into()),
                }],
            },
            &mut events,
        );
        // The unknown tool surfaces as an inline assistant text note.
        let has_error_text = events.iter().any(|e| match &e.r#type {
            Some(api::response_event::Type::ClientActions(a)) => a.actions.iter().any(|act| {
                matches!(
                    &act.action,
                    Some(api::client_action::Action::AppendToMessageContent(amc))
                    if matches!(
                        amc.message.as_ref().and_then(|m| m.message.as_ref()),
                        Some(api::message::Message::AgentOutput(out)) if out.text.contains("nonexistent_tool")
                    )
                )
            }),
            _ => false,
        });
        assert!(has_error_text, "unknown tool name should surface as inline error text");
    }
}
