//! Tool registry — converts between Warp's typed protobuf tool calls and
//! OpenAI's generic `tool_calls` JSON format.
//!
//! Local mode only advertises the subset of Warp tools that are common
//! enough to make a working agent: `run_shell_command`, `read_files`,
//! `grep`, `file_glob`, `apply_file_diffs`. Other tools won't be advertised
//! to the model — the model simply won't know they exist. The system
//! prompt nudges it toward this subset.
//!
//! The registry has two responsibilities:
//!
//! 1. **Outbound (LLM ← Warp)**: build `OpenAiTool` definitions with JSON
//!    schemas. These go in the `tools` array of every chat-completions
//!    request when local mode is active.
//! 2. **Inbound (LLM → Warp)**: parse an OpenAI `tool_calls` payload
//!    (function name + JSON arg blob) and construct the corresponding
//!    typed `api::message::tool_call::Tool` so the existing Warp client
//!    machinery can execute it.
//! 3. **History reconstruction (both directions)**: when replaying a
//!    multi-turn conversation, convert past Warp `ToolCall` and
//!    `ToolCallResult` proto messages back into OpenAI assistant/tool
//!    messages so the model sees its own previous tool use in context.

use ai::local_provider::{
    OpenAiChatMessage, OpenAiFunctionCall, OpenAiFunctionDef, OpenAiTool, OpenAiToolCall,
};
use serde::Deserialize;
use serde_json::{json, Value};
use warp_multi_agent_api::{self as api};

/// Names of the tools we advertise to local models. Keeping this list
/// short and stable makes it easy for users to reason about what their
/// model can do, and keeps the system prompt focused.
pub const TOOL_RUN_SHELL_COMMAND: &str = "run_shell_command";
pub const TOOL_READ_FILES: &str = "read_files";
pub const TOOL_GREP: &str = "grep";
pub const TOOL_FILE_GLOB: &str = "file_glob";
pub const TOOL_APPLY_FILE_DIFFS: &str = "apply_file_diffs";

/// Returns the OpenAI tool definitions that get sent to the local model.
/// Each definition includes a JSON Schema derived from the proto field
/// names so the model produces arguments we can decode without guessing.
pub fn supported_tools() -> Vec<OpenAiTool> {
    vec![
        run_shell_command_def(),
        read_files_def(),
        grep_def(),
        file_glob_def(),
        apply_file_diffs_def(),
    ]
}

fn function_tool(name: &str, description: &str, parameters: Value) -> OpenAiTool {
    OpenAiTool {
        kind: "function".into(),
        function: OpenAiFunctionDef {
            name: name.into(),
            description: description.into(),
            parameters,
        },
    }
}

fn run_shell_command_def() -> OpenAiTool {
    function_tool(
        TOOL_RUN_SHELL_COMMAND,
        "Run a shell command in the user's terminal and return its stdout, \
         stderr, and exit code. The user may approve, edit, or reject the \
         command before it executes — assume the result reflects what \
         actually ran.",
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute. Multi-line scripts are allowed."
                }
            },
            "required": ["command"]
        }),
    )
}

fn read_files_def() -> OpenAiTool {
    function_tool(
        TOOL_READ_FILES,
        "Read one or more files from the user's working directory. \
         Returns the contents of each file. Prefer this over `cat`.",
        json!({
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Relative or absolute paths of files to read."
                }
            },
            "required": ["paths"]
        }),
    )
}

fn grep_def() -> OpenAiTool {
    function_tool(
        TOOL_GREP,
        "Search file contents for one or more substrings/patterns. Returns \
         the file paths and line numbers of matches.",
        json!({
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Substrings or regex patterns to search for."
                },
                "path": {
                    "type": "string",
                    "description": "Relative path to the file or directory to search in. Defaults to current directory."
                }
            },
            "required": ["queries"]
        }),
    )
}

fn file_glob_def() -> OpenAiTool {
    function_tool(
        TOOL_FILE_GLOB,
        "Find files whose names match shell-style glob patterns (?, *, []). \
         Returns matching paths.",
        json!({
            "type": "object",
            "properties": {
                "patterns": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Glob patterns to match against file names."
                },
                "search_dir": {
                    "type": "string",
                    "description": "Relative path to the directory to search. Defaults to current directory."
                }
            },
            "required": ["patterns"]
        }),
    )
}

fn apply_file_diffs_def() -> OpenAiTool {
    function_tool(
        TOOL_APPLY_FILE_DIFFS,
        "Edit existing files via search-and-replace, create new files, or \
         delete files. Each diff replaces an exact substring within a file. \
         The `search` text must occur exactly once in the file.",
        json!({
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-sentence description of the change."
                },
                "diffs": {
                    "type": "array",
                    "description": "Search-and-replace edits to apply.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_path": { "type": "string" },
                            "search": { "type": "string", "description": "Exact substring to replace." },
                            "replace": { "type": "string", "description": "Replacement text." }
                        },
                        "required": ["file_path", "search", "replace"]
                    }
                },
                "new_files": {
                    "type": "array",
                    "description": "Files to create.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_path": { "type": "string" },
                            "content": { "type": "string" }
                        },
                        "required": ["file_path", "content"]
                    }
                },
                "deleted_files": {
                    "type": "array",
                    "description": "Files to delete.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_path": { "type": "string" }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            "required": ["summary"]
        }),
    )
}

/// Parse an OpenAI `function.arguments` JSON blob and build the
/// corresponding typed Warp [`api::message::tool_call::Tool`]. Returns
/// `None` if the function name is unknown or the args don't match the
/// expected schema (the model hallucinated). The caller surfaces failures
/// as a tool error message back to the LLM.
pub fn try_build_warp_tool(name: &str, args_json: &str) -> Option<api::message::tool_call::Tool> {
    use api::message::tool_call::Tool;
    match name {
        TOOL_RUN_SHELL_COMMAND => {
            let args: RunShellArgs = serde_json::from_str(args_json).ok()?;
            Some(Tool::RunShellCommand(api::message::tool_call::RunShellCommand {
                command: args.command,
                ..Default::default()
            }))
        }
        TOOL_READ_FILES => {
            let args: ReadFilesArgs = serde_json::from_str(args_json).ok()?;
            let files = args
                .paths
                .into_iter()
                .map(|name| api::message::tool_call::read_files::File {
                    name,
                    line_ranges: vec![],
                })
                .collect();
            Some(Tool::ReadFiles(api::message::tool_call::ReadFiles { files }))
        }
        TOOL_GREP => {
            let args: GrepArgs = serde_json::from_str(args_json).ok()?;
            Some(Tool::Grep(api::message::tool_call::Grep {
                queries: args.queries,
                path: args.path.unwrap_or_default(),
            }))
        }
        TOOL_FILE_GLOB => {
            let args: FileGlobArgs = serde_json::from_str(args_json).ok()?;
            Some(Tool::FileGlobV2(api::message::tool_call::FileGlobV2 {
                patterns: args.patterns,
                search_dir: args.search_dir.unwrap_or_default(),
                ..Default::default()
            }))
        }
        TOOL_APPLY_FILE_DIFFS => {
            let args: ApplyFileDiffsArgs = serde_json::from_str(args_json).ok()?;
            let diffs = args
                .diffs
                .unwrap_or_default()
                .into_iter()
                .map(|d| api::message::tool_call::apply_file_diffs::FileDiff {
                    file_path: d.file_path,
                    search: d.search,
                    replace: d.replace,
                })
                .collect();
            let new_files = args
                .new_files
                .unwrap_or_default()
                .into_iter()
                .map(|f| api::message::tool_call::apply_file_diffs::NewFile {
                    file_path: f.file_path,
                    content: f.content,
                })
                .collect();
            let deleted_files = args
                .deleted_files
                .unwrap_or_default()
                .into_iter()
                .map(|f| api::message::tool_call::apply_file_diffs::DeleteFile {
                    file_path: f.file_path,
                })
                .collect();
            Some(Tool::ApplyFileDiffs(
                api::message::tool_call::ApplyFileDiffs {
                    summary: args.summary,
                    diffs,
                    new_files,
                    deleted_files,
                    v4a_updates: vec![],
                },
            ))
        }
        _ => None,
    }
}

/// Convert a Warp [`api::message::tool_call::Tool`] back into the OpenAI
/// representation used in chat history (function name + args JSON). Used
/// to reconstruct the assistant's prior tool calls so the model has
/// continuity in multi-turn sessions.
pub fn warp_tool_to_openai_tool_call(
    tool: &api::message::tool_call::Tool,
    tool_call_id: &str,
) -> Option<OpenAiToolCall> {
    use api::message::tool_call::Tool;
    let (name, args) = match tool {
        Tool::RunShellCommand(c) => (
            TOOL_RUN_SHELL_COMMAND,
            json!({ "command": c.command }),
        ),
        Tool::ReadFiles(rf) => (
            TOOL_READ_FILES,
            json!({
                "paths": rf.files.iter().map(|f| &f.name).collect::<Vec<_>>(),
            }),
        ),
        Tool::Grep(g) => (
            TOOL_GREP,
            json!({
                "queries": g.queries,
                "path": g.path,
            }),
        ),
        Tool::FileGlobV2(fg) => (
            TOOL_FILE_GLOB,
            json!({
                "patterns": fg.patterns,
                "search_dir": fg.search_dir,
            }),
        ),
        Tool::ApplyFileDiffs(afd) => (
            TOOL_APPLY_FILE_DIFFS,
            json!({
                "summary": afd.summary,
                "diffs": afd.diffs.iter().map(|d| json!({
                    "file_path": d.file_path,
                    "search": d.search,
                    "replace": d.replace,
                })).collect::<Vec<_>>(),
                "new_files": afd.new_files.iter().map(|f| json!({
                    "file_path": f.file_path,
                    "content": f.content,
                })).collect::<Vec<_>>(),
                "deleted_files": afd.deleted_files.iter().map(|f| json!({
                    "file_path": f.file_path,
                })).collect::<Vec<_>>(),
            }),
        ),
        // Anything else: not advertised to local models, so it shouldn't
        // appear in their conversation history. Skip it rather than
        // surface garbage.
        _ => return None,
    };
    Some(OpenAiToolCall {
        id: tool_call_id.into(),
        kind: "function".into(),
        function: OpenAiFunctionCall {
            name: name.into(),
            arguments: serde_json::to_string(&args).unwrap_or_default(),
        },
    })
}

/// Build an OpenAI `tool` role message containing the textual rendering
/// of a Warp `ToolCallResult`. Returns `None` for result variants that
/// don't correspond to a tool we advertise.
pub fn warp_tool_result_to_openai_message(
    tool_call_id: String,
    result: &api::request::input::tool_call_result::Result,
) -> Option<OpenAiChatMessage> {
    let content = format_tool_call_result(result)?;
    Some(OpenAiChatMessage {
        role: "tool".into(),
        content: Some(content),
        tool_call_id: Some(tool_call_id),
        tool_calls: None,
    })
}

fn format_tool_call_result(
    result: &api::request::input::tool_call_result::Result,
) -> Option<String> {
    use api::request::input::tool_call_result::Result as R;
    match result {
        R::RunShellCommand(r) => Some(format_run_shell_result(r)),
        R::ReadFiles(r) => Some(format_read_files_result(r)),
        R::Grep(r) => Some(format_grep_result(r)),
        R::FileGlobV2(r) => Some(format_file_glob_v2_result(r)),
        R::ApplyFileDiffs(r) => Some(format_apply_file_diffs_result(r)),
        _ => None,
    }
}

fn format_run_shell_result(r: &api::RunShellCommandResult) -> String {
    use api::run_shell_command_result::Result as R;
    let mut out = String::new();
    if !r.command.is_empty() {
        out.push_str("command: ");
        out.push_str(&r.command);
        out.push('\n');
    }
    match &r.result {
        Some(R::CommandFinished(f)) => {
            out.push_str(&format!("exit_code: {}\n", f.exit_code));
            if !f.output.is_empty() {
                out.push_str("output:\n");
                out.push_str(&f.output);
            }
        }
        Some(R::LongRunningCommandSnapshot(_)) => {
            out.push_str("status: long_running (output not yet final)\n");
        }
        Some(R::PermissionDenied(_)) => {
            out.push_str("status: permission_denied (user rejected the command)\n");
        }
        None => {
            // Older clients populated the deprecated top-level fields.
            if !r.output.is_empty() {
                out.push_str(&format!("exit_code: {}\noutput:\n{}", r.exit_code, r.output));
            } else {
                out.push_str("status: empty");
            }
        }
    }
    out
}

fn format_read_files_result(r: &api::ReadFilesResult) -> String {
    use api::read_files_result::Result as R;
    match &r.result {
        Some(R::TextFilesSuccess(s)) => {
            let mut out = String::new();
            for f in &s.files {
                out.push_str(&format!("=== {} ===\n", f.file_path));
                out.push_str(&f.content);
                out.push('\n');
            }
            out
        }
        Some(R::AnyFilesSuccess(s)) => {
            // AnyFileContent only carries the content (image/binary
            // payload), no path. Surface a count instead.
            format!("({} non-text file(s) returned)", s.files.len())
        }
        Some(R::Error(e)) => format!("error: {}", e.message),
        None => "error: empty result".into(),
    }
}

fn format_grep_result(r: &api::GrepResult) -> String {
    use api::grep_result::Result as R;
    match &r.result {
        Some(R::Success(s)) => {
            let mut out = String::new();
            for fm in &s.matched_files {
                let lines = fm
                    .matched_lines
                    .iter()
                    .map(|m| m.line_number.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                out.push_str(&format!("{}: lines {}\n", fm.file_path, lines));
            }
            if out.is_empty() {
                out.push_str("(no matches)");
            }
            out
        }
        Some(R::Error(e)) => format!("error: {}", e.message),
        None => "error: empty result".into(),
    }
}

fn format_file_glob_v2_result(r: &api::FileGlobV2Result) -> String {
    use api::file_glob_v2_result::Result as R;
    match &r.result {
        Some(R::Success(s)) => {
            if s.matched_files.is_empty() {
                "(no matches)".into()
            } else {
                s.matched_files
                    .iter()
                    .map(|m| m.file_path.clone())
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
        Some(R::Error(e)) => format!("error: {}", e.message),
        None => "error: empty result".into(),
    }
}

fn format_apply_file_diffs_result(r: &api::ApplyFileDiffsResult) -> String {
    use api::apply_file_diffs_result::Result as R;
    match &r.result {
        Some(R::Success(s)) => {
            let updated = s.updated_files_v2.len();
            let deleted = s.deleted_files.len();
            format!(
                "applied: {} file(s) updated, {} file(s) deleted",
                updated, deleted
            )
        }
        Some(R::Error(e)) => format!("error: {}", e.message),
        None => "error: empty result".into(),
    }
}

// ---- Argument deserialization shapes ---------------------------------------
// These mirror the JSON Schemas exposed via [`supported_tools`]. We accept
// them permissively (extra fields are ignored) but reject obviously
// malformed payloads (missing required keys → return None upstream).

#[derive(Deserialize)]
struct RunShellArgs {
    command: String,
}

#[derive(Deserialize)]
struct ReadFilesArgs {
    paths: Vec<String>,
}

#[derive(Deserialize)]
struct GrepArgs {
    queries: Vec<String>,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Deserialize)]
struct FileGlobArgs {
    patterns: Vec<String>,
    #[serde(default)]
    search_dir: Option<String>,
}

#[derive(Deserialize)]
struct ApplyFileDiffsArgs {
    summary: String,
    #[serde(default)]
    diffs: Option<Vec<ApplyFileDiffEntry>>,
    #[serde(default)]
    new_files: Option<Vec<NewFileEntry>>,
    #[serde(default)]
    deleted_files: Option<Vec<DeletedFileEntry>>,
}

#[derive(Deserialize)]
struct ApplyFileDiffEntry {
    file_path: String,
    search: String,
    replace: String,
}

#[derive(Deserialize)]
struct NewFileEntry {
    file_path: String,
    content: String,
}

#[derive(Deserialize)]
struct DeletedFileEntry {
    file_path: String,
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definitions_have_unique_names() {
        let names: Vec<_> = supported_tools()
            .into_iter()
            .map(|t| t.function.name)
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(names.len(), sorted.len());
    }

    #[test]
    fn build_run_shell_command() {
        let tool = try_build_warp_tool(
            TOOL_RUN_SHELL_COMMAND,
            r#"{"command":"ls -la"}"#,
        )
        .unwrap();
        match tool {
            api::message::tool_call::Tool::RunShellCommand(c) => {
                assert_eq!(c.command, "ls -la");
            }
            _ => panic!("expected RunShellCommand"),
        }
    }

    #[test]
    fn build_read_files_with_multiple_paths() {
        let tool = try_build_warp_tool(
            TOOL_READ_FILES,
            r#"{"paths":["src/main.rs","Cargo.toml"]}"#,
        )
        .unwrap();
        match tool {
            api::message::tool_call::Tool::ReadFiles(rf) => {
                assert_eq!(rf.files.len(), 2);
                assert_eq!(rf.files[0].name, "src/main.rs");
                assert_eq!(rf.files[1].name, "Cargo.toml");
            }
            _ => panic!("expected ReadFiles"),
        }
    }

    #[test]
    fn build_grep_with_optional_path_omitted() {
        let tool = try_build_warp_tool(
            TOOL_GREP,
            r#"{"queries":["TODO","FIXME"]}"#,
        )
        .unwrap();
        match tool {
            api::message::tool_call::Tool::Grep(g) => {
                assert_eq!(g.queries, vec!["TODO".to_string(), "FIXME".to_string()]);
                assert_eq!(g.path, "");
            }
            _ => panic!("expected Grep"),
        }
    }

    #[test]
    fn build_file_glob_with_search_dir() {
        let tool = try_build_warp_tool(
            TOOL_FILE_GLOB,
            r#"{"patterns":["*.rs"],"search_dir":"src"}"#,
        )
        .unwrap();
        match tool {
            api::message::tool_call::Tool::FileGlobV2(fg) => {
                assert_eq!(fg.patterns, vec!["*.rs".to_string()]);
                assert_eq!(fg.search_dir, "src");
            }
            _ => panic!("expected FileGlobV2"),
        }
    }

    #[test]
    fn build_apply_file_diffs() {
        let json = r#"{
            "summary": "fix typo",
            "diffs": [{"file_path":"a.rs","search":"foo","replace":"bar"}],
            "new_files": [{"file_path":"b.rs","content":"// new"}],
            "deleted_files": [{"file_path":"c.rs"}]
        }"#;
        let tool = try_build_warp_tool(TOOL_APPLY_FILE_DIFFS, json).unwrap();
        match tool {
            api::message::tool_call::Tool::ApplyFileDiffs(afd) => {
                assert_eq!(afd.summary, "fix typo");
                assert_eq!(afd.diffs.len(), 1);
                assert_eq!(afd.diffs[0].file_path, "a.rs");
                assert_eq!(afd.diffs[0].search, "foo");
                assert_eq!(afd.diffs[0].replace, "bar");
                assert_eq!(afd.new_files.len(), 1);
                assert_eq!(afd.new_files[0].file_path, "b.rs");
                assert_eq!(afd.deleted_files.len(), 1);
                assert_eq!(afd.deleted_files[0].file_path, "c.rs");
            }
            _ => panic!("expected ApplyFileDiffs"),
        }
    }

    #[test]
    fn build_apply_file_diffs_with_only_new_files() {
        let json = r#"{
            "summary": "create",
            "new_files": [{"file_path":"a.rs","content":"x"}]
        }"#;
        let tool = try_build_warp_tool(TOOL_APPLY_FILE_DIFFS, json).unwrap();
        match tool {
            api::message::tool_call::Tool::ApplyFileDiffs(afd) => {
                assert!(afd.diffs.is_empty());
                assert_eq!(afd.new_files.len(), 1);
                assert!(afd.deleted_files.is_empty());
            }
            _ => panic!("expected ApplyFileDiffs"),
        }
    }

    #[test]
    fn unknown_tool_name_returns_none() {
        assert!(try_build_warp_tool("does_not_exist", "{}").is_none());
    }

    #[test]
    fn malformed_json_returns_none() {
        assert!(try_build_warp_tool(TOOL_RUN_SHELL_COMMAND, "not json").is_none());
    }

    #[test]
    fn missing_required_field_returns_none() {
        // run_shell_command requires `command`.
        assert!(try_build_warp_tool(TOOL_RUN_SHELL_COMMAND, "{}").is_none());
    }

    #[test]
    fn round_trip_run_shell_command() {
        let tool = api::message::tool_call::Tool::RunShellCommand(
            api::message::tool_call::RunShellCommand {
                command: "echo hello".into(),
                ..Default::default()
            },
        );
        let openai = warp_tool_to_openai_tool_call(&tool, "call_1").unwrap();
        assert_eq!(openai.function.name, TOOL_RUN_SHELL_COMMAND);
        assert!(openai.function.arguments.contains("echo hello"));
        let rebuilt = try_build_warp_tool(&openai.function.name, &openai.function.arguments)
            .expect("should rebuild");
        match rebuilt {
            api::message::tool_call::Tool::RunShellCommand(c) => {
                assert_eq!(c.command, "echo hello");
            }
            _ => panic!("round trip lost type"),
        }
    }

    #[test]
    fn format_run_shell_result_includes_exit_code_and_output() {
        let result = api::RunShellCommandResult {
            command: "ls".into(),
            result: Some(api::run_shell_command_result::Result::CommandFinished(
                api::ShellCommandFinished {
                    exit_code: 0,
                    output: "file.txt\n".into(),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let s = format_run_shell_result(&result);
        assert!(s.contains("exit_code: 0"));
        assert!(s.contains("file.txt"));
    }

    #[test]
    fn format_grep_result_renders_matches() {
        let result = api::GrepResult {
            result: Some(api::grep_result::Result::Success(api::grep_result::Success {
                matched_files: vec![api::grep_result::success::GrepFileMatch {
                    file_path: "src/main.rs".into(),
                    matched_lines: vec![
                        api::grep_result::success::grep_file_match::GrepLineMatch {
                            line_number: 42,
                        },
                        api::grep_result::success::grep_file_match::GrepLineMatch {
                            line_number: 99,
                        },
                    ],
                }],
            })),
        };
        let s = format_grep_result(&result);
        assert!(s.contains("src/main.rs"));
        assert!(s.contains("42"));
        assert!(s.contains("99"));
    }
}
