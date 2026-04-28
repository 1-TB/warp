//! Client-side dispatcher for the [`LLMProvider::Local`] variant — routes
//! agent requests to a user-configured OpenAI-compatible endpoint (Ollama,
//! LM Studio, llama.cpp server, vLLM, hosted gateways) instead of Warp's
//! GraphQL backend.
//!
//! # Status
//!
//! Stage 1 (this file): configuration plumbing only. Stage 2 will fill in
//! [`LocalDispatcher::stream`] with the actual HTTP call and translation
//! between OpenAI ChatCompletion SSE chunks and Warp's `ResponseEvent`
//! protobuf vocabulary.
//!
//! # Architecture sketch (for Stage 2)
//!
//! 1. Convert `warp_multi_agent_api::ConversationData` → OpenAI
//!    `ChatCompletionRequest` (system / user / assistant / tool messages,
//!    plus the OpenAI `tools` array derived from Warp's tool definitions).
//! 2. POST to `{endpoint}/chat/completions` with `stream: true`.
//! 3. Parse the SSE stream — for each `delta`, synthesize a
//!    `ResponseEvent` (text deltas, tool-call deltas, finish events).
//! 4. Emit terminal events (`end_of_turn`, errors) so the agent UI
//!    advances state machines correctly.
//!
//! The interception point in Stage 2 will be at the boundary where
//! `app/src/server/server_api/ai.rs` builds a `Request` and hands it to
//! `send_graphql_request` — we'll branch on
//! `ApiKeyManager::keys().has_local_endpoint()` (and/or model id) before
//! that call and route through this module instead.

use serde::{Deserialize, Serialize};

use crate::api_keys::ApiKeys;

/// Resolved configuration for talking to a user's local OpenAI-compatible
/// endpoint. Built from [`ApiKeys`] at request time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalEndpointConfig {
    /// Base URL — typically ends in `/v1`. Must not be empty.
    pub base_url: String,
    /// Model identifier sent in the `model` field of each request.
    pub model: String,
    /// Optional bearer token. Most local runtimes ignore auth; vLLM and
    /// hosted gateways may require it.
    pub api_key: Option<String>,
}

impl LocalEndpointConfig {
    /// Builds a config from the user's stored [`ApiKeys`], or returns
    /// `None` if either the endpoint URL or model name is missing.
    pub fn from_api_keys(keys: &ApiKeys) -> Option<Self> {
        let base_url = keys.local_endpoint.as_ref()?.trim().trim_end_matches('/');
        if base_url.is_empty() {
            return None;
        }
        let model = keys.local_model.as_ref()?.trim();
        if model.is_empty() {
            return None;
        }
        Some(Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            api_key: keys
                .local_api_key
                .as_ref()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
        })
    }

    /// Full URL to POST chat completions to.
    pub fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

/// Errors returned from the local dispatcher. Stage 2 will extend this with
/// HTTP / parse / streaming variants; this stub keeps just enough surface
/// area for the eventual integration sites to compile.
#[derive(Debug, thiserror::Error)]
pub enum LocalDispatcherError {
    #[error("local endpoint is not configured")]
    NotConfigured,
    #[error("local dispatcher is not yet implemented (Stage 2)")]
    NotImplemented,
}

/// OpenAI ChatCompletion request shape. Kept here (rather than pulling in a
/// crate dep) so we can serialize exactly what Ollama / LM Studio / vLLM
/// expect, including streamed responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiChatRequest {
    pub model: String,
    pub messages: Vec<OpenAiChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAiTool>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OpenAiFunctionDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiFunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OpenAiFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_config_requires_url_and_model() {
        let mut keys = ApiKeys::default();
        assert!(LocalEndpointConfig::from_api_keys(&keys).is_none());

        keys.local_endpoint = Some("http://localhost:11434/v1".into());
        assert!(LocalEndpointConfig::from_api_keys(&keys).is_none(),
            "endpoint without model should not yield a config");

        keys.local_model = Some("qwen2.5-coder:7b".into());
        let cfg = LocalEndpointConfig::from_api_keys(&keys).unwrap();
        assert_eq!(cfg.base_url, "http://localhost:11434/v1");
        assert_eq!(cfg.chat_completions_url(),
            "http://localhost:11434/v1/chat/completions");
        assert!(cfg.api_key.is_none());
    }

    #[test]
    fn endpoint_config_strips_trailing_slash() {
        let keys = ApiKeys {
            local_endpoint: Some("http://localhost:11434/v1/".into()),
            local_model: Some("qwen2.5-coder:7b".into()),
            ..Default::default()
        };
        let cfg = LocalEndpointConfig::from_api_keys(&keys).unwrap();
        assert_eq!(cfg.base_url, "http://localhost:11434/v1");
    }

    #[test]
    fn endpoint_config_treats_blank_strings_as_unset() {
        let keys = ApiKeys {
            local_endpoint: Some("   ".into()),
            local_model: Some("model".into()),
            ..Default::default()
        };
        assert!(LocalEndpointConfig::from_api_keys(&keys).is_none());
    }

    #[test]
    fn has_local_endpoint_matches_config_resolution() {
        let mut keys = ApiKeys::default();
        assert!(!keys.has_local_endpoint());
        keys.local_endpoint = Some("http://localhost:11434/v1".into());
        assert!(keys.has_local_endpoint());
        keys.local_endpoint = Some("   ".into());
        assert!(!keys.has_local_endpoint());
    }
}
