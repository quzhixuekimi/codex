use super::ApiTelemetry;
use super::AuthRequestTelemetryContext;
use super::ModelClient;
use super::PendingUnauthorizedRetry;
use super::RequestRouteTelemetry;
use super::UnauthorizedRecoveryExecution;
use super::emit_sentry_auth_failure_if_unauthorized;
use super::emit_terminal_auth_failure_after_failed_recovery;
use crate::auth_env_telemetry::AuthEnvTelemetry;
use crate::error::CodexErr;
use crate::response_debug_context::ResponseDebugContext;
use crate::util::FeedbackRequestTags;
use codex_api::ApiError;
use codex_api::RequestTelemetry;
use codex_api::WebsocketTelemetry;
use codex_otel::SessionTelemetry;
use codex_protocol::ThreadId;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;
use http::StatusCode;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use tracing::Event;
use tracing::Subscriber;
use tracing::field::Visit;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;

fn test_model_client(session_source: SessionSource) -> ModelClient {
    let provider = crate::model_provider_info::create_oss_provider_with_base_url(
        "https://example.com/v1",
        crate::model_provider_info::WireApi::Responses,
    );
    ModelClient::new(
        /*auth_manager*/ None,
        ThreadId::new(),
        provider,
        session_source,
        /*model_verbosity*/ None,
        /*enable_request_compression*/ false,
        /*include_timing_metrics*/ false,
        /*beta_features_header*/ None,
    )
}

fn test_model_info() -> ModelInfo {
    serde_json::from_value(json!({
        "slug": "gpt-test",
        "display_name": "gpt-test",
        "description": "desc",
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": [
            {"effort": "medium", "description": "medium"}
        ],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": true,
        "priority": 1,
        "upgrade": null,
        "base_instructions": "base instructions",
        "model_messages": null,
        "supports_reasoning_summaries": false,
        "support_verbosity": false,
        "default_verbosity": null,
        "apply_patch_tool_type": null,
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_parallel_tool_calls": false,
        "supports_image_detail_original": false,
        "context_window": 272000,
        "auto_compact_token_limit": null,
        "experimental_supported_tools": []
    }))
    .expect("deserialize test model info")
}

fn test_session_telemetry() -> SessionTelemetry {
    SessionTelemetry::new(
        ThreadId::new(),
        "gpt-test",
        "gpt-test",
        /*account_id*/ None,
        /*account_email*/ None,
        /*auth_mode*/ None,
        "test-originator".to_string(),
        /*log_user_prompts*/ false,
        "test-terminal".to_string(),
        SessionSource::Cli,
    )
}

#[derive(Default)]
struct TagCollectorVisitor {
    tags: BTreeMap<String, String>,
}

impl Visit for TagCollectorVisitor {
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.tags
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.tags
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.tags
            .insert(field.name().to_string(), format!("{value:?}"));
    }
}

#[derive(Clone)]
struct TagCollectorLayer {
    target: &'static str,
    tags: Arc<Mutex<BTreeMap<String, String>>>,
    event_count: Arc<Mutex<usize>>,
}

impl<S> Layer<S> for TagCollectorLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        if event.metadata().target() != self.target {
            return;
        }
        let mut visitor = TagCollectorVisitor::default();
        event.record(&mut visitor);
        self.tags.lock().unwrap().extend(visitor.tags);
        *self.event_count.lock().unwrap() += 1;
    }
}

fn empty_auth_env_telemetry() -> AuthEnvTelemetry {
    AuthEnvTelemetry {
        openai_api_key_env_present: false,
        codex_api_key_env_present: false,
        codex_api_key_env_enabled: false,
        provider_env_key_name: None,
        provider_env_key_present: None,
        refresh_token_url_override_present: false,
    }
}

#[test]
fn build_subagent_headers_sets_other_subagent_label() {
    let client = test_model_client(SessionSource::SubAgent(SubAgentSource::Other(
        "memory_consolidation".to_string(),
    )));
    let headers = client.build_subagent_headers();
    let value = headers
        .get("x-openai-subagent")
        .and_then(|value| value.to_str().ok());
    assert_eq!(value, Some("memory_consolidation"));
}

#[tokio::test]
async fn summarize_memories_returns_empty_for_empty_input() {
    let client = test_model_client(SessionSource::Cli);
    let model_info = test_model_info();
    let session_telemetry = test_session_telemetry();

    let output = client
        .summarize_memories(
            Vec::new(),
            &model_info,
            /*effort*/ None,
            &session_telemetry,
        )
        .await
        .expect("empty summarize request should succeed");
    assert_eq!(output.len(), 0);
}

#[test]
fn auth_request_telemetry_context_tracks_attached_auth_and_retry_phase() {
    let auth_context = AuthRequestTelemetryContext::new(
        Some(crate::auth::AuthMode::Chatgpt),
        &crate::api_bridge::CoreAuthProvider::for_test(Some("access-token"), Some("workspace-123")),
        /*has_followup_unauthorized_retry*/ true,
        PendingUnauthorizedRetry::from_recovery(UnauthorizedRecoveryExecution {
            mode: "managed",
            phase: "refresh_token",
        }),
    );

    assert_eq!(auth_context.auth_mode, Some("Chatgpt"));
    assert!(auth_context.auth_header_attached);
    assert_eq!(auth_context.auth_header_name, Some("authorization"));
    assert!(auth_context.retry_after_unauthorized);
    assert_eq!(auth_context.recovery_mode, Some("managed"));
    assert_eq!(auth_context.recovery_phase, Some("refresh_token"));
}

#[test]
fn api_telemetry_only_emits_sentry_auth_failure_after_unauthorized_retry() {
    let tags = Arc::new(Mutex::new(BTreeMap::new()));
    let event_count = Arc::new(Mutex::new(0));
    let _guard = tracing_subscriber::registry()
        .with(TagCollectorLayer {
            target: crate::util::SENTRY_AUTH_FAILURES_TARGET,
            tags: Arc::clone(&tags),
            event_count: Arc::clone(&event_count),
        })
        .set_default();

    let telemetry = ApiTelemetry::new(
        test_session_telemetry(),
        AuthRequestTelemetryContext {
            auth_mode: Some("Chatgpt"),
            auth_header_attached: true,
            auth_header_name: Some("authorization"),
            has_followup_unauthorized_retry: true,
            retry_after_unauthorized: false,
            recovery_mode: Some("managed"),
            recovery_phase: Some("refresh_token"),
        },
        RequestRouteTelemetry::for_endpoint("/responses"),
        empty_auth_env_telemetry(),
        true,
    );

    telemetry.on_request(1, Some(StatusCode::UNAUTHORIZED), None, Default::default());

    assert_eq!(*event_count.lock().unwrap(), 0);
    assert!(tags.lock().unwrap().is_empty());
}

#[test]
fn websocket_telemetry_only_emits_sentry_auth_failure_after_unauthorized_retry() {
    let tags = Arc::new(Mutex::new(BTreeMap::new()));
    let event_count = Arc::new(Mutex::new(0));
    let _guard = tracing_subscriber::registry()
        .with(TagCollectorLayer {
            target: crate::util::SENTRY_AUTH_FAILURES_TARGET,
            tags: Arc::clone(&tags),
            event_count: Arc::clone(&event_count),
        })
        .set_default();

    let telemetry = ApiTelemetry::new(
        test_session_telemetry(),
        AuthRequestTelemetryContext {
            auth_mode: Some("Chatgpt"),
            auth_header_attached: true,
            auth_header_name: Some("authorization"),
            has_followup_unauthorized_retry: true,
            retry_after_unauthorized: false,
            recovery_mode: Some("managed"),
            recovery_phase: Some("refresh_token"),
        },
        RequestRouteTelemetry::for_endpoint("/responses"),
        empty_auth_env_telemetry(),
        true,
    );

    telemetry.on_ws_request(
        Default::default(),
        Some(&ApiError::Api {
            status: StatusCode::UNAUTHORIZED,
            message: String::new(),
        }),
        false,
    );

    assert_eq!(*event_count.lock().unwrap(), 0);
    assert!(tags.lock().unwrap().is_empty());
}

#[test]
fn api_telemetry_skips_sentry_auth_failure_for_non_openai_provider() {
    let tags = Arc::new(Mutex::new(BTreeMap::new()));
    let event_count = Arc::new(Mutex::new(0));
    let _guard = tracing_subscriber::registry()
        .with(TagCollectorLayer {
            target: crate::util::SENTRY_AUTH_FAILURES_TARGET,
            tags: Arc::clone(&tags),
            event_count: Arc::clone(&event_count),
        })
        .set_default();

    let telemetry = ApiTelemetry::new(
        test_session_telemetry(),
        AuthRequestTelemetryContext {
            auth_mode: Some("Chatgpt"),
            auth_header_attached: true,
            auth_header_name: Some("authorization"),
            has_followup_unauthorized_retry: true,
            retry_after_unauthorized: true,
            recovery_mode: Some("managed"),
            recovery_phase: Some("refresh_token"),
        },
        RequestRouteTelemetry::for_endpoint("/responses"),
        empty_auth_env_telemetry(),
        false,
    );

    telemetry.on_request(1, Some(StatusCode::UNAUTHORIZED), None, Default::default());

    assert_eq!(*event_count.lock().unwrap(), 0);
    assert!(tags.lock().unwrap().is_empty());
}

#[test]
fn websocket_telemetry_skips_sentry_auth_failure_for_non_openai_provider() {
    let tags = Arc::new(Mutex::new(BTreeMap::new()));
    let event_count = Arc::new(Mutex::new(0));
    let _guard = tracing_subscriber::registry()
        .with(TagCollectorLayer {
            target: crate::util::SENTRY_AUTH_FAILURES_TARGET,
            tags: Arc::clone(&tags),
            event_count: Arc::clone(&event_count),
        })
        .set_default();

    let telemetry = ApiTelemetry::new(
        test_session_telemetry(),
        AuthRequestTelemetryContext {
            auth_mode: Some("Chatgpt"),
            auth_header_attached: true,
            auth_header_name: Some("authorization"),
            has_followup_unauthorized_retry: true,
            retry_after_unauthorized: true,
            recovery_mode: Some("managed"),
            recovery_phase: Some("refresh_token"),
        },
        RequestRouteTelemetry::for_endpoint("/responses"),
        empty_auth_env_telemetry(),
        false,
    );

    telemetry.on_ws_request(
        Default::default(),
        Some(&ApiError::Api {
            status: StatusCode::UNAUTHORIZED,
            message: String::new(),
        }),
        false,
    );

    assert_eq!(*event_count.lock().unwrap(), 0);
    assert!(tags.lock().unwrap().is_empty());
}

#[test]
fn websocket_handshake_failure_emits_sentry_auth_failure_when_no_followup_retry_remains() {
    let tags = Arc::new(Mutex::new(BTreeMap::new()));
    let event_count = Arc::new(Mutex::new(0));
    let _guard = tracing_subscriber::registry()
        .with(TagCollectorLayer {
            target: crate::util::SENTRY_AUTH_FAILURES_TARGET,
            tags: Arc::clone(&tags),
            event_count: Arc::clone(&event_count),
        })
        .set_default();

    let feedback_tags = FeedbackRequestTags {
        endpoint: "/responses",
        auth_header_attached: true,
        auth_header_name: Some("authorization"),
        auth_mode: Some("Chatgpt"),
        auth_retry_after_unauthorized: Some(true),
        auth_recovery_mode: Some("managed"),
        auth_recovery_phase: Some("refresh_token"),
        auth_connection_reused: Some(false),
        auth_request_id: Some("req_123"),
        auth_cf_ray: Some("ray_123"),
        auth_error: Some(""),
        auth_error_code: Some("invalid_api_key"),
        auth_recovery_followup_success: Some(false),
        auth_recovery_followup_status: Some(StatusCode::UNAUTHORIZED.as_u16()),
    };

    emit_sentry_auth_failure_if_unauthorized(
        &feedback_tags,
        &empty_auth_env_telemetry(),
        true,
        Some(StatusCode::UNAUTHORIZED.as_u16()),
        false,
    );

    assert_eq!(*event_count.lock().unwrap(), 1);
    let tags = tags.lock().unwrap();
    assert_eq!(tags.get("endpoint"), Some(&"/responses".to_string()));
    assert_eq!(
        tags.get("auth_retry_after_unauthorized"),
        Some(&"true".to_string())
    );
    assert_eq!(tags.get("auth_request_id"), Some(&"req_123".to_string()));
}

#[test]
fn unauthorized_retry_with_followup_recovery_does_not_emit_sentry_auth_failure() {
    let tags = Arc::new(Mutex::new(BTreeMap::new()));
    let event_count = Arc::new(Mutex::new(0));
    let _guard = tracing_subscriber::registry()
        .with(TagCollectorLayer {
            target: crate::util::SENTRY_AUTH_FAILURES_TARGET,
            tags: Arc::clone(&tags),
            event_count: Arc::clone(&event_count),
        })
        .set_default();

    let feedback_tags = FeedbackRequestTags {
        endpoint: "/responses",
        auth_header_attached: true,
        auth_header_name: Some("authorization"),
        auth_mode: Some("Chatgpt"),
        auth_retry_after_unauthorized: Some(true),
        auth_recovery_mode: Some("managed"),
        auth_recovery_phase: Some("reload"),
        auth_connection_reused: Some(false),
        auth_request_id: Some("req_reload"),
        auth_cf_ray: Some("ray_reload"),
        auth_error: Some(""),
        auth_error_code: Some("token_expired"),
        auth_recovery_followup_success: Some(false),
        auth_recovery_followup_status: Some(StatusCode::UNAUTHORIZED.as_u16()),
    };

    emit_sentry_auth_failure_if_unauthorized(
        &feedback_tags,
        &empty_auth_env_telemetry(),
        true,
        Some(StatusCode::UNAUTHORIZED.as_u16()),
        true,
    );

    assert_eq!(*event_count.lock().unwrap(), 0);
    assert!(tags.lock().unwrap().is_empty());
}

#[test]
fn terminal_unauthorized_without_retry_path_emits_sentry_auth_failure() {
    let tags = Arc::new(Mutex::new(BTreeMap::new()));
    let event_count = Arc::new(Mutex::new(0));
    let _guard = tracing_subscriber::registry()
        .with(TagCollectorLayer {
            target: crate::util::SENTRY_AUTH_FAILURES_TARGET,
            tags: Arc::clone(&tags),
            event_count: Arc::clone(&event_count),
        })
        .set_default();

    let feedback_tags = FeedbackRequestTags {
        endpoint: "/responses/compact",
        auth_header_attached: true,
        auth_header_name: Some("authorization"),
        auth_mode: Some("Chatgpt"),
        auth_retry_after_unauthorized: Some(false),
        auth_recovery_mode: None,
        auth_recovery_phase: None,
        auth_connection_reused: None,
        auth_request_id: Some("req_terminal"),
        auth_cf_ray: Some("ray_terminal"),
        auth_error: Some(""),
        auth_error_code: Some("invalid_api_key"),
        auth_recovery_followup_success: None,
        auth_recovery_followup_status: None,
    };

    emit_sentry_auth_failure_if_unauthorized(
        &feedback_tags,
        &empty_auth_env_telemetry(),
        true,
        Some(StatusCode::UNAUTHORIZED.as_u16()),
        false,
    );

    assert_eq!(*event_count.lock().unwrap(), 1);
    let tags = tags.lock().unwrap();
    assert_eq!(
        tags.get("endpoint"),
        Some(&"/responses/compact".to_string())
    );
    assert_eq!(
        tags.get("auth_retry_after_unauthorized"),
        Some(&"false".to_string())
    );
    assert_eq!(
        tags.get("auth_request_id"),
        Some(&"req_terminal".to_string())
    );
}

#[test]
fn failed_recovery_before_retry_emits_terminal_sentry_auth_failure() {
    let tags = Arc::new(Mutex::new(BTreeMap::new()));
    let event_count = Arc::new(Mutex::new(0));
    let _guard = tracing_subscriber::registry()
        .with(TagCollectorLayer {
            target: crate::util::SENTRY_AUTH_FAILURES_TARGET,
            tags: Arc::clone(&tags),
            event_count: Arc::clone(&event_count),
        })
        .set_default();

    let auth_context = AuthRequestTelemetryContext {
        auth_mode: Some("Chatgpt"),
        auth_header_attached: true,
        auth_header_name: Some("authorization"),
        has_followup_unauthorized_retry: true,
        retry_after_unauthorized: false,
        recovery_mode: Some("managed"),
        recovery_phase: Some("reload"),
    };
    let debug = ResponseDebugContext {
        request_id: Some("req_failed_recovery".to_string()),
        cf_ray: Some("ray_failed_recovery".to_string()),
        auth_error: None,
        auth_error_code: Some("token_expired".to_string()),
    };

    emit_terminal_auth_failure_after_failed_recovery(
        auth_context,
        RequestRouteTelemetry::for_endpoint("/responses"),
        &empty_auth_env_telemetry(),
        &debug,
        &CodexErr::Io(std::io::Error::other("transient refresh failure")),
        true,
    );

    assert_eq!(*event_count.lock().unwrap(), 1);
    let tags = tags.lock().unwrap();
    assert_eq!(tags.get("endpoint"), Some(&"/responses".to_string()));
    assert_eq!(tags.get("auth_recovery_mode"), Some(&"managed".to_string()));
    assert_eq!(tags.get("auth_recovery_phase"), Some(&"reload".to_string()));
    assert_eq!(
        tags.get("auth_retry_after_unauthorized"),
        Some(&"false".to_string())
    );
    assert_eq!(
        tags.get("auth_request_id"),
        Some(&"req_failed_recovery".to_string())
    );
}

#[test]
fn refresh_token_failed_recovery_does_not_emit_terminal_sentry_auth_failure() {
    let tags = Arc::new(Mutex::new(BTreeMap::new()));
    let event_count = Arc::new(Mutex::new(0));
    let _guard = tracing_subscriber::registry()
        .with(TagCollectorLayer {
            target: crate::util::SENTRY_AUTH_FAILURES_TARGET,
            tags: Arc::clone(&tags),
            event_count: Arc::clone(&event_count),
        })
        .set_default();

    let auth_context = AuthRequestTelemetryContext {
        auth_mode: Some("Chatgpt"),
        auth_header_attached: true,
        auth_header_name: Some("authorization"),
        has_followup_unauthorized_retry: true,
        retry_after_unauthorized: false,
        recovery_mode: Some("managed"),
        recovery_phase: Some("refresh_token"),
    };
    let debug = ResponseDebugContext {
        request_id: Some("req_refresh_failed".to_string()),
        cf_ray: Some("ray_refresh_failed".to_string()),
        auth_error: None,
        auth_error_code: Some("refresh_token_reused".to_string()),
    };

    emit_terminal_auth_failure_after_failed_recovery(
        auth_context,
        RequestRouteTelemetry::for_endpoint("/responses"),
        &empty_auth_env_telemetry(),
        &debug,
        &CodexErr::RefreshTokenFailed(codex_login::auth::RefreshTokenFailedError::new(
            codex_login::auth::RefreshTokenFailedReason::Revoked,
            "refresh token reused",
        )),
        true,
    );

    assert_eq!(*event_count.lock().unwrap(), 0);
    assert!(tags.lock().unwrap().is_empty());
}

#[test]
fn transient_refresh_token_failed_recovery_emits_terminal_sentry_auth_failure() {
    let tags = Arc::new(Mutex::new(BTreeMap::new()));
    let event_count = Arc::new(Mutex::new(0));
    let _guard = tracing_subscriber::registry()
        .with(TagCollectorLayer {
            target: crate::util::SENTRY_AUTH_FAILURES_TARGET,
            tags: Arc::clone(&tags),
            event_count: Arc::clone(&event_count),
        })
        .set_default();

    let auth_context = AuthRequestTelemetryContext {
        auth_mode: Some("Chatgpt"),
        auth_header_attached: true,
        auth_header_name: Some("authorization"),
        has_followup_unauthorized_retry: true,
        retry_after_unauthorized: false,
        recovery_mode: Some("managed"),
        recovery_phase: Some("refresh_token"),
    };
    let debug = ResponseDebugContext {
        request_id: Some("req_refresh_transient".to_string()),
        cf_ray: Some("ray_refresh_transient".to_string()),
        auth_error: None,
        auth_error_code: Some("timeout".to_string()),
    };

    emit_terminal_auth_failure_after_failed_recovery(
        auth_context,
        RequestRouteTelemetry::for_endpoint("/responses"),
        &empty_auth_env_telemetry(),
        &debug,
        &CodexErr::Io(std::io::Error::other("refresh timeout")),
        true,
    );

    assert_eq!(*event_count.lock().unwrap(), 1);
    let tags = tags.lock().unwrap();
    assert_eq!(tags.get("endpoint"), Some(&"/responses".to_string()));
    assert_eq!(
        tags.get("auth_recovery_phase"),
        Some(&"refresh_token".to_string())
    );
    assert_eq!(
        tags.get("auth_request_id"),
        Some(&"req_refresh_transient".to_string())
    );
}
