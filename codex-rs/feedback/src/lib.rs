use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::collections::btree_map::Entry;
use std::fs;
use std::io::Write;
use std::io::{self};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::Result;
use anyhow::anyhow;
use codex_protocol::ThreadId;
use codex_protocol::protocol::SessionSource;
use feedback_diagnostics::FEEDBACK_DIAGNOSTICS_ATTACHMENT_FILENAME;
use feedback_diagnostics::FeedbackDiagnostics;
use tracing::Event;
use tracing::Level;
use tracing::field::Visit;
use tracing_subscriber::Layer;
use tracing_subscriber::filter;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::fmt::writer::MakeWriter;
use tracing_subscriber::layer::Filter;
use tracing_subscriber::registry::LookupSpan;

pub mod feedback_diagnostics;

const DEFAULT_MAX_BYTES: usize = 4 * 1024 * 1024; // 4 MiB
const SENTRY_DSN: &str =
    "https://ae32ed50620d7a7792c1ce5df38b3e3e@o33249.ingest.us.sentry.io/4510195390611458";
const SENTRY_DSN_OVERRIDE_ENV_VAR: &str = "CODEX_SENTRY_DSN_OVERRIDE";
const UPLOAD_TIMEOUT_SECS: u64 = 10;
const FEEDBACK_TAGS_TARGET: &str = "feedback_tags";
const SENTRY_AUTH_FAILURES_TARGET: &str = "sentry_auth_failures";
const AUTH_FAILURE_REPORT_KIND: &str = "auth_failure_auto";
const MAX_FEEDBACK_TAGS: usize = 64;
const AUTH_FAILURE_UPLOAD_QUEUE_CAPACITY: usize = 32;
const AUTH_FAILURE_QUEUE_FLUSH_BUFFER_SECS: u64 = 5;
const AUTH_FAILURE_QUEUE_MAX_FLUSHED_UPLOADS: u64 = 3;
const FALLBACK_AUTH_FAILURE_UPLOAD_QUEUE_CAPACITY: usize = 32;

struct AuthFailureUploadTask {
    tags: BTreeMap<String, String>,
    dsn_override: Option<String>,
}

enum AuthFailureUploadCommand {
    Upload(AuthFailureUploadTask),
    Flush(std::sync::mpsc::Sender<()>),
}

static AUTH_FAILURE_UPLOAD_QUEUE: OnceLock<std::sync::mpsc::SyncSender<AuthFailureUploadCommand>> =
    OnceLock::new();
static FALLBACK_AUTH_FAILURE_UPLOAD_QUEUE: OnceLock<
    std::sync::mpsc::SyncSender<AuthFailureUploadCommand>,
> = OnceLock::new();

#[derive(Clone)]
pub struct CodexFeedback {
    inner: Arc<FeedbackInner>,
}

impl Default for CodexFeedback {
    fn default() -> Self {
        Self::new()
    }
}

impl CodexFeedback {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_BYTES)
    }

    pub(crate) fn with_capacity(max_bytes: usize) -> Self {
        Self {
            inner: Arc::new(FeedbackInner::new(max_bytes)),
        }
    }

    pub fn make_writer(&self) -> FeedbackMakeWriter {
        FeedbackMakeWriter {
            inner: self.inner.clone(),
        }
    }

    /// Returns a [`tracing_subscriber`] layer that captures full-fidelity logs into this feedback
    /// ring buffer.
    ///
    /// This is intended for initialization code so call sites don't have to duplicate the exact
    /// `fmt::layer()` configuration and filter logic.
    pub fn logger_layer<S>(&self) -> impl Layer<S> + Send + Sync + 'static
    where
        S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    {
        tracing_subscriber::fmt::layer()
            .with_writer(self.make_writer())
            .with_timer(tracing_subscriber::fmt::time::SystemTime)
            .with_ansi(false)
            .with_target(false)
            // Capture everything, regardless of the caller's `RUST_LOG`, so feedback includes the
            // full trace when the user uploads a report.
            .with_filter(Targets::new().with_default(Level::TRACE))
    }

    /// Returns a [`tracing_subscriber`] layer that collects structured metadata for feedback.
    ///
    /// Events with `target: "feedback_tags"` are treated as key/value tags to attach to feedback
    /// uploads later.
    pub fn metadata_layer<S>(&self) -> impl Layer<S> + Send + Sync + 'static
    where
        S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    {
        FeedbackMetadataLayer {
            inner: self.inner.clone(),
        }
        .with_filter(Targets::new().with_target(FEEDBACK_TAGS_TARGET, Level::TRACE))
    }

    /// Returns a [`tracing_subscriber`] layer that uploads lightweight auth-failure Sentry events.
    ///
    /// Events with `target: "sentry_auth_failures"` are converted into searchable Sentry issues
    /// without relying on a manual `/feedback` upload.
    pub fn auth_event_layer<S>(&self) -> impl Layer<S> + Send + Sync + 'static
    where
        S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    {
        AuthFailureSentryLayer
            .with_filter(Targets::new().with_target(SENTRY_AUTH_FAILURES_TARGET, Level::TRACE))
    }

    pub fn snapshot(&self, session_id: Option<ThreadId>) -> FeedbackSnapshot {
        let bytes = {
            let guard = self.inner.ring.lock().expect("mutex poisoned");
            guard.snapshot_bytes()
        };
        let tags = {
            let guard = self.inner.tags.lock().expect("mutex poisoned");
            guard.clone()
        };
        FeedbackSnapshot {
            bytes,
            tags,
            feedback_diagnostics: FeedbackDiagnostics::collect_from_env(),
            thread_id: session_id
                .map(|id| id.to_string())
                .unwrap_or("no-active-thread-".to_string() + &ThreadId::new().to_string()),
        }
    }
}

pub fn exclude_auth_failure_events<S>() -> impl Filter<S> + Clone {
    filter::filter_fn(|metadata| metadata.target() != SENTRY_AUTH_FAILURES_TARGET)
}

struct FeedbackInner {
    ring: Mutex<RingBuffer>,
    tags: Mutex<BTreeMap<String, String>>,
}

impl FeedbackInner {
    fn new(max_bytes: usize) -> Self {
        Self {
            ring: Mutex::new(RingBuffer::new(max_bytes)),
            tags: Mutex::new(BTreeMap::new()),
        }
    }
}

#[derive(Clone)]
pub struct FeedbackMakeWriter {
    inner: Arc<FeedbackInner>,
}

impl<'a> MakeWriter<'a> for FeedbackMakeWriter {
    type Writer = FeedbackWriter;

    fn make_writer(&'a self) -> Self::Writer {
        FeedbackWriter {
            inner: self.inner.clone(),
        }
    }
}

pub struct FeedbackWriter {
    inner: Arc<FeedbackInner>,
}

impl Write for FeedbackWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut guard = self.inner.ring.lock().map_err(|_| io::ErrorKind::Other)?;
        guard.push_bytes(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct RingBuffer {
    max: usize,
    buf: VecDeque<u8>,
}

impl RingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            max: capacity,
            buf: VecDeque::with_capacity(capacity),
        }
    }

    fn len(&self) -> usize {
        self.buf.len()
    }

    fn push_bytes(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }

        // If the incoming chunk is larger than capacity, keep only the trailing bytes.
        if data.len() >= self.max {
            self.buf.clear();
            let start = data.len() - self.max;
            self.buf.extend(data[start..].iter().copied());
            return;
        }

        // Evict from the front if we would exceed capacity.
        let needed = self.len() + data.len();
        if needed > self.max {
            let to_drop = needed - self.max;
            for _ in 0..to_drop {
                let _ = self.buf.pop_front();
            }
        }

        self.buf.extend(data.iter().copied());
    }

    fn snapshot_bytes(&self) -> Vec<u8> {
        self.buf.iter().copied().collect()
    }
}

pub struct FeedbackSnapshot {
    bytes: Vec<u8>,
    tags: BTreeMap<String, String>,
    feedback_diagnostics: FeedbackDiagnostics,
    pub thread_id: String,
}

impl FeedbackSnapshot {
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn feedback_diagnostics(&self) -> &FeedbackDiagnostics {
        &self.feedback_diagnostics
    }

    pub fn with_feedback_diagnostics(mut self, feedback_diagnostics: FeedbackDiagnostics) -> Self {
        self.feedback_diagnostics = feedback_diagnostics;
        self
    }

    pub fn feedback_diagnostics_attachment_text(&self, include_logs: bool) -> Option<String> {
        if !include_logs {
            return None;
        }

        self.feedback_diagnostics.attachment_text()
    }

    pub fn save_to_temp_file(&self) -> io::Result<PathBuf> {
        let dir = std::env::temp_dir();
        let filename = format!("codex-feedback-{}.log", self.thread_id);
        let path = dir.join(filename);
        fs::write(&path, self.as_bytes())?;
        Ok(path)
    }

    /// Upload feedback to Sentry with optional attachments.
    pub fn upload_feedback(
        &self,
        classification: &str,
        reason: Option<&str>,
        include_logs: bool,
        extra_attachment_paths: &[PathBuf],
        session_source: Option<SessionSource>,
        logs_override: Option<Vec<u8>>,
    ) -> Result<()> {
        use sentry::protocol::Envelope;
        use sentry::protocol::EnvelopeItem;
        use sentry::protocol::Event;
        use sentry::protocol::Level;
        use std::collections::BTreeMap;

        let client = build_sentry_client(/*dsn_override*/ None)?;

        let cli_version = env!("CARGO_PKG_VERSION");
        let mut tags = BTreeMap::from([
            (String::from("thread_id"), self.thread_id.to_string()),
            (String::from("classification"), classification.to_string()),
            (String::from("cli_version"), cli_version.to_string()),
        ]);
        if let Some(source) = session_source.as_ref() {
            tags.insert(String::from("session_source"), source.to_string());
        }
        if let Some(r) = reason {
            tags.insert(String::from("reason"), r.to_string());
        }

        let reserved = [
            "thread_id",
            "classification",
            "cli_version",
            "session_source",
            "reason",
        ];
        for (key, value) in &self.tags {
            if reserved.contains(&key.as_str()) {
                continue;
            }
            if let Entry::Vacant(entry) = tags.entry(key.clone()) {
                entry.insert(value.clone());
            }
        }

        let level = match classification {
            "bug" | "bad_result" | "safety_check" => Level::Error,
            _ => Level::Info,
        };

        let mut envelope = Envelope::new();
        let title = format!(
            "[{}]: Codex session {}",
            display_classification(classification),
            self.thread_id
        );

        let mut event = Event {
            level,
            message: Some(title.clone()),
            tags,
            ..Default::default()
        };
        if let Some(r) = reason {
            use sentry::protocol::Exception;
            use sentry::protocol::Values;

            event.exception = Values::from(vec![Exception {
                ty: title.clone(),
                value: Some(r.to_string()),
                ..Default::default()
            }]);
        }
        envelope.add_item(EnvelopeItem::Event(event));

        for attachment in
            self.feedback_attachments(include_logs, extra_attachment_paths, logs_override)
        {
            envelope.add_item(EnvelopeItem::Attachment(attachment));
        }

        client.send_envelope(envelope);
        client.flush(Some(Duration::from_secs(UPLOAD_TIMEOUT_SECS)));
        Ok(())
    }

    fn feedback_attachments(
        &self,
        include_logs: bool,
        extra_attachment_paths: &[PathBuf],
        logs_override: Option<Vec<u8>>,
    ) -> Vec<sentry::protocol::Attachment> {
        use sentry::protocol::Attachment;

        let mut attachments = Vec::new();

        if include_logs {
            attachments.push(Attachment {
                buffer: logs_override.unwrap_or_else(|| self.bytes.clone()),
                filename: String::from("codex-logs.log"),
                content_type: Some("text/plain".to_string()),
                ty: None,
            });
        }

        if let Some(text) = self.feedback_diagnostics_attachment_text(include_logs) {
            attachments.push(Attachment {
                buffer: text.into_bytes(),
                filename: FEEDBACK_DIAGNOSTICS_ATTACHMENT_FILENAME.to_string(),
                content_type: Some("text/plain".to_string()),
                ty: None,
            });
        }

        for path in extra_attachment_paths {
            let data = match fs::read(path) {
                Ok(data) => data,
                Err(err) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %err,
                        "failed to read log attachment; skipping"
                    );
                    continue;
                }
            };
            let filename = path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "extra-log.log".to_string());
            attachments.push(Attachment {
                buffer: data,
                filename,
                content_type: Some("text/plain".to_string()),
                ty: None,
            });
        }

        attachments
    }
}

fn build_sentry_client(dsn_override: Option<&str>) -> Result<sentry::Client> {
    build_sentry_client_with_dsn_override(dsn_override)
}

fn build_sentry_client_with_dsn_override(dsn_override: Option<&str>) -> Result<sentry::Client> {
    use std::str::FromStr;
    use std::sync::Arc;

    use sentry::Client;
    use sentry::ClientOptions;
    use sentry::transports::DefaultTransportFactory;
    use sentry::types::Dsn;

    let dsn = dsn_override.unwrap_or(SENTRY_DSN);
    Ok(Client::from_config(ClientOptions {
        dsn: Some(Dsn::from_str(dsn).map_err(|e| anyhow!("invalid DSN: {e}"))?),
        transport: Some(Arc::new(DefaultTransportFactory {})),
        ..Default::default()
    }))
}

fn display_classification(classification: &str) -> String {
    match classification {
        "bug" => "Bug".to_string(),
        "bad_result" => "Bad result".to_string(),
        "good_result" => "Good result".to_string(),
        "safety_check" => "Safety check".to_string(),
        _ => "Other".to_string(),
    }
}

#[derive(Clone)]
struct FeedbackMetadataLayer {
    inner: Arc<FeedbackInner>,
}

impl<S> Layer<S> for FeedbackMetadataLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        // This layer is filtered by `Targets`, but keep the guard anyway in case it is used without
        // the filter.
        if event.metadata().target() != FEEDBACK_TAGS_TARGET {
            return;
        }

        let mut visitor = FeedbackTagsVisitor::default();
        event.record(&mut visitor);
        if visitor.tags.is_empty() {
            return;
        }

        let mut guard = self.inner.tags.lock().expect("mutex poisoned");
        for (key, value) in visitor.tags {
            if guard.len() >= MAX_FEEDBACK_TAGS && !guard.contains_key(&key) {
                continue;
            }
            guard.insert(key, value);
        }
    }
}

#[derive(Clone, Copy)]
struct AuthFailureSentryLayer;

impl<S> Layer<S> for AuthFailureSentryLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        if event.metadata().target() != SENTRY_AUTH_FAILURES_TARGET {
            return;
        }

        let mut visitor = FeedbackTagsVisitor::default();
        event.record(&mut visitor);
        if visitor.tags.is_empty() {
            return;
        }

        if let Err(err) = enqueue_fallback_auth_failure_event_tags(visitor.tags) {
            tracing::warn!(error = %err, "failed to enqueue fallback auth failure event");
        }
    }
}

#[derive(Default)]
struct FeedbackTagsVisitor {
    tags: BTreeMap<String, String>,
}

impl Visit for FeedbackTagsVisitor {
    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.tags
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.tags
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.tags
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
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

fn finalize_auth_failure_tags(mut tags: BTreeMap<String, String>) -> BTreeMap<String, String> {
    tags.retain(|_, value| !value.is_empty());
    tags.insert(
        String::from("report_kind"),
        AUTH_FAILURE_REPORT_KIND.to_string(),
    );
    tags
}

fn auth_failure_grouping_key(tags: &BTreeMap<String, String>) -> Vec<String> {
    let endpoint = tags
        .get("endpoint")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let error_code = tags
        .get("auth_error_code")
        .filter(|value| !value.is_empty())
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let auth_header_attached = tags
        .get("auth_header_attached")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    vec![
        "codex".to_string(),
        AUTH_FAILURE_REPORT_KIND.to_string(),
        endpoint,
        error_code,
        auth_header_attached,
    ]
}

fn build_auth_failure_event(tags: BTreeMap<String, String>) -> sentry::protocol::Event<'static> {
    use std::borrow::Cow;

    use sentry::protocol::Event;
    use sentry::protocol::Level;

    let endpoint = tags
        .get("endpoint")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let fingerprint = auth_failure_grouping_key(&tags)
        .into_iter()
        .map(Cow::Owned)
        .collect::<Vec<Cow<'static, str>>>();
    Event {
        level: Level::Error,
        message: Some(format!("Codex client auth failure on {endpoint}")),
        fingerprint: fingerprint.into(),
        tags,
        ..Default::default()
    }
}

pub fn upload_auth_failure_event_tags(tags: BTreeMap<String, String>) -> Result<()> {
    upload_auth_failure_event_with_dsn_override(
        finalize_auth_failure_tags(tags),
        std::env::var(SENTRY_DSN_OVERRIDE_ENV_VAR).ok().as_deref(),
    )
}

pub fn enqueue_auth_failure_event_tags(tags: BTreeMap<String, String>) -> bool {
    let dsn_override = std::env::var(SENTRY_DSN_OVERRIDE_ENV_VAR).ok();
    if enqueue_auth_failure_event_with_dsn_override(tags.clone(), dsn_override.clone()) {
        true
    } else {
        enqueue_fallback_auth_failure_event_with_dsn_override(tags, dsn_override).is_ok()
    }
}

pub fn flush_auth_failure_event_queue(timeout: Duration) {
    let deadline = std::time::Instant::now() + timeout;
    flush_auth_failure_upload_queue(deadline);
    flush_fallback_auth_failure_upload_queue(deadline);
}

pub fn auth_failure_event_queue_flush_timeout() -> Duration {
    Duration::from_secs(
        UPLOAD_TIMEOUT_SECS * AUTH_FAILURE_QUEUE_MAX_FLUSHED_UPLOADS
            + AUTH_FAILURE_QUEUE_FLUSH_BUFFER_SECS,
    )
}

pub fn flush_auth_failure_event_queue_and_exit(timeout: Duration, code: i32) -> ! {
    flush_auth_failure_event_queue(timeout);
    std::process::exit(code);
}

pub struct AuthFailureEventQueueFlushGuard {
    timeout: Duration,
}

impl AuthFailureEventQueueFlushGuard {
    pub fn new(timeout: Duration) -> Self {
        Self { timeout }
    }
}

impl Drop for AuthFailureEventQueueFlushGuard {
    fn drop(&mut self) {
        flush_auth_failure_event_queue(self.timeout);
    }
}

fn enqueue_auth_failure_event_with_dsn_override(
    tags: BTreeMap<String, String>,
    dsn_override: Option<String>,
) -> bool {
    let tags = finalize_auth_failure_tags(tags);
    let sender = auth_failure_upload_queue_sender();
    match sender.try_send(AuthFailureUploadCommand::Upload(AuthFailureUploadTask {
        tags,
        dsn_override,
    })) {
        Ok(()) => true,
        Err(err) => {
            tracing::warn!(error = %err, "failed to enqueue auth failure event");
            false
        }
    }
}

fn auth_failure_upload_queue_sender()
-> &'static std::sync::mpsc::SyncSender<AuthFailureUploadCommand> {
    AUTH_FAILURE_UPLOAD_QUEUE.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<AuthFailureUploadCommand>(
            AUTH_FAILURE_UPLOAD_QUEUE_CAPACITY,
        );
        std::thread::spawn(move || {
            while let Ok(command) = rx.recv() {
                match command {
                    AuthFailureUploadCommand::Upload(task) => {
                        if let Err(err) = upload_auth_failure_event_with_dsn_override(
                            task.tags,
                            task.dsn_override.as_deref(),
                        ) {
                            tracing::warn!(error = %err, "failed to upload auth failure event");
                        }
                    }
                    AuthFailureUploadCommand::Flush(done) => {
                        let _ = done.send(());
                    }
                }
            }
        });
        tx
    })
}

fn fallback_auth_failure_upload_queue_sender()
-> &'static std::sync::mpsc::SyncSender<AuthFailureUploadCommand> {
    FALLBACK_AUTH_FAILURE_UPLOAD_QUEUE.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<AuthFailureUploadCommand>(
            FALLBACK_AUTH_FAILURE_UPLOAD_QUEUE_CAPACITY,
        );
        std::thread::spawn(move || {
            while let Ok(command) = rx.recv() {
                match command {
                    AuthFailureUploadCommand::Upload(task) => {
                        if let Err(err) = upload_auth_failure_event_with_dsn_override(
                            task.tags,
                            task.dsn_override.as_deref(),
                        ) {
                            tracing::warn!(
                                error = %err,
                                "failed to upload fallback auth failure event"
                            );
                        }
                    }
                    AuthFailureUploadCommand::Flush(done) => {
                        let _ = done.send(());
                    }
                }
            }
        });
        tx
    })
}

fn enqueue_fallback_auth_failure_event_tags(tags: BTreeMap<String, String>) -> Result<()> {
    enqueue_fallback_auth_failure_event_with_dsn_override(
        tags,
        std::env::var(SENTRY_DSN_OVERRIDE_ENV_VAR).ok(),
    )
}

fn enqueue_fallback_auth_failure_event_with_dsn_override(
    tags: BTreeMap<String, String>,
    dsn_override: Option<String>,
) -> Result<()> {
    let tags = finalize_auth_failure_tags(tags);
    let sender = fallback_auth_failure_upload_queue_sender();
    if let Err(err) = sender.try_send(AuthFailureUploadCommand::Upload(AuthFailureUploadTask {
        tags,
        dsn_override,
    })) {
        return Err(anyhow!("failed to send fallback auth failure event: {err}"));
    }
    Ok(())
}

fn flush_auth_failure_upload_queue(deadline: std::time::Instant) {
    let (tx, rx) = std::sync::mpsc::channel();
    if let Err(err) =
        send_auth_failure_flush_command(auth_failure_upload_queue_sender(), tx, deadline)
    {
        tracing::warn!(error = %err, "failed to request auth failure queue flush");
        return;
    }
    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
    if let Err(err) = rx.recv_timeout(remaining) {
        tracing::warn!(error = %err, "timed out flushing auth failure queue");
    }
}

fn flush_fallback_auth_failure_upload_queue(deadline: std::time::Instant) {
    let (tx, rx) = std::sync::mpsc::channel();
    if let Err(err) =
        send_auth_failure_flush_command(fallback_auth_failure_upload_queue_sender(), tx, deadline)
    {
        tracing::warn!(error = %err, "failed to request fallback auth failure queue flush");
        return;
    }
    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
    if let Err(err) = rx.recv_timeout(remaining) {
        tracing::warn!(error = %err, "timed out flushing fallback auth failure queue");
    }
}

fn send_auth_failure_flush_command(
    sender: &std::sync::mpsc::SyncSender<AuthFailureUploadCommand>,
    done: std::sync::mpsc::Sender<()>,
    deadline: std::time::Instant,
) -> std::result::Result<(), std::sync::mpsc::TrySendError<AuthFailureUploadCommand>> {
    let mut command = AuthFailureUploadCommand::Flush(done);
    loop {
        match sender.try_send(command) {
            Ok(()) => return Ok(()),
            Err(std::sync::mpsc::TrySendError::Disconnected(command)) => {
                return Err(std::sync::mpsc::TrySendError::Disconnected(command));
            }
            Err(std::sync::mpsc::TrySendError::Full(returned_command)) => {
                if std::time::Instant::now() >= deadline {
                    return Err(std::sync::mpsc::TrySendError::Full(returned_command));
                }
                command = returned_command;
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

fn upload_auth_failure_event_with_dsn_override(
    tags: BTreeMap<String, String>,
    dsn_override: Option<&str>,
) -> Result<()> {
    use sentry::protocol::Envelope;
    use sentry::protocol::EnvelopeItem;

    let client = build_sentry_client_with_dsn_override(dsn_override)?;
    let mut envelope = Envelope::new();
    envelope.add_item(EnvelopeItem::Event(build_auth_failure_event(tags)));
    client.send_envelope(envelope);
    ensure_auth_failure_event_flushed(
        client.flush(Some(Duration::from_secs(UPLOAD_TIMEOUT_SECS))),
    )?;
    Ok(())
}

fn ensure_auth_failure_event_flushed(flushed: bool) -> Result<()> {
    if flushed {
        Ok(())
    } else {
        Err(anyhow!("timed out flushing auth failure event"))
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::fs;
    use std::io::Read;
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration as StdDuration;

    use super::*;
    use feedback_diagnostics::FeedbackDiagnostic;
    use pretty_assertions::assert_eq;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    #[test]
    fn ring_buffer_drops_front_when_full() {
        let fb = CodexFeedback::with_capacity(/*max_bytes*/ 8);
        {
            let mut w = fb.make_writer().make_writer();
            w.write_all(b"abcdefgh").unwrap();
            w.write_all(b"ij").unwrap();
        }
        let snap = fb.snapshot(/*session_id*/ None);
        // Capacity 8: after writing 10 bytes, we should keep the last 8.
        pretty_assertions::assert_eq!(std::str::from_utf8(snap.as_bytes()).unwrap(), "cdefghij");
    }

    #[test]
    fn metadata_layer_records_tags_from_feedback_target() {
        let fb = CodexFeedback::new();
        let _guard = tracing_subscriber::registry()
            .with(fb.metadata_layer())
            .set_default();

        tracing::info!(target: FEEDBACK_TAGS_TARGET, model = "gpt-5", cached = true, "tags");

        let snap = fb.snapshot(/*session_id*/ None);
        pretty_assertions::assert_eq!(snap.tags.get("model").map(String::as_str), Some("gpt-5"));
        pretty_assertions::assert_eq!(snap.tags.get("cached").map(String::as_str), Some("true"));
    }

    #[test]
    fn finalize_auth_failure_tags_adds_report_kind_and_drops_empty_values() {
        let tags = finalize_auth_failure_tags(BTreeMap::from([
            (String::from("endpoint"), String::from("/responses")),
            (String::from("auth_request_id"), String::new()),
        ]));

        assert_eq!(
            tags.get("report_kind").map(String::as_str),
            Some(AUTH_FAILURE_REPORT_KIND)
        );
        assert_eq!(tags.get("endpoint").map(String::as_str), Some("/responses"));
        assert!(!tags.contains_key("auth_request_id"));
    }

    #[test]
    fn auth_failure_grouping_key_uses_endpoint_code_and_header_state() {
        let grouping_key = auth_failure_grouping_key(&BTreeMap::from([
            (String::from("endpoint"), String::from("/responses")),
            (
                String::from("auth_error_code"),
                String::from("token_expired"),
            ),
            (String::from("auth_header_attached"), String::from("true")),
        ]));

        assert_eq!(
            grouping_key,
            vec![
                "codex".to_string(),
                AUTH_FAILURE_REPORT_KIND.to_string(),
                "/responses".to_string(),
                "token_expired".to_string(),
                "true".to_string(),
            ]
        );
    }

    #[test]
    fn auth_failure_grouping_key_does_not_use_raw_error_text() {
        let grouping_key = auth_failure_grouping_key(&BTreeMap::from([
            (String::from("endpoint"), String::from("/responses")),
            (
                String::from("auth_error"),
                String::from("request-specific plaintext body"),
            ),
            (String::from("auth_header_attached"), String::from("true")),
        ]));

        assert_eq!(
            grouping_key,
            vec![
                "codex".to_string(),
                AUTH_FAILURE_REPORT_KIND.to_string(),
                "/responses".to_string(),
                "unknown".to_string(),
                "true".to_string(),
            ]
        );
    }

    #[test]
    fn build_auth_failure_event_sets_stable_message_and_tags() {
        let event = build_auth_failure_event(BTreeMap::from([
            (
                String::from("report_kind"),
                AUTH_FAILURE_REPORT_KIND.to_string(),
            ),
            (String::from("endpoint"), String::from("/responses")),
            (String::from("auth_header_attached"), String::from("true")),
        ]));

        assert_eq!(
            event.message.as_deref(),
            Some("Codex client auth failure on /responses")
        );
        assert_eq!(
            event.tags.get("report_kind").map(String::as_str),
            Some(AUTH_FAILURE_REPORT_KIND)
        );
        assert_eq!(
            event.tags.get("auth_header_attached").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn auth_failure_upload_posts_envelope_to_overridden_dsn() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind local listener");
        listener
            .set_nonblocking(false)
            .expect("listener should stay blocking");
        let addr = listener.local_addr().expect("local addr");
        let (tx, rx) = mpsc::channel();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept envelope request");
            let mut buffer = Vec::new();
            let mut headers_end = None;
            while headers_end.is_none() {
                let mut chunk = [0_u8; 4096];
                let read = stream.read(&mut chunk).expect("read request headers");
                buffer.extend_from_slice(&chunk[..read]);
                headers_end = buffer.windows(4).position(|window| window == b"\r\n\r\n");
            }
            let headers_end = headers_end.expect("headers terminator should exist") + 4;
            let headers = String::from_utf8_lossy(&buffer[..headers_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    if name.eq_ignore_ascii_case("content-length") {
                        Some(value.trim().parse::<usize>().expect("content-length"))
                    } else {
                        None
                    }
                })
                .expect("content-length header");
            while buffer.len() < headers_end + content_length {
                let mut chunk = [0_u8; 4096];
                let read = stream.read(&mut chunk).expect("read request body");
                buffer.extend_from_slice(&chunk[..read]);
            }
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .expect("write response");
            tx.send(buffer).expect("capture request");
        });

        let dsn = format!("http://public@127.0.0.1:{}/1", addr.port());
        upload_auth_failure_event_with_dsn_override(
            finalize_auth_failure_tags(BTreeMap::from([
                (String::from("endpoint"), String::from("/oauth/token")),
                (
                    String::from("auth_error_code"),
                    String::from("refresh_token_reused"),
                ),
                (String::from("auth_header_attached"), String::from("true")),
            ])),
            Some(&dsn),
        )
        .expect("upload auth failure event");

        let request = rx
            .recv_timeout(StdDuration::from_secs(5))
            .expect("receive envelope request");
        server.join().expect("server thread should exit");

        let request_text = String::from_utf8_lossy(&request);
        assert!(request_text.contains("POST /api/1/envelope/"));
        assert!(request_text.contains("\"report_kind\":\"auth_failure_auto\""));
        assert!(request_text.contains("\"endpoint\":\"/oauth/token\""));
        assert!(request_text.contains("\"auth_error_code\":\"refresh_token_reused\""));
    }

    #[test]
    fn enqueue_auth_failure_event_tags_posts_envelope_to_overridden_dsn() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind local listener");
        listener
            .set_nonblocking(false)
            .expect("listener should stay blocking");
        let addr = listener.local_addr().expect("local addr");
        let (tx, rx) = mpsc::channel();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept envelope request");
            let mut buffer = Vec::new();
            let mut headers_end = None;
            while headers_end.is_none() {
                let mut chunk = [0_u8; 4096];
                let read = stream.read(&mut chunk).expect("read request headers");
                buffer.extend_from_slice(&chunk[..read]);
                headers_end = buffer.windows(4).position(|window| window == b"\r\n\r\n");
            }
            let headers_end = headers_end.expect("headers terminator should exist") + 4;
            let headers = String::from_utf8_lossy(&buffer[..headers_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    if name.eq_ignore_ascii_case("content-length") {
                        Some(value.trim().parse::<usize>().expect("content-length"))
                    } else {
                        None
                    }
                })
                .expect("content-length header");
            while buffer.len() < headers_end + content_length {
                let mut chunk = [0_u8; 4096];
                let read = stream.read(&mut chunk).expect("read request body");
                buffer.extend_from_slice(&chunk[..read]);
            }
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .expect("write response");
            tx.send(buffer).expect("capture request");
        });

        enqueue_auth_failure_event_with_dsn_override(
            BTreeMap::from([
                (String::from("endpoint"), String::from("/oauth/token")),
                (
                    String::from("auth_error_code"),
                    String::from("refresh_token_reused"),
                ),
                (String::from("auth_header_attached"), String::from("false")),
            ]),
            Some(format!("http://public@127.0.0.1:{}/1", addr.port())),
        );

        let request = rx
            .recv_timeout(StdDuration::from_secs(5))
            .expect("receive envelope request");
        server.join().expect("server thread should exit");

        let request_text = String::from_utf8_lossy(&request);
        assert!(request_text.contains("POST /api/1/envelope/"));
        assert!(request_text.contains("\"report_kind\":\"auth_failure_auto\""));
        assert!(request_text.contains("\"endpoint\":\"/oauth/token\""));
        assert!(request_text.contains("\"auth_header_attached\":\"false\""));
    }

    #[test]
    fn flush_auth_failure_event_queue_waits_for_enqueued_upload() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind local listener");
        listener
            .set_nonblocking(false)
            .expect("listener should stay blocking");
        let addr = listener.local_addr().expect("local addr");
        let (tx, rx) = mpsc::channel();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept envelope request");
            let mut buffer = Vec::new();
            let mut headers_end = None;
            while headers_end.is_none() {
                let mut chunk = [0_u8; 4096];
                let read = stream.read(&mut chunk).expect("read request headers");
                buffer.extend_from_slice(&chunk[..read]);
                headers_end = buffer.windows(4).position(|window| window == b"\r\n\r\n");
            }
            let headers_end = headers_end.expect("headers terminator should exist") + 4;
            let headers = String::from_utf8_lossy(&buffer[..headers_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    if name.eq_ignore_ascii_case("content-length") {
                        Some(value.trim().parse::<usize>().expect("content-length"))
                    } else {
                        None
                    }
                })
                .expect("content-length header");
            while buffer.len() < headers_end + content_length {
                let mut chunk = [0_u8; 4096];
                let read = stream.read(&mut chunk).expect("read request body");
                buffer.extend_from_slice(&chunk[..read]);
            }
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .expect("write response");
            tx.send(buffer).expect("capture request");
        });

        enqueue_auth_failure_event_with_dsn_override(
            BTreeMap::from([
                (String::from("endpoint"), String::from("/oauth/token")),
                (
                    String::from("auth_error_code"),
                    String::from("refresh_token_reused"),
                ),
                (String::from("auth_header_attached"), String::from("false")),
            ]),
            Some(format!("http://public@127.0.0.1:{}/1", addr.port())),
        );

        flush_auth_failure_event_queue(StdDuration::from_secs(5));

        let request = rx
            .recv_timeout(StdDuration::from_secs(5))
            .expect("receive envelope request");
        server.join().expect("server thread should exit");

        let request_text = String::from_utf8_lossy(&request);
        assert!(request_text.contains("POST /api/1/envelope/"));
        assert!(request_text.contains("\"endpoint\":\"/oauth/token\""));
    }

    #[test]
    fn send_auth_failure_flush_command_times_out_when_queue_stays_full() {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        tx.try_send(AuthFailureUploadCommand::Upload(AuthFailureUploadTask {
            tags: BTreeMap::new(),
            dsn_override: None,
        }))
        .expect("fill queue");
        std::mem::forget(rx);

        let (done_tx, _done_rx) = mpsc::channel();
        let start = std::time::Instant::now();
        let err = send_auth_failure_flush_command(
            &tx,
            done_tx,
            std::time::Instant::now() + StdDuration::from_millis(50),
        )
        .expect_err("flush enqueue should time out");

        assert!(matches!(err, std::sync::mpsc::TrySendError::Full(_)));
        assert!(start.elapsed() < StdDuration::from_secs(1));
    }

    #[test]
    fn ensure_auth_failure_event_flushed_rejects_timeouts() {
        assert!(ensure_auth_failure_event_flushed(true).is_ok());
        assert_eq!(
            ensure_auth_failure_event_flushed(false)
                .expect_err("flush timeout should fail")
                .to_string(),
            "timed out flushing auth failure event"
        );
    }

    #[test]
    fn auth_failure_queue_flush_timeout_is_bounded_to_primary_and_fallback_budget() {
        let timeout = auth_failure_event_queue_flush_timeout();
        assert!(timeout > StdDuration::from_secs(UPLOAD_TIMEOUT_SECS * 3));
        assert!(timeout < StdDuration::from_secs(UPLOAD_TIMEOUT_SECS * 4));
    }

    #[test]
    fn feedback_attachments_gate_connectivity_diagnostics() {
        let extra_filename = format!("codex-feedback-extra-{}.jsonl", ThreadId::new());
        let extra_path = std::env::temp_dir().join(&extra_filename);
        fs::write(&extra_path, "rollout").expect("extra attachment should be written");

        let snapshot_with_diagnostics = CodexFeedback::new()
            .snapshot(/*session_id*/ None)
            .with_feedback_diagnostics(FeedbackDiagnostics::new(vec![FeedbackDiagnostic {
                headline: "OPENAI_BASE_URL is set and may affect connectivity.".to_string(),
                details: vec!["OPENAI_BASE_URL = https://example.com/v1".to_string()],
            }]));

        let attachments_with_diagnostics = snapshot_with_diagnostics.feedback_attachments(
            /*include_logs*/ true,
            std::slice::from_ref(&extra_path),
            Some(vec![1]),
        );

        assert_eq!(
            attachments_with_diagnostics
                .iter()
                .map(|attachment| attachment.filename.as_str())
                .collect::<Vec<_>>(),
            vec![
                "codex-logs.log",
                FEEDBACK_DIAGNOSTICS_ATTACHMENT_FILENAME,
                extra_filename.as_str()
            ]
        );
        assert_eq!(attachments_with_diagnostics[0].buffer, vec![1]);
        assert_eq!(
            attachments_with_diagnostics[1].buffer,
            b"Connectivity diagnostics\n\n- OPENAI_BASE_URL is set and may affect connectivity.\n  - OPENAI_BASE_URL = https://example.com/v1".to_vec()
        );
        assert_eq!(attachments_with_diagnostics[2].buffer, b"rollout".to_vec());
        assert_eq!(
            OsStr::new(attachments_with_diagnostics[2].filename.as_str()),
            OsStr::new(extra_filename.as_str())
        );
        let attachments_without_diagnostics = CodexFeedback::new()
            .snapshot(/*session_id*/ None)
            .feedback_attachments(/*include_logs*/ true, &[], Some(vec![1]));

        assert_eq!(
            attachments_without_diagnostics
                .iter()
                .map(|attachment| attachment.filename.as_str())
                .collect::<Vec<_>>(),
            vec!["codex-logs.log"]
        );
        assert_eq!(attachments_without_diagnostics[0].buffer, vec![1]);
        fs::remove_file(extra_path).expect("extra attachment should be removed");
    }
}
