//! Sync engine (SPEC §4.3, §7). Batches pending events (≤500 per request) to the
//! PHR backend, idempotent via `client_event_uuid`; per-event `accepted` /
//! `duplicate` mark the row uploaded, `rejected` is surfaced and left pending.
//! Local tombstones (uploaded-then-deleted) sync as a DELETE batch. Failures use
//! exponential backoff (30 s → 30 min cap) with full jitter.
//!
//! **Offline-strict is structural, not a flag** (SPEC §4.3, §8): [`SyncEngine`] is
//! the only thing that owns an HTTP client, and [`SyncEngine::for_mode`] returns
//! `None` for [`Mode::OfflineStrict`]. In that mode no engine exists, so there is
//! no reachable network I/O path at all.

use std::time::Duration;

use chrono::Utc;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::store::Store;
use crate::token::TokenStore;
use crate::types::{Event, Source};

/// Sync behaviour mode (SPEC §4.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    AutoBatch,
    OfflineFirst,
    OfflineStrict,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::AutoBatch => "auto-batch",
            Mode::OfflineFirst => "offline-first",
            Mode::OfflineStrict => "offline-strict",
        }
    }

    /// Whether this mode is permitted to perform network I/O at all.
    pub fn allows_network(self) -> bool {
        !matches!(self, Mode::OfflineStrict)
    }
}

/// Per-event server verdict (SPEC §7 contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EventStatus {
    Accepted,
    Duplicate,
    Rejected,
}

#[derive(Debug, Clone, Deserialize)]
struct PerEventResult {
    uuid: String,
    status: EventStatus,
    #[allow(dead_code)]
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct BatchResponse {
    results: Vec<PerEventResult>,
}

/// The per-event payload uploaded to the backend (SPEC §7 — metadata only, never
/// audio).
#[derive(Debug, Clone, Serialize)]
struct UploadEvent<'a> {
    uuid: &'a str,
    event_type: &'a str,
    occurred_at: String,
    tz_offset_min: i32,
    duration_ms: i64,
    confidence: f32,
    burst_count: i64,
}

impl<'a> From<&'a Event> for UploadEvent<'a> {
    fn from(e: &'a Event) -> Self {
        UploadEvent {
            uuid: &e.uuid,
            event_type: e.event_type.as_str(),
            occurred_at: e.occurred_at.to_rfc3339(),
            tz_offset_min: e.tz_offset_min,
            duration_ms: e.duration_ms,
            confidence: e.confidence,
            burst_count: e.burst_count,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct BatchRequest<'a> {
    device_id: &'a str,
    source: &'a str,
    model_version: &'a str,
    events: Vec<UploadEvent<'a>>,
}

#[derive(Debug, Clone, Serialize)]
struct DeleteRequest<'a> {
    uuids: &'a [String],
}

/// Outcome of a flush.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SyncOutcome {
    pub accepted: usize,
    pub duplicate: usize,
    pub rejected: usize,
    pub deleted: usize,
}

/// Sync engine configuration.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Backend base URL, e.g. `https://example.test` (no trailing slash).
    pub base_url: String,
    /// PHR patient id the events bind to.
    pub patient_id: i64,
    pub device_id: String,
    pub source: Source,
    pub model_version: String,
    /// Max events per request (SPEC: ≤500).
    pub batch_size: usize,
}

impl SyncConfig {
    fn batch_url(&self) -> String {
        format!(
            "{}/api/phr/patients/{}/respiratory-events/batch",
            self.base_url.trim_end_matches('/'),
            self.patient_id
        )
    }
}

/// Exponential backoff with full jitter (SPEC §4.3): 30 s → 30 min cap.
#[derive(Debug, Clone)]
pub struct Backoff {
    base: Duration,
    cap: Duration,
    attempt: u32,
}

impl Default for Backoff {
    fn default() -> Self {
        Backoff {
            base: Duration::from_secs(30),
            cap: Duration::from_secs(30 * 60),
            attempt: 0,
        }
    }
}

impl Backoff {
    pub fn new(base: Duration, cap: Duration) -> Self {
        Backoff {
            base,
            cap,
            attempt: 0,
        }
    }

    /// Upper bound (pre-jitter) for the current attempt.
    fn ceiling(&self) -> Duration {
        let factor = 2u64.saturating_pow(self.attempt);
        let scaled = self.base.saturating_mul(factor.min(u32::MAX as u64) as u32);
        scaled.min(self.cap)
    }

    /// Next delay to wait (full jitter in `[0, ceiling]`), advancing the attempt.
    pub fn next_delay(&mut self) -> Duration {
        let ceil = self.ceiling();
        self.attempt = self.attempt.saturating_add(1);
        let millis = ceil.as_millis() as u64;
        if millis == 0 {
            return Duration::ZERO;
        }
        let jittered = rand::thread_rng().gen_range(0..=millis);
        Duration::from_millis(jittered)
    }

    /// Reset after a successful flush.
    pub fn reset(&mut self) {
        self.attempt = 0;
    }

    pub fn attempt(&self) -> u32 {
        self.attempt
    }
}

/// The sync engine. Constructed only for network-capable modes.
pub struct SyncEngine<T: TokenStore> {
    cfg: SyncConfig,
    client: reqwest::blocking::Client,
    token: T,
}

impl<T: TokenStore> SyncEngine<T> {
    /// Construct an engine for `mode`, or `None` for [`Mode::OfflineStrict`] — the
    /// mode in which no HTTP client is ever created (SPEC §4.3).
    pub fn for_mode(mode: Mode, cfg: SyncConfig, token: T) -> Option<Self> {
        if !mode.allows_network() {
            return None;
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .ok()?;
        Some(SyncEngine { cfg, client, token })
    }

    fn bearer(&self) -> Result<String> {
        self.token
            .get_token()?
            .ok_or_else(|| Error::Token("no API token configured".into()))
    }

    /// Upload one batch and return the server's per-event verdicts.
    fn upload_batch(&self, events: &[Event]) -> Result<Vec<PerEventResult>> {
        let body = BatchRequest {
            device_id: &self.cfg.device_id,
            source: self.cfg.source.as_str(),
            model_version: &self.cfg.model_version,
            events: events.iter().map(UploadEvent::from).collect(),
        };
        let resp = self
            .client
            .post(self.cfg.batch_url())
            .bearer_auth(self.bearer()?)
            .json(&body)
            .send()
            .map_err(|e| Error::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(Error::Http(format!("batch upload HTTP {}", resp.status())));
        }
        let parsed: BatchResponse = resp.json().map_err(|e| Error::Http(e.to_string()))?;
        Ok(parsed.results)
    }

    /// DELETE a batch of tombstoned uuids.
    fn delete_batch(&self, uuids: &[String]) -> Result<()> {
        let resp = self
            .client
            .delete(self.cfg.batch_url())
            .bearer_auth(self.bearer()?)
            .json(&DeleteRequest { uuids })
            .send()
            .map_err(|e| Error::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(Error::Http(format!(
                "tombstone delete HTTP {}",
                resp.status()
            )));
        }
        Ok(())
    }

    /// Flush all pending events and tombstones from `store`. Accepted/duplicate
    /// events are marked uploaded; rejected are counted and left pending;
    /// confirmed tombstones are purged locally.
    pub fn flush(&self, store: &Store) -> Result<SyncOutcome> {
        let mut outcome = SyncOutcome::default();

        loop {
            let batch = store.pending_events(self.cfg.batch_size)?;
            if batch.is_empty() {
                break;
            }
            let results = self.upload_batch(&batch)?;
            let mut uploaded = Vec::new();
            let mut rejected = Vec::new();
            for r in results {
                match r.status {
                    EventStatus::Accepted => {
                        outcome.accepted += 1;
                        uploaded.push(r.uuid);
                    }
                    EventStatus::Duplicate => {
                        outcome.duplicate += 1;
                        uploaded.push(r.uuid);
                    }
                    EventStatus::Rejected => {
                        outcome.rejected += 1;
                        rejected.push(r.uuid);
                    }
                }
            }
            let now = Utc::now();
            store.mark_uploaded(&uploaded, now)?;
            // Record rejections so a permanently-rejected event stops being re-sent
            // after `MAX_REJECTIONS` (SPEC §4.3) instead of looping forever.
            store.mark_rejected(&rejected, now)?;
            // If nothing was marked uploaded, avoid an infinite loop on a
            // fully-rejected batch — the rejected rows are now counted and will
            // drop out of `pending_events` once they pass the cap.
            if uploaded.is_empty() {
                break;
            }
        }

        loop {
            let tombstones = store.pending_tombstones(self.cfg.batch_size)?;
            if tombstones.is_empty() {
                break;
            }
            self.delete_batch(&tombstones)?;
            store.purge(&tombstones)?;
            outcome.deleted += tombstones.len();
        }

        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::InMemoryTokenStore;
    use crate::types::EventType;
    use std::sync::mpsc;

    fn sample_event(uuid: &str) -> Event {
        Event {
            uuid: uuid.to_string(),
            event_type: EventType::Cough,
            occurred_at: Utc::now(),
            tz_offset_min: 0,
            duration_ms: 500,
            confidence: 0.6,
            burst_count: 1,
            model_version: "test@0".into(),
            source: Source::DesktopMac,
            device_id: "dev".into(),
            uploaded_at: None,
            deleted: false,
            reject_count: 0,
            rejected_at: None,
        }
    }

    /// A one-shot mock HTTP server. Returns the canned body for the first request
    /// and reports the method + captured body back over a channel.
    fn spawn_mock(
        response_body: String,
        tx: mpsc::Sender<(String, String)>,
    ) -> (String, std::thread::JoinHandle<()>) {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let url = format!("http://{}", server.server_addr());
        let handle = std::thread::spawn(move || {
            if let Ok(mut req) = server.recv() {
                let method = req.method().as_str().to_string();
                let mut body = String::new();
                req.as_reader().read_to_string(&mut body).ok();
                tx.send((method, body)).ok();
                let resp = tiny_http::Response::from_string(response_body)
                    .with_status_code(200)
                    .with_header(
                        tiny_http::Header::from_bytes(
                            &b"Content-Type"[..],
                            &b"application/json"[..],
                        )
                        .unwrap(),
                    );
                req.respond(resp).ok();
            }
        });
        (url, handle)
    }

    fn config(base_url: String) -> SyncConfig {
        SyncConfig {
            base_url,
            patient_id: 7,
            device_id: "dev-1".into(),
            source: Source::DesktopMac,
            model_version: "test@0".into(),
            batch_size: 500,
        }
    }

    #[test]
    fn offline_strict_constructs_no_engine() {
        let engine = SyncEngine::for_mode(
            Mode::OfflineStrict,
            config("http://unused".into()),
            InMemoryTokenStore::with_token("t"),
        );
        assert!(
            engine.is_none(),
            "offline-strict must never build an engine"
        );
    }

    #[test]
    fn flush_uploads_and_marks_accepted_and_duplicate() {
        let (tx, rx) = mpsc::channel();
        let body = r#"{"results":[
            {"uuid":"a","status":"accepted"},
            {"uuid":"b","status":"duplicate"}
        ]}"#
        .to_string();
        let (url, handle) = spawn_mock(body, tx);

        let store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("a")).unwrap();
        store.insert_event(&sample_event("b")).unwrap();

        let engine = SyncEngine::for_mode(
            Mode::AutoBatch,
            config(url),
            InMemoryTokenStore::with_token("secret"),
        )
        .unwrap();
        let outcome = engine.flush(&store).unwrap();

        assert_eq!(outcome.accepted, 1);
        assert_eq!(outcome.duplicate, 1);
        assert_eq!(outcome.rejected, 0);
        // Both should now be marked uploaded → no longer pending.
        assert!(store.pending_events(10).unwrap().is_empty());

        let (method, sent) = rx.recv().unwrap();
        assert_eq!(method, "POST");
        assert!(sent.contains("\"device_id\":\"dev-1\""));
        assert!(sent.contains("\"uuid\":\"a\""));
        // Privacy: never any audio field.
        assert!(!sent.contains("audio"));
        handle.join().ok();
    }

    #[test]
    fn rejected_events_stay_pending() {
        let (tx, _rx) = mpsc::channel();
        let body = r#"{"results":[{"uuid":"a","status":"rejected","reason":"bad"}]}"#.to_string();
        let (url, handle) = spawn_mock(body, tx);

        let store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("a")).unwrap();

        let engine = SyncEngine::for_mode(
            Mode::AutoBatch,
            config(url),
            InMemoryTokenStore::with_token("secret"),
        )
        .unwrap();
        let outcome = engine.flush(&store).unwrap();
        assert_eq!(outcome.rejected, 1);
        assert_eq!(store.pending_events(10).unwrap().len(), 1);
        handle.join().ok();
    }

    #[test]
    fn tombstones_sync_as_delete_batch() {
        let (tx, rx) = mpsc::channel();
        let (url, handle) = spawn_mock("{}".to_string(), tx);

        let store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("x")).unwrap();
        store.mark_uploaded(&["x".into()], Utc::now()).unwrap();
        store.mark_deleted("x").unwrap();

        let engine = SyncEngine::for_mode(
            Mode::AutoBatch,
            config(url),
            InMemoryTokenStore::with_token("secret"),
        )
        .unwrap();
        let outcome = engine.flush(&store).unwrap();
        assert_eq!(outcome.deleted, 1);
        assert!(store.get_event("x").unwrap().is_none());

        let (method, sent) = rx.recv().unwrap();
        assert_eq!(method, "DELETE");
        assert!(sent.contains("\"uuids\":[\"x\"]"));
        handle.join().ok();
    }

    #[test]
    fn missing_token_errors() {
        let store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("a")).unwrap();
        let engine = SyncEngine::for_mode(
            Mode::AutoBatch,
            config("http://127.0.0.1:1".into()),
            InMemoryTokenStore::new(),
        )
        .unwrap();
        assert!(matches!(engine.flush(&store), Err(Error::Token(_))));
    }

    #[test]
    fn backoff_grows_and_caps() {
        let mut b = Backoff::new(Duration::from_secs(30), Duration::from_secs(30 * 60));
        // Ceilings: 30, 60, 120, ... capped at 1800 s. With full jitter each delay
        // is within [0, ceiling]; verify the ceiling schedule via monotonic caps.
        for expected_ceiling in [30u64, 60, 120, 240, 480, 960, 1800, 1800] {
            let ceil = b.ceiling().as_secs();
            assert_eq!(ceil, expected_ceiling, "attempt {}", b.attempt());
            let d = b.next_delay();
            assert!(d.as_secs() <= expected_ceiling);
        }
        b.reset();
        assert_eq!(b.attempt(), 0);
        assert_eq!(b.ceiling().as_secs(), 30);
    }
}
