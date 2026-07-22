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
    #[serde(rename = "client_event_uuid")]
    uuid: &'a str,
    event_type: &'a str,
    occurred_at: String,
    tz_offset_min: i32,
    duration_ms: i64,
    confidence: f32,
    burst_count: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    peak_dbfs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mean_dbfs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    noise_floor_dbfs: Option<f32>,
    source: &'a str,
    device_id: &'a str,
    model_version: &'a str,
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
            peak_dbfs: e.peak_dbfs,
            mean_dbfs: e.mean_dbfs,
            noise_floor_dbfs: e.noise_floor_dbfs,
            source: e.source.as_str(),
            device_id: &e.device_id,
            model_version: &e.model_version,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct BatchRequest<'a> {
    events: Vec<UploadEvent<'a>>,
}

#[derive(Debug, Clone, Serialize)]
struct DeleteRequest<'a> {
    uuids: &'a [String],
}

/// One declarative flag: the *current* local state, not a delta. Clearing a
/// flag (the user's Undo) is `false_positive: false` with a null
/// `corrected_to`, which is why this can express an undo at all.
#[derive(Debug, Clone, Serialize)]
struct FlagItem<'a> {
    uuid: &'a str,
    false_positive: bool,
    corrected_to: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize)]
struct FlagRequest<'a> {
    items: Vec<FlagItem<'a>>,
}

#[derive(Debug, Clone, Deserialize)]
struct FlagResult {
    #[allow(dead_code)]
    uuid: String,
    #[allow(dead_code)]
    status: String,
}

#[derive(Debug, Clone, Deserialize)]
struct FlagResponse {
    #[allow(dead_code)]
    #[serde(default)]
    results: Vec<FlagResult>,
}

#[derive(Debug, Clone, Serialize)]
struct SettingsRequest<'a> {
    settings: &'a std::collections::BTreeMap<String, String>,
    updated_at: String,
    device_id: &'a str,
}

#[derive(Debug, Clone, Deserialize)]
struct SettingsDocWire {
    #[serde(default)]
    settings: std::collections::BTreeMap<String, String>,
    updated_at: String,
}

#[derive(Debug, Clone, Deserialize)]
struct SettingsShowResponse {
    sinus_settings: Option<SettingsDocWire>,
}

#[derive(Debug, Clone, Deserialize)]
struct SettingsUpdateResponse {
    #[allow(dead_code)]
    applied: bool,
    sinus_settings: Option<SettingsDocWire>,
}

/// Wire form of an enrollment. `client_enrollment_uuid` and `embedding` are
/// base64 of raw bytes — the same bytes the local SQLite BLOBs hold, so no
/// float is ever reformatted in transit.
#[derive(Debug, Clone, Serialize)]
struct UploadEnrollment {
    client_enrollment_uuid: String,
    class: &'static str,
    is_negative: bool,
    negative_scoped: bool,
    embedding: String,
    embedding_dim: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    similarity: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    separation: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    peak_dbfs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_event_uuid: Option<String>,
    device_id: String,
    captured_at: String,
}

#[derive(Debug, Clone, Serialize)]
struct EnrollmentBatchRequest {
    enrollments: Vec<UploadEnrollment>,
}

#[derive(Debug, Clone, Deserialize)]
struct DownloadEnrollment {
    client_enrollment_uuid: String,
    class: String,
    #[serde(default)]
    is_negative: bool,
    #[serde(default)]
    negative_scoped: bool,
    embedding: String,
    #[serde(default)]
    similarity: Option<f32>,
    #[serde(default)]
    separation: Option<f32>,
    #[serde(default)]
    peak_dbfs: Option<f32>,
    #[serde(default)]
    model_version: Option<String>,
    #[serde(default)]
    source_event_uuid: Option<String>,
    captured_at: String,
}

#[derive(Debug, Clone, Deserialize)]
struct EnrollmentIndexResponse {
    #[serde(default)]
    sinus_enrollments: Vec<DownloadEnrollment>,
}

/// Per-item verdict for an enrollment push. Same shape as the event batch.
#[derive(Debug, Clone, Deserialize)]
struct EnrollmentResult {
    uuid: Option<String>,
    status: String,
    #[allow(dead_code)]
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct EnrollmentBatchResponse {
    #[serde(default)]
    results: Vec<EnrollmentResult>,
}

/// Outcome of a flush.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SyncOutcome {
    pub accepted: usize,
    pub duplicate: usize,
    pub rejected: usize,
    pub deleted: usize,
    /// False-positive / correction flags pushed.
    pub flagged: usize,
    /// Teach-mode examples pushed up.
    pub enrollments_pushed: usize,
    /// Teach-mode examples pulled down (a fresh machine inheriting training).
    pub enrollments_pulled: usize,
    /// Settings keys adopted from the server.
    pub settings_pulled: usize,
    /// Whether anything changed that the capture thread must reload.
    pub reload_settings: bool,
    pub reload_enrollments: bool,
}

/// Sync engine configuration.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Backend base URL or complete respiratory-event batch endpoint.
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
    /// The configured value may be either a server root or the complete
    /// respiratory-events batch endpoint (the historical form). Normalize to a
    /// root so the other endpoints can be derived from it.
    fn server_root(&self) -> &str {
        let configured = self.base_url.trim_end_matches('/');
        configured
            .strip_suffix("/respiratory-events/batch")
            .and_then(|rest| rest.rsplit_once("/api/phr/patients/"))
            .map(|(root, _)| root)
            .unwrap_or(configured)
    }

    fn batch_url(&self) -> String {
        let configured = self.base_url.trim_end_matches('/');
        if configured.ends_with("/respiratory-events/batch") {
            configured.to_string()
        } else {
            format!(
                "{configured}/api/phr/patients/{}/respiratory-events/batch",
                self.patient_id
            )
        }
    }

    fn endpoint(&self, suffix: &str) -> String {
        format!(
            "{}/api/phr/patients/{}/{suffix}",
            self.server_root(),
            self.patient_id
        )
    }

    fn flag_url(&self) -> String {
        self.endpoint("respiratory-events/flag-batch")
    }

    fn settings_url(&self) -> String {
        self.endpoint("sinus-settings")
    }

    fn enrollments_url(&self) -> String {
        self.endpoint("sinus-enrollments")
    }

    fn enrollment_batch_url(&self) -> String {
        self.endpoint("sinus-enrollments/batch")
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
            events: events.iter().map(UploadEvent::from).collect(),
        };
        let resp = self
            .client
            .post(self.cfg.batch_url())
            .header(reqwest::header::ACCEPT, "application/json")
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
            .header(reqwest::header::ACCEPT, "application/json")
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

    /// Whether a response means "this PHR predates the feature" rather than a
    /// real failure.
    ///
    /// 404 is ambiguous — `PhrPatientAccessService` also 404s an inaccessible or
    /// non-existent patient — so a mistyped `patient_id` makes these sync steps
    /// quietly no-op. That is tolerable only because the *events* endpoint fails
    /// loudly on the same misconfiguration, so the user still sees sync break;
    /// the UI additionally surfaces the skipped feature.
    fn is_unsupported(status: reqwest::StatusCode) -> bool {
        status == reqwest::StatusCode::NOT_FOUND
            || status == reqwest::StatusCode::METHOD_NOT_ALLOWED
    }

    /// Push the current state of every locally-changed false-positive /
    /// correction flag.
    fn flush_flags(&self, store: &Store) -> Result<usize> {
        let mut pushed = 0;

        loop {
            let pending = store.pending_flags(self.cfg.batch_size)?;
            if pending.is_empty() {
                break;
            }

            let items: Vec<FlagItem> = pending
                .iter()
                .map(|flag| FlagItem {
                    uuid: &flag.uuid,
                    false_positive: flag.false_positive,
                    corrected_to: flag.corrected_to.map(|c| c.as_str()),
                })
                .collect();

            let resp = self
                .client
                .post(self.cfg.flag_url())
                .header(reqwest::header::ACCEPT, "application/json")
                .bearer_auth(self.bearer()?)
                .json(&FlagRequest { items })
                .send()
                .map_err(|e| Error::Http(e.to_string()))?;

            if Self::is_unsupported(resp.status()) {
                return Ok(pushed);
            }
            if !resp.status().is_success() {
                return Err(Error::Http(format!("flag batch HTTP {}", resp.status())));
            }
            let _: FlagResponse = resp.json().map_err(|e| Error::Http(e.to_string()))?;

            // `not_found` is deliberately treated as terminal alongside
            // `flagged`: an event the server never accepted (rejected past
            // MAX_REJECTIONS, or deleted there) would otherwise be re-sent on
            // every flush forever.
            store.mark_flags_synced(&pending)?;
            pushed += pending.len();

            if pending.len() < self.cfg.batch_size {
                break;
            }
        }

        Ok(pushed)
    }

    /// Reconcile settings with the PHR, last-write-wins.
    ///
    /// Returns the number of keys adopted from the server; a non-zero count means
    /// the capture thread must re-read its detection config.
    fn sync_settings(&self, store: &mut Store) -> Result<usize> {
        let local = store.settings_doc()?;

        let resp = self
            .client
            .get(self.cfg.settings_url())
            .header(reqwest::header::ACCEPT, "application/json")
            .bearer_auth(self.bearer()?)
            .send()
            .map_err(|e| Error::Http(e.to_string()))?;

        if Self::is_unsupported(resp.status()) {
            return Ok(0);
        }
        if !resp.status().is_success() {
            return Err(Error::Http(format!("settings GET HTTP {}", resp.status())));
        }
        let remote: SettingsShowResponse = resp.json().map_err(|e| Error::Http(e.to_string()))?;
        let remote_doc = remote.sinus_settings.and_then(parse_settings_doc);

        let local_newer = match (&local.updated_at, &remote_doc) {
            (Some(local_time), Some(remote)) => remote
                .updated_at
                .is_none_or(|remote_time| *local_time > remote_time),
            (Some(_), None) => true,
            (None, _) => false,
        };

        if local_newer && !local.is_empty() {
            // Whole seconds: the server stores DATETIME and echoes back
            // truncated, so sending sub-second precision would make our own copy
            // look strictly newer than the one the server just confirmed, and we
            // would re-push the identical document on every flush forever.
            let updated_at =
                crate::store::truncate_to_second(local.updated_at.unwrap_or_else(Utc::now))
                    .to_rfc3339();

            let resp = self
                .client
                .put(self.cfg.settings_url())
                .header(reqwest::header::ACCEPT, "application/json")
                .bearer_auth(self.bearer()?)
                .json(&SettingsRequest {
                    settings: &local.values,
                    updated_at,
                    device_id: &self.cfg.device_id,
                })
                .send()
                .map_err(|e| Error::Http(e.to_string()))?;

            if Self::is_unsupported(resp.status()) {
                return Ok(0);
            }
            // A 422 here is a verdict on this document — most likely a clock
            // more than MAX_CLOCK_SKEW_MINUTES fast. Failing the flush would
            // block the enrollment step behind it until wall-clock caught up,
            // so skip this cycle instead.
            if resp.status() == reqwest::StatusCode::UNPROCESSABLE_ENTITY {
                eprintln!("sync: the server rejected these settings; check this device's clock");
                return Ok(0);
            }
            if !resp.status().is_success() {
                return Err(Error::Http(format!("settings PUT HTTP {}", resp.status())));
            }

            // The response always carries the winning document, so a device that
            // lost the race adopts server state here rather than needing another
            // round trip.
            let updated: SettingsUpdateResponse =
                resp.json().map_err(|e| Error::Http(e.to_string()))?;
            let winner = updated.sinus_settings.and_then(parse_settings_doc);
            return match winner {
                Some(doc) => Ok(store.apply_settings_doc(&doc)?.len()),
                None => Ok(0),
            };
        }

        match remote_doc {
            Some(doc) => Ok(store.apply_settings_doc(&doc)?.len()),
            None => Ok(0),
        }
    }

    /// Push locally-taught examples and deletions, then pull down anything this
    /// machine has never seen.
    ///
    /// Returns `(pushed, pulled)`. A non-zero `pulled` means the capture thread
    /// must rebuild its prototype matcher — otherwise a fresh machine downloads
    /// the user's training and keeps detecting as though it had none.
    fn sync_enrollments(&self, store: &Store) -> Result<(usize, usize)> {
        let mut pushed = 0;

        // Push new examples.
        loop {
            let pending = store.pending_enrollments(ENROLLMENT_BATCH_SIZE)?;
            if pending.is_empty() {
                break;
            }

            let body = EnrollmentBatchRequest {
                enrollments: pending
                    .iter()
                    .map(|stored| upload_enrollment(stored, &self.cfg.device_id))
                    .collect(),
            };

            let resp = self
                .client
                .post(self.cfg.enrollment_batch_url())
                .header(reqwest::header::ACCEPT, "application/json")
                .bearer_auth(self.bearer()?)
                .json(&body)
                .send()
                .map_err(|e| Error::Http(e.to_string()))?;

            if Self::is_unsupported(resp.status()) {
                return Ok((pushed, 0));
            }
            if !resp.status().is_success() {
                return Err(Error::Http(format!(
                    "enrollment batch HTTP {}",
                    resp.status()
                )));
            }

            // Honour the per-item verdicts. Marking the whole batch synced
            // regardless would silently strand every `rejected` item: never
            // retried, never on the server, never inherited by another machine.
            let parsed: EnrollmentBatchResponse =
                resp.json().map_err(|e| Error::Http(e.to_string()))?;

            let accepted: std::collections::HashSet<&str> = parsed
                .results
                .iter()
                .filter(|r| r.status == "accepted" || r.status == "duplicate")
                .filter_map(|r| r.uuid.as_deref())
                .collect();

            let mut accepted_uuids = Vec::new();
            let mut rejected = 0usize;
            for stored in &pending {
                if accepted.contains(b64(&stored.uuid).as_str()) {
                    accepted_uuids.push(stored.uuid.clone());
                } else {
                    rejected += 1;
                    eprintln!(
                        "sync: the server rejected enrollment {} ({}); it will not be retried",
                        b64(&stored.uuid),
                        stored.enrollment.class.as_str(),
                    );
                }
            }

            // A rejection is a deterministic verdict on the payload, not a
            // transient failure, so retrying it forever is pointless — mark the
            // whole batch resolved and log the rejects. `upload_enrollment`
            // clamps the values the server range-checks, so a rejection here
            // means a genuine contract mismatch worth seeing in the log.
            let mut resolved = accepted_uuids.clone();
            if rejected > 0 {
                resolved = pending.iter().map(|s| s.uuid.clone()).collect();
            }
            store.mark_enrollments_synced(&resolved)?;
            pushed += accepted_uuids.len();

            if pending.len() < ENROLLMENT_BATCH_SIZE {
                break;
            }
        }

        // Push deletions.
        loop {
            let tombstones = store.pending_enrollment_deletions(ENROLLMENT_BATCH_SIZE)?;
            if tombstones.is_empty() {
                break;
            }
            let encoded: Vec<String> = tombstones.iter().map(|u| b64(u)).collect();

            let resp = self
                .client
                .delete(self.cfg.enrollment_batch_url())
                .header(reqwest::header::ACCEPT, "application/json")
                .bearer_auth(self.bearer()?)
                .json(&DeleteRequest { uuids: &encoded })
                .send()
                .map_err(|e| Error::Http(e.to_string()))?;

            if Self::is_unsupported(resp.status()) {
                break;
            }
            if !resp.status().is_success() {
                return Err(Error::Http(format!(
                    "enrollment delete HTTP {}",
                    resp.status()
                )));
            }
            store.purge_enrollments(&tombstones)?;
        }

        // Pull anything this machine has never seen.
        let resp = self
            .client
            .get(self.cfg.enrollments_url())
            .header(reqwest::header::ACCEPT, "application/json")
            .bearer_auth(self.bearer()?)
            .send()
            .map_err(|e| Error::Http(e.to_string()))?;

        if Self::is_unsupported(resp.status()) {
            return Ok((pushed, 0));
        }
        if !resp.status().is_success() {
            return Err(Error::Http(format!(
                "enrollment index HTTP {}",
                resp.status()
            )));
        }
        let index: EnrollmentIndexResponse = resp.json().map_err(|e| Error::Http(e.to_string()))?;

        let mut pulled = 0;
        for remote in &index.sinus_enrollments {
            if let Some(parsed) = parse_remote_enrollment(remote) {
                if store.upsert_remote_enrollment(&parsed)? {
                    pulled += 1;
                }
            }
        }

        Ok((pushed, pulled))
    }

    /// Flush all pending events and tombstones from `store`. Accepted/duplicate
    /// events are marked uploaded; rejected are counted and left pending;
    /// confirmed tombstones are purged locally.
    pub fn flush(&self, store: &mut Store) -> Result<SyncOutcome> {
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

        // Flags run after events so a flag on a just-uploaded event finds it
        // server-side rather than reporting `not_found`.
        outcome.flagged = self.flush_flags(store)?;

        outcome.settings_pulled = self.sync_settings(store)?;
        outcome.reload_settings = outcome.settings_pulled > 0;

        let (pushed, pulled) = self.sync_enrollments(store)?;
        outcome.enrollments_pushed = pushed;
        outcome.enrollments_pulled = pulled;
        outcome.reload_enrollments = pulled > 0;

        Ok(outcome)
    }
}

/// Enrollments carry embeddings, so batches are far smaller than the 500-event
/// limit: 500 x 16 KB would be ~10.9 MB of base64, over PHP's default 8 MB
/// `post_max_size`.
const ENROLLMENT_BATCH_SIZE: usize = 100;

fn b64(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn unb64(text: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.decode(text).ok()
}

/// Little-endian f32 bytes — the same encoding the local SQLite BLOB uses, so
/// an embedding makes the whole round trip without a float ever being
/// reformatted.
fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn upload_enrollment(stored: &crate::store::StoredEnrollment, device_id: &str) -> UploadEnrollment {
    let embedding = f32_to_bytes(&stored.enrollment.embedding);
    UploadEnrollment {
        client_enrollment_uuid: b64(&stored.uuid),
        class: stored.enrollment.class.as_str(),
        is_negative: stored.enrollment.is_negative,
        negative_scoped: stored.enrollment.negative_scoped,
        embedding_dim: stored.enrollment.embedding.len(),
        embedding: b64(&embedding),
        model_version: stored.model_version.clone(),
        // Clamped to the ranges the server range-checks. Cosine similarity is an
        // unclamped f32 dot product, so near-identical takes can land a hair
        // outside [-1,1]; a very quiet hop can put peak_dbfs near -180. Neither
        // is worth a rejection that would silently lose the take.
        similarity: stored.similarity.map(|v| v.clamp(-1.0, 1.0)),
        separation: stored.separation.map(|v| v.clamp(-2.0, 2.0)),
        peak_dbfs: stored.peak_dbfs.map(|v| v.clamp(-120.0, 20.0)),
        source_event_uuid: stored.source_event_uuid.clone(),
        device_id: device_id.to_string(),
        captured_at: stored.created_at.clone(),
    }
}

fn parse_remote_enrollment(remote: &DownloadEnrollment) -> Option<crate::store::RemoteEnrollment> {
    let uuid = unb64(&remote.client_enrollment_uuid)?;
    if uuid.len() != 16 {
        return None;
    }
    let class = crate::types::EventType::parse(&remote.class)?;
    let embedding = bytes_to_f32(&unb64(&remote.embedding)?);
    if embedding.is_empty() {
        return None;
    }
    Some(crate::store::RemoteEnrollment {
        uuid,
        class,
        embedding,
        is_negative: remote.is_negative,
        created_at: remote.captured_at.clone(),
        similarity: remote.similarity,
        separation: remote.separation,
        peak_dbfs: remote.peak_dbfs,
        model_version: remote.model_version.clone(),
        source_event_uuid: remote.source_event_uuid.clone(),
        negative_scoped: remote.negative_scoped,
    })
}

fn parse_settings_doc(wire: SettingsDocWire) -> Option<crate::store::SettingsDoc> {
    let updated_at = chrono::DateTime::parse_from_rfc3339(&wire.updated_at)
        .ok()
        .map(|d| d.with_timezone(&Utc))?;
    Some(crate::store::SettingsDoc {
        values: wire.settings,
        updated_at: Some(updated_at),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::InMemoryTokenStore;
    use crate::types::EventType;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    fn sample_event(uuid: &str) -> Event {
        Event {
            uuid: uuid.to_string(),
            event_type: EventType::Cough,
            occurred_at: Utc::now(),
            tz_offset_min: 0,
            duration_ms: 500,
            confidence: 0.6,
            burst_count: 1,
            peak_dbfs: Some(-9.5),
            mean_dbfs: Some(-21.0),
            noise_floor_dbfs: Some(-50.0),
            model_version: "test@0".into(),
            source: Source::DesktopMac,
            device_id: "dev".into(),
            uploaded_at: None,
            deleted: false,
            false_positive_at: None,
            corrected_to: None,
            corrected_at: None,
            reject_count: 0,
            rejected_at: None,
        }
    }

    /// One captured request.
    #[derive(Debug, Clone)]
    struct Captured {
        method: String,
        path: String,
        body: String,
    }

    /// A routing mock server.
    ///
    /// A flush is several requests now (events → flags → settings →
    /// enrollments), so a one-shot mock would deadlock the first step that
    /// followed it. Routes are matched by `"METHOD /path"`; anything unmatched
    /// answers 404, which the engine treats as "this PHR predates the feature"
    /// and skips — exactly the behaviour the older-server tests want.
    struct Mock {
        url: String,
        captured: Arc<Mutex<Vec<Captured>>>,
        shutdown: Arc<AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl Mock {
        fn start(routes: Vec<(&'static str, &'static str)>) -> Mock {
            let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
            let url = format!("http://{}", server.server_addr());
            let captured = Arc::new(Mutex::new(Vec::new()));
            let shutdown = Arc::new(AtomicBool::new(false));

            let routes: std::collections::HashMap<String, String> = routes
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            let sink = Arc::clone(&captured);
            let stop = Arc::clone(&shutdown);

            let handle = std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let Ok(Some(mut req)) =
                        server.recv_timeout(std::time::Duration::from_millis(50))
                    else {
                        continue;
                    };
                    let method = req.method().as_str().to_string();
                    let path = req.url().to_string();
                    let mut body = String::new();
                    req.as_reader().read_to_string(&mut body).ok();

                    let last_body = body.clone();
                    sink.lock().unwrap().push(Captured {
                        method: method.clone(),
                        path: path.clone(),
                        body,
                    });

                    let key = format!("{method} {path}");
                    let (code, payload) = match routes.get(&key) {
                        // The client now checks per-item verdicts, and uuids are
                        // random, so a canned body cannot name them. This
                        // sentinel makes the mock accept whatever it was sent.
                        Some(body) if body == ACCEPT_ALL => (200, accept_all_response(&last_body)),
                        Some(body) => (200, body.clone()),
                        None => (404, "{}".to_string()),
                    };

                    let resp = tiny_http::Response::from_string(payload)
                        .with_status_code(code)
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

            Mock {
                url,
                captured,
                shutdown,
                handle: Some(handle),
            }
        }

        fn requests(&self) -> Vec<Captured> {
            self.captured.lock().unwrap().clone()
        }

        fn body_for(&self, method: &str, path: &str) -> Option<String> {
            self.requests()
                .into_iter()
                .find(|r| r.method == method && r.path == path)
                .map(|r| r.body)
        }
    }

    impl Drop for Mock {
        fn drop(&mut self) {
            self.shutdown.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                handle.join().ok();
            }
        }
    }

    /// Route body meaning "accept every item in the request", since the client
    /// now honours per-item verdicts and the uuids are random.
    const ACCEPT_ALL: &str = "__ACCEPT_ALL__";

    /// Echo an `accepted` verdict for every `client_enrollment_uuid` in a
    /// request body.
    fn accept_all_response(body: &str) -> String {
        let uuids: Vec<String> = serde_json::from_str::<serde_json::Value>(body)
            .ok()
            .and_then(|v| v.get("enrollments").cloned())
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default()
            .iter()
            .filter_map(|e| {
                e.get("client_enrollment_uuid")
                    .and_then(|u| u.as_str())
                    .map(str::to_string)
            })
            .collect();

        let results: Vec<String> = uuids
            .iter()
            .map(|u| format!(r#"{{"uuid":"{u}","status":"accepted"}}"#))
            .collect();
        format!(r#"{{"results":[{}]}}"#, results.join(","))
    }

    const EVENTS_BATCH: &str = "/api/phr/patients/7/respiratory-events/batch";
    const FLAG_BATCH: &str = "/api/phr/patients/7/respiratory-events/flag-batch";
    const SETTINGS: &str = "/api/phr/patients/7/sinus-settings";
    const ENROLLMENTS_BATCH: &str = "/api/phr/patients/7/sinus-enrollments/batch";

    /// Routes that make every non-event step a no-op, for tests focused on the
    /// event path.
    fn quiet_routes() -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "POST /api/phr/patients/7/respiratory-events/flag-batch",
                r#"{"results":[]}"#,
            ),
            (
                "GET /api/phr/patients/7/sinus-settings",
                r#"{"sinus_settings":null}"#,
            ),
            (
                "GET /api/phr/patients/7/sinus-enrollments",
                r#"{"sinus_enrollments":[]}"#,
            ),
        ]
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

    fn engine_for(url: String) -> SyncEngine<InMemoryTokenStore> {
        SyncEngine::for_mode(
            Mode::AutoBatch,
            config(url),
            InMemoryTokenStore::with_token("secret"),
        )
        .unwrap()
    }

    #[test]
    fn flush_uploads_and_marks_accepted_and_duplicate() {
        let mut routes = quiet_routes();
        routes.push((
            "POST /api/phr/patients/7/respiratory-events/batch",
            r#"{"results":[{"uuid":"a","status":"accepted"},{"uuid":"b","status":"duplicate"}]}"#,
        ));
        let mock = Mock::start(routes);

        let mut store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("a")).unwrap();
        store.insert_event(&sample_event("b")).unwrap();

        let outcome = engine_for(mock.url.clone()).flush(&mut store).unwrap();

        assert_eq!(outcome.accepted, 1);
        assert_eq!(outcome.duplicate, 1);
        assert_eq!(outcome.rejected, 0);
        // Both should now be marked uploaded → no longer pending.
        assert!(store.pending_events(10).unwrap().is_empty());

        let sent = mock.body_for("POST", EVENTS_BATCH).unwrap();
        assert!(sent.contains("\"device_id\":\"dev\""));
        assert!(sent.contains("\"client_event_uuid\":\"a\""));
        // Loudness rides along with the metadata.
        assert!(sent.contains("\"peak_dbfs\":-9.5"));
        assert!(sent.contains("\"noise_floor_dbfs\":-50.0"));
        // Privacy: never any audio field.
        assert!(!sent.contains("audio"));
    }

    #[test]
    fn rejected_events_stay_pending() {
        let mut routes = quiet_routes();
        routes.push((
            "POST /api/phr/patients/7/respiratory-events/batch",
            r#"{"results":[{"uuid":"a","status":"rejected","reason":"bad"}]}"#,
        ));
        let mock = Mock::start(routes);

        let mut store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("a")).unwrap();

        let outcome = engine_for(mock.url.clone()).flush(&mut store).unwrap();
        assert_eq!(outcome.rejected, 1);
        assert_eq!(store.pending_events(10).unwrap().len(), 1);
    }

    #[test]
    fn tombstones_sync_as_delete_batch() {
        let mut routes = quiet_routes();
        routes.push(("DELETE /api/phr/patients/7/respiratory-events/batch", "{}"));
        let mock = Mock::start(routes);

        let mut store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("x")).unwrap();
        store.mark_uploaded(&["x".into()], Utc::now()).unwrap();
        store.mark_deleted("x").unwrap();

        let outcome = engine_for(mock.url.clone()).flush(&mut store).unwrap();
        assert_eq!(outcome.deleted, 1);
        assert!(store.get_event("x").unwrap().is_none());

        let sent = mock.body_for("DELETE", EVENTS_BATCH).unwrap();
        assert!(sent.contains("\"uuids\":[\"x\"]"));
    }

    #[test]
    fn missing_token_errors() {
        let mut store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("a")).unwrap();
        let engine = SyncEngine::for_mode(
            Mode::AutoBatch,
            config("http://127.0.0.1:1".into()),
            InMemoryTokenStore::new(),
        )
        .unwrap();
        assert!(matches!(engine.flush(&mut store), Err(Error::Token(_))));
    }

    #[test]
    fn flags_push_current_state_and_clear_the_queue() {
        let mut routes = quiet_routes();
        routes.push((
            "POST /api/phr/patients/7/respiratory-events/batch",
            r#"{"results":[{"uuid":"fp","status":"accepted"},{"uuid":"fix","status":"accepted"}]}"#,
        ));
        let mock = Mock::start(routes);

        let mut store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("fp")).unwrap();
        store.insert_event(&sample_event("fix")).unwrap();
        store.mark_false_positive("fp").unwrap();
        store
            .recharacterize("fix", crate::types::EventType::NoseBlow)
            .unwrap();

        let outcome = engine_for(mock.url.clone()).flush(&mut store).unwrap();

        assert_eq!(outcome.flagged, 2);
        assert!(
            store.pending_flags(10).unwrap().is_empty(),
            "a synced flag must leave the queue"
        );

        let sent = mock.body_for("POST", FLAG_BATCH).unwrap();
        assert!(sent.contains("\"false_positive\":true"));
        assert!(sent.contains("\"corrected_to\":\"nose_blow\""));
    }

    /// Undo has to reach the server. A queue keyed on "is flagged" would drop
    /// the cleared row silently and leave the PHR marked forever.
    #[test]
    fn clearing_a_flag_re_queues_it() {
        let store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("u")).unwrap();

        store.mark_false_positive("u").unwrap();
        let queued = store.pending_flags(10).unwrap();
        assert_eq!(queued.len(), 1);
        assert!(queued[0].false_positive);

        store.mark_flags_synced(&queued).unwrap();
        assert!(store.pending_flags(10).unwrap().is_empty());

        store.clear_flag("u").unwrap();
        let requeued = store.pending_flags(10).unwrap();
        assert_eq!(requeued.len(), 1, "an undo must sync too");
        assert!(!requeued[0].false_positive);
        assert_eq!(requeued[0].corrected_to, None);
    }

    /// `not_found` is terminal. An event the server never accepted would
    /// otherwise have its flag re-sent on every flush forever.
    #[test]
    fn a_flag_for_an_unknown_event_is_not_retried() {
        let mut routes = quiet_routes();
        routes.push((
            "POST /api/phr/patients/7/respiratory-events/flag-batch",
            r#"{"results":[{"uuid":"ghost","status":"not_found"}]}"#,
        ));
        let mock = Mock::start(routes);

        let mut store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("ghost")).unwrap();
        store.mark_uploaded(&["ghost".into()], Utc::now()).unwrap();
        store.mark_false_positive("ghost").unwrap();

        engine_for(mock.url.clone()).flush(&mut store).unwrap();
        assert!(store.pending_flags(10).unwrap().is_empty());
    }

    #[test]
    fn local_settings_are_pushed_when_newer() {
        let mut routes = quiet_routes();
        routes.push((
            "PUT /api/phr/patients/7/sinus-settings",
            r#"{"applied":true,"sinus_settings":{"settings":{"sensitivity":"0.8"},"updated_at":"2026-07-01T00:00:00Z"}}"#,
        ));
        let mock = Mock::start(routes);

        let mut store = Store::open_in_memory().unwrap();
        store.setting_set("sensitivity", "0.8").unwrap();

        engine_for(mock.url.clone()).flush(&mut store).unwrap();

        let sent = mock.body_for("PUT", SETTINGS).unwrap();
        assert!(sent.contains("\"sensitivity\":\"0.8\""));
        assert!(sent.contains("\"device_id\":\"dev-1\""));
    }

    #[test]
    fn a_newer_server_document_is_adopted() {
        let mut routes = quiet_routes();
        routes.retain(|(key, _)| !key.starts_with("GET /api/phr/patients/7/sinus-settings"));
        routes.push((
            "GET /api/phr/patients/7/sinus-settings",
            r#"{"sinus_settings":{"settings":{"sensitivity":"0.9"},"updated_at":"2999-01-01T00:00:00Z"}}"#,
        ));
        let mock = Mock::start(routes);

        let mut store = Store::open_in_memory().unwrap();
        store.setting_set("sensitivity", "0.2").unwrap();

        let outcome = engine_for(mock.url.clone()).flush(&mut store).unwrap();

        assert_eq!(outcome.settings_pulled, 1);
        assert!(outcome.reload_settings, "capture must re-read sensitivity");
        assert_eq!(
            store.setting_get("sensitivity").unwrap().as_deref(),
            Some("0.9")
        );
        // No PUT: the server was newer, so there was nothing to push.
        assert!(mock.body_for("PUT", SETTINGS).is_none());
    }

    /// Adopting a server value must not look like a fresh local edit, or the
    /// two machines ping-pong the same document forever.
    #[test]
    fn adopting_a_server_value_does_not_re_push_it() {
        let mut store = Store::open_in_memory().unwrap();
        store.setting_set("sensitivity", "0.2").unwrap();

        let server_time = Utc::now() + chrono::Duration::hours(1);
        let doc = crate::store::SettingsDoc {
            values: [("sensitivity".to_string(), "0.9".to_string())]
                .into_iter()
                .collect(),
            updated_at: Some(server_time),
        };
        store.apply_settings_doc(&doc).unwrap();

        let after = store.settings_doc().unwrap();
        assert_eq!(after.updated_at, Some(server_time));
    }

    /// The UI thread can write a setting between the sync thread's GET and its
    /// apply; that edit must survive.
    #[test]
    fn a_newer_local_edit_is_not_clobbered_by_the_server() {
        let mut store = Store::open_in_memory().unwrap();
        let doc = crate::store::SettingsDoc {
            values: [("sensitivity".to_string(), "0.9".to_string())]
                .into_iter()
                .collect(),
            updated_at: Some(Utc::now() - chrono::Duration::hours(1)),
        };

        store.setting_set("sensitivity", "0.3").unwrap();
        let changed = store.apply_settings_doc(&doc).unwrap();

        assert!(changed.is_empty());
        assert_eq!(
            store.setting_get("sensitivity").unwrap().as_deref(),
            Some("0.3")
        );
    }

    #[test]
    fn enrollments_push_and_pull_with_byte_exact_embeddings() {
        let embedding = vec![0.5f32, -0.25, 0.125, -1.0];
        let remote_uuid = b64(&[7u8; 16]);
        let remote_embedding = b64(&f32_to_bytes(&embedding));
        let index_body = format!(
            r#"{{"sinus_enrollments":[{{"client_enrollment_uuid":"{remote_uuid}","class":"sniffle","is_negative":false,"embedding":"{remote_embedding}","embedding_dim":4,"captured_at":"2026-07-01T00:00:00Z"}}]}}"#
        );
        let index_body: &'static str = Box::leak(index_body.into_boxed_str());

        let mut routes = quiet_routes();
        routes.retain(|(key, _)| !key.starts_with("GET /api/phr/patients/7/sinus-enrollments"));
        routes.push(("GET /api/phr/patients/7/sinus-enrollments", index_body));
        routes.push((
            "POST /api/phr/patients/7/sinus-enrollments/batch",
            ACCEPT_ALL,
        ));
        let mock = Mock::start(routes);

        let mut store = Store::open_in_memory().unwrap();
        store
            .add_enrollment(crate::types::EventType::Hawk, &embedding, false)
            .unwrap();

        let outcome = engine_for(mock.url.clone()).flush(&mut store).unwrap();

        assert_eq!(outcome.enrollments_pushed, 1);
        assert_eq!(outcome.enrollments_pulled, 1);
        assert!(
            outcome.reload_enrollments,
            "a pulled take must rebuild the matcher, or a fresh machine keeps detecting as if untrained"
        );

        // Pushed embedding survives as bytes, not reformatted floats.
        let sent = mock.body_for("POST", ENROLLMENTS_BATCH).unwrap();
        assert!(sent.contains(&remote_embedding));

        // Pulled example landed with its vector intact.
        let all = store.enrollments().unwrap();
        assert_eq!(all.len(), 2);
        let pulled = all
            .iter()
            .find(|e| e.enrollment.class == crate::types::EventType::Sniffle)
            .expect("pulled enrollment");
        assert_eq!(pulled.enrollment.embedding, embedding);

        // A second flush must not re-import it.
        let again = engine_for(mock.url.clone()).flush(&mut store).unwrap();
        assert_eq!(again.enrollments_pulled, 0);
        let _ = mock.requests();
    }

    /// A PHR that predates these endpoints answers 404. That must be a skip, not
    /// a sync failure that trips the backoff and stalls event upload.
    #[test]
    fn an_older_server_is_skipped_rather_than_failing_the_flush() {
        // Only the events route exists.
        let mock = Mock::start(vec![(
            "POST /api/phr/patients/7/respiratory-events/batch",
            r#"{"results":[{"uuid":"a","status":"accepted"}]}"#,
        )]);

        let mut store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("a")).unwrap();
        store.setting_set("sensitivity", "0.7").unwrap();
        store
            .add_enrollment(crate::types::EventType::Hawk, &[0.1, 0.2], false)
            .unwrap();

        let outcome = engine_for(mock.url.clone()).flush(&mut store).unwrap();

        assert_eq!(outcome.accepted, 1, "events must still upload");
        assert_eq!(outcome.settings_pulled, 0);
        assert_eq!(outcome.enrollments_pulled, 0);
    }

    #[test]
    fn complete_batch_endpoint_is_used_verbatim() {
        let endpoint =
            "https://example.test/api/phr/patients/1/respiratory-events/batch".to_string();
        assert_eq!(config(endpoint.clone()).batch_url(), endpoint);
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
