//! UniFFI boundary for native Apple clients.
//!
//! Swift owns capture, lifecycle, Core ML, and presentation. Rust accepts
//! converted 16 kHz mono PCM and owns the detector, persistence, and projections.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use sinus_app::instance::InstanceGuard;
use sinus_app::monitor::{MonitoringConfig, MonitoringEngine};
use sinus_app::sync::{FlushPolicy, SyncDriver, TickInput};
use sinus_core::classify::embed::{Embedder, WindowFeatures, AUDIOSET_CLASSES, EMBED_DIM};
use sinus_core::error::{Error as CoreError, Result as CoreResult};
use sinus_core::mel::MelPatch;
use sinus_core::store::Store;
use sinus_core::token::TokenStore;
use sinus_core::types::{Event, EventType, Source};
use thiserror::Error;

uniffi::setup_scaffolding!();

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum ApplePlatform {
    Macos,
    Ios,
}

impl ApplePlatform {
    fn source(self) -> Source {
        match self {
            Self::Macos => Source::DesktopMac,
            Self::Ios => Source::MobileIos,
        }
    }
}

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum AppleEventType {
    Cough,
    ThroatClearing,
    Sniffle,
    Sneeze,
    NoseBlow,
    Hawk,
    SnortSuck,
}

impl From<EventType> for AppleEventType {
    fn from(value: EventType) -> Self {
        match value {
            EventType::Cough => Self::Cough,
            EventType::ThroatClearing => Self::ThroatClearing,
            EventType::Sniffle => Self::Sniffle,
            EventType::Sneeze => Self::Sneeze,
            EventType::NoseBlow => Self::NoseBlow,
            EventType::Hawk => Self::Hawk,
            EventType::SnortSuck => Self::SnortSuck,
        }
    }
}

impl From<AppleEventType> for EventType {
    fn from(value: AppleEventType) -> Self {
        match value {
            AppleEventType::Cough => Self::Cough,
            AppleEventType::ThroatClearing => Self::ThroatClearing,
            AppleEventType::Sniffle => Self::Sniffle,
            AppleEventType::Sneeze => Self::Sneeze,
            AppleEventType::NoseBlow => Self::NoseBlow,
            AppleEventType::Hawk => Self::Hawk,
            AppleEventType::SnortSuck => Self::SnortSuck,
        }
    }
}

/// Only what the platform knows. The device identity, sensitivity and battery
/// policy all live in the database, which this machine's other Sinus Sentinel
/// shell reads too.
#[derive(Debug, Clone, uniffi::Record)]
pub struct AppleEngineConfig {
    pub database_path: String,
    pub platform: ApplePlatform,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelOutput {
    pub audioset_scores: Vec<f32>,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Error, uniffi::Error)]
pub enum ModelError {
    #[error("model inference failed: {message}")]
    Failed { message: String },
}

impl From<uniffi::UnexpectedUniFFICallbackError> for ModelError {
    fn from(value: uniffi::UnexpectedUniFFICallbackError) -> Self {
        Self::Failed {
            message: value.reason,
        }
    }
}

/// Implemented in Swift with Core ML. Calls happen only for gate-active patches,
/// not continuously for quiet audio.
#[uniffi::export(foreign)]
pub trait ModelRunner: Send + Sync {
    fn model_version(&self) -> Result<String, ModelError>;

    fn infer(
        &self,
        log_mel: Vec<f32>,
        frames: u32,
        bands: u32,
        energy_peak: bool,
    ) -> Result<ModelOutput, ModelError>;
}

struct ForeignEmbedder {
    runner: Arc<dyn ModelRunner>,
    version: String,
}

impl fmt::Debug for ForeignEmbedder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ForeignEmbedder")
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl Embedder for ForeignEmbedder {
    fn model_version(&self) -> String {
        self.version.clone()
    }

    fn embed(&self, patch: &MelPatch, energy_peak: bool) -> CoreResult<WindowFeatures> {
        let output = self
            .runner
            .infer(
                patch.data.clone(),
                patch.frames as u32,
                patch.bands as u32,
                energy_peak,
            )
            .map_err(|error| CoreError::ModelUnavailable(error.to_string()))?;
        if output.audioset_scores.len() != AUDIOSET_CLASSES {
            return Err(CoreError::ModelUnavailable(format!(
                "Core ML returned {} scores; expected {AUDIOSET_CLASSES}",
                output.audioset_scores.len()
            )));
        }
        if output.embedding.len() != EMBED_DIM {
            return Err(CoreError::ModelUnavailable(format!(
                "Core ML returned an {}-value embedding; expected {EMBED_DIM}",
                output.embedding.len()
            )));
        }
        Ok(WindowFeatures {
            audioset_scores: Some(output.audioset_scores),
            embedding: output.embedding,
            energy_peak,
        })
    }
}

#[derive(Debug, Error, uniffi::Error)]
pub enum TokenError {
    #[error("keychain failure: {message}")]
    Keychain { message: String },
}

impl From<uniffi::UnexpectedUniFFICallbackError> for TokenError {
    fn from(value: uniffi::UnexpectedUniFFICallbackError) -> Self {
        Self::Keychain {
            message: value.reason,
        }
    }
}

/// Implemented in Swift over Security.framework. Rust never stores the PHR
/// bearer token itself on Apple platforms — the Keychain is the only copy, and
/// it is bound to the app's code signature rather than to a file Rust could read.
#[uniffi::export(foreign)]
pub trait TokenProvider: Send + Sync {
    fn get_token(&self) -> Result<Option<String>, TokenError>;
    fn set_token(&self, token: String) -> Result<(), TokenError>;
    fn clear_token(&self) -> Result<(), TokenError>;
}

/// Adapts a Swift [`TokenProvider`] to [`sinus_core::token::TokenStore`].
///
/// `SyncEngine::bearer()` is called once per HTTP request, and a flush makes
/// up to five of those; without a cache, each one becomes a Keychain round
/// trip, turning a backoff storm into a Keychain-prompt storm. This is the
/// same reason `KeyringTokenStore` caches (see `crates/core/src/token.rs`) —
/// mirrored here because the source of truth on Apple platforms is a Swift
/// callback rather than a local `keyring::Entry`.
struct ForeignTokenStore {
    provider: Arc<dyn TokenProvider>,
    /// Outer `None` means "not yet read from the provider"; `Some(None)` is a
    /// cached "provider has no token" answer.
    cached: Mutex<Option<Option<String>>>,
}

impl ForeignTokenStore {
    fn new(provider: Arc<dyn TokenProvider>) -> Self {
        ForeignTokenStore {
            provider,
            cached: Mutex::new(None),
        }
    }
}

impl TokenStore for ForeignTokenStore {
    fn get_token(&self) -> CoreResult<Option<String>> {
        // A poisoned cache is not fatal: the token is still readable straight
        // from the provider, and failing a sync because a lock was poisoned
        // would be a worse outcome than a redundant Keychain read.
        match self.cached.lock() {
            Ok(mut cache) => {
                if let Some(token) = cache.as_ref() {
                    return Ok(token.clone());
                }
                let token = self
                    .provider
                    .get_token()
                    .map_err(|error| CoreError::Token(error.to_string()))?;
                *cache = Some(token.clone());
                Ok(token)
            }
            Err(_) => self
                .provider
                .get_token()
                .map_err(|error| CoreError::Token(error.to_string())),
        }
    }

    fn set_token(&self, token: &str) -> CoreResult<()> {
        self.provider
            .set_token(token.to_string())
            .map_err(|error| CoreError::Token(error.to_string()))?;
        // Only updated once the provider has actually accepted the write, so a
        // failed write cannot leave the cache claiming a token it never stored.
        if let Ok(mut cache) = self.cached.lock() {
            *cache = Some(Some(token.to_string()));
        }
        Ok(())
    }

    fn clear(&self) -> CoreResult<()> {
        self.provider
            .clear_token()
            .map_err(|error| CoreError::Token(error.to_string()))?;
        if let Ok(mut cache) = self.cached.lock() {
            *cache = Some(None);
        }
        Ok(())
    }
}

#[derive(Debug, Error, uniffi::Error)]
pub enum AppleEngineError {
    #[error("invalid argument: {message}")]
    InvalidArgument { message: String },
    #[error("engine failure: {message}")]
    Engine { message: String },
    #[error("model failure: {message}")]
    Model { message: String },
    /// Another Sinus Sentinel — the menu-bar app, or a second copy of this one —
    /// already owns this machine's database. Two detectors on one microphone
    /// would log every event twice, so the second one refuses to start.
    #[error("another Sinus Sentinel is already running on this computer")]
    AlreadyRunning,
    /// The event uuid the UI holds no longer exists — a stale list, or a row the
    /// PHR sync removed between render and tap.
    #[error("no such event: {uuid}")]
    NotFound { uuid: String },
}

impl From<CoreError> for AppleEngineError {
    fn from(value: CoreError) -> Self {
        Self::Engine {
            message: value.to_string(),
        }
    }
}

/// `event_type` is the effective type — what every list should render by
/// default, and what a correction overrides. `original_event_type` and
/// `corrected_to` are the audit trail behind it: what the classifier said,
/// and, if the user corrected it, what they said instead. A UI wanting to
/// render "Sniffle (was Cough)" needs both.
#[derive(Debug, Clone, uniffi::Record)]
pub struct AppleEvent {
    pub uuid: String,
    pub event_type: AppleEventType,
    /// What the classifier originally decided, before any correction.
    pub original_event_type: AppleEventType,
    /// What the user corrected the event to, if they did.
    pub corrected_to: Option<AppleEventType>,
    pub occurred_at_epoch_ms: i64,
    pub timezone_offset_minutes: i32,
    pub duration_ms: i64,
    pub confidence: f32,
    pub burst_count: i64,
    pub peak_dbfs: Option<f32>,
    pub mean_dbfs: Option<f32>,
    pub noise_floor_dbfs: Option<f32>,
    pub model_version: String,
    pub false_positive: bool,
}

impl From<Event> for AppleEvent {
    fn from(value: Event) -> Self {
        let event_type = value.effective_type().into();
        let original_event_type = value.event_type.into();
        let corrected_to = value.corrected_to.map(Into::into);
        Self {
            uuid: value.uuid,
            event_type,
            original_event_type,
            corrected_to,
            occurred_at_epoch_ms: value.occurred_at.timestamp_millis(),
            timezone_offset_minutes: value.tz_offset_min,
            duration_ms: value.duration_ms,
            confidence: value.confidence,
            burst_count: value.burst_count,
            peak_dbfs: value.peak_dbfs,
            mean_dbfs: value.mean_dbfs,
            noise_floor_dbfs: value.noise_floor_dbfs,
            model_version: value.model_version,
            false_positive: value.false_positive_at.is_some(),
        }
    }
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct EventCount {
    pub event_type: AppleEventType,
    pub count: u64,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct DayBucket {
    pub date_iso8601: String,
    pub counts: Vec<EventCount>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct HistorySnapshot {
    pub today: Vec<EventCount>,
    pub days: Vec<DayBucket>,
    pub recent_events: Vec<AppleEvent>,
    pub congestion_score_per_monitored_hour: f64,
    /// Hours elapsed since local midnight, unclamped — so a UI can say "3.2
    /// monitored hours" next to the score instead of showing a rate with no
    /// denominator. `congestion_score_per_monitored_hour` divides by a clamped
    /// version of this internally (to stay finite moments after midnight); this
    /// field reports the real elapsed time, including a true zero.
    pub monitored_hours: f64,
}

/// Samples of lead-in the shell should discard before it starts buffering a
/// take, exported so Swift's countdown and Rust's expectations cannot drift.
#[uniffi::export]
pub fn teach_countdown_samples() -> u32 {
    sinus_app::teach::TEACH_COUNTDOWN_SAMPLES as u32
}

/// Samples one take must contain. Swift buffers exactly this many before calling
/// `enroll_take`.
#[uniffi::export]
pub fn teach_take_samples() -> u32 {
    sinus_app::teach::TEACH_TAKE_SAMPLES as u32
}

/// Takes a class needs before it can fire on its own — what "2 more takes"
/// counts down to in the UI.
#[uniffi::export]
pub fn teach_min_takes() -> u32 {
    sinus_app::teach::MIN_TAKES as u32
}

/// Where a class stands in Settings. Mirrors `sinus_app::teach::ClassStatus`.
#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum TrainingStatus {
    /// No takes recorded.
    Untrained,
    /// Some takes, but fewer than `teach_min_takes()`; `needed` more to go.
    Inactive { needed: u32 },
    /// Enough takes to fire, but the most recent one scored poorly.
    Active,
    /// Enough takes, and the latest one was clean.
    Ready,
}

impl From<sinus_app::teach::ClassStatus> for TrainingStatus {
    fn from(value: sinus_app::teach::ClassStatus) -> Self {
        match value {
            sinus_app::teach::ClassStatus::Untrained => Self::Untrained,
            sinus_app::teach::ClassStatus::Inactive { needed } => Self::Inactive {
                needed: needed as u32,
            },
            sinus_app::teach::ClassStatus::Active => Self::Active,
            sinus_app::teach::ClassStatus::Ready => Self::Ready,
        }
    }
}

/// One recorded take, as the Training list renders it.
#[derive(Debug, Clone, uniffi::Record)]
pub struct TeachTake {
    /// Row id — the handle `delete_take` takes.
    pub id: i64,
    /// RFC3339, as stored.
    pub created_at: String,
    /// Similarity to the class's existing prototype when this take was recorded.
    /// `None` for a class's first take, which had nothing to score against.
    pub similarity: Option<f32>,
    /// Same-class similarity minus the closest other class. `None` alongside `similarity`.
    pub separation: Option<f32>,
    pub peak_dbfs: Option<f32>,
    pub model_version: Option<String>,
    /// Whether the PHR has this example.
    pub synced: bool,
}

impl From<sinus_core::store::StoredEnrollment> for TeachTake {
    fn from(value: sinus_core::store::StoredEnrollment) -> Self {
        Self {
            id: value.id,
            created_at: value.created_at,
            similarity: value.similarity,
            separation: value.separation,
            peak_dbfs: value.peak_dbfs,
            model_version: value.model_version,
            synced: value.synced,
        }
    }
}

/// One class's training, in `EventType::ALL` order.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ClassTraining {
    pub event_type: AppleEventType,
    pub status: TrainingStatus,
    /// Positive takes, oldest first, so the last entry is the newest.
    pub takes: Vec<TeachTake>,
}

/// The whole Training pane in one read.
#[derive(Debug, Clone, uniffi::Record)]
pub struct TrainingSnapshot {
    pub classes: Vec<ClassTraining>,
    /// Learned false-positive suppressions, across all classes.
    pub negative_count: u32,
}

/// The outcome of enrolling one take.
#[derive(Debug, Clone, uniffi::Record)]
pub struct TeachResult {
    pub event_type: AppleEventType,
    /// Total positive takes for this class, including the one just recorded.
    pub examples: u32,
    /// Negative when this was the class's first take — nothing existed to score against.
    pub similarity: f32,
    pub separation: f32,
    pub peak_dbfs: Option<f32>,
    /// Whether this take alone clears the bar `status` applies. Computed by
    /// `sinus_app::teach::TeachResult::is_good` so the two cannot disagree.
    pub good: bool,
}

impl From<sinus_app::teach::TeachResult> for TeachResult {
    fn from(value: sinus_app::teach::TeachResult) -> Self {
        let good = value.is_good();
        Self {
            event_type: value.class.into(),
            examples: value.examples as u32,
            similarity: value.similarity,
            separation: value.separation,
            peak_dbfs: value.peak_dbfs,
            good,
        }
    }
}

/// What a flag operation changed, as the UI needs to see it.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FlagResult {
    /// The event as it now stands, so the caller can replace its row without refetching.
    pub event: AppleEvent,
    /// An enrollment was written, so the detector actually changed. False when
    /// the event's embedding had already been pruned — worth telling the user,
    /// since the flag alone will not stop the sound recurring.
    pub trained: bool,
}

/// How aggressively this device uploads. Device-local: a laptop on metered
/// wifi and a desktop on ethernet reasonably want different answers for the
/// same patient. Mirrors `sinus_core::sync::Mode`.
#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum SyncMode {
    AutoBatch,
    OfflineFirst,
    OfflineStrict,
}

impl From<sinus_core::sync::Mode> for SyncMode {
    fn from(value: sinus_core::sync::Mode) -> Self {
        match value {
            sinus_core::sync::Mode::AutoBatch => Self::AutoBatch,
            sinus_core::sync::Mode::OfflineFirst => Self::OfflineFirst,
            sinus_core::sync::Mode::OfflineStrict => Self::OfflineStrict,
        }
    }
}

impl From<SyncMode> for sinus_core::sync::Mode {
    fn from(value: SyncMode) -> Self {
        match value {
            SyncMode::AutoBatch => Self::AutoBatch,
            SyncMode::OfflineFirst => Self::OfflineFirst,
            SyncMode::OfflineStrict => Self::OfflineStrict,
        }
    }
}

/// A local-time window during which detections are not recorded. Both bounds
/// are hours, 0–23. `start == end` is not a zero-length window but "no window",
/// which is why the accessors use `Option` rather than encoding that here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Record)]
pub struct QuietHours {
    pub start_hour: u32,
    pub end_hour: u32,
}

/// Whether capture is suspended, and until when.
#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum PauseKind {
    Running,
    /// Paused until an absolute instant; `until_epoch_ms` on the snapshot says when.
    Timed,
    /// Paused until the user resumes.
    Indefinite,
}

#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct PauseSnapshot {
    pub kind: PauseKind,
    /// Set only for `Timed`.
    pub until_epoch_ms: Option<i64>,
    /// Whether the pause is in force *now* — a timed pause whose deadline has
    /// passed reports `Timed` with a past deadline but `paused == false`, so a
    /// UI need not re-derive expiry.
    pub paused: bool,
}

/// Converts stored pause state into the wire snapshot. `is_paused` and the
/// variant are derived from the same `state` and `now`, so the two fields on
/// `PauseSnapshot` cannot disagree with each other the way two independent FFI
/// calls taken moments apart could.
fn pause_snapshot(state: sinus_app::state::PauseState, now: DateTime<Utc>) -> PauseSnapshot {
    let paused = state.is_paused(now);
    match state {
        sinus_app::state::PauseState::Running => PauseSnapshot {
            kind: PauseKind::Running,
            until_epoch_ms: None,
            paused,
        },
        sinus_app::state::PauseState::PausedIndefinite => PauseSnapshot {
            kind: PauseKind::Indefinite,
            until_epoch_ms: None,
            paused,
        },
        sinus_app::state::PauseState::PausedUntil(until) => PauseSnapshot {
            kind: PauseKind::Timed,
            until_epoch_ms: Some(until.timestamp_millis()),
            paused,
        },
    }
}

/// Everything the PHR pane shows except the token, which never crosses this
/// boundary — Swift owns it in the Keychain.
#[derive(Debug, Clone, uniffi::Record)]
pub struct PhrSettings {
    pub server_url: String,
    pub patient_id: Option<i64>,
    pub mode: SyncMode,
    /// This machine's stable identity in the PHR.
    pub device_id: String,
}

/// Everything the menu bar renders, in one read, so a UI refresh is one FFI
/// call rather than six.
#[derive(Debug, Clone, uniffi::Record)]
pub struct EngineStatus {
    pub monitoring: bool,
    /// The energy gate is open — a sound is arriving and is being classified.
    /// Drives the "heard something — classifying…" indicator.
    pub gate_open: bool,
    /// When the gate last opened, for a UI that wants "last heard 4s ago"
    /// rather than a flicker. `None` if it has not opened this session.
    pub last_heard_epoch_ms: Option<i64>,
    pub sensitivity: f32,
    pub pause: PauseSnapshot,
    pub quiet_hours: Option<QuietHours>,
    /// Whether *now* falls inside the quiet-hours window.
    pub in_quiet_hours: bool,
    pub pause_on_low_power: bool,
    pub model_version: String,
}

/// How many events the "recent" list carries across the FFI boundary. The UI
/// shows a handful; fetching a whole week would copy every row on every refresh.
const RECENT_EVENT_LIMIT: usize = 50;

#[derive(uniffi::Object)]
pub struct AppleEngine {
    inner: Mutex<MonitoringEngine<ForeignEmbedder>>,
    /// A second, read-only connection to the same database. Projections are read
    /// from the UI thread while `inner` can be held for the length of a Core ML
    /// inference on the audio queue; sharing one lock would stall the UI behind
    /// the model. SQLite's WAL mode is built for exactly this — concurrent
    /// readers never block on the writer.
    reader: Mutex<Store>,
    /// Published from `push_pcm_16k` so the UI can poll the gate without touching
    /// the writer mutex, which the audio thread can hold across a Core ML
    /// inference. A frame-late answer is fine for an indicator; a stalled UI is not.
    ///
    /// The same reasoning covers `monitoring` and `model_version`: between them
    /// these four fields are the whole of `status()` that does not come from the
    /// read connection, which is what lets a UI poll several times a second
    /// without ever contending with the audio thread.
    gate_open: AtomicBool,
    /// Epoch milliseconds when the gate was last observed open; 0 for "never".
    last_heard_epoch_ms: AtomicI64,
    /// Mirrors `MonitoringEngine::is_monitoring`, updated on the transitions.
    monitoring: AtomicBool,
    /// Fixed for the engine's lifetime — the embedder is constructed once, from
    /// one Core ML model — so it needs no lock and no interior mutability.
    model_version: String,
    /// Held for the engine's lifetime. Dropping it — including on a crash, since
    /// the OS owns the underlying file lock — hands this machine back to whichever
    /// shell the user opens next.
    instance: InstanceGuard,
    /// Kept so `SyncController::new` can open its own connection to the same
    /// database without Swift threading the path through a second time.
    database_path: String,
    /// Kept so `SyncController::new` uploads under the right `Source` — an
    /// iPhone must not report as `desktop-mac`.
    platform: ApplePlatform,
}

#[uniffi::export]
impl AppleEngine {
    #[uniffi::constructor]
    pub fn new(
        config: AppleEngineConfig,
        model: Arc<dyn ModelRunner>,
    ) -> Result<Arc<Self>, AppleEngineError> {
        if config.database_path.trim().is_empty() {
            return Err(AppleEngineError::InvalidArgument {
                message: "database_path cannot be empty".to_string(),
            });
        }
        let database_path = std::path::Path::new(&config.database_path);
        let data_dir = database_path
            .parent()
            .ok_or_else(|| AppleEngineError::InvalidArgument {
                message: "database_path must name a file inside a data directory".to_string(),
            })?;
        // Claim the machine before touching the microphone or the database, so a
        // refusal costs nothing and leaves no partially-initialized state.
        let instance = InstanceGuard::try_acquire(data_dir)
            .map_err(|error| AppleEngineError::Engine {
                message: format!("could not establish single-instance ownership: {error}"),
            })?
            .ok_or(AppleEngineError::AlreadyRunning)?;

        let version = model
            .model_version()
            .map_err(|error| AppleEngineError::Model {
                message: error.to_string(),
            })?;
        let embedder = ForeignEmbedder {
            runner: model,
            version: version.clone(),
        };
        let engine = MonitoringEngine::open(
            &config.database_path,
            embedder,
            MonitoringConfig::new(config.platform.source()),
        )?;
        let reader = Store::open(&config.database_path)?;
        let database_path = config.database_path.clone();
        Ok(Arc::new(Self {
            inner: Mutex::new(engine),
            reader: Mutex::new(reader),
            gate_open: AtomicBool::new(false),
            last_heard_epoch_ms: AtomicI64::new(0),
            monitoring: AtomicBool::new(false),
            model_version: version,
            instance,
            database_path,
            platform: config.platform,
        }))
    }

    pub fn start_monitoring(
        &self,
        started_at_epoch_ms: i64,
        timezone_offset_minutes: i32,
    ) -> Result<(), AppleEngineError> {
        let started_at = timestamp(started_at_epoch_ms)?;
        check_timezone_offset(timezone_offset_minutes)?;
        self.lock()?
            .start_session(started_at, timezone_offset_minutes);
        self.gate_open.store(false, Ordering::Relaxed);
        self.monitoring.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub fn push_pcm_16k(&self, samples: Vec<f32>) -> Result<Vec<AppleEvent>, AppleEngineError> {
        let mut engine = self.lock()?;
        let events = engine.push_pcm_16k(&samples)?;
        // Read while still holding the lock we already have, rather than taking
        // it a second time: the audio thread calls this several times a second,
        // and a second acquisition would be a needless contention point against
        // any UI thread waiting on `status()`.
        let gate_open = engine.gate_open();
        self.gate_open.store(gate_open, Ordering::Relaxed);
        if gate_open {
            self.last_heard_epoch_ms
                .store(Utc::now().timestamp_millis(), Ordering::Relaxed);
        }
        Ok(events.into_iter().map(AppleEvent::from).collect())
    }

    pub fn stop_monitoring(&self) -> Result<Vec<AppleEvent>, AppleEngineError> {
        let mut engine = self.lock()?;
        let stopped = engine.stop_session();
        // Published from what the engine actually did, not from the fact that
        // stop was called: a failed flush leaves the session open, and an
        // indicator that claimed otherwise would be lying about a live mic.
        self.monitoring
            .store(engine.is_monitoring(), Ordering::Relaxed);
        // `last_heard_epoch_ms` is left untouched: it is "last heard this
        // session or before", not "last heard while monitoring", so a UI
        // checked after the mic closes can still show how long ago the last
        // sound arrived instead of the indicator vanishing.
        self.gate_open.store(false, Ordering::Relaxed);
        Ok(stopped?.into_iter().map(AppleEvent::from).collect())
    }

    /// Reads the published mirror rather than the engine, so a UI asking while a
    /// Core ML inference is in flight gets an answer instead of a stall.
    pub fn is_monitoring(&self) -> Result<bool, AppleEngineError> {
        Ok(self.monitoring.load(Ordering::Relaxed))
    }

    pub fn set_sensitivity(&self, sensitivity: f32) -> Result<(), AppleEngineError> {
        if !sensitivity.is_finite() {
            return Err(AppleEngineError::InvalidArgument {
                message: "sensitivity must be finite".to_string(),
            });
        }
        self.lock()?.set_sensitivity(sensitivity)?;
        Ok(())
    }

    pub fn history(
        &self,
        days: u32,
        now_epoch_ms: i64,
        timezone_offset_minutes: i32,
    ) -> Result<HistorySnapshot, AppleEngineError> {
        if !(1..=90).contains(&days) {
            return Err(AppleEngineError::InvalidArgument {
                message: "history days must be between 1 and 90".to_string(),
            });
        }
        check_timezone_offset(timezone_offset_minutes)?;
        let now = timestamp(now_epoch_ms)?;
        let store = self.reader()?;
        let today = sinus_app::state::today_counts_at_offset(&store, now, timezone_offset_minutes);
        let histogram = sinus_app::state::daily_histogram_at_offset(
            &store,
            days as i64,
            now,
            timezone_offset_minutes,
        );
        let recent_events = store
            .recent_events_limited(
                now - chrono::Duration::days(days as i64),
                now,
                RECENT_EVENT_LIMIT,
            )?
            .into_iter()
            .map(AppleEvent::from)
            .collect();
        let monitored_hours = (now
            - sinus_app::state::local_midnight_at_offset(now, timezone_offset_minutes))
        .num_minutes() as f64
            / 60.0;

        Ok(HistorySnapshot {
            today: event_counts(&today),
            days: histogram
                .into_iter()
                .map(|day| DayBucket {
                    date_iso8601: day.date.to_string(),
                    counts: event_counts(&day.counts),
                })
                .collect(),
            recent_events,
            congestion_score_per_monitored_hour: sinus_app::state::congestion_score(
                &today,
                monitored_hours.max(0.1),
            ),
            monitored_hours,
        })
    }

    /// Whether the shell should release the microphone while the OS reports Low
    /// Power Mode. Same row, same default as the desktop tray's battery policy,
    /// so a Mac running both shells behaves consistently.
    pub fn pause_on_low_power(&self) -> Result<bool, AppleEngineError> {
        let store = self.reader()?;
        Ok(sinus_app::settings::pause_on_low_power(&store))
    }

    pub fn set_pause_on_low_power(&self, enabled: bool) -> Result<(), AppleEngineError> {
        sinus_app::settings::set_pause_on_low_power(self.lock()?.store(), enabled)?;
        Ok(())
    }

    pub fn get_setting(&self, key: String) -> Result<Option<String>, AppleEngineError> {
        Ok(self.reader()?.setting_get(&key)?)
    }

    pub fn set_setting(&self, key: String, value: String) -> Result<(), AppleEngineError> {
        self.lock()?.store().setting_set(&key, &value)?;
        Ok(())
    }

    /// Report a misdetection: the event stops counting here and in the PHR, and,
    /// if its embedding was retained, the detector is trained not to call that
    /// sound the class it fired as. Policy lives in `sinus_app::flag`; see there
    /// for the rules.
    pub fn report_false_positive(
        &self,
        event_uuid: String,
    ) -> Result<FlagResult, AppleEngineError> {
        self.flag(event_uuid, |store, uuid| {
            sinus_app::flag::report_false_positive(store, uuid)
        })
    }

    /// Record what a misdetected sound actually was. Correcting an event back to
    /// the class the classifier originally fired is treated as an undo, not a
    /// correction — see `sinus_app::flag::recharacterize`.
    pub fn recharacterize(
        &self,
        event_uuid: String,
        corrected: AppleEventType,
    ) -> Result<FlagResult, AppleEngineError> {
        let corrected: EventType = corrected.into();
        self.flag(event_uuid, move |store, uuid| {
            sinus_app::flag::recharacterize(store, uuid, corrected)
        })
    }

    /// Undo a false-positive report or a correction. Any training the flag
    /// produced is kept; only the flag on this event is reverted.
    pub fn clear_flag(&self, event_uuid: String) -> Result<FlagResult, AppleEngineError> {
        self.flag(event_uuid, |store, uuid| {
            sinus_app::flag::clear_flag(store, uuid)
        })
    }

    /// The whole Training pane. Read through the read connection, not the writer,
    /// so refreshing Settings cannot queue behind a Core ML inference.
    pub fn training(&self) -> Result<TrainingSnapshot, AppleEngineError> {
        let store = self.reader()?;
        let (classes, negative_count) = sinus_app::teach::training_snapshot(&store)?;
        Ok(TrainingSnapshot {
            classes: classes
                .into_iter()
                .map(|class| ClassTraining {
                    event_type: class.class.into(),
                    status: class.status.into(),
                    takes: class.takes.into_iter().map(TeachTake::from).collect(),
                })
                .collect(),
            negative_count: negative_count as u32,
        })
    }

    /// Suppress persistence for the duration of a take. Detections keep running —
    /// the gate, noise floor and cooldowns stay continuous — but the coughs the
    /// user deliberately performs for training must not also be logged as events.
    pub fn begin_teach_take(&self) -> Result<(), AppleEngineError> {
        self.lock()?.set_suppress_persistence(true);
        Ok(())
    }

    /// Abandon a take in progress and resume persisting.
    pub fn cancel_teach_take(&self) -> Result<(), AppleEngineError> {
        self.lock()?.set_suppress_persistence(false);
        Ok(())
    }

    /// Score and store one take of exactly `teach_take_samples()` samples, then
    /// reload the live detector's prototypes. Always resumes persistence, including
    /// on failure — `MonitoringEngine::enroll_take` clears `suppress_persistence`
    /// on both the success and error paths, so there is nothing left to reset here.
    pub fn enroll_take(
        &self,
        event_type: AppleEventType,
        samples: Vec<f32>,
    ) -> Result<TeachResult, AppleEngineError> {
        // A short buffer would still produce a take — Rust just picks the loudest
        // window out of whatever it got — so a Swift bug that hands over, say, one
        // second instead of three would silently degrade scores instead of failing
        // loudly. Reject anything but exactly the agreed length.
        let expected = sinus_app::teach::TEACH_TAKE_SAMPLES;
        if samples.len() != expected {
            return Err(AppleEngineError::InvalidArgument {
                message: format!(
                    "teach take must contain exactly {expected} samples, got {}",
                    samples.len()
                ),
            });
        }
        let result = self.lock()?.enroll_take(event_type.into(), &samples)?;
        Ok(result.into())
    }

    /// Remove one take by its `TeachTake::id` — the "delete this recording" row
    /// action. Returns how many rows went, so the caller need not assume.
    pub fn delete_take(&self, id: i64) -> Result<u32, AppleEngineError> {
        self.delete_training(sinus_app::teach::Deletion::One(id))
    }

    /// Remove every take of one class, returning the class to untrained.
    pub fn delete_class_training(
        &self,
        event_type: AppleEventType,
    ) -> Result<u32, AppleEngineError> {
        self.delete_training(sinus_app::teach::Deletion::Class(event_type.into()))
    }

    /// Remove every negative enrollment — everything the detector learned from
    /// false-positive reports and corrections — while keeping the taught takes.
    pub fn delete_learned_suppressions(&self) -> Result<u32, AppleEngineError> {
        self.delete_training(sinus_app::teach::Deletion::Negatives)
    }

    /// Remove all personalization, positive and negative alike. The detector
    /// falls back to the generic decision rules.
    pub fn delete_all_training(&self) -> Result<u32, AppleEngineError> {
        self.delete_training(sinus_app::teach::Deletion::All)
    }

    /// Everything the menu bar renders, in one read, so a UI refresh is one FFI
    /// call rather than six.
    ///
    /// Deliberately never takes the writer lock. This is the call a menu bar
    /// polls several times a second, and the audio thread holds that lock across
    /// a Core ML inference — so everything here comes either from the read
    /// connection, which WAL lets run concurrently with the writer, or from the
    /// atomics the audio thread publishes as it goes.
    pub fn status(&self) -> Result<EngineStatus, AppleEngineError> {
        let store = self.reader()?;
        let now = Utc::now();
        let pause = pause_snapshot(sinus_app::settings::pause_state(&store), now);
        let quiet_hours =
            sinus_app::settings::quiet_hours(&store).map(|(start_hour, end_hour)| QuietHours {
                start_hour,
                end_hour,
            });
        let last_heard = self.last_heard_epoch_ms.load(Ordering::Relaxed);
        Ok(EngineStatus {
            monitoring: self.monitoring.load(Ordering::Relaxed),
            gate_open: self.gate_open.load(Ordering::Relaxed),
            last_heard_epoch_ms: if last_heard == 0 {
                None
            } else {
                Some(last_heard)
            },
            sensitivity: sinus_app::settings::sensitivity(&store),
            pause,
            quiet_hours,
            in_quiet_hours: sinus_app::sync::in_quiet_hours_now(&store),
            pause_on_low_power: sinus_app::settings::pause_on_low_power(&store),
            model_version: self.model_version.clone(),
        })
    }

    pub fn sensitivity(&self) -> Result<f32, AppleEngineError> {
        let store = self.reader()?;
        Ok(sinus_app::settings::sensitivity(&store))
    }

    pub fn quiet_hours(&self) -> Result<Option<QuietHours>, AppleEngineError> {
        let store = self.reader()?;
        Ok(
            sinus_app::settings::quiet_hours(&store).map(|(start_hour, end_hour)| QuietHours {
                start_hour,
                end_hour,
            }),
        )
    }

    /// Passing `None` clears the window. Written unsynced-dirty so the next flush
    /// carries it to the PHR — quiet hours follow the patient between machines.
    pub fn set_quiet_hours(&self, hours: Option<QuietHours>) -> Result<(), AppleEngineError> {
        if let Some(QuietHours {
            start_hour,
            end_hour,
        }) = hours
        {
            for (name, hour) in [("start_hour", start_hour), ("end_hour", end_hour)] {
                if hour > 23 {
                    return Err(AppleEngineError::InvalidArgument {
                        message: format!("{name} must be 0..=23, got {hour}"),
                    });
                }
            }
        }
        sinus_app::settings::set_quiet_hours(
            self.lock()?.store(),
            hours.map(|window| (window.start_hour, window.end_hour)),
        )?;
        Ok(())
    }

    /// Apply quiet hours to detection logging. The window itself is a setting the
    /// shell can already read; this is the lever that acts on it.
    ///
    /// Quiet hours suppress *logging*, not detection — the pipeline keeps running
    /// so the gate, noise floor and cooldowns stay continuous across the window,
    /// exactly as the tray app does at its own write site. Tearing the session
    /// down instead would reset the sample clock and skew the timestamps of every
    /// event after the window closed.
    ///
    /// Driven by the shell rather than checked here per push: `push_pcm_16k` runs
    /// several times a second and this answer changes on the hour, so reading the
    /// setting on every buffer would be a database round trip to learn nothing.
    /// The shell already polls `status()`, whose `in_quiet_hours` is the value to
    /// pass here.
    pub fn set_quiet_suppression(&self, suppressed: bool) -> Result<(), AppleEngineError> {
        self.lock()?.set_quiet(suppressed);
        Ok(())
    }

    pub fn in_quiet_hours(&self) -> Result<bool, AppleEngineError> {
        let store = self.reader()?;
        Ok(sinus_app::sync::in_quiet_hours_now(&store))
    }

    pub fn phr_settings(&self) -> Result<PhrSettings, AppleEngineError> {
        let store = self.reader()?;
        Ok(PhrSettings {
            server_url: sinus_app::settings::server_url(&store),
            patient_id: sinus_app::settings::patient_id(&store),
            mode: sinus_app::settings::mode(&store).into(),
            // Read directly rather than through `ensure_device_id`, which can
            // write: this is a read path, and by the time an engine exists the
            // id was already minted when `MonitoringEngine::from_store` opened
            // the writer connection.
            device_id: store
                .setting_get(sinus_app::settings::DEVICE_ID)?
                .unwrap_or_default(),
        })
    }

    pub fn set_server_url(&self, url: String) -> Result<(), AppleEngineError> {
        sinus_app::settings::set_server_url(self.lock()?.store(), &url)?;
        Ok(())
    }

    pub fn set_patient_id(&self, patient_id: Option<i64>) -> Result<(), AppleEngineError> {
        sinus_app::settings::set_patient_id(self.lock()?.store(), patient_id)?;
        Ok(())
    }

    pub fn set_sync_mode(&self, mode: SyncMode) -> Result<(), AppleEngineError> {
        sinus_app::settings::set_mode(self.lock()?.store(), mode.into())?;
        Ok(())
    }

    pub fn pause(&self) -> Result<PauseSnapshot, AppleEngineError> {
        let store = self.reader()?;
        let state = sinus_app::settings::pause_state(&store);
        Ok(pause_snapshot(state, Utc::now()))
    }

    /// `until_epoch_ms` is `Some` only for `PauseKind::Timed`; supplying it with any
    /// other kind, or omitting it for `Timed`, is an `InvalidArgument`.
    pub fn set_pause(
        &self,
        kind: PauseKind,
        until_epoch_ms: Option<i64>,
    ) -> Result<(), AppleEngineError> {
        let state = match (kind, until_epoch_ms) {
            (PauseKind::Running, None) => sinus_app::state::PauseState::Running,
            (PauseKind::Indefinite, None) => sinus_app::state::PauseState::PausedIndefinite,
            (PauseKind::Timed, Some(ms)) => {
                sinus_app::state::PauseState::PausedUntil(timestamp(ms)?)
            }
            _ => {
                return Err(AppleEngineError::InvalidArgument {
                    message:
                        "until_epoch_ms must be set for Timed and omitted for every other kind"
                            .to_string(),
                })
            }
        };
        sinus_app::settings::set_pause_state(self.lock()?.store(), state)?;
        Ok(())
    }

    /// Whether another Sinus Sentinel asked this one to show itself. A second
    /// launch cannot take the machine, so it leaves a marker and exits; polling
    /// this is how the running app learns to raise its window instead of the user
    /// seeing nothing happen. Consumes the request.
    pub fn take_activation_request(&self) -> bool {
        self.instance.take_activation_request()
    }
}

impl AppleEngine {
    fn lock(&self) -> Result<MutexGuard<'_, MonitoringEngine<ForeignEmbedder>>, AppleEngineError> {
        self.inner.lock().map_err(|_| AppleEngineError::Engine {
            message: "engine state lock was poisoned".to_string(),
        })
    }

    fn reader(&self) -> Result<MutexGuard<'_, Store>, AppleEngineError> {
        self.reader.lock().map_err(|_| AppleEngineError::Engine {
            message: "engine read lock was poisoned".to_string(),
        })
    }

    /// Shared body of the three flagging methods: resolve the uuid, run `op`,
    /// reload the matcher's prototypes if `op` actually trained anything, then
    /// re-read the event so the caller gets back exactly what changed.
    ///
    /// Everything happens under the writer lock and on the writer's connection,
    /// including the existence check. Checking on the read connection first
    /// would be a wasted round trip in the common case and, worse, would report
    /// `NotFound` for an event this very engine had just written but not yet
    /// checkpointed.
    fn flag(
        &self,
        event_uuid: String,
        op: impl FnOnce(&Store, &str) -> CoreResult<sinus_app::flag::FlagOutcome>,
    ) -> Result<FlagResult, AppleEngineError> {
        let mut engine = self.lock()?;
        // Resolved here rather than left to `sinus_app::flag`, which reports a
        // missing uuid as `Error::Config` — the blanket `From<CoreError>` would
        // flatten that into `Engine`, which Swift cannot tell apart from a real
        // failure. A stale list needs a different message than a broken engine.
        if engine.store().get_event(&event_uuid)?.is_none() {
            return Err(AppleEngineError::NotFound { uuid: event_uuid });
        }
        let outcome = op(engine.store(), &event_uuid)?;
        if outcome.trained {
            // The running matcher's prototypes are built at load time; without this
            // reload, the user reports a false positive and the very next identical
            // sound still fires.
            engine.reload_enrollments()?;
        }
        let updated =
            engine
                .store()
                .get_event(&event_uuid)?
                .ok_or_else(|| AppleEngineError::NotFound {
                    uuid: event_uuid.clone(),
                })?;
        Ok(FlagResult {
            event: updated.into(),
            trained: outcome.trained,
        })
    }

    /// Shared body of the four delete methods: remove rows, then reload the live
    /// detector's prototypes unconditionally. Unlike `flag`, this does not check
    /// whether anything was actually trained on those rows first — deleting a
    /// class's training while monitoring is running must not leave the stale
    /// matcher in place, and a reload against an unchanged prototype set is cheap.
    fn delete_training(&self, what: sinus_app::teach::Deletion) -> Result<u32, AppleEngineError> {
        let mut engine = self.lock()?;
        let removed = sinus_app::teach::delete(engine.store(), what)?;
        engine.reload_enrollments()?;
        Ok(removed as u32)
    }
}

/// Sync health, mirroring `sinus_app::sync::SyncState`. A separate type so this
/// crate's FFI surface has no dependency on any shell's UI types, matching why
/// `sinus_app::sync` keeps its own `SyncState` rather than reusing the desktop
/// tray's.
#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum SyncState {
    Idle,
    Syncing,
    Failed,
}

impl From<sinus_app::sync::SyncState> for SyncState {
    fn from(value: sinus_app::sync::SyncState) -> Self {
        match value {
            sinus_app::sync::SyncState::Idle => Self::Idle,
            sinus_app::sync::SyncState::Syncing => Self::Syncing,
            sinus_app::sync::SyncState::Failed => Self::Failed,
        }
    }
}

/// One tick's worth of sync state, pushed to the shell rather than polled —
/// the driver thread sleeps between ticks, so a poll would either be stale or
/// force it awake.
#[derive(Debug, Clone, uniffi::Record)]
pub struct SyncStatusSnapshot {
    pub state: SyncState,
    pub mode: SyncMode,
    /// Badge count — events only, which is what "pending" means to a user.
    pub pending_events: u32,
    /// The flush gate: events plus tombstones, flags and enrollments.
    pub pending_work: u32,
    /// Whether the current local hour falls in the quiet-hours window.
    pub quiet: bool,
    /// The failure text from the last attempt, if it failed. Before this
    /// existed the reason only reached stderr, so a user saw "sync failing"
    /// with no way to learn why.
    ///
    /// The shell distinguishes exactly two failures by substring rather than
    /// by a structured error code: text containing "no API token configured"
    /// (see `sinus_core::sync::SyncEngine::bearer`) means the PHR API token
    /// has not been set and the user should be sent to Settings › PHR; text
    /// containing "keychain" (see `TokenError::Keychain`) means the Keychain
    /// read itself failed. This comment is the contract — do not add parsing
    /// or error-code plumbing in Rust for it.
    pub error: Option<String>,
    /// Epoch milliseconds of the last successful flush; `None` if none yet.
    pub last_success_epoch_ms: Option<i64>,
}

impl SyncStatusSnapshot {
    /// What `SyncController::status()` returns before the driver thread has
    /// completed its first tick.
    fn idle() -> Self {
        SyncStatusSnapshot {
            state: SyncState::Idle,
            mode: SyncMode::AutoBatch,
            pending_events: 0,
            pending_work: 0,
            quiet: false,
            error: None,
            last_success_epoch_ms: None,
        }
    }
}

/// Implemented in Swift to receive sync status pushes.
#[uniffi::export(foreign)]
pub trait SyncObserver: Send + Sync {
    /// Called on the driver thread, not the main thread — Swift implementations
    /// must hop to the main actor before touching UI state.
    fn on_status(&self, status: SyncStatusSnapshot);
}

/// Runs [`sinus_app::sync::SyncDriver`] on its own thread against its own
/// database connection, pushing every tick's result to a Swift-side
/// [`SyncObserver`].
///
/// Takes `Arc<AppleEngine>` rather than the raw config the engine was built
/// from, and holds onto it for as long as the controller lives: a
/// `SyncController` can then structurally not exist unless this process holds
/// the machine's `InstanceGuard` (owned by `AppleEngine`), so two shells can
/// never sync the same database concurrently, and dropping the caller's own
/// `Arc<AppleEngine>` while keeping the controller cannot release the guard
/// out from under it.
#[derive(uniffi::Object)]
pub struct SyncController {
    /// Kept alive only for its `InstanceGuard` — the controller opens its own
    /// `Store` connection below rather than sharing the engine's, since
    /// `Store` is not `Sync` and `SyncDriver::tick` needs `&mut Store`.
    _engine: Arc<AppleEngine>,
    /// The same object the driver thread reads from, so `set_token` /
    /// `clear_token` update the one cache the thread sees. A Swift caller that
    /// wrote the Keychain directly instead would leave that cache serving a
    /// stale token until relaunch.
    tokens: Arc<dyn TokenStore>,
    /// Wakes the driver thread immediately for a manual sync or a shutdown,
    /// rather than letting it sleep through the request until its next
    /// scheduled tick.
    wake: Arc<(Mutex<u64>, Condvar)>,
    manual_requested: Arc<AtomicBool>,
    quitting: Arc<AtomicBool>,
    /// The last snapshot the driver thread pushed, for a caller that reads
    /// `status()` before the first tick or between ticks.
    status: Arc<Mutex<SyncStatusSnapshot>>,
    /// Taken by `shutdown`; `None` once a shutdown has run, so a second call
    /// finds nothing left to join instead of panicking.
    thread: Mutex<Option<JoinHandle<()>>>,
}

#[uniffi::export]
impl SyncController {
    #[uniffi::constructor]
    pub fn new(
        engine: Arc<AppleEngine>,
        tokens: Arc<dyn TokenProvider>,
        observer: Arc<dyn SyncObserver>,
    ) -> Result<Arc<SyncController>, AppleEngineError> {
        let store = Store::open(&engine.database_path)?;
        let source = engine.platform.source();
        let token_store: Arc<dyn TokenStore> = Arc::new(ForeignTokenStore::new(tokens));
        let wake = Arc::new((Mutex::new(0u64), Condvar::new()));
        let manual_requested = Arc::new(AtomicBool::new(false));
        let quitting = Arc::new(AtomicBool::new(false));
        let status = Arc::new(Mutex::new(SyncStatusSnapshot::idle()));

        let handle = {
            let tokens = Arc::clone(&token_store);
            let wake = Arc::clone(&wake);
            let manual_requested = Arc::clone(&manual_requested);
            let quitting = Arc::clone(&quitting);
            let status = Arc::clone(&status);
            // The thread holds its own reference to the engine, not just the
            // controller does. `Drop` signals stop without joining, so the
            // thread can outlive the controller; without this the engine — and
            // with it the machine's `InstanceGuard` — could be released while
            // this thread was still writing to the database, letting another
            // shell take the machine mid-flush.
            let engine = Arc::clone(&engine);
            std::thread::spawn(move || {
                run_sync_driver(
                    store,
                    source,
                    tokens,
                    observer,
                    wake,
                    manual_requested,
                    quitting,
                    status,
                );
                drop(engine);
            })
        };

        Ok(Arc::new(SyncController {
            _engine: engine,
            tokens: token_store,
            wake,
            manual_requested,
            quitting,
            status,
            thread: Mutex::new(Some(handle)),
        }))
    }

    /// Ask for a flush now, regardless of schedule. Returns immediately; the
    /// result arrives via the observer.
    pub fn sync_now(&self) {
        self.manual_requested.store(true, Ordering::Relaxed);
        signal_wake(&self.wake);
    }

    /// Routes through the controller's `TokenStore` rather than letting Swift
    /// write the Keychain directly, so the `ForeignTokenStore` cache the
    /// driver thread reads from stays coherent instead of serving a stale
    /// token until relaunch.
    pub fn set_token(&self, token: String) -> Result<(), AppleEngineError> {
        self.tokens.set_token(&token)?;
        Ok(())
    }

    /// See [`Self::set_token`] — routes through the same store for the same reason.
    pub fn clear_token(&self) -> Result<(), AppleEngineError> {
        self.tokens.clear()?;
        Ok(())
    }

    pub fn has_token(&self) -> Result<bool, AppleEngineError> {
        Ok(self
            .tokens
            .get_token()?
            .is_some_and(|token| !token.is_empty()))
    }

    /// The last snapshot pushed to the observer, for a view that appears after
    /// a tick rather than before it.
    pub fn status(&self) -> SyncStatusSnapshot {
        self.status
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_else(|_| SyncStatusSnapshot::idle())
    }

    /// Stop the driver thread and wait up to `timeout_ms` for it to finish its
    /// final flush. The deterministic counterpart to `Drop`, which signals stop
    /// but does not join.
    pub fn shutdown(&self, timeout_ms: u64) {
        self.quitting.store(true, Ordering::Relaxed);
        signal_wake(&self.wake);

        let Ok(mut slot) = self.thread.lock() else {
            return;
        };
        let Some(handle) = slot.take() else {
            // Already shut down — nothing left to join.
            return;
        };
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        // `JoinHandle` has no timed join, so poll `is_finished` instead of
        // blocking on `join` outright.
        while !handle.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        if handle.is_finished() {
            let _ = handle.join();
        }
        // Otherwise the deadline passed with a flush still in flight. Dropping
        // the handle here lets the thread finish in the background rather than
        // blocking the caller — typically the UI thread — indefinitely.
    }
}

impl Drop for SyncController {
    /// Signals the driver thread to stop but does not join it: `Drop` running
    /// on whatever thread drops the last `Arc` must not block on a flush that
    /// can take up to the HTTP client's 30-second timeout. `shutdown` is the
    /// deterministic, bounded way to stop and wait.
    fn drop(&mut self) {
        self.quitting.store(true, Ordering::Relaxed);
        signal_wake(&self.wake);
    }
}

/// The driver thread body: tick `SyncDriver` in a loop, publish each result to
/// `observer` and the shared `status` cell, and sleep on `wake` between ticks.
/// Mirrors `apps/desktop/src/sync.rs::run_sync`; the difference is that state
/// reaches the shell via a push (`SyncObserver::on_status`) instead of a
/// shared-memory struct the UI polls.
#[allow(clippy::too_many_arguments)]
fn run_sync_driver(
    mut store: Store,
    source: Source,
    tokens: Arc<dyn TokenStore>,
    observer: Arc<dyn SyncObserver>,
    wake: Arc<(Mutex<u64>, Condvar)>,
    manual_requested: Arc<AtomicBool>,
    quitting: Arc<AtomicBool>,
    status: Arc<Mutex<SyncStatusSnapshot>>,
) {
    let mut driver = SyncDriver::new(source, FlushPolicy::default(), tokens);
    let mut last_success_epoch_ms: Option<i64> = None;
    let mut observed = wake_generation(&wake);

    loop {
        let input = TickInput {
            manual: manual_requested.swap(false, Ordering::Relaxed),
            quitting: quitting.load(Ordering::Relaxed),
        };
        let output = driver.tick(&mut store, input);

        if output.flushed.is_some() {
            last_success_epoch_ms = Some(Utc::now().timestamp_millis());
        }

        let snapshot = SyncStatusSnapshot {
            state: output.state.into(),
            mode: output.mode.into(),
            pending_events: output.pending_events as u32,
            pending_work: output.pending_work as u32,
            quiet: output.quiet,
            error: output.error,
            last_success_epoch_ms,
        };
        if let Ok(mut slot) = status.lock() {
            *slot = snapshot.clone();
        }
        observer.on_status(snapshot);

        if output.done {
            break;
        }
        if output.next_wait == Duration::ZERO {
            // A flush just succeeded; there may be more pending work to drain.
            continue;
        }
        observed = wait_for_wake(&wake, observed, output.next_wait);
    }
}

fn wake_generation(wake: &(Mutex<u64>, Condvar)) -> u64 {
    wake.0.lock().map_or(0, |generation| *generation)
}

fn signal_wake(wake: &(Mutex<u64>, Condvar)) {
    if let Ok(mut generation) = wake.0.lock() {
        *generation = generation.wrapping_add(1);
        wake.1.notify_one();
    }
}

fn wait_for_wake(wake: &(Mutex<u64>, Condvar), observed: u64, timeout: Duration) -> u64 {
    let (generation, condvar) = wake;
    let Ok(generation) = generation.lock() else {
        // Practically unreachable — nothing inside this lock can panic — but the
        // fallback still has to sleep. Returning straight away would turn a
        // poisoned mutex into a loop that ticks the driver as fast as it can,
        // hammering the database and the server instead of degrading quietly.
        std::thread::sleep(timeout);
        return observed;
    };
    if *generation != observed {
        return *generation;
    }
    condvar
        .wait_timeout_while(generation, timeout, |current| *current == observed)
        .map_or(observed, |(current, _)| *current)
}

fn check_timezone_offset(offset_minutes: i32) -> Result<(), AppleEngineError> {
    if (-1_439..=1_439).contains(&offset_minutes) {
        Ok(())
    } else {
        Err(AppleEngineError::InvalidArgument {
            message: "timezone offset must be between -1439 and 1439 minutes".to_string(),
        })
    }
}

fn timestamp(epoch_ms: i64) -> Result<DateTime<Utc>, AppleEngineError> {
    DateTime::from_timestamp_millis(epoch_ms).ok_or_else(|| AppleEngineError::InvalidArgument {
        message: format!("invalid epoch timestamp: {epoch_ms}"),
    })
}

fn event_counts(counts: &std::collections::HashMap<EventType, i64>) -> Vec<EventCount> {
    EventType::ALL
        .into_iter()
        .map(|event_type| EventCount {
            event_type: event_type.into(),
            count: counts.get(&event_type).copied().unwrap_or(0).max(0) as u64,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sinus_core::classify::embed::BandHeuristicEmbedder;

    #[derive(Debug)]
    struct TestModel;

    impl ModelRunner for TestModel {
        fn model_version(&self) -> Result<String, ModelError> {
            Ok("test-model@1".to_string())
        }

        fn infer(
            &self,
            log_mel: Vec<f32>,
            frames: u32,
            bands: u32,
            energy_peak: bool,
        ) -> Result<ModelOutput, ModelError> {
            let patch = MelPatch {
                frames: frames as usize,
                bands: bands as usize,
                data: log_mel,
            };
            let output = BandHeuristicEmbedder
                .embed(&patch, energy_peak)
                .map_err(|error| ModelError::Failed {
                    message: error.to_string(),
                })?;
            Ok(ModelOutput {
                audioset_scores: output.audioset_scores.unwrap(),
                embedding: output.embedding,
            })
        }
    }

    /// A private data directory per test: the engine takes a single-instance lock
    /// on the database's parent, so tests sharing one would contend.
    fn temp_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("sinus-apple-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn engine_in(dir: &std::path::Path) -> Result<Arc<AppleEngine>, AppleEngineError> {
        AppleEngine::new(
            AppleEngineConfig {
                database_path: dir.join("events.db").to_string_lossy().into_owned(),
                platform: ApplePlatform::Ios,
            },
            Arc::new(TestModel),
        )
    }

    fn temp_engine() -> (Arc<AppleEngine>, std::path::PathBuf) {
        let dir = temp_dir();
        let engine = engine_in(&dir).unwrap();
        (engine, dir)
    }

    #[test]
    fn battery_policy_defaults_on_and_round_trips() {
        let (engine, dir) = temp_engine();
        assert!(engine.pause_on_low_power().unwrap());
        engine.set_pause_on_low_power(false).unwrap();
        assert!(!engine.pause_on_low_power().unwrap());
        engine.set_pause_on_low_power(true).unwrap();
        assert!(engine.pause_on_low_power().unwrap());
        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn bridge_rejects_an_impossible_timezone_offset() {
        let (engine, dir) = temp_engine();
        let error = engine.start_monitoring(0, 5_000).unwrap_err();
        assert!(error.to_string().contains("timezone offset"));
        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn bridge_rejects_pcm_before_session_start() {
        let (engine, dir) = temp_engine();
        let error = engine.push_pcm_16k(vec![0.0; 800]).unwrap_err();
        assert!(error.to_string().contains("start a monitoring session"));
        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_second_shell_on_the_same_machine_is_refused() {
        let dir = temp_dir();
        let first = engine_in(&dir).unwrap();
        assert!(matches!(
            engine_in(&dir),
            Err(AppleEngineError::AlreadyRunning)
        ));
        // Quitting the owner hands the machine back.
        drop(first);
        let second = engine_in(&dir).unwrap();
        drop(second);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn sensitivity_comes_from_the_shared_store_not_the_shell() {
        let dir = temp_dir();
        let engine = engine_in(&dir).unwrap();
        engine.set_sensitivity(0.8).unwrap();
        drop(engine);

        // A fresh launch must not reset what the user (or a PHR sync) chose.
        let engine = engine_in(&dir).unwrap();
        assert_eq!(
            engine
                .get_setting("sensitivity".to_string())
                .unwrap()
                .as_deref(),
            Some("0.8")
        );
        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn bridge_session_emits_and_projects_an_event() {
        let (engine, dir) = temp_engine();
        let now = Utc::now();
        let event = emit_one_event(&engine, now);

        assert!(matches!(event.event_type, AppleEventType::Cough));
        let history = engine
            .history(7, (now + chrono::Duration::hours(1)).timestamp_millis(), 0)
            .unwrap();
        assert_eq!(
            history
                .today
                .iter()
                .find(|count| matches!(count.event_type, AppleEventType::Cough))
                .unwrap()
                .count,
            1
        );
        assert_eq!(history.recent_events.len(), 1);

        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn quiet_suppression_logs_nothing_and_leaves_the_gate_running() {
        let (engine, dir) = temp_engine();
        let now = Utc::now();

        engine.set_quiet_suppression(true).unwrap();
        engine.start_monitoring(now.timestamp_millis(), 0).unwrap();
        let mut signal = sinus_core::synth::white_noise(16_000, 0.003, 1);
        signal.extend(sinus_core::synth::sine(32_000, 16_000, 300.0, 0.6));
        signal.extend(sinus_core::synth::white_noise(16_000, 0.003, 2));
        let mut heard = false;
        for chunk in signal.chunks(777) {
            assert!(engine.push_pcm_16k(chunk.to_vec()).unwrap().is_empty());
            // The point of suppressing at the write site rather than by cutting
            // the audio off: the gate still opens, so the indicator and the
            // noise floor behave the same inside the window as outside it.
            heard |= engine.status().unwrap().gate_open;
        }
        assert!(engine.stop_monitoring().unwrap().is_empty());
        assert!(heard, "the gate never opened under quiet suppression");
        assert_eq!(
            engine
                .history(7, now.timestamp_millis(), 0)
                .unwrap()
                .recent_events
                .len(),
            0
        );

        // And the flag is not session state: stopping did not clear it.
        engine.set_quiet_suppression(false).unwrap();
        assert!(!emit_one_event(&engine, now).uuid.is_empty());

        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Drives a synthetic session that produces exactly one detected cough, and
    /// returns it — the starting point for every test that needs a real,
    /// persisted row rather than a hand-built one.
    fn emit_one_event(engine: &Arc<AppleEngine>, now: DateTime<Utc>) -> AppleEvent {
        engine.start_monitoring(now.timestamp_millis(), 0).unwrap();
        let mut signal = sinus_core::synth::white_noise(16_000, 0.003, 1);
        signal.extend(sinus_core::synth::sine(32_000, 16_000, 300.0, 0.6));
        signal.extend(sinus_core::synth::white_noise(16_000, 0.003, 2));
        let mut emitted = Vec::new();
        for chunk in signal.chunks(777) {
            emitted.extend(engine.push_pcm_16k(chunk.to_vec()).unwrap());
        }
        emitted.extend(engine.stop_monitoring().unwrap());
        assert_eq!(emitted.len(), 1);
        emitted.into_iter().next().unwrap()
    }

    #[test]
    fn reporting_an_unknown_uuid_is_distinguishable_from_a_real_failure() {
        let (engine, dir) = temp_engine();
        assert!(matches!(
            engine.report_false_positive("not-a-real-uuid".to_string()),
            Err(AppleEngineError::NotFound { uuid }) if uuid == "not-a-real-uuid"
        ));
        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn recharacterize_updates_the_effective_type_and_keeps_the_audit_trail() {
        let (engine, dir) = temp_engine();
        let now = Utc::now();
        let event = emit_one_event(&engine, now);
        assert!(matches!(event.event_type, AppleEventType::Cough));

        let result = engine
            .recharacterize(event.uuid.clone(), AppleEventType::Sniffle)
            .unwrap();

        assert!(matches!(result.event.event_type, AppleEventType::Sniffle));
        assert!(matches!(
            result.event.original_event_type,
            AppleEventType::Cough
        ));
        assert!(matches!(
            result.event.corrected_to,
            Some(AppleEventType::Sniffle)
        ));
        assert!(result.trained);

        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn clear_flag_after_a_report_restores_the_event() {
        let (engine, dir) = temp_engine();
        let now = Utc::now();
        let event = emit_one_event(&engine, now);

        let reported = engine.report_false_positive(event.uuid.clone()).unwrap();
        assert!(reported.event.false_positive);

        let restored = engine.clear_flag(event.uuid.clone()).unwrap();
        assert!(!restored.event.false_positive);

        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn monitored_hours_reports_elapsed_time_since_local_midnight() {
        let (engine, dir) = temp_engine();
        let now = Utc::now();
        let midnight = sinus_app::state::local_midnight_at_offset(now, 0);
        let an_hour_after_midnight = midnight + chrono::Duration::hours(1);

        let history = engine
            .history(1, an_hour_after_midnight.timestamp_millis(), 0)
            .unwrap();

        assert!(
            (history.monitored_hours - 1.0).abs() < 0.01,
            "expected roughly 1.0 monitored hours, got {}",
            history.monitored_hours
        );

        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A take loud enough for `loudest_patch` to select, exactly
    /// `teach_take_samples()` long.
    fn take_signal() -> Vec<f32> {
        sinus_core::synth::sine(teach_take_samples() as usize, 16_000, 300.0, 0.6)
    }

    #[test]
    fn enroll_take_rejects_the_wrong_sample_count() {
        let (engine, dir) = temp_engine();
        let error = engine
            .enroll_take(AppleEventType::Hawk, vec![0.0; 100])
            .unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains(&teach_take_samples().to_string()),
            "expected message to name the expected count, got: {message}"
        );
        assert!(
            message.contains("100"),
            "expected message to name the actual count, got: {message}"
        );
        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_fresh_engine_reports_every_class_untrained() {
        let (engine, dir) = temp_engine();
        let training = engine.training().unwrap();
        assert_eq!(training.negative_count, 0);
        assert_eq!(training.classes.len(), EventType::ALL.len());
        for class in &training.classes {
            assert!(matches!(class.status, TrainingStatus::Untrained));
            assert!(class.takes.is_empty());
        }
        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enrolling_takes_moves_a_class_off_untrained() {
        let (engine, dir) = temp_engine();

        let first = engine
            .enroll_take(AppleEventType::Hawk, take_signal())
            .unwrap();
        assert_eq!(first.examples, 1);
        let second = engine
            .enroll_take(AppleEventType::Hawk, take_signal())
            .unwrap();
        assert_eq!(second.examples, 2);
        let third = engine
            .enroll_take(AppleEventType::Hawk, take_signal())
            .unwrap();
        assert_eq!(third.examples, 3);

        let training = engine.training().unwrap();
        for class in &training.classes {
            match class.event_type {
                AppleEventType::Hawk => {
                    assert_eq!(class.takes.len(), 3);
                    assert!(!matches!(class.status, TrainingStatus::Untrained));
                }
                _ => assert!(matches!(class.status, TrainingStatus::Untrained)),
            }
        }

        let removed = engine.delete_class_training(AppleEventType::Hawk).unwrap();
        assert_eq!(removed, 3);
        let training = engine.training().unwrap();
        let hawk = training
            .classes
            .iter()
            .find(|class| matches!(class.event_type, AppleEventType::Hawk))
            .unwrap();
        assert!(matches!(hawk.status, TrainingStatus::Untrained));
        assert!(hawk.takes.is_empty());

        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_take_in_progress_suppresses_persistence_until_cancelled() {
        let (engine, dir) = temp_engine();
        let now = Utc::now();
        engine.start_monitoring(now.timestamp_millis(), 0).unwrap();

        engine.begin_teach_take().unwrap();
        let mut signal = sinus_core::synth::white_noise(16_000, 0.003, 1);
        signal.extend(sinus_core::synth::sine(32_000, 16_000, 300.0, 0.6));
        signal.extend(sinus_core::synth::white_noise(16_000, 0.003, 2));
        let mut emitted = Vec::new();
        for chunk in signal.chunks(777) {
            emitted.extend(engine.push_pcm_16k(chunk.to_vec()).unwrap());
        }
        assert!(emitted.is_empty());

        engine.cancel_teach_take().unwrap();
        let event = emit_one_event(&engine, now + chrono::Duration::seconds(10));
        assert!(matches!(event.event_type, AppleEventType::Cough));

        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn quiet_hours_round_trip_and_reject_bad_hours() {
        let (engine, dir) = temp_engine();
        assert!(engine.quiet_hours().unwrap().is_none());

        engine
            .set_quiet_hours(Some(QuietHours {
                start_hour: 22,
                end_hour: 7,
            }))
            .unwrap();
        assert_eq!(
            engine.quiet_hours().unwrap(),
            Some(QuietHours {
                start_hour: 22,
                end_hour: 7
            })
        );

        // `start == end` is the documented "disabled" encoding, not a
        // zero-length window.
        engine
            .set_quiet_hours(Some(QuietHours {
                start_hour: 9,
                end_hour: 9,
            }))
            .unwrap();
        assert!(engine.quiet_hours().unwrap().is_none());

        let error = engine
            .set_quiet_hours(Some(QuietHours {
                start_hour: 24,
                end_hour: 7,
            }))
            .unwrap_err();
        assert!(matches!(error, AppleEngineError::InvalidArgument { .. }));

        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn set_pause_validates_kind_and_deadline_pairing() {
        let (engine, dir) = temp_engine();

        assert!(matches!(
            engine.set_pause(PauseKind::Timed, None),
            Err(AppleEngineError::InvalidArgument { .. })
        ));
        assert!(matches!(
            engine.set_pause(PauseKind::Indefinite, Some(1)),
            Err(AppleEngineError::InvalidArgument { .. })
        ));
        assert!(matches!(
            engine.set_pause(PauseKind::Running, Some(1)),
            Err(AppleEngineError::InvalidArgument { .. })
        ));

        let now = Utc::now();
        let future = (now + chrono::Duration::hours(1)).timestamp_millis();
        engine.set_pause(PauseKind::Timed, Some(future)).unwrap();
        let snapshot = engine.pause().unwrap();
        assert!(matches!(snapshot.kind, PauseKind::Timed));
        assert_eq!(snapshot.until_epoch_ms, Some(future));
        assert!(snapshot.paused);

        let past = (now - chrono::Duration::hours(1)).timestamp_millis();
        engine.set_pause(PauseKind::Timed, Some(past)).unwrap();
        let snapshot = engine.pause().unwrap();
        assert!(matches!(snapshot.kind, PauseKind::Timed));
        assert!(!snapshot.paused);

        engine.set_pause(PauseKind::Running, None).unwrap();
        let snapshot = engine.pause().unwrap();
        assert!(matches!(snapshot.kind, PauseKind::Running));
        assert!(!snapshot.paused);

        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_pause_survives_a_relaunch() {
        let dir = temp_dir();
        let engine = engine_in(&dir).unwrap();
        engine.set_pause(PauseKind::Indefinite, None).unwrap();
        drop(engine);

        // The `pause_until` row is device-local settings, not session state —
        // this is the whole point of storing it rather than holding it in
        // memory, so a relaunch mid-pause does not silently resume capture.
        let engine = engine_in(&dir).unwrap();
        let snapshot = engine.pause().unwrap();
        assert!(matches!(snapshot.kind, PauseKind::Indefinite));
        assert!(snapshot.paused);

        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn phr_settings_default_and_round_trip_through_a_relaunch() {
        let dir = temp_dir();
        let engine = engine_in(&dir).unwrap();
        let defaults = engine.phr_settings().unwrap();
        assert_eq!(defaults.server_url, "");
        assert_eq!(defaults.patient_id, None);
        assert!(matches!(defaults.mode, SyncMode::AutoBatch));
        assert!(!defaults.device_id.is_empty());

        engine
            .set_server_url("https://phr.example".to_string())
            .unwrap();
        engine.set_patient_id(Some(42)).unwrap();
        engine.set_sync_mode(SyncMode::OfflineFirst).unwrap();
        drop(engine);

        // A shell must not reset settings a previous launch (or the user) chose.
        let engine = engine_in(&dir).unwrap();
        let settings = engine.phr_settings().unwrap();
        assert_eq!(settings.server_url, "https://phr.example");
        assert_eq!(settings.patient_id, Some(42));
        assert!(matches!(settings.mode, SyncMode::OfflineFirst));
        assert_eq!(settings.device_id, defaults.device_id);

        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn status_on_a_fresh_engine_reports_defaults() {
        let (engine, dir) = temp_engine();
        let status = engine.status().unwrap();
        assert!(!status.monitoring);
        assert!(!status.gate_open);
        assert_eq!(status.last_heard_epoch_ms, None);
        assert_eq!(status.sensitivity, sinus_app::settings::DEFAULT_SENSITIVITY);
        assert!(matches!(status.pause.kind, PauseKind::Running));
        assert!(!status.pause.paused);
        assert!(status.quiet_hours.is_none());
        assert!(!status.in_quiet_hours);

        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn status_reports_the_gate_opening_from_a_real_session() {
        let (engine, dir) = temp_engine();
        let now = Utc::now();
        emit_one_event(&engine, now);

        let status = engine.status().unwrap();
        assert!(status.last_heard_epoch_ms.is_some());
        // The mic is closed again, but "last heard" is history, not live state.
        assert!(!status.gate_open);

        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// `is_monitoring` and `status().monitoring` read a mirror of the engine's
    /// own flag rather than the engine itself, so that a UI poll never waits on
    /// the audio thread. A mirror can drift; this checks it does not.
    #[test]
    fn the_published_monitoring_flag_tracks_the_engine() {
        let (engine, dir) = temp_engine();
        assert!(!engine.is_monitoring().unwrap());

        engine
            .start_monitoring(Utc::now().timestamp_millis(), 0)
            .unwrap();
        assert!(engine.is_monitoring().unwrap());
        assert!(engine.status().unwrap().monitoring);
        assert_eq!(
            engine.is_monitoring().unwrap(),
            engine.lock().unwrap().is_monitoring()
        );

        engine.stop_monitoring().unwrap();
        assert!(!engine.is_monitoring().unwrap());
        assert!(!engine.status().unwrap().monitoring);
        assert_eq!(
            engine.is_monitoring().unwrap(),
            engine.lock().unwrap().is_monitoring()
        );

        drop(engine);
        let _ = std::fs::remove_dir_all(dir);
    }

    // --- Sync bridge ----------------------------------------------------

    use std::sync::atomic::AtomicUsize;

    /// A `TokenProvider` over an in-memory cell, counting `get_token` calls so
    /// tests can assert `ForeignTokenStore` actually caches rather than
    /// forwarding every read.
    #[derive(Debug, Default)]
    struct TestTokenProvider {
        token: Mutex<Option<String>>,
        get_calls: AtomicUsize,
        fail_writes: AtomicBool,
    }

    impl TestTokenProvider {
        fn new(initial: Option<&str>) -> Self {
            TestTokenProvider {
                token: Mutex::new(initial.map(str::to_string)),
                get_calls: AtomicUsize::new(0),
                fail_writes: AtomicBool::new(false),
            }
        }
    }

    impl TokenProvider for TestTokenProvider {
        fn get_token(&self) -> Result<Option<String>, TokenError> {
            self.get_calls.fetch_add(1, Ordering::Relaxed);
            Ok(self.token.lock().unwrap().clone())
        }

        fn set_token(&self, token: String) -> Result<(), TokenError> {
            if self.fail_writes.load(Ordering::Relaxed) {
                return Err(TokenError::Keychain {
                    message: "write denied".to_string(),
                });
            }
            *self.token.lock().unwrap() = Some(token);
            Ok(())
        }

        fn clear_token(&self) -> Result<(), TokenError> {
            if self.fail_writes.load(Ordering::Relaxed) {
                return Err(TokenError::Keychain {
                    message: "write denied".to_string(),
                });
            }
            *self.token.lock().unwrap() = None;
            Ok(())
        }
    }

    /// A `SyncObserver` that records every pushed snapshot for later inspection.
    #[derive(Debug, Default)]
    struct TestObserver {
        snapshots: Mutex<Vec<SyncStatusSnapshot>>,
    }

    impl SyncObserver for TestObserver {
        fn on_status(&self, status: SyncStatusSnapshot) {
            self.snapshots.lock().unwrap().push(status);
        }
    }

    impl TestObserver {
        fn count(&self) -> usize {
            self.snapshots.lock().unwrap().len()
        }
    }

    #[test]
    fn foreign_token_store_caches_reads_and_absorbs_a_write() {
        let provider = Arc::new(TestTokenProvider::new(Some("first")));
        let store = ForeignTokenStore::new(Arc::clone(&provider) as Arc<dyn TokenProvider>);

        // The important assertion: three reads, one provider call.
        assert_eq!(store.get_token().unwrap().as_deref(), Some("first"));
        assert_eq!(store.get_token().unwrap().as_deref(), Some("first"));
        assert_eq!(store.get_token().unwrap().as_deref(), Some("first"));
        assert_eq!(provider.get_calls.load(Ordering::Relaxed), 1);

        store.set_token("second").unwrap();
        assert_eq!(store.get_token().unwrap().as_deref(), Some("second"));
        // The write updated the cache directly; no extra provider read needed.
        assert_eq!(provider.get_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn foreign_token_store_caches_an_absent_token() {
        let provider = Arc::new(TestTokenProvider::new(None));
        let store = ForeignTokenStore::new(Arc::clone(&provider) as Arc<dyn TokenProvider>);

        assert_eq!(store.get_token().unwrap(), None);
        assert_eq!(store.get_token().unwrap(), None);
        assert_eq!(provider.get_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_failed_write_leaves_the_cache_untouched() {
        let provider = Arc::new(TestTokenProvider::new(Some("old")));
        let store = ForeignTokenStore::new(Arc::clone(&provider) as Arc<dyn TokenProvider>);
        assert_eq!(store.get_token().unwrap().as_deref(), Some("old"));

        provider.fail_writes.store(true, Ordering::Relaxed);
        assert!(store.set_token("new").is_err());

        // The failed write must not have poisoned the cache with the rejected
        // value, and reading it back must not have cost a second provider call.
        assert_eq!(store.get_token().unwrap().as_deref(), Some("old"));
        assert_eq!(provider.get_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn sync_controller_token_round_trip_and_repeated_shutdown() {
        let (engine, dir) = temp_engine();
        let provider: Arc<dyn TokenProvider> = Arc::new(TestTokenProvider::new(None));
        let observer: Arc<dyn SyncObserver> = Arc::new(TestObserver::default());
        let controller = SyncController::new(engine, provider, observer).unwrap();

        assert!(!controller.has_token().unwrap());
        controller.set_token("a-token".to_string()).unwrap();
        assert!(controller.has_token().unwrap());
        controller.clear_token().unwrap();
        assert!(!controller.has_token().unwrap());

        controller.shutdown(2_000);
        // A second shutdown, after the thread has already stopped, must not panic.
        controller.shutdown(2_000);

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Before the driver thread has run a tick with real PHR settings
    /// configured, its output is indistinguishable from the seeded idle
    /// default — no server URL means no flush is ever attempted — so this
    /// holds whether or not a tick has actually landed by the time `status()`
    /// is called, without asserting on any network outcome.
    #[test]
    fn status_never_panics_and_starts_idle() {
        let (engine, dir) = temp_engine();
        let provider: Arc<dyn TokenProvider> = Arc::new(TestTokenProvider::new(None));
        let observer: Arc<dyn SyncObserver> = Arc::new(TestObserver::default());
        let controller = SyncController::new(engine, provider, observer).unwrap();

        let status = controller.status();
        assert!(matches!(status.state, SyncState::Idle));
        assert_eq!(status.pending_events, 0);
        assert_eq!(status.last_success_epoch_ms, None);

        controller.shutdown(2_000);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Exercises the whole thread wiring end to end: the driver spawns, ticks,
    /// and pushes its result to the Swift-side observer. Everything else here
    /// tests the pieces; nothing else tests that they are actually connected.
    #[test]
    fn the_driver_thread_pushes_status_to_the_observer() {
        let (engine, dir) = temp_engine();
        let observer = Arc::new(TestObserver::default());
        let controller = SyncController::new(
            engine,
            Arc::new(TestTokenProvider::new(None)),
            Arc::clone(&observer) as Arc<dyn SyncObserver>,
        )
        .unwrap();

        controller.sync_now();
        // Polled rather than slept on a fixed delay: the first tick is
        // near-immediate, and a fixed sleep would be either flaky or slow.
        let deadline = Instant::now() + Duration::from_secs(5);
        while observer.count() == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(observer.count() > 0, "the observer was never called");

        controller.shutdown(2_000);
        let _ = std::fs::remove_dir_all(dir);
    }
}
