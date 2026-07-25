//! UniFFI boundary for native Apple clients.
//!
//! Swift owns capture, lifecycle, Core ML, and presentation. Rust accepts
//! converted 16 kHz mono PCM and owns the detector, persistence, and projections.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use chrono::{DateTime, Utc};
use sinus_app::instance::InstanceGuard;
use sinus_app::monitor::{MonitoringConfig, MonitoringEngine};
use sinus_core::classify::embed::{Embedder, WindowFeatures, AUDIOSET_CLASSES, EMBED_DIM};
use sinus_core::error::{Error as CoreError, Result as CoreResult};
use sinus_core::mel::MelPatch;
use sinus_core::store::Store;
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
    /// Held for the engine's lifetime. Dropping it — including on a crash, since
    /// the OS owns the underlying file lock — hands this machine back to whichever
    /// shell the user opens next.
    _instance: InstanceGuard,
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
            version,
        };
        let engine = MonitoringEngine::open(
            &config.database_path,
            embedder,
            MonitoringConfig::new(config.platform.source()),
        )?;
        let reader = Store::open(&config.database_path)?;
        Ok(Arc::new(Self {
            inner: Mutex::new(engine),
            reader: Mutex::new(reader),
            _instance: instance,
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
        Ok(())
    }

    pub fn push_pcm_16k(&self, samples: Vec<f32>) -> Result<Vec<AppleEvent>, AppleEngineError> {
        Ok(self
            .lock()?
            .push_pcm_16k(&samples)?
            .into_iter()
            .map(AppleEvent::from)
            .collect())
    }

    pub fn stop_monitoring(&self) -> Result<Vec<AppleEvent>, AppleEngineError> {
        Ok(self
            .lock()?
            .stop_session()?
            .into_iter()
            .map(AppleEvent::from)
            .collect())
    }

    pub fn is_monitoring(&self) -> Result<bool, AppleEngineError> {
        Ok(self.lock()?.is_monitoring())
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
}
