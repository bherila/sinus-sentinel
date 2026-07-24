//! UniFFI boundary for native Apple clients.
//!
//! Swift owns capture, lifecycle, Core ML, and presentation. Rust accepts
//! converted 16 kHz mono PCM and owns the detector, persistence, and projections.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use chrono::{DateTime, Utc};
use sinus_app::monitor::{MonitoringConfig, MonitoringEngine};
use sinus_core::classify::embed::{Embedder, WindowFeatures, AUDIOSET_CLASSES, EMBED_DIM};
use sinus_core::error::{Error as CoreError, Result as CoreResult};
use sinus_core::mel::MelPatch;
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

#[derive(Debug, Clone, uniffi::Record)]
pub struct AppleEngineConfig {
    pub database_path: String,
    pub device_id: String,
    pub platform: ApplePlatform,
    pub sensitivity: f32,
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
}

impl From<CoreError> for AppleEngineError {
    fn from(value: CoreError) -> Self {
        Self::Engine {
            message: value.to_string(),
        }
    }
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct AppleEvent {
    pub uuid: String,
    pub event_type: AppleEventType,
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
        Self {
            uuid: value.uuid,
            event_type,
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
}

#[derive(uniffi::Object)]
pub struct AppleEngine {
    inner: Mutex<MonitoringEngine<ForeignEmbedder>>,
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
        if config.device_id.trim().is_empty() {
            return Err(AppleEngineError::InvalidArgument {
                message: "device_id cannot be empty".to_string(),
            });
        }
        let version = model
            .model_version()
            .map_err(|error| AppleEngineError::Model {
                message: error.to_string(),
            })?;
        let embedder = ForeignEmbedder {
            runner: model,
            version,
        };
        let mut monitoring_config =
            MonitoringConfig::new(config.platform.source(), config.device_id);
        monitoring_config.sensitivity = config.sensitivity;
        let engine = MonitoringEngine::open(&config.database_path, embedder, monitoring_config)?;
        Ok(Arc::new(Self {
            inner: Mutex::new(engine),
        }))
    }

    pub fn start_monitoring(
        &self,
        started_at_epoch_ms: i64,
        timezone_offset_minutes: i32,
    ) -> Result<(), AppleEngineError> {
        let started_at = timestamp(started_at_epoch_ms)?;
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
        if !(-1_439..=1_439).contains(&timezone_offset_minutes) {
            return Err(AppleEngineError::InvalidArgument {
                message: "timezone offset must be between -1439 and 1439 minutes".to_string(),
            });
        }
        let now = timestamp(now_epoch_ms)?;
        let engine = self.lock()?;
        let today =
            sinus_app::state::today_counts_at_offset(engine.store(), now, timezone_offset_minutes);
        let histogram = sinus_app::state::daily_histogram_at_offset(
            engine.store(),
            days as i64,
            now,
            timezone_offset_minutes,
        );
        let recent_events = engine
            .store()
            .recent_events(now - chrono::Duration::days(days as i64), now)?
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
        })
    }

    pub fn get_setting(&self, key: String) -> Result<Option<String>, AppleEngineError> {
        Ok(self.lock()?.store().setting_get(&key)?)
    }

    pub fn set_setting(&self, key: String, value: String) -> Result<(), AppleEngineError> {
        self.lock()?.store().setting_set(&key, &value)?;
        Ok(())
    }
}

impl AppleEngine {
    fn lock(&self) -> Result<MutexGuard<'_, MonitoringEngine<ForeignEmbedder>>, AppleEngineError> {
        self.inner.lock().map_err(|_| AppleEngineError::Engine {
            message: "engine state lock was poisoned".to_string(),
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

    #[test]
    fn bridge_rejects_pcm_before_session_start() {
        let path =
            std::env::temp_dir().join(format!("sinus-apple-test-{}.db", uuid::Uuid::new_v4()));
        let engine = AppleEngine::new(
            AppleEngineConfig {
                database_path: path.to_string_lossy().into_owned(),
                device_id: "test-device".to_string(),
                platform: ApplePlatform::Ios,
                sensitivity: 0.5,
            },
            Arc::new(TestModel),
        )
        .unwrap();
        let error = engine.push_pcm_16k(vec![0.0; 800]).unwrap_err();
        assert!(error.to_string().contains("start a monitoring session"));
        drop(engine);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bridge_session_emits_and_projects_an_event() {
        let path =
            std::env::temp_dir().join(format!("sinus-apple-test-{}.db", uuid::Uuid::new_v4()));
        let engine = AppleEngine::new(
            AppleEngineConfig {
                database_path: path.to_string_lossy().into_owned(),
                device_id: "test-device".to_string(),
                platform: ApplePlatform::Ios,
                sensitivity: 0.5,
            },
            Arc::new(TestModel),
        )
        .unwrap();
        let now = Utc::now();
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
        assert!(matches!(emitted[0].event_type, AppleEventType::Cough));
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
        let _ = std::fs::remove_file(path);
    }
}
