//! A platform-neutral monitoring session.
//!
//! The host captures and converts microphone audio to 16 kHz mono `f32` PCM,
//! then feeds it here. Keeping device APIs outside this crate lets macOS and iOS
//! use `AVAudioSession`/`AVAudioEngine` while the detector and persistence remain
//! identical across clients.

use std::path::Path;

use chrono::{DateTime, Utc};
use sinus_core::classify::embed::Embedder;
use sinus_core::classify::proto::PrototypeMatcher;
use sinus_core::error::{Error, Result};
use sinus_core::pipeline::{EventContext, PipelineConfig, StreamingPipeline};
use sinus_core::store::Store;
use sinus_core::types::{Event, Source};

const PROTOTYPE_SIM_THRESHOLD: f32 = 0.65;
const PROTOTYPE_NEGATIVE_MARGIN: f32 = 0.05;

#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub source: Source,
    pub device_id: String,
    pub sensitivity: f32,
}

impl MonitoringConfig {
    pub fn new(source: Source, device_id: impl Into<String>) -> Self {
        Self {
            source,
            device_id: device_id.into(),
            sensitivity: 0.5,
        }
    }
}

/// Owns one store connection and one streaming detector. It deliberately does
/// not open a microphone or spawn a thread; those are platform lifecycle
/// responsibilities.
pub struct MonitoringEngine<E: Embedder> {
    store: Store,
    pipeline: StreamingPipeline<E>,
    config: MonitoringConfig,
    context: Option<EventContext>,
}

impl<E: Embedder> MonitoringEngine<E> {
    pub fn open(db_path: impl AsRef<Path>, embedder: E, config: MonitoringConfig) -> Result<Self> {
        Self::from_store(Store::open(db_path)?, embedder, config)
    }

    pub fn from_store(store: Store, embedder: E, config: MonitoringConfig) -> Result<Self> {
        let mut pipeline = StreamingPipeline::new(PipelineConfig::default(), embedder);
        pipeline.set_sensitivity(config.sensitivity);
        if let Some(prototypes) = prototypes_from_store(&store)? {
            pipeline = pipeline.with_prototypes(prototypes);
        }
        Ok(Self {
            store,
            pipeline,
            config,
            context: None,
        })
    }

    pub fn start_session(&mut self, started_at: DateTime<Utc>, tz_offset_min: i32) {
        self.pipeline.reset_stream();
        self.context = Some(EventContext {
            base_time: started_at,
            tz_offset_min,
            device_id: self.config.device_id.clone(),
            source: self.config.source,
            model_version: self.pipeline.model_version(),
        });
    }

    pub fn is_monitoring(&self) -> bool {
        self.context.is_some()
    }

    pub fn push_pcm_16k(&mut self, samples: &[f32]) -> Result<Vec<Event>> {
        if self.context.is_none() {
            return Err(Error::Config(
                "start a monitoring session before pushing audio".to_string(),
            ));
        }
        let detected = self.pipeline.push(samples)?;
        self.persist(detected)
    }

    /// Flush tail state, persist any final event, and make subsequent PCM invalid
    /// until a new session is explicitly started.
    pub fn stop_session(&mut self) -> Result<Vec<Event>> {
        if self.context.is_none() {
            return Ok(Vec::new());
        }
        let detected = self.pipeline.flush()?;
        let events = self.persist(detected)?;
        self.context = None;
        self.pipeline.reset_stream();
        Ok(events)
    }

    pub fn set_sensitivity(&mut self, sensitivity: f32) -> Result<()> {
        let sensitivity = sensitivity.clamp(0.0, 1.0);
        self.pipeline.set_sensitivity(sensitivity);
        self.config.sensitivity = sensitivity;
        self.store
            .setting_set("sensitivity", &sensitivity.to_string())
    }

    pub fn reload_enrollments(&mut self) -> Result<()> {
        self.pipeline
            .set_prototypes(prototypes_from_store(&self.store)?);
        Ok(())
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    fn persist(&mut self, detected: Vec<sinus_core::session::DetectedEvent>) -> Result<Vec<Event>> {
        let context = self
            .context
            .as_ref()
            .ok_or_else(|| Error::Config("monitoring session is not active".to_string()))?;
        let mut events = Vec::with_capacity(detected.len());
        for detected in detected {
            let event = self.pipeline.to_event(&detected, context);
            self.store.insert_event(&event)?;
            self.store
                .put_event_embedding(&event.uuid, &detected.embedding)?;
            events.push(event);
        }
        Ok(events)
    }
}

fn prototypes_from_store(store: &Store) -> Result<Option<PrototypeMatcher>> {
    let enrollments: Vec<_> = store
        .enrollments()?
        .into_iter()
        .map(|stored| stored.enrollment)
        .collect();
    Ok((!enrollments.is_empty()).then(|| {
        PrototypeMatcher::from_enrollments(
            &enrollments,
            PROTOTYPE_SIM_THRESHOLD,
            PROTOTYPE_NEGATIVE_MARGIN,
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sinus_core::classify::embed::BandHeuristicEmbedder;
    use sinus_core::synth;
    use sinus_core::types::EventType;

    fn engine() -> MonitoringEngine<BandHeuristicEmbedder> {
        MonitoringEngine::from_store(
            Store::open_in_memory().unwrap(),
            BandHeuristicEmbedder,
            MonitoringConfig::new(Source::MobileIos, "test-device"),
        )
        .unwrap()
    }

    #[test]
    fn pcm_requires_an_explicit_session() {
        let mut engine = engine();
        let error = engine.push_pcm_16k(&[0.0; 800]).unwrap_err();
        assert!(error.to_string().contains("start a monitoring session"));
    }

    #[test]
    fn session_persists_events_and_stops_cleanly() {
        let mut engine = engine();
        let now = Utc::now();
        engine.start_session(now, 0);

        let mut signal = synth::white_noise(16_000, 0.003, 1);
        signal.extend(synth::sine(32_000, 16_000, 300.0, 0.6));
        signal.extend(synth::white_noise(16_000, 0.003, 2));

        let mut emitted = Vec::new();
        for chunk in signal.chunks(777) {
            emitted.extend(engine.push_pcm_16k(chunk).unwrap());
        }
        emitted.extend(engine.stop_session().unwrap());

        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0].event_type, EventType::Cough);
        assert_eq!(emitted[0].source, Source::MobileIos);
        assert_eq!(engine.store().event_count().unwrap(), 1);
        assert!(!engine.is_monitoring());
    }
}
