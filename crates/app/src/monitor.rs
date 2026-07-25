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
use sinus_core::types::{Event, EventType, Source};

use crate::teach::{self, TeachResult};

/// Cosine similarity a personalized prototype must reach to fire, and how far a
/// negative prototype must beat the positive to veto it. Shared so the desktop
/// capture thread, the Apple bridge and Teach-mode scoring cannot drift apart —
/// a take enrolled under one threshold and matched under another would score
/// differently in the UI than in the detector.
pub const PROTOTYPE_SIM_THRESHOLD: f32 = 0.65;
pub const PROTOTYPE_NEGATIVE_MARGIN: f32 = 0.05;

/// What the *platform* contributes. Everything else — sensitivity, the device
/// identity, the battery policy — is read from the store the shells share, so
/// two clients on one machine cannot disagree about it.
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub source: Source,
}

impl MonitoringConfig {
    pub fn new(source: Source) -> Self {
        Self { source }
    }
}

/// Owns one store connection and one streaming detector. It deliberately does
/// not open a microphone or spawn a thread; those are platform lifecycle
/// responsibilities.
pub struct MonitoringEngine<E: Embedder> {
    store: Store,
    pipeline: StreamingPipeline<E>,
    config: MonitoringConfig,
    device_id: String,
    context: Option<EventContext>,
    suppress_persistence: bool,
}

impl<E: Embedder> MonitoringEngine<E> {
    pub fn open(db_path: impl AsRef<Path>, embedder: E, config: MonitoringConfig) -> Result<Self> {
        Self::from_store(Store::open(db_path)?, embedder, config)
    }

    pub fn from_store(store: Store, embedder: E, config: MonitoringConfig) -> Result<Self> {
        let mut pipeline = StreamingPipeline::new(PipelineConfig::default(), embedder);
        // Sensitivity is a synced, user-visible setting: honor whatever the store
        // holds instead of letting each shell impose its own starting value.
        pipeline.set_sensitivity(crate::settings::sensitivity(&store));
        if let Some(prototypes) = prototypes_from_store(&store)? {
            pipeline = pipeline.with_prototypes(prototypes);
        }
        let device_id = crate::settings::ensure_device_id(&store);
        Ok(Self {
            store,
            pipeline,
            config,
            device_id,
            context: None,
            suppress_persistence: false,
        })
    }

    pub fn start_session(&mut self, started_at: DateTime<Utc>, tz_offset_min: i32) {
        self.pipeline.reset_stream();
        self.context = Some(EventContext {
            base_time: started_at,
            tz_offset_min,
            device_id: self.device_id.clone(),
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
        let result = self.stop_session_inner();
        // Cleared unconditionally, even on the error path above: a crash or a
        // forgotten un-suppress must not leave the *next* session permanently mute.
        self.suppress_persistence = false;
        result
    }

    fn stop_session_inner(&mut self) -> Result<Vec<Event>> {
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
        crate::settings::set_sensitivity(&self.store, sensitivity)
    }

    pub fn reload_enrollments(&mut self) -> Result<()> {
        self.pipeline
            .set_prototypes(prototypes_from_store(&self.store)?);
        Ok(())
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    /// The energy gate is open — a sound is arriving and is being classified.
    /// Drives the UI's "heard something" indicator.
    pub fn gate_open(&self) -> bool {
        self.pipeline.gate_open()
    }

    pub fn model_version(&self) -> String {
        self.pipeline.model_version()
    }

    /// While set, detections are still computed — so the gate, the noise floor and
    /// the sessionizer's cooldowns stay continuous — but nothing is persisted.
    /// Quiet hours and Teach takes both need exactly this; tearing the stream down
    /// instead would reset the sample clock and skew later event timestamps.
    pub fn set_suppress_persistence(&mut self, suppress: bool) {
        self.suppress_persistence = suppress;
    }

    pub fn suppress_persistence(&self) -> bool {
        self.suppress_persistence
    }

    /// Re-read the settings a PHR pull can change under us (today: sensitivity).
    pub fn reload_settings(&mut self) -> Result<()> {
        self.pipeline
            .set_sensitivity(crate::settings::sensitivity(&self.store));
        Ok(())
    }

    /// Score and persist a Teach take against this engine's own pipeline and store,
    /// then reload prototypes so the new example takes effect immediately.
    /// Always clears `suppress_persistence` — a take that fails must not leave the
    /// detector permanently mute.
    pub fn enroll_take(&mut self, class: EventType, samples: &[f32]) -> Result<TeachResult> {
        let result =
            teach::enroll_take(&self.store, &self.pipeline, class, samples).and_then(|result| {
                self.pipeline
                    .set_prototypes(prototypes_from_store(&self.store)?);
                Ok(result)
            });
        // Cleared on both branches: a take that fails must not leave the detector
        // permanently mute (see `set_suppress_persistence` doc comment).
        self.suppress_persistence = false;
        result
    }

    fn persist(&mut self, detected: Vec<sinus_core::session::DetectedEvent>) -> Result<Vec<Event>> {
        // The pipeline must run regardless of suppression (see push_pcm_16k) so the
        // gate, noise floor and cooldowns stay continuous; only the store write is
        // skipped here.
        if self.suppress_persistence {
            return Ok(Vec::new());
        }
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

/// Build the personalized matcher from every stored Teach-mode enrollment, or
/// `None` when the user has taught nothing yet (in which case the pipeline runs
/// the generic decision rules alone).
pub fn prototypes_from_store(store: &Store) -> Result<Option<PrototypeMatcher>> {
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

    fn engine() -> MonitoringEngine<BandHeuristicEmbedder> {
        MonitoringEngine::from_store(
            Store::open_in_memory().unwrap(),
            BandHeuristicEmbedder,
            MonitoringConfig::new(Source::MobileIos),
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

    /// The cough signal `session_persists_events_and_stops_cleanly` relies on —
    /// factored out so the suppression test can push it twice.
    fn cough_signal() -> Vec<f32> {
        let mut signal = synth::white_noise(16_000, 0.003, 1);
        signal.extend(synth::sine(32_000, 16_000, 300.0, 0.6));
        signal.extend(synth::white_noise(16_000, 0.003, 2));
        signal
    }

    #[test]
    fn suppressed_pcm_still_advances_the_pipeline_but_persists_nothing() {
        let mut engine = engine();
        let signal = cough_signal();

        engine.start_session(Utc::now(), 0);
        engine.set_suppress_persistence(true);
        assert!(engine.suppress_persistence());
        for chunk in signal.chunks(777) {
            // The pipeline still runs — and returns its usual detections — under
            // suppression; only the store write is skipped.
            engine.push_pcm_16k(chunk).unwrap();
        }
        engine.stop_session().unwrap();
        assert_eq!(engine.store().event_count().unwrap(), 0);

        // Clearing suppression and feeding the identical signal again must now
        // persist, proving the flag alone gated the write, not some broken state
        // left over from the suppressed pass.
        engine.set_suppress_persistence(false);
        assert!(!engine.suppress_persistence());
        engine.start_session(Utc::now(), 0);
        let mut emitted = Vec::new();
        for chunk in signal.chunks(777) {
            emitted.extend(engine.push_pcm_16k(chunk).unwrap());
        }
        emitted.extend(engine.stop_session().unwrap());
        assert_eq!(emitted.len(), 1);
        assert_eq!(engine.store().event_count().unwrap(), 1);
    }

    #[test]
    fn stop_session_clears_suppression_defensively() {
        let mut active_engine = engine();
        active_engine.start_session(Utc::now(), 0);
        active_engine.set_suppress_persistence(true);
        active_engine.stop_session().unwrap();
        assert!(!active_engine.suppress_persistence());

        // Also clears when no session was active — a stray cancel-before-start
        // must not leave a future session muted.
        let mut idle_engine = engine();
        idle_engine.set_suppress_persistence(true);
        idle_engine.stop_session().unwrap();
        assert!(!idle_engine.suppress_persistence());
    }

    /// A cough score that only clears the effective threshold at neutral-or-higher
    /// sensitivity: cough's base threshold is 0.30, and the sensitivity factor
    /// ranges ×0.5 (loosest) to ×1.5 (strictest), so 0.35 clears 0.30 (neutral)
    /// but not 0.45 (sensitivity 0.0). `MockEmbedder` fixes the score regardless
    /// of the audio fed in, which is what lets a plain amplitude signal (needed
    /// to open the real energy gate) drive a controlled decision score.
    fn mock_cough_engine(cough_score: f32) -> MonitoringEngine<sinus_core::classify::MockEmbedder> {
        use sinus_core::classify::{
            embed::AUDIOSET_CLASSES, AudiosetMap, MockEmbedder, WindowFeatures, EMBED_DIM,
        };

        let mut audioset_scores = vec![0.0f32; AUDIOSET_CLASSES];
        audioset_scores[AudiosetMap::default().cough] = cough_score;
        let embedder = MockEmbedder {
            features: WindowFeatures {
                audioset_scores: Some(audioset_scores),
                embedding: vec![0.0; EMBED_DIM],
                energy_peak: false,
            },
            version: "mock@test".to_string(),
        };
        MonitoringEngine::from_store(
            Store::open_in_memory().unwrap(),
            embedder,
            MonitoringConfig::new(Source::MobileIos),
        )
        .unwrap()
    }

    #[test]
    fn reload_settings_picks_up_a_sensitivity_written_directly_to_the_store() {
        let mut engine = mock_cough_engine(0.35);
        let signal = cough_signal();

        // At the store's default sensitivity (0.5, neutral factor) 0.35 clears
        // cough's 0.30 threshold and fires.
        engine.start_session(Utc::now(), 0);
        let mut emitted = Vec::new();
        for chunk in signal.chunks(777) {
            emitted.extend(engine.push_pcm_16k(chunk).unwrap());
        }
        emitted.extend(engine.stop_session().unwrap());
        assert_eq!(
            emitted.len(),
            1,
            "0.35 should clear cough's threshold at neutral sensitivity"
        );

        // Written straight to the store, bypassing `set_sensitivity`, the way a
        // PHR pull lands a synced value: the live pipeline must not see it until
        // `reload_settings` is called.
        crate::settings::set_sensitivity(engine.store(), 0.0).unwrap();
        engine.start_session(Utc::now(), 0);
        let mut emitted = Vec::new();
        for chunk in signal.chunks(777) {
            emitted.extend(engine.push_pcm_16k(chunk).unwrap());
        }
        emitted.extend(engine.stop_session().unwrap());
        assert_eq!(
            emitted.len(),
            1,
            "unreloaded pipeline should still be at the old sensitivity"
        );

        engine.reload_settings().unwrap();
        engine.start_session(Utc::now(), 0);
        let mut emitted = Vec::new();
        for chunk in signal.chunks(777) {
            emitted.extend(engine.push_pcm_16k(chunk).unwrap());
        }
        emitted.extend(engine.stop_session().unwrap());
        assert!(
            emitted.is_empty(),
            "reload_settings should have applied sensitivity=0.0, raising the \
             threshold above 0.35"
        );
    }

    #[test]
    fn enroll_take_scores_a_live_take_and_clears_suppression() {
        let mut engine = engine();
        engine.set_suppress_persistence(true);

        let samples = synth::sine(48_000, 16_000, 300.0, 0.6);
        let result = engine.enroll_take(EventType::Cough, &samples).unwrap();

        assert_eq!(result.class, EventType::Cough);
        assert_eq!(result.examples, 1);
        assert!(!engine.suppress_persistence());
    }
}
