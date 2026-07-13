//! End-to-end offline pipeline (SPEC §4). Runs the *identical* stages as the live
//! app — gate ① → log-mel ② → embed ③ → decision ④ → sessionizer ⑤ — over a
//! buffer of 16 kHz mono samples, producing detected events. The CLI `classify`
//! command and the golden-corpus test both drive this so tests cover the real
//! path without needing `yamnet.onnx` (any [`Embedder`] plugs in).

use chrono::{DateTime, Utc};

use crate::classify::decision::{DecisionEngine, WindowHit, WindowScores};
use crate::classify::embed::Embedder;
use crate::classify::native::AudiosetMap;
use crate::classify::proto::PrototypeMatcher;
use crate::error::Result;
use crate::gate::{Gate, GateConfig, GateEdge};
use crate::mel::{MelFrontend, PATCH_FRAMES, PATCH_HOP_FRAMES};
use crate::session::{DetectedEvent, SessionConfig, Sessionizer, WindowObservation};
use crate::types::{Event, EventType, Source};

/// Pipeline tuning — the per-stage configs (SPEC §4.1).
#[derive(Debug, Clone, Default)]
pub struct PipelineConfig {
    pub gate: GateConfig,
    pub decision: crate::classify::decision::DecisionConfig,
    pub session: SessionConfig,
}

/// Per-window diagnostic record (drives `cli classify` output).
#[derive(Debug, Clone)]
pub struct WindowRecord {
    pub time_ms: i64,
    pub active: bool,
    pub energy_peak: bool,
    pub speech: f32,
    /// Per-class scores, highest first.
    pub scores: Vec<(EventType, f32)>,
    /// The decided firing class, if any.
    pub hit: Option<WindowHit>,
}

/// Result of running the pipeline over a buffer.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub windows: Vec<WindowRecord>,
    pub events: Vec<DetectedEvent>,
}

/// Metadata stamped onto persisted events.
#[derive(Debug, Clone)]
pub struct EventContext {
    pub base_time: DateTime<Utc>,
    pub tz_offset_min: i32,
    pub device_id: String,
    pub source: Source,
    pub model_version: String,
}

/// The offline pipeline. Generic over the backbone so tests use a deterministic
/// [`Embedder`] and production uses ONNX YAMNet.
pub struct Pipeline<E: Embedder> {
    cfg: PipelineConfig,
    mel: MelFrontend,
    embedder: E,
    decision: DecisionEngine,
    proto: Option<PrototypeMatcher>,
    audioset_map: AudiosetMap,
}

impl<E: Embedder> Pipeline<E> {
    pub fn new(cfg: PipelineConfig, embedder: E) -> Self {
        let decision = DecisionEngine::new(cfg.decision.clone());
        Pipeline {
            cfg,
            mel: MelFrontend::new(crate::types::SAMPLE_RATE),
            embedder,
            decision,
            proto: None,
            audioset_map: AudiosetMap::default(),
        }
    }

    /// Attach a prototype matcher for the enrolled custom classes (SPEC §5 B-lite).
    pub fn with_prototypes(mut self, proto: PrototypeMatcher) -> Self {
        self.proto = Some(proto);
        self
    }

    pub fn model_version(&self) -> String {
        self.embedder.model_version()
    }

    /// Run the gate over the whole signal, returning per-hop `(active, peak)`.
    /// `active` includes the tail (the gate holds open) and a pre-roll extension
    /// back from each opening edge (SPEC §4.1 stage ①).
    fn run_gate(&self, samples: &[f32]) -> (Vec<bool>, Vec<bool>) {
        let hop = self.cfg.gate.hop_samples();
        let preroll = self.cfg.gate.preroll_hops();
        let mut gate = Gate::new(self.cfg.gate.clone());
        let n_hops = samples.len() / hop;
        let mut active = vec![false; n_hops];
        let mut peak = vec![false; n_hops];
        let mut opening_edges = Vec::new();
        for h in 0..n_hops {
            let report = gate.process_hop(&samples[h * hop..(h + 1) * hop]);
            active[h] = report.open;
            peak[h] = report.energy_peak;
            if report.edge == Some(GateEdge::Opened) {
                opening_edges.push(h);
            }
        }
        // Extend `active` backward from each opening edge to include the pre-roll.
        for h in opening_edges {
            let start = h.saturating_sub(preroll);
            for a in active.iter_mut().take(h).skip(start) {
                *a = true;
            }
        }
        (active, peak)
    }

    /// Process a buffer of 16 kHz mono samples.
    pub fn process(&self, samples: &[f32]) -> Result<PipelineResult> {
        let (active, peak) = self.run_gate(samples);
        let frames = self.mel.log_mel_frames(samples);
        let patches = MelFrontend::frames_to_patches(&frames);

        // 0.5 s patch hop = 10 gate hops (50 ms); a 0.96 s patch spans ~20 hops.
        let hops_per_patch_hop =
            (PATCH_HOP_FRAMES * crate::mel::HOP_LEN) / self.cfg.gate.hop_samples();
        let patch_span_hops = (PATCH_FRAMES * crate::mel::HOP_LEN) / self.cfg.gate.hop_samples();

        let mut sessionizer = Sessionizer::new(self.cfg.session.clone());
        let mut windows = Vec::with_capacity(patches.len());
        let mut events = Vec::new();

        for (p, patch) in patches.iter().enumerate() {
            let time_ms = (p * PATCH_HOP_FRAMES * crate::mel::HOP_LEN * 1000
                / crate::types::SAMPLE_RATE as usize) as i64;
            let hop_start = p * hops_per_patch_hop;
            let hop_end = (hop_start + patch_span_hops).min(active.len());
            let is_active = active
                .get(hop_start..hop_end)
                .map(|s| s.iter().any(|&a| a))
                .unwrap_or(false);
            let energy_peak = peak
                .get(hop_start..hop_end)
                .map(|s| s.iter().any(|&a| a))
                .unwrap_or(false);

            if !is_active {
                windows.push(WindowRecord {
                    time_ms,
                    active: false,
                    energy_peak,
                    speech: 0.0,
                    scores: Vec::new(),
                    hit: None,
                });
                continue;
            }

            let features = self.embedder.embed(patch, energy_peak)?;
            let native = match &features.audioset_scores {
                Some(a) => self.audioset_map.native_scores(a),
                None => Default::default(),
            };
            let proto_hit = self
                .proto
                .as_ref()
                .and_then(|m| m.best_match(&features.embedding));
            let proto_vec: Vec<(EventType, f32)> = proto_hit.into_iter().collect();

            let scores = WindowScores::merge(&native, &proto_vec, energy_peak);
            let hit = self.decision.decide(&scores);

            let mut sorted: Vec<(EventType, f32)> =
                scores.scores.iter().map(|(&k, &v)| (k, v)).collect();
            // `total_cmp` is NaN-safe: a zero-norm embedding can make a cosine score
            // NaN, which would panic `partial_cmp().unwrap()` (SPEC §4.1 ④).
            sorted.sort_by(|a, b| b.1.total_cmp(&a.1));

            if let Some(h) = hit {
                for ev in sessionizer.observe(WindowObservation {
                    event_type: h.event_type,
                    confidence: h.confidence,
                    timestamp_ms: time_ms,
                    energy_peak,
                }) {
                    events.push(ev);
                }
            }

            windows.push(WindowRecord {
                time_ms,
                active: true,
                energy_peak,
                speech: scores.speech,
                scores: sorted,
                hit,
            });
        }

        events.extend(sessionizer.flush());
        Ok(PipelineResult { windows, events })
    }

    /// Convert a detected event to a persistable [`Event`] with metadata.
    pub fn to_event(&self, d: &DetectedEvent, ctx: &EventContext) -> Event {
        Event {
            uuid: uuid::Uuid::new_v4().to_string(),
            event_type: d.event_type,
            occurred_at: ctx.base_time + chrono::Duration::milliseconds(d.start_ms),
            tz_offset_min: ctx.tz_offset_min,
            duration_ms: d.duration_ms,
            confidence: d.confidence,
            burst_count: d.burst_count,
            model_version: ctx.model_version.clone(),
            source: ctx.source,
            device_id: ctx.device_id.clone(),
            uploaded_at: None,
            deleted: false,
            reject_count: 0,
            rejected_at: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::embed::BandHeuristicEmbedder;
    use crate::synth;

    /// Silence in → gate stays shut → no windows analyzed, no events.
    #[test]
    fn silence_produces_no_events() {
        let pipe = Pipeline::new(PipelineConfig::default(), BandHeuristicEmbedder);
        let silence = synth::silence(16_000 * 3);
        let result = pipe.process(&silence).unwrap();
        assert!(result.events.is_empty());
        assert!(result.windows.iter().all(|w| !w.active));
    }

    /// A loud cough-frequency burst surrounded by quiet → exactly one cough event.
    #[test]
    fn cough_tone_burst_yields_one_cough_event() {
        let pipe = Pipeline::new(PipelineConfig::default(), BandHeuristicEmbedder);
        let mut sig = synth::white_noise(16_000, 0.003, 1); // ~1 s quiet
        sig.extend(synth::sine(16_000 * 2, 16_000, 300.0, 0.6)); // 2 s cough-band tone
        sig.extend(synth::white_noise(16_000, 0.003, 2)); // ~1 s quiet
        let result = pipe.process(&sig).unwrap();
        assert_eq!(result.events.len(), 1, "events: {:?}", result.events);
        assert_eq!(result.events[0].event_type, EventType::Cough);
        assert!(result.windows.iter().any(|w| w.active));
    }
}
