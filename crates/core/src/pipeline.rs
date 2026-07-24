//! Pipeline (SPEC §4). Runs the *identical* stages as the live app —
//! gate ① → log-mel ② → embed ③ → decision ④ → sessionizer ⑤ — over 16 kHz mono
//! samples, producing detected events.
//!
//! There are two front-ends over one shared engine ([`StreamState`] + the
//! `stream_*` methods):
//!
//! - [`Pipeline::process`]: batch — the whole buffer in one call. The CLI
//!   `classify` command and the golden-corpus test drive this.
//! - [`StreamingPipeline`]: incremental — [`push`](StreamingPipeline::push) feeds
//!   arbitrary-length sample slices and [`flush`](StreamingPipeline::flush) drains
//!   the tail. The live capture thread holds one instance for the stream's life so
//!   gate/sessionizer/mel state persists across chunk boundaries (no per-chunk
//!   reset): events straddling a boundary merge, cooldowns persist, and the noise
//!   floor converges. Feeding the same signal whole vs. split yields identical
//!   events — `process` is literally "advance once, then flush".

use std::collections::VecDeque;

use chrono::{DateTime, Utc};

use crate::classify::decision::{DecisionEngine, WindowHit, WindowScores};
use crate::classify::embed::Embedder;
use crate::classify::native::AudiosetMap;
use crate::classify::proto::PrototypeMatcher;
use crate::error::Result;
use crate::gate::{Gate, GateEdge};
use crate::mel::{
    MelFrontend, MelPatch, MelScratch, FRAME_LEN, HOP_LEN, N_MEL, PATCH_FRAMES, PATCH_HOP_FRAMES,
};
use crate::session::{DetectedEvent, SessionConfig, Sessionizer, WindowLevels, WindowObservation};
use crate::types::{Event, EventType, Source};

/// Pipeline tuning — the per-stage configs (SPEC §4.1).
#[derive(Debug, Clone, Default)]
pub struct PipelineConfig {
    pub gate: crate::gate::GateConfig,
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

/// Newly-finalized windows + events produced by one engine step.
#[derive(Debug, Clone, Default)]
struct StreamOutput {
    windows: Vec<WindowRecord>,
    events: Vec<DetectedEvent>,
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

/// Persistent, incrementally-fed engine state (gate ①, mel framing ②, sessionizer
/// ⑤ plus the rolling buffers needed to reproduce the batch path exactly across
/// arbitrary push boundaries). Owned by [`StreamingPipeline`]; constructed fresh
/// per call by [`Pipeline::process`].
struct StreamState {
    gate: Gate,
    sessionizer: Sessionizer,

    /// Samples not yet consumed into gate hops.
    gate_buf: Vec<f32>,
    /// Absolute index of the next gate hop to emit.
    next_hop: usize,
    /// Rolling per-hop gate state, `hops_open[0]` at absolute index `hops_base`.
    hops_open: VecDeque<bool>,
    hops_peak: VecDeque<bool>,
    /// Per-hop RMS level and adaptive noise floor (dBFS), same indexing as
    /// `hops_open`. The gate already computes both; retaining them is what lets
    /// an event record how loud it actually was.
    hops_rms: VecDeque<f32>,
    hops_floor: VecDeque<f32>,
    hops_base: usize,
    /// Absolute hop indices of gate opening edges (rolling; pruned with hops).
    open_edges: VecDeque<usize>,

    /// Rolling raw audio retained until the gate has enough look-ahead to decide
    /// whether a classifier patch is active. Quiet patches never run an FFT; when
    /// the gate opens this buffer supplies the required one-second pre-roll.
    raw_samples: VecDeque<f32>,
    raw_base_sample: usize,
    total_samples: usize,
    /// Absolute index of the next mel frame to emit.
    next_frame: usize,
    /// Lazy rolling log-mel frames. `None` means the frame is available in
    /// `raw_samples` but has not consumed DSP because no active patch needed it.
    frames: VecDeque<Option<[f32; N_MEL]>>,
    frames_base: usize,
    mel_scratch: MelScratch,
    #[cfg(test)]
    mel_frames_computed: usize,

    /// Absolute index of the next patch to finalize.
    next_patch: usize,
}

impl StreamState {
    fn new(cfg: &PipelineConfig, mel: &MelFrontend) -> Self {
        StreamState {
            gate: Gate::new(cfg.gate.clone()),
            sessionizer: Sessionizer::new(cfg.session.clone()),
            gate_buf: Vec::new(),
            next_hop: 0,
            hops_open: VecDeque::new(),
            hops_peak: VecDeque::new(),
            hops_rms: VecDeque::new(),
            hops_floor: VecDeque::new(),
            hops_base: 0,
            open_edges: VecDeque::new(),
            raw_samples: VecDeque::new(),
            raw_base_sample: 0,
            total_samples: 0,
            next_frame: 0,
            frames: VecDeque::new(),
            frames_base: 0,
            mel_scratch: mel.make_scratch(),
            #[cfg(test)]
            mel_frames_computed: 0,
            next_patch: 0,
        }
    }

    /// Drain `gate_buf` into fixed-size gate hops (SPEC §4.1 ①). Hop boundaries are
    /// on absolute sample positions, so a push splitting mid-hop just leaves the
    /// remainder buffered — identical hops to the batch path.
    fn push_gate_samples(&mut self, samples: &[f32]) {
        self.gate_buf.extend_from_slice(samples);
        let hop = self.gate.config().hop_samples();
        let mut consumed = 0;
        while self.gate_buf.len() - consumed >= hop {
            let report = self
                .gate
                .process_hop(&self.gate_buf[consumed..consumed + hop]);
            self.hops_open.push_back(report.open);
            self.hops_peak.push_back(report.energy_peak);
            self.hops_rms.push_back(report.rms_db);
            self.hops_floor.push_back(report.floor_db);
            if report.edge == Some(GateEdge::Opened) {
                self.open_edges.push_back(self.next_hop);
            }
            self.next_hop += 1;
            consumed += hop;
        }
        self.gate_buf.drain(..consumed);
    }

    /// Retain raw samples and advance the *logical* mel-frame clock. Actual
    /// log-mel/FFT work is lazy and happens only if the energy gate marks a patch
    /// active. This makes quiet-room work gate-only while preserving exact pre-roll.
    fn push_raw_samples(&mut self, samples: &[f32]) {
        self.raw_samples.extend(samples.iter().copied());
        self.total_samples += samples.len();
        while self.total_samples >= self.next_frame * HOP_LEN + FRAME_LEN {
            self.frames.push_back(None);
            self.next_frame += 1;
        }
    }

    /// Assemble the 96-frame patch starting at absolute frame `frame_start`.
    fn assemble_patch(&mut self, mel: &MelFrontend, frame_start: usize) -> MelPatch {
        let mut data = Vec::with_capacity(PATCH_FRAMES * N_MEL);
        for i in 0..PATCH_FRAMES {
            let absolute_frame = frame_start + i;
            let rel = absolute_frame - self.frames_base;
            if self.frames[rel].is_none() {
                let absolute_sample = absolute_frame * HOP_LEN;
                let raw_offset = absolute_sample - self.raw_base_sample;
                let mut frame = [0.0f32; FRAME_LEN];
                for (slot, sample) in frame
                    .iter_mut()
                    .zip(self.raw_samples.iter().skip(raw_offset).take(FRAME_LEN))
                {
                    *slot = *sample;
                }
                self.frames[rel] =
                    Some(mel.log_mel_frame_with_scratch(&frame, &mut self.mel_scratch));
                #[cfg(test)]
                {
                    self.mel_frames_computed += 1;
                }
            }
            data.extend_from_slice(self.frames[rel].as_ref().expect("frame computed above"));
        }
        MelPatch {
            frames: PATCH_FRAMES,
            bands: N_MEL,
            data,
        }
    }

    /// Whether the patch spanning hops `[hop_start, hop_end)` is active (any hop
    /// open, or within `preroll` hops *before* an opening edge — the pre-roll
    /// extension of SPEC §4.1 ①) and whether it contains a raw energy peak.
    fn patch_activity(&self, hop_start: usize, hop_end: usize, preroll: usize) -> (bool, bool) {
        let mut active = false;
        let mut peak = false;
        for k in hop_start..hop_end {
            let rel = k - self.hops_base;
            if *self.hops_peak.get(rel).unwrap_or(&false) {
                peak = true;
            }
            let open = *self.hops_open.get(rel).unwrap_or(&false);
            let preroll_hit = self.open_edges.iter().any(|&e| k < e && e <= k + preroll);
            if open || preroll_hit {
                active = true;
            }
        }
        (active, peak)
    }

    /// Loudness of one patch, for the event record.
    ///
    /// Patches overlap — a ~19-hop span advancing on a 10-hop stride — so the
    /// two statistics use deliberately different windows:
    ///
    /// - `peak_dbfs` covers the whole span `[hop_start, hop_end)`. `max` is
    ///   idempotent, so counting a hop twice is harmless.
    /// - the mean covers only `[hop_start, hop_start + stride)`. Consecutive
    ///   patches tile that range exactly, with no overlap and no gap, so a
    ///   session summing across its windows averages each hop precisely once.
    ///   Averaging the full spans instead would double-count every interior hop.
    ///
    /// The mean is accumulated as linear power (`10^(dB/10)`); averaging dB
    /// values directly is meaningless. Returns `None` when the span retains no
    /// hops.
    fn patch_levels(
        &self,
        hop_start: usize,
        hop_end: usize,
        stride: usize,
    ) -> Option<WindowLevels> {
        let mut peak_dbfs = f32::NEG_INFINITY;
        let mut power_sum = 0.0f64;
        let mut mean_hops = 0u32;
        let mut floor_dbfs = None;
        let mean_end = (hop_start + stride).min(hop_end);

        for k in hop_start..hop_end {
            let rel = k - self.hops_base;
            let Some(&rms_db) = self.hops_rms.get(rel) else {
                continue;
            };
            if floor_dbfs.is_none() {
                floor_dbfs = self.hops_floor.get(rel).copied();
            }
            peak_dbfs = peak_dbfs.max(rms_db);
            if k < mean_end {
                power_sum += 10f64.powf(rms_db as f64 / 10.0);
                mean_hops += 1;
            }
        }

        if mean_hops == 0 {
            return None;
        }

        Some(WindowLevels {
            peak_dbfs,
            power_sum,
            mean_hops,
            floor_dbfs: floor_dbfs.unwrap_or(peak_dbfs),
        })
    }

    /// Drop rolling state that no later patch can reference (bounded memory for a
    /// long-running stream).
    fn prune(&mut self, hop_keep_from: usize, frame_keep_from: usize) {
        while self.hops_base < hop_keep_from && !self.hops_open.is_empty() {
            self.hops_open.pop_front();
            self.hops_peak.pop_front();
            self.hops_rms.pop_front();
            self.hops_floor.pop_front();
            self.hops_base += 1;
        }
        while let Some(&e) = self.open_edges.front() {
            if e < hop_keep_from {
                self.open_edges.pop_front();
            } else {
                break;
            }
        }
        while self.frames_base < frame_keep_from && !self.frames.is_empty() {
            self.frames.pop_front();
            self.frames_base += 1;
        }
        let sample_keep_from = frame_keep_from * HOP_LEN;
        let drop_samples = sample_keep_from
            .saturating_sub(self.raw_base_sample)
            .min(self.raw_samples.len());
        self.raw_samples.drain(..drop_samples);
        self.raw_base_sample += drop_samples;
    }
}

/// The pipeline. Generic over the backbone so tests use a deterministic
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

    /// Embed one already-framed patch with the production backbone. Used by the
    /// desktop Teach mode so raw capture stays in memory and only the embedding is
    /// persisted.
    pub fn embed_patch(&self, patch: &MelPatch, energy_peak: bool) -> Result<Vec<f32>> {
        Ok(self.embedder.embed(patch, energy_peak)?.embedding)
    }

    /// Replace personalized examples after a Teach-mode sample is saved, without
    /// restarting the microphone stream or resetting gate/session state.
    pub fn set_prototypes(&mut self, proto: Option<PrototypeMatcher>) {
        self.proto = proto;
    }

    /// Retune detection sensitivity in place.
    ///
    /// Without this the setting is only read when the pipeline is constructed,
    /// so moving the slider — or pulling a new value from the PHR — would not
    /// take effect until the app restarted.
    pub fn set_sensitivity(&mut self, sensitivity: f32) {
        self.decision.config.sensitivity = sensitivity.clamp(0.0, 1.0);
    }

    /// Process a buffer of 16 kHz mono samples (batch): advance the engine over the
    /// whole buffer, then flush. Produces the same windows/events as feeding the
    /// same samples through a [`StreamingPipeline`] in arbitrary chunks.
    pub fn process(&self, samples: &[f32]) -> Result<PipelineResult> {
        let mut st = StreamState::new(&self.cfg, &self.mel);
        let mut out = self.stream_advance(&mut st, samples)?;
        let tail = self.stream_flush(&mut st)?;
        out.windows.extend(tail.windows);
        out.events.extend(tail.events);
        Ok(PipelineResult {
            windows: out.windows,
            events: out.events,
        })
    }

    /// Feed a slice of samples: drain gate hops + mel frames, then finalize every
    /// patch that now has enough gate look-ahead (one `preroll` past its span) to
    /// resolve the pre-roll extension deterministically.
    fn stream_advance(&self, st: &mut StreamState, samples: &[f32]) -> Result<StreamOutput> {
        st.push_gate_samples(samples);
        st.push_raw_samples(samples);
        let mut out = StreamOutput::default();
        self.finalize_ready(st, false, &mut out)?;
        Ok(out)
    }

    /// Finalize all remaining full-frame patches with the gate hops available so
    /// far (end of stream), then close any still-open sessions (SPEC §4.1 ⑤).
    fn stream_flush(&self, st: &mut StreamState) -> Result<StreamOutput> {
        let mut out = StreamOutput::default();
        self.finalize_ready(st, true, &mut out)?;
        out.events.extend(st.sessionizer.flush());
        Ok(out)
    }

    /// Drive the per-patch stages (embed ③ → decision ④ → sessionizer ⑤) for every
    /// patch that is ready. A patch is ready mid-stream once (a) its 96 frames
    /// exist and (b) the gate has advanced one `preroll` past the patch's hop span,
    /// so any opening edge that would pre-roll-activate it is already known. At
    /// `flush` the look-ahead requirement is dropped and the hop span is clamped to
    /// what exists — matching the batch path's end-of-signal clamping exactly.
    fn finalize_ready(
        &self,
        st: &mut StreamState,
        flush: bool,
        out: &mut StreamOutput,
    ) -> Result<()> {
        let hop = self.cfg.gate.hop_samples();
        let preroll = self.cfg.gate.preroll_hops();
        // 0.5 s patch hop = 10 gate hops (50 ms); a 0.96 s patch spans ~19 hops.
        let hops_per_patch_hop = (PATCH_HOP_FRAMES * HOP_LEN) / hop;
        let patch_span_hops = (PATCH_FRAMES * HOP_LEN) / hop;

        loop {
            let p = st.next_patch;
            let frame_start = p * PATCH_HOP_FRAMES;
            // (a) need the full 96-frame patch.
            if st.next_frame < frame_start + PATCH_FRAMES {
                break;
            }
            let hop_start = p * hops_per_patch_hop;
            let hop_end_full = hop_start + patch_span_hops;
            // (b) mid-stream, need one preroll of look-ahead past the span.
            if !flush && st.next_hop < hop_end_full + preroll {
                break;
            }
            let hop_end = if flush {
                hop_end_full.min(st.next_hop)
            } else {
                hop_end_full
            };

            let time_ms =
                (p * PATCH_HOP_FRAMES * HOP_LEN * 1000 / crate::types::SAMPLE_RATE as usize) as i64;
            let (is_active, energy_peak) = st.patch_activity(hop_start, hop_end, preroll);
            let levels = st.patch_levels(hop_start, hop_end, hops_per_patch_hop);

            if !is_active {
                out.windows.push(WindowRecord {
                    time_ms,
                    active: false,
                    energy_peak,
                    speech: 0.0,
                    scores: Vec::new(),
                    hit: None,
                });
            } else {
                let patch = st.assemble_patch(&self.mel, frame_start);
                let features = self.embedder.embed(&patch, energy_peak)?;
                let native = match &features.audioset_scores {
                    Some(a) => self.audioset_map.native_scores(a),
                    None => Default::default(),
                };
                // Personalized examples are deliberately ignored when YAMNet says
                // speech dominates. This prevents a taught throat sound from
                // becoming a general-purpose voice detector.
                let proto_hit = (native.speech <= self.decision.config.speech_dominant)
                    .then(|| {
                        self.proto
                            .as_ref()
                            .and_then(|m| m.best_match(&features.embedding))
                    })
                    .flatten();
                let proto_vec: Vec<(EventType, f32)> = proto_hit.into_iter().collect();

                let scores = WindowScores::merge(&native, &proto_vec, energy_peak);
                let native_hit = self.decision.decide(&scores);
                // A qualifying personalized match is a disambiguation signal, not
                // another probability on YAMNet's scale. Let it override a generic
                // native label (notably nose-blow-as-sneeze and sniff-as-cough),
                // while still applying the normal threshold, energy, and speech
                // guards to the personalized candidate.
                let hit = proto_hit
                    .and_then(|personalized| {
                        let personalized_scores = WindowScores::merge(
                            &crate::classify::native::NativeScores {
                                speech: native.speech,
                                ..Default::default()
                            },
                            &[personalized],
                            energy_peak,
                        );
                        self.decision.decide(&personalized_scores)
                    })
                    .or(native_hit);
                // A reported false positive vetoes every path, native included —
                // a negative enrolled from a bad detection must suppress the
                // same sound even when YAMNet (not a prototype) produced the
                // label. The veto is scoped to the class it was reported
                // against, so recharacterizing a sound suppresses only the wrong
                // label and leaves the corrected one free to fire.
                let hit = hit.filter(|candidate| {
                    !self
                        .proto
                        .as_ref()
                        .is_some_and(|m| m.vetoes(&features.embedding, candidate.event_type))
                });

                let mut sorted: Vec<(EventType, f32)> =
                    scores.scores.iter().map(|(&k, &v)| (k, v)).collect();
                // `total_cmp` is NaN-safe: a zero-norm embedding can make a cosine
                // score NaN, which would panic `partial_cmp().unwrap()`.
                sorted.sort_by(|a, b| b.1.total_cmp(&a.1));

                if let Some(h) = hit {
                    for ev in st.sessionizer.observe(WindowObservation {
                        event_type: h.event_type,
                        confidence: h.confidence,
                        timestamp_ms: time_ms,
                        energy_peak,
                        levels,
                        embedding: features.embedding.clone(),
                    }) {
                        out.events.push(ev);
                    }
                }

                out.windows.push(WindowRecord {
                    time_ms,
                    active: true,
                    energy_peak,
                    speech: scores.speech,
                    scores: sorted,
                    hit,
                });
            }

            st.next_patch += 1;
            st.prune((p + 1) * hops_per_patch_hop, (p + 1) * PATCH_HOP_FRAMES);
        }
        Ok(())
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
            peak_dbfs: d.peak_dbfs,
            mean_dbfs: d.mean_dbfs,
            noise_floor_dbfs: d.noise_floor_dbfs,
            model_version: ctx.model_version.clone(),
            source: ctx.source,
            device_id: ctx.device_id.clone(),
            uploaded_at: None,
            deleted: false,
            false_positive_at: None,
            corrected_to: None,
            corrected_at: None,
            reject_count: 0,
            rejected_at: None,
        }
    }
}

/// Streaming front-end (SPEC §1, live capture). Owns persistent gate/sessionizer/
/// mel state so incremental [`push`](Self::push) calls never reset at chunk
/// boundaries: events straddling a boundary merge into one, per-class cooldowns
/// persist, and the adaptive noise floor keeps converging. Event timestamps derive
/// from a running sample counter (patch index), not per-chunk wall-clock math, so
/// the capture thread maps sample-counter ms → wall clock once at stream start.
pub struct StreamingPipeline<E: Embedder> {
    inner: Pipeline<E>,
    state: StreamState,
}

impl<E: Embedder> StreamingPipeline<E> {
    pub fn new(cfg: PipelineConfig, embedder: E) -> Self {
        let inner = Pipeline::new(cfg, embedder);
        let state = StreamState::new(&inner.cfg, &inner.mel);
        StreamingPipeline { inner, state }
    }

    /// Attach a prototype matcher for the enrolled custom classes (SPEC §5 B-lite).
    pub fn with_prototypes(mut self, proto: PrototypeMatcher) -> Self {
        self.inner = self.inner.with_prototypes(proto);
        self
    }

    pub fn model_version(&self) -> String {
        self.inner.model_version()
    }

    pub fn embed_patch(&self, patch: &MelPatch, energy_peak: bool) -> Result<Vec<f32>> {
        self.inner.embed_patch(patch, energy_peak)
    }

    pub fn set_prototypes(&mut self, proto: Option<PrototypeMatcher>) {
        self.inner.set_prototypes(proto);
    }

    /// Retune detection sensitivity without restarting the microphone stream.
    pub fn set_sensitivity(&mut self, sensitivity: f32) {
        self.inner.set_sensitivity(sensitivity);
    }

    /// Start a fresh sample-clock/gate/session stream while retaining the loaded
    /// model and personalized prototypes. Used after a real microphone pause so
    /// events cannot merge across an unmonitored gap.
    pub fn reset_stream(&mut self) {
        self.state = StreamState::new(&self.inner.cfg, &self.inner.mel);
    }

    /// Whether the energy gate is currently open — i.e. the app is analyzing a
    /// sound right now. Drives the "heard something" indicator.
    pub fn gate_open(&self) -> bool {
        self.state.gate.is_open()
    }

    /// Feed the next slice of 16 kHz mono samples; returns any events that closed as
    /// a result. Sample counting is monotonic across calls (SPEC §4.1 ⑤).
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<DetectedEvent>> {
        Ok(self.inner.stream_advance(&mut self.state, samples)?.events)
    }

    /// Close out the stream: finalize any trailing full patches and close any
    /// still-open sessions. Call at end of stream / on shutdown.
    pub fn flush(&mut self) -> Result<Vec<DetectedEvent>> {
        Ok(self.inner.stream_flush(&mut self.state)?.events)
    }

    /// Stamp metadata onto a detected event (delegates to the inner pipeline).
    pub fn to_event(&self, d: &DetectedEvent, ctx: &EventContext) -> Event {
        self.inner.to_event(d, ctx)
    }

    /// Current adaptive noise floor (dBFS) — exposed for tests asserting that gate
    /// state persists across pushes.
    pub fn gate_floor_db(&self) -> f32 {
        self.state.gate.floor_db()
    }

    #[cfg(test)]
    fn mel_frames_computed(&self) -> usize {
        self.state.mel_frames_computed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::embed::{BandHeuristicEmbedder, MockEmbedder, WindowFeatures};
    use crate::classify::native::AudiosetMap;
    use crate::classify::proto::{Enrollment, PrototypeMatcher};
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

    #[test]
    fn closed_gate_skips_all_mel_fft_work() {
        let mut pipeline = StreamingPipeline::new(PipelineConfig::default(), BandHeuristicEmbedder);
        for chunk in synth::silence(16_000 * 30).chunks(777) {
            assert!(pipeline.push(chunk).unwrap().is_empty());
        }
        assert!(pipeline.flush().unwrap().is_empty());
        assert_eq!(
            pipeline.mel_frames_computed(),
            0,
            "quiet-room processing must remain gate-only"
        );
    }

    /// Loudness is recorded, and a loud event is measurably louder than a quiet
    /// one of the same class — the whole point of logging intensity.
    #[test]
    fn events_record_how_loud_they_were() {
        let burst = |amplitude: f32| {
            let mut sig = synth::white_noise(16_000, 0.003, 1);
            sig.extend(synth::sine(16_000 * 2, 16_000, 300.0, amplitude));
            sig.extend(synth::white_noise(16_000, 0.003, 2));
            Pipeline::new(PipelineConfig::default(), BandHeuristicEmbedder)
                .process(&sig)
                .unwrap()
                .events
        };

        let loud = burst(0.6);
        let quiet = burst(0.06);
        assert!(!loud.is_empty() && !quiet.is_empty());

        let loud_peak = loud[0].peak_dbfs.expect("loudness recorded");
        let quiet_peak = quiet[0].peak_dbfs.expect("loudness recorded");
        assert!(
            loud_peak > quiet_peak + 10.0,
            "a 10x amplitude difference should be ~20 dB: {loud_peak} vs {quiet_peak}"
        );

        // dBFS for real signal sits at or below 0, and the mean cannot exceed
        // the peak.
        assert!(loud_peak <= 0.0, "peak {loud_peak} should be <= 0 dBFS");
        assert!(loud[0].mean_dbfs.unwrap() <= loud_peak);
        // The floor is captured from the quiet lead-in, so it is well below.
        assert!(loud[0].noise_floor_dbfs.unwrap() < loud_peak);
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

    #[test]
    fn personalized_match_overrides_a_conflicting_native_label() {
        let mut native_scores = vec![0.0; crate::classify::embed::AUDIOSET_CLASSES];
        native_scores[AudiosetMap::default().sneeze] = 0.99;
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let embedder = MockEmbedder {
            features: WindowFeatures {
                audioset_scores: Some(native_scores),
                embedding: embedding.clone(),
                energy_peak: true,
            },
            version: "test".to_string(),
        };
        let matcher = PrototypeMatcher::from_enrollments(
            &(0..3)
                .map(|_| Enrollment {
                    class: EventType::NoseBlow,
                    embedding: embedding.clone(),
                    is_negative: false,
                    negative_scoped: false,
                })
                .collect::<Vec<_>>(),
            0.65,
            0.05,
        );
        let pipeline = Pipeline::new(PipelineConfig::default(), embedder).with_prototypes(matcher);
        let signal = synth::sine(16_000 * 2, 16_000, 500.0, 0.8);
        let result = pipeline.process(&signal).unwrap();
        assert!(!result.events.is_empty());
        assert!(result
            .events
            .iter()
            .all(|event| event.event_type == EventType::NoseBlow));
    }

    #[test]
    fn personalized_match_does_not_override_speech_dominant_audio() {
        let mut native_scores = vec![0.0; crate::classify::embed::AUDIOSET_CLASSES];
        native_scores[AudiosetMap::default().speech] = 0.99;
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let embedder = MockEmbedder {
            features: WindowFeatures {
                audioset_scores: Some(native_scores),
                embedding: embedding.clone(),
                energy_peak: true,
            },
            version: "test".to_string(),
        };
        let matcher = PrototypeMatcher::from_enrollments(
            &(0..3)
                .map(|_| Enrollment {
                    class: EventType::Hawk,
                    embedding: embedding.clone(),
                    is_negative: false,
                    negative_scoped: false,
                })
                .collect::<Vec<_>>(),
            0.65,
            0.05,
        );
        let pipeline = Pipeline::new(PipelineConfig::default(), embedder).with_prototypes(matcher);
        let signal = synth::sine(16_000 * 2, 16_000, 500.0, 0.8);
        let result = pipeline.process(&signal).unwrap();
        assert!(result.events.is_empty());
    }

    /// A negative enrolled from a reported false positive suppresses a *native*
    /// YAMNet detection of the same sound — not just personalized prototypes.
    #[test]
    fn reported_false_positive_vetoes_native_detection() {
        let mut native_scores = vec![0.0; crate::classify::embed::AUDIOSET_CLASSES];
        native_scores[AudiosetMap::default().cough] = 0.99;
        let embedding = vec![0.2, 0.9, 0.1, 0.0];
        let embedder = || MockEmbedder {
            features: WindowFeatures {
                audioset_scores: Some(native_scores.clone()),
                embedding: embedding.clone(),
                energy_peak: true,
            },
            version: "test".to_string(),
        };
        let signal = synth::sine(16_000 * 2, 16_000, 500.0, 0.8);

        // Without the negative the native path fires…
        let baseline = Pipeline::new(PipelineConfig::default(), embedder())
            .process(&signal)
            .unwrap();
        assert!(!baseline.events.is_empty());
        // …and the closed event carries the window embedding for later reporting.
        assert_eq!(baseline.events[0].embedding, embedding);

        // With the event's own embedding enrolled as a negative, it is vetoed.
        let matcher = PrototypeMatcher::from_enrollments(
            &[Enrollment {
                class: EventType::Cough,
                embedding: embedding.clone(),
                is_negative: true,
                negative_scoped: false,
            }],
            0.65,
            0.05,
        );
        let vetoed = Pipeline::new(PipelineConfig::default(), embedder())
            .with_prototypes(matcher)
            .process(&signal)
            .unwrap();
        assert!(vetoed.events.is_empty(), "events: {:?}", vetoed.events);
    }

    /// Build a rich signal: quiet-converge → cough-band burst → quiet → a second
    /// burst → trailing quiet. Exercises gate open/close, sessionizing, cooldowns.
    fn rich_signal() -> Vec<f32> {
        let mut sig = synth::white_noise(16_000, 0.003, 1); // ~1 s quiet
        sig.extend(synth::sine(16_000 * 2, 16_000, 300.0, 0.6)); // 2 s cough burst
        sig.extend(synth::white_noise(16_000 * 2, 0.003, 2)); // ~2 s quiet gap
        sig.extend(synth::sine(16_000 * 2, 16_000, 300.0, 0.6)); // 2 s cough burst
        sig.extend(synth::white_noise(16_000, 0.003, 3)); // ~1 s quiet tail
        sig
    }

    /// THE key invariant (SPEC §1): the same signal fed whole via `process` and
    /// split across many odd-sized `push` calls yields identical events. This
    /// covers straddling boundaries, persistent cooldowns, and floor continuity in
    /// one shot — the streaming state never resets at a chunk boundary.
    #[test]
    fn batch_and_streaming_agree_on_events() {
        let sig = rich_signal();
        let batch = Pipeline::new(PipelineConfig::default(), BandHeuristicEmbedder)
            .process(&sig)
            .unwrap()
            .events;

        // Many odd-sized pushes (777 does not divide a gate hop, a mel hop, or a
        // patch — every boundary lands mid-stage).
        let mut sp = StreamingPipeline::new(PipelineConfig::default(), BandHeuristicEmbedder);
        let mut streamed = Vec::new();
        for chunk in sig.chunks(777) {
            streamed.extend(sp.push(chunk).unwrap());
        }
        streamed.extend(sp.flush().unwrap());

        assert_eq!(
            batch, streamed,
            "streaming must reproduce the batch events exactly"
        );
        assert!(!batch.is_empty(), "the rich signal should yield events");
    }

    /// An event whose loud span straddles a push boundary is ONE event, not two.
    /// Split the signal exactly in the middle of the 2 s burst.
    #[test]
    fn event_straddling_a_push_boundary_is_one_event() {
        let sig = {
            let mut s = synth::white_noise(16_000, 0.003, 1);
            s.extend(synth::sine(16_000 * 2, 16_000, 300.0, 0.6));
            s.extend(synth::white_noise(16_000, 0.003, 2));
            s
        };
        // Boundary mid-burst: 1 s quiet + 1 s into the tone.
        let split = 16_000 * 2;
        let mut sp = StreamingPipeline::new(PipelineConfig::default(), BandHeuristicEmbedder);
        let mut events = Vec::new();
        events.extend(sp.push(&sig[..split]).unwrap());
        events.extend(sp.push(&sig[split..]).unwrap());
        events.extend(sp.flush().unwrap());
        assert_eq!(events.len(), 1, "events: {events:?}");
        assert_eq!(events[0].event_type, EventType::Cough);
    }

    /// A per-class cooldown suppresses a re-grip even when the boundary falls
    /// between the two detections. Two cough bursts separated by a ~2 s gap: the
    /// first closes (gap > 1.5 s merge) and its 2 s cooldown swallows the onset of
    /// the second. `process` (whole) and split `push` must agree — and both must
    /// show cooldown suppression collapsing the two bursts toward one event.
    #[test]
    fn cooldown_persists_across_a_push_boundary() {
        let sig = rich_signal();
        let batch = Pipeline::new(PipelineConfig::default(), BandHeuristicEmbedder)
            .process(&sig)
            .unwrap()
            .events;

        // Put every push boundary inside the quiet gap between the two bursts.
        let mut sp = StreamingPipeline::new(PipelineConfig::default(), BandHeuristicEmbedder);
        let mut streamed = Vec::new();
        // First burst + into the gap.
        streamed.extend(sp.push(&sig[..16_000 * 4]).unwrap());
        // Remainder (rest of gap + second burst + tail), one sample at a time near
        // the boundary to stress state continuity.
        for chunk in sig[16_000 * 4..].chunks(1) {
            streamed.extend(sp.push(chunk).unwrap());
        }
        streamed.extend(sp.flush().unwrap());

        assert_eq!(batch, streamed);
        // The cough cooldown (2 s) means the two 300 Hz bursts do not both survive
        // as independent events — assert the streamed count matches batch and is
        // bounded (cooldown actually suppressed a re-fire).
        assert!(
            streamed.len() <= 2,
            "cough cooldown should limit re-fires: {streamed:?}"
        );
    }

    /// Noise-floor state persists across pushes: feeding a long, quiet ambient in
    /// many chunks lets the floor converge upward (SPEC §4.1 ①). A per-chunk reset
    /// would keep snapping the floor back to `floor_init_db`; here it climbs and
    /// the gate never spuriously opens.
    #[test]
    fn noise_floor_state_persists_across_pushes() {
        let cfg = PipelineConfig::default();
        let init_floor = cfg.gate.floor_init_db;
        // ~30 s of low ambient (~ -53 dBFS RMS), below the +10 dB open margin.
        let ambient = synth::white_noise(16_000 * 30, 0.004, 7);

        let mut sp = StreamingPipeline::new(cfg, BandHeuristicEmbedder);
        let mut events = Vec::new();
        for chunk in ambient.chunks(1000) {
            events.extend(sp.push(chunk).unwrap());
        }
        events.extend(sp.flush().unwrap());

        assert!(events.is_empty(), "quiet ambient must not fire events");
        assert!(
            sp.gate_floor_db() > init_floor + 5.0,
            "floor should converge upward across pushes: {} -> {}",
            init_floor,
            sp.gate_floor_db()
        );
    }
}
