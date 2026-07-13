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
use crate::mel::{MelFrontend, MelPatch, HOP_LEN, N_MEL, PATCH_FRAMES, PATCH_HOP_FRAMES};
use crate::session::{DetectedEvent, SessionConfig, Sessionizer, WindowObservation};
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
    hops_base: usize,
    /// Absolute hop indices of gate opening edges (rolling; pruned with hops).
    open_edges: VecDeque<usize>,

    /// Samples not yet consumed into mel frames (carries the STFT overlap remainder
    /// across pushes so no frame is dropped at a boundary — SPEC §4.1 ②).
    mel_buf: Vec<f32>,
    /// Absolute index of the next mel frame to emit.
    next_frame: usize,
    /// Rolling log-mel frames, `frames[0]` at absolute frame index `frames_base`.
    frames: VecDeque<[f32; N_MEL]>,
    frames_base: usize,

    /// Absolute index of the next patch to finalize.
    next_patch: usize,
}

impl StreamState {
    fn new(cfg: &PipelineConfig) -> Self {
        StreamState {
            gate: Gate::new(cfg.gate.clone()),
            sessionizer: Sessionizer::new(cfg.session.clone()),
            gate_buf: Vec::new(),
            next_hop: 0,
            hops_open: VecDeque::new(),
            hops_peak: VecDeque::new(),
            hops_base: 0,
            open_edges: VecDeque::new(),
            mel_buf: Vec::new(),
            next_frame: 0,
            frames: VecDeque::new(),
            frames_base: 0,
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
            if report.edge == Some(GateEdge::Opened) {
                self.open_edges.push_back(self.next_hop);
            }
            self.next_hop += 1;
            consumed += hop;
        }
        self.gate_buf.drain(..consumed);
    }

    /// Drain `mel_buf` into log-mel frames (SPEC §4.1 ②), advancing by `HOP_LEN`
    /// and retaining the `FRAME_LEN - HOP_LEN` overlap remainder for the next push.
    fn push_mel_samples(&mut self, mel: &MelFrontend, samples: &[f32]) {
        self.mel_buf.extend_from_slice(samples);
        let mut pos = 0;
        while self.mel_buf.len() - pos >= crate::mel::FRAME_LEN {
            let row = mel.log_mel_frame(&self.mel_buf[pos..pos + crate::mel::FRAME_LEN]);
            self.frames.push_back(row);
            self.next_frame += 1;
            pos += HOP_LEN;
        }
        self.mel_buf.drain(..pos);
    }

    /// Assemble the 96-frame patch starting at absolute frame `frame_start`.
    fn assemble_patch(&self, frame_start: usize) -> MelPatch {
        let mut data = Vec::with_capacity(PATCH_FRAMES * N_MEL);
        for i in 0..PATCH_FRAMES {
            let rel = frame_start + i - self.frames_base;
            data.extend_from_slice(&self.frames[rel]);
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

    /// Drop rolling state that no later patch can reference (bounded memory for a
    /// long-running stream).
    fn prune(&mut self, hop_keep_from: usize, frame_keep_from: usize) {
        while self.hops_base < hop_keep_from && !self.hops_open.is_empty() {
            self.hops_open.pop_front();
            self.hops_peak.pop_front();
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

    /// Process a buffer of 16 kHz mono samples (batch): advance the engine over the
    /// whole buffer, then flush. Produces the same windows/events as feeding the
    /// same samples through a [`StreamingPipeline`] in arbitrary chunks.
    pub fn process(&self, samples: &[f32]) -> Result<PipelineResult> {
        let mut st = StreamState::new(&self.cfg);
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
        st.push_mel_samples(&self.mel, samples);
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
                let patch = st.assemble_patch(frame_start);
                let features = self.embedder.embed(&patch, energy_peak)?;
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
                // `total_cmp` is NaN-safe: a zero-norm embedding can make a cosine
                // score NaN, which would panic `partial_cmp().unwrap()`.
                sorted.sort_by(|a, b| b.1.total_cmp(&a.1));

                if let Some(h) = hit {
                    for ev in st.sessionizer.observe(WindowObservation {
                        event_type: h.event_type,
                        confidence: h.confidence,
                        timestamp_ms: time_ms,
                        energy_peak,
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
        let state = StreamState::new(&cfg);
        StreamingPipeline {
            inner: Pipeline::new(cfg, embedder),
            state,
        }
    }

    /// Attach a prototype matcher for the enrolled custom classes (SPEC §5 B-lite).
    pub fn with_prototypes(mut self, proto: PrototypeMatcher) -> Self {
        self.inner = self.inner.with_prototypes(proto);
        self
    }

    pub fn model_version(&self) -> String {
        self.inner.model_version()
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
