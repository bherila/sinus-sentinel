//! Stage ① — energy gate (SPEC §4.1). Runs always at ≈0 CPU (no FFT): per 50 ms
//! hop it computes RMS energy in dB and maintains an adaptive noise floor (EMA
//! that rises slowly / falls fast). The gate opens at `floor + open_margin` and,
//! once open, stays open until energy sits below the (hysteresis-lowered) close
//! threshold for a full tail. A 1 s pre-roll is pulled from the ring buffer on
//! the opening edge so the *onset* of the sound is analyzed, not just its tail.

use crate::types::SAMPLE_RATE;

/// Gate tuning. Defaults follow SPEC §4.1.
#[derive(Debug, Clone)]
pub struct GateConfig {
    pub sample_rate: u32,
    /// Analysis hop length in milliseconds (SPEC: 50 ms).
    pub hop_ms: u32,
    /// Gate opens at `floor + open_margin_db` (SPEC: ~10 dB).
    pub open_margin_db: f32,
    /// Hysteresis: the close threshold is `open_margin_db - hysteresis_db` above
    /// the floor, so a briefly-quieter moment does not immediately close.
    pub hysteresis_db: f32,
    /// Gate stays open until energy is below the close threshold for this long
    /// (SPEC: 1 s tail).
    pub tail_ms: u32,
    /// Pre-roll pulled from the ring buffer on the opening edge (SPEC: 1 s).
    pub preroll_ms: u32,
    /// Noise-floor rise time constant (SPEC: slow, ~3 s).
    pub rise_tau_s: f32,
    /// Noise-floor fall time constant (fast).
    pub fall_tau_s: f32,
    /// Initial floor estimate in dBFS.
    pub floor_init_db: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        GateConfig {
            sample_rate: SAMPLE_RATE,
            hop_ms: 50,
            open_margin_db: 10.0,
            hysteresis_db: 4.0,
            tail_ms: 1000,
            preroll_ms: 1000,
            rise_tau_s: 3.0,
            fall_tau_s: 0.2,
            floor_init_db: -60.0,
        }
    }
}

impl GateConfig {
    /// Samples per analysis hop.
    pub fn hop_samples(&self) -> usize {
        (self.sample_rate as u64 * self.hop_ms as u64 / 1000) as usize
    }

    /// Number of hops in the pre-roll window.
    pub fn preroll_hops(&self) -> usize {
        (self.preroll_ms / self.hop_ms) as usize
    }

    fn tail_hops(&self) -> u32 {
        self.tail_ms / self.hop_ms
    }
}

/// Edge emitted by [`Gate::process_hop`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateEdge {
    /// The gate transitioned closed → open on this hop.
    Opened,
    /// The gate transitioned open → closed on this hop.
    Closed,
}

/// Per-hop report from the gate.
#[derive(Debug, Clone, Copy)]
pub struct HopReport {
    /// RMS level of this hop, dBFS.
    pub rms_db: f32,
    /// Current adaptive noise floor, dBFS.
    pub floor_db: f32,
    /// Whether the gate is open after processing this hop.
    pub open: bool,
    /// Whether this hop is an energy *peak* relative to the local floor — used
    /// by the weak-class coincidence rule and burst counting (SPEC §4.1 ④/⑤).
    pub energy_peak: bool,
    /// Transition edge, if any.
    pub edge: Option<GateEdge>,
}

/// Adaptive-noise-floor energy gate.
#[derive(Debug, Clone)]
pub struct Gate {
    cfg: GateConfig,
    floor_db: f32,
    open: bool,
    hops_below: u32,
    alpha_rise: f32,
    alpha_fall: f32,
}

impl Gate {
    pub fn new(cfg: GateConfig) -> Self {
        let dt = cfg.hop_ms as f32 / 1000.0;
        let alpha_rise = 1.0 - (-dt / cfg.rise_tau_s).exp();
        let alpha_fall = 1.0 - (-dt / cfg.fall_tau_s).exp();
        let floor_db = cfg.floor_init_db;
        Gate {
            cfg,
            floor_db,
            open: false,
            hops_below: 0,
            alpha_rise,
            alpha_fall,
        }
    }

    pub fn with_defaults() -> Self {
        Gate::new(GateConfig::default())
    }

    pub fn config(&self) -> &GateConfig {
        &self.cfg
    }

    pub fn floor_db(&self) -> f32 {
        self.floor_db
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Compute the RMS level of a hop in dBFS.
    fn rms_db(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return -120.0;
        }
        let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
        let rms = (sum_sq / samples.len() as f64).sqrt();
        20.0 * (rms + 1e-9).log10() as f32
    }

    /// Process one 50 ms hop of samples and update gate state.
    pub fn process_hop(&mut self, samples: &[f32]) -> HopReport {
        let rms_db = Self::rms_db(samples);

        let open_threshold = self.floor_db + self.cfg.open_margin_db;
        let close_threshold = self.floor_db + self.cfg.open_margin_db - self.cfg.hysteresis_db;
        // An energy peak = well above the floor, used for weak-class coincidence
        // and burst counting. Uses half the open margin so it flags the loud core
        // of a segment even mid-session.
        let energy_peak = rms_db > self.floor_db + self.cfg.open_margin_db * 0.5;

        let mut edge = None;
        if self.open {
            if rms_db < close_threshold {
                self.hops_below += 1;
                if self.hops_below >= self.cfg.tail_hops() {
                    self.open = false;
                    self.hops_below = 0;
                    edge = Some(GateEdge::Closed);
                }
            } else {
                self.hops_below = 0;
            }
        } else if rms_db > open_threshold {
            self.open = true;
            self.hops_below = 0;
            edge = Some(GateEdge::Opened);
        }

        // Update the adaptive floor: slow rise, fast fall — but ONLY while the gate
        // is closed. Freezing the floor while the gate is open is standard
        // noise-gate practice: an open gate is measuring signal+noise, not the room
        // floor, so letting the ~3 s rise track a loud event would erode the
        // open/close margin for the duration of a long event (SPEC §4.1 ①). The
        // slow-rise "a persistent fan raises the floor" behaviour still holds while
        // the gate is closed, where the ambient level is what's being measured.
        if !self.open {
            let alpha = if rms_db > self.floor_db {
                self.alpha_rise
            } else {
                self.alpha_fall
            };
            self.floor_db += alpha * (rms_db - self.floor_db);
        }

        HopReport {
            rms_db,
            floor_db: self.floor_db,
            open: self.open,
            energy_peak: energy_peak && self.open,
            edge,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;

    /// Feed silence-ish noise, then a loud burst, then quiet again, and assert
    /// the gate opens on the burst and closes ~1 s (tail) after it ends.
    #[test]
    fn opens_on_burst_and_closes_after_tail() {
        let cfg = GateConfig::default();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();

        // 40 hops of low-level noise (~ -50 dBFS), then 20 hops loud (~ -6 dBFS),
        // then 60 hops quiet again.
        let quiet = synth::white_noise_hops(40, hop, 0.003, 1);
        let loud = synth::sine_hops(20, hop, cfg.sample_rate, 900.0, 0.5);
        let quiet2 = synth::white_noise_hops(60, hop, 0.003, 7);

        let mut opened_at = None;
        let mut closed_at = None;
        let mut idx = 0usize;
        for block in [quiet, loud, quiet2] {
            for chunk in block.chunks(hop) {
                let r = gate.process_hop(chunk);
                match r.edge {
                    Some(GateEdge::Opened) => opened_at = Some(idx),
                    Some(GateEdge::Closed) => closed_at = Some(idx),
                    None => {}
                }
                idx += 1;
            }
        }

        let opened_at = opened_at.expect("gate should open on the burst");
        let closed_at = closed_at.expect("gate should close after the burst");
        // Opens shortly after the burst starts at hop 40.
        assert!(
            (40..45).contains(&opened_at),
            "opened_at = {opened_at}, expected ~40"
        );
        // Burst ends at hop 60; closes ~1 s (20 hops) later.
        assert!(
            (78..86).contains(&closed_at),
            "closed_at = {closed_at}, expected ~80"
        );
    }

    #[test]
    fn stays_closed_on_pure_silence() {
        let cfg = GateConfig::default();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();
        let silence = vec![0.0f32; hop];
        for _ in 0..200 {
            let r = gate.process_hop(&silence);
            assert!(!r.open, "gate must never open on silence");
            assert!(r.edge.is_none());
        }
    }

    #[test]
    fn floor_rises_toward_ambient_while_gate_closed() {
        // A steady low hiss that stays *below* the open threshold keeps the gate
        // closed; the adaptive floor should slow-rise toward it (SPEC §4.1 — "a
        // persistent fan raises the floor"). Amplitude 0.004 (~ -53 dBFS RMS) sits
        // above the -60 dBFS init floor but below floor + 10 dB (-50), so the gate
        // never opens and the floor tracks the ambient upward.
        let cfg = GateConfig::default();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();
        let start_floor = gate.floor_db();
        let noise = synth::white_noise_hops(600, hop, 0.004, 3);
        for chunk in noise.chunks(hop) {
            let r = gate.process_hop(chunk);
            assert!(!r.open, "quiet ambient must not open the gate");
        }
        assert!(
            gate.floor_db() > start_floor + 5.0,
            "floor should rise toward ambient while closed: {} -> {}",
            start_floor,
            gate.floor_db()
        );
    }

    #[test]
    fn floor_frozen_while_gate_open() {
        // Standard noise-gate behaviour: once the gate opens on a loud event, the
        // noise floor is frozen so the long event can't erode the margin (SPEC
        // §4.1, review finding #3). Verify the floor barely moves across a
        // multi-second open span even though the signal is far above it.
        let cfg = GateConfig::default();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();

        // Open the gate with a loud tone.
        let loud = synth::sine_hops(1, hop, cfg.sample_rate, 900.0, 0.5);
        let r = gate.process_hop(&loud);
        assert!(r.open, "loud tone should open the gate");
        let floor_at_open = gate.floor_db();

        // Hold it open for ~2 s of loud signal; the frozen floor must not drift.
        let hold = synth::sine_hops(40, hop, cfg.sample_rate, 900.0, 0.5);
        for chunk in hold.chunks(hop) {
            let r = gate.process_hop(chunk);
            assert!(r.open, "gate should stay open under sustained signal");
        }
        assert!(
            (gate.floor_db() - floor_at_open).abs() < 0.01,
            "floor must stay frozen while open: {floor_at_open} -> {}",
            gate.floor_db()
        );
    }

    #[test]
    fn hop_and_preroll_sizes() {
        let cfg = GateConfig::default();
        assert_eq!(cfg.hop_samples(), 800); // 16 kHz * 50 ms
        assert_eq!(cfg.preroll_hops(), 20); // 1 s / 50 ms
    }
}
