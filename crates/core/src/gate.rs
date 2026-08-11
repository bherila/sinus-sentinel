//! Stage ① — energy gate (SPEC §4.1). Runs always at ≈0 CPU (no FFT): per 50 ms
//! hop it computes RMS energy in dB and maintains an adaptive noise floor.
//! Startup calibration is an explicit live-capture policy; finite clips default
//! to immediate detection. A lower-quantile estimator tracks sustained ambient
//! sound without treating frequent foreground bursts as the room. The gate opens
//! at `floor + open_margin` and, once open, stays
//! open until energy sits below the (hysteresis-lowered) close threshold for a
//! full tail. A 1 s pre-roll is pulled from the ring buffer on the opening edge
//! so the *onset* of the sound is analyzed, not just its tail.

use std::collections::VecDeque;

use crate::types::SAMPLE_RATE;

/// Whether a new gate is immediately usable or first observes a live room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateStartupMode {
    /// Detect from the first hop. Used by finite clips and user-triggered mobile
    /// sessions, where consuming an ambient lead-in would silently lose events.
    Immediate,
    /// Hold the gate closed while collecting a candidate room estimate.
    Calibrate { duration_ms: u32 },
}

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
    /// Levels below this are treated as invalid device/digital silence rather
    /// than evidence about the physical room.
    pub min_valid_dbfs: f32,
    /// A live calibration cannot lower the authoritative floor below this bound.
    pub min_calibrated_floor_dbfs: f32,
    /// Robust quantile over the recent level horizon (0.25 means foreground
    /// must occupy roughly 75% before it can replace ordinary quiet gaps).
    pub ambient_quantile: f32,
    /// Startup behavior. The shared default is immediate; passive live capture
    /// opts into calibration explicitly.
    pub startup_mode: GateStartupMode,
    /// Rolling RMS window used to distinguish sustained ambient sound from a
    /// short foreground event.
    pub ambient_window_ms: u32,
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
            min_valid_dbfs: -115.0,
            min_calibrated_floor_dbfs: -95.0,
            ambient_quantile: 0.25,
            startup_mode: GateStartupMode::Immediate,
            ambient_window_ms: 10_000,
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

    fn calibration_hops(&self) -> u32 {
        match self.startup_mode {
            GateStartupMode::Immediate => 0,
            GateStartupMode::Calibrate { duration_ms } => duration_ms.div_ceil(self.hop_ms),
        }
    }

    fn ambient_window_hops(&self) -> usize {
        self.ambient_window_ms.div_ceil(self.hop_ms).max(1) as usize
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

/// Current gate estimator state, kept separate so a miss can be attributed to
/// the short floor, robust floor, or their effective maximum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateDiagnostics {
    pub calibrating: bool,
    pub short_floor_db: f32,
    pub ambient_floor_db: f32,
    pub effective_floor_db: f32,
    pub current_rms_db: f32,
    pub open: bool,
}

/// Adaptive-noise-floor energy gate.
#[derive(Debug, Clone)]
pub struct Gate {
    cfg: GateConfig,
    /// Fast closed-gate EMA.
    floor_db: f32,
    /// Robust baseline across recent audio, including open-gate spans.
    ambient_floor_db: f32,
    /// One entry per hop so invalid/silent spans still age old observations out.
    ambient_levels: VecDeque<Option<f32>>,
    calibration_levels: Vec<f32>,
    calibration_hops_remaining: u32,
    hops_until_ambient_refresh: u32,
    quieter_closed_hops: u32,
    /// Once a formerly loud room is falling, stale loud observations must not
    /// promote it again until recent robust evidence has replaced that source.
    ambient_recovery_active: bool,
    open: bool,
    hops_below: u32,
    alpha_rise: f32,
    alpha_fall: f32,
    current_rms_db: f32,
}

impl Gate {
    pub fn new(cfg: GateConfig) -> Self {
        let dt = cfg.hop_ms as f32 / 1000.0;
        let alpha_rise = 1.0 - (-dt / cfg.rise_tau_s).exp();
        let alpha_fall = 1.0 - (-dt / cfg.fall_tau_s).exp();
        let floor_db = cfg.floor_init_db;
        let calibration_hops_remaining = cfg.calibration_hops();
        let ambient_capacity = cfg.ambient_window_hops();
        Gate {
            cfg,
            floor_db,
            ambient_floor_db: floor_db,
            ambient_levels: VecDeque::with_capacity(ambient_capacity),
            calibration_levels: Vec::with_capacity(calibration_hops_remaining as usize),
            calibration_hops_remaining,
            hops_until_ambient_refresh: 0,
            quieter_closed_hops: 0,
            ambient_recovery_active: false,
            open: false,
            hops_below: 0,
            alpha_rise,
            alpha_fall,
            current_rms_db: floor_db,
        }
    }

    pub fn with_defaults() -> Self {
        Gate::new(GateConfig::default())
    }

    pub fn config(&self) -> &GateConfig {
        &self.cfg
    }

    pub fn floor_db(&self) -> f32 {
        self.floor_db.max(self.ambient_floor_db)
    }

    pub fn diagnostics(&self) -> GateDiagnostics {
        GateDiagnostics {
            calibrating: self.calibration_hops_remaining > 0,
            short_floor_db: self.floor_db,
            ambient_floor_db: self.ambient_floor_db,
            effective_floor_db: self.floor_db(),
            current_rms_db: self.current_rms_db,
            open: self.open,
        }
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Reset only state that could join events across an unmonitored gap. Room
    /// observations and an in-progress first calibration deliberately survive.
    pub fn reset_temporal(&mut self) {
        self.open = false;
        self.hops_below = 0;
        self.quieter_closed_hops = 0;
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

    fn valid_level(&self, rms_db: f32) -> bool {
        // Digital silence is about -180 dBFS with the RMS epsilon above. It is
        // not evidence about microphone gain or the physical room.
        rms_db.is_finite() && rms_db >= self.cfg.min_valid_dbfs
    }

    fn quantile(mut levels: Vec<f32>, fraction: f32) -> Option<f32> {
        if levels.is_empty() {
            return None;
        }
        let index = ((levels.len() - 1) as f32 * fraction.clamp(0.0, 1.0)).round() as usize;
        let (_, value, _) = levels.select_nth_unstable_by(index, f32::total_cmp);
        Some(*value)
    }

    fn ambient_quantile(&self) -> Option<f32> {
        Self::quantile(
            self.ambient_levels.iter().flatten().copied().collect(),
            self.cfg.ambient_quantile,
        )
    }

    fn stable_recent_level(&self) -> Option<f32> {
        let count = 3000u32.div_ceil(self.cfg.hop_ms).max(1) as usize;
        if self.ambient_levels.len() < count {
            return None;
        }
        let recent: Vec<f32> = self
            .ambient_levels
            .iter()
            .rev()
            .take(count)
            .copied()
            .collect::<Option<Vec<_>>>()?;
        let q25 = Self::quantile(recent.clone(), 0.25)?;
        let q75 = Self::quantile(recent, 0.75)?;
        (q75 - q25 <= 2.0).then_some(q25)
    }

    fn finish_calibration(&mut self) {
        let expected = self.cfg.calibration_hops() as usize;
        // Calibration is a fixed wall-clock window. Missing hops are invalid
        // evidence, not permission to learn only the foreground that followed
        // a muted/zero-filled device startup.
        if self.calibration_levels.len() != expected {
            return;
        }
        let q25 = Self::quantile(self.calibration_levels.clone(), 0.25).unwrap();
        let q75 = Self::quantile(self.calibration_levels.clone(), 0.75).unwrap();
        // A stable room is safe to adopt. Handling noise, speech and coughs are
        // variable; retain the prior/default estimate and keep adapting instead.
        if q75 - q25 <= 6.0 {
            let estimate = q25.max(self.cfg.min_calibrated_floor_dbfs);
            self.floor_db = estimate;
            self.ambient_floor_db = estimate;
        }
    }

    /// Record every hop, including open-gate audio. The 25th percentile means a
    /// foreground level must occupy roughly 75% of the time horizon before it
    /// can replace ordinary quiet gaps. A stable source can be promoted sooner.
    fn observe_ambient(&mut self, rms_db: f32) {
        let capacity = self.cfg.ambient_window_hops();
        if self.ambient_levels.len() == capacity {
            self.ambient_levels.pop_front();
        }
        self.ambient_levels
            .push_back(self.valid_level(rms_db).then_some(rms_db));

        let minimum = (capacity / 2).max(1);
        let valid_count = self.ambient_levels.iter().flatten().count();

        if self.ambient_recovery_active {
            // Keep stale upward candidates disabled until either the full robust
            // horizon or a consecutive stable lower window says the old source
            // has actually been replaced. Always return on the exit hop so the
            // same observation cannot both end recovery and promote upward.
            let robust_replaced = valid_count >= minimum
                && self
                    .ambient_quantile()
                    .is_some_and(|level| level <= self.ambient_floor_db);
            let stable_lower = self
                .stable_recent_level()
                .is_some_and(|level| level <= self.ambient_floor_db);
            if robust_replaced || stable_lower {
                self.ambient_recovery_active = false;
                self.quieter_closed_hops = 0;
            }
            return;
        }

        // Refresh at 2 Hz; recalculating on every 50 ms hop adds no useful
        // responsiveness. Closed-gate evidence that the room became quieter is
        // handled separately below and must not be overwritten by stale history.
        if valid_count < minimum {
            return;
        }
        if self.hops_until_ambient_refresh > 0 {
            self.hops_until_ambient_refresh -= 1;
            return;
        }
        if let Some(stable) = self.stable_recent_level() {
            if stable > self.ambient_floor_db + 3.0 {
                self.ambient_floor_db = stable;
            }
        } else if let Some(candidate) = self.ambient_quantile() {
            if candidate > self.ambient_floor_db {
                self.ambient_floor_db = candidate;
            }
        }
        self.hops_until_ambient_refresh = 500u32.div_ceil(self.cfg.hop_ms).max(1) - 1;
    }

    /// Process one 50 ms hop of samples and update gate state.
    pub fn process_hop(&mut self, samples: &[f32]) -> HopReport {
        let rms_db = Self::rms_db(samples);
        self.current_rms_db = rms_db;

        let calibrating = self.calibration_hops_remaining > 0;
        if calibrating {
            if self.valid_level(rms_db) {
                self.calibration_levels.push(rms_db);
            }
            self.observe_ambient(rms_db);
            self.calibration_hops_remaining -= 1;
            if self.calibration_hops_remaining == 0 {
                self.finish_calibration();
            }
            return HopReport {
                rms_db,
                floor_db: self.floor_db(),
                open: false,
                energy_peak: false,
                edge: None,
            };
        }

        self.observe_ambient(rms_db);

        let effective_floor = self.floor_db();
        let open_threshold = effective_floor + self.cfg.open_margin_db;
        let close_threshold = effective_floor + self.cfg.open_margin_db - self.cfg.hysteresis_db;
        // An energy peak = well above the floor, used for weak-class coincidence
        // and burst counting. Uses half the open margin so it flags the loud core
        // of a segment even mid-session.
        let energy_peak = rms_db > effective_floor + self.cfg.open_margin_db * 0.5;

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

        // Update the short-term EMA only while closed: a brief open-gate event is
        // signal+noise, not the room floor. The separate rolling median observes
        // every hop and only moves when a level dominates the longer window, so a
        // truly sustained fan can still become background without one cough
        // eroding the margin.
        if !self.open && self.valid_level(rms_db) {
            let alpha = if rms_db > self.floor_db {
                self.alpha_rise
            } else {
                self.alpha_fall
            };
            self.floor_db += alpha * (rms_db - self.floor_db);

            // The short EMA can lag low immediately after a fan is promoted.
            // Require the current closed-gate audio to be lower too, otherwise
            // the still-running fan would trigger its own downward recovery.
            if rms_db < self.ambient_floor_db - 3.0 && self.floor_db < self.ambient_floor_db - 3.0 {
                self.quieter_closed_hops += 1;
                let recovery_hops = 500u32.div_ceil(self.cfg.hop_ms).max(1);
                if self.quieter_closed_hops >= recovery_hops {
                    self.ambient_recovery_active = true;
                }
            } else if !self.ambient_recovery_active {
                self.quieter_closed_hops = 0;
            }
            if self.ambient_recovery_active && self.floor_db < self.ambient_floor_db {
                self.ambient_floor_db += self.alpha_fall * (self.floor_db - self.ambient_floor_db);
            }
        } else if self.open {
            self.quieter_closed_hops = 0;
        }

        HopReport {
            rms_db,
            floor_db: self.floor_db(),
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

    fn live_config() -> GateConfig {
        GateConfig {
            startup_mode: GateStartupMode::Calibrate { duration_ms: 1000 },
            ..GateConfig::default()
        }
    }

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
    fn a_short_open_gate_does_not_redefine_the_ambient_floor() {
        // Fill the baseline with quiet room noise, then open for two seconds. The
        // foreground sound occupies less than half the ten-second window, so it
        // must not become the new ambient level.
        let cfg = GateConfig::default();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();

        let quiet = synth::white_noise_hops(200, hop, 0.003, 31);
        for chunk in quiet.chunks(hop) {
            assert!(!gate.process_hop(chunk).open);
        }
        let floor_at_open = gate.floor_db();

        let hold = synth::sine_hops(40, hop, cfg.sample_rate, 900.0, 0.5);
        for chunk in hold.chunks(hop) {
            let r = gate.process_hop(chunk);
            assert!(r.open, "gate should stay open under sustained signal");
        }
        assert!(
            (gate.floor_db() - floor_at_open).abs() < 0.5,
            "a short foreground sound must not redefine ambient: {floor_at_open} -> {}",
            gate.floor_db()
        );
    }

    #[test]
    fn startup_calibration_ignores_an_already_running_background() {
        let cfg = live_config();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();

        // About -40 dBFS RMS: loud enough to open immediately against the old
        // fixed -60 dBFS initialization, but now learned as the room baseline.
        let background = synth::sine_hops(40, hop, cfg.sample_rate, 120.0, 0.014);
        for chunk in background.chunks(hop) {
            let report = gate.process_hop(chunk);
            assert!(
                !report.open,
                "calibrated background must stay below the gate"
            );
        }
        assert!(gate.floor_db() > -43.0, "floor = {}", gate.floor_db());

        let foreground = synth::sine_hops(1, hop, cfg.sample_rate, 900.0, 0.2);
        let report = gate.process_hop(&foreground);
        assert!(
            report.open,
            "a real foreground burst must still open the gate"
        );
    }

    #[test]
    fn sustained_new_background_becomes_the_baseline() {
        let cfg = GateConfig::default();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();

        let quiet = synth::white_noise_hops(200, hop, 0.003, 41);
        for chunk in quiet.chunks(hop) {
            gate.process_hop(chunk);
        }

        // A fan starting after calibration initially opens the gate, but once it
        // occupies most of the rolling window it becomes ambient and the gate
        // closes instead of classifying it forever.
        let fan = synth::sine_hops(240, hop, cfg.sample_rate, 120.0, 0.014);
        let mut opened = false;
        for chunk in fan.chunks(hop) {
            let report = gate.process_hop(chunk);
            opened |= report.edge == Some(GateEdge::Opened);
        }
        assert!(
            opened,
            "the abrupt background onset should initially be heard"
        );
        assert!(
            !gate.is_open(),
            "sustained background should be gated back out"
        );
        assert!(gate.floor_db() > -43.0, "floor = {}", gate.floor_db());
    }

    #[test]
    fn hop_and_preroll_sizes() {
        let cfg = GateConfig::default();
        assert_eq!(cfg.hop_samples(), 800); // 16 kHz * 50 ms
        assert_eq!(cfg.preroll_hops(), 20); // 1 s / 50 ms
        assert_eq!(cfg.calibration_hops(), 0); // finite clips detect immediately
        assert_eq!(live_config().calibration_hops(), 20); // 1 s / 50 ms
        assert_eq!(cfg.ambient_window_hops(), 200); // 10 s / 50 ms
    }

    #[test]
    fn digital_silence_does_not_poison_live_calibration() {
        let cfg = live_config();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();

        for _ in 0..cfg.calibration_hops() {
            assert!(!gate.process_hop(&vec![0.0; hop]).open);
        }
        assert_eq!(gate.diagnostics().ambient_floor_db, cfg.floor_init_db);

        let room = synth::white_noise_hops(80, hop, 0.003, 71);
        for chunk in room.chunks(hop) {
            assert!(
                !gate.process_hop(chunk).open,
                "ordinary room noise after a zero-filled device startup must stay gated"
            );
        }
    }

    #[test]
    fn variable_foreground_during_calibration_is_not_adopted_as_the_room() {
        let cfg = live_config();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();
        let mut startup = synth::sine_hops(8, hop, cfg.sample_rate, 300.0, 0.6);
        startup.extend(synth::white_noise_hops(12, hop, 0.003, 73));
        for chunk in startup.chunks(hop) {
            gate.process_hop(chunk);
        }

        assert!(
            gate.diagnostics().ambient_floor_db < -50.0,
            "a variable startup foreground must not leave a high ambient floor: {:?}",
            gate.diagnostics()
        );
        let event = synth::sine_hops(1, hop, cfg.sample_rate, 300.0, 0.1);
        assert!(gate.process_hop(&event).open);
    }

    #[test]
    fn incomplete_valid_calibration_does_not_adopt_the_following_foreground() {
        let cfg = live_config();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();
        let silence = vec![0.0; hop];
        for _ in 0..10 {
            gate.process_hop(&silence);
        }
        let foreground = synth::sine_hops(11, hop, cfg.sample_rate, 300.0, 0.2);
        for chunk in foreground[..10 * hop].chunks(hop) {
            assert!(!gate.process_hop(chunk).open);
        }

        assert_eq!(gate.diagnostics().ambient_floor_db, cfg.floor_init_db);
        assert!(
            gate.process_hop(&foreground[10 * hop..]).open,
            "the continuing foreground must open immediately after the fixed calibration window"
        );
    }

    #[test]
    fn invalid_slots_do_not_make_one_foreground_hop_ready_for_promotion() {
        let cfg = GateConfig::default();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();
        let silence = vec![0.0; hop];
        for _ in 0..109 {
            gate.process_hop(&silence);
        }
        let before = gate.diagnostics().ambient_floor_db;
        let foreground = synth::sine_hops(1, hop, cfg.sample_rate, 300.0, 0.2);
        let report = gate.process_hop(&foreground);
        assert_eq!(gate.diagnostics().ambient_floor_db, before);
        assert!(report.open, "one valid foreground hop must open the gate");
    }

    #[test]
    fn ambient_promotion_waits_for_enough_valid_observations_after_invalid_audio() {
        let cfg = GateConfig::default();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();
        let silence = vec![0.0; hop];
        for _ in 0..200 {
            gate.process_hop(&silence);
        }
        let initial = gate.diagnostics().ambient_floor_db;
        let room = synth::white_noise_hops(100, hop, 0.003, 75);
        for chunk in room[..99 * hop].chunks(hop) {
            gate.process_hop(chunk);
        }
        assert_eq!(gate.diagnostics().ambient_floor_db, initial);

        gate.process_hop(&room[99 * hop..]);
        assert!(
            gate.diagnostics().ambient_floor_db > initial,
            "promotion should become eligible at the configured valid-observation minimum"
        );
    }

    #[test]
    fn frequent_bursts_do_not_become_the_room() {
        let cfg = GateConfig::default();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();
        for chunk in synth::white_noise_hops(200, hop, 0.003, 79).chunks(hop) {
            gate.process_hop(chunk);
        }

        let mut saw_peak_late = false;
        for cycle in 0..15 {
            for chunk in synth::sine_hops(12, hop, cfg.sample_rate, 300.0, 0.2).chunks(hop) {
                let report = gate.process_hop(chunk);
                if cycle >= 10 {
                    saw_peak_late |= report.energy_peak;
                }
            }
            for chunk in synth::white_noise_hops(8, hop, 0.003, 100 + cycle).chunks(hop) {
                gate.process_hop(chunk);
            }
        }

        assert!(saw_peak_late, "late symptom bursts must remain foreground");
        assert!(
            gate.diagnostics().ambient_floor_db < -45.0,
            "60% symptom occupancy must not replace the quiet baseline: {:?}",
            gate.diagnostics()
        );
    }

    #[test]
    fn ambient_floor_recovers_quickly_without_rebounding_after_a_fan_stops() {
        let cfg = GateConfig::default();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();
        for chunk in synth::white_noise_hops(200, hop, 0.003, 91).chunks(hop) {
            gate.process_hop(chunk);
        }
        for chunk in synth::sine_hops(240, hop, cfg.sample_rate, 120.0, 0.014).chunks(hop) {
            gate.process_hop(chunk);
        }
        assert!(gate.diagnostics().ambient_floor_db > -43.0);

        let quiet = synth::white_noise_hops(100, hop, 0.003, 92);
        let mut chunks = quiet.chunks(hop);
        let mut recovered_at = None;
        for index in 0..60 {
            gate.process_hop(chunks.next().unwrap());
            if gate.diagnostics().ambient_floor_db < -48.0 {
                recovered_at = Some(index + 1);
                break;
            }
        }
        let recovered_at = recovered_at.expect("ambient floor should recover after the fan stops");
        assert!(
            recovered_at <= 40,
            "recovery took {recovered_at} hops ({:.2}s)",
            recovered_at as f32 * cfg.hop_ms as f32 / 1000.0
        );

        for chunk in chunks.by_ref().take(40) {
            gate.process_hop(chunk);
            assert!(
                gate.diagnostics().ambient_floor_db < -48.0,
                "recovered ambient floor rebounded: {:?}",
                gate.diagnostics()
            );
        }
    }

    #[test]
    fn a_quiet_event_opens_during_the_former_rebound_window() {
        let cfg = GateConfig::default();
        let mut gate = Gate::new(cfg.clone());
        let hop = cfg.hop_samples();
        for chunk in synth::white_noise_hops(200, hop, 0.003, 93).chunks(hop) {
            gate.process_hop(chunk);
        }
        for chunk in synth::sine_hops(240, hop, cfg.sample_rate, 120.0, 0.014).chunks(hop) {
            gate.process_hop(chunk);
        }

        let quiet = synth::white_noise_hops(80, hop, 0.003, 94);
        let mut quiet_chunks = quiet.chunks(hop);
        while gate.diagnostics().ambient_floor_db >= -48.0 {
            gate.process_hop(
                quiet_chunks
                    .next()
                    .expect("floor should recover before quiet input ends"),
            );
        }
        for chunk in quiet_chunks.by_ref().take(10) {
            gate.process_hop(chunk);
        }

        let quiet_event = synth::sine_hops(1, hop, cfg.sample_rate, 900.0, 0.03);
        assert!(
            gate.process_hop(&quiet_event).open,
            "a quiet event at the former stale-history rebound must open"
        );
    }
}
