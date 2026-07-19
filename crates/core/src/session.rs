//! Stage ⑤ — sessionizer (SPEC §4.1). Consecutive same-class windows with gaps
//! under 1.5 s merge into one event: `duration_ms` is the merged span,
//! `confidence` is the max window score, and `burst_count` is the number of
//! distinct energy peaks inside the span (a 5-cough fit is one event, burst 5).
//! Per-class cooldowns after an event closes suppress re-grips (e.g. 10 s for
//! `nose_blow`, 2 s for `cough`).

use std::collections::HashMap;

use crate::types::EventType;

/// Sessionizer tuning (SPEC §4.1 ⑤).
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Same-class windows closer than this merge (SPEC: 1.5 s).
    pub merge_gap_ms: i64,
    /// Nominal window length added to a single-window span so a lone detection
    /// has a sensible duration (the 0.5 s patch hop).
    pub window_ms: i64,
    /// Minimum spacing between energy peaks to count them as distinct bursts.
    pub peak_min_gap_ms: i64,
    /// Per-class cooldown after an event closes.
    pub cooldowns_ms: HashMap<EventType, i64>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        let mut cooldowns_ms = HashMap::new();
        cooldowns_ms.insert(EventType::NoseBlow, 10_000);
        cooldowns_ms.insert(EventType::Hawk, 5_000);
        cooldowns_ms.insert(EventType::Cough, 2_000);
        cooldowns_ms.insert(EventType::Sneeze, 2_000);
        cooldowns_ms.insert(EventType::ThroatClearing, 3_000);
        cooldowns_ms.insert(EventType::Sniffle, 2_000);
        cooldowns_ms.insert(EventType::SnortSuck, 3_000);
        SessionConfig {
            merge_gap_ms: 1_500,
            window_ms: 500,
            peak_min_gap_ms: 150,
            cooldowns_ms,
        }
    }
}

impl SessionConfig {
    fn cooldown(&self, et: EventType) -> i64 {
        self.cooldowns_ms.get(&et).copied().unwrap_or(2_000)
    }
}

/// A per-window firing observation fed to the sessionizer.
#[derive(Debug, Clone)]
pub struct WindowObservation {
    pub event_type: EventType,
    pub confidence: f32,
    /// Window start time, ms (monotonic within a stream).
    pub timestamp_ms: i64,
    pub energy_peak: bool,
    /// Backbone embedding of the window (empty when unavailable). The closed
    /// event carries the embedding of its highest-confidence window so a false
    /// positive can later be enrolled as a negative example.
    pub embedding: Vec<f32>,
}

/// A merged, sessionized detection (pre-persistence; the pipeline stamps uuid /
/// occurred_at / model_version onto it).
#[derive(Debug, Clone, PartialEq)]
pub struct DetectedEvent {
    pub event_type: EventType,
    pub start_ms: i64,
    pub end_ms: i64,
    pub duration_ms: i64,
    pub confidence: f32,
    pub burst_count: i64,
    /// Embedding of the highest-confidence window in the session (empty when the
    /// backbone produced none). Kept locally only — never uploaded.
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
struct OpenSession {
    start_ms: i64,
    last_ms: i64,
    max_conf: f32,
    bursts: i64,
    last_peak_ms: Option<i64>,
    best_embedding: Vec<f32>,
}

/// The sessionizer. Feed it window observations in non-decreasing time order;
/// it emits [`DetectedEvent`]s as sessions close, plus any still-open sessions on
/// [`Sessionizer::flush`].
#[derive(Debug, Clone)]
pub struct Sessionizer {
    cfg: SessionConfig,
    open: HashMap<EventType, OpenSession>,
    cooldown_until: HashMap<EventType, i64>,
}

impl Sessionizer {
    pub fn new(cfg: SessionConfig) -> Self {
        Sessionizer {
            cfg,
            open: HashMap::new(),
            cooldown_until: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Sessionizer::new(SessionConfig::default())
    }

    fn close(&mut self, et: EventType) -> DetectedEvent {
        let s = self.open.remove(&et).expect("close: session exists");
        let event = DetectedEvent {
            event_type: et,
            start_ms: s.start_ms,
            end_ms: s.last_ms,
            duration_ms: (s.last_ms - s.start_ms) + self.cfg.window_ms,
            confidence: s.max_conf,
            burst_count: s.bursts.max(1),
            embedding: s.best_embedding,
        };
        self.cooldown_until
            .insert(et, event.end_ms + self.cfg.cooldown(et));
        event
    }

    /// Close any open sessions whose last window is older than `now - merge_gap`.
    fn flush_stale(&mut self, now_ms: i64, out: &mut Vec<DetectedEvent>) {
        let stale: Vec<EventType> = self
            .open
            .iter()
            .filter(|(_, s)| s.last_ms + self.cfg.merge_gap_ms < now_ms)
            .map(|(&et, _)| et)
            .collect();
        for et in stale {
            let ev = self.close(et);
            out.push(ev);
        }
    }

    fn record_peak(session: &mut OpenSession, obs: &WindowObservation, min_gap: i64) {
        if !obs.energy_peak {
            return;
        }
        let distinct = match session.last_peak_ms {
            None => true,
            Some(prev) => obs.timestamp_ms - prev >= min_gap,
        };
        if distinct {
            session.bursts += 1;
            session.last_peak_ms = Some(obs.timestamp_ms);
        }
    }

    /// Feed one observation; returns any events that closed as a result.
    pub fn observe(&mut self, obs: WindowObservation) -> Vec<DetectedEvent> {
        let mut out = Vec::new();
        self.flush_stale(obs.timestamp_ms, &mut out);

        let et = obs.event_type;

        // Respect the per-class cooldown: drop observations that arrive while the
        // class is cooling down and has no currently-open session.
        if !self.open.contains_key(&et) {
            if let Some(&until) = self.cooldown_until.get(&et) {
                if obs.timestamp_ms < until {
                    return out;
                }
            }
        }

        match self.open.get_mut(&et) {
            Some(session) => {
                session.last_ms = obs.timestamp_ms;
                if obs.confidence > session.max_conf {
                    session.max_conf = obs.confidence;
                    session.best_embedding = obs.embedding.clone();
                }
                Self::record_peak(session, &obs, self.cfg.peak_min_gap_ms);
            }
            None => {
                let mut session = OpenSession {
                    start_ms: obs.timestamp_ms,
                    last_ms: obs.timestamp_ms,
                    max_conf: obs.confidence,
                    bursts: 0,
                    last_peak_ms: None,
                    best_embedding: obs.embedding.clone(),
                };
                Self::record_peak(&mut session, &obs, self.cfg.peak_min_gap_ms);
                self.open.insert(et, session);
            }
        }
        out
    }

    /// Close all still-open sessions (call at end of stream / gate close).
    pub fn flush(&mut self) -> Vec<DetectedEvent> {
        let ets: Vec<EventType> = self.open.keys().copied().collect();
        ets.into_iter().map(|et| self.close(et)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(et: EventType, t: i64, conf: f32, peak: bool) -> WindowObservation {
        WindowObservation {
            event_type: et,
            confidence: conf,
            timestamp_ms: t,
            energy_peak: peak,
            embedding: Vec::new(),
        }
    }

    #[test]
    fn merges_same_class_windows_within_gap() {
        let mut s = Sessionizer::with_defaults();
        assert!(s.observe(obs(EventType::Cough, 0, 0.4, false)).is_empty());
        assert!(s.observe(obs(EventType::Cough, 500, 0.7, false)).is_empty());
        assert!(s
            .observe(obs(EventType::Cough, 1000, 0.5, false))
            .is_empty());
        let events = s.flush();
        assert_eq!(events.len(), 1);
        let e = &events[0];
        assert_eq!(e.event_type, EventType::Cough);
        assert_eq!(e.start_ms, 0);
        assert_eq!(e.end_ms, 1000);
        assert_eq!(e.duration_ms, 1500); // span 1000 + one window
        assert!((e.confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn splits_when_gap_exceeds_merge_window() {
        let mut s = Sessionizer::with_defaults();
        s.observe(obs(EventType::Cough, 0, 0.6, false));
        // 3 s later — beyond both the 1.5 s merge gap and the 2 s cough cooldown.
        let closed = s.observe(obs(EventType::Cough, 3_000, 0.6, false));
        assert_eq!(closed.len(), 1, "first session should have closed");
        let rest = s.flush();
        assert_eq!(rest.len(), 1, "second session closes on flush");
    }

    #[test]
    fn counts_distinct_energy_peaks_as_bursts() {
        let mut s = Sessionizer::with_defaults();
        for t in [0, 200, 400, 600, 800] {
            s.observe(obs(EventType::Cough, t, 0.8, true));
        }
        let events = s.flush();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].burst_count, 5);
    }

    #[test]
    fn merged_peaks_too_close_do_not_overcount() {
        let mut s = Sessionizer::with_defaults();
        // Peaks 50 ms apart (< 150 ms min gap) count as one burst.
        for t in [0, 50, 100] {
            s.observe(obs(EventType::Cough, t, 0.8, true));
        }
        let events = s.flush();
        assert_eq!(events[0].burst_count, 1);
    }

    #[test]
    fn per_class_cooldown_suppresses_regrips() {
        let mut s = Sessionizer::with_defaults();
        // One nose blow, then a re-grip 2 s later. Nose-blow cooldown is 10 s, so
        // the re-grip after the first session closes must be dropped.
        s.observe(obs(EventType::NoseBlow, 0, 0.9, true));
        // Force the first session to close via a stale flush at t=2000.
        let closed = s.observe(obs(EventType::NoseBlow, 2_000, 0.9, true));
        assert_eq!(closed.len(), 1, "first nose blow closed");
        // t=2000 < cooldown_until (0 + 10000) → dropped, nothing open.
        assert!(s.flush().is_empty(), "re-grip within cooldown suppressed");

        // After the cooldown, a new blow is accepted.
        s.observe(obs(EventType::NoseBlow, 11_000, 0.9, true));
        assert_eq!(s.flush().len(), 1);
    }

    #[test]
    fn event_carries_embedding_of_highest_confidence_window() {
        let mut s = Sessionizer::with_defaults();
        let mut low = obs(EventType::Cough, 0, 0.4, false);
        low.embedding = vec![1.0, 0.0];
        let mut high = obs(EventType::Cough, 500, 0.9, false);
        high.embedding = vec![0.0, 1.0];
        let mut later_low = obs(EventType::Cough, 1000, 0.5, false);
        later_low.embedding = vec![0.5, 0.5];
        s.observe(low);
        s.observe(high);
        s.observe(later_low);
        let events = s.flush();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].embedding, vec![0.0, 1.0]);
    }

    #[test]
    fn interleaved_classes_form_separate_sessions() {
        let mut s = Sessionizer::with_defaults();
        s.observe(obs(EventType::Cough, 0, 0.6, true));
        s.observe(obs(EventType::Sniffle, 500, 0.5, true));
        s.observe(obs(EventType::Cough, 1000, 0.7, true));
        let events = s.flush();
        assert_eq!(events.len(), 2);
        let cough = events
            .iter()
            .find(|e| e.event_type == EventType::Cough)
            .unwrap();
        // The two coughs (0 and 1000, gap 1000 < 1500) merged.
        assert_eq!(cough.start_ms, 0);
        assert_eq!(cough.end_ms, 1000);
        assert!(events.iter().any(|e| e.event_type == EventType::Sniffle));
    }
}
