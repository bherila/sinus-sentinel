//! Stage ④ — decision logic (SPEC §4.1). Per-class calibrated thresholds scaled
//! by a user sensitivity slider; transient classes fire on a single window ≥ θ_c;
//! weak/short classes additionally require a coincident gate energy peak; and a
//! speech guard suppresses everything except very-high-confidence cough when the
//! Speech score dominates the window.

use std::collections::HashMap;

use crate::classify::native::NativeScores;
use crate::types::EventType;

/// Per-window scores fed to the decision engine: a merged view of the native head
/// and the prototype matcher, plus the Speech score and the energy-peak flag.
#[derive(Debug, Clone)]
pub struct WindowScores {
    pub scores: HashMap<EventType, f32>,
    pub speech: f32,
    pub energy_peak: bool,
}

impl WindowScores {
    /// Merge native head scores with prototype-matcher similarities into one view.
    /// If both produce a score for a class, the larger wins.
    pub fn merge(native: &NativeScores, proto: &[(EventType, f32)], energy_peak: bool) -> Self {
        let mut scores: HashMap<EventType, f32> = HashMap::new();
        for et in [
            EventType::Cough,
            EventType::ThroatClearing,
            EventType::Sniffle,
            EventType::Sneeze,
        ] {
            scores.insert(et, native.score_for(et));
        }
        for &(et, sim) in proto {
            let entry = scores.entry(et).or_insert(0.0);
            *entry = entry.max(sim);
        }
        WindowScores {
            scores,
            speech: native.speech,
            energy_peak,
        }
    }
}

/// Decision tuning (SPEC §4.1 ④).
#[derive(Debug, Clone)]
pub struct DecisionConfig {
    /// Per-class base thresholds θ_c.
    pub thresholds: HashMap<EventType, f32>,
    /// Sensitivity in [0,1]; 0.5 is neutral. Higher lowers all thresholds.
    pub sensitivity: f32,
    /// Speech is "dominant" above this score.
    pub speech_dominant: f32,
    /// …and when it exceeds `speech_ratio` × the candidate class score.
    pub speech_ratio: f32,
    /// A cough this confident survives the speech guard.
    pub high_cough: f32,
    /// Per-class enable flags (disabled classes never fire).
    pub enabled: HashMap<EventType, bool>,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        // Placeholder defaults; real θ_c come from `cli calibrate` on the golden
        // corpus (SPEC §4.1 accuracy loop).
        let mut thresholds = HashMap::new();
        thresholds.insert(EventType::Cough, 0.30);
        thresholds.insert(EventType::ThroatClearing, 0.30);
        thresholds.insert(EventType::Sniffle, 0.25);
        thresholds.insert(EventType::Sneeze, 0.30);
        thresholds.insert(EventType::NoseBlow, 0.55);
        thresholds.insert(EventType::Hawk, 0.55);
        thresholds.insert(EventType::SnortSuck, 0.50);
        let enabled = EventType::ALL.into_iter().map(|e| (e, true)).collect();
        DecisionConfig {
            thresholds,
            sensitivity: 0.5,
            speech_dominant: 0.5,
            speech_ratio: 1.5,
            high_cough: 0.8,
            enabled,
        }
    }
}

impl DecisionConfig {
    /// Effective threshold for a class after sensitivity scaling. Sensitivity 0 →
    /// ×1.5 (stricter), 0.5 → ×1.0, 1.0 → ×0.5 (looser).
    pub fn effective_threshold(&self, et: EventType) -> f32 {
        let base = self.thresholds.get(&et).copied().unwrap_or(0.5);
        let factor = (1.5 - self.sensitivity).clamp(0.25, 1.5);
        base * factor
    }

    fn is_enabled(&self, et: EventType) -> bool {
        self.enabled.get(&et).copied().unwrap_or(true)
    }
}

/// A firing decision for one window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowHit {
    pub event_type: EventType,
    pub confidence: f32,
}

/// The stage-④ decision engine.
#[derive(Debug, Clone, Default)]
pub struct DecisionEngine {
    pub config: DecisionConfig,
}

impl DecisionEngine {
    pub fn new(config: DecisionConfig) -> Self {
        DecisionEngine { config }
    }

    /// Decide whether any class fires for this window; returns the highest-scoring
    /// qualifying class, or `None`.
    pub fn decide(&self, w: &WindowScores) -> Option<WindowHit> {
        let speech_dominates = w.speech > self.config.speech_dominant;
        let mut best: Option<WindowHit> = None;

        for (&et, &score) in &w.scores {
            if !self.config.is_enabled(et) {
                continue;
            }
            let thr = self.config.effective_threshold(et);
            if score < thr {
                continue;
            }
            // Weak/short classes need a coincident energy peak.
            if et.is_weak() && !w.energy_peak {
                continue;
            }
            // Speech guard: when speech dominates and out-scores the candidate,
            // suppress it — except a very-high-confidence cough.
            if speech_dominates && w.speech > self.config.speech_ratio * score {
                let survives = et == EventType::Cough && score >= self.config.high_cough;
                if !survives {
                    continue;
                }
            }
            let hit = WindowHit {
                event_type: et,
                confidence: score,
            };
            best = match best {
                Some(b) if b.confidence >= hit.confidence => Some(b),
                _ => Some(hit),
            };
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(pairs: &[(EventType, f32)], speech: f32, peak: bool) -> WindowScores {
        WindowScores {
            scores: pairs.iter().copied().collect(),
            speech,
            energy_peak: peak,
        }
    }

    #[test]
    fn transient_fires_on_single_window() {
        let eng = DecisionEngine::default();
        let w = scores(&[(EventType::Cough, 0.5)], 0.0, false);
        assert_eq!(eng.decide(&w).unwrap().event_type, EventType::Cough);
    }

    #[test]
    fn weak_class_requires_energy_peak() {
        let eng = DecisionEngine::default();
        let no_peak = scores(&[(EventType::Sniffle, 0.5)], 0.0, false);
        assert!(eng.decide(&no_peak).is_none());
        let with_peak = scores(&[(EventType::Sniffle, 0.5)], 0.0, true);
        assert_eq!(
            eng.decide(&with_peak).unwrap().event_type,
            EventType::Sniffle
        );
    }

    #[test]
    fn speech_guard_suppresses_weak_candidates() {
        let eng = DecisionEngine::default();
        // Speech dominant and >1.5× the sniffle score → suppressed.
        let w = scores(&[(EventType::Sniffle, 0.4)], 0.9, true);
        assert!(eng.decide(&w).is_none());
    }

    #[test]
    fn very_high_cough_survives_speech_guard() {
        let eng = DecisionEngine::default();
        let w = scores(&[(EventType::Cough, 0.85)], 0.95, false);
        assert_eq!(eng.decide(&w).unwrap().event_type, EventType::Cough);
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn sensitivity_scales_thresholds() {
        let mut cfg = DecisionConfig::default();
        // A score of 0.2 is below the cough default (0.30) at neutral sensitivity.
        cfg.sensitivity = 0.5;
        let eng = DecisionEngine::new(cfg.clone());
        assert!(eng
            .decide(&scores(&[(EventType::Cough, 0.2)], 0.0, false))
            .is_none());
        // Crank sensitivity up → threshold drops below 0.2 and it fires.
        cfg.sensitivity = 1.0;
        let eng = DecisionEngine::new(cfg);
        assert!(eng
            .decide(&scores(&[(EventType::Cough, 0.2)], 0.0, false))
            .is_some());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn disabled_class_never_fires() {
        let mut cfg = DecisionConfig::default();
        cfg.enabled.insert(EventType::Cough, false);
        let eng = DecisionEngine::new(cfg);
        assert!(eng
            .decide(&scores(&[(EventType::Cough, 0.9)], 0.0, false))
            .is_none());
    }

    #[test]
    fn picks_highest_scoring_candidate() {
        let eng = DecisionEngine::default();
        let w = scores(
            &[(EventType::Cough, 0.6), (EventType::Sneeze, 0.9)],
            0.0,
            false,
        );
        assert_eq!(eng.decide(&w).unwrap().event_type, EventType::Sneeze);
    }
}
