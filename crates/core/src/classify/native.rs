//! Native head (SPEC §5 Phase A): map YAMNet's AudioSet class scores to our four
//! natively-covered event types, plus the Speech score used by the speech guard.

use crate::types::EventType;

/// Indices into YAMNet's 521-class AudioSet output for the classes we consume.
///
/// Defaults are the standard `yamnet_class_map.csv` positions. They are kept
/// configurable (not hard-coded at the call site) so a model export that renumbers
/// classes can be corrected without touching decision logic — see model/README.md.
#[derive(Debug, Clone, Copy)]
pub struct AudiosetMap {
    pub speech: usize,
    pub cough: usize,
    pub throat_clearing: usize,
    pub sneeze: usize,
    pub sniff: usize,
}

impl Default for AudiosetMap {
    fn default() -> Self {
        AudiosetMap {
            speech: 0,
            cough: 42,
            throat_clearing: 43,
            sneeze: 44,
            sniff: 45,
        }
    }
}

/// Per-class native scores derived from one window's AudioSet output.
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeScores {
    pub cough: f32,
    pub throat_clearing: f32,
    pub sniffle: f32,
    pub sneeze: f32,
    /// Speech score — feeds the speech guard, not an event type.
    pub speech: f32,
}

impl AudiosetMap {
    /// Extract native scores from a 521-length AudioSet score vector. Out-of-range
    /// indices read as 0 (defensive against a mismatched export).
    pub fn native_scores(&self, audioset: &[f32]) -> NativeScores {
        let get = |i: usize| audioset.get(i).copied().unwrap_or(0.0);
        NativeScores {
            cough: get(self.cough),
            throat_clearing: get(self.throat_clearing),
            sniffle: get(self.sniff),
            sneeze: get(self.sneeze),
            speech: get(self.speech),
        }
    }
}

impl NativeScores {
    /// Score for a given native event type (0 for non-native types).
    pub fn score_for(&self, et: EventType) -> f32 {
        match et {
            EventType::Cough => self.cough,
            EventType::ThroatClearing => self.throat_clearing,
            EventType::Sniffle => self.sniffle,
            EventType::Sneeze => self.sneeze,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_mapped_indices() {
        let map = AudiosetMap::default();
        let mut v = vec![0.0f32; 521];
        v[0] = 0.7; // speech
        v[42] = 0.6; // cough
        v[45] = 0.4; // sniff → sniffle
        let ns = map.native_scores(&v);
        assert_eq!(ns.speech, 0.7);
        assert_eq!(ns.cough, 0.6);
        assert_eq!(ns.sniffle, 0.4);
        assert_eq!(ns.sneeze, 0.0);
    }

    #[test]
    fn out_of_range_reads_zero() {
        let map = AudiosetMap::default();
        let ns = map.native_scores(&[0.1, 0.2]);
        assert_eq!(ns.cough, 0.0);
    }
}
