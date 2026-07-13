//! The backbone abstraction (SPEC §4.1 stage ③): something that turns a log-mel
//! patch into per-window features — the 1024-d YAMNet embedding and, when
//! available, the 521 AudioSet class scores. One forward pass serves both the
//! native head (§5 Phase A) and the prototype matcher (§5 Phase B-lite).
//!
//! The real backbone is the ort-backed YAMNet ([`crate::classify::yamnet`], behind
//! the `onnx` feature). This module also provides deterministic, model-free
//! backbones so the whole pipeline is testable without `yamnet.onnx`.

use crate::error::Result;
use crate::mel::MelPatch;

/// AudioSet-scale embedding dimension emitted by YAMNet.
pub const EMBED_DIM: usize = 1024;
/// Number of AudioSet classes YAMNet scores.
pub const AUDIOSET_CLASSES: usize = 521;

/// Features for a single 0.96 s window.
#[derive(Debug, Clone)]
pub struct WindowFeatures {
    /// 521 AudioSet class scores (0..1), if the backbone produces them.
    pub audioset_scores: Option<Vec<f32>>,
    /// 1024-d YAMNet embedding.
    pub embedding: Vec<f32>,
    /// Whether a gate energy peak coincided with this window (SPEC §4.1 ④).
    pub energy_peak: bool,
}

/// A backbone that embeds mel patches. Implemented by the ONNX YAMNet and by the
/// deterministic test backbones below.
pub trait Embedder {
    /// Model version string recorded on every event (SPEC §5, e.g.
    /// `yamnet+proto@N` or `yamnet-onnx`).
    fn model_version(&self) -> String;

    /// Run one forward pass over a mel patch.
    fn embed(&self, patch: &MelPatch, energy_peak: bool) -> Result<WindowFeatures>;
}

/// Canonical corpus tone → class mapping. The synthetic golden corpus emits a
/// tone burst at each of these frequencies; [`BandHeuristicEmbedder`] maps the
/// dominant mel band back to the class, giving a fully deterministic,
/// model-free classification path for tests and the CLI fallback.
pub const CORPUS_TONES: [(crate::types::EventType, f32); 4] = [
    (crate::types::EventType::Cough, 300.0),
    (crate::types::EventType::ThroatClearing, 700.0),
    (crate::types::EventType::Sneeze, 1500.0),
    (crate::types::EventType::Sniffle, 4500.0),
];

/// A deterministic, model-free backbone. It reads the dominant mel band of a
/// patch, maps its center frequency to the nearest [`CORPUS_TONES`] class, and
/// synthesizes both AudioSet scores and an embedding from the band energies.
/// This is *not* a real recognizer — it exists so gate→mel→decision→session can
/// be exercised end-to-end without `yamnet.onnx`.
#[derive(Debug, Clone, Default)]
pub struct BandHeuristicEmbedder;

impl BandHeuristicEmbedder {
    /// Center frequency (Hz) of mel band `j`, mirroring the filterbank layout.
    fn band_center_hz(j: usize) -> f32 {
        let lo = crate::mel::hz_to_mel(crate::mel::MEL_MIN_HZ);
        let hi = crate::mel::hz_to_mel(crate::mel::MEL_MAX_HZ);
        let mel = lo + (hi - lo) * (j as f32 + 1.0) / (crate::mel::N_MEL + 1) as f32;
        // Invert HTK mel: hz = 700 * (exp(mel/1127) - 1).
        700.0 * ((mel / 1127.0).exp() - 1.0)
    }
}

impl Embedder for BandHeuristicEmbedder {
    fn model_version(&self) -> String {
        "band-heuristic@0".to_string()
    }

    fn embed(&self, patch: &MelPatch, energy_peak: bool) -> Result<WindowFeatures> {
        let means = patch.band_means();
        let argmax = means
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let dominant_hz = Self::band_center_hz(argmax);

        // Nearest corpus tone → class.
        let (class, _) = CORPUS_TONES
            .iter()
            .min_by(|a, b| {
                (a.1 - dominant_hz)
                    .abs()
                    .partial_cmp(&(b.1 - dominant_hz).abs())
                    .unwrap()
            })
            .copied()
            .unwrap();

        // Synthesize AudioSet scores: put a strong score at the mapped native
        // index, leave the rest low. Uses the same indices as the real head.
        let map = super::native::AudiosetMap::default();
        let mut scores = vec![0.02f32; AUDIOSET_CLASSES];
        let idx = match class {
            crate::types::EventType::Cough => map.cough,
            crate::types::EventType::ThroatClearing => map.throat_clearing,
            crate::types::EventType::Sneeze => map.sneeze,
            crate::types::EventType::Sniffle => map.sniff,
            _ => map.cough,
        };
        if idx < scores.len() {
            scores[idx] = 0.9;
        }

        // Synthesize a 1024-d embedding by tiling the 64 band means.
        let mut embedding = Vec::with_capacity(EMBED_DIM);
        while embedding.len() < EMBED_DIM {
            embedding.extend_from_slice(&means);
        }
        embedding.truncate(EMBED_DIM);

        Ok(WindowFeatures {
            audioset_scores: Some(scores),
            embedding,
            energy_peak,
        })
    }
}

/// A fully-scripted backbone for unit tests: returns whatever features you hand
/// it, ignoring the patch.
#[derive(Debug, Clone)]
pub struct MockEmbedder {
    pub features: WindowFeatures,
    pub version: String,
}

impl Embedder for MockEmbedder {
    fn model_version(&self) -> String {
        self.version.clone()
    }

    fn embed(&self, _patch: &MelPatch, energy_peak: bool) -> Result<WindowFeatures> {
        let mut f = self.features.clone();
        f.energy_peak = energy_peak;
        Ok(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mel::MelFrontend;
    use crate::synth;
    use crate::types::EventType;

    #[test]
    fn band_heuristic_maps_tone_bursts_to_corpus_classes() {
        let fe = MelFrontend::new(16_000);
        let emb = BandHeuristicEmbedder;
        for (class, freq) in CORPUS_TONES {
            let tone = synth::sine(16_000, 16_000, freq, 0.8);
            let patch = &fe.patches(&tone)[0];
            let feat = emb.embed(patch, true).unwrap();
            let scores = feat.audioset_scores.unwrap();
            let argmax = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            let map = super::super::native::AudiosetMap::default();
            let expected_idx = match class {
                EventType::Cough => map.cough,
                EventType::ThroatClearing => map.throat_clearing,
                EventType::Sneeze => map.sneeze,
                EventType::Sniffle => map.sniff,
                _ => unreachable!(),
            };
            assert_eq!(
                argmax, expected_idx,
                "tone {freq} Hz should map to {class:?}"
            );
            assert_eq!(feat.embedding.len(), EMBED_DIM);
        }
    }
}
