//! Full-stack integration test against the REAL converted `model/yamnet.onnx`:
//! Rust synth sine → core's own log-mel frontend → ort inference → compare the
//! top classes with the Python oracle recorded in `model/CONVERSION.md`
//! (Keras/ONNX top-5 for a 440 Hz sine: Telephone 383, Dial tone 387,
//! Busy signal 388, Alarm 382, Sidetone 512).
//!
//! Skips (passes vacuously, with a note) when the gitignored model file or the
//! ONNX Runtime dylib is unavailable — CI has neither; developers run it via:
//! `ORT_DYLIB_PATH=/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib \
//!  cargo test -p sinus-core --features onnx --test yamnet_real`
#![cfg(feature = "onnx")]

use sinus_core::classify::embed::{Embedder, AUDIOSET_CLASSES, EMBED_DIM};
use sinus_core::classify::yamnet::YamnetOnnx;
use sinus_core::error::Error;
use sinus_core::mel::MelFrontend;
use sinus_core::synth;

const TONE_CLASSES: [usize; 3] = [383, 387, 388]; // Telephone, Dial tone, Busy signal
const ORACLE_TOP5: [usize; 5] = [383, 387, 388, 382, 512];

#[test]
fn real_model_matches_python_oracle_on_sine() {
    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../model/yamnet.onnx");
    let yamnet = match YamnetOnnx::load(model_path) {
        Ok(y) => y,
        Err(Error::ModelUnavailable(why)) => {
            eprintln!("SKIP yamnet_real: {why}");
            return;
        }
        Err(other) => panic!("unexpected error loading model: {other}"),
    };

    let samples = synth::sine(16_000, 16_000, 440.0, 0.5);
    let frontend = MelFrontend::new(16_000);
    let patches = frontend.patches(&samples);
    assert!(!patches.is_empty(), "1 s of audio must yield ≥1 patch");

    let features = yamnet
        .embed(&patches[0], true)
        .expect("inference on the real model");

    assert_eq!(features.embedding.len(), EMBED_DIM);
    let scores = features
        .audioset_scores
        .expect("ONNX embedder must surface AudioSet scores");
    assert_eq!(scores.len(), AUDIOSET_CLASSES);

    let mut ranked: Vec<usize> = (0..scores.len()).collect();
    ranked.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let top5 = &ranked[..5];
    eprintln!(
        "top-5: {:?}",
        top5.iter().map(|&i| (i, scores[i])).collect::<Vec<_>>()
    );

    assert!(
        TONE_CLASSES.contains(&top5[0]),
        "top-1 {} should be a steady-tone class {TONE_CLASSES:?}; scores diverge from the \
         Python oracle — suspect the Rust mel frontend",
        top5[0]
    );
    let overlap = top5.iter().filter(|i| ORACLE_TOP5.contains(i)).count();
    assert!(
        overlap >= 3,
        "only {overlap}/5 of the oracle's top-5 classes present in {top5:?}"
    );
}
