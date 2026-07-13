//! Golden-corpus test (SPEC §4.1 accuracy loop). Replays the committed synthetic
//! WAV corpus through the *identical* core pipeline (gate → mel → embed →
//! decision → sessionizer) and asserts a deterministic scorecard. The
//! deterministic `BandHeuristicEmbedder` stands in for YAMNet so this runs in CI
//! without `yamnet.onnx`.
//!
//! Regenerate the corpus with: `cargo run -p sinus-cli -- gen-testdata testdata`.

use std::path::PathBuf;

use sinus_core::audio::{AudioSource, BufferedAudioSource};
use sinus_core::classify::embed::BandHeuristicEmbedder;
use sinus_core::pipeline::{Pipeline, PipelineConfig, PipelineResult};
use sinus_core::types::EventType;

fn testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("testdata")
}

fn run(name: &str) -> PipelineResult {
    let path = testdata_dir().join(name);
    let mut src = BufferedAudioSource::open_wav(&path)
        .unwrap_or_else(|e| panic!("open {path:?}: {e} (run `gen-testdata testdata`)"));
    let mut samples = Vec::new();
    let mut buf = vec![0.0f32; 16_000];
    loop {
        let n = src.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        samples.extend_from_slice(&buf[..n]);
    }
    Pipeline::new(PipelineConfig::default(), BandHeuristicEmbedder)
        .process(&samples)
        .unwrap()
}

#[test]
fn positives_yield_exactly_one_event_of_the_labeled_class() {
    let cases = [
        ("cough.wav", EventType::Cough),
        ("throat_clearing.wav", EventType::ThroatClearing),
        ("sneeze.wav", EventType::Sneeze),
        ("sniffle.wav", EventType::Sniffle),
    ];
    for (file, class) in cases {
        let result = run(file);
        assert_eq!(
            result.events.len(),
            1,
            "{file}: expected 1 event, got {:?}",
            result.events
        );
        assert_eq!(result.events[0].event_type, class, "{file}");
        // burst_count comes from distinct energy peaks in the loud span.
        assert!(result.events[0].burst_count >= 1, "{file}");
        assert!(result.events[0].duration_ms > 0, "{file}");
    }
}

#[test]
fn quiet_negatives_produce_no_events_and_never_open_the_gate() {
    for file in ["silence.wav", "quiet_room.wav"] {
        let result = run(file);
        assert!(
            result.events.is_empty(),
            "{file}: expected no events, got {:?}",
            result.events
        );
        assert!(
            result.windows.iter().all(|w| !w.active),
            "{file}: gate must stay closed on a quiet negative"
        );
    }
}

#[test]
fn loud_negative_opens_the_gate() {
    // The hard negative is loud, so the gate opens and windows are analyzed — the
    // deterministic backbone cannot *reject* it (that is YAMNet's job, covered by
    // the speech guard and enrolled-negative unit tests). We only assert the gate
    // and analysis path engage.
    let result = run("noise_burst.wav");
    assert!(
        result.windows.iter().any(|w| w.active),
        "loud negative should open the gate"
    );
}
