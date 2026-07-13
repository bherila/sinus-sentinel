//! Live capture thread (SPEC §4, §6) — behind the `live-audio` feature. Opens the
//! microphone via `cpal`, runs the identical core pipeline, and writes detected
//! events to the store. Requires OS mic permission (granted on first run).
//!
//! Backbone selection (SPEC §4 stage ③): built with `--features onnx` the thread
//! tries [`YamnetOnnx::load`] (model path from the `model_path` setting, default
//! `model/yamnet.onnx`, honoring `ORT_DYLIB_PATH` via ort's load-dynamic). On any
//! load failure it falls back to the model-free [`BandHeuristicEmbedder`] and
//! surfaces a "model missing" state in the tray. Without the feature it always
//! uses the heuristic backbone. The pipeline stages are otherwise identical.

use std::path::PathBuf;
use std::thread::JoinHandle;

use chrono::Utc;
use sinus_core::audio::{AudioSource, CpalAudioSource};
use sinus_core::classify::embed::{BandHeuristicEmbedder, Embedder, WindowFeatures};
use sinus_core::error::Result as CoreResult;
use sinus_core::mel::MelPatch;
use sinus_core::pipeline::{EventContext, PipelineConfig, StreamingPipeline};
use sinus_core::store::Store;
use sinus_core::types::Source;

use crate::shared::{ModelStatus, SharedStatus};

/// The backbone the capture thread runs. An enum (not `dyn`) so the generic
/// [`Pipeline`] stays monomorphized and the ONNX variant only exists when the
/// feature is on.
enum CaptureEmbedder {
    Heuristic(BandHeuristicEmbedder),
    #[cfg(feature = "onnx")]
    Yamnet(sinus_core::classify::yamnet::YamnetOnnx),
}

impl Embedder for CaptureEmbedder {
    fn model_version(&self) -> String {
        match self {
            CaptureEmbedder::Heuristic(e) => e.model_version(),
            #[cfg(feature = "onnx")]
            CaptureEmbedder::Yamnet(e) => e.model_version(),
        }
    }

    fn embed(&self, patch: &MelPatch, energy_peak: bool) -> CoreResult<WindowFeatures> {
        match self {
            CaptureEmbedder::Heuristic(e) => e.embed(patch, energy_peak),
            #[cfg(feature = "onnx")]
            CaptureEmbedder::Yamnet(e) => e.embed(patch, energy_peak),
        }
    }
}

/// Pick the backbone, publishing the resulting [`ModelStatus`] to the tray.
fn build_embedder(_store: &Store, shared: &SharedStatus) -> CaptureEmbedder {
    #[cfg(feature = "onnx")]
    {
        let path = _store
            .setting_get("model_path")
            .ok()
            .flatten()
            .unwrap_or_else(|| "model/yamnet.onnx".to_string());
        match sinus_core::classify::yamnet::YamnetOnnx::load(&path) {
            Ok(y) => {
                shared.set_model(ModelStatus::Onnx);
                CaptureEmbedder::Yamnet(y)
            }
            Err(e) => {
                eprintln!("capture: ONNX model unavailable ({e}); falling back to band-heuristic");
                shared.set_model(ModelStatus::Missing);
                CaptureEmbedder::Heuristic(BandHeuristicEmbedder)
            }
        }
    }
    #[cfg(not(feature = "onnx"))]
    {
        shared.set_model(ModelStatus::Heuristic);
        CaptureEmbedder::Heuristic(BandHeuristicEmbedder)
    }
}

/// Spawn the capture thread. Returns its handle; the thread runs until the process
/// exits. Errors (no device, permission denied) are logged, not fatal.
pub fn spawn_capture(db_path: PathBuf, shared: SharedStatus) -> JoinHandle<()> {
    std::thread::spawn(move || {
        if let Err(e) = run(db_path, shared) {
            eprintln!("capture: {e}");
        }
    })
}

fn run(db_path: PathBuf, shared: SharedStatus) -> Result<(), String> {
    let store = Store::open(&db_path).map_err(|e| e.to_string())?;
    let device_id = store
        .setting_get("device_id")
        .ok()
        .flatten()
        .unwrap_or_else(|| "unknown".to_string());

    let mut source = CpalAudioSource::open_default().map_err(|e| e.to_string())?;
    let embedder = build_embedder(&store, &shared);

    // One StreamingPipeline for the life of the stream (SPEC §1, live capture):
    // gate/sessionizer/mel state persists across reads, so events straddling a read
    // boundary merge, cooldowns persist, and the noise floor converges. Detected
    // events carry sample-counter timestamps relative to the stream start; map that
    // origin to wall-clock ONCE, here, rather than doing per-chunk `Utc::now()` math.
    let mut pipeline = StreamingPipeline::new(PipelineConfig::default(), embedder);
    let stream_start = Utc::now();
    let ctx = EventContext {
        base_time: stream_start,
        tz_offset_min: local_offset_minutes(),
        device_id,
        source: Source::current_desktop(),
        model_version: pipeline.model_version(),
    };

    let mut buf = vec![0.0f32; 4096];
    loop {
        let n = source.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            continue;
        }
        if let Ok(events) = pipeline.push(&buf[..n]) {
            for detected in &events {
                let event = pipeline.to_event(detected, &ctx);
                let _ = store.insert_event(&event);
            }
        }
    }
}

/// Local UTC offset in minutes.
fn local_offset_minutes() -> i32 {
    chrono::Local::now().offset().local_minus_utc() / 60
}
