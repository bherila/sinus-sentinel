//! Live capture thread (SPEC §4, §6) — behind the `live-audio` feature. Opens the
//! microphone via `cpal`, runs the identical core pipeline, and writes detected
//! events to the store. Requires OS mic permission (granted on first run).
//!
//! This uses the model-free `BandHeuristicEmbedder` as a placeholder backbone; a
//! real deployment builds with `--features onnx` and swaps in `YamnetOnnx`
//! (see model/README.md). The pipeline stages are otherwise identical to the CLI.

use std::path::PathBuf;
use std::thread::JoinHandle;

use chrono::Utc;
use sinus_core::audio::{AudioSource, CpalAudioSource};
use sinus_core::classify::embed::BandHeuristicEmbedder;
use sinus_core::pipeline::{EventContext, Pipeline, PipelineConfig};
use sinus_core::store::Store;
use sinus_core::types::{Source, SAMPLE_RATE};

/// Spawn the capture thread. Returns its handle; the thread runs until the process
/// exits. Errors (no device, permission denied) are logged, not fatal.
pub fn spawn_capture(db_path: PathBuf) -> JoinHandle<()> {
    std::thread::spawn(move || {
        if let Err(e) = run(db_path) {
            eprintln!("capture: {e}");
        }
    })
}

fn run(db_path: PathBuf) -> Result<(), String> {
    let store = Store::open(&db_path).map_err(|e| e.to_string())?;
    let device_id = store
        .setting_get("device_id")
        .ok()
        .flatten()
        .unwrap_or_else(|| "unknown".to_string());

    let mut source = CpalAudioSource::open_default().map_err(|e| e.to_string())?;
    let pipeline = Pipeline::new(PipelineConfig::default(), BandHeuristicEmbedder);

    // Analyze in ~3 s chunks (the gate keeps downstream near-zero when quiet).
    let chunk_samples = (SAMPLE_RATE as usize) * 3;
    let mut buf = vec![0.0f32; 4096];
    let mut window: Vec<f32> = Vec::with_capacity(chunk_samples);

    loop {
        let n = source.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            continue;
        }
        window.extend_from_slice(&buf[..n]);
        if window.len() < chunk_samples {
            continue;
        }

        let base_time = Utc::now()
            - chrono::Duration::milliseconds((window.len() * 1000 / SAMPLE_RATE as usize) as i64);
        if let Ok(result) = pipeline.process(&window) {
            let ctx = EventContext {
                base_time,
                tz_offset_min: local_offset_minutes(),
                device_id: device_id.clone(),
                source: Source::current_desktop(),
                model_version: pipeline.model_version(),
            };
            for detected in &result.events {
                let event = pipeline.to_event(detected, &ctx);
                let _ = store.insert_event(&event);
            }
        }
        window.clear();
    }
}

/// Local UTC offset in minutes.
fn local_offset_minutes() -> i32 {
    chrono::Local::now().offset().local_minus_utc() / 60
}
