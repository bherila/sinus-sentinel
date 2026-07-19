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
use sinus_core::classify::proto::PrototypeMatcher;
use sinus_core::error::Result as CoreResult;
use sinus_core::mel::{loudest_patch, MelPatch};
use sinus_core::pipeline::{EventContext, PipelineConfig, StreamingPipeline};
use sinus_core::store::Store;
use sinus_core::types::Source;

use crate::shared::{ModelStatus, SharedStatus};

const PROTOTYPE_SIM_THRESHOLD: f32 = 0.65;
const PROTOTYPE_NEGATIVE_MARGIN: f32 = 0.05;
const TEACH_CAPTURE_SAMPLES: usize = sinus_core::types::SAMPLE_RATE as usize * 3;

struct TeachCapture {
    class: sinus_core::types::EventType,
    samples: Vec<f32>,
}

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
fn build_embedder(store: &Store, shared: &SharedStatus) -> CaptureEmbedder {
    #[cfg(feature = "onnx")]
    {
        configure_ort_dylib_path();
        let path = store
            .setting_get("model_path")
            .ok()
            .flatten()
            .map(PathBuf::from)
            .unwrap_or_else(default_model_path);
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
        let _ = store;
        shared.set_model(ModelStatus::Heuristic);
        CaptureEmbedder::Heuristic(BandHeuristicEmbedder)
    }
}

#[cfg(feature = "onnx")]
fn default_model_path() -> PathBuf {
    #[cfg(target_os = "macos")]
    if let Ok(exe) = std::env::current_exe() {
        if let Some(contents) = exe.parent().and_then(std::path::Path::parent) {
            let bundled = contents.join("Resources/model/yamnet.onnx");
            if bundled.exists() {
                return bundled;
            }
        }
    }
    PathBuf::from("model/yamnet.onnx")
}

#[cfg(all(feature = "onnx", target_os = "macos"))]
fn configure_ort_dylib_path() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }
    for candidate in [
        "/opt/homebrew/lib/libonnxruntime.dylib",
        "/usr/local/lib/libonnxruntime.dylib",
    ] {
        if std::path::Path::new(candidate).exists() {
            std::env::set_var("ORT_DYLIB_PATH", candidate);
            break;
        }
    }
}

#[cfg(all(feature = "onnx", not(target_os = "macos")))]
fn configure_ort_dylib_path() {}

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

    let prototypes = prototypes_from_store(&store)?;

    // One StreamingPipeline for the life of the stream (SPEC §1, live capture):
    // gate/sessionizer/mel state persists across reads, so events straddling a read
    // boundary merge, cooldowns persist, and the noise floor converges. Detected
    // events carry sample-counter timestamps relative to the stream start; map that
    // origin to wall-clock ONCE, here, rather than doing per-chunk `Utc::now()` math.
    let mut config = PipelineConfig::default();
    config.decision.sensitivity = store
        .setting_get("sensitivity")
        .ok()
        .flatten()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.5);
    let mut pipeline = StreamingPipeline::new(config, embedder);
    if let Some(prototypes) = prototypes {
        pipeline = pipeline.with_prototypes(prototypes);
    }
    let stream_start = Utc::now();
    let ctx = EventContext {
        base_time: stream_start,
        tz_offset_min: local_offset_minutes(),
        device_id,
        source: Source::current_desktop(),
        model_version: pipeline.model_version(),
    };

    let mut buf = vec![0.0f32; 4096];
    let mut teach_capture: Option<TeachCapture> = None;
    loop {
        if teach_capture.is_none() {
            if let Some(class) = shared.take_teach_request() {
                shared.set_teach_recording(class);
                teach_capture = Some(TeachCapture {
                    class,
                    samples: Vec::with_capacity(TEACH_CAPTURE_SAMPLES),
                });
            }
        }

        let n = source.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            continue;
        }
        let teaching = teach_capture.is_some();
        if let Some(capture) = teach_capture.as_mut() {
            let remaining = TEACH_CAPTURE_SAMPLES.saturating_sub(capture.samples.len());
            capture.samples.extend_from_slice(&buf[..n.min(remaining)]);
        }

        if let Ok(events) = pipeline.push(&buf[..n]) {
            // Quiet hours suppress detection *logging* (SPEC §6): keep running the
            // pipeline (so state/floor/cooldowns stay continuous) but drop the
            // events at the write site instead of persisting them. The flag is
            // published by the sync thread from the quiet-hours setting.
            if !shared.quiet() && !teaching {
                for detected in &events {
                    let event = pipeline.to_event(detected, &ctx);
                    let _ = store.insert_event(&event);
                }
            }
        }

        if teach_capture
            .as_ref()
            .is_some_and(|capture| capture.samples.len() >= TEACH_CAPTURE_SAMPLES)
        {
            let capture = teach_capture.take().expect("checked above");
            match save_teach_sample(&store, &mut pipeline, capture.class, &capture.samples) {
                Ok((examples, similarity, separation)) => {
                    shared.finish_teach(capture.class, examples, similarity, separation);
                }
                Err(error) => {
                    eprintln!("teach: {error}");
                    shared.fail_teach(capture.class);
                }
            }
        }
    }
}

fn prototypes_from_store(store: &Store) -> Result<Option<PrototypeMatcher>, String> {
    let enrollments: Vec<_> = store
        .enrollments()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|stored| stored.enrollment)
        .collect();
    Ok((!enrollments.is_empty()).then(|| {
        PrototypeMatcher::from_enrollments(
            &enrollments,
            PROTOTYPE_SIM_THRESHOLD,
            PROTOTYPE_NEGATIVE_MARGIN,
        )
    }))
}

fn save_teach_sample(
    store: &Store,
    pipeline: &mut StreamingPipeline<CaptureEmbedder>,
    class: sinus_core::types::EventType,
    samples: &[f32],
) -> Result<(usize, f32, f32), String> {
    let patch = loudest_patch(samples).ok_or("no complete analysis window captured")?;
    let embedding = pipeline
        .embed_patch(&patch, true)
        .map_err(|e| e.to_string())?;

    let existing: Vec<_> = store
        .enrollments()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|stored| stored.enrollment)
        .collect();
    let has_same_class = existing.iter().any(|example| example.class == class);
    let (similarity, separation) = if existing.is_empty() || !has_same_class {
        (-1.0, 0.0)
    } else {
        let matcher = PrototypeMatcher::from_enrollments(
            &existing,
            PROTOTYPE_SIM_THRESHOLD,
            PROTOTYPE_NEGATIVE_MARGIN,
        );
        let similarities = matcher.similarities(&embedding);
        let same = similarities
            .iter()
            .find(|(candidate, _)| *candidate == class)
            .map(|(_, score)| *score)
            .unwrap_or(-1.0);
        let other = similarities
            .iter()
            .filter(|(candidate, _)| *candidate != class)
            .map(|(_, score)| *score)
            .fold(-1.0f32, f32::max);
        (same, same - other)
    };

    store
        .add_enrollment(class, &embedding, false)
        .map_err(|e| e.to_string())?;
    pipeline.set_prototypes(prototypes_from_store(store)?);
    let examples = store
        .enrollment_counts()
        .map_err(|e| e.to_string())?
        .get(&class)
        .copied()
        .unwrap_or(0) as usize;
    Ok((examples, similarity, separation))
}

/// Local UTC offset in minutes.
fn local_offset_minutes() -> i32 {
    chrono::Local::now().offset().local_minus_utc() / 60
}
