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

use sinus_app::monitor::prototypes_from_store;
use sinus_app::settings;
use sinus_app::state::local_offset_minutes;
use sinus_app::teach::{self, TakeBuffer};

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

/// Event embeddings only exist to let the user report a recent false positive;
/// past this age the event is no longer surfaced for reporting.
const EVENT_EMBEDDING_RETENTION_DAYS: i64 = 30;

fn run(db_path: PathBuf, shared: SharedStatus) -> Result<(), String> {
    shared.register_capture_thread();
    let store = Store::open(&db_path).map_err(|e| e.to_string())?;
    let _ = store.prune_event_embeddings(
        Utc::now() - chrono::Duration::days(EVENT_EMBEDDING_RETENTION_DAYS),
    );
    let device_id = settings::ensure_device_id(&store);

    let embedder = build_embedder(&store, &shared);

    let prototypes = prototypes_from_store(&store).map_err(|e| e.to_string())?;

    // One StreamingPipeline for the life of the stream (SPEC §1, live capture):
    // gate/sessionizer/mel state persists across reads, so events straddling a read
    // boundary merge, cooldowns persist, and the noise floor converges. Detected
    // events carry sample-counter timestamps relative to the stream start; map that
    // origin to wall-clock ONCE, here, rather than doing per-chunk `Utc::now()` math.
    let mut config = PipelineConfig::default();
    config.decision.sensitivity = settings::sensitivity(&store);
    let mut pipeline = StreamingPipeline::new(config, embedder);
    if let Some(prototypes) = prototypes {
        pipeline = pipeline.with_prototypes(prototypes);
    }
    let mut ctx = EventContext {
        base_time: Utc::now(),
        tz_offset_min: local_offset_minutes(),
        device_id,
        source: Source::current_desktop(),
        model_version: pipeline.model_version(),
    };

    let mut buf = vec![0.0f32; 4096];
    let mut source: Option<CpalAudioSource> = None;
    let mut teach_capture: Option<TakeBuffer> = None;
    let mut gate_was_open = false;
    let mut pause_on_low_power = settings::pause_on_low_power(&store);
    let mut last_power_check: Option<std::time::Instant> = None;
    loop {
        if shared.quitting() {
            return Ok(());
        }

        if shared.take_enrollment_reload() {
            pipeline.set_prototypes(prototypes_from_store(&store).map_err(|e| e.to_string())?);
        }

        // Sensitivity and power policy can change locally or arrive from sync.
        if shared.take_settings_reload() {
            pipeline.set_sensitivity(settings::sensitivity(&store));
            pause_on_low_power = settings::pause_on_low_power(&store);
            last_power_check = None;
        }
        if last_power_check
            .is_none_or(|checked| checked.elapsed() >= std::time::Duration::from_secs(60))
        {
            shared.set_low_power(pause_on_low_power && crate::power::low_power_mode_enabled());
            last_power_check = Some(std::time::Instant::now());
        }

        let now = Utc::now();
        if shared.capture_suspended(now) {
            // Dropping the stream releases the microphone device and lets the
            // hardware/driver power down. Reset the streaming state because a
            // paused gap must not merge sessions or skew sample-clock timestamps.
            source = None;
            pipeline.reset_stream();
            if let Some(capture) = teach_capture.take() {
                shared.fail_teach(capture.class());
            }
            gate_was_open = false;
            shared.set_analyzing(false);
            std::thread::park_timeout(shared.suspension_wait(now));
            continue;
        }

        if source.is_none() {
            source = Some(CpalAudioSource::open_default().map_err(|e| e.to_string())?);
            pipeline.reset_stream();
            ctx.base_time = Utc::now();
            ctx.tz_offset_min = local_offset_minutes();
            gate_was_open = false;
        }

        if teach_capture.is_none() {
            if let Some(class) = shared.take_teach_request() {
                teach_capture = Some(TakeBuffer::new(class));
            }
        }

        let n = source
            .as_mut()
            .expect("source opened above")
            .read(&mut buf)
            .map_err(|e| e.to_string())?;
        if n == 0 {
            continue;
        }
        let teaching = teach_capture.is_some();
        if let Some(capture) = teach_capture.as_mut() {
            if capture.push(&buf[..n]) {
                shared.set_teach_recording(capture.class());
            }
        }

        // Publish "we can hear something" so the UI isn't silent during the
        // second or two between a sound arriving and its classification.
        let gate_open = pipeline.gate_open();
        if gate_open && !gate_was_open {
            shared.note_heard(Utc::now());
        }
        gate_was_open = gate_open;
        shared.set_analyzing(gate_open && !teaching);

        if let Ok(events) = pipeline.push(&buf[..n]) {
            // Quiet hours suppress detection *logging* (SPEC §6): keep running the
            // pipeline (so state/floor/cooldowns stay continuous) but drop the
            // events at the write site instead of persisting them. The flag is
            // published by the sync thread from the quiet-hours setting.
            if !shared.quiet() && !teaching {
                for detected in &events {
                    let event = pipeline.to_event(detected, &ctx);
                    if store.insert_event(&event).is_ok() {
                        shared.notify_event_persisted();
                    }
                    // Kept locally only (never uploaded) so a false-positive
                    // report can enroll this exact sound as a negative example.
                    let _ = store.put_event_embedding(&event.uuid, &detected.embedding);
                }
            }
        }

        if teach_capture.as_ref().is_some_and(TakeBuffer::is_complete) {
            let capture = teach_capture.take().expect("checked above");
            match teach::enroll_take(&store, &pipeline, capture.class(), capture.samples()) {
                Ok(result) => {
                    pipeline
                        .set_prototypes(prototypes_from_store(&store).map_err(|e| e.to_string())?);
                    shared.finish_teach(
                        result.class,
                        result.examples,
                        result.similarity,
                        result.separation,
                    );
                    // Push the new take so another machine inherits it without
                    // waiting for an unrelated event flush.
                    shared.request_sync_now();
                }
                Err(error) => {
                    eprintln!("teach: {error}");
                    shared.fail_teach(capture.class());
                }
            }
        }
    }
}
