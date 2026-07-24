//! ONNX-backed YAMNet embedder (SPEC §4.1 stage ③, §5 Phase A). Behind the
//! `onnx` feature so the default build and tests never require the ONNX Runtime
//! shared library or the `yamnet.onnx` model file.
//!
//! Loading is fail-soft: if the model file is absent or fails to load, the
//! constructor returns [`Error::ModelUnavailable`] and the app surfaces a "model
//! missing" state instead of crashing (SPEC §4). The expected model contract
//! (input `[1, 96, 64]` log-mel, outputs 521 AudioSet scores + 1024-d embedding)
//! is documented in `model/README.md`.

use std::path::Path;
use std::sync::Mutex;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use crate::classify::embed::{Embedder, WindowFeatures, AUDIOSET_CLASSES, EMBED_DIM};
use crate::error::{Error, Result};
use crate::mel::MelPatch;

/// Names of the model's input/output tensors. YAMNet ONNX exports vary; these are
/// overridable so a differently-named export works without code changes.
#[derive(Debug, Clone)]
pub struct TensorNames {
    pub input: String,
    pub scores: String,
    pub embedding: String,
}

impl Default for TensorNames {
    fn default() -> Self {
        TensorNames {
            input: "input".to_string(),
            scores: "scores".to_string(),
            embedding: "embeddings".to_string(),
        }
    }
}

/// The ort-backed YAMNet backbone. `Session::run` takes `&mut self`, so the
/// session is behind a `Mutex` — the analysis thread holds it briefly per window.
pub struct YamnetOnnx {
    session: Mutex<Session>,
    names: TensorNames,
    version: String,
}

impl YamnetOnnx {
    /// Load `yamnet.onnx` from `path`. Returns [`Error::ModelUnavailable`] if the
    /// file is missing or cannot be loaded — never panics.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_names(path, TensorNames::default())
    }

    pub fn load_with_names(path: impl AsRef<Path>, names: TensorNames) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(Error::ModelUnavailable(format!(
                "model file not found: {}",
                path.display()
            )));
        }
        let build = || -> ort::Result<Session> {
            let mut builder = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                // YAMNet patches arrive only twice per second while the gate is
                // open. A many-core pool saves little latency but wakes the whole
                // package, so keep the fallback CPU path single-threaded.
                .with_intra_threads(1)?
                .with_inter_threads(1)?;
            #[cfg(target_os = "macos")]
            let mut builder = builder.with_execution_providers([ort::ep::CoreML::default()
                .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndNeuralEngine)
                .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                .with_static_input_shapes(true)
                .build()])?;
            builder.commit_from_file(path)
        };
        let session = build().map_err(|e| {
            Error::ModelUnavailable(format!("failed to load {}: {e}", path.display()))
        })?;
        Ok(YamnetOnnx {
            session: Mutex::new(session),
            names,
            version: "yamnet-onnx".to_string(),
        })
    }
}

impl Embedder for YamnetOnnx {
    fn model_version(&self) -> String {
        self.version.clone()
    }

    fn embed(&self, patch: &MelPatch, energy_peak: bool) -> Result<WindowFeatures> {
        let shape = vec![1i64, patch.frames as i64, patch.bands as i64];
        let input = Tensor::from_array((shape, patch.data.clone()))
            .map_err(|e| Error::ModelUnavailable(format!("tensor build: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::ModelUnavailable("session mutex poisoned".to_string()))?;
        let outputs = session
            .run(ort::inputs![self.names.input.as_str() => input])
            .map_err(|e| Error::ModelUnavailable(format!("inference failed: {e}")))?;

        let extract = |name: &str, expect: usize| -> Result<Vec<f32>> {
            let val = outputs.get(name).ok_or_else(|| {
                Error::ModelUnavailable(format!("missing output tensor `{name}`"))
            })?;
            let (_, data) = val
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::ModelUnavailable(format!("extract `{name}`: {e}")))?;
            // If the model emits a per-frame matrix, average across frames down to
            // `expect` values.
            if data.len() == expect {
                Ok(data.to_vec())
            } else if !data.is_empty() && data.len() % expect == 0 {
                let frames = data.len() / expect;
                let mut acc = vec![0.0f32; expect];
                for f in 0..frames {
                    for (a, &v) in acc.iter_mut().zip(&data[f * expect..(f + 1) * expect]) {
                        *a += v;
                    }
                }
                for a in &mut acc {
                    *a /= frames as f32;
                }
                Ok(acc)
            } else {
                Err(Error::ModelUnavailable(format!(
                    "output `{name}` length {} not a multiple of {expect}",
                    data.len()
                )))
            }
        };

        let audioset = extract(&self.names.scores, AUDIOSET_CLASSES)?;
        let embedding = extract(&self.names.embedding, EMBED_DIM)?;

        Ok(WindowFeatures {
            audioset_scores: Some(audioset),
            embedding,
            energy_peak,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_is_graceful_not_a_panic() {
        let result = YamnetOnnx::load("/nonexistent/yamnet.onnx");
        assert!(matches!(result, Err(Error::ModelUnavailable(_))));
    }
}
