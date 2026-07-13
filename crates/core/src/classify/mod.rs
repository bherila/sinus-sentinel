//! Stage ③/④ — classification (SPEC §4.1, §5).
//!
//! - [`embed`]: the [`Embedder`] backbone trait (ONNX YAMNet + model-free test
//!   backbones) producing per-window features.
//! - [`native`]: Phase A native head mapping AudioSet scores to our classes.
//! - [`proto`]: Phase B-lite few-shot [`PrototypeMatcher`].
//! - [`decision`]: stage-④ [`DecisionEngine`] (thresholds, weak-class coincidence,
//!   speech guard).

pub mod decision;
pub mod embed;
pub mod native;
pub mod proto;

#[cfg(feature = "onnx")]
pub mod yamnet;

pub use decision::{DecisionConfig, DecisionEngine, WindowHit, WindowScores};
pub use embed::{BandHeuristicEmbedder, Embedder, MockEmbedder, WindowFeatures, EMBED_DIM};
pub use native::{AudiosetMap, NativeScores};
pub use proto::{cosine, Enrollment, Prototype, PrototypeMatcher};
