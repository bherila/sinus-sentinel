//! Cross-thread status shared between the capture thread and the egui/tray UI
//! thread (SPEC §6). Everything is lock-free atomics so the UI thread never blocks
//! on a worker (SPEC §9). Cloning a [`SharedStatus`] shares the same cells.
//!
//! Issue #2 extends this bus with sync health, pending count, quiet-hours, and the
//! manual "Sync now" request.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// Which backbone the capture thread ended up running (SPEC §4 stage ③). Surfaced
/// in the tray so a missing/failed ONNX model is visible rather than silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    /// The deterministic, model-free `BandHeuristicEmbedder` (default build).
    Heuristic,
    /// The real ort-backed YAMNet model loaded successfully.
    Onnx,
    /// An ONNX build was requested but the model could not be loaded; capture fell
    /// back to the heuristic backbone (SPEC §4 — fail-soft).
    Missing,
}

impl ModelStatus {
    fn to_u8(self) -> u8 {
        match self {
            ModelStatus::Heuristic => 0,
            ModelStatus::Onnx => 1,
            ModelStatus::Missing => 2,
        }
    }

    fn from_u8(v: u8) -> ModelStatus {
        match v {
            1 => ModelStatus::Onnx,
            2 => ModelStatus::Missing,
            _ => ModelStatus::Heuristic,
        }
    }

    /// Short label for the tray/status line.
    pub fn label(self) -> &'static str {
        match self {
            ModelStatus::Heuristic => "heuristic",
            ModelStatus::Onnx => "yamnet-onnx",
            ModelStatus::Missing => "model missing",
        }
    }
}

/// Shared status cells. Clone to hand a view to another thread.
#[derive(Clone)]
pub struct SharedStatus {
    model: Arc<AtomicU8>,
}

impl Default for SharedStatus {
    fn default() -> Self {
        SharedStatus {
            model: Arc::new(AtomicU8::new(ModelStatus::Heuristic.to_u8())),
        }
    }
}

impl SharedStatus {
    /// Only the capture thread publishes the model status, so this is unused in a
    /// build without `live-audio` (the UI still reads the default Heuristic state).
    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn set_model(&self, s: ModelStatus) {
        self.model.store(s.to_u8(), Ordering::Relaxed);
    }

    pub fn model(&self) -> ModelStatus {
        ModelStatus::from_u8(self.model.load(Ordering::Relaxed))
    }
}
