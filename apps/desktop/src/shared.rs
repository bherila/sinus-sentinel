//! Cross-thread status shared between the capture thread, the background sync
//! thread, and the egui/tray UI thread (SPEC §6). Everything is lock-free atomics
//! so the UI thread never blocks on a worker (SPEC §9). Cloning a [`SharedStatus`]
//! shares the same underlying cells.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

use sinus_core::types::EventType;

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

/// Background-sync health, surfaced in the tray (SPEC §6 — "⚠ sync failing").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStatus {
    /// Idle / last flush succeeded.
    Idle,
    /// A flush is in progress.
    Syncing,
    /// The last flush attempt failed (retrying with backoff).
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TeachState {
    Idle,
    Armed,
    Recording,
    Saved,
    Failed,
}

impl TeachState {
    fn to_u8(self) -> u8 {
        match self {
            TeachState::Idle => 0,
            TeachState::Armed => 1,
            TeachState::Recording => 2,
            TeachState::Saved => 3,
            TeachState::Failed => 4,
        }
    }

    fn from_u8(value: u8) -> Self {
        match value {
            1 => TeachState::Armed,
            2 => TeachState::Recording,
            3 => TeachState::Saved,
            4 => TeachState::Failed,
            _ => TeachState::Idle,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TeachFeedback {
    pub state: TeachState,
    pub class: Option<EventType>,
    pub examples: usize,
    /// Similarity to the closest prior same-class example. Negative means this
    /// was the first example and no cross-sample validation was possible yet.
    pub similarity: f32,
    /// Same-class similarity minus the closest other-class similarity.
    pub separation: f32,
}

impl SyncStatus {
    fn to_u8(self) -> u8 {
        match self {
            SyncStatus::Idle => 0,
            SyncStatus::Syncing => 1,
            SyncStatus::Failed => 2,
        }
    }

    fn from_u8(v: u8) -> SyncStatus {
        match v {
            1 => SyncStatus::Syncing,
            2 => SyncStatus::Failed,
            _ => SyncStatus::Idle,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SyncStatus::Idle => "idle",
            SyncStatus::Syncing => "syncing…",
            SyncStatus::Failed => "sync failing",
        }
    }
}

/// Shared status cells. Clone to hand a view to another thread.
#[derive(Clone)]
pub struct SharedStatus {
    model: Arc<AtomicU8>,
    sync: Arc<AtomicU8>,
    pending: Arc<AtomicUsize>,
    quiet: Arc<AtomicBool>,
    sync_now: Arc<AtomicBool>,
    quitting: Arc<AtomicBool>,
    teach_request: Arc<AtomicU8>,
    teach_state: Arc<AtomicU8>,
    teach_class: Arc<AtomicU8>,
    teach_examples: Arc<AtomicUsize>,
    teach_similarity: Arc<AtomicU32>,
    teach_separation: Arc<AtomicU32>,
}

impl Default for SharedStatus {
    fn default() -> Self {
        SharedStatus {
            model: Arc::new(AtomicU8::new(ModelStatus::Heuristic.to_u8())),
            sync: Arc::new(AtomicU8::new(SyncStatus::Idle.to_u8())),
            pending: Arc::new(AtomicUsize::new(0)),
            quiet: Arc::new(AtomicBool::new(false)),
            sync_now: Arc::new(AtomicBool::new(false)),
            quitting: Arc::new(AtomicBool::new(false)),
            teach_request: Arc::new(AtomicU8::new(0)),
            teach_state: Arc::new(AtomicU8::new(TeachState::Idle.to_u8())),
            teach_class: Arc::new(AtomicU8::new(0)),
            teach_examples: Arc::new(AtomicUsize::new(0)),
            teach_similarity: Arc::new(AtomicU32::new((-1.0f32).to_bits())),
            teach_separation: Arc::new(AtomicU32::new(0.0f32.to_bits())),
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

    pub fn set_sync(&self, s: SyncStatus) {
        self.sync.store(s.to_u8(), Ordering::Relaxed);
    }

    pub fn sync(&self) -> SyncStatus {
        SyncStatus::from_u8(self.sync.load(Ordering::Relaxed))
    }

    pub fn set_pending(&self, n: usize) {
        self.pending.store(n, Ordering::Relaxed);
    }

    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::Relaxed)
    }

    /// Only the capture thread suppresses persistence during quiet hours, so this
    /// setter is unused without `live-audio` (the UI still reads the flag).
    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn set_quiet(&self, on: bool) {
        self.quiet.store(on, Ordering::Relaxed);
    }

    pub fn quiet(&self) -> bool {
        self.quiet.load(Ordering::Relaxed)
    }

    /// Tray "Sync now" sets this; the sync thread consumes it with [`Self::take_sync_now`].
    pub fn request_sync_now(&self) {
        self.sync_now.store(true, Ordering::Relaxed);
    }

    /// Atomically read-and-clear the manual-sync request.
    pub fn take_sync_now(&self) -> bool {
        self.sync_now.swap(false, Ordering::Relaxed)
    }

    /// Set on Quit so the sync thread can attempt a final flush (SPEC §4.3).
    pub fn set_quitting(&self, on: bool) {
        self.quitting.store(on, Ordering::Relaxed);
    }

    pub fn quitting(&self) -> bool {
        self.quitting.load(Ordering::Relaxed)
    }

    pub fn request_teach(&self, class: EventType) {
        let encoded = encode_event_type(class);
        self.teach_class.store(encoded, Ordering::Relaxed);
        self.teach_request.store(encoded, Ordering::Release);
        self.teach_state
            .store(TeachState::Armed.to_u8(), Ordering::Relaxed);
    }

    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn take_teach_request(&self) -> Option<EventType> {
        decode_event_type(self.teach_request.swap(0, Ordering::AcqRel))
    }

    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn set_teach_recording(&self, class: EventType) {
        self.teach_class
            .store(encode_event_type(class), Ordering::Relaxed);
        self.teach_state
            .store(TeachState::Recording.to_u8(), Ordering::Relaxed);
    }

    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn finish_teach(
        &self,
        class: EventType,
        examples: usize,
        similarity: f32,
        separation: f32,
    ) {
        self.teach_class
            .store(encode_event_type(class), Ordering::Relaxed);
        self.teach_examples.store(examples, Ordering::Relaxed);
        self.teach_similarity
            .store(similarity.to_bits(), Ordering::Relaxed);
        self.teach_separation
            .store(separation.to_bits(), Ordering::Relaxed);
        self.teach_state
            .store(TeachState::Saved.to_u8(), Ordering::Release);
    }

    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn fail_teach(&self, class: EventType) {
        self.teach_class
            .store(encode_event_type(class), Ordering::Relaxed);
        self.teach_state
            .store(TeachState::Failed.to_u8(), Ordering::Release);
    }

    pub fn teach_feedback(&self) -> TeachFeedback {
        TeachFeedback {
            state: TeachState::from_u8(self.teach_state.load(Ordering::Acquire)),
            class: decode_event_type(self.teach_class.load(Ordering::Relaxed)),
            examples: self.teach_examples.load(Ordering::Relaxed),
            similarity: f32::from_bits(self.teach_similarity.load(Ordering::Relaxed)),
            separation: f32::from_bits(self.teach_separation.load(Ordering::Relaxed)),
        }
    }
}

fn encode_event_type(class: EventType) -> u8 {
    EventType::ALL
        .iter()
        .position(|candidate| *candidate == class)
        .map(|index| index as u8 + 1)
        .unwrap_or(0)
}

fn decode_event_type(value: u8) -> Option<EventType> {
    value
        .checked_sub(1)
        .and_then(|index| EventType::ALL.get(index as usize).copied())
}
