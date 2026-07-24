//! Cross-thread status shared between the capture thread, the background sync
//! thread, and the egui/tray UI thread (SPEC §6). Hot status reads are lock-free;
//! the few mutexes only hold wake handles/generations and are never held while
//! doing I/O or inference. Cloning a [`SharedStatus`] shares the same state.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::Thread;
use std::time::Duration;

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
    low_power: Arc<AtomicBool>,
    sync_now: Arc<AtomicBool>,
    quitting: Arc<AtomicBool>,
    teach_request: Arc<AtomicU8>,
    teach_state: Arc<AtomicU8>,
    teach_class: Arc<AtomicU8>,
    teach_examples: Arc<AtomicUsize>,
    teach_similarity: Arc<AtomicU32>,
    teach_separation: Arc<AtomicU32>,
    enrollment_reload: Arc<AtomicBool>,
    settings_reload: Arc<AtomicBool>,
    analyzing: Arc<AtomicBool>,
    last_heard_ms: Arc<AtomicI64>,
    pause_until_ms: Arc<AtomicI64>,
    capture_thread: Arc<Mutex<Option<Thread>>>,
    sync_signal: Arc<(Mutex<u64>, Condvar)>,
    repaint_context: Arc<Mutex<Option<eframe::egui::Context>>>,
    history_generation: Arc<AtomicUsize>,
}

impl Default for SharedStatus {
    fn default() -> Self {
        SharedStatus {
            model: Arc::new(AtomicU8::new(ModelStatus::Heuristic.to_u8())),
            sync: Arc::new(AtomicU8::new(SyncStatus::Idle.to_u8())),
            pending: Arc::new(AtomicUsize::new(0)),
            quiet: Arc::new(AtomicBool::new(false)),
            low_power: Arc::new(AtomicBool::new(false)),
            sync_now: Arc::new(AtomicBool::new(false)),
            quitting: Arc::new(AtomicBool::new(false)),
            teach_request: Arc::new(AtomicU8::new(0)),
            teach_state: Arc::new(AtomicU8::new(TeachState::Idle.to_u8())),
            teach_class: Arc::new(AtomicU8::new(0)),
            teach_examples: Arc::new(AtomicUsize::new(0)),
            teach_similarity: Arc::new(AtomicU32::new((-1.0f32).to_bits())),
            teach_separation: Arc::new(AtomicU32::new(0.0f32.to_bits())),
            enrollment_reload: Arc::new(AtomicBool::new(false)),
            settings_reload: Arc::new(AtomicBool::new(false)),
            analyzing: Arc::new(AtomicBool::new(false)),
            last_heard_ms: Arc::new(AtomicI64::new(0)),
            pause_until_ms: Arc::new(AtomicI64::new(0)),
            capture_thread: Arc::new(Mutex::new(None)),
            sync_signal: Arc::new((Mutex::new(0), Condvar::new())),
            repaint_context: Arc::new(Mutex::new(None)),
            history_generation: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl SharedStatus {
    fn notify_ui(&self) {
        if let Ok(context) = self.repaint_context.lock() {
            if let Some(context) = context.as_ref() {
                context.request_repaint();
            }
        }
    }

    pub fn attach_repaint_context(&self, context: eframe::egui::Context) {
        if let Ok(mut slot) = self.repaint_context.lock() {
            *slot = Some(context);
        }
    }

    fn wake_capture(&self) {
        if let Ok(thread) = self.capture_thread.lock() {
            if let Some(thread) = thread.as_ref() {
                thread.unpark();
            }
        }
    }

    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn register_capture_thread(&self) {
        if let Ok(mut slot) = self.capture_thread.lock() {
            *slot = Some(std::thread::current());
        }
    }

    pub fn notify_sync(&self) {
        let (generation, wake) = &*self.sync_signal;
        if let Ok(mut generation) = generation.lock() {
            *generation = generation.wrapping_add(1);
            wake.notify_one();
        }
    }

    pub fn sync_generation(&self) -> u64 {
        self.sync_signal
            .0
            .lock()
            .map_or(0, |generation| *generation)
    }

    pub fn wait_for_sync_signal(&self, observed: u64, timeout: Duration) -> u64 {
        let (generation, wake) = &*self.sync_signal;
        let Ok(generation) = generation.lock() else {
            return observed;
        };
        if *generation != observed {
            return *generation;
        }
        wake.wait_timeout_while(generation, timeout, |current| *current == observed)
            .map_or(observed, |(current, _)| *current)
    }

    /// Only the capture thread publishes the model status, so this is unused in a
    /// build without `live-audio` (the UI still reads the default Heuristic state).
    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn set_model(&self, s: ModelStatus) {
        if self.model.swap(s.to_u8(), Ordering::Relaxed) != s.to_u8() {
            self.notify_ui();
        }
    }

    pub fn model(&self) -> ModelStatus {
        ModelStatus::from_u8(self.model.load(Ordering::Relaxed))
    }

    pub fn set_sync(&self, s: SyncStatus) {
        if self.sync.swap(s.to_u8(), Ordering::Relaxed) != s.to_u8() {
            self.notify_ui();
        }
    }

    pub fn sync(&self) -> SyncStatus {
        SyncStatus::from_u8(self.sync.load(Ordering::Relaxed))
    }

    pub fn set_pending(&self, n: usize) {
        if self.pending.swap(n, Ordering::Relaxed) != n {
            self.notify_ui();
        }
    }

    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::Relaxed)
    }

    /// Only the capture thread suppresses persistence during quiet hours, so this
    /// setter is unused without `live-audio` (the UI still reads the flag).
    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn set_quiet(&self, on: bool) {
        if self.quiet.swap(on, Ordering::Relaxed) != on {
            self.wake_capture();
            self.notify_ui();
        }
    }

    pub fn quiet(&self) -> bool {
        self.quiet.load(Ordering::Relaxed)
    }

    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn set_low_power(&self, on: bool) {
        if self.low_power.swap(on, Ordering::Relaxed) != on {
            self.wake_capture();
            self.notify_ui();
        }
    }

    pub fn low_power(&self) -> bool {
        self.low_power.load(Ordering::Relaxed)
    }

    pub fn pause_until(&self, until: chrono::DateTime<chrono::Utc>) {
        self.pause_until_ms
            .store(until.timestamp_millis().max(1), Ordering::Release);
        self.wake_capture();
        self.notify_ui();
    }

    pub fn pause_indefinitely(&self) {
        self.pause_until_ms.store(i64::MAX, Ordering::Release);
        self.wake_capture();
        self.notify_ui();
    }

    pub fn resume_capture(&self) {
        self.pause_until_ms.store(0, Ordering::Release);
        self.wake_capture();
        self.notify_ui();
    }

    pub fn capture_paused(&self, now: chrono::DateTime<chrono::Utc>) -> bool {
        let until = self.pause_until_ms.load(Ordering::Acquire);
        if until == i64::MAX {
            return true;
        }
        if until > now.timestamp_millis() {
            return true;
        }
        if until != 0 {
            let _ =
                self.pause_until_ms
                    .compare_exchange(until, 0, Ordering::AcqRel, Ordering::Relaxed);
            self.notify_ui();
        }
        false
    }

    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn capture_suspended(&self, now: chrono::DateTime<chrono::Utc>) -> bool {
        self.quiet() || self.low_power() || self.capture_paused(now)
    }

    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn suspension_wait(&self, now: chrono::DateTime<chrono::Utc>) -> Duration {
        if self.low_power() {
            return Duration::from_secs(60);
        }
        if self.quiet() {
            return Duration::from_secs(60 * 60);
        }
        match self.pause_until_ms.load(Ordering::Acquire) {
            i64::MAX => Duration::from_secs(60 * 60),
            0 => Duration::ZERO,
            until => Duration::from_millis(
                until
                    .saturating_sub(now.timestamp_millis())
                    .max(1)
                    .try_into()
                    .unwrap_or(u64::MAX),
            ),
        }
    }

    /// Tray "Sync now" sets this; the sync thread consumes it with [`Self::take_sync_now`].
    pub fn request_sync_now(&self) {
        self.sync_now.store(true, Ordering::Relaxed);
        self.notify_sync();
    }

    /// Atomically read-and-clear the manual-sync request.
    pub fn take_sync_now(&self) -> bool {
        self.sync_now.swap(false, Ordering::Relaxed)
    }

    /// Set on Quit so the sync thread can attempt a final flush (SPEC §4.3).
    #[cfg_attr(test, allow(dead_code))]
    pub fn set_quitting(&self, on: bool) {
        self.quitting.store(on, Ordering::Relaxed);
        self.wake_capture();
        self.notify_sync();
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
        self.wake_capture();
        self.notify_ui();
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
        self.notify_ui();
    }

    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn fail_teach(&self, class: EventType) {
        self.teach_class
            .store(encode_event_type(class), Ordering::Relaxed);
        self.teach_state
            .store(TeachState::Failed.to_u8(), Ordering::Release);
        self.notify_ui();
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

    /// Tell the capture worker that enrollment rows changed outside its thread.
    pub fn request_enrollment_reload(&self) {
        self.enrollment_reload.store(true, Ordering::Release);
    }

    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn take_enrollment_reload(&self) -> bool {
        self.enrollment_reload.swap(false, Ordering::AcqRel)
    }

    /// Tell the capture worker its detection settings changed — from the
    /// slider here, or from a value pulled off the PHR.
    ///
    /// Without this, sensitivity is only read when the pipeline is built, so a
    /// change would not take effect until the app restarted.
    pub fn request_settings_reload(&self) {
        self.settings_reload.store(true, Ordering::Release);
        self.wake_capture();
    }

    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn take_settings_reload(&self) -> bool {
        self.settings_reload.swap(false, Ordering::AcqRel)
    }

    /// Published by the capture thread while the energy gate is open — the app
    /// has heard something and is classifying it.
    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn set_analyzing(&self, on: bool) {
        if self.analyzing.swap(on, Ordering::Relaxed) != on {
            self.notify_ui();
        }
    }

    pub fn analyzing(&self) -> bool {
        self.analyzing.load(Ordering::Relaxed)
    }

    /// Stamp the moment a sound was first picked up (gate closed → open).
    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn note_heard(&self, at: chrono::DateTime<chrono::Utc>) {
        self.last_heard_ms
            .store(at.timestamp_millis(), Ordering::Relaxed);
        self.notify_ui();
    }

    /// When the app last started analyzing a sound, if ever.
    pub fn last_heard(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        let millis = self.last_heard_ms.load(Ordering::Relaxed);
        (millis > 0)
            .then(|| chrono::DateTime::from_timestamp_millis(millis))
            .flatten()
    }

    pub fn reset_teach_feedback(&self) {
        self.teach_request.store(0, Ordering::Relaxed);
        self.teach_state
            .store(TeachState::Idle.to_u8(), Ordering::Release);
        self.teach_class.store(0, Ordering::Relaxed);
        self.teach_examples.store(0, Ordering::Relaxed);
        self.teach_similarity
            .store((-1.0f32).to_bits(), Ordering::Relaxed);
        self.teach_separation
            .store(0.0f32.to_bits(), Ordering::Relaxed);
        self.notify_ui();
    }

    pub fn history_generation(&self) -> usize {
        self.history_generation.load(Ordering::Acquire)
    }

    pub fn notify_history_changed(&self) {
        self.history_generation.fetch_add(1, Ordering::AcqRel);
        self.notify_ui();
    }

    #[cfg_attr(not(feature = "live-audio"), allow(dead_code))]
    pub fn notify_event_persisted(&self) {
        self.notify_history_changed();
        self.notify_sync();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timed_and_indefinite_pause_are_shared_with_capture() {
        let shared = SharedStatus::default();
        let now = chrono::Utc::now();
        shared.pause_until(now + chrono::Duration::minutes(15));
        assert!(shared.capture_paused(now));
        assert!(!shared.capture_paused(now + chrono::Duration::minutes(16)));

        shared.pause_indefinitely();
        assert!(shared.capture_paused(now + chrono::Duration::days(1)));
        shared.resume_capture();
        assert!(!shared.capture_paused(now));
    }

    #[test]
    fn quiet_hours_suspend_capture() {
        let shared = SharedStatus::default();
        shared.set_quiet(true);
        assert!(shared.capture_suspended(chrono::Utc::now()));
        shared.set_quiet(false);
        assert!(!shared.capture_suspended(chrono::Utc::now()));
    }

    #[test]
    fn os_low_power_mode_suspends_capture() {
        let shared = SharedStatus::default();
        shared.set_low_power(true);
        assert!(shared.capture_suspended(chrono::Utc::now()));
        shared.set_low_power(false);
        assert!(!shared.capture_suspended(chrono::Utc::now()));
    }

    #[test]
    fn sync_wait_is_generation_driven() {
        let shared = SharedStatus::default();
        let observed = shared.sync_generation();
        shared.notify_sync();
        let next = shared.wait_for_sync_signal(observed, Duration::from_secs(1));
        assert_ne!(next, observed);
    }
}
