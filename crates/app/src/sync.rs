//! Background PHR sync scheduler (SPEC §4.3, §6, §7). The core
//! [`SyncEngine`](sinus_core::sync::SyncEngine) is complete and tested; [`SyncDriver`]
//! drives it so every shell — the egui/tray desktop app and the SwiftUI
//! macOS/iOS clients — gets the same backoff, engine-rebuild and quiet-hours
//! behavior instead of a second implementation that can drift:
//!
//! - **Auto-batch**: flush at 50 pending / 5 min elapsed / app quit.
//! - **Offline-first**: flush hourly or on explicit "Sync now" (metered — no
//!   threshold, no quit flush).
//! - **Offline-strict**: no engine is ever constructed (structural no-network,
//!   SPEC §4.3/§8).
//!
//! The when-to-flush decision and the quiet-hours check are **pure functions**
//! (unit-tested below). [`SyncDriver::tick`] is a thin driver on top of them
//! that also wires [`Backoff`] on failure. It owns no thread and touches no
//! platform status bridge — each shell drives it from its own loop/timer and
//! maps [`TickOutput`] onto its own UI, exactly the seam that lets the desktop
//! tray and the SwiftUI clients share this scheduler instead of reimplementing
//! it.

use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Timelike;
use sinus_core::store::Store;
use sinus_core::sync::{Backoff, SyncConfig, SyncEngine, SyncOutcome};
use sinus_core::token::TokenStore;
use sinus_core::types::Source;

use crate::settings;

pub use sinus_core::sync::Mode;

/// When-to-flush thresholds (SPEC §4.3).
#[derive(Debug, Clone)]
pub struct FlushPolicy {
    /// Auto-batch flushes once this many events are pending.
    pub batch_threshold: usize,
    /// Auto-batch flushes at least this often while work is pending.
    pub auto_interval: Duration,
    /// Offline-first flushes on this schedule (else only on demand).
    pub offline_first_interval: Duration,
}

impl Default for FlushPolicy {
    fn default() -> Self {
        FlushPolicy {
            batch_threshold: 50,
            auto_interval: Duration::from_secs(5 * 60),
            offline_first_interval: Duration::from_secs(60 * 60),
        }
    }
}

/// Why a flush was decided (diagnostic; the flush itself is uniform).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushReason {
    /// Pending count crossed the auto-batch threshold.
    PendingThreshold,
    /// The mode's flush interval elapsed with work pending.
    IntervalElapsed,
    /// Explicit "Sync now".
    Manual,
    /// App is quitting (auto-batch only).
    Quit,
}

/// Pure when-to-flush decision (SPEC §4.3). `pending` is the total pending work
/// (events + tombstones). Offline-strict never flushes (no engine exists).
pub fn should_flush(
    mode: Mode,
    pending: usize,
    since_last_flush: Duration,
    manual_requested: bool,
    quitting: bool,
    policy: &FlushPolicy,
) -> Option<FlushReason> {
    // Offline-strict is structural: there is no engine and no network path.
    if mode == Mode::OfflineStrict {
        return None;
    }
    // Explicit user request wins in any network-capable mode (on demand).
    if manual_requested {
        return Some(FlushReason::Manual);
    }
    // Nothing to do — never wake the network for an empty queue.
    if pending == 0 {
        return None;
    }
    match mode {
        Mode::AutoBatch => {
            if quitting {
                Some(FlushReason::Quit)
            } else if pending >= policy.batch_threshold {
                Some(FlushReason::PendingThreshold)
            } else if since_last_flush >= policy.auto_interval {
                Some(FlushReason::IntervalElapsed)
            } else {
                None
            }
        }
        // Offline-first: scheduled only. Metered connections → no threshold flush
        // and no flush-on-quit; the queue simply waits for the next schedule tick
        // or an explicit "Sync now".
        Mode::OfflineFirst => {
            if since_last_flush >= policy.offline_first_interval {
                Some(FlushReason::IntervalElapsed)
            } else {
                None
            }
        }
        Mode::OfflineStrict => None,
    }
}

/// Pure quiet-hours check (SPEC §6): is `hour` (0–23, local) within `[start, end)`,
/// wrapping past midnight when `start > end`? `start == end` disables it.
pub fn in_quiet_hours(hour: u32, start: u32, end: u32) -> bool {
    if start == end {
        return false;
    }
    if start < end {
        hour >= start && hour < end
    } else {
        // Wraps midnight, e.g. 23:00–07:00.
        hour >= start || hour < end
    }
}

/// The settings the engine is built from, so a change to either rebuilds it.
/// Without this, filling in the server URL or patient id for the first time
/// leaves sync dead until the app is relaunched or the mode toggled.
fn sync_config_key(store: &Store) -> (String, Option<i64>) {
    (settings::server_url(store), settings::patient_id(store))
}

/// The engine configuration for `source`, or `None` if unconfigured. `source`
/// is a parameter rather than an assumed desktop host: an iPhone must not
/// upload as `desktop-mac`.
pub fn sync_config(store: &Store, source: Source) -> Option<SyncConfig> {
    let base_url = settings::server_url(store);
    if base_url.is_empty() {
        return None;
    }
    let patient_id = settings::patient_id(store)?;
    let device_id = store
        .setting_get("device_id")
        .ok()
        .flatten()
        .unwrap_or_else(|| "unknown".to_string());
    let model_version = store
        .setting_get("model_version")
        .ok()
        .flatten()
        .unwrap_or_else(|| "yamnet+proto@0".to_string());
    Some(SyncConfig {
        base_url,
        patient_id,
        device_id,
        source,
        model_version,
        batch_size: 500,
    })
}

/// Construct the engine for `mode`, or `None` if offline-strict or unconfigured.
/// Private to the driver: every caller goes through [`SyncDriver::tick`], which
/// is what keeps the rebuild condition and the backoff reset next to each other.
fn build_engine(
    mode: Mode,
    store: &Store,
    source: Source,
    token: Arc<dyn TokenStore>,
) -> Option<SyncEngine<Arc<dyn TokenStore>>> {
    let cfg = sync_config(store, source)?;
    SyncEngine::for_mode(mode, cfg, token)
}

/// Whether the current local time falls in the configured quiet-hours window
/// (SPEC §6), or `false` if none is configured.
pub fn in_quiet_hours_now(store: &Store) -> bool {
    match settings::quiet_hours(store) {
        Some((start, end)) => in_quiet_hours(chrono::Local::now().hour(), start, end),
        None => false,
    }
}

fn until_next_local_hour() -> Duration {
    let now = chrono::Local::now();
    let elapsed = now.minute() as u64 * 60 + now.second() as u64;
    Duration::from_secs((60 * 60 - elapsed).max(1))
}

fn next_driver_wait(
    mode: Mode,
    pending: usize,
    last_flush: Instant,
    retry_at: Option<Instant>,
    policy: &FlushPolicy,
) -> Duration {
    let now = Instant::now();
    let mut wait = until_next_local_hour();
    if let Some(retry_at) = retry_at {
        wait = wait.min(retry_at.saturating_duration_since(now));
    } else if pending > 0 {
        let interval = match mode {
            Mode::AutoBatch => Some(policy.auto_interval),
            Mode::OfflineFirst => Some(policy.offline_first_interval),
            Mode::OfflineStrict => None,
        };
        if let Some(interval) = interval {
            wait = wait.min(interval.saturating_sub(last_flush.elapsed()));
        }
    }
    wait.max(Duration::from_millis(1))
}

/// Sync health, as reported by [`TickOutput::state`]. Named `SyncState` rather
/// than reusing the desktop's `SyncStatus` so this crate has no dependency on
/// any shell's UI types — each shell maps this onto its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncState {
    /// Idle / last flush succeeded.
    #[default]
    Idle,
    /// A flush is in progress.
    Syncing,
    /// The last flush attempt failed (retrying with backoff).
    Failed,
}

/// What the caller observed since the previous tick — the driver has no
/// thread and no status bridge of its own, so this is how "Sync now" and
/// "app is quitting" reach it.
#[derive(Debug, Clone, Copy, Default)]
pub struct TickInput {
    pub manual: bool,
    pub quitting: bool,
}

/// The result of one [`SyncDriver::tick`]. The caller maps this onto its own
/// status display and decides how long to wait before ticking again.
#[derive(Debug, Clone)]
pub struct TickOutput {
    pub mode: Mode,
    /// Badge count — events only, which is what "pending" means to a user.
    pub pending_events: usize,
    /// The flush gate — events + tombstones + flags + enrollments.
    pub pending_work: usize,
    pub quiet: bool,
    pub state: SyncState,
    pub flushed: Option<SyncOutcome>,
    /// The failure text. Today this only reaches stderr; every shell should be
    /// able to show it.
    pub error: Option<String>,
    /// How long the caller should wait before ticking again, absent a signal.
    /// Zero means "tick again now" — a flush just succeeded and there may be
    /// more pending work to drain immediately, mirroring the old loop's
    /// `if flushed { continue }`.
    pub next_wait: Duration,
    /// The caller asked to quit and the final flush (if the mode allows one)
    /// has been made — stop looping.
    pub done: bool,
}

impl Default for TickOutput {
    // Hand-written rather than derived: `Mode` has no `Default` of its own
    // (deriving one would mean touching sinus-core for this crate's sake), and
    // `AutoBatch` is the same safe fallback `settings::mode` already uses for
    // an absent/corrupt row.
    fn default() -> Self {
        TickOutput {
            mode: Mode::AutoBatch,
            pending_events: 0,
            pending_work: 0,
            quiet: false,
            state: SyncState::default(),
            flushed: None,
            error: None,
            next_wait: Duration::ZERO,
            done: false,
        }
    }
}

/// Drives [`SyncEngine`] off pure policy decisions. Owns no thread — the
/// caller supplies the store and a [`TickInput`] each tick and is responsible
/// for the sleep/wake loop around it (a background thread for the desktop
/// tray, a timer/task for SwiftUI).
pub struct SyncDriver {
    source: Source,
    policy: FlushPolicy,
    backoff: Backoff,
    engine: Option<SyncEngine<Arc<dyn TokenStore>>>,
    engine_mode: Option<Mode>,
    engine_config: (String, Option<i64>),
    /// Last known health, carried across ticks so a quiet tick (nothing
    /// scheduled) reports the same state as the last transition rather than
    /// resetting to `Idle` — unlike the old thread loop, `TickOutput` is a
    /// fresh value every call, not a shared cell.
    state: SyncState,
    last_flush: Instant,
    /// When a failure schedules a retry: no flush attempt is made before this.
    retry_at: Option<Instant>,
    token: Arc<dyn TokenStore>,
}

impl SyncDriver {
    pub fn new(source: Source, policy: FlushPolicy, token: Arc<dyn TokenStore>) -> Self {
        SyncDriver {
            source,
            policy,
            backoff: Backoff::default(),
            engine: None,
            engine_mode: None,
            engine_config: (String::new(), None),
            state: SyncState::default(),
            last_flush: Instant::now(),
            retry_at: None,
            token,
        }
    }

    /// One iteration of the scheduler: rebuild the engine if config changed,
    /// decide whether to flush, attempt it if so, and report what happened
    /// plus how long to wait before the next tick.
    pub fn tick(&mut self, store: &mut Store, input: TickInput) -> TickOutput {
        let quiet = in_quiet_hours_now(store);

        let pending_events = store.pending_count().unwrap_or(0) as usize;
        // Everything awaiting the server, not just events: a flag or a teach
        // take made while offline must be able to schedule its own retry. The
        // badge still shows only the event count, which is what "pending"
        // means to a user.
        let pending = store.pending_work_count().unwrap_or(0) as usize;

        let mode = settings::mode(store);
        // Rebuild on a mode switch (offline-strict drops the engine entirely,
        // preserving the structural no-network property — SPEC §4.3) *and* when
        // the server URL / patient id changes, so filling those in for the first
        // time starts syncing without a relaunch.
        let config_key = sync_config_key(store);
        if self.engine_mode != Some(mode) || self.engine_config != config_key {
            self.engine = build_engine(mode, store, self.source, Arc::clone(&self.token));
            self.engine_mode = Some(mode);
            self.engine_config = config_key;
            self.backoff.reset();
            self.retry_at = None;
            self.state = SyncState::Idle;
        }

        let mut output = TickOutput {
            mode,
            pending_events,
            pending_work: pending,
            quiet,
            state: self.state,
            ..TickOutput::default()
        };

        let scheduled = should_flush(
            mode,
            pending,
            self.last_flush.elapsed(),
            input.manual,
            input.quitting,
            &self.policy,
        )
        .is_some();
        let retry_due = self
            .retry_at
            .is_some_and(|deadline| Instant::now() >= deadline);
        let mut flushed = false;
        if scheduled || retry_due {
            let ready = self.retry_at.is_none_or(|t| Instant::now() >= t);
            if ready {
                if let Some(eng) = &self.engine {
                    self.state = SyncState::Syncing;
                    output.state = SyncState::Syncing;
                    match eng.flush(store) {
                        Ok(outcome) => {
                            // Anything pulled down has to reach the caller's
                            // capture path, or a machine that just inherited the
                            // user's settings and training keeps detecting as
                            // though it had neither.
                            self.backoff.reset();
                            self.retry_at = None;
                            self.last_flush = Instant::now();
                            output.pending_events = store.pending_count().unwrap_or(0) as usize;
                            self.state = SyncState::Idle;
                            output.state = SyncState::Idle;
                            output.flushed = Some(outcome);
                            flushed = true;
                        }
                        Err(e) => {
                            output.error = Some(e.to_string());
                            // Wire the backoff cadence (SPEC §4.3): schedule the next
                            // attempt after a jittered delay; reset on success above.
                            self.retry_at = Some(Instant::now() + self.backoff.next_delay());
                            self.state = SyncState::Failed;
                            output.state = SyncState::Failed;
                        }
                    }
                }
            }
        }

        if input.quitting {
            output.done = true;
            return output;
        }
        if flushed {
            // Tick again now — there may be more pending work to drain.
            output.next_wait = Duration::ZERO;
            return output;
        }
        output.next_wait =
            next_driver_wait(mode, pending, self.last_flush, self.retry_at, &self.policy);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> FlushPolicy {
        FlushPolicy::default()
    }

    #[test]
    fn offline_strict_never_flushes() {
        // Even an explicit request and a full queue must not flush (no engine).
        assert_eq!(
            should_flush(
                Mode::OfflineStrict,
                999,
                Duration::from_secs(0),
                true,
                true,
                &policy()
            ),
            None
        );
    }

    #[test]
    fn auto_batch_flush_triggers() {
        let p = policy();
        // Below threshold, before interval, no request → hold.
        assert_eq!(
            should_flush(Mode::AutoBatch, 1, Duration::from_secs(1), false, false, &p),
            None
        );
        // 50 pending → threshold.
        assert_eq!(
            should_flush(
                Mode::AutoBatch,
                50,
                Duration::from_secs(1),
                false,
                false,
                &p
            ),
            Some(FlushReason::PendingThreshold)
        );
        // 5 min elapsed with work → interval.
        assert_eq!(
            should_flush(
                Mode::AutoBatch,
                1,
                Duration::from_secs(5 * 60),
                false,
                false,
                &p
            ),
            Some(FlushReason::IntervalElapsed)
        );
        // Quitting with work → quit flush.
        assert_eq!(
            should_flush(Mode::AutoBatch, 1, Duration::from_secs(1), false, true, &p),
            Some(FlushReason::Quit)
        );
        // Manual beats everything.
        assert_eq!(
            should_flush(Mode::AutoBatch, 0, Duration::from_secs(0), true, false, &p),
            Some(FlushReason::Manual)
        );
        // Empty queue, not manual → nothing (never wake the network for nothing).
        assert_eq!(
            should_flush(
                Mode::AutoBatch,
                0,
                Duration::from_secs(9999),
                false,
                true,
                &p
            ),
            None
        );
    }

    #[test]
    fn offline_first_is_scheduled_or_on_demand_only() {
        let p = policy();
        // A full queue does NOT trigger a threshold flush (metered).
        assert_eq!(
            should_flush(
                Mode::OfflineFirst,
                999,
                Duration::from_secs(1),
                false,
                false,
                &p
            ),
            None
        );
        // Quitting does NOT flush (metered — unlike auto-batch).
        assert_eq!(
            should_flush(
                Mode::OfflineFirst,
                999,
                Duration::from_secs(1),
                false,
                true,
                &p
            ),
            None
        );
        // Hourly schedule with work → interval flush.
        assert_eq!(
            should_flush(
                Mode::OfflineFirst,
                1,
                Duration::from_secs(60 * 60),
                false,
                false,
                &p
            ),
            Some(FlushReason::IntervalElapsed)
        );
        // On demand always works.
        assert_eq!(
            should_flush(
                Mode::OfflineFirst,
                1,
                Duration::from_secs(0),
                true,
                false,
                &p
            ),
            Some(FlushReason::Manual)
        );
    }

    #[test]
    fn quiet_hours_windows() {
        // Daytime window 22–23 (not wrapping).
        assert!(in_quiet_hours(22, 22, 23));
        assert!(!in_quiet_hours(23, 22, 23));
        assert!(!in_quiet_hours(21, 22, 23));
        // Overnight window 23:00–07:00 (wraps midnight).
        assert!(in_quiet_hours(23, 23, 7));
        assert!(in_quiet_hours(0, 23, 7));
        assert!(in_quiet_hours(6, 23, 7));
        assert!(!in_quiet_hours(7, 23, 7));
        assert!(!in_quiet_hours(12, 23, 7));
        // start == end disables.
        assert!(!in_quiet_hours(5, 0, 0));
    }

    // --- SyncDriver integration test -----------------------------------

    use sinus_core::token::InMemoryTokenStore;
    use sinus_core::types::{Event, EventType};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn sample_event(uuid: &str) -> Event {
        Event {
            uuid: uuid.to_string(),
            event_type: EventType::Cough,
            occurred_at: chrono::Utc::now(),
            tz_offset_min: 0,
            duration_ms: 500,
            confidence: 0.6,
            burst_count: 1,
            peak_dbfs: Some(-9.5),
            mean_dbfs: Some(-21.0),
            noise_floor_dbfs: Some(-50.0),
            model_version: "test@0".into(),
            source: Source::DesktopMac,
            device_id: "dev".into(),
            uploaded_at: None,
            deleted: false,
            false_positive_at: None,
            corrected_to: None,
            corrected_at: None,
            reject_count: 0,
            rejected_at: None,
        }
    }

    /// A server that always answers 500, so every flush it sees fails. Counts
    /// requests so the retry-gate test can assert a second immediate tick does
    /// not attempt another flush.
    struct FailingServer {
        url: String,
        requests: Arc<AtomicUsize>,
        shutdown: Arc<std::sync::atomic::AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl FailingServer {
        fn start() -> Self {
            let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
            let url = format!("http://{}", server.server_addr());
            let requests = Arc::new(AtomicUsize::new(0));
            let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));

            let counter = Arc::clone(&requests);
            let stop = Arc::clone(&shutdown);
            let handle = std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let Ok(Some(req)) = server.recv_timeout(std::time::Duration::from_millis(50))
                    else {
                        continue;
                    };
                    counter.fetch_add(1, Ordering::Relaxed);
                    let resp = tiny_http::Response::from_string("server error".to_string())
                        .with_status_code(500);
                    req.respond(resp).ok();
                }
            });

            FailingServer {
                url,
                requests,
                shutdown,
                handle: Some(handle),
            }
        }

        fn request_count(&self) -> usize {
            self.requests.load(Ordering::Relaxed)
        }
    }

    impl Drop for FailingServer {
        fn drop(&mut self) {
            self.shutdown.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                handle.join().ok();
            }
        }
    }

    #[test]
    fn a_failed_flush_backs_off_and_does_not_retry_immediately() {
        let server = FailingServer::start();

        let mut store = Store::open_in_memory().unwrap();
        settings::set_server_url(&store, &server.url).unwrap();
        settings::set_patient_id(&store, Some(7)).unwrap();
        settings::set_mode(&store, Mode::AutoBatch).unwrap();
        store.insert_event(&sample_event("a")).unwrap();

        let token: Arc<dyn TokenStore> = Arc::new(InMemoryTokenStore::with_token("secret"));
        let mut driver = SyncDriver::new(Source::DesktopMac, FlushPolicy::default(), token);

        // A single pending event under auto-batch defaults (threshold 50,
        // interval 5 min) would not schedule itself; force the attempt the way
        // a user's "Sync now" would.
        let first = driver.tick(
            &mut store,
            TickInput {
                manual: true,
                quitting: false,
            },
        );
        assert_eq!(first.state, SyncState::Failed);
        assert!(first.error.is_some(), "the failure text must be reported");
        assert!(
            first.next_wait > Duration::ZERO,
            "a failure must schedule a non-zero backoff wait"
        );
        assert_eq!(
            server.request_count(),
            1,
            "the first tick must attempt exactly one flush"
        );

        // Immediately again: the retry deadline has not passed, so this must
        // not touch the network at all.
        let second = driver.tick(&mut store, TickInput::default());
        assert_eq!(second.state, SyncState::Failed);
        assert_eq!(
            server.request_count(),
            1,
            "a tick before the retry deadline must not attempt another flush"
        );
    }
}
