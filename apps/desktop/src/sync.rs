//! Background sync scheduler + thread (SPEC §4.3, §6, §7). The core
//! [`SyncEngine`](sinus_core::sync::SyncEngine) is complete and tested; this module
//! drives it off the UI thread:
//!
//! - **Auto-batch**: flush at 50 pending / 5 min elapsed / app quit.
//! - **Offline-first**: flush hourly or on explicit "Sync now" (metered — no
//!   threshold, no quit flush).
//! - **Offline-strict**: no engine is ever constructed (structural no-network,
//!   SPEC §4.3/§8).
//!
//! The when-to-flush decision and the quiet-hours check are **pure functions**
//! (unit-tested below); the thread loop is a thin driver that also wires
//! [`Backoff`] on failure, publishes pending count / sync health / quiet state to
//! the tray via [`SharedStatus`], and honors the manual "Sync now" request.

use std::path::PathBuf;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use chrono::Timelike;
use sinus_core::store::Store;
use sinus_core::sync::{Backoff, Mode, SyncConfig, SyncEngine};
use sinus_core::token::TokenStore;
use sinus_core::types::Source;

use crate::shared::{SharedStatus, SyncStatus};

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

/// Spawn the background sync thread. Runs regardless of audio/model availability so
/// previously-stored events still upload and quiet-hours state is always published.
pub fn spawn_sync(db_path: PathBuf, shared: SharedStatus) -> JoinHandle<()> {
    std::thread::spawn(move || {
        if let Err(e) = run_sync(db_path, shared) {
            eprintln!("sync: {e}");
        }
    })
}

/// How often the driver wakes to re-check the queue and publish status. Coarse to
/// keep wakeups coalesced (SPEC §9).
const TICK: Duration = Duration::from_secs(3);

fn read_mode(store: &Store) -> Mode {
    store
        .setting_get("mode")
        .ok()
        .flatten()
        .map(|s| match s.as_str() {
            "offline-first" => Mode::OfflineFirst,
            "offline-strict" => Mode::OfflineStrict,
            _ => Mode::AutoBatch,
        })
        .unwrap_or(Mode::AutoBatch)
}

fn sync_config(store: &Store) -> Option<SyncConfig> {
    let base_url = store
        .setting_get("server_url")
        .ok()
        .flatten()
        .filter(|s| !s.is_empty())?;
    let patient_id: i64 = store
        .setting_get("patient_id")
        .ok()
        .flatten()
        .and_then(|s| s.parse().ok())?;
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
        source: Source::current_desktop(),
        model_version,
        batch_size: 500,
    })
}

/// Pick the bearer-token store. Under `keyring` this is the OS keychain (SPEC §7/§8);
/// otherwise an empty in-memory store (a flush then fails with a token error, which
/// the driver treats as a retryable failure — dev builds without keychain access).
fn new_token_store() -> Box<dyn TokenStore> {
    #[cfg(feature = "keyring")]
    {
        Box::new(sinus_core::token::KeyringTokenStore::new(
            "SinusSentinel",
            "phr-api-token",
        ))
    }
    #[cfg(not(feature = "keyring"))]
    {
        Box::new(sinus_core::token::InMemoryTokenStore::new())
    }
}

/// Construct the engine for `mode`, or `None` if offline-strict or unconfigured.
fn build_engine(mode: Mode, store: &Store) -> Option<SyncEngine<Box<dyn TokenStore>>> {
    let cfg = sync_config(store)?;
    SyncEngine::for_mode(mode, cfg, new_token_store())
}

fn eval_quiet(store: &Store) -> bool {
    let get = |k: &str| {
        store
            .setting_get(k)
            .ok()
            .flatten()
            .and_then(|s| s.parse().ok())
    };
    match (get("quiet_start"), get("quiet_end")) {
        (Some(start), Some(end)) => in_quiet_hours(chrono::Local::now().hour(), start, end),
        _ => false,
    }
}

fn run_sync(db_path: PathBuf, shared: SharedStatus) -> Result<(), String> {
    let store = Store::open(&db_path).map_err(|e| e.to_string())?;
    let policy = FlushPolicy::default();
    let mut backoff = Backoff::default();
    let mut engine: Option<SyncEngine<Box<dyn TokenStore>>> = None;
    let mut engine_mode: Option<Mode> = None;
    let mut last_flush = Instant::now();
    // When a failure schedules a retry: no flush attempt is made before this.
    let mut retry_at: Option<Instant> = None;

    loop {
        shared.set_quiet(eval_quiet(&store));

        let pending_events = store.pending_count().unwrap_or(0) as usize;
        let has_tombstones = store
            .pending_tombstones(1)
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        let pending = pending_events + usize::from(has_tombstones);
        shared.set_pending(pending_events);

        let mode = read_mode(&store);
        // Mode switch: tear down / recreate the engine (offline-strict drops it
        // entirely, preserving the structural no-network property — SPEC §4.3).
        if engine_mode != Some(mode) {
            engine = build_engine(mode, &store);
            engine_mode = Some(mode);
            backoff.reset();
            retry_at = None;
            shared.set_sync(SyncStatus::Idle);
        }

        let manual = shared.take_sync_now();
        let quitting = shared.quitting();

        if should_flush(
            mode,
            pending,
            last_flush.elapsed(),
            manual,
            quitting,
            &policy,
        )
        .is_some()
        {
            let ready = retry_at.is_none_or(|t| Instant::now() >= t);
            if ready {
                if let Some(eng) = &engine {
                    shared.set_sync(SyncStatus::Syncing);
                    match eng.flush(&store) {
                        Ok(_) => {
                            backoff.reset();
                            retry_at = None;
                            last_flush = Instant::now();
                            shared.set_pending(store.pending_count().unwrap_or(0) as usize);
                            shared.set_sync(SyncStatus::Idle);
                        }
                        Err(e) => {
                            eprintln!("sync: flush failed: {e}");
                            // Wire the backoff cadence (SPEC §4.3): schedule the next
                            // attempt after a jittered delay; reset on success above.
                            retry_at = Some(Instant::now() + backoff.next_delay());
                            shared.set_sync(SyncStatus::Failed);
                        }
                    }
                }
            }
        }

        if quitting {
            break;
        }
        std::thread::sleep(TICK);
    }
    Ok(())
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
}
