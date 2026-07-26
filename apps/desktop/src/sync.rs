//! Background sync thread (SPEC §4.3, §6, §7). The scheduling policy — when to
//! flush, quiet hours, backoff, engine rebuilds — lives in
//! [`sinus_app::sync::SyncDriver`] so the SwiftUI macOS/iOS clients share it
//! rather than reimplementing it. This module is just the desktop-specific
//! wiring: a background thread that ticks the driver, the OS-keychain token
//! store, and publishing the result to the tray via [`SharedStatus`].

use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use sinus_app::sync::{FlushPolicy, SyncDriver, SyncState, TickInput};
use sinus_core::store::Store;
use sinus_core::token::TokenStore;
use sinus_core::types::Source;

use crate::shared::{SharedStatus, SyncStatus};

/// Spawn the background sync thread. Runs regardless of audio/model availability so
/// previously-stored events still upload and quiet-hours state is always published.
pub fn spawn_sync(db_path: PathBuf, shared: SharedStatus) -> JoinHandle<()> {
    std::thread::spawn(move || {
        if let Err(e) = run_sync(db_path, shared) {
            eprintln!("sync: {e}");
        }
    })
}

/// Pick the bearer-token store. Under `keyring` this is the OS keychain (SPEC §7/§8);
/// otherwise an empty in-memory store (a flush then fails with a token error, which
/// the driver treats as a retryable failure — dev builds without keychain access).
///
/// `Arc` rather than `Box`: the driver rebuilds its engine whenever mode or server
/// config changes, and cloning an `Arc` shares the keychain-backed store's cache
/// instead of re-constructing it on every rebuild.
fn new_token_store() -> Arc<dyn TokenStore> {
    #[cfg(feature = "keyring")]
    {
        Arc::new(sinus_core::token::KeyringTokenStore::new(
            "SinusSentinel",
            "phr-api-token",
        ))
    }
    #[cfg(not(feature = "keyring"))]
    {
        Arc::new(sinus_core::token::InMemoryTokenStore::new())
    }
}

fn map_sync_state(state: SyncState) -> SyncStatus {
    match state {
        SyncState::Idle => SyncStatus::Idle,
        SyncState::Syncing => SyncStatus::Syncing,
        SyncState::Failed => SyncStatus::Failed,
    }
}

fn run_sync(db_path: PathBuf, shared: SharedStatus) -> Result<(), String> {
    let mut store = Store::open(&db_path).map_err(|e| e.to_string())?;
    let mut driver = SyncDriver::new(
        Source::current_desktop(),
        FlushPolicy::default(),
        new_token_store(),
    );
    let mut observed_signal = shared.sync_generation();

    loop {
        let input = TickInput {
            manual: shared.take_sync_now(),
            quitting: shared.quitting(),
        };
        let output = driver.tick(&mut store, input);

        // Publishes the raw clock window only — whether that suppresses capture
        // depends on whether the user is actually away, which only the capture
        // worker (with a live idle read) can decide. See `SharedStatus::quiet`.
        shared.set_quiet_window(output.quiet);
        shared.set_pending(output.pending_events);
        shared.set_sync(map_sync_state(output.state));

        if let Some(outcome) = &output.flushed {
            // Anything pulled down has to reach the capture thread, or a
            // machine that just inherited the user's settings and training
            // keeps detecting as though it had neither.
            if outcome.reload_settings {
                shared.request_settings_reload();
            }
            if outcome.reload_enrollments {
                shared.request_enrollment_reload();
            }
        }
        if let Some(error) = &output.error {
            eprintln!("sync: flush failed: {error}");
        }

        if output.done {
            break;
        }
        if output.next_wait == Duration::ZERO {
            // A flush just succeeded; there may be more pending work to drain.
            continue;
        }
        observed_signal = shared.wait_for_sync_signal(observed_signal, output.next_wait);
    }
    Ok(())
}
