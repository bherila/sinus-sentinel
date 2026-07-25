//! Typed accessors for the `settings` rows every client shares.
//!
//! The keys are a cross-platform contract: the desktop tray, the Apple clients
//! and the PHR sync engine all read and write the same rows, so the parsing and
//! the defaults live here once rather than being re-derived per shell. A missing
//! or malformed row always falls back to the documented default instead of
//! failing — settings are advisory, and a corrupt value must not stop capture.

use chrono::{DateTime, Utc};

use sinus_core::error::{Error, Result};
use sinus_core::store::Store;
use sinus_core::sync::Mode;

use crate::state::PauseState;

/// Detection sensitivity, 0.0–1.0. Synced with the PHR.
pub const SENSITIVITY: &str = "sensitivity";
/// This machine's stable identity in the PHR.
pub const DEVICE_ID: &str = "device_id";
/// Whether to release the microphone while the OS reports low-power mode.
/// Device-local: battery policy is a property of the machine, not the patient.
pub const PAUSE_LOW_POWER: &str = "pause_low_power";
/// Sync behaviour (SPEC §4.3). Device-local: a laptop on metered wifi and a
/// desktop on ethernet reasonably want different modes for the same patient.
pub const MODE: &str = "mode";
/// PHR base URL. Device-local, like `MODE`.
pub const SERVER_URL: &str = "server_url";
/// The PHR patient these events belong to. Device-local, like `MODE`.
pub const PATIENT_ID: &str = "patient_id";
/// Quiet-hours window start, local hour 0-23. Synced with the PHR so the
/// window follows the patient between machines.
pub const QUIET_START: &str = "quiet_start";
/// Quiet-hours window end, local hour 0-23. See `QUIET_START`.
pub const QUIET_END: &str = "quiet_end";
/// Device-local: when a pause set in one shell should still be in force after a
/// relaunch. Absent means running.
pub const PAUSE_UNTIL: &str = "pause_until";

/// Neutral sensitivity, used when the row is absent or unparseable.
pub const DEFAULT_SENSITIVITY: f32 = 0.5;

/// The literal `pause_until` value that marks an indefinite pause, as opposed
/// to an RFC3339 timestamp for a timed one.
const PAUSE_INDEFINITE: &str = "indefinite";

/// Stored sensitivity, clamped to the slider's range.
pub fn sensitivity(store: &Store) -> f32 {
    store
        .setting_get(SENSITIVITY)
        .ok()
        .flatten()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .map(|value| value.clamp(0.0, 1.0))
        .unwrap_or(DEFAULT_SENSITIVITY)
}

pub fn set_sensitivity(store: &Store, sensitivity: f32) -> Result<()> {
    let sensitivity = if sensitivity.is_finite() {
        sensitivity.clamp(0.0, 1.0)
    } else {
        DEFAULT_SENSITIVITY
    };
    store.setting_set(SENSITIVITY, &sensitivity.to_string())
}

/// Battery policy, on by default: only an explicit `false` opts out, so a fresh
/// install and an install that predates the setting behave the same.
pub fn pause_on_low_power(store: &Store) -> bool {
    store
        .setting_get(PAUSE_LOW_POWER)
        .ok()
        .flatten()
        .is_none_or(|value| value != "false")
}

pub fn set_pause_on_low_power(store: &Store, enabled: bool) -> Result<()> {
    store.setting_set(PAUSE_LOW_POWER, if enabled { "true" } else { "false" })
}

/// This machine's device id, minted on first use.
///
/// It lives in the database rather than in each shell's own preferences, so the
/// tray app and the SwiftUI app are one device to the PHR instead of two — they
/// are, after all, one machine sharing one event history.
pub fn ensure_device_id(store: &Store) -> String {
    if let Ok(Some(id)) = store.setting_get(DEVICE_ID) {
        if !id.is_empty() {
            return id;
        }
    }
    let id = uuid::Uuid::new_v4().to_string();
    let _ = store.setting_set(DEVICE_ID, &id);
    id
}

/// Sync mode. An absent or unrecognized row (e.g. from a future version's new
/// variant) falls back to the safest default that still syncs.
pub fn mode(store: &Store) -> Mode {
    store
        .setting_get(MODE)
        .ok()
        .flatten()
        .map(|value| match value.as_str() {
            "offline-first" => Mode::OfflineFirst,
            "offline-strict" => Mode::OfflineStrict,
            _ => Mode::AutoBatch,
        })
        .unwrap_or(Mode::AutoBatch)
}

pub fn set_mode(store: &Store, mode: Mode) -> Result<()> {
    store.setting_set(MODE, mode.as_str())
}

/// PHR base URL, empty when never configured.
pub fn server_url(store: &Store) -> String {
    store
        .setting_get(SERVER_URL)
        .ok()
        .flatten()
        .unwrap_or_default()
}

pub fn set_server_url(store: &Store, url: &str) -> Result<()> {
    store.setting_set(SERVER_URL, url)
}

/// The configured patient id, or `None` if it is absent, blank, unparseable,
/// or not a positive id — none of those are a patient sync can target.
pub fn patient_id(store: &Store) -> Option<i64> {
    store
        .setting_get(PATIENT_ID)
        .ok()
        .flatten()
        .filter(|value| !value.trim().is_empty())
        .and_then(|value| value.trim().parse::<i64>().ok())
        .filter(|id| *id > 0)
}

/// `None` clears the row (written as `""`, which `patient_id` reads back as
/// `None`) rather than leaving the previous id in place.
pub fn set_patient_id(store: &Store, id: Option<i64>) -> Result<()> {
    store.setting_set(PATIENT_ID, &id.map(|id| id.to_string()).unwrap_or_default())
}

/// Quiet-hours window as local hours `[start, end)`, or `None` when disabled —
/// either row absent/unparseable, or `start == end`, which is the documented
/// "no quiet hours" encoding (see `set_quiet_hours`).
pub fn quiet_hours(store: &Store) -> Option<(u32, u32)> {
    let get = |key: &str| {
        store
            .setting_get(key)
            .ok()
            .flatten()
            .and_then(|value| value.parse::<u32>().ok())
    };
    match (get(QUIET_START), get(QUIET_END)) {
        (Some(start), Some(end)) if start != end => Some((start, end)),
        _ => None,
    }
}

/// Set or clear the quiet-hours window. Both hours must be `0..=23`; an
/// out-of-range hour is a caller bug (unlike a corrupt stored row, which just
/// falls back to disabled), so it is rejected rather than clamped.
/// `None` clears the window by writing `start == end`, the encoding
/// `quiet_hours` already treats as disabled.
///
/// Written with `setting_set`, not `setting_set_synced`: these two keys are in
/// `SYNCED_SETTING_KEYS`, and a *local* write must be marked dirty (stamped
/// `updated_at = now`) so the next flush pushes it to the PHR.
/// `setting_set_synced` is for adopting a value that already came *from* the
/// server, which this is not.
pub fn set_quiet_hours(store: &Store, hours: Option<(u32, u32)>) -> Result<()> {
    let (start, end) = hours.unwrap_or((0, 0));
    if start > 23 || end > 23 {
        return Err(Error::Config(format!(
            "quiet hours must be 0..=23, got start={start} end={end}"
        )));
    }
    store.setting_set(QUIET_START, &start.to_string())?;
    store.setting_set(QUIET_END, &end.to_string())?;
    Ok(())
}

/// Device-local pause state. Reads only what is stored — the caller applies
/// `PauseState::normalized(now)` itself, since only the caller knows `now` and
/// whether an expired timed pause should be treated as running yet.
pub fn pause_state(store: &Store) -> PauseState {
    match store.setting_get(PAUSE_UNTIL).ok().flatten() {
        None => PauseState::Running,
        Some(value) if value.is_empty() => PauseState::Running,
        Some(value) if value == PAUSE_INDEFINITE => PauseState::PausedIndefinite,
        Some(value) => DateTime::parse_from_rfc3339(&value)
            .map(|t| PauseState::PausedUntil(t.with_timezone(&Utc)))
            .unwrap_or(PauseState::Running),
    }
}

pub fn set_pause_state(store: &Store, state: PauseState) -> Result<()> {
    match state {
        PauseState::Running => store.setting_set(PAUSE_UNTIL, ""),
        PauseState::PausedIndefinite => store.setting_set(PAUSE_UNTIL, PAUSE_INDEFINITE),
        PauseState::PausedUntil(t) => store.setting_set(PAUSE_UNTIL, &t.to_rfc3339()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensitivity_defaults_and_clamps() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(sensitivity(&store), DEFAULT_SENSITIVITY);

        store.setting_set(SENSITIVITY, "not a number").unwrap();
        assert_eq!(sensitivity(&store), DEFAULT_SENSITIVITY);

        store.setting_set(SENSITIVITY, "2.5").unwrap();
        assert_eq!(sensitivity(&store), 1.0);

        set_sensitivity(&store, -3.0).unwrap();
        assert_eq!(sensitivity(&store), 0.0);

        set_sensitivity(&store, f32::NAN).unwrap();
        assert_eq!(sensitivity(&store), DEFAULT_SENSITIVITY);
    }

    #[test]
    fn device_id_is_minted_once_and_shared_by_every_shell() {
        let store = Store::open_in_memory().unwrap();
        let first = ensure_device_id(&store);
        assert!(!first.is_empty());
        assert_eq!(ensure_device_id(&store), first);
    }

    #[test]
    fn low_power_pause_is_opt_out() {
        let store = Store::open_in_memory().unwrap();
        assert!(pause_on_low_power(&store));

        set_pause_on_low_power(&store, false).unwrap();
        assert!(!pause_on_low_power(&store));

        set_pause_on_low_power(&store, true).unwrap();
        assert!(pause_on_low_power(&store));
    }

    #[test]
    fn mode_round_trips_and_falls_back_on_unknown_string() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(mode(&store), Mode::AutoBatch);

        set_mode(&store, Mode::OfflineFirst).unwrap();
        assert_eq!(mode(&store), Mode::OfflineFirst);

        set_mode(&store, Mode::OfflineStrict).unwrap();
        assert_eq!(mode(&store), Mode::OfflineStrict);

        store.setting_set(MODE, "not-a-mode").unwrap();
        assert_eq!(mode(&store), Mode::AutoBatch);
    }

    #[test]
    fn server_url_defaults_empty_and_round_trips() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(server_url(&store), "");

        set_server_url(&store, "https://phr.example").unwrap();
        assert_eq!(server_url(&store), "https://phr.example");
    }

    #[test]
    fn patient_id_rejects_non_positive_and_garbage() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(patient_id(&store), None);

        store.setting_set(PATIENT_ID, "").unwrap();
        assert_eq!(patient_id(&store), None);

        store.setting_set(PATIENT_ID, "0").unwrap();
        assert_eq!(patient_id(&store), None);

        store.setting_set(PATIENT_ID, "-5").unwrap();
        assert_eq!(patient_id(&store), None);

        store.setting_set(PATIENT_ID, "not a number").unwrap();
        assert_eq!(patient_id(&store), None);

        set_patient_id(&store, Some(42)).unwrap();
        assert_eq!(patient_id(&store), Some(42));

        set_patient_id(&store, None).unwrap();
        assert_eq!(patient_id(&store), None);
    }

    #[test]
    fn quiet_hours_round_trips_and_start_eq_end_disables() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(quiet_hours(&store), None);

        set_quiet_hours(&store, Some((22, 7))).unwrap();
        assert_eq!(quiet_hours(&store), Some((22, 7)));

        set_quiet_hours(&store, Some((9, 9))).unwrap();
        assert_eq!(quiet_hours(&store), None);

        set_quiet_hours(&store, None).unwrap();
        assert_eq!(quiet_hours(&store), None);

        assert!(set_quiet_hours(&store, Some((24, 7))).is_err());
        assert!(set_quiet_hours(&store, Some((7, 24))).is_err());
    }

    #[test]
    fn pause_state_round_trips_all_variants_and_falls_back_on_garbage() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(pause_state(&store), PauseState::Running);

        set_pause_state(&store, PauseState::PausedIndefinite).unwrap();
        assert_eq!(pause_state(&store), PauseState::PausedIndefinite);

        let until = Utc::now() + chrono::Duration::minutes(15);
        set_pause_state(&store, PauseState::PausedUntil(until)).unwrap();
        // RFC3339 round-trips through the store at second precision.
        match pause_state(&store) {
            PauseState::PausedUntil(t) => {
                assert_eq!(t.timestamp(), until.timestamp());
            }
            other => panic!("expected PausedUntil, got {other:?}"),
        }

        set_pause_state(&store, PauseState::Running).unwrap();
        assert_eq!(pause_state(&store), PauseState::Running);

        store.setting_set(PAUSE_UNTIL, "not a timestamp").unwrap();
        assert_eq!(pause_state(&store), PauseState::Running);
    }
}
