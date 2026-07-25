//! Typed accessors for the `settings` rows every client shares.
//!
//! The keys are a cross-platform contract: the desktop tray, the Apple clients
//! and the PHR sync engine all read and write the same rows, so the parsing and
//! the defaults live here once rather than being re-derived per shell. A missing
//! or malformed row always falls back to the documented default instead of
//! failing — settings are advisory, and a corrupt value must not stop capture.

use sinus_core::error::Result;
use sinus_core::store::Store;

/// Detection sensitivity, 0.0–1.0. Synced with the PHR.
pub const SENSITIVITY: &str = "sensitivity";
/// This machine's stable identity in the PHR.
pub const DEVICE_ID: &str = "device_id";
/// Whether to release the microphone while the OS reports low-power mode.
/// Device-local: battery policy is a property of the machine, not the patient.
pub const PAUSE_LOW_POWER: &str = "pause_low_power";

/// Neutral sensitivity, used when the row is absent or unparseable.
pub const DEFAULT_SENSITIVITY: f32 = 0.5;

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
}
