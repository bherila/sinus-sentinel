//! Platform low-power-mode and idle-time detection. The capture worker samples
//! low-power-mode once per minute; user-idle is sampled far more often (see
//! `capture.rs`) because it feeds quiet-hours' absence check (SPEC §13 q4),
//! which needs to notice a returning user within moments, not an hour.

#[cfg(target_os = "macos")]
pub fn low_power_mode_enabled() -> bool {
    objc2_foundation::NSProcessInfo::processInfo().isLowPowerModeEnabled()
}

#[cfg(target_os = "windows")]
pub fn low_power_mode_enabled() -> bool {
    let mut status = windows_sys::Win32::System::Power::SYSTEM_POWER_STATUS::default();
    // SystemStatusFlag is non-zero when Windows battery saver is active.
    unsafe {
        windows_sys::Win32::System::Power::GetSystemPowerStatus(&mut status) != 0
            && status.SystemStatusFlag != 0
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn low_power_mode_enabled() -> bool {
    false
}

/// Time since the last keyboard/mouse/trackpad input, or `None` if the platform
/// offers no honest signal. `None` deliberately does not collapse to
/// `Some(Duration::ZERO)`: zero would read as "the user just touched the
/// machine", which would hold the microphone open all night on every query
/// failure. `sinus_app::sync::suppress_for_quiet_hours` treats `None` as "no
/// absence signal" and falls back to the literal window instead — the safe
/// direction when idle can't honestly be measured (e.g. iOS, where there is no
/// absence to detect because the device is in the user's pocket).
#[cfg(target_os = "macos")]
pub fn user_idle() -> Option<std::time::Duration> {
    // One CoreGraphics call: the framework is already linked transitively via
    // AppKit (winit/eframe need it for the window), so declaring this call
    // directly avoids pulling in the `core-graphics` crate for a single symbol.
    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGEventSourceSecondsSinceLastEventType(state_id: i32, event_type: u32) -> f64;
    }
    // CGEventSourceStateID::kCGEventSourceStateHIDSystemState.
    const HID_SYSTEM_STATE: i32 = 1;
    // CGEventType::kCGAnyInputEventType == (CGEventType)~0.
    const ANY_INPUT_EVENT_TYPE: u32 = u32::MAX;

    let seconds =
        unsafe { CGEventSourceSecondsSinceLastEventType(HID_SYSTEM_STATE, ANY_INPUT_EVENT_TYPE) };
    (seconds.is_finite() && seconds >= 0.0).then(|| std::time::Duration::from_secs_f64(seconds))
}

#[cfg(target_os = "windows")]
pub fn user_idle() -> Option<std::time::Duration> {
    use windows_sys::Win32::System::SystemInformation::GetTickCount64;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetLastInputInfo, LASTINPUTINFO};

    let mut info = LASTINPUTINFO {
        cbSize: std::mem::size_of::<LASTINPUTINFO>() as u32,
        dwTime: 0,
    };
    if unsafe { GetLastInputInfo(&mut info) } == 0 {
        return None;
    }
    // dwTime is stamped from the 32-bit GetTickCount, which wraps every ~49.7
    // days; GetTickCount64 does not. Compare in the same 32-bit width (wrapping
    // subtraction) rather than promoting dwTime into the 64-bit counter, which
    // would read as an enormous idle time for any machine that has been up
    // longer than dwTime's range.
    let now_low = GetTickCount64() as u32;
    let idle_ms = now_low.wrapping_sub(info.dwTime);
    Some(std::time::Duration::from_millis(idle_ms as u64))
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn user_idle() -> Option<std::time::Duration> {
    None
}
