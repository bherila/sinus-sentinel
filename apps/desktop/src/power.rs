//! Platform low-power-mode detection. The capture worker samples this only once
//! per minute; OS battery saver can then fully suspend microphone monitoring when
//! the user leaves the default "pause in low-power mode" setting enabled.

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
