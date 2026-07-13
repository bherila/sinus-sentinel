//! Sinus Sentinel desktop app (SPEC §6) — tray-first, single process, egui UI.
//!
//! macOS Dock-hiding (LSUIElement) is set in the app bundle's `Info.plist` by the
//! packaging step (cargo-dist, SPEC §11), not programmatically, to avoid coupling
//! to winit platform internals — see `macos` note below.

mod app;
#[cfg(feature = "live-audio")]
mod capture;
mod shared;
mod state;
mod sync;

use std::path::PathBuf;

use sinus_core::store::Store;

/// Platform application-data directory (SPEC §8: user-only app-data dir).
fn data_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join("Library/Application Support/SinusSentinel");
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(appdata) = std::env::var("APPDATA") {
            return PathBuf::from(appdata).join("SinusSentinel");
        }
    }
    std::env::temp_dir().join("SinusSentinel")
}

fn main() -> eframe::Result<()> {
    let dir = data_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("warning: could not create data dir {dir:?}: {e}");
    }
    let store = Store::open(dir.join("events.db")).unwrap_or_else(|e| {
        eprintln!("fatal: could not open event store: {e}");
        std::process::exit(1);
    });

    // Shared, lock-free status bus between the capture/sync worker threads and the
    // tray/UI (SPEC §6). Cloned into each thread; the same cells are observed.
    let shared = shared::SharedStatus::default();

    // The background sync thread drives SyncEngine off the UI thread (SPEC §4.3).
    // It runs in every build (independent of audio) so stored events still upload
    // and quiet-hours state is always published.
    let _sync = sync::spawn_sync(dir.join("events.db"), shared.clone());

    // The live capture thread is only spawned when built with `live-audio` and
    // requires mic permission (granted by the OS on first run).
    #[cfg(feature = "live-audio")]
    let _capture = capture::spawn_capture(dir.join("events.db"), shared.clone());

    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([760.0, 520.0])
            .with_title("Sinus Sentinel"),
        ..Default::default()
    };

    eframe::run_native(
        "Sinus Sentinel",
        native_options,
        Box::new(move |cc| Ok(Box::new(app::SinusApp::new(cc, store, shared)))),
    )
}
