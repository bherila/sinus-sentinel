# Apple SwiftUI client

This directory is a native SwiftUI shell shared by macOS and iPhone — one
target, two shells. The Rust engine and the generated binding contract are
tested on Linux; the Swift compile, microphone lifecycle, Core ML model,
simulator, and real-device background behavior have to be exercised on a Mac.

macOS is menu-bar-first: a History window, a `MenuBarExtra`, and a `Settings`
scene under ⌘,. iPhone has neither a menu bar nor a `Settings` scene, so
`RootTabView` presents the same views as History / Train / Settings tabs, with
pause beside Start/Stop and "Sync now" in Settings › PHR because their macOS
homes do not exist there. Backgrounding the app with no monitoring session
running stops the sync driver and rebuilds it on return. Everything else —
every model and the entire Rust surface — is shared; `#if os(...)` is confined
to `SinusSentinelApp`, `AppDelegate`, `MenuBarContent`, `RootTabView` and a few
marked lines.

The macOS shell is at feature parity with the menu-bar tray app, which is now
the Windows client. The iPhone shell has been compiled but never run.

## One-command build and run

Prerequisites:

- macOS 14 or newer
- Xcode with an iOS 17+ Simulator runtime (`xcode-select` pointing at it)
- stable Rust installed with Rustup

From the repository root:

```bash
make apple-ios-run
```

This command does not assume a prebuilt Rust artifact. It:

1. adds the Rust target matching the current Simulator architecture;
2. compiles `sinus-apple` as a static library;
3. regenerates `SinusApple.swift`, the C header, and module map from that exact
   library;
4. compiles and links the SwiftUI app with `xcrun swiftc`;
5. boots/selects an iOS Simulator, installs the app, and launches it.

The equivalent native macOS command is:

```bash
make apple-macos-run
```

Generated files under `Generated/` are committed so the API is reviewable and
Linux CI can fail if Rust metadata and Swift bindings drift.

## Model

The Swift `ModelRunner` trait is called only when Rust's energy gate marks a
patch active. `CoreMLYamnetRunner` expects:

- input `input`: float32 `[1, 96, 64]`
- output `scores`: float32 `[1, 521]`
- output `embeddings`: float32 `[1, 1024]`

Place a compiled model at `Resources/yamnet.mlmodelc`; the build command copies
it into the app bundle. Without it, `PreviewModelRunner` returns neutral,
correctly shaped values so the UI and capture lifecycle can run without false
detections.

## One app per machine

The SwiftUI app and the menu-bar app share one database, so only one may run at a
time — two detectors on one microphone would log every event twice. Whichever
starts first owns the machine through a lock on the shared data directory; the
other refuses to start and says so. Quit the running one first.

For the same reason both bundles use `com.bherila.sinus-sentinel` and take their
device id, sensitivity and battery policy from the shared database rather than
from per-app preferences.

## Battery

Monitoring pauses while the OS reports Low Power Mode, releasing the microphone
and resuming on its own when it turns off. The toggle in the app writes the same
`pause_low_power` row the desktop tray reads, so both shells on one Mac share the
policy. See [the architecture notes](../../docs/APPLE_ARCHITECTURE.md#battery-policy).

## Still requires a Mac / device

- run the iPhone shell for the first time — it is compiled by CI's
  `apple-native` job but has never been launched, on a Simulator or a device;
  a Mac with only the Command Line Tools has no `iphonesimulator` SDK and can
  build just the macOS shell;
- convert and validate YAMNet as a Core ML package;
- handle audio interruptions, route changes, and media-services resets;
- verify that an active session continues through iPhone screen lock;
- measure real battery draw on a physical iPhone and Apple-silicon Mac, including
  the Low Power Mode transition;
- validate the declared background-audio use with App Review requirements;
- add signing, provisioning, XCTest/UI tests, and distributable XCFrameworks.
