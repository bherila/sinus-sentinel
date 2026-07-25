# Apple client architecture prototype

## Direction

Apple clients use native SwiftUI, AVFoundation, Core ML, Charts, and platform
lifecycle APIs. Rust remains the shared source of truth for:

- energy gate, log-mel frontend, decision rules, and sessionization;
- event persistence and settings;
- personalized prototype matching;
- history buckets and congestion scoring.

The existing egui desktop client stays functional while Apple work proceeds.
Windows is maintenance-only for this phase; no Windows code is removed.

## Boundary

`crates/app` is a UI-independent application library. A platform starts an
explicit monitoring session, converts microphone input to 16 kHz mono float PCM,
and pushes chunks into `MonitoringEngine`. Rust returns only structured events.
Stopping a session flushes its tail, resets detector clocks/cooldowns, and rejects
later PCM until another session starts.

`crates/apple` exposes that layer with UniFFI. Swift implements `ModelRunner`
using Core ML and receives gate-active `[96, 64]` log-mel patches. Keeping Core ML
native avoids shipping ONNX Runtime on iPhone and keeps quiet audio entirely on
the Rust gate-only path.

The Swift shell owns:

- microphone permission and user-visible session controls;
- `AVAudioSession` / `AVAudioEngine`;
- sample-rate/channel conversion;
- background, interruption, and route lifecycle;
- Core ML execution;
- Swift Charts and all accessibility/presentation behavior.

No raw PCM crosses into persistence or networking.

## One app per machine

Both macOS shells read `~/Library/Application Support/SinusSentinel/events.db`,
which is the point — one history, one set of settings, one device identity in the
PHR. It also means two of them listening at once would run two independent
detectors over one microphone and log every cough twice.

So `sinus_app::instance` is now shared rather than desktop-private, and the lock
is scoped to the *data directory*, not to a shell or a bundle id. The OS owns the
file lock, so it is released on a crash with no stale-PID handling. Whichever app
starts first owns the machine:

- the tray app losing the race writes an activation marker and exits, as before;
- the Apple shell losing the race fails `AppleEngine::new` with `AlreadyRunning`
  *before* opening the database or the microphone, and shows an explanatory
  screen instead of a half-live UI.

Known gap: the marker handoff only works owner-to-owner between tray apps. If the
SwiftUI app is the owner, launching the tray app exits silently instead of
revealing the running window — it does not yet consume activation requests.

Everything that is not platform knowledge now comes from that shared database.
`AppleEngineConfig` carries only the database path and the platform;
sensitivity, the battery policy and the device id are read from the store, so a
shell cannot reset a synced setting just by launching (it previously forced
sensitivity back to 0.5 and minted a second device id in `UserDefaults`).

Both Apple bundles use `com.bherila.sinus-sentinel`, matching the shipped tray
app: one product identity for the microphone grant, the keychain item and any
future notarized replacement. `open -n` launches the built prototype by path so
LaunchServices cannot substitute the other bundle registered under that id.

## Battery policy

`ProcessInfo.isLowPowerModeEnabled` is the one signal both Apple platforms share
(iOS 9+, macOS 12+), and the system posts `NSProcessInfoPowerStateDidChange` on
every transition, so `LowPowerMonitor` never polls. When Low Power Mode turns on
and the `pause_low_power` setting is enabled — the default, and the same store
row the desktop tray uses, so a Mac running both shells agrees with itself — the
shell tears down the `AVAudioEngine` tap and deactivates the audio session. That
releases the microphone rather than merely skipping analysis, which is what
actually lets the audio hardware and its 20-per-second wake-ups idle.

The user's intent is kept separately from whether capture is running
(`sessionRequested` vs `isCapturing`), so a session suspended for battery resumes
by itself when the device leaves Low Power Mode instead of silently ending.

## Cost of the hot path

Two things run continuously while a session is active, and both are kept off the
allocator:

- the converted 16 kHz buffer is allocated once per session and reused across
  the ~20 tap callbacks a second, not rebuilt per callback;
- Core ML outputs are copied out through `withUnsafeBufferPointer`, because the
  `MLMultiArray` subscript boxes each element in an `NSNumber` — roughly 1,500
  allocations per inference, on every gate-active window.

Projections (`history`) read through a second SQLite connection rather than the
mutex that serializes the detector. WAL readers never block on the writer, so a
UI refresh cannot stall behind a Core ML inference holding the engine lock.

## Monitoring through screen lock

The iOS prototype declares `UIBackgroundModes = audio`, activates an
`AVAudioSession` recording category only after the user starts monitoring, and
deactivates it when monitoring stops. This is the intended mechanism for an
active recording session to continue when the app backgrounds or the phone
locks. Apple explicitly recommends that recording apps keep the session active
only while they are recording.

This behavior cannot be proven on Linux or fully represented by Simulator
testing. The Mac handoff must verify it on a physical iPhone, including phone
calls, alarms, Siri, Bluetooth route changes, media-service resets, thermal
pressure, and low-power mode.

Apple references:

- [Audio playback, recording, and processing](https://developer.apple.com/documentation/avfoundation/audio-playback-recording-and-processing)
- [Requesting microphone permission](https://developer.apple.com/documentation/avfaudio/avaudioapplication/requestrecordpermission(completionhandler:))
- [Audio session activation and background recording](https://developer.apple.com/library/archive/documentation/Audio/Conceptual/AudioSessionProgrammingGuide/ConfiguringanAudioSession/ConfiguringanAudioSession.html)
- [Swift Charts](https://developer.apple.com/documentation/Charts)

## Build contract

On a Mac:

```bash
make apple-ios-run
make apple-macos-run
```

Both commands compile Rust first, generate bindings from that exact binary, and
then invoke Swift. There is no stale/prebuilt library fallback.

On Linux:

```bash
cargo test -p sinus-app -p sinus-apple
make apple-bindings-check
```

CI regenerates the committed Swift/header files and fails on any diff. UniFFI
generates Swift sources, a C header, and a module map; it does not build or sign
the Apple binaries, which is why the Mac continuation remains mandatory.
