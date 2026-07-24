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
