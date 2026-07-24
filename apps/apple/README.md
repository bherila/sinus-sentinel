# Apple SwiftUI prototype

This directory is a native SwiftUI shell shared by macOS and iPhone. It is a
prototype handoff: the Rust engine and generated binding contract are tested on
Linux, while the Swift compile, microphone lifecycle, Core ML model, simulator,
and real-device background behavior must be completed on a Mac.

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

## Still requires a Mac / device

- compile both Swift destinations and address any SDK/Swift concurrency drift;
- convert and validate YAMNet as a Core ML package;
- handle audio interruptions, route changes, and media-services resets;
- verify that an active session continues through iPhone screen lock;
- profile battery use on a physical iPhone and Apple-silicon Mac;
- validate the declared background-audio use with App Review requirements;
- add signing, provisioning, XCTest/UI tests, and distributable XCFrameworks.
