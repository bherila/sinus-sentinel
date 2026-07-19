# Sinus Sentinel

Menubar/tray app (macOS + Windows) that passively detects sinus-related sounds
(cough, throat clearing, sniffle, nose blow, hawk, snort/suck) via microphone,
classifies them **entirely on-device**, and logs structured events to a personal
health record backend. Offline-first; includes a strict never-upload mode.

**No raw audio is ever persisted or transmitted.** Classification runs locally
(YAMNet ONNX + few-shot prototype matching); only event metadata (type,
timestamp, duration, confidence) is stored or synced.

📄 **Read [docs/SPEC.md](docs/SPEC.md) first** — it is the source of truth for
architecture, event taxonomy, sync modes, and milestones.

## Status

Alpha. The pure-Rust core pipeline is complete and tested end-to-end; the desktop
shell compiles and runs (mic permission + tray require a real desktop session).

### Roadmap (SPEC §11 milestones)

- [x] **M1** — audio pipeline + log-mel + classifier scaffolding, CLI harness +
  golden corpus. Gate ①, YAMNet log-mel ②, ONNX/prototype classifier ③④,
  sessionizer ⑤, `cli classify/soak/calibrate`, deterministic golden test.
- [x] **M2** — SQLite store (WAL) + sessionizer + history view (egui_plot);
  events survive restart; false-positive ✕ tombstoning.
- [x] **M3 (client side)** — batch sync engine: idempotent upload, per-event
  accepted/duplicate/rejected, DELETE tombstones, jittered backoff, three modes.
  *Backend endpoints (2025-website) are tracked separately.*
- [x] **M4 (offline-strict)** — offline-strict is structurally incapable of
  network I/O (no engine constructed); CSV/JSON export is still to wire into UI.
- [~] **M0/M2 desktop shell** — tray + settings/history windows compile and run;
  live cpal capture behind `--features live-audio`.
- [~] **M5** — local Teach-mode capture + Phase B-lite live recognition are wired;
  broader real-world calibration and enrollment management remain.
- [ ] **M6** — mobile companion (stretch).

The ONNX YAMNet backbone is behind the `onnx` feature; everything builds and
tests **without** `model/yamnet.onnx` (a deterministic backbone drives tests).

## Layout

- `crates/core` — audio pipeline, gate, DSP, inference, sessionizer, store, sync (no UI deps)
- `crates/cli` — headless harness: classify WAV files, soak, threshold calibration
- `apps/desktop` — tray-icon + winit + egui app
- `model/` — ONNX model artifacts (fetched, not committed — see model/README.md)
- `testdata/` — synthetic golden-corpus WAVs

## Build

```bash
cargo build --workspace
cargo test  --workspace                       # incl. the golden-WAV corpus test

# Feature-gated pieces (off by default so CI needs no system libs / model file):
cargo build -p sinus-core --features onnx        # ort-backed YAMNet
cargo build -p sinus-desktop --features live-audio   # cpal microphone capture

# CLI
cargo run -p sinus-cli -- gen-testdata testdata
cargo run -p sinus-cli -- classify testdata/cough.wav
cargo run -p sinus-cli -- soak --secs 10
cargo run -p sinus-cli -- calibrate testdata

# Real-model diagnostics and local enrollment (requires ONNX Runtime):
ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
  cargo run -p sinus-cli --features onnx -- \
  classify sample.wav --model model/yamnet.onnx
ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
  cargo run -p sinus-cli --features onnx -- \
  enroll "$HOME/Library/Application Support/SinusSentinel/events.db" \
  sniffle sample.wav --model model/yamnet.onnx
```

The desktop Settings screen also has a fully local **Teach mode**. Pick a class,
make one clear sound during the three-second capture, and repeat until the UI shows
good repeat similarity and class separation (usually 3–5 varied samples). Raw
training audio is discarded; only 1024-value YAMNet embeddings are stored locally.
Saved takes can be removed individually, reset by class, or reset together.

The desktop app enforces one process per user. Launching it again activates the
existing History window instead of starting duplicate microphone, sync, tray, or
Keychain workers. Closing the window hides it; the menu-bar/tray item keeps running
until **Quit** is selected.
