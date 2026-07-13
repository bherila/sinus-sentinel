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

Pre-alpha; scaffold + core pipeline under construction (spec milestones M0–M2).

## Layout

- `crates/core` — audio pipeline, gate, DSP, inference, sessionizer, store, sync (no UI deps)
- `crates/cli` — headless harness: classify WAV files, benchmarks, threshold calibration
- `apps/desktop` — tray-icon + winit + egui app
- `model/` — ONNX model artifacts (fetched, not committed — see model/README.md)

## Build

```bash
cargo build          # workspace
cargo test           # core tests incl. golden-WAV corpus (skips if model absent)
```
