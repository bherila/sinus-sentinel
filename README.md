# Sinus Sentinel

Menu-bar/tray app (macOS + Windows) that passively detects sinus-related sounds —
cough, throat clearing, sniffle, sneeze, nose blow, hawk, snort/suck — via the
microphone, classifies them **entirely on-device**, and logs structured events to
a personal health record (PHR) backend.

**No raw audio is ever persisted or transmitted.** Classification runs locally
(YAMNet ONNX + few-shot prototype matching); what leaves the device is event
metadata (type, timestamp, duration, confidence, burst count, loudness) and —
only once you connect a PHR — the derived embeddings behind Teach mode, so your
personalized detector follows you between machines. Embeddings are opaque float
vectors from which audio cannot be reconstructed, and they never leave the device
in offline modes. Offline-first, with a strict mode that is structurally
incapable of network I/O.

📄 **[docs/SPEC.md](docs/SPEC.md)** is the source of truth for architecture,
event taxonomy, sync modes, and milestones.

## Quick start

```bash
# Prerequisites: stable Rust (rustup), and for the real model path on macOS:
brew install onnxruntime

# Build and run the desktop app with the full feature set
cargo run --release -p sinus-desktop --features live-audio,onnx,keyring
```

Grant microphone permission on first launch. The app lives in the menu bar/tray:
closing the window hides it, monitoring keeps running, and only the tray **Quit**
exits. Launching a second copy just focuses the running one.

Tray states: 🟢 listening · ⏸ paused · ⚠ model unavailable · 📴 offline-strict.

### The model file

`model/yamnet.onnx` is **not committed** (size + provenance). Reproduce it from
the upstream TF-Hub module with `model/fetch.sh` (needs `python3` with
`tensorflow`, `tensorflow-hub`, `tf2onnx` — see [model/README.md](model/README.md)
for the exact tensor contract). Everything builds, tests, and runs **without**
it: the app falls back to a deterministic band-heuristic backbone and shows the
⚠ tray state.

At runtime the `onnx` feature loads the ONNX Runtime shared library dynamically.
Homebrew's install is found automatically on macOS; elsewhere point
`ORT_DYLIB_PATH` at `libonnxruntime.{dylib,so}` / `onnxruntime.dll`.

## Install

Prebuilt archives for macOS (arm64) and Windows (x64) are published on
[GitHub Releases](https://github.com/bherila/sinus-sentinel/releases) from `v*`
tags. Unpack and run the binary; there is no installer yet (a signed/notarized
macOS bundle is tracked in issue #5).

From source:

```bash
cargo build --release -p sinus-desktop --features live-audio,onnx,keyring
# → target/release/sinus-desktop  (copy wherever you like)
```

## Connecting to a PHR backend

In **Settings** set the server URL and patient id, and paste an API token (stored
in the OS keychain — requires the `keyring` feature; never written to disk or
displayed). Failures back off exponentially (30 s → 30 min, jittered).

A flush syncs, in order:

| What | Endpoint (under `{server}/api/phr/patients/{patient_id}/`) |
|---|---|
| Events, idempotent batches of ≤500 | `respiratory-events/batch` |
| Tombstones | `respiratory-events/batch` (DELETE) |
| False-positive / correction flags | `respiratory-events/flag-batch` |
| Sensitivity + quiet hours, last-write-wins | `sinus-settings` |
| Teach-mode enrollments, batches of ≤100 | `sinus-enrollments/batch` |

A PHR that predates the settings/enrollment endpoints answers 404; those steps
are skipped rather than failing the flush, so events keep uploading.

Three sync modes (tray menu): **auto-batch** (flushes on a pending-count
threshold, a timer, or quit), **offline-first** (metered-friendly: a longer
scheduled interval or an explicit "Sync now" — never on threshold or quit), and
**offline-strict** — in which no HTTP client is ever constructed, so no network
I/O path exists at all.

## Using the app

- **History** shows today's per-class counts, a congestion score, a stacked
  7-day trend chart (per-class colors, hover for exact counts), and recent
  events with their peak loudness in dBFS. While a sound is being analyzed a
  **🔊 heard something** line appears, so the app is not silent during the
  second or two before a classification lands.
- **✕ Report false positive** on any recent event stops it counting — here and
  in the PHR — and enrolls the sound's embedding as a negative for the class
  that fired, so the detector stops applying that label to near-identical
  sounds. The event is flagged rather than deleted: the record that the
  classifier got something wrong is worth keeping, and **↺** undoes it. Learned
  suppressions are listed in Teach mode and can be forgotten there.
- **Recharacterize** an event (the **…** picker beside it) if you know what the
  sound really was. It enrolls the embedding as a negative for the wrong class
  *and* a positive for the right one, and relabels the event so it keeps
  counting — as the corrected class — locally and in the PHR. The wrong label
  stops firing immediately; teach the correct class a few more times for it to
  be recognized on its own.
- **Teach mode** (Settings) trains personalized classes: pick a class, make one
  clear sound after the countdown, repeat with 3–5 varied takes until the UI
  shows good repeat similarity and separation. Raw audio is discarded; only
  1024-value YAMNet embeddings are stored. With a PHR connected these sync, so a
  second machine inherits your trained detector.
- **Pause** (15 min / 1 h / until resumed) from the tray releases the microphone
  and stops analysis entirely, then opens a fresh capture session on resume.
  Quiet hours do the same, and the default battery policy automatically pauses
  while macOS/Windows low-power mode is active (toggleable in Settings).
  A sensitivity slider
  in Settings scales all detection thresholds, takes effect immediately, and —
  along with quiet hours — syncs across your machines via the PHR. Server URL,
  patient id and sync mode stay device-local.

### Data locations

| Platform | Path |
|---|---|
| macOS | `~/Library/Application Support/SinusSentinel/events.db` |
| Windows | `%APPDATA%\SinusSentinel\events.db` |

SQLite in WAL mode; events survive kill/relaunch. Detection embeddings kept so a
recent event can be reported or recharacterized are local-only and pruned after
30 days — distinct from Teach-mode enrollments, which do sync when a PHR is
connected.

## CLI

A headless harness over the identical pipeline (same code path as live capture):

```bash
cargo run -p sinus-cli -- gen-testdata testdata     # synthesize the golden WAVs
cargo run -p sinus-cli -- classify testdata/cough.wav
cargo run -p sinus-cli -- soak --secs 10            # long-running stability run
cargo run -p sinus-cli -- calibrate testdata        # per-class threshold sweep

# With the real model (requires ONNX Runtime):
cargo run -p sinus-cli --features onnx -- \
  classify sample.wav --model model/yamnet.onnx

# Enroll a WAV as a Teach-mode example straight into the app's database:
cargo run -p sinus-cli --features onnx -- \
  enroll "$HOME/Library/Application Support/SinusSentinel/events.db" \
  sniffle sample.wav --model model/yamnet.onnx
```

## Development

```bash
cargo test --workspace          # includes the golden-WAV corpus test
cargo clippy --workspace --all-targets --all-features
```

Features are **off by default** so CI and plain builds need no system libraries
or model file:

| Feature | Crate(s) | Enables |
|---|---|---|
| `live-audio` | desktop, core | cpal microphone capture |
| `onnx` | desktop, cli, core | real YAMNet backbone via ONNX Runtime (`load-dynamic`) |
| `keyring` | desktop, core | API token in the OS keychain |

### Layout

- `crates/core` — audio pipeline (gate → log-mel → embed → decision →
  sessionizer), store, sync engine; no UI deps
- `crates/cli` — headless harness: classify, soak, calibrate, enroll
- `apps/desktop` — tray + egui app, live capture, Teach mode, sync scheduler
- `model/` — ONNX model artifacts (fetched, not committed)
- `testdata/` — synthetic golden-corpus WAVs
- `docs/SPEC.md` — the spec

A key invariant, enforced by test: feeding a signal to the streaming pipeline in
arbitrary chunks yields **exactly** the same events as one batch call — the CLI,
tests, and live capture share one engine.

The quiet path is gate-only: raw audio is retained just long enough to supply
pre-roll if the gate opens, while log-mel FFTs and model inference are skipped.
Audio callbacks wake the worker in 50 ms batches; the hidden UI and sync worker
sleep until an event or real deadline rather than polling.

## Status

Alpha (`v0.2.0-alpha`). Core pipeline complete and tested end-to-end; desktop
shell with live capture, Teach mode, false-positive training, sync, and releases
working. Remaining: signed/notarized macOS bundle (#5), real-world accuracy
corpus + evaluation (#4), CSV/JSON export in the UI, mobile companion (stretch).

## License

MIT
