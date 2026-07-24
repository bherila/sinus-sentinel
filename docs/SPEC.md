# Sinus Sentinel — Acoustic Sinus-Event Monitor (Spec v0.3)

**Status:** Draft for review · **Date:** 2026-07-13
**Scope:** Cross-platform menubar/tray desktop app (macOS + Windows) that passively detects sinus-related sounds via microphone, classifies them **entirely on-device**, and logs structured events to the owner's PHR system (`2025-website` Laravel backend). Stretch: a mobile companion built from the same core.

> **v0.2:** desktop UI switched from Tauri 2 webview to **pure-Rust egui** (single process, no webview) after resource-usage review; Tauri 2 demoted to one of three candidate *mobile* shells (§10). Core crate unchanged.
> **v0.3:** custom-class recognition restructured around **user enrollment ("teach mode") + few-shot prototype matching** (§5 Phase B-lite) — record your own sniffs/hawks/blows and recognize them by embedding similarity, no training loop; the fine-tuned MLP head becomes an optional later upgrade (Phase B-full).

---

## 1. Goals & non-goals

### Goals
1. **Passive acoustic detection** of sinus/airway events while the app runs: throat clearing, sniffling, hawking, nose blowing, snort/suck-back, coughing (sneeze as a freebie — the base model detects it well).
2. **Log every instance** with timestamp, type, confidence, and duration to the PHR (`phr` patient records), so congestion patterns can be charted over time and correlated with medications, weather, and the sinus CT already in the PHR imaging module.
3. **Low resource usage**: idle CPU ≈ 0 when quiet (energy-gated), single-digit % CPU during classification bursts, <100 MB RSS, no GPU required.
4. **Privacy by construction**: raw audio never leaves the device and is never written to disk (except short opt-in clips for model improvement, stored locally only).
5. **Offline-first**: events queue locally in SQLite and upload in **batches** when connectivity allows. Killing/relaunching the app never loses events.
6. **Offline mode (strict)**: a user-selectable mode in which the app **never uploads anything** and instead provides an in-app history UX (daily counts, timeline, trends). Leaving offline mode offers — but does not force — a backfill upload.
7. **Menubar/tray-first UX**: at-a-glance status, today's counts, pause/mute, mode switch. A real window exists only for settings/history.

### Non-goals (v1)
- Diagnosing anything. This is a symptom diary, not a medical device. No alerts, no health claims.
- Multi-user / multi-patient. v1 hardcodes one PHR patient binding per device (configurable).
- Speech recognition or retention of any intelligible audio. The classifier consumes log-mel frames; no transcript path exists.
- Background (screen-off) capture on iOS. See §10.

---

## 2. Event taxonomy

| `event_type` | Sound | Semantic bucket | Base-model coverage (AudioSet/YAMNet) |
|---|---|---|---|
| `cough` | Single or burst cough | irritation | ✅ native class |
| `throat_clearing` | "Ahem" | clearing action | ✅ native class |
| `sniffle` | Short nasal inhale | **blockage indicator** | ✅ native class ("Sniff") |
| `sneeze` | Sneeze | irritation | ✅ native class |
| `nose_blow` | Blowing nose into tissue | clearing action | ⚠️ weak/absent — needs fine-tune |
| `hawk` | Deep forceful mucus clear ("hocking") | clearing action | ❌ custom class |
| `snort_suck` | Backward suction/snort of mucus | **blockage indicator** | ❌ custom class |

Two-phase model strategy (§5): ship v1 with the four native classes; add the three custom classes via a fine-tuned head trained on self-labeled clips.

**Derived metric — daily congestion score (v1, deliberately simple):**
`score = 2·(sniffle + snort_suck) + 1·(nose_blow + hawk + throat_clearing) + 0.5·cough`, normalized per monitored-hour. Blockage indicators weigh double because they indicate the state; clearing actions indicate the response. Rendered as a 7/30-day trend both in-app and later in the PHR web UI.

---

## 3. Technology choice

### Decision matrix

| | **Pure Rust: tray-icon + winit + egui** (recommended) | Rust + Tauri 2 | Go (systray + fyne) | C++ / Qt |
|---|---|---|---|---|
| Menubar/tray Mac+Win | ✅ (`tray-icon` crate) | ✅ first-class | ✅ | ✅ |
| Extra processes | **none — single process** | WebContent (macOS) / `msedgewebview2.exe` family (Windows) while a window is open | none | none |
| Idle footprint (tray only) | ~20–35 MB | ~30–50 MB | ~50 MB | ~40 MB |
| Window-open footprint | +~30 MB in-process (renderer torn down on close) | +100–300 MB webview processes | +~40 MB | +~40 MB |
| On-device ML | ✅ `ort` (ONNX Runtime) | ✅ `ort` | ⚠️ cgo bindings, weak story | ✅ ONNX RT C++ |
| Audio capture | ✅ `cpal` | ✅ `cpal` | ⚠️ portaudio/malgo | ✅ Qt Multimedia |
| Mobile from same core | core ✅ / UI shell per §10 | ✅ core + webview UI | ❌ | ⚠️ Qt mobile ($$, heavy) |
| Settings/history UI cost | Medium (immediate-mode, `egui_plot` for charts) | Low (web UI) | Medium | High |
| Toolchain | cargo only | cargo + Node/JS bundler | go | cmake/qmake |

**Recommendation (v0.2): pure Rust — `tray-icon` + `winit` + `egui`/`eframe`, single process, no webview.**
The always-on part (audio → gate → classifier → queue → sync) is a **pure-Rust core crate with no UI dependency** in every option — that's where the low-resource requirement lives. The differentiator is the occasionally-open settings/history window: egui renders in-process (GPU-accelerated via glow/wgpu, event-driven repaint — no continuous render loop), so opening the history view never spawns webview helper processes, and the repo carries no JS toolchain. The costs are honest but acceptable for a personal utility: utilitarian aesthetics, and charts hand-built with `egui_plot` instead of a JS chart library. Tauri 2 remains the fallback if the UI ambitions grow, and one of three candidate shells for mobile (§10) — the core crate is identical either way. C++/Qt would be similarly performant but costs far more development effort; Go's on-device ML and audio stories are the weakest of the four.

### Core dependencies
- `cpal` — cross-platform audio capture (CoreAudio / WASAPI / AAudio / iOS CoreAudio) + `ringbuf` (lock-free SPSC between audio callback and analysis thread) + `rubato` (resampling to 16 kHz when the device won't open at 16 kHz natively)
- `realfft`/`rustfft` — STFT for the log-mel frontend (§5.1)
- `ort` — ONNX Runtime inference (CPU EP; CoreML/DirectML optional, not required)
- `rusqlite` — event store + upload queue (WAL mode)
- `reqwest` (rustls) — batch uploader
- `tray-icon` + `winit` + `eframe`/`egui` + `egui_plot` — tray, event loop, settings/history window
- `keyring` (OS keychain), `auto-launch` (login item / registry autostart), `notify-rust` (optional notifications)
- Releases via `cargo-dist` (signed/notarized macOS bundle, Windows installer)
- Model artifacts: YAMNet ONNX (~4 MB) + custom head (~100 KB), bundled in-app

---

## 4. Architecture

```
┌─────────────────────────── desktop app ────────────────────────────┐
│                                                                    │
│  mic ──► cpal stream (16 kHz mono f32)                             │
│            │                                                       │
│            ▼                                                       │
│      ring buffer (3 s)                                             │
│            │                                                       │
│            ▼                                                       │
│   ①  energy + spectral-flux gate        (~0 CPU when quiet)        │
│            │  trips                                                │
│            ▼                                                       │
│   ②  log-mel frontend (0.975 s window, 0.5 s hop)                  │
│            ▼                                                       │
│   ③  YAMNet embedding (ONNX, CPU, ~ms)                             │
│            ▼                                                       │
│   ④  class head: 4 native + 3 custom classes + calibration         │
│            ▼                                                       │
│   ⑤  debounce/sessionizer (merge <1.5 s gaps, per-class cooldown,  │
│       confidence threshold w/ user sensitivity)                    │
│            ▼                                                       │
│   ⑥  SQLite event store (uuid, type, t, dur, conf, model_ver)      │
│            │                                    ▲                  │
│            ▼                                    │                  │
│   ⑦  sync engine ──► batch POST /api/phr/.../respiratory-events   │
│       (modes: auto-batch | offline-first | offline-strict)         │
│                                                                    │
│  tray icon/menu ◄── status bus ──► settings+history window (egui)  │
└────────────────────────────────────────────────────────────────────┘
```

Stages ①–⑦ live in `crates/core` (no UI imports) so they compile unchanged into the mobile shells and into a headless CLI used for tests/benchmarks.

### 4.1 Audio analysis, specifically

**Capture & transport.** `cpal` opens the selected input in shared mode, mono. If the device opens at 16 kHz natively, use it; otherwise capture at the device rate (typically 44.1/48 kHz) and resample to 16 kHz with `rubato` (fixed-ratio polyphase). The audio callback is real-time-safe: it only copies samples into a lock-free SPSC ring buffer (`ringbuf`, 3 s capacity) — no allocation, no locks, no inference. A dedicated analysis thread drains it in 50 ms hops.

**Important platform caveat:** OS "voice processing" (macOS Voice Isolation, Windows audio enhancements, echo cancellation) is tuned to *remove* exactly the non-speech sounds we detect. Open the raw input stream (no VPIO on macOS; raw mode on WASAPI where available) and document "disable mic enhancements" for Windows devices that force them.

**Stage ① — energy gate (runs always, ≈0 CPU).** Per 50 ms hop compute RMS and a first-difference (high-passed) energy ratio — no FFT. Maintain an adaptive noise floor: EMA that rises slowly (~3 s time constant) and falls fast, so a persistent fan raises the floor but a sudden sniff doesn't. Gate opens at floor + ~10 dB with hysteresis; it stays open until energy sits below threshold for 1 s. Everything downstream runs **only while the gate is open** plus a 1 s pre-roll pulled from the ring buffer (so the *onset* of the sound is always analyzed, not just its tail).

**Stage ② — log-mel frontend (YAMNet's exact recipe — do not improvise).** 16 kHz mono → STFT with 25 ms Hann window, 10 ms hop → 64 mel bands spanning 125–7500 Hz → `log(mel + 0.001)`. A classifier window is 0.975 s = 96 mel frames × 64 bands. Windows hop 0.5 s while the gate is open. No per-window normalization, no AGC in the feature path — YAMNet was trained on this exact scaling, and "improving" it silently wrecks calibration. (`realfft` + a precomputed mel filterbank; microseconds per window.)

**Stage ③ — model inference.** YAMNet (AudioSet-pretrained, MobileNet-v1, ~3.7 M params) exported to ONNX, run with `ort` on CPU: ~2–5 ms per window on Apple Silicon, similar on modern x86. Two outputs per window: 521 AudioSet class scores *and* the 1024-d embedding. Phase A uses the scores directly (Cough, Throat clearing, Sniff, Sneeze, Speech); Phase B feeds the same embedding to the custom head (§5) in the same pass — one backbone run serves both.

**Stage ④ — decision logic.**
- Per-class calibrated thresholds θ_c (defaults set from the golden corpus, scaled by the user sensitivity slider).
- Transient classes (`cough`, `sneeze`, `nose_blow`, `hawk`): a single window ≥ θ_c fires.
- Weak/short classes (`sniffle`, `snort_suck`, `throat_clearing`): require score ≥ θ_c **coincident with** a gate-energy peak, which suppresses the classic false positive of breathy speech scoring low-grade "Sniff" for minutes.
- **Speech guard:** if the Speech score dominates the window (Speech > 0.5 and > 1.5× the candidate class), suppress everything except a very-high-confidence cough — plosives and inhales during talking are the main false-positive source.
- When another app grabs the mic (call detected) and auto-pause is on, the gate is forced closed (§4.1 last ¶).

**Stage ⑤ — sessionizer.** Consecutive same-class windows with gaps <1.5 s merge into one event: `duration_ms` = merged span, `confidence` = max window score, `burst_count` = count of distinct energy peaks inside the span (a 5-cough fit = one event, burst_count 5). Per-class cooldowns after an event closes (e.g. 10 s for `nose_blow` — one blow with re-grips is still one blow; 2 s for `cough`).

**Accuracy loop.** A golden WAV corpus lives in the repo: recorded positives per class plus hard negatives (speech with plosives, keyboard, chair squeak, kettle, packaging crinkle). The `cli` crate replays them through the *identical* pipeline (`cli classify file.wav`) in CI and emits a per-class precision/recall table per release; threshold defaults are derived from this corpus, and every model/threshold change shows its scorecard in the PR. Live "test detection" mode in settings shows the raw per-class scores as you sniff at the mic — used for personal calibration.

- Mic contention: capture uses shared mode; the app keeps working during calls but a settings toggle "auto-pause when another app uses the mic" (default **on**, detected via OS APIs) avoids logging your Zoom cough-track twice… and avoids the perception problem of listening during meetings.

### 4.2 Event store (SQLite, WAL)
```sql
CREATE TABLE events (
  uuid          TEXT PRIMARY KEY,       -- client_event_uuid, idempotency key
  event_type    TEXT NOT NULL,
  occurred_at   TEXT NOT NULL,          -- UTC ISO-8601
  tz_offset_min INTEGER NOT NULL,       -- local context for "morning vs night" charts
  duration_ms   INTEGER NOT NULL,
  confidence    REAL NOT NULL,
  burst_count   INTEGER NOT NULL DEFAULT 1,
  peak_dbfs        REAL NULL,           -- loudest 50 ms hop in the event
  mean_dbfs        REAL NULL,           -- power-domain mean across the event
  noise_floor_dbfs REAL NULL,           -- adaptive floor at onset; peak - floor = loudness vs the room
  model_version TEXT NOT NULL,
  source        TEXT NOT NULL,          -- desktop-mac | desktop-win | mobile-ios | mobile-android
  device_id     TEXT NOT NULL,          -- stable per-install UUID
  uploaded_at   TEXT NULL,              -- NULL = pending; never uploaded in offline-strict
  deleted       INTEGER NOT NULL DEFAULT 0, -- hard local removal (tombstone syncs as DELETE)
  false_positive_at TEXT NULL,          -- reported misdetection: retained, but never counted
  corrected_to      TEXT NULL,          -- recharacterized: still counts, as this class
  corrected_at      TEXT NULL,
  flag_updated_at   TEXT NULL,          -- last flag mutation, incl. clearing one
  flag_synced_at    TEXT NULL           -- pending when < flag_updated_at
);
CREATE INDEX idx_events_pending ON events (uploaded_at) WHERE uploaded_at IS NULL;
CREATE INDEX idx_events_day ON events (occurred_at);
CREATE INDEX idx_events_flag_pending ON events (flag_updated_at) WHERE flag_updated_at IS NOT NULL;
```

**Loudness.** The gate (stage ①) already computes per-hop RMS and an adaptive
noise floor; these three columns retain them so a quiet throat-clear and a
violent one are distinguishable. `mean_dbfs` is averaged in the **linear power**
domain over hops that tile the event without overlap — averaging dB values, or
averaging per-window means across overlapping patches, would both be wrong.

**False positive vs. correction.** A false positive is a misdetection: retained
(a health record should keep the fact that the classifier erred) but excluded
from every count, chart and score. A correction is a real event under the wrong
label: it keeps counting, as `corrected_to`. `flag_updated_at` is stamped by
every flag mutation *including clearing one*, which is what makes an undo
syncable — a queue keyed on "is currently flagged" cannot represent a cleared
flag and would leave the PHR marked forever.

### 4.3 Sync engine — three modes
| Mode | Behavior |
|---|---|
| **Auto-batch** (default) | Flush pending events when: 50 pending **or** 5 min elapsed **or** app quitting. Exponential backoff on failure (30 s → 30 min cap), jittered. |
| **Offline-first** | Same queue, but flush only on explicit "Sync now" or on a schedule (e.g. hourly) — for flaky/metered connections. |
| **Offline-strict** | **Never uploads.** No network I/O at all (feature-flagged out at the sync layer, not just a toggle). History lives entirely in the in-app UX. Switching back to a syncing mode prompts: "Upload 1,214 stored events? [Upload all] [Upload from today] [Keep local]". |

Batches are ≤500 events per request, idempotent via `uuid`; server replies per-event `accepted | duplicate | rejected` so a half-applied batch just re-sends (duplicates no-op). Local `deleted` tombstones for already-uploaded events sync as a small `DELETE` batch (same idempotency).

---

## 5. Classifier

### Phase A (ship first): pretrained YAMNet
- YAMNet (AudioSet, 521 classes, ~3.7 M params, MobileNet-v1 backbone) → ONNX. CPU inference is ~2–5 ms per 0.975 s frame on Apple Silicon / modern x86 — nothing else needed.
- Map native classes: Cough, Throat clearing, Sniff, Sneeze (+ Speech for the guard). Per-class thresholds calibrated on the developer's own environment; a "sensitivity" slider scales all thresholds.

### Phase B-lite (ships early): enrollment + few-shot prototype matching for `nose_blow`, `hawk`, `snort_suck`
The user **records their own examples** and the app recognizes those sounds later by similarity — no training loop at all.

- **Teach mode (settings window)**: a guided flow — "perform a hawk … again … again" — captures 10–20 deliberate examples per class (plus the same for sniffle/blow to *personalize* the native classes). Each example is stored as its **1024-d YAMNet embedding** (raw clip kept locally only if the user opts in, for later Phase-B-full training).
- **Negative enrollment**: same flow for "sounds that must NOT count" — your keyboard, your espresso machine, a plosive-heavy spoken phrase. False-positive ✕ clicks in history also add negatives automatically.
- **Runtime**: every gate-triggered window already produces an embedding (§4.1 stage ③, same forward pass). Classify by cosine similarity: nearest class prototype (mean of enrolled embeddings; k-NN over individual examples as a refinement) must exceed a similarity threshold **and** beat the nearest enrolled negative. Cost: a few dot products — microseconds.
- **Instant improvement loop**: a missed event or false positive is fixed by one tap ("add as example" / "✕ not me") — the prototype updates immediately, no retrain, no model file, no version bump. Enrollment data lives in SQLite next to events.
- **Personalization as a feature**: prototypes are of *your* sounds, so matching is biased toward you and against housemates/officemates polluting the log. (Honest limit: embeddings aren't a speaker-ID system — a very similar-sounding person can still match. §12.)
- **Deliberate-vs-spontaneous caveat**: performed hawks differ acoustically from reflexive ones (well documented for voluntary vs. reflex cough). Enrollment **bootstraps**; the passive labeling loop below refines with real events. Expect to add a handful of organic examples in week one before precision settles.

### Phase B-full (optional upgrade): fine-tuned head
- When organic labeled clips accumulate (≥100–200/class), train a tiny MLP head (1024→128→7) on YAMNet embeddings offline (Python notebook in the repo); the app loads `head.onnx` and bumps `model_version`. Adopt it **only if** it beats the prototype matcher on the held-out set — for a single-user app, prototypes may simply remain sufficient.
- **Labeling loop built into the app** (feeds both B-lite negatives/examples and the B-full training set): opt-in retention of gate-triggered 2 s clips **locally only**; history UI offers one-tap "was this a hawk / blow / suck / other / not-me?" labeling.
- Every event records `model_version` (`yamnet+proto@N` for B-lite, where N counts enrollment revisions) so charts can be re-interpreted after recognizer changes.

---

## 6. Desktop UX

### Tray / menubar
- Icon states: 🟢 listening · ⏸ paused · 📴 offline-strict (distinct glyph) · ⚠ mic permission missing / sync failing.
- Menu: today's counts by type ("👃 12 sniffles · 🤧 2 blows · 😤 1 hawk"), congestion-score sparkline (last 7 d), Pause 15 min / 1 h / until resumed, Mode selector, Sync now (+ pending count), Open History, Settings, Quit.
- macOS: `NSStatusItem` via `tray-icon`, LSUIElement (no Dock icon). Windows: notification-area icon; autostart via `auto-launch` (login item / Run registry key).

### Settings window (egui, created on demand — renderer torn down on close)
- PHR connection: server URL + API token (§7), patient binding, "Test connection".
- Input device picker + live level meter + "test detection" (make a sniff, see it classify).
- **Teach mode** (§5 Phase B-lite): guided enrollment of your own sniffs/hawks/blows (+ negatives like your keyboard); shows per-class example counts and lets you audition/delete individual examples.
- Sensitivity slider; per-class enable/disable; quiet hours; auto-pause-on-call toggle.
- Privacy panel: exactly what is stored/sent (event metadata only — show a sample JSON), clip-retention opt-in for labeling, "wipe local data".

### History window (egui + `egui_plot`; the offline-mode UX, but available in every mode)
- **Today**: timeline strip (dots colored by type, hover = time/confidence/duration), counts by type.
- **Trends**: 7/30-day stacked bars by type + congestion-score line; monitored-hours overlay so a quiet *unmonitored* day isn't a "good" day.
- **Event list**: filter by type/date; per-event ✕ false-positive (feeds Phase B, tombstones).
- **Export**: CSV / JSON of any date range — works in offline-strict mode (the escape hatch if the user never wants the network path at all).

---

## 7. PHR backend integration (`2025-website`)

> Tracked as a GitHub issue in the 2025-website repo; summary here so this spec stands alone.

### Auth
- The PHR API (`/api/phr/*`) is currently session-cookie only (`['web','auth']`). The repo already has a bearer path: `AuthenticateWebOrMcpRequest` accepts `Authorization: Bearer <token>` validated against sha256 `users.mcp_api_key` (issued from the account API-key page).
- **MVP**: mount the new respiratory-event routes with that middleware — zero new auth infrastructure; the desktop app stores the existing user API key (OS keychain via `keyring` crate — Keychain on macOS, Credential Manager on Windows).
- **Later (nice-to-have)**: scoped, per-device revocable tokens (`phr:respiratory-events:write` only), so a lost laptop doesn't hold a full-access key. Out of MVP scope.

### Endpoints (new)
```
POST   /api/phr/patients/{patient}/respiratory-events/batch
       { device_id, source, model_version, events: [ {uuid, event_type,
         occurred_at, tz_offset_min, duration_ms, confidence, burst_count,
         peak_dbfs?, mean_dbfs?, noise_floor_dbfs?}, ≤500 ] }
  →    { results: [ {uuid, status: accepted|duplicate|rejected, reason?} ] }

DELETE /api/phr/patients/{patient}/respiratory-events/batch   { uuids: [...] }
POST   /api/phr/patients/{patient}/respiratory-events/flag-batch
       { items: [ {uuid, false_positive, corrected_to}, ≤500 ] }
  →    { flagged, results: [ {uuid, status: flagged|not_found} ] }
GET    /api/phr/patients/{patient}/respiratory-events?from=&to=&type=&include_false_positives=
GET    /api/phr/patients/{patient}/respiratory-events/summary?from=&to=&bucket=day

GET    /api/phr/patients/{patient}/sinus-settings
PUT    /api/phr/patients/{patient}/sinus-settings
       { settings: {...}, updated_at, device_id }
  →    { applied, sinus_settings: { settings, updated_at, ... } }

GET    /api/phr/patients/{patient}/sinus-enrollments
POST   /api/phr/patients/{patient}/sinus-enrollments/batch    { enrollments: [ ..., ≤100 ] }
DELETE /api/phr/patients/{patient}/sinus-enrollments/batch    { uuids: [...] }
```

`flag-batch` is declarative — the current state, not a delta — so clearing a
flag is `false_positive: false` with a null `corrected_to`. `not_found` is
terminal client-side, so a flag on an event the server never accepted cannot
retry forever. Reads exclude false positives by default and bucket by
`COALESCE(corrected_to_event_type, event_type)`.

A server predating the settings/enrollment endpoints answers 404; the client
skips those steps rather than failing the flush.

### Table `phr_respiratory_events`
Mirrors the client store: `id`, `phr_patient_id` FK, `client_event_uuid` (unique per patient — the idempotency key), `event_type` (string, validated against the taxonomy), `occurred_at` datetime + `tz_offset_min`, `duration_ms`, `confidence`, `burst_count`, `peak_dbfs`/`mean_dbfs`/`noise_floor_dbfs`, `source`, `device_id`, `model_version`, `false_positive_at`, `corrected_to_event_type`, `corrected_at`, timestamps. Model uses `SerializesDatesAsLocal`, `BelongsTo` PhrPatient, follows existing `Phr*` conventions; patient authorization mirrors the other PHR controllers (owner/access grants).

### Tables `phr_sinus_settings`, `phr_sinus_enrollments`
`phr_sinus_settings`: one row per patient — a JSON `settings` document plus the client's `settings_updated_at` (the last-write-wins comparand, rejected if more than 5 minutes in the future) and the server's own `received_at`. Only detection-shaping keys sync (`sensitivity`, `quiet_start`, `quiet_end`); `server_url`, `patient_id`, `device_id`, `model_path` and `mode` are device-local — sync mode is per-machine network policy, and pulling `offline-strict` onto a second machine would silently disable its sync.

`phr_sinus_enrollments`: Teach-mode examples keyed on `client_enrollment_uuid`. Both `client_enrollment_uuid` (`BINARY(16)`) and `embedding` (`VARBINARY(16384)`) are raw binary — byte-identical to the device's SQLite BLOBs, so no float is reformatted in the round trip — carried over the wire as base64. A negative may set `source_event_uuid`, linking it to the event whose misdetection produced it.

See `docs/phr/sinus-sentinel.md` in the PHR repo for the full contract.

### Web UI (later, separate PR)
A "Respiratory" card on the PHR patient page: trend chart (reuse the vitals-trend chart pattern), correlation with medication start/stop dates. Not required for the desktop app to ship.

---

## 8. Privacy & security posture

1. **No raw audio persistence, no raw audio egress** — the network layer can only serialize the event and enrollment schemas; there is no code path that uploads audio. Opt-in labeling clips are local-only files, wiped by "wipe local data".
1a. **Derived embeddings do egress, and only to a PHR the user connected.** Teach-mode enrollments sync so a personalized detector follows the user between machines. These are opaque 1024-value YAMNet vectors from which audio cannot be reconstructed; they never leave the device in offline-first-without-sync or offline-strict. The per-event embeddings retained for false-positive reporting are a separate, strictly local set, pruned after 30 days. The mic-permission string and the Settings footer must both describe this accurately — "only event metadata is ever sent" is no longer a true statement of the system.
2. On-device inference only. No cloud ASR/classification of any kind.
3. macOS mic-permission usage string states plainly: "detects coughs/sniffles locally; no audio is recorded or sent." The OS mic indicator (orange dot) is on while listening. Pause, quiet hours, and the default OS low-power-mode policy release the microphone completely and open a fresh capture session on resume.
4. Event payloads contain no free text and no identifiers beyond `device_id` (random UUID) and the patient id.
5. Token in OS keychain, never in config files. TLS only (rustls); server URL pinned in settings.
6. SQLite DB in the platform app-data dir with user-only permissions. (SQLCipher considered; deferred — the threat model is a personal machine.)

---

## 9. Resource budget (acceptance targets — single process, no helper processes ever)
| State | CPU | RSS |
|---|---|---|
| Quiet room (gate closed) | <0.5 % of one core | <50 MB |
| Active classification burst | <8 % of one core | <70 MB |
| Settings/history window open | event-driven repaint only | +~30 MB, released on close |
Battery-relevant: the callback coalesces worker wakes into 50 ms gate hops; quiet
audio retains bounded raw pre-roll but performs no FFT or inference. There is no
busy polling or continuous render loop: worker status changes request egui
repaints, hidden history performs no database/chart work, and sync sleeps until a
queue signal or exact flush/quiet-hours deadline. ONNX CPU fallback is
single-threaded; macOS prefers CoreML on CPU + Neural Engine. CI drives the
complete streaming pipeline through a 10-minute quiet-room soak.

---

## 10. Mobile companion (stretch, same core)

- `crates/core` (pipeline, store, sync, offline-strict) compiles for iOS/Android regardless of shell. **Shell chosen at M6** from three candidates:
  1. **Tauri 2 mobile shell** — webview UI. On mobile a webview is idiomatic and WKWebView is cheap; the resource argument against desktop webviews doesn't transfer. Lowest-effort path; UI written in web tech for mobile only.
  2. **egui via winit on Android** (works today; iOS support is experimental) — maximum code share with the desktop window.
  3. **Slint** — native-rendered declarative UI, royalty-free license, Android supported / iOS maturing.
- **Foreground-only capture v1**, with **keep-awake** (`FLAG_KEEP_SCREEN_ON` / `isIdleTimerDisabled`, or the shell's plugin) holding the screen on while the session runs (matches the "keeps phone awake while running" idea) — a deliberate dodge of iOS background-audio entitlement review and Android foreground-service policy for the personal-use MVP.
  - Android later: `FOREGROUND_SERVICE_MICROPHONE` service → true background monitoring is feasible.
  - iOS later: `audio` background mode technically works but is App Store-fragile; personal sideload (free dev profile / TestFlight) doesn't care.
- Sync/queue/offline-strict behave identically (same crate). `source` = `mobile-ios|mobile-android`.
- Distribution: personal use only — sideload iOS via Xcode/TestFlight; Android via direct APK. No store submission planned.
- Explicit non-goal: always-on wearable-style monitoring on phone. Desktop is the primary sensor; mobile is for on-the-go sessions (commute, bedside).

## 11. Repo & delivery plan

**Repo:** `sinus-sentinel` (name confirmed 2026-07-13; github.com/bherila/sinus-sentinel, private):
```
sinus-sentinel/
├── crates/
│   ├── core/          # audio, gate, dsp, inference, sessionizer, store, sync (no UI deps)
│   └── cli/           # headless: run pipeline on wav files; bench; calibrate thresholds
├── apps/desktop/      # eframe app: tray-icon + winit + egui settings/history window
├── apps/mobile/       # M6; shell TBD per §10 (Tauri 2 / egui-android / Slint)
├── model/             # yamnet.onnx, head.onnx, labels.json, training notebook
├── testdata/          # golden WAV corpus (positives per class + hard negatives)
├── docs/SPEC.md       # this document, kept current
└── .github/workflows/ # fmt+clippy+test+golden-corpus scorecard; cargo-dist releases
```
CI: `cargo fmt --check`, `clippy -D warnings`, `cargo test` (core runs the golden-wav corpus and emits the precision/recall scorecard), `cargo-dist` release jobs for macOS (signed + notarized) and Windows. Pure-cargo toolchain — no Node.

### Milestones
| | Deliverable | Acceptance |
|---|---|---|
| **M0** | Repo bootstrap, CI, tray skeleton (tray-icon + winit) w/ pause toggle | icon runs on both OSes |
| **M1** | Audio pipeline + YAMNet 4-class detection, CLI harness + golden corpus | golden-wav scorecard in CI; live sniff detected E2E |
| **M2** | SQLite store + sessionizer + history window (egui_plot) | events survive restart; false-positive ✕ works |
| **M3** | Sync engine + backend endpoints live (2025-website issue) | batch upload idempotent; backoff verified by yanking Wi-Fi |
| **M4** | Offline-strict mode + export; settings polish; signed builds | offline-strict provably makes zero network calls (test: DNS-blackhole run) |
| **M5** | Teach mode + Phase B-lite prototype recognition (hawk / nose_blow / snort_suck) + labeling loop | enrolled classes detected live; ≥80 % precision on a held-out self-labeled set; B-full head only if it beats prototypes |
| **M6** | Mobile shell (path chosen per §10) w/ keep-awake (stretch) | session on phone logs to same PHR patient |

## 12. Risks & mitigations
- **False positives during speech** → YAMNet speech-guard suppression (§4.1); per-class thresholds; ✕ feedback loop.
- **AudioSet coverage gaps for hawk/suck** → enrollment + prototype matching (Phase B-lite) rather than hoping; sniffle/cough/throat-clear alone already produce a useful congestion signal in v1.
- **Enrolled prototypes aren't speaker-ID** → personalization biases toward the enrolled user but a similar-sounding housemate can still match; mitigations: negatives enrollment, ✕ feedback, and honesty in the UI ("events near others may include them").
- **Deliberate enrollment samples ≠ spontaneous events** → treat enrollment as bootstrap; refine with organic one-tap examples during week one (§5).
- **Quiet events (sniffle at 3 m)** → detection radius is honestly ~arm's-length-to-desk; document it; mobile companion covers away-from-desk.
- **"Monitored hours" bias** (fewer events because app was off) → history charts always normalize per monitored-hour (§2, §6).
- **macOS notarization friction for mic apps** → standard entitlements + usage string; budgeted in M4, not an afterthought.
- **Scope creep toward medical claims** → taxonomy + score stay descriptive; no thresholds framed as diagnoses.

## 13. Open questions
1. Repo name — `sinus-sentinel` acceptable?
2. Should the PHR web UI card (§7) land with M3 or wait for real data volume?
3. Sneeze: log by default or keep off (allergy noise vs sinus signal)?
4. Quiet-hours default (e.g. suppress 23:00–07:00 when the machine is on but you're likely elsewhere/asleep)?
5. Scoped device tokens: worth doing before mobile (which multiplies stored-token copies)?
6. Mobile shell (§10): decide Tauri 2 vs egui-android vs Slint at M6 — default lean is Tauri 2 for lowest effort, since webview economics on mobile differ from desktop.
