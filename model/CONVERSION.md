# Converting YAMNet -> `model/yamnet.onnx`

This is the reproducible path from the published YAMNet weights to the
`model/yamnet.onnx` this app loads (`crates/core/src/classify/yamnet.rs`). It
supersedes the TF-Hub/`export_yamnet.py` sketch in `model/README.md` /
`model/fetch.sh`: that route (raw waveform -> internal `tf.signal.stft` ->
scores/embeddings) reliably fails to convert with `tf2onnx` because of the
`tf.signal.stft` / `tf.signal.frame` ops. This app computes the log-mel
frontend itself in Rust (SPEC §4.1 stage ②), so we only need the
**patches-in** part of the graph, and there's a route that avoids the
STFT ops entirely — see "Why patches-in, not waveform-in" below.

`model/yamnet.onnx` itself is **not committed** (gitignored via `model/*.onnx`,
size + provenance) — regenerate it locally with `model/convert_yamnet.py`.

## Result: patches-input ONNX (not the waveform-input fallback)

The conversion below succeeded on the first architecture tried — patches in,
scores + embeddings out — so there was no need to fall back to a
waveform-input model. **The Rust side loads a patches-input model**, matching
the frontend it already computes itself (SPEC §4.1/§5); no waveform
preprocessing needs to happen inside the ONNX graph.

## Tensor contract (verified in the exported graph, matches `TensorNames::default()`)

| Tensor | Name | Shape | Dtype |
|---|---|---|---|
| input | `input` | `[N, 96, 64]` (N = batch of patches, dynamic) | `float32` |
| output | `scores` | `[N, 521]` | `float32` |
| output | `embeddings` | `[N, 1024]` | `float32` |

- `N` is a dynamic dimension (any batch size, including `N=1` for a single
  0.96 s patch, which is what `YamnetOnnx::embed` in
  `crates/core/src/classify/yamnet.rs` sends today).
- `scores` are raw sigmoid outputs (one score per class, not softmax —
  matches the original YAMNet; multiple classes can legitimately score high
  at once).
- `embeddings` is the pre-logit `GlobalAveragePooling2D` output (1024-d),
  used unchanged by both the native-class scores and the Phase B-lite
  prototype matcher (SPEC §5).
- Class index -> name mapping is `model/yamnet_class_map.csv` (521 rows;
  provenance below). `crates/core/src/classify/embed.rs`'s
  `AUDIOSET_CLASSES`/native index constants (`Speech=0, Cough=42,
  Throat clearing=43, Sneeze=44, Sniff=45`) are this same official ordering.

## Why patches-in, not waveform-in

The TF-Hub `google/yamnet/1` SavedModel takes a raw 16 kHz waveform and
computes the log-mel spectrogram internally using `tf.signal.stft` /
`tf.signal.frame`. `tf2onnx` frequently cannot lower those ops (missing
RFFT/STFT op support), so converting that SavedModel directly is the known
unreliable route.

Instead this script uses the **Keras reimplementation** in
[`tensorflow/models`](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
(`yamnet.py`/`params.py`/`features.py`): it builds the MobileNet-v1 backbone +
classifier head as a plain Keras functional model with an explicit `(N, 96,
64)` patches input, loads the officially published `yamnet.h5` weights into
it, and exports *that* — the waveform/STFT frontend (`features.py`) is never
part of the exported graph. All ops in the exported graph are
Conv2D/DepthwiseConv2D/BatchNorm/ReLU/Dense/Sigmoid/GlobalAveragePooling —
all natively `tf2onnx`- and `onnxruntime`-friendly. See
`model/convert_yamnet.py`'s module docstring and `build_patches_model()` for
the exact construction.

### Gotcha hit during conversion: by-name weight loading silently drops the classifier head

The published `yamnet.h5` was saved from a model whose classifier
`Dense`/`Activation` layers are named `logits`/`prediction`. The *current*
`tensorflow/models` `yamnet.py` builds those two layers without an explicit
`name=`, so Keras auto-names them `dense`/`activation`. Backbone layers
(`layer1`...`layer14`, explicitly named in `yamnet.py`) match the H5 file by
name either way, but `model.load_weights('yamnet.h5', by_name=True)` silently
**skips** the mismatched `dense`/`activation` layers (leaves them at random
init) while everything upstream loads fine — the embeddings come out looking
perfectly plausible while the class scores are pure noise. This is easy to
miss because nothing raises an error.

Fix: load weights **positionally** — `model.load_weights('yamnet.h5')`
(default `by_name=False`) — which matches by traversal order, not name, and
is what `build_patches_model()` does. `convert_yamnet.py` verifies this is
correct by also building the unmodified, upstream `yamnet_frames_model`
(waveform-in) as an oracle and diffing outputs on the same input end-to-end
(see the "Cross-check vs unmodified yamnet_frames_model" section of the
validation transcript below) — before this fix that cross-check failed with
a max abs diff around 0.95 on `scores` (only `embeddings` matched); after the
fix both match to ~1e-6/1e-7.

## Class map provenance

`model/yamnet_class_map.csv` (521 AudioSet classes, committed — the Rust
runtime reads it for label lookups) is fetched verbatim from
`tensorflow/models`, `research/audioset/yamnet/yamnet_class_map.csv`
(same commit/ref as `yamnet.py` — see the URLs in `convert_yamnet.py`). It is
the canonical index-ordering YAMNet's Dense classifier head was trained
against; do not regenerate or hand-edit it.

## Pinned dependencies

Convert on macOS arm64 with a **Python 3.12** venv created *outside* the
repo (a 3.14 system Python will not resolve these pins):

```bash
/opt/homebrew/bin/python3.12 -m venv /path/outside/repo/yamnet-venv
source /path/outside/repo/yamnet-venv/bin/activate
pip install \
  "tensorflow==2.16.2" \
  "tf-keras==2.16.0" \
  "tf2onnx==1.16.1" \
  "onnx==1.16.2" \
  "onnxruntime==1.18.1" \
  "numpy==1.26.4"
```

- `tensorflow==2.16.2` defaults to Keras 3; `tensorflow/models`' `yamnet.py`
  imports the standalone `tf_keras` (Keras 2 API) package explicitly, so
  `tf-keras==2.16.0` must also be installed even though nothing imports it
  directly from this repo's script.
- Plain `tensorflow` (not `tensorflow-macos`) works fine on this Apple
  Silicon host at 2.16.2; fall back to `tensorflow-macos` only if pip
  resolution fights on your machine (see AGENTS.md-adjacent guidance in the
  task brief).
- `numpy==1.26.4` — TensorFlow 2.16 requires `numpy<2`.

## Running the conversion

```bash
source /path/outside/repo/yamnet-venv/bin/activate
python model/convert_yamnet.py
# optional: --output <path>  --opset <int, default 17>
#           --cache-dir <dir, default ~/.cache/sinus-sentinel-yamnet>
```

The script:
1. Downloads `yamnet.py`/`params.py`/`features.py`/`yamnet_class_map.csv` and
   the published `yamnet.h5` weights from `tensorflow/models` /
   `storage.googleapis.com/audioset` into `--cache-dir` (outside the repo;
   re-run is a no-op cache hit).
2. Copies the fetched class map to `model/yamnet_class_map.csv` (committed).
3. Builds the patches-input Keras sub-model and loads the published weights
   positionally (see gotcha above).
4. Builds the unmodified upstream `yamnet_frames_model` (waveform-in) purely
   as a cross-validation oracle (not exported).
5. Exports the patches-input model to ONNX via `tf2onnx.convert.from_keras`
   (opset 17), asserting the input/output tensor names match the Rust
   contract.
6. Validates: runs 3 deterministic inputs (seeded random patch, all-zeros,
   and a patch computed from a synthetic 440 Hz/16 kHz sine via the
   reference `features.py`) through both the Keras model and the exported
   ONNX (`onnxruntime`), asserts scores/embeddings agree to <1e-4 max abs
   diff, and prints the top-5 AudioSet classes for the sine input from both
   backends to confirm identical ordering. Also cross-checks the patches
   sub-model's sine-input output against the unmodified `yamnet_frames_model`
   oracle end-to-end.

## Validation transcript (2026-07-13, this exact run)

```
TensorFlow 2.16.2
Fetching tensorflow/models research/audioset/yamnet reference source:
  downloading (1/3): https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet.py
  downloading (1/3): https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/params.py
  downloading (1/3): https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/features.py
  downloading (1/3): https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv
Fetching published yamnet.h5 weights:
  downloading (1/3): https://storage.googleapis.com/audioset/yamnet.h5
  yamnet.h5 sha256: 13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744
Wrote model/yamnet_class_map.csv (committed; Rust needs it at runtime)

Building patches-input Keras model and loading published weights...
Building unmodified reference yamnet_frames_model for cross-validation...

Exporting to ONNX (opset 17) -> model/yamnet.onnx
  input:  'input'      shape [N, 96, 64] float32
  output: 'scores'     shape [N, 521]    float32
  output: 'embeddings' shape [N, 1024]   float32
  file size: 14,938,132 bytes
  sha256: 9e58907dea8376a63dc6ee281a54fa1eea37d61ac9ec0d3521d05b0cda5639bb

=== Keras vs ONNX validation (3 deterministic inputs) ===
  input[0] = random(seed=20260713)
  input[1] = zeros
  input[2] = sine(440Hz,16kHz)
  scores shape: keras=(3, 521) onnx=(3, 521)
  embeddings shape: keras=(3, 1024) onnx=(3, 1024)
  max abs diff (scores):     8.494e-07
  max abs diff (embeddings): 6.676e-06
  tolerance 1e-04: PASS

  Top-5 AudioSet classes for the sine(440Hz) input:
    rank idx   keras score   onnx score    name
    1    383   0.990763      0.990763      Telephone
    2    387   0.985571      0.985571      Dial tone
    3    388   0.972617      0.972618      Busy signal
    4    382   0.929680      0.929680      Alarm
    5    512   0.147160      0.147159      Sidetone
  top-5 ordering identical: PASS

  Cross-check vs unmodified yamnet_frames_model (waveform-in oracle):
    max abs diff (scores):     1.937e-07
    max abs diff (embeddings): 2.384e-06
    tolerance 1e-04: PASS

Validation PASSED.
```

(A pure 440 Hz tone landing on Telephone/Dial tone/Busy signal/Alarm/Sidetone
is the expected qualitative sanity check — these are exactly AudioSet's
narrow-band steady-tone classes.)

## Remaining steps for the Rust side

None required for the tensor contract — `crates/core/src/classify/yamnet.rs`'s
`TensorNames::default()` already expects input `"input"` and outputs
`"scores"`/`"embeddings"`, which is what this export produces. To use the
model:

1. Run `python model/convert_yamnet.py` locally (see above) to produce
   `model/yamnet.onnx` (gitignored, ~14.9 MB, not committed).
2. Build with the `onnx` Cargo feature enabled; `ort` loads the ONNX Runtime
   shared library at runtime (`ORT_DYLIB_PATH` or a system install) —
   nothing repo-side to configure.
3. `YamnetOnnx::load("model/yamnet.onnx")` (or wherever the app is configured
   to look) works with zero code changes.
