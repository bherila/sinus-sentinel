# Model artifacts

Sinus Sentinel runs [YAMNet](https://tfhub.dev/google/yamnet/1) (AudioSet,
MobileNet-v1 backbone, ~3.7 M params) on-device via ONNX Runtime. **No binary is
committed** (size + provenance); everything builds and tests without it. When the
model file is absent, `YamnetOnnx::load` returns `Error::ModelUnavailable` and the
app surfaces a "model missing" state — it never panics (SPEC §4 stage ③).

Expected location: `model/yamnet.onnx` (override via the loader path). The `onnx`
Cargo feature is **off by default** and uses `ort`'s `load-dynamic`, so building it
requires no network download; at runtime it loads the ONNX Runtime shared library
(`ORT_DYLIB_PATH` or a system install).

## Contract (what the code expects)

Our log-mel frontend (`sinus_core::mel`, SPEC §4.1 stage ②) computes features and
feeds YAMNet the **features** variant — not raw waveform:

| Tensor | Name (default, overridable via `TensorNames`) | Shape | Notes |
|---|---|---|---|
| input | `input` | `[1, 96, 64]` f32 | 0.96 s log-mel patch, 96 frames × 64 mel bands, `log(mel + 0.001)`, no normalization |
| output | `scores` | `[1, 521]` or `[N, 521]` f32 | AudioSet class scores; per-frame matrices are averaged |
| output | `embeddings` | `[1, 1024]` or `[N, 1024]` f32 | 1024-d embedding for the prototype matcher (SPEC §5) |

AudioSet indices consumed by the native head (`AudiosetMap`, overridable):
`Speech=0, Cough=42, Throat clearing=43, Sneeze=44, Sniff=45`.

If your export renumbers classes or names tensors differently, pass a custom
`AudiosetMap` / `TensorNames` — no code change required.

## Converting YAMNet → ONNX

YAMNet ships as a TensorFlow SavedModel / TF-Hub module. Convert with
[`tf2onnx`](https://github.com/onnx/tensorflow-onnx):

```bash
python -m pip install tensorflow tensorflow-hub tf2onnx onnx

# 1. Export the TF-Hub module to a SavedModel that takes a log-mel patch and
#    returns (scores, embeddings). See model/export_yamnet.py (write to taste):
#    - input:  float32 [1, 96, 64] log-mel patch
#    - outputs: scores [1, 521], embeddings [1, 1024]
python model/export_yamnet.py --out saved_model/

# 2. Convert SavedModel → ONNX (opset 17).
python -m tf2onnx.convert \
  --saved-model saved_model/ \
  --output model/yamnet.onnx \
  --opset 17

# 3. Sanity-check the shapes.
python -c "import onnx; m=onnx.load('model/yamnet.onnx'); print(m.graph.input, m.graph.output)"
```

The stock TF-Hub YAMNet takes a 1-D waveform and computes mel internally. Because
this app computes the *exact* YAMNet log-mel itself (to keep the CLI/tests and the
live app on one code path), export the **features-in** variant above, or adapt
`TensorNames`/pre-processing to whatever your export exposes.

## `head.onnx` (Phase B-full, optional — SPEC §5)

Not required for v1. When enough organic labeled clips accumulate, a tiny MLP head
(`1024 → 128 → 7`) can be trained offline and loaded as `model/head.onnx`, bumping
`model_version`. Until then, custom classes use the zero-training prototype matcher
(`sinus_core::classify::proto`).

## `labels.json`

Optional map of AudioSet index → human label, for debugging `cli classify`
output. Not consumed by the runtime.
