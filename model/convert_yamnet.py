#!/usr/bin/env python3
"""Convert YAMNet (AudioSet, MobileNet-v1) to a patches-in ONNX model.

End-to-end: fetch the reference Keras source + published weights from
tensorflow/models -> build a patches-input sub-model (skip the waveform/STFT
frontend entirely, since sinus-sentinel computes the log-mel frontend itself in
Rust, see docs/SPEC.md §4.1/§5) -> export to ONNX -> validate Keras vs ONNX
agreement on deterministic inputs.

Why the Keras-reimplementation route and not the TF-Hub SavedModel:
The TF-Hub `google/yamnet/1` SavedModel takes a raw waveform and computes the
log-mel spectrogram internally with `tf.signal.stft` / `tf.signal.frame` ops that
`tf2onnx` frequently cannot convert (unsupported RFFT/FrameOp lowering). The
reference Keras reimplementation in tensorflow/models (research/audioset/yamnet)
lets us build *just* the patches -> (scores, embeddings) graph (conv/dense/bn only,
all tf2onnx-friendly) and load the published `yamnet.h5` weights into it directly,
skipping the frontend altogether.

Usage:
    python model/convert_yamnet.py [--output model/yamnet.onnx] [--opset 17]
                                    [--cache-dir ~/.cache/sinus-sentinel-yamnet]

Pinned dependencies (see model/CONVERSION.md for the exact `pip install` line and
the full validation transcript):
    tensorflow==2.16.2  tf-keras==2.16.0  tf2onnx==1.16.1
    onnx==1.16.2  onnxruntime==1.18.1  numpy==1.26.4
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np

TF_MODELS_RAW_BASE = (
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet"
)
YAMNET_WEIGHTS_URL = "https://storage.googleapis.com/audioset/yamnet.h5"

SOURCE_FILES = ["yamnet.py", "params.py", "features.py", "yamnet_class_map.csv"]

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = MODEL_DIR / "yamnet.onnx"
CLASS_MAP_OUTPUT = MODEL_DIR / "yamnet_class_map.csv"

# Tensor contract expected by crates/core/src/classify/yamnet.rs
# (`TensorNames::default()`). Keep these in sync with that file.
INPUT_NAME = "input"
SCORES_NAME = "scores"
EMBEDDINGS_NAME = "embeddings"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, *, retries: int = 3) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"  downloading ({attempt}/{retries}): {url}")
            with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as out:
                shutil.copyfileobj(resp, out)
            return
        except Exception as exc:  # noqa: BLE001 - retry loop, re-raised below
            last_err = exc
            if dest.exists():
                dest.unlink()
    raise RuntimeError(f"failed to download {url} after {retries} attempts") from last_err


def fetch_reference_sources(cache_dir: Path) -> Path:
    """Download the tensorflow/models yamnet reference source + weights."""
    src_dir = cache_dir / "yamnet-src"
    src_dir.mkdir(parents=True, exist_ok=True)
    print("Fetching tensorflow/models research/audioset/yamnet reference source:")
    for name in SOURCE_FILES:
        download(f"{TF_MODELS_RAW_BASE}/{name}", src_dir / name)
    print("Fetching published yamnet.h5 weights:")
    weights_path = src_dir / "yamnet.h5"
    download(YAMNET_WEIGHTS_URL, weights_path)
    print(f"  yamnet.h5 sha256: {sha256_of(weights_path)}")
    return src_dir


def import_module_from_path(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_patches_model(src_dir: Path):
    """Build the patches [N, 96, 64] -> (scores [N, 521], embeddings [N, 1024])
    sub-model, and load the published weights into it.

    Loads weights *positionally* (`load_weights(path)`, no `by_name=True`):
    the published `yamnet.h5` names its classifier layers `logits`/`prediction`,
    but the current tensorflow/models `yamnet.py` builds them unnamed (Keras
    auto-names them `dense`/`activation`). By-name loading silently skips the
    mismatched classifier layers (leaving it randomly initialized) while the
    backbone layers 1-14 *do* match by name and load fine -- a subtle,
    easy-to-miss bug where embeddings look correct but scores are garbage.
    Positional loading matches by traversal order instead of name and is
    verified byte-for-byte against the reference `yamnet_frames_model` below.
    """
    import tensorflow as tf
    from tf_keras import Model, layers

    # yamnet.py does `import features as features_lib` (plain, not relative), so
    # the fetched source dir must be on sys.path before it's imported.
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    params_lib = import_module_from_path("yamnet_params", src_dir / "params.py")
    yamnet_lib = import_module_from_path("yamnet_model", src_dir / "yamnet.py")

    params = params_lib.Params()
    assert (params.patch_frames, params.patch_bands) == (96, 64), (
        "yamnet.py params changed patch shape away from (96, 64) -- "
        "sinus-sentinel's Rust frontend assumes (96, 64), see docs/SPEC.md §4.1"
    )

    patches_in = layers.Input(
        batch_shape=(None, params.patch_frames, params.patch_bands),
        dtype=tf.float32,
        name=INPUT_NAME,
    )
    predictions, embeddings = yamnet_lib.yamnet(patches_in, params)
    scores = layers.Activation("linear", name=SCORES_NAME)(predictions)
    embeddings_out = layers.Activation("linear", name=EMBEDDINGS_NAME)(embeddings)
    model = Model(name="yamnet_patches", inputs=patches_in, outputs=[scores, embeddings_out])
    model.load_weights(str(src_dir / "yamnet.h5"))
    return model, params, yamnet_lib, params_lib


def build_reference_frames_model(src_dir: Path, params, yamnet_lib):
    """Build the *unmodified* waveform-in reference model (yamnet_frames_model)
    for cross-validation -- this is the model tensorflow/models ships and its
    weight loading is unambiguous (no name collisions), so it's our oracle."""
    model = yamnet_lib.yamnet_frames_model(params)
    model.load_weights(str(src_dir / "yamnet.h5"))
    return model


def export_onnx(model, output_path: Path, opset: int) -> None:
    import tensorflow as tf
    import tf2onnx

    spec = (tf.TensorSpec((None, 96, 64), tf.float32, name=INPUT_NAME),)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=opset, output_path=str(output_path)
    )
    in_names = [n.name for n in model_proto.graph.input]
    out_names = [n.name for n in model_proto.graph.output]
    assert in_names == [INPUT_NAME], f"unexpected ONNX input names: {in_names}"
    assert out_names == [SCORES_NAME, EMBEDDINGS_NAME], f"unexpected ONNX output names: {out_names}"


def synthetic_sine_patch(src_dir: Path, params, features_lib) -> np.ndarray:
    """One (96, 64) log-mel patch computed via the *reference* features.py from
    a synthetic 16 kHz sine, for a real (non-random, non-zero) sanity check."""
    import tensorflow as tf

    sample_rate = 16000
    freq_hz = 440.0
    duration_s = 1.0
    t = np.arange(int(sample_rate * duration_s), dtype=np.float32) / sample_rate
    waveform = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)

    waveform_padded = features_lib.pad_waveform(tf.constant(waveform), params)
    _, patches = features_lib.waveform_to_log_mel_spectrogram_patches(waveform_padded, params)
    patches_np = patches.numpy()
    assert patches_np.shape[1:] == (96, 64), patches_np.shape
    return patches_np[0:1]  # first patch, shape (1, 96, 64)


def load_class_names(class_map_csv: Path) -> list[str]:
    import csv

    with class_map_csv.open() as f:
        reader = csv.reader(f)
        next(reader)  # header
        return [row[2] for row in reader]


def run_validation(model, ref_frames_model, params, yamnet_lib, features_lib, src_dir: Path, onnx_path: Path) -> bool:
    import onnxruntime as ort

    class_names = load_class_names(src_dir / "yamnet_class_map.csv")

    rng = np.random.default_rng(20260713)
    random_patch = rng.standard_normal((96, 64)).astype(np.float32)
    zero_patch = np.zeros((96, 64), dtype=np.float32)
    sine_patch = synthetic_sine_patch(src_dir, params, features_lib)[0]

    batch = np.stack([random_patch, zero_patch, sine_patch], axis=0)
    labels = ["random(seed=20260713)", "zeros", "sine(440Hz,16kHz)"]

    keras_scores, keras_emb = model.predict(batch, steps=1, verbose=0)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_scores, onnx_emb = sess.run(None, {INPUT_NAME: batch})

    scores_diff = float(np.max(np.abs(keras_scores - onnx_scores)))
    emb_diff = float(np.max(np.abs(keras_emb - onnx_emb)))

    print()
    print("=== Keras vs ONNX validation (3 deterministic inputs) ===")
    for i, label in enumerate(labels):
        print(f"  input[{i}] = {label}")
    print(f"  scores shape: keras={keras_scores.shape} onnx={onnx_scores.shape}")
    print(f"  embeddings shape: keras={keras_emb.shape} onnx={onnx_emb.shape}")
    print(f"  max abs diff (scores):     {scores_diff:.3e}")
    print(f"  max abs diff (embeddings): {emb_diff:.3e}")

    tolerance = 1e-4
    ok = scores_diff < tolerance and emb_diff < tolerance
    print(f"  tolerance {tolerance:.0e}: {'PASS' if ok else 'FAIL'}")

    # Top-5 for the sine input (index 2), both backends, must agree in order.
    sine_idx = 2
    keras_top5 = np.argsort(-keras_scores[sine_idx])[:5]
    onnx_top5 = np.argsort(-onnx_scores[sine_idx])[:5]
    order_matches = np.array_equal(keras_top5, onnx_top5)
    print()
    print("  Top-5 AudioSet classes for the sine(440Hz) input:")
    print(f"    {'rank':<5}{'idx':<6}{'keras score':<14}{'onnx score':<14}name")
    for rank in range(5):
        idx = int(keras_top5[rank])
        print(
            f"    {rank + 1:<5}{idx:<6}{keras_scores[sine_idx, idx]:<14.6f}"
            f"{onnx_scores[sine_idx, idx]:<14.6f}{class_names[idx]}"
        )
    print(f"  top-5 ordering identical: {'PASS' if order_matches else 'FAIL'}")

    # Cross-check the patches sub-model against the unmodified reference
    # yamnet_frames_model on the same sine waveform end-to-end (frontend
    # included), to catch any drift between the two model constructions.
    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    ref_preds, ref_emb, _ = ref_frames_model.predict(waveform, steps=1, verbose=0)
    ref_diff_scores = float(np.max(np.abs(ref_preds[0] - keras_scores[sine_idx])))
    ref_diff_emb = float(np.max(np.abs(ref_emb[0] - keras_emb[sine_idx])))
    print()
    print("  Cross-check vs unmodified yamnet_frames_model (waveform-in oracle):")
    print(f"    max abs diff (scores):     {ref_diff_scores:.3e}")
    print(f"    max abs diff (embeddings): {ref_diff_emb:.3e}")
    ref_ok = ref_diff_scores < tolerance and ref_diff_emb < tolerance
    print(f"    tolerance {tolerance:.0e}: {'PASS' if ref_ok else 'FAIL'}")

    return ok and order_matches and ref_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "sinus-sentinel-yamnet",
        help="download cache for reference source + weights (outside the repo)",
    )
    args = parser.parse_args()

    import tensorflow as tf

    print(f"TensorFlow {tf.__version__}")
    src_dir = fetch_reference_sources(args.cache_dir)

    shutil.copyfile(src_dir / "yamnet_class_map.csv", CLASS_MAP_OUTPUT)
    print(f"Wrote {CLASS_MAP_OUTPUT} (committed; Rust needs it at runtime)")

    print("\nBuilding patches-input Keras model and loading published weights...")
    model, params, yamnet_lib, params_lib = build_patches_model(src_dir)

    print("Building unmodified reference yamnet_frames_model for cross-validation...")
    features_lib = import_module_from_path("yamnet_features", src_dir / "features.py")
    ref_frames_model = build_reference_frames_model(src_dir, params, yamnet_lib)

    print(f"\nExporting to ONNX (opset {args.opset}) -> {args.output}")
    export_onnx(model, args.output, args.opset)
    print(f"  input:  '{INPUT_NAME}'      shape [N, 96, 64] float32")
    print(f"  output: '{SCORES_NAME}'     shape [N, 521]    float32")
    print(f"  output: '{EMBEDDINGS_NAME}' shape [N, 1024]   float32")
    onnx_size = args.output.stat().st_size
    print(f"  file size: {onnx_size:,} bytes")
    print(f"  sha256: {sha256_of(args.output)}")

    passed = run_validation(model, ref_frames_model, params, yamnet_lib, features_lib, src_dir, args.output)

    print()
    if not passed:
        print("VALIDATION FAILED -- do not ship this model.", file=sys.stderr)
        return 1
    print("Validation PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
