#!/usr/bin/env python3
"""Convert YAMNet to a patches-in Core ML model for the Apple SwiftUI shell.

Reuses convert_yamnet.py's builders — the same Keras patches-input model and
published yamnet.h5 weights that produce model/yamnet.onnx — so the two
exports cannot drift: one graph, two serializations. See model/CONVERSION.md
for why patches-in (Rust computes the log-mel frontend itself, SPEC §4.1).

Contract expected by apps/apple/Sources/Platform/ModelRunners.swift
(CoreMLYamnetRunner): input `input` float32 [1, 96, 64], outputs `scores`
[1, 521] and `embeddings` [1, 1024].

Usage:
    python model/convert_yamnet_coreml.py [--output model/yamnet.mlpackage]
                                          [--cache-dir ~/.cache/sinus-sentinel-yamnet]

Then compile and place where the Apple build script picks it up:
    xcrun coremlcompiler compile model/yamnet.mlpackage apps/apple/Resources/

Dependencies: the convert_yamnet.py pins plus coremltools (macOS only —
validation runs a real Core ML prediction).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR))

from convert_yamnet import (  # noqa: E402
    EMBEDDINGS_NAME,
    INPUT_NAME,
    SCORES_NAME,
    build_patches_model,
    fetch_reference_sources,
    import_module_from_path,
    sha256_of,
    synthetic_sine_patch,
)

DEFAULT_OUTPUT = MODEL_DIR / "yamnet.mlpackage"


def export_coreml(model, output_path: Path):
    import coremltools as ct
    import tensorflow as tf

    # The model is built with tf-keras (Keras 2), whose Functional class
    # coremltools does not recognize as a tf.keras.Model — hand it a concrete
    # function instead, which both sides agree on.
    @tf.function
    def patches_fn(patches):
        scores, embeddings = model(patches, training=False)
        return {SCORES_NAME: scores, EMBEDDINGS_NAME: embeddings}

    concrete = patches_fn.get_concrete_function(
        tf.TensorSpec((1, 96, 64), tf.float32, name=INPUT_NAME)
    )

    mlmodel = ct.convert(
        [concrete],
        source="tensorflow",
        inputs=[ct.TensorType(name=INPUT_NAME, shape=(1, 96, 64), dtype=np.float32)],
        convert_to="mlprogram",
        # FLOAT32, not the FP16 default: embeddings feed cosine similarity
        # against enrolled prototypes that were (and on other machines still
        # are) computed by the float32 ONNX path — keep the two backends
        # numerically interchangeable rather than shipping a smaller file.
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
    )
    # The converter names outputs after internal graph tensors; rename to the
    # contract CoreMLYamnetRunner expects, identifying each by its width
    # (521 classes vs the 1024-d embedding) rather than trusting output order.
    spec = mlmodel.get_spec()
    for out in spec.description.output:
        width = out.type.multiArrayType.shape[-1]
        target = {521: SCORES_NAME, 1024: EMBEDDINGS_NAME}.get(width)
        assert target is not None, f"unexpected output width {width}"
        if out.name != target:
            ct.utils.rename_feature(spec, out.name, target)
    mlmodel = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)

    if output_path.exists():
        shutil.rmtree(output_path)
    mlmodel.save(str(output_path))
    return mlmodel


def run_validation(keras_model, mlmodel, params, features_lib, src_dir: Path) -> bool:
    rng = np.random.default_rng(20260713)
    inputs = {
        "random(seed=20260713)": rng.standard_normal((1, 96, 64)).astype(np.float32),
        "zeros": np.zeros((1, 96, 64), dtype=np.float32),
        "sine(440Hz,16kHz)": synthetic_sine_patch(src_dir, params, features_lib),
    }

    print()
    print("=== Keras vs Core ML validation (3 deterministic inputs) ===")
    # Looser than the ONNX gate's 1e-4: Core ML runs the conv stack through
    # its own kernels (and possibly the ANE), so bit-level agreement is not
    # on offer even at FLOAT32. 1e-3 on sigmoid scores and unit-scale
    # embedding components is far below anything the decision thresholds or
    # the prototype matcher can distinguish.
    tolerance = 1e-3
    ok = True
    for label, patch in inputs.items():
        keras_scores, keras_emb = keras_model.predict(patch, steps=1, verbose=0)
        out = mlmodel.predict({INPUT_NAME: patch})
        scores_diff = float(np.max(np.abs(keras_scores - out[SCORES_NAME])))
        emb_diff = float(np.max(np.abs(keras_emb - out[EMBEDDINGS_NAME])))
        line_ok = scores_diff < tolerance and emb_diff < tolerance
        ok = ok and line_ok
        print(
            f"  {label:<24} scores {scores_diff:.3e}  embeddings {emb_diff:.3e}  "
            f"{'PASS' if line_ok else 'FAIL'}"
        )
        top_keras = np.argsort(-keras_scores[0])[:5]
        top_coreml = np.argsort(-out[SCORES_NAME][0])[:5]
        if not np.array_equal(top_keras, top_coreml):
            ok = False
            print(f"    top-5 ordering differs: keras={top_keras} coreml={top_coreml}")
    print(f"  tolerance {tolerance:.0e}: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "sinus-sentinel-yamnet",
    )
    args = parser.parse_args()

    import tensorflow as tf

    print(f"TensorFlow {tf.__version__}")
    src_dir = fetch_reference_sources(args.cache_dir)

    print("\nBuilding patches-input Keras model and loading published weights...")
    model, params, _yamnet_lib, _params_lib = build_patches_model(src_dir)
    features_lib = import_module_from_path("yamnet_features", src_dir / "features.py")

    print(f"\nConverting to Core ML (mlprogram, FLOAT32) -> {args.output}")
    mlmodel = export_coreml(model, args.output)
    print(f"  input:  '{INPUT_NAME}'      shape [1, 96, 64] float32")
    print(f"  output: '{SCORES_NAME}'     shape [1, 521]    float32")
    print(f"  output: '{EMBEDDINGS_NAME}' shape [1, 1024]   float32")

    passed = run_validation(model, mlmodel, params, features_lib, src_dir)

    print()
    if not passed:
        print("VALIDATION FAILED -- do not ship this model.", file=sys.stderr)
        return 1
    print("Validation PASSED. Compile with:")
    print(f"  xcrun coremlcompiler compile {args.output} apps/apple/Resources/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
