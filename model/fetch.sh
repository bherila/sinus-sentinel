#!/usr/bin/env bash
# Convert YAMNet (TF-Hub) to model/yamnet.onnx. See model/README.md for the
# tensor contract. Requires: python3 with tensorflow, tensorflow-hub, tf2onnx.
#
# We deliberately do NOT vendor a prebuilt .onnx (size + provenance); this script
# reproduces it from the upstream TF-Hub module.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${HERE}/yamnet.onnx"
SAVED="${HERE}/.saved_model"

if [[ -f "${OUT}" ]]; then
  echo "model already present: ${OUT}"
  exit 0
fi

if ! command -v python3 >/dev/null; then
  echo "python3 not found; install it plus: pip install tensorflow tensorflow-hub tf2onnx onnx" >&2
  exit 1
fi

if [[ ! -f "${HERE}/export_yamnet.py" ]]; then
  cat >&2 <<'EOF'
model/export_yamnet.py not found.

Provide an exporter that writes a SavedModel taking a [1,96,64] log-mel patch and
returning scores [1,521] + embeddings [1,1024], then re-run this script. See
model/README.md for the exact contract and the tf2onnx command.
EOF
  exit 1
fi

echo "exporting SavedModel -> ${SAVED}"
python3 "${HERE}/export_yamnet.py" --out "${SAVED}"

echo "converting SavedModel -> ${OUT}"
python3 -m tf2onnx.convert --saved-model "${SAVED}" --output "${OUT}" --opset 17

echo "done: ${OUT}"
