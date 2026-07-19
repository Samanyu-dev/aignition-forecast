#!/usr/bin/env bash
set -euo pipefail

# Accept arguments, fall back to defaults for local runs
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEATURES_PATH="$(mktemp -t aignition_features_XXXXXX).parquet"
trap 'rm -f "$FEATURES_PATH"' EXIT

mkdir -p "$(dirname "$OUTPUT_PATH")"

# 1. Generate the features the model expects from the data
python3 "$SCRIPT_DIR/src/generate_features.py" \
  --data-dir "$DATA_DIR" \
  --out "$FEATURES_PATH"

# 2. Load the pickled model and produce predictions (no retraining, no network)
python3 "$SCRIPT_DIR/src/predict.py" \
  --features "$FEATURES_PATH" \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_PATH"

echo "Done. Predictions written to $OUTPUT_PATH"
