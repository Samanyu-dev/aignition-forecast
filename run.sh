#!/usr/bin/env bash
set -euo pipefail

# Accept arguments, fall back to defaults for local runs
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect virtualenv python if present, else fallback to PATH python3
PYTHON_BIN="python3"
if [ -f "$SCRIPT_DIR/.venv/bin/python3" ]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -f "$VIRTUAL_ENV/bin/python3" ]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python3"
fi

# Cross-platform Python-managed temporary file creation (POSIX macOS + Linux compatible)
FEATURES_PATH=$("$PYTHON_BIN" -c "import tempfile; print(tempfile.NamedTemporaryFile(suffix='.parquet', delete=False).name)")
trap 'rm -f "$FEATURES_PATH"' EXIT

OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
mkdir -p "$OUTPUT_DIR"

# 1. Generate features dynamically from evaluation dataset
"$PYTHON_BIN" "$SCRIPT_DIR/src/generate_features.py" \
  --data-dir "$DATA_DIR" \
  --out "$FEATURES_PATH"

# 2. Dynamic evaluation-driven probabilistic forecasting
"$PYTHON_BIN" "$SCRIPT_DIR/src/predict.py" \
  --features "$FEATURES_PATH" \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_PATH"

# 3. Generate AI business insights & causal narrative
INSIGHTS_JSON="$OUTPUT_DIR/insights.json"
INSIGHTS_TXT="$OUTPUT_DIR/insights.txt"
"$PYTHON_BIN" "$SCRIPT_DIR/src/llm_summary.py" \
  --features "$FEATURES_PATH" \
  --model "$MODEL_PATH" \
  --out-json "$INSIGHTS_JSON" \
  --out-txt "$INSIGHTS_TXT" || true

echo "Done. Predictions written to $OUTPUT_PATH"
echo "AI Insights written to $INSIGHTS_JSON and $INSIGHTS_TXT"


