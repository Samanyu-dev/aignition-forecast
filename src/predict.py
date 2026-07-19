"""
Load the pre-trained model + freshly generated features, produce
output/predictions.csv. This is the second (and final) step of run.sh's
critical path — no retraining, no network calls.

The model already encapsulates fitted per-segment trend/seasonality/elasticity
parameters (see src/train.py); --features is read to confirm the freshly
generated feature table is well-formed for this run, and the baseline
(budget-multiplier=1.0, i.e. "continue current run-rate") forecast is written.
Budget-scenario overrides are exposed separately through src/api.py, not here.
"""
import argparse
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from forecasting import forecast, HORIZONS, SEED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print(f"Loading features from {args.features}", file=sys.stderr)
    features = pd.read_parquet(args.features)
    if features.empty:
        raise ValueError(f"{args.features} is empty")

    print(f"Loading model from {args.model}", file=sys.stderr)
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    print(f"Forecasting horizons {HORIZONS} (seed={SEED})", file=sys.stderr)
    predictions = forecast(model, horizons=HORIZONS, seed=SEED)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Wrote {len(predictions)} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
