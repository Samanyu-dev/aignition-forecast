"""
Load the pre-trained model + freshly generated features, produce
output/predictions.csv. This is the second (and final) step of run.sh's
critical path — no retraining, no network calls.

The model encapsulates fitted per-segment trend/seasonality/elasticity
parameters; --features is read and passed to forecast() to ensure the forecast
origin, run-rate baselines, and campaign rosters align dynamically with the
evaluation dataset.
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

    model = {}
    if os.path.exists(args.model):
        print(f"Loading model from {args.model}", file=sys.stderr)
        try:
            with open(args.model, "rb") as f:
                model = pickle.load(f)
        except Exception as exc:
            print(f"Warning: Failed to load {args.model} ({exc}), falling back to dynamic initialization", file=sys.stderr)
            model = {"segments": {}, "campaign_segments": {}}
    else:
        print(f"Model path {args.model} not found; using dynamic feature-driven initialization", file=sys.stderr)
        model = {"segments": {}, "campaign_segments": {}}

    print(f"Forecasting horizons {HORIZONS} (seed={SEED}) dynamically from evaluation features", file=sys.stderr)
    predictions = forecast(model, horizons=HORIZONS, eval_features=features, seed=SEED)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Wrote {len(predictions)} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

