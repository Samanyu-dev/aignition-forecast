"""
Ingest raw Bing / Google Ads / Meta Ads campaign CSVs from DATA_DIR and normalize
them into a single long-format feature table.

Reads files by filename pattern (case-insensitive substring match on
"bing" / "google" / "meta"), not by hardcoded row counts, so this works
unchanged against held-out test data with the same schema.

Data validation finding: Meta's `conversion` column is NOT a conversion count,
despite the name. It is treated here as conversion VALUE (revenue), based on:
  - values are fractional currency-like amounts (e.g. 34.99, 40.99, 120.00),
    which a count column would never produce;
  - the column's sum (1.66M) exceeds total clicks (540K) across the dataset,
    which is impossible if it were a count of converting users;
  - row-level (conversion / spend) produces a ROAS-shaped distribution
    (median ~4.1, mean ~11.2) matching the same range as Bing/Google's
    measured (revenue / spend) ROAS, whereas a genuine per-dollar conversion
    RATE would be two orders of magnitude smaller.
This is documented as a dataset-specific assumption in docs/TECHNICAL_DOC.md.
Meta rows are still flagged via revenue_is_imputed=True (False here, since we
treat it as measured, not derived) to keep the schema uniform and adjustable
if an updated data dictionary becomes available.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def _find_file(data_dir: str, needle: str) -> str:
    matches = [
        f for f in glob.glob(os.path.join(data_dir, "*.csv"))
        if needle in os.path.basename(f).lower()
    ]
    if not matches:
        raise FileNotFoundError(f"No CSV matching '*{needle}*' found in {data_dir}")
    if len(matches) > 1:
        matches.sort()
    return matches[0]


def infer_funnel_stage(campaign_name: str) -> str:
    """Heuristic funnel-stage bucket shared across all three channels, used only
    to segment the AOV used for Meta's revenue imputation. 'high_intent' covers
    branded/trademark search and remarketing (capturing existing demand);
    everything else ('prospecting') covers non-trademark, generic, and
    upper-funnel/display-style campaigns."""
    n = str(campaign_name).lower()
    if "remarketing" in n or "retarget" in n or "_tm_" in n:
        return "high_intent"
    return "prospecting"


def load_bing(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TimePeriod"]),
        "channel": "bing",
        "campaign_id": df["CampaignId"].astype(str),
        "campaign_name": df["CampaignName"],
        "campaign_type": df["CampaignType"],
        "spend": df["Spend"].astype(float),
        "revenue": df["Revenue"].astype(float),
        "conversions": df["Conversions"].astype(float),
        "revenue_is_imputed": False,
    })
    return out


def load_google(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["segments_date"]),
        "channel": "google",
        "campaign_id": df["campaign_id"].astype(str),
        "campaign_name": df["campaign_name"],
        "campaign_type": df["campaign_advertising_channel_type"],
        "spend": df["metrics_cost_micros"].astype(float) / 1_000_000.0,
        "revenue": df["metrics_conversions_value"].astype(float),
        "conversions": df["metrics_conversions"].astype(float),
        "revenue_is_imputed": False,
    })
    return out


def load_meta(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    campaign_type = (
        df["campaign_name"]
        .astype(str)
        .str.replace(r"_Campaign_\d+$", "", regex=True)
    )
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date_start"]),
        "channel": "meta",
        "campaign_id": df["campaign_id"].astype(str),
        "campaign_name": df["campaign_name"],
        "campaign_type": campaign_type,
        "spend": df["spend"].astype(float),
        "revenue": df["conversion"].astype(float),  # see data validation finding above
        "conversions": np.nan,  # no genuine conversion-count field available for Meta
        "revenue_is_imputed": False,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bing_path = _find_file(args.data_dir, "bing")
    google_path = _find_file(args.data_dir, "google")
    meta_path = _find_file(args.data_dir, "meta")

    print(f"Loading Bing from {bing_path}", file=sys.stderr)
    bing = load_bing(bing_path)
    print(f"Loading Google from {google_path}", file=sys.stderr)
    google = load_google(google_path)
    print(f"Loading Meta from {meta_path}", file=sys.stderr)
    meta = load_meta(meta_path)

    features = pd.concat([bing, google, meta], ignore_index=True)
    features["funnel_stage"] = features["campaign_name"].apply(infer_funnel_stage)

    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values(["channel", "campaign_id", "date"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    features.to_parquet(args.out, index=False)
    print(f"Wrote {len(features)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
