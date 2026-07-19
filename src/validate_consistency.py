"""
Formal campaign consistency validation -- a distinct pipeline step, not
folded into feature generation or forecasting, satisfying the brief's
"validating campaign consistency" Working Prototype requirement as its own
capability rather than something inferred from ad hoc findings elsewhere.

Reads the raw per-channel CSVs directly (not the trimmed unified feature
schema in generate_features.py, which drops budget/click columns not needed
for forecasting). Produces one structured JSON report covering, per channel:

  - campaign_id -> campaign_name/campaign_type stability over time
  - date-coverage gaps per campaign (>14 consecutive days with no activity
    inside an otherwise-active campaign)
  - spend exceeding the campaign's stated daily_budget
  - conversions exceeding clicks (Bing/Google only -- Meta has no genuine
    conversion-count field, see generate_features.py's data-validation note)
  - negative/impossible values in any numeric column
  - zero-lifetime-revenue campaigns with nonzero spend (generalizes the
    Bing finding from docs/TECHNICAL_DOC.md to all three channels, computed
    the same way for each rather than special-cased to Bing)

Run standalone (writes docs/validation_report.json) or call run_validation()
directly -- src/llm_summary.py's compute_stats() and app_streamlit.py both
consume its output. Not part of run.sh's critical path.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from generate_features import _find_file

DATE_GAP_THRESHOLD_DAYS = 14
BUDGET_OVERSPEND_TOLERANCE = 1.05  # allow 5% over stated daily_budget before flagging

RAW_SCHEMA = {
    "bing": {
        "date": "TimePeriod", "campaign_id": "CampaignId", "campaign_name": "CampaignName",
        "campaign_type": "CampaignType", "spend": "Spend", "revenue": "Revenue",
        "clicks": "Clicks", "conversions": "Conversions", "impressions": "Impressions",
        "daily_budget": "DailyBudget",
    },
    "google": {
        "date": "segments_date", "campaign_id": "campaign_id", "campaign_name": "campaign_name",
        "campaign_type": "campaign_advertising_channel_type", "spend_micros": "metrics_cost_micros",
        "revenue": "metrics_conversions_value", "clicks": "metrics_clicks",
        "conversions": "metrics_conversions", "impressions": "metrics_impressions",
        "daily_budget": "campaign_budget_amount",
    },
    "meta": {
        # Meta's "conversion" column is revenue, not a conversion count -- see
        # generate_features.py's data-validation finding. No genuine clicks-vs-
        # conversions check is possible here for that reason.
        "date": "date_start", "campaign_id": "campaign_id", "campaign_name": "campaign_name",
        "spend": "spend", "revenue_proxy": "conversion", "clicks": "clicks",
        "impressions": "impressions", "daily_budget": "daily_budget",
    },
}


def _load_raw(data_dir: str, channel: str) -> pd.DataFrame:
    return pd.read_csv(_find_file(data_dir, channel))


def _check_campaign_id_stability(df, id_col, name_col, type_col=None) -> list:
    issues = []
    for cid, group in df.groupby(id_col):
        names = group[name_col].dropna().unique()
        if len(names) > 1:
            issues.append({"campaign_id": str(cid), "issue": "unstable_campaign_name",
                            "values": [str(v) for v in names]})
        if type_col and type_col in df.columns:
            types = group[type_col].dropna().unique()
            if len(types) > 1:
                issues.append({"campaign_id": str(cid), "issue": "unstable_campaign_type",
                                "values": [str(v) for v in types]})
    return issues


def _check_date_gaps(df, id_col, date_col, threshold_days=DATE_GAP_THRESHOLD_DAYS) -> list:
    out = []
    for cid, group in df.groupby(id_col):
        dates = pd.to_datetime(group[date_col]).sort_values().unique()
        if len(dates) < 2:
            continue
        gaps = np.diff(dates) / np.timedelta64(1, "D")
        max_gap = float(gaps.max()) if len(gaps) else 0.0
        if max_gap > threshold_days:
            out.append({"campaign_id": str(cid), "issue": "date_coverage_gap",
                         "max_gap_days": int(max_gap)})
    return out


def _check_budget_exceeded(df, spend_col, budget_col, id_col, tolerance=BUDGET_OVERSPEND_TOLERANCE) -> list:
    d = df.dropna(subset=[budget_col])
    d = d[d[budget_col] > 0]
    violations = d[d[spend_col] > d[budget_col] * tolerance]
    out = []
    for cid, group in violations.groupby(id_col):
        overspend_pct = ((group[spend_col] - group[budget_col]) / group[budget_col]).max() * 100
        out.append({"campaign_id": str(cid), "issue": "spend_exceeds_daily_budget",
                     "n_violating_days": int(len(group)), "max_overspend_pct": round(float(overspend_pct), 1)})
    return out


def _check_conversions_exceed_clicks(df, conv_col, click_col, id_col) -> list:
    violations = df[df[conv_col] > df[click_col]]
    out = []
    for cid, group in violations.groupby(id_col):
        out.append({"campaign_id": str(cid), "issue": "conversions_exceed_clicks",
                     "n_violating_rows": int(len(group))})
    return out


def _check_negative_values(df, cols, id_col) -> list:
    out = []
    for col in cols:
        if col not in df.columns:
            continue
        neg = df[df[col] < 0]
        for cid, group in neg.groupby(id_col):
            out.append({"campaign_id": str(cid), "issue": f"negative_{col}", "n_rows": int(len(group))})
    return out


def _check_zero_revenue_with_spend(df, spend_col, revenue_col, id_col, name_col) -> list:
    lifetime = df.groupby([id_col, name_col]).agg(
        spend=(spend_col, "sum"), revenue=(revenue_col, "sum")
    ).reset_index()
    flagged = lifetime[(lifetime.spend > 0) & (lifetime.revenue == 0)]
    return [
        {"campaign_id": str(r[id_col]), "campaign_name": str(r[name_col]),
         "issue": "zero_revenue_with_spend", "total_spend": round(float(r["spend"]), 2)}
        for _, r in flagged.iterrows()
    ]


def validate_channel(data_dir: str, channel: str) -> dict:
    df = _load_raw(data_dir, channel)
    schema = RAW_SCHEMA[channel]
    id_col, name_col = schema["campaign_id"], schema["campaign_name"]

    spend_col = schema.get("spend")
    if channel == "google":
        df["_spend_derived"] = df[schema["spend_micros"]] / 1_000_000.0
        spend_col = "_spend_derived"

    revenue_col = schema.get("revenue") or schema.get("revenue_proxy")
    budget_col = schema.get("daily_budget")

    checks = {
        "campaign_id_stability": _check_campaign_id_stability(df, id_col, name_col, schema.get("campaign_type")),
        "date_coverage_gaps": _check_date_gaps(df, id_col, schema["date"]),
        "budget_exceeded": (
            _check_budget_exceeded(df, spend_col, budget_col, id_col) if budget_col in df.columns else []
        ),
        "conversions_exceed_clicks": (
            _check_conversions_exceed_clicks(df, schema["conversions"], schema["clicks"], id_col)
            if "conversions" in schema and "clicks" in schema else []
        ),
        "negative_values": _check_negative_values(
            df, [spend_col, revenue_col, schema.get("clicks"), schema.get("conversions"), schema.get("impressions")],
            id_col,
        ),
        "zero_revenue_with_spend": _check_zero_revenue_with_spend(df, spend_col, revenue_col, id_col, name_col),
    }

    return {
        "channel": channel,
        "n_campaigns": int(df[id_col].nunique()),
        "n_rows": int(len(df)),
        "n_issues_flagged": sum(len(v) for v in checks.values()),
        "checks": checks,
    }


def run_validation(data_dir: str = "./data") -> dict:
    channels = {ch: validate_channel(data_dir, ch) for ch in ("bing", "google", "meta")}
    return {
        "generated_from": data_dir,
        "channels": channels,
        "summary": {
            "total_campaigns": sum(c["n_campaigns"] for c in channels.values()),
            "total_issues_flagged": sum(c["n_issues_flagged"] for c in channels.values()),
            "by_check_type": {
                check_name: sum(len(c["checks"][check_name]) for c in channels.values())
                for check_name in next(iter(channels.values()))["checks"]
            },
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="./docs/validation_report.json")
    args = ap.parse_args()

    report = run_validation(args.data_dir)

    print("=== CAMPAIGN CONSISTENCY VALIDATION ===")
    print(json.dumps(report["summary"], indent=2))
    for ch, c in report["channels"].items():
        print(f"\n{ch}: {c['n_campaigns']} campaigns, {c['n_rows']} rows, {c['n_issues_flagged']} issues flagged")
        for check_name, issues in c["checks"].items():
            if issues:
                print(f"  {check_name}: {len(issues)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
