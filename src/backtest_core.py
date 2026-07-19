"""
Shared walk-forward evaluation core, used by both src/train.py (to pick the
winning model per segment and calibrate interval width at training time) and
src/backtest.py (the standalone CLI report). Deliberately has NO dependency
on train.py, to keep train.py -> backtest_core.py one-directional (train.py
needs this at fit time; backtest_core.py must not need train.py back).
"""
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from forecasting import _simulate_segment_daily

TEST_HORIZON_DAYS = 30
N_CUTOFFS = 3
MIN_TRAIN_DAYS = 45
N_SIMS = 2000


def pinball_loss(y_true: float, y_pred: float, tau: float) -> float:
    diff = y_true - y_pred
    return max(tau * diff, (tau - 1) * diff)


def generate_cutoffs(daily: pd.DataFrame, n_cutoffs: int, test_horizon_days: int = TEST_HORIZON_DAYS) -> list:
    """All candidate cutoff dates, oldest first, spaced test_horizon_days apart,
    ending test_horizon_days before the segment's last observed date."""
    last_date = daily["date"].max()
    return [last_date - pd.Timedelta(days=test_horizon_days * i) for i in range(n_cutoffs, 0, -1)]


def valid_cutoffs(daily: pd.DataFrame, cutoffs: list, test_horizon_days: int, min_train_days: int) -> list:
    """Filter to cutoffs that actually have enough training history before
    them and enough test data after them to be scoreable."""
    out = []
    for cutoff in cutoffs:
        train_slice = daily[daily.date <= cutoff]
        test_slice = daily[(daily.date > cutoff) & (daily.date <= cutoff + pd.Timedelta(days=test_horizon_days))]
        if len(train_slice) >= min_train_days and len(test_slice) >= test_horizon_days:
            out.append(cutoff)
    return out


def evaluate_segment_method(
    daily: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], dict],
    rng: np.random.Generator,
    base_uncertainty_inflation: float = 1.0,
    residual_scale: float = 1.0,
    test_horizon_days: int = TEST_HORIZON_DAYS,
    n_cutoffs: int = N_CUTOFFS,
    min_train_days: int = MIN_TRAIN_DAYS,
    n_sims: int = N_SIMS,
    cutoffs: Optional[List[pd.Timestamp]] = None,
) -> dict:
    """Walk-forward evaluate one (segment, fitting function) pair.

    residual_scale multiplies the segment's residual-bootstrap noise on top
    of base_uncertainty_inflation -- used both to score a candidate model
    as-is (residual_scale=1.0) and to grid-search a calibration factor that
    widens under-covered intervals (residual_scale>1.0).

    cutoffs: explicit list of cutoff dates to evaluate, overriding the
    default "last n_cutoffs windows" generation. Used for rolling-origin
    calibration -- pass the earlier half of available cutoffs to *tune* a
    calibration scale, then the later, never-touched-during-tuning half to
    *report* coverage, so the reported number isn't evaluated on the same
    data it was fit to (see docs/TECHNICAL_DOC.md sec 3.8).
    """
    if cutoffs is None:
        cutoffs = generate_cutoffs(daily, n_cutoffs, test_horizon_days)

    per_cutoff = []
    for cutoff in cutoffs:
        train_slice = daily[daily.date <= cutoff]
        test_slice = daily[(daily.date > cutoff) & (daily.date <= cutoff + pd.Timedelta(days=test_horizon_days))]

        if len(train_slice) < min_train_days or len(test_slice) < test_horizon_days:
            per_cutoff.append({"cutoff": str(cutoff.date()), "skipped": True})
            continue

        try:
            seg = fit_fn(train_slice)
        except Exception as exc:  # noqa: BLE001 -- a candidate method failing to fit is a valid (bad) score
            per_cutoff.append({"cutoff": str(cutoff.date()), "skipped": True, "reason": f"fit failed: {exc}"})
            continue

        seg["uncertainty_inflation"] = base_uncertainty_inflation * residual_scale
        sims = _simulate_segment_daily(seg, test_horizon_days, n_sims, rng)
        revenue_sim = sims.sum(axis=1)
        p10, p50, p90 = np.percentile(revenue_sim, [10, 50, 90])
        actual = float(test_slice["revenue"].sum())

        ape = abs(actual - p50) / max(actual, 1.0) * 100
        pinball = (
            pinball_loss(actual, p10, 0.1) + pinball_loss(actual, p50, 0.5) + pinball_loss(actual, p90, 0.9)
        ) / 3
        per_cutoff.append({
            "cutoff": str(cutoff.date()), "skipped": False,
            "actual": actual, "p10": float(p10), "p50": float(p50), "p90": float(p90),
            "ape_pct": ape, "pinball_loss": pinball, "covered": bool(p10 <= actual <= p90),
        })

    scored = [r for r in per_cutoff if not r["skipped"]]
    if not scored:
        return {"per_cutoff": per_cutoff, "n_scored": 0, "mean_ape_pct": None,
                "mean_pinball_loss": None, "coverage_pct": None}

    return {
        "per_cutoff": per_cutoff,
        "n_scored": len(scored),
        "mean_ape_pct": float(np.mean([r["ape_pct"] for r in scored])),
        "mean_pinball_loss": float(np.mean([r["pinball_loss"] for r in scored])),
        "coverage_pct": float(np.mean([r["covered"] for r in scored]) * 100),
    }
