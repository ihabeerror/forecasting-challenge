from __future__ import annotations

import numpy as np
import pandas as pd

from cwfc.metrics import mae, normalized_mae, rmse
from cwfc.models import Forecaster
from cwfc.preprocess import TARGET


def backtest(forecasters: list[Forecaster], history: pd.DataFrame, origins: list[int], horizon: int) -> pd.DataFrame:
    """Fit on rows [0, L) and forecast rows [L, L + horizon) for every origin L.

    Returns one row per (model, origin, step). A forecaster that cannot be fit
    at some origin is dropped entirely so every model is scored on the same folds.
    Gap-filled rows (observed == False) are forecast but not scored.
    """
    rows, failed = [], set()
    for origin in origins:
        known = history.iloc[:origin]
        actual = history.iloc[origin:origin + horizon]
        scored = actual["observed"].to_numpy(dtype=bool) if "observed" in actual else np.ones(len(actual), bool)
        for forecaster in forecasters:
            if forecaster.name in failed:
                continue
            try:
                forecast = forecaster.fit(known).predict(known, horizon)
            except ValueError:
                failed.add(forecaster.name)
                continue
            rows.append(pd.DataFrame({
                "model": forecaster.name,
                "origin": origin,
                "h": np.arange(1, horizon + 1),
                "timestamp": actual["timestamp"].to_numpy(),
                "actual": actual[TARGET].to_numpy(dtype=float),
                "forecast": forecast,
            })[scored])
    if not rows:
        return pd.DataFrame(columns=["model", "origin", "h", "timestamp", "actual", "forecast", "error"])
    results = pd.concat(rows, ignore_index=True)
    results = results[~results.model.isin(failed)].reset_index(drop=True)
    results["error"] = results.forecast - results.actual
    return results


def summarize(results: pd.DataFrame, peak_threshold: float, reference: str = "persistence") -> pd.DataFrame:
    """Per-model scores over all folds.

    mae_vs_reference: model MAE / reference MAE (below 1 beats the reference).
    peak_mae: MAE on steps where the actual value is at or above peak_threshold.
    bias: mean(forecast - actual); negative means the model under-forecasts.
    under_rate: share of steps where the actual exceeded the forecast.
    """
    keys = ["origin", "h"]
    ref = results[results.model == reference].set_index(keys).forecast if reference in set(results.model) else None
    out = []
    for name, group in results.groupby("model", sort=False):
        peak = group[group.actual >= peak_threshold]
        fold_mae = group.groupby("origin").apply(lambda g: mae(*_pair(g)), include_groups=False)
        model_mae = mae(*_pair(group))
        out.append({
            "model": name,
            "mae": model_mae,
            "rmse": rmse(*_pair(group)),
            "mae_vs_reference": (
                normalized_mae(group.actual, group.forecast, ref.loc[pd.MultiIndex.from_frame(group[keys])])
                if ref is not None else np.nan
            ),
            "peak_mae": mae(*_pair(peak)) if len(peak) else np.nan,
            "worst_fold_mae": fold_mae.max(),
            "bias": group.error.mean(),
            "under_rate": (group.error < 0).mean(),
        })
    return pd.DataFrame(out).sort_values("mae").reset_index(drop=True)


def mae_by_horizon(results: pd.DataFrame) -> pd.DataFrame:
    """MAE per forecast step: rows are h, columns are models."""
    return results.assign(abs_error=results.error.abs()).pivot_table(index="h", columns="model", values="abs_error")


def headroom_margin(results: pd.DataFrame, model: str, quantile: float = 0.95) -> float:
    """Margin above the forecast that covered `quantile` of validation outcomes."""
    shortfall = -results.loc[results.model == model, "error"]
    return float(max(shortfall.quantile(quantile), 0.0))


def _pair(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return frame.actual.to_numpy(), frame.forecast.to_numpy()
