"""End-to-end pipeline: preprocess, validate, select, forecast the holdout.

Run: python -m cwfc.predict --data challenge_data --output predictions.csv

Model selection uses only backtests inside the training window. The holdout
targets are read once, after predictions.csv is written, to score the result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.evaluation import backtest, headroom_margin, mae_by_horizon, summarize
from cwfc.metrics import mae, rmse
from cwfc.models import Persistence, default_forecasters
from cwfc.preprocess import (
    COUNT,
    HORIZON,
    TARGET,
    build_fleet_series,
    build_history,
    build_target_series,
    load_readings,
    split_train_holdout,
)
from cwfc.validation import expanding_origins

MIN_TRAIN = 60
ORIGIN_STEP = 6
PEAK_QUANTILE = 0.9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    data, reports = Path(args.data), Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)

    readings = load_readings(str(data / "vm_cpu_readings.csv"))
    metadata = pd.read_csv(data / "vm_metadata.csv")
    history = build_history(build_target_series(readings), build_fleet_series(readings, metadata))
    train, holdout = split_train_holdout(history, HORIZON)
    print(f"history: {len(history)} rows -> train={len(train)} holdout={len(holdout)}")

    # Validation and selection: training window only.
    forecasters = default_forecasters()
    origins = expanding_origins(len(train), HORIZON, min(MIN_TRAIN, len(train) - HORIZON), ORIGIN_STEP) \
        if len(train) > HORIZON else []
    peak_threshold = float(train[TARGET].quantile(PEAK_QUANTILE))
    results = backtest(forecasters, train, origins, HORIZON) if origins else pd.DataFrame()
    if results.empty:
        best_name, margin = "persistence", 0.0
        print("not enough history to validate; falling back to persistence")
    else:
        scores = summarize(results, peak_threshold)
        best_name = scores.model.iloc[0]
        margin = headroom_margin(results, best_name)
        results.to_csv(reports / "validation_errors.csv", index=False)
        scores.to_csv(reports / "validation_scores.csv", index=False)
        mae_by_horizon(results).to_csv(reports / "validation_mae_by_horizon.csv")
        print(f"validated {scores.shape[0]} models on {len(origins)} origins -> wrote {reports / 'validation_scores.csv'}")
    print(f"selected: {best_name}")

    # Final forecast from the end of the training window.
    best = next((f for f in forecasters if f.name == best_name), Persistence())
    cpu_forecast = np.clip(best.fit(train).predict(train, HORIZON), 0.0, 100.0)
    count_forecast = int(train[COUNT].iloc[-1])
    predictions = pd.DataFrame({
        "timestamp": holdout["timestamp"].to_numpy(),
        COUNT: count_forecast,
        TARGET: cpu_forecast,
    })
    if not np.isfinite(predictions[[COUNT, TARGET]].to_numpy(dtype=float)).all():
        raise ValueError("non-finite prediction")
    predictions.to_csv(args.output, index=False)
    print(f"wrote {args.output}")

    bands = predictions.assign(cpu_upper=np.clip(cpu_forecast + margin, 0.0, 100.0), margin=margin)
    bands.to_csv(reports / "forecast_bands.csv", index=False)
    print(f"wrote {reports / 'forecast_bands.csv'} (headroom margin {margin:.3f} pts)")

    # Final scoring on the holdout, after selection and after predictions are written.
    holdout_rows = []
    for forecaster in forecasters:
        try:
            forecast = np.clip(forecaster.fit(train).predict(train, HORIZON), 0.0, 100.0)
        except ValueError:
            continue
        actual = holdout[TARGET].to_numpy(dtype=float)
        peak = actual >= peak_threshold
        holdout_rows.append({
            "model": forecaster.name,
            "selected": forecaster.name == best_name,
            "mae": mae(actual, forecast),
            "rmse": rmse(actual, forecast),
            "peak_mae": mae(actual[peak], forecast[peak]) if peak.any() else np.nan,
            "bias": float(np.mean(forecast - actual)),
            "covered_by_margin": float(np.mean(actual <= forecast + margin)) if forecaster.name == best_name else np.nan,
        })
    holdout_scores = pd.DataFrame(holdout_rows)
    reference = holdout_scores.loc[holdout_scores.model == "persistence", "mae"]
    if len(reference) and reference.iloc[0] > 0:
        holdout_scores["mae_vs_persistence"] = holdout_scores.mae / reference.iloc[0]
    holdout_scores["count_mae"] = mae(holdout[COUNT], np.full(len(holdout), count_forecast))
    holdout_scores.to_csv(reports / "holdout_scores.csv", index=False)
    print(f"wrote {reports / 'holdout_scores.csv'}")


if __name__ == "__main__":
    main()
