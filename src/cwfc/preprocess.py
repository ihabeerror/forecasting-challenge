from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STEP_SECONDS = 300
HORIZON = 24

TARGET = "aggregate_cpu_utilization"
COUNT = "active_vm_count"
CPU_LAGS = (1, 2, 5)
ROLL_WINDOW = 6
HOT_THRESHOLD = 50.0


def load_readings(path: str) -> pd.DataFrame:
    """Load per-VM CPU readings, sorted by timestamp then vm_id."""
    readings = pd.read_csv(path)
    return readings.sort_values(["timestamp", "vm_id"]).reset_index(drop=True)


def build_target_series(readings: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-VM readings into one row per timestamp.

    active_vm_count: distinct VMs with a reading at the timestamp.
    aggregate_cpu_utilization: mean of VM-level avg_cpu; missing readings are
    skipped by the mean, so the value is over the VMs actually reported.
    """
    series = (
        readings.groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .sort_index()
        .reset_index()
    )
    steps = series["timestamp"].diff().dropna()
    if not (steps == STEP_SECONDS).all():
        raise ValueError("timestamps are not evenly spaced at 5 minutes")
    return series


def active_vm_count_from_lifetimes(metadata: pd.DataFrame, timestamps: pd.Series) -> pd.Series:
    """Count VMs with created <= t < deleted. A cross-check only: deletion times
    are not known in advance in production, so this is not used to forecast."""
    t = np.asarray(timestamps)[None, :]
    created = metadata["timestamp_vm_created"].to_numpy()[:, None]
    deleted = metadata["timestamp_vm_deleted"].to_numpy()[:, None]
    counts = ((created <= t) & (t < deleted)).sum(axis=0)
    return pd.Series(counts, index=timestamps.index, name=COUNT)


def build_fleet_series(readings: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Per-timestamp fleet statistics observed at that timestamp."""
    grouped = readings.groupby("timestamp")
    fleet = pd.DataFrame({
        "cpu_p95": grouped.avg_cpu.quantile(0.95),
        "cpu_std_across_vms": grouped.avg_cpu.std(),
        "share_above_50": readings.assign(hot=readings.avg_cpu > HOT_THRESHOLD).groupby("timestamp").hot.mean(),
        "max_cpu_mean": grouped.max_cpu.mean(),
    })
    merged = readings.merge(metadata[["vm_id", "vm_category"]], on="vm_id", how="left")
    by_category = merged.groupby(["timestamp", "vm_category"]).avg_cpu.mean().unstack()
    by_category.columns = [f"cpu_{str(c).lower().replace('-', '_')}" for c in by_category.columns]
    return fleet.join(by_category).sort_index().reset_index()


def build_history(series: pd.DataFrame, fleet: pd.DataFrame) -> pd.DataFrame:
    """One row per timestamp: targets plus the fleet statistics observed then."""
    history = series.merge(fleet, on="timestamp", how="left")
    return history.sort_values("timestamp").reset_index(drop=True)


def fleet_columns(history: pd.DataFrame) -> list[str]:
    return [c for c in history.columns if c not in ("timestamp", COUNT, TARGET)]


def target_features(cpu: pd.Series) -> pd.DataFrame:
    """Features from the target's own history, known at each row.

    Row o uses only values at o and earlier, so it describes what is known at
    a forecast origin o. Early rows are NaN where history is too short.
    """
    cpu = cpu.astype(float)
    features = pd.DataFrame({"cpu_now": cpu}, index=cpu.index)
    for k in CPU_LAGS:
        features[f"cpu_lag{k}"] = cpu.shift(k)
    features["cpu_change"] = cpu - cpu.shift(1)
    features[f"cpu_roll_mean_{ROLL_WINDOW}"] = cpu.rolling(ROLL_WINDOW).mean()
    features[f"cpu_roll_std_{ROLL_WINDOW}"] = cpu.rolling(ROLL_WINDOW).std()
    return features


def origin_features(history: pd.DataFrame) -> pd.DataFrame:
    """Everything known at each forecast origin: target history plus fleet stats at that row."""
    return target_features(history[TARGET]).join(history[fleet_columns(history)])


def split_train_holdout(frame: pd.DataFrame, horizon: int = HORIZON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the final `horizon` rows are the holdout."""
    if len(frame) <= horizon:
        raise ValueError(f"need more than {horizon} rows, got {len(frame)}")
    return frame.iloc[:-horizon].copy(), frame.iloc[-horizon:].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the history and origin features, run basic checks.")
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.out)

    readings = load_readings(str(data / "vm_cpu_readings.csv"))
    metadata = pd.read_csv(data / "vm_metadata.csv")
    series = build_target_series(readings)
    history = build_history(series, build_fleet_series(readings, metadata))
    train, holdout = split_train_holdout(history)
    features = origin_features(history)
    print(f"built history: {len(history)} rows, {features.shape[1]} origin features")
    print(f"split: train={len(train)} holdout={len(holdout)}")

    counts = active_vm_count_from_lifetimes(metadata, series["timestamp"])
    print(f"lifetime-based active_vm_count matches readings: {bool((counts == series[COUNT]).all())}")
    print(f"origins with incomplete features (skipped when fitting): {int(features.isna().any(axis=1).sum())}")

    out.mkdir(parents=True, exist_ok=True)
    history[["timestamp", COUNT, TARGET]].join(features).to_csv(out / "origin_features.csv", index=False)
    print(f"wrote {out / 'origin_features.csv'}")


if __name__ == "__main__":
    main()
