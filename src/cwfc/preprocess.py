from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.data_files import data_file

STEP_SECONDS = 300
HORIZON = 24

TARGET = "aggregate_cpu_utilization"
COUNT = "active_vm_count"
CPU_LAGS = (1, 2, 5)
ROLL_WINDOW = 6
HOT_THRESHOLD = 50.0


def load_readings(path: str) -> pd.DataFrame:
    """Load per-VM CPU readings, keep valid ones, sort by timestamp then vm_id."""
    raw = pd.read_csv(path)
    readings = clean_readings(raw)
    dropped = len(raw) - len(readings)
    if dropped:
        warnings.warn(f"dropped {dropped} readings with missing avg_cpu or a duplicate (vm_id, timestamp)")
    return readings


def clean_readings(readings: pd.DataFrame) -> pd.DataFrame:
    """Keep valid readings only: avg_cpu present, one row per (vm_id, timestamp).

    A duplicate pair keeps its first row, so a VM is counted and weighted once.
    """
    valid = readings.dropna(subset=["timestamp", "vm_id", "avg_cpu"])
    valid = valid.drop_duplicates(["timestamp", "vm_id"], keep="first")
    return valid.sort_values(["timestamp", "vm_id"]).reset_index(drop=True)


def build_target_series(readings: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-VM readings into one row per timestamp on the 5-minute grid.

    active_vm_count: distinct VMs with a valid reading at the timestamp.
    aggregate_cpu_utilization: mean of VM-level avg_cpu over those VMs.

    Timestamps with no readings at all are added to the grid and filled with
    the last observed values (what an operator would know at that moment).
    The `observed` column marks real rows; filled rows are never scored and
    never used as holdout timestamps.
    """
    series = (
        clean_readings(readings).groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .sort_index()
    )
    if series.empty:
        raise ValueError("no valid readings")
    start = series.index[0]
    if ((series.index - start) % STEP_SECONDS != 0).any():
        raise ValueError("timestamps do not lie on a 5-minute grid")
    grid = np.arange(start, series.index[-1] + STEP_SECONDS, STEP_SECONDS)
    series = series.reindex(grid)
    series.index.name = "timestamp"
    series["observed"] = series[COUNT].notna()
    missing = int((~series["observed"]).sum())
    if missing:
        warnings.warn(f"{missing} missing 5-minute timestamps filled with the last observed values")
    series[[COUNT, TARGET]] = series[[COUNT, TARGET]].ffill()
    series[COUNT] = series[COUNT].astype(int)
    return series.reset_index()


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
    readings = clean_readings(readings)
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
    history = series.merge(fleet, on="timestamp", how="left").sort_values("timestamp").reset_index(drop=True)
    columns = fleet_columns(history)
    history[columns] = history[columns].ffill()  # gap rows carry the last observed fleet stats
    return history


def fleet_columns(history: pd.DataFrame) -> list[str]:
    return [c for c in history.columns if c not in ("timestamp", COUNT, TARGET, "observed")]


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
    """Chronological split: the final `horizon` observed timestamps are the holdout.

    Training is every grid row before the first holdout timestamp. Gap-filled
    rows inside the holdout span are dropped: they are neither trained on nor scored.
    """
    observed = frame["observed"] if "observed" in frame else pd.Series(True, index=frame.index)
    observed_rows = frame[observed.astype(bool)]
    if len(observed_rows) <= horizon:
        raise ValueError(f"need more than {horizon} observed timestamps, got {len(observed_rows)}")
    holdout = observed_rows.iloc[-horizon:].copy()
    train = frame[frame["timestamp"] < holdout["timestamp"].iloc[0]].copy()
    return train, holdout


def steps_ahead(train: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """Forecast step (1 = next 5 minutes) of each holdout timestamp from the training origin."""
    return ((holdout["timestamp"].to_numpy() - train["timestamp"].iloc[-1]) // STEP_SECONDS).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the history and origin features, run basic checks.")
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.out)

    readings = load_readings(str(data_file(data, "vm_cpu_readings")))
    metadata = pd.read_csv(data_file(data, "vm_metadata"))
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
