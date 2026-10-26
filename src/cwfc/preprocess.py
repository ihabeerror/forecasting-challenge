from __future__ import annotations

import numpy as np
import pandas as pd

STEP_SECONDS = 300
TRAIN_SIZE = 110
HORIZON = 24

TARGET = "aggregate_cpu_utilization"
CPU_LAGS = (1, 2, 3, 6)
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
    """Count VMs with created <= t < deleted, from the known VM lifetimes."""
    t = np.asarray(timestamps)[None, :]
    created = metadata["timestamp_vm_created"].to_numpy()[:, None]
    deleted = metadata["timestamp_vm_deleted"].to_numpy()[:, None]
    counts = ((created <= t) & (t < deleted)).sum(axis=0)
    return pd.Series(counts, index=timestamps.index, name="active_vm_count")


def build_fleet_series(readings: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Per-timestamp fleet statistics (same-timestamp values, not yet lagged)."""
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


def build_feature_table(series: pd.DataFrame, fleet: pd.DataFrame) -> pd.DataFrame:
    """One row per timestamp: targets plus features built only from earlier rows.

    Every feature at row t is computed from shifted values (t-1 and earlier), so
    it is available before the value at t is observed. The first rows contain
    NaN where there is not enough history; drop them when fitting.
    """
    frame = series.merge(fleet, on="timestamp", how="left").sort_values("timestamp").reset_index(drop=True)
    cpu = frame[TARGET]

    for k in CPU_LAGS:
        frame[f"cpu_lag{k}"] = cpu.shift(k)
    frame["cpu_lag1_diff"] = frame["cpu_lag1"] - frame["cpu_lag2"]
    past = cpu.shift(1)
    frame[f"cpu_roll_mean_{ROLL_WINDOW}"] = past.rolling(ROLL_WINDOW).mean()
    frame[f"cpu_roll_std_{ROLL_WINDOW}"] = past.rolling(ROLL_WINDOW).std()

    for col in fleet.columns.drop("timestamp"):
        frame[f"{col}_lag1"] = frame[col].shift(1)
        frame = frame.drop(columns=col)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Columns that are model inputs: everything except timestamp and targets."""
    return [c for c in frame.columns if c not in ("timestamp", "active_vm_count", TARGET)]


def split_train_holdout(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: first 110 rows train, final 24 rows holdout."""
    if len(frame) != TRAIN_SIZE + HORIZON:
        raise ValueError(f"expected {TRAIN_SIZE + HORIZON} rows, got {len(frame)}")
    return frame.iloc[:TRAIN_SIZE].copy(), frame.iloc[TRAIN_SIZE:].copy()
