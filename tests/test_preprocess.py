import numpy as np
import pandas as pd

from cwfc.preprocess import (
    TARGET,
    active_vm_count_from_lifetimes,
    build_feature_table,
    feature_columns,
)


def _toy_frames(n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    series = pd.DataFrame({
        "timestamp": np.arange(n) * 300,
        "active_vm_count": 10,
        TARGET: rng.uniform(5, 15, n),
    })
    fleet = pd.DataFrame({"timestamp": series.timestamp, "cpu_p95": rng.uniform(20, 40, n)})
    return series, fleet


def test_features_ignore_current_target():
    series, fleet = _toy_frames()
    base = build_feature_table(series, fleet)
    changed = series.copy()
    changed.loc[12, TARGET] += 100.0
    altered = build_feature_table(changed, fleet)
    cols = feature_columns(base)
    pd.testing.assert_frame_equal(base.loc[:12, cols], altered.loc[:12, cols])


def test_lag1_is_previous_value():
    series, fleet = _toy_frames()
    frame = build_feature_table(series, fleet)
    assert frame.loc[5, "cpu_lag1"] == series.loc[4, TARGET]
    assert frame.loc[5, "cpu_lag1_diff"] == series.loc[4, TARGET] - series.loc[3, TARGET]


def test_active_count_from_lifetimes():
    metadata = pd.DataFrame({
        "timestamp_vm_created": [0, 0, 600],
        "timestamp_vm_deleted": [1000, 300, 2000],
    })
    ts = pd.Series([0, 300, 600, 1200])
    assert active_vm_count_from_lifetimes(metadata, ts).tolist() == [2, 1, 2, 1]
