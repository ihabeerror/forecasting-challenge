import numpy as np
import pandas as pd
import pytest

from cwfc.preprocess import (
    TARGET,
    active_vm_count_from_lifetimes,
    origin_features,
    split_train_holdout,
)


def _toy_history(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp": np.arange(n) * 300,
        "active_vm_count": 10,
        TARGET: rng.uniform(5, 15, n),
        "cpu_p95": rng.uniform(20, 40, n),
    })


def test_origin_features_ignore_later_rows():
    history = _toy_history()
    base = origin_features(history)
    changed = history.copy()
    changed.loc[13:, [TARGET, "cpu_p95"]] += 100.0
    altered = origin_features(changed)
    pd.testing.assert_frame_equal(base.loc[:12], altered.loc[:12])


def test_origin_feature_values():
    history = _toy_history()
    features = origin_features(history)
    cpu = history[TARGET]
    assert features.loc[8, "cpu_now"] == cpu[8]
    assert features.loc[8, "cpu_lag1"] == cpu[7]
    assert features.loc[8, "cpu_change"] == cpu[8] - cpu[7]
    assert features.loc[8, "cpu_roll_mean_6"] == pytest.approx(cpu[3:9].mean())
    assert features.loc[8, "cpu_p95"] == history.loc[8, "cpu_p95"]


def test_split_uses_final_rows_as_holdout():
    history = _toy_history(30)
    train, holdout = split_train_holdout(history, horizon=24)
    assert len(train) == 6 and len(holdout) == 24
    assert holdout.index[0] == 6
    with pytest.raises(ValueError):
        split_train_holdout(history.iloc[:24], horizon=24)


def test_active_count_from_lifetimes():
    metadata = pd.DataFrame({
        "timestamp_vm_created": [0, 0, 600],
        "timestamp_vm_deleted": [1000, 300, 2000],
    })
    ts = pd.Series([0, 300, 600, 1200])
    assert active_vm_count_from_lifetimes(metadata, ts).tolist() == [2, 1, 2, 1]
