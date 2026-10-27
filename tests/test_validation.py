import pandas as pd
import pytest
from cwfc.validation import expanding_origins, rolling_origin_splits


def test_rolling_origin_splits_are_temporal():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC")
    })
    splits = rolling_origin_splits(frame, horizon=4, min_train_size=8, step=4)
    assert len(splits) == 3
    assert splits[0].train_end < splits[0].validation_start
    assert splits[0].validation_end < splits[1].validation_start


def test_expanding_origins_end_at_last_row():
    assert expanding_origins(110, 24, 60, 6) == [62, 68, 74, 80, 86]
    assert expanding_origins(30, 24, 10, 6) == []
    with pytest.raises(ValueError):
        expanding_origins(110, 24, 0, 6)
