import pytest

from cwfc.metrics import mae, normalized_mae, rmse


def test_metrics_basic():
    y = [1, 2, 3]
    p = [1, 3, 1]
    assert mae(y, p) == 1.0
    assert round(rmse(y, p), 8) == round((5 / 3) ** 0.5, 8)


def test_normalized_mae():
    y = [1, 2, 3]
    assert normalized_mae(y, [1, 2, 4], [1, 1, 1]) == pytest.approx((1 / 3) / 1.0)
    assert normalized_mae(y, y, y) == 0.0
    assert normalized_mae(y, [1, 2, 4], y) == float("inf")
