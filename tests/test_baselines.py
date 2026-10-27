import pandas as pd
import pytest

from cwfc.baselines import moving_average_forecast


def test_moving_average_uses_last_window_values():
    train = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    template = pd.DataFrame(index=[5, 6, 7])
    forecast = moving_average_forecast(train, template, "y", window=3)
    assert forecast.tolist() == [4.0, 4.0, 4.0]
    assert forecast.index.tolist() == [5, 6, 7]


def test_moving_average_rejects_short_history_and_bad_window():
    train = pd.DataFrame({"y": [1.0, 2.0]})
    template = pd.DataFrame(index=[2])
    with pytest.raises(ValueError):
        moving_average_forecast(train, template, "y", window=3)
    with pytest.raises(ValueError):
        moving_average_forecast(train, template, "y", window=0)
