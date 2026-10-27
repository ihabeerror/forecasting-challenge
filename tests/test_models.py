import numpy as np
import pandas as pd
import pytest

from cwfc.evaluation import backtest, headroom_margin, summarize
from cwfc.models import DirectModel, Persistence, RecursiveModel, default_forecasters, ridge
from cwfc.preprocess import TARGET
from cwfc.validation import expanding_origins


def _history(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    t = np.arange(n)
    return pd.DataFrame({
        "timestamp": t * 300,
        "active_vm_count": 10,
        TARGET: 9 + 0.3 * np.sin(t / 5) + rng.normal(0, 0.05, n),
        "cpu_p95": 30 + rng.normal(0, 1, n),
    })


@pytest.mark.parametrize("forecaster", default_forecasters(), ids=lambda f: f.name)
def test_every_forecaster_returns_horizon_values(forecaster):
    history = _history()
    forecast = forecaster.fit(history).predict(history, 24)
    assert forecast.shape == (24,)
    assert np.isfinite(forecast).all()


@pytest.mark.parametrize("model", [DirectModel(ridge(), "d", horizon=6), RecursiveModel(ridge(), "r")])
def test_models_follow_a_linear_trend(model):
    t = np.arange(60, dtype=float)
    history = pd.DataFrame({"timestamp": t * 300, "active_vm_count": 1, TARGET: 5 + 0.1 * t})
    forecast = model.fit(history).predict(history, 6)
    np.testing.assert_allclose(forecast, 5 + 0.1 * np.arange(60, 66), atol=0.05)


def test_direct_model_with_history_shorter_than_horizon():
    history = _history(16)
    forecast = DirectModel(ridge(), "d", horizon=24).fit(history).predict(history, 24)
    assert forecast.shape == (24,)


def test_backtest_never_fits_on_forecast_rows():
    seen = []

    class Spy(Persistence):
        name = "spy"

        def fit(self, history):
            seen.append(len(history))
            return self

    history = _history()
    origins = expanding_origins(len(history), 24, 30, 6)
    results = backtest([Spy()], history, origins, 24)
    assert seen == origins
    assert (results.groupby("origin").timestamp.min() == history.timestamp.iloc[origins].to_numpy()).all()


def test_summary_and_margin():
    history = _history()
    results = backtest([Persistence(), RecursiveModel(ridge(), "ridge_recursive")], history, [40, 56], 24)
    scores = summarize(results, peak_threshold=9.2)
    assert set(scores.model) == {"persistence", "ridge_recursive"}
    assert scores.loc[scores.model == "persistence", "mae_vs_reference"].iloc[0] == pytest.approx(1.0)
    assert headroom_margin(results, "persistence") >= 0
