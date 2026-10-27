"""Forecasters behind one interface: fit on known history, predict the next steps.

A forecaster only ever receives rows up to the forecast origin, so it cannot
see the values it is asked to predict.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cwfc.baselines import moving_average_forecast, persistence_forecast, seasonal_naive_forecast
from cwfc.preprocess import CPU_LAGS, HORIZON, ROLL_WINDOW, TARGET, origin_features, target_features

# Rows of history needed for target_features to be complete at the last row.
FEATURE_WINDOW = max(ROLL_WINDOW, max(CPU_LAGS) + 1)


def _template(horizon: int) -> pd.DataFrame:
    return pd.DataFrame(index=range(horizon))


class Forecaster(ABC):
    name = "forecaster"

    def fit(self, history: pd.DataFrame) -> Forecaster:
        return self

    @abstractmethod
    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        """Forecast the `horizon` steps after the last row of `history`."""


class Persistence(Forecaster):
    name = "persistence"

    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        return persistence_forecast(history, _template(horizon), TARGET).to_numpy()


class MovingAverage(Forecaster):
    def __init__(self, window: int = 6):
        self.window = window
        self.name = f"moving_average_{window}"

    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        return moving_average_forecast(history, _template(horizon), TARGET, self.window).to_numpy()


class SeasonalNaive(Forecaster):
    def __init__(self, season_length: int = 24):
        self.season_length = season_length
        self.name = f"seasonal_naive_{season_length}"

    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        return seasonal_naive_forecast(history, _template(horizon), TARGET, self.season_length).to_numpy()


class DirectModel(Forecaster):
    """One model for every horizon.

    Each training sample is (features at origin o, h) -> cpu(o + h) - cpu(o),
    using only origins whose target lies inside the given history. Predicting
    copies the features at the last row for h = 1..horizon.
    """

    def __init__(self, estimator: RegressorMixin, name: str, horizon: int = HORIZON):
        self.estimator = estimator
        self.name = name
        self.horizon = horizon

    def fit(self, history: pd.DataFrame) -> DirectModel:
        features = origin_features(history)
        self.columns_ = list(features.columns)
        values = features.to_numpy()
        complete = ~np.isnan(values).any(axis=1)
        cpu = history[TARGET].to_numpy(dtype=float)
        xs, ys = [], []
        for h in range(1, min(self.horizon, len(cpu) - 1) + 1):
            origins = np.flatnonzero(complete[: len(cpu) - h])
            if len(origins) == 0:
                continue
            xs.append(np.column_stack([values[origins], np.full(len(origins), h)]))
            ys.append(cpu[origins + h] - cpu[origins])
        if not xs:
            raise ValueError("not enough history to fit a direct model")
        self.model_ = clone(self.estimator).fit(np.vstack(xs), np.concatenate(ys))
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        last = origin_features(history)[self.columns_].iloc[-1].to_numpy(dtype=float)
        if np.isnan(last).any():
            raise ValueError("history too short for origin features")
        x = np.column_stack([np.tile(last, (horizon, 1)), np.arange(1, horizon + 1)])
        return float(history[TARGET].iloc[-1]) + self.model_.predict(x)


class RecursiveModel(Forecaster):
    """One-step model on the target's own history, rolled forward on its predictions.

    Training samples are (target features at o) -> cpu(o + 1) - cpu(o). Fleet
    features are not used because they cannot be rolled forward.
    """

    def __init__(self, estimator: RegressorMixin, name: str):
        self.estimator = estimator
        self.name = name

    def fit(self, history: pd.DataFrame) -> RecursiveModel:
        cpu = history[TARGET].reset_index(drop=True).astype(float)
        features = target_features(cpu).iloc[:-1]
        change = (cpu.shift(-1) - cpu).iloc[:-1]
        complete = features.notna().all(axis=1)
        if not complete.any():
            raise ValueError("not enough history to fit a recursive model")
        self.model_ = clone(self.estimator).fit(features[complete].to_numpy(), change[complete].to_numpy())
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> np.ndarray:
        cpu = history[TARGET].astype(float).tolist()
        if len(cpu) < FEATURE_WINDOW:
            raise ValueError("history too short for target features")
        out = []
        for _ in range(horizon):
            row = target_features(pd.Series(cpu[-FEATURE_WINDOW:])).iloc[-1].to_numpy()
            nxt = cpu[-1] + float(self.model_.predict(row[None, :])[0])
            cpu.append(nxt)
            out.append(nxt)
        return np.array(out)


def ridge() -> RegressorMixin:
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def boosting() -> RegressorMixin:
    # Shallow, slow-learning trees with large leaves: ~100 rows overfit easily.
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=10, random_state=0
    )


def default_forecasters() -> list[Forecaster]:
    return [
        Persistence(),
        MovingAverage(6),
        SeasonalNaive(24),
        RecursiveModel(ridge(), "ridge_recursive"),
        DirectModel(ridge(), "ridge_direct"),
        RecursiveModel(boosting(), "boosting_recursive"),
        DirectModel(boosting(), "boosting_direct"),
    ]
