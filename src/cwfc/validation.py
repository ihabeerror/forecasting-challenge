from __future__ import annotations


def expanding_origins(n_rows: int, horizon: int, min_train: int, step: int) -> list[int]:
    """Forecast origins for an expanding-window backtest, as history lengths.

    Origin L means: fit on rows [0, L), forecast rows [L, L + horizon).
    Origins are spaced `step` apart and the last one ends exactly at `n_rows`.
    """
    if horizon <= 0 or min_train <= 0 or step <= 0:
        raise ValueError("horizon, min_train and step must be positive")
    last = n_rows - horizon
    if last < min_train:
        return []
    return list(range(last, min_train - 1, -step))[::-1]
