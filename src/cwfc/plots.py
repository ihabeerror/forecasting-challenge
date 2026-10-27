"""Model comparison figures, drawn from the files written by cwfc.predict.

Run: python -m cwfc.plots --reports reports
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.preprocess import STEP_SECONDS, TARGET
from cwfc.style import AQUA, BLUE, INK, MUTED, ORANGE, SECONDARY_INK, YELLOW, plt, save_figure, style_axes

# Fixed colour per baseline so it reads the same in every image; the model under comparison is always blue.
BASELINE_COLORS = {"persistence": ORANGE, "moving_average_6": AQUA, "seasonal_naive_24": YELLOW}
TRAIN_TAIL = 36


def _label(model: str) -> str:
    return f"{model} (baseline)" if model in BASELINE_COLORS else model


def plot_relative_mae(validation: pd.DataFrame, holdout: pd.DataFrame, path: Path) -> None:
    """Validation vs holdout MAE, each divided by persistence MAE."""
    models = validation.model.tolist()
    val = validation.set_index("model").mae_vs_reference.reindex(models)
    hold = holdout.set_index("model").mae_vs_persistence.reindex(models)
    y = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for offset, values, color, name in ((-0.2, val, BLUE, "validation"), (0.2, hold, ORANGE, "holdout")):
        ax.barh(y + offset, values, height=0.36, color=color, label=name)
        for yi, v in zip(y + offset, values):
            ax.text(v + 0.03, yi, f"{v:.2f}", va="center", color=INK, fontsize=8)
    ax.axvline(1.0, color=INK, linewidth=1)
    ax.set_yticks(y, [_label(m) for m in models])
    ax.invert_yaxis()
    ax.set_xlim(0, max(val.max(), hold.max()) * 1.15)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)
    style_axes(ax, "MAE relative to persistence (left of the line beats it)", "model MAE / persistence MAE", "",
               grid_axis="x")
    save_figure(fig, path)


def plot_model_vs_baselines(model: str, selected: str, history: pd.DataFrame, forecasts: pd.DataFrame,
                            by_horizon: pd.DataFrame, validation: pd.DataFrame, holdout: pd.DataFrame,
                            bands: pd.DataFrame, path: Path) -> None:
    """One model against the three baselines: holdout forecasts (left) and validation MAE by step (right)."""
    series = {**BASELINE_COLORS, model: BLUE}
    val_mae = validation.set_index("model").mae
    hold_mae = holdout.set_index("model").mae
    train = history[history.split == "train"].tail(TRAIN_TAIL)
    actual = pd.concat([train, history[history.split == "holdout"]])
    minutes = by_horizon.index * STEP_SECONDS / 60

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2))
    if model == selected:
        left.fill_between(bands.timestamp / 3600, bands[TARGET], bands.cpu_upper, color=BLUE, alpha=0.12,
                          linewidth=0, label="headroom margin")
    left.plot(actual.timestamp / 3600, actual[TARGET], color=SECONDARY_INK, linewidth=1.5, label="actual")
    for name, color in series.items():
        rows = forecasts[forecasts.model == name]
        width = 2.5 if name == model else 1.8
        left.plot(rows.timestamp / 3600, rows.forecast, color=color, linewidth=width,
                  label=f"{_label(name)}: validation MAE {val_mae[name]:.3f}, holdout MAE {hold_mae[name]:.3f}")
        right.plot(minutes, by_horizon[name], color=color, linewidth=width)
    left.axvline(train.timestamp.iloc[-1] / 3600, color=MUTED, linewidth=1, linestyle="--")

    tag = " (selected)" if model == selected else ""
    style_axes(left, f"Holdout forecast: {model}{tag} vs baselines", "hours since trace start (dashed: origin)",
               "aggregate CPU (%)")
    style_axes(right, "Validation MAE by forecast step", "minutes ahead", "MAE (CPU points)")
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.32, wspace=0.22)
    save_figure(fig, path, tight=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    reports = Path(args.reports)
    out = reports / "figures"
    out.mkdir(parents=True, exist_ok=True)
    if not (reports / "validation_scores.csv").exists():
        print("no validation scores (history too short to validate); skipping comparison figures")
        return

    validation = pd.read_csv(reports / "validation_scores.csv")
    holdout = pd.read_csv(reports / "holdout_scores.csv")
    selected = holdout.loc[holdout.selected, "model"].iloc[0]

    plot_relative_mae(validation, holdout, out / "relative_mae.png")
    print(f"wrote {out / 'relative_mae.png'}")

    history = pd.read_csv(reports / "history.csv")
    forecasts = pd.read_csv(reports / "holdout_forecasts.csv")
    by_horizon = pd.read_csv(reports / "validation_mae_by_horizon.csv", index_col="h")
    bands = pd.read_csv(reports / "forecast_bands.csv")
    models = [m for m in validation.model if m not in BASELINE_COLORS and m in set(holdout.model)]
    for model in models:
        path = out / f"compare_{model}.png"
        plot_model_vs_baselines(model, selected, history, forecasts, by_horizon, validation, holdout, bands, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
