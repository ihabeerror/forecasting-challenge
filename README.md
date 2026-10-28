# Cloud Workload Forecasting

Forecasts `active_vm_count` and `aggregate_cpu_utilization` for the final two hours (24 five-minute steps) of the Azure VM trace in the `data-v1` release.

The assignment is in [`ASSIGNMENT.md`](ASSIGNMENT.md), the data description in [`DATASET.md`](DATASET.md), and the output contract in [`SUBMISSION.md`](SUBMISSION.md). The written analysis is in `reports/final_report.pdf`.

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,analysis]"
```

| Extra | Contents | Needed for |
|---|---|---|
| core | numpy, pandas, scikit-learn | the prediction pipeline |
| `dev` | pytest | the tests |
| `analysis` | matplotlib | the EDA and comparison figures |

## Data

Download `data-v1.tar.gz` from the [data-v1 release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1) and extract it into `challenge_data/`:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

The pipeline reads `vm_cpu_readings` and `vm_metadata` as either `.csv.gz` (as shipped) or `.csv` (if extracted), so no further unpacking is needed.

## Produce the predictions

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This writes `reports/predictions.csv`: 24 rows, with columns `timestamp,active_vm_count,aggregate_cpu_utilization`. A bare file name for `--output` goes into `--reports` (default `reports/`). A path such as `--output out/predictions.csv` is used as given.

## Optional analysis

Run these after `cwfc.predict`. They need the `analysis` extra.

```bash
python -m cwfc.eda --data challenge_data --out reports     # workload analysis (section A)
python -m cwfc.plots --reports reports                     # model vs baseline figures
python -m pytest -q                                        # tests
```

## Outputs (`reports/`, generated, not committed)

| File | Contents |
|---|---|
| `predictions.csv` | the submission forecast |
| `validation_scores.csv` | per-model MAE, RMSE, MAE relative to persistence, peak MAE, worst fold, bias and under-forecast rate across backtest windows |
| `validation_mae_by_horizon.csv` | validation MAE for each step ahead |
| `validation_errors.csv` | every backtest forecast and its error |
| `forecast_bands.csv` | the forecast plus a headroom margin (`cpu_upper`) |
| `holdout_scores.csv` | each model scored on the holdout, after the forecast is written |
| `holdout_forecasts.csv`, `history.csv` | data for the figures |
| `workload_summary.txt`, `figures/` | EDA summary and charts |

## Method

**Targets.** Per-VM readings are grouped by timestamp:
- `active_vm_count` is the number of distinct VMs reporting.
- `aggregate_cpu_utilization` is the mean of their `avg_cpu`.

**Missing readings and gaps.** The release has no null values, no duplicate (VM, timestamp) pairs and no gaps in the 5-minute cadence (`cwfc.eda` checks this), so nothing is imputed for it. The pipeline still handles them for other data, with a warning each time:
- A reading with no `avg_cpu` is dropped, so that VM does not count as active at that timestamp. A duplicate (VM, timestamp) row keeps its first reading, so each VM is counted and weighted once.
- The mean is taken over the VMs with a valid reading at each timestamp.
- A timestamp with no readings at all is added to the 5-minute grid and filled with the last observed values, which is what an operator would know at that moment. These rows are marked as not observed. They keep lags and rolling features on a regular grid, but they are never scored in validation and never used as holdout timestamps.
- The holdout is the last 24 *observed* timestamps. If a gap falls inside them, the model forecasts far enough ahead and the steps that match those timestamps are kept.
- Timestamps that are not on a 5-minute grid stop the pipeline with an error.

**Split.** The last 24 observed timestamps are the holdout and the rest is training. Holdout values are never used for fitting, feature selection, tuning or model selection. They are read only after `predictions.csv` is written, to score the result.

**Features.** Each forecast origin uses only rows at or before it:
- current CPU
- lags 1, 2 and 5
- the last change
- rolling mean and standard deviation over 6 steps
- fleet statistics observed at that timestamp: p95, spread across VMs, share above 50%, mean max CPU and per-category mean

Metadata lifetime statistics (`avg_cpu`, `max_cpu`, `p95_max_cpu`) and deletion times describe the whole VM life, so they are excluded as leakage.

Calendar features (hour, day of week) are not used. The trace covers about 11 hours, so there is no full daily or weekly cycle to learn from, and the raw timestamp would only let a model extrapolate a trend. Every feature is computed from timestamps at or before the origin.

**Models** ([`models.py`](src/cwfc/models.py)):
- Baselines: persistence, 6-step moving average and seasonal-naive (period 24).
- Ridge regression and gradient-boosted trees, each in two forms:
  - *Recursive:* a one-step model whose predictions are fed back in as inputs.
  - *Direct:* one model with the step ahead as an input.
- The learned models predict the change from the current value.

**Validation** ([`validation.py`](src/cwfc/validation.py), [`evaluation.py`](src/cwfc/evaluation.py)): an expanding-window backtest inside the training window. Each window forecasts 24 steps, the first window has at least 60 rows of history, and windows start every 6 steps. The model with the lowest validation MAE is selected and refit on all training rows. If there is too little history to validate, the pipeline falls back to persistence.

**Active VM count.** The count is forecast by persistence (the last training value). VM deletion times are not known in advance in production, so they are not used.

**Headroom.** The margin is the 95th percentile of under-forecast error in validation for the selected model. `cpu_upper = forecast + margin` is the suggested capacity line.

## Repository layout

```text
src/cwfc/
  data_files.py   locate .csv / .csv.gz inputs
  eda.py          section A workload analysis
  preprocess.py   target series, fleet statistics, origin features, split
  baselines.py    persistence, moving average, seasonal-naive
  models.py       forecaster classes (baselines, ridge, boosting; recursive and direct)
  validation.py   rolling-origin windows
  evaluation.py   backtest, summary metrics, headroom margin
  metrics.py      MAE, RMSE
  predict.py      end-to-end pipeline (entry point)
  plots.py, style.py  comparison figures and shared chart style
tests/            unit tests (leakage checks, models, baselines, validation)
reports/          generated output; final_report.pdf is committed
```

## Small datasets

The pipeline is not tied to the supplied files. It needs more than 24 timestamps (to leave a training window before the holdout). Backtesting needs more than 24 training timestamps (49 or more in total). With fewer, it falls back to persistence and the comparison figures are skipped. Models that cannot be fitted on very short history are dropped from the comparison.

## Source

Microsoft Azure Public Dataset V2: https://github.com/Azure/AzurePublicDataset
