"""Section A workload analysis: prints a summary and writes figures.

Run: python -m cwfc.eda --data challenge_data --out reports
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.data_files import data_file
from cwfc.style import AQUA, BLUE, GRID, INK, ORANGE, SURFACE, plt
from cwfc.style import save_figure as _save
from cwfc.style import style_axes as _style

STEP_SECONDS = 300


def coverage_report(readings: pd.DataFrame) -> list[str]:
    """Cadence, gaps, duplicates and missingness. Structure only, no CPU values."""
    ts = np.sort(readings["timestamp"].unique())
    per_ts = readings.groupby("timestamp").size()
    per_vm = readings.groupby("vm_id").size()
    return [
        "== cadence and coverage ==",
        f"rows={len(readings)} vms={readings['vm_id'].nunique()} timestamps={len(ts)}",
        f"first={ts[0]}s last={ts[-1]}s step(s)={sorted(set(np.diff(ts).tolist()))} (expected {STEP_SECONDS})",
        f"nulls={int(readings.isna().sum().sum())} duplicate(vm,ts)={int(readings.duplicated(['vm_id', 'timestamp']).sum())}",
        f"vms per timestamp min/max={per_ts.min()}/{per_ts.max()}",
        f"readings per vm min/max={per_vm.min()}/{per_vm.max()}",
        "no resampling or imputation applied; native 5-minute grid",
    ]


def churn_report(metadata: pd.DataFrame, window_start: int, window_end: int) -> list[str]:
    created = metadata.timestamp_vm_created
    deleted = metadata.timestamp_vm_deleted
    return [
        "",
        f"== VM lifetimes and churn (window {window_start}s-{window_end}s) ==",
        f"created at/before window start: {(created <= window_start).sum()} of {len(metadata)}",
        f"created inside window: {((created > window_start) & (created <= window_end)).sum()}",
        f"deleted inside window: {(deleted <= window_end).sum()}",
        f"planned lifetime quantiles (hours): {((deleted - created) / 3600).quantile([0.1, 0.5, 0.9]).round(1).to_dict()}",
    ]


def distribution_report(readings: pd.DataFrame) -> list[str]:
    q = [0.5, 0.75, 0.9, 0.95, 0.99]
    vm_mean = readings.groupby("vm_id").avg_cpu.mean()
    return [
        "",
        "== CPU utilization distribution ==",
        f"reading quantiles: {readings.avg_cpu.quantile(q).round(2).to_dict()}",
        f"per-VM mean quantiles: {vm_mean.quantile(q).round(2).to_dict()}",
        f"VMs with mean < 5%: {(vm_mean < 5).mean():.1%}   mean > 50%: {(vm_mean > 50).mean():.1%}",
    ]


def concentration(readings: pd.DataFrame) -> pd.Series:
    """Share of total CPU consumed by the top x% of VMs."""
    totals = readings.groupby("vm_id").avg_cpu.sum().sort_values(ascending=False)
    cum = totals.cumsum() / totals.sum()
    n = len(totals)
    return pd.Series({f"top {p}%": cum.iloc[int(n * p / 100) - 1] for p in (1, 5, 10, 25, 50)})


def temporal_report(mean_cpu: pd.Series) -> list[str]:
    acf = [mean_cpu.autocorr(k) for k in (1, 2, 3, 6, 12, 24)]
    return [
        "",
        "== temporal pattern (mean CPU per timestamp) ==",
        f"min={mean_cpu.min():.2f} max={mean_cpu.max():.2f} std={mean_cpu.std():.3f} mean={mean_cpu.mean():.2f}",
        f"peak 3 timestamps: {mean_cpu.nlargest(3).round(2).to_dict()}",
        f"low 3 timestamps: {mean_cpu.nsmallest(3).round(2).to_dict()}",
        f"autocorr lags 1,2,3,6,12,24: {np.round(acf, 3).tolist()}",
    ]


def segment_report(readings: pd.DataFrame, metadata: pd.DataFrame) -> list[str]:
    merged = readings.merge(metadata[["vm_id", "vm_category", "virtual_core_bucket"]], on="vm_id")
    out = ["", "== segments =="]
    for col in ("vm_category", "virtual_core_bucket"):
        stats = merged.groupby(col).avg_cpu.agg(["mean", "median", "std", "count"]).round(2)
        out += [f"-- by {col} --", stats.to_string()]
    return out


def group_contribution_report(readings: pd.DataFrame, metadata: pd.DataFrame, top: int = 8) -> list[str]:
    """How much each subscription / deployment contributes to the aggregate CPU.

    A group's contribution at time t is the sum of its VMs' avg_cpu divided by the
    fleet size, so contributions add up to the aggregate. `share_of_cpu` is the
    group's slice of the aggregate level; `share_of_variation` is its slice of the
    aggregate's movement over time (covariance with the aggregate / its variance,
    which sums to 100% across groups).
    """
    merged = readings.merge(metadata[["vm_id", "subscription_id", "deployment_id"]], on="vm_id")
    n_vms = readings.vm_id.nunique()
    agg = readings.groupby("timestamp").avg_cpu.mean()
    agg_dev = agg - agg.mean()
    out = ["", "== group contribution to aggregate CPU =="]
    for col in ("subscription_id", "deployment_id"):
        contrib = merged.groupby(["timestamp", col]).avg_cpu.sum().unstack(fill_value=0) / n_vms
        sizes = metadata.groupby(col).size()
        table = pd.DataFrame({
            "vms": sizes,
            "mean_cpu": contrib.mean() * n_vms / sizes,
            "share_of_cpu": contrib.mean() / agg.mean(),
            "share_of_variation": (contrib - contrib.mean()).mul(agg_dev, axis=0).sum() / (len(agg) - 1) / agg.var(),
        })
        ranked = table.sort_values("share_of_cpu", ascending=False)
        cum = ranked.share_of_cpu.cumsum()
        n = len(ranked)
        conc = {f"top {p}%": round(float(cum.iloc[max(int(n * p / 100), 1) - 1]), 3) for p in (1, 5, 10, 25)}
        shown = ranked.head(top).copy()
        shown.index = [str(i)[:8] for i in shown.index]
        out += [
            f"-- {col}: {n} groups, share of aggregate CPU held by top x% of groups --",
            str(conc),
            f"top {top} by share of CPU:",
            shown.round(3).to_string(),
            f"top {top} by share of variation:",
            _short_index(table.sort_values("share_of_variation", ascending=False).head(top)).round(3).to_string(),
        ]
    return out


def _entropy(counts: pd.Series) -> float:
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def id_relationship_report(metadata: pd.DataFrame) -> list[str]:
    """How redundant subscription_id and deployment_id are with each other."""
    pairs = metadata.groupby(["subscription_id", "deployment_id"]).size()
    n_sub = metadata.subscription_id.nunique()
    n_dep = metadata.deployment_id.nunique()
    deps_per_sub = metadata.groupby("subscription_id").deployment_id.nunique()
    subs_per_dep = metadata.groupby("deployment_id").subscription_id.nunique()
    vm_deps = metadata.subscription_id.map(deps_per_sub)
    vm_subs = metadata.deployment_id.map(subs_per_dep)
    h_sub = _entropy(metadata.groupby("subscription_id").size())
    h_dep = _entropy(metadata.groupby("deployment_id").size())
    h_joint = _entropy(pairs)
    nmi = 2 * (h_sub + h_dep - h_joint) / (h_sub + h_dep)
    return [
        "",
        "== subscription vs deployment relationship ==",
        f"subscriptions={n_sub} deployments={n_dep} distinct pairs={len(pairs)} "
        f"(identical grouping if all three are equal)",
        f"deployments per subscription: {deps_per_sub.value_counts().sort_index().to_dict()}",
        f"subscriptions per deployment: {subs_per_dep.value_counts().sort_index().to_dict()}",
        f"VMs whose subscription has 1 deployment: {(vm_deps == 1).mean():.1%}; "
        f"whose deployment has 1 subscription: {(vm_subs == 1).mean():.1%}",
        f"normalized mutual information: {nmi:.3f} (1.0 = one determines the other)",
    ]


def _short_index(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = [str(i)[:8] for i in frame.index]
    return frame


def plot_mean_cpu(mean_cpu: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(mean_cpu.index / 3600, mean_cpu.values, color=BLUE, linewidth=2)
    _style(ax, "Mean CPU per timestamp", "hours since trace start", "mean avg_cpu (%)")
    _save(fig, path)


def plot_distribution(readings: pd.DataFrame, path: Path) -> None:
    vm_mean = readings.groupby("vm_id").avg_cpu.mean()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(vm_mean, bins=np.arange(0, 101, 2), color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.set_yscale("log")
    for q, label in ((0.5, "median"), (0.95, "p95")):
        x = vm_mean.quantile(q)
        ax.axvline(x, color=ORANGE, linewidth=1.5)
        ax.text(x + 1, ax.get_ylim()[1], f"{label} {x:.1f}%", va="top", color=INK, fontsize=9)
    _style(ax, "Distribution of per-VM mean CPU", "per-VM mean avg_cpu (%)", "VMs (log scale)")
    _save(fig, path)


def plot_segments(readings: pd.DataFrame, metadata: pd.DataFrame, path: Path) -> None:
    merged = readings.merge(metadata[["vm_id", "vm_category"]], on="vm_id")
    by_cat = merged.groupby(["timestamp", "vm_category"]).avg_cpu.mean().unstack()
    colors = {"Interactive": BLUE, "Delay-insensitive": ORANGE, "Unknown": AQUA}
    hours = by_cat.index / 3600
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for name, color in colors.items():
        ax.plot(hours, by_cat[name], color=color, linewidth=2, label=name)
        ax.text(hours[-1] + 0.1, by_cat[name].iloc[-1], name, color=INK, va="center", fontsize=9)
    ax.set_xlim(right=hours[-1] + 3)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9)
    _style(ax, "Mean CPU by VM category", "hours since trace start", "mean avg_cpu (%)")
    _save(fig, path)


def plot_concentration(share: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.bar(share.index, share.values * 100, color=BLUE, width=0.6)
    for bar, v in zip(bars, share.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 100 + 1, f"{v:.0%}", ha="center", color=INK, fontsize=9)
    ax.set_ylim(0, 105)
    _style(ax, "Share of total CPU by top-ranked VMs", "VMs ranked by total CPU", "share of all CPU (%)")
    _save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    readings = pd.read_csv(data_file(data, "vm_cpu_readings")).sort_values(["timestamp", "vm_id"])
    metadata = pd.read_csv(data_file(data, "vm_metadata"))

    mean_cpu = readings.groupby("timestamp").avg_cpu.mean()
    share = concentration(readings)

    lines = (coverage_report(readings)
             + churn_report(metadata, int(readings.timestamp.min()), int(readings.timestamp.max()))
             + distribution_report(readings) + temporal_report(mean_cpu)
             + segment_report(readings, metadata)
             + id_relationship_report(metadata)
             + group_contribution_report(readings, metadata)
             + ["", "== concentration (share of all CPU) ==", share.round(3).to_string()])
    text = "\n".join(lines)
    summary_path = out / "workload_summary.txt"
    summary_path.write_text(text, encoding="utf-8")
    print(f"wrote {summary_path}")

    figures = {
        "mean_cpu.png": lambda p: plot_mean_cpu(mean_cpu, p),
        "cpu_distribution.png": lambda p: plot_distribution(readings, p),
        "segments.png": lambda p: plot_segments(readings, metadata, p),
        "concentration.png": lambda p: plot_concentration(share, p),
    }
    for name, draw in figures.items():
        draw(out / "figures" / name)
        print(f"wrote {out / 'figures' / name}")


if __name__ == "__main__":
    main()
