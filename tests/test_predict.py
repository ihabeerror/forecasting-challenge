"""End-to-end run on a tiny dataset with the irregularities a reviewer's smoke data might have."""
import numpy as np
import pandas as pd

from cwfc.predict import main


def _write_smoke_data(folder, n_steps=60, gap=40):
    rng = np.random.default_rng(1)
    rows = [
        (t * 300, f"vm{v}", 0.0, 100.0, float(rng.uniform(0, 100)))
        for t in range(n_steps) if t != gap
        for v in range(4)
        if not (v == 3 and t < 10)                        # one VM starts late
    ]
    readings = pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])
    readings.loc[0, "avg_cpu"] = np.nan                   # a missing reading
    readings = pd.concat([readings, readings.iloc[[5]]])  # a duplicate row
    readings.to_csv(folder / "vm_cpu_readings.csv", index=False)
    pd.DataFrame({
        "vm_id": [f"vm{v}" for v in range(4)],
        "vm_category": ["Interactive", "Delay-insensitive", "Unknown", "Interactive"],
    }).to_csv(folder / "vm_metadata.csv", index=False)
    return sorted(readings.timestamp.unique())


def test_predict_handles_gap_in_holdout(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    timestamps = _write_smoke_data(data)
    output = tmp_path / "predictions.csv"
    main(["--data", str(data), "--output", str(output), "--reports", str(tmp_path / "reports")])

    predictions = pd.read_csv(output)
    assert list(predictions.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert predictions["timestamp"].tolist() == timestamps[-24:]
    assert np.isfinite(predictions[["active_vm_count", "aggregate_cpu_utilization"]]).all().all()
    assert predictions["aggregate_cpu_utilization"].between(0, 100).all()
    assert (predictions["active_vm_count"] >= 0).all()
