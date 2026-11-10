#!/usr/bin/env python3
"""Score trajectory rows with the saved Contact Model v2."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "data/contact_v2/model/contact_model.joblib"
DEFAULT_INPUT = ROOT / "data/contact_v2/phase3/trajectories_features.csv"
DEFAULT_OUTPUT = ROOT / "data/contact_v2/model/predictions.csv"


def get_float(row, key):
    value = row.get(key, "")
    return float(value) if value not in (None, "") else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="CSV with episode_id, time_s, current kinematics, and derived TTC/THW/DRAC fields")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    features = bundle["features"]
    threshold = float(bundle["threshold"])
    episodes = defaultdict(list)
    with args.input.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"episode_id", "scenario_id", "time_s", "ego_speed_mps", "lead_speed_mps",
                    "ego_acceleration_mps2", "lead_acceleration_mps2", "gap_m", "relative_speed_mps",
                    "ttc_s", "ttc_valid", "thw_s", "thw_valid", "drac_mps2", "drac_valid"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
        for row in reader:
            episodes[row["episode_id"]].append(row)
    rows_out = []
    for episode_id, rows in sorted(episodes.items()):
        rows.sort(key=lambda r: float(r["time_s"]))
        by_time = {round(float(r["time_s"]), 7): r for r in rows}
        for row in rows:
            if int(row.get("contact", "0") or 0):
                continue
            t = float(row["time_s"])
            past = by_time.get(round(t - 1.0, 7))
            values = {
                "ego_speed_mps": get_float(row, "ego_speed_mps"),
                "lead_speed_mps": get_float(row, "lead_speed_mps"),
                "ego_acceleration_mps2": get_float(row, "ego_acceleration_mps2"),
                "lead_acceleration_mps2": get_float(row, "lead_acceleration_mps2"),
                "gap_m": get_float(row, "gap_m"),
                "relative_speed_mps": get_float(row, "relative_speed_mps"),
                "ttc_s": get_float(row, "ttc_s"), "ttc_valid": get_float(row, "ttc_valid"),
                "thw_s": get_float(row, "thw_s"), "thw_valid": get_float(row, "thw_valid"),
                "drac_mps2": get_float(row, "drac_mps2"), "drac_valid": get_float(row, "drac_valid"),
            }
            if past:
                values.update({
                    "ego_speed_delta_1s": values["ego_speed_mps"] - get_float(past, "ego_speed_mps"),
                    "lead_speed_delta_1s": values["lead_speed_mps"] - get_float(past, "lead_speed_mps"),
                    "gap_delta_1s": values["gap_m"] - get_float(past, "gap_m"),
                    "relative_speed_delta_1s": values["relative_speed_mps"] - get_float(past, "relative_speed_mps"),
                })
            else:
                for name in ("ego_speed_delta_1s", "lead_speed_delta_1s", "gap_delta_1s", "relative_speed_delta_1s"):
                    values[name] = float("nan")
            rows_out.append((row, [values[name] for name in features]))
    x = np.asarray([item[1] for item in rows_out], dtype=float)
    raw = bundle["model"].predict_proba(x)[:, 1]
    logits = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))[:, None]
    calibrated = bundle["calibrator"].predict_proba(logits)[:, 1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        fields = ["episode_id", "scenario_id", "time_s", "predicted_contact_probability_3s",
                  "threshold", "contact_alert", "model_version"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (row, _), probability in zip(rows_out, calibrated):
            writer.writerow({"episode_id": row["episode_id"], "scenario_id": row["scenario_id"],
                             "time_s": row["time_s"], "predicted_contact_probability_3s": f"{probability:.8f}",
                             "threshold": f"{threshold:.4f}", "contact_alert": int(probability >= threshold),
                             "model_version": "contact_model_v2"})
    print(json.dumps({"input_rows_scored": len(rows_out), "episodes": len(episodes),
                      "output": str(args.output), "threshold": threshold,
                      "model_version": "contact_model_v2"}, indent=2))


if __name__ == "__main__":
    main()
