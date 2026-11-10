#!/usr/bin/env python3
"""Run a traceable current-state TTC baseline against Phase 4 targets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "phase4" / "trajectories_labeled.csv"
DEFAULT_CONFIG = ROOT / "config" / "phase5.json"
DEFAULT_OUTPUT = ROOT / "data" / "phase5"


def current_risk_and_reason(row: dict[str, str]) -> tuple[str, str]:
    if int(row["contact"]):
        return "critical", "CURRENT_CONTACT"
    if int(row["ttc_valid"]) != 1 or row["ttc_s"] == "":
        return "low", "TTC_UNDEFINED_OR_NOT_CLOSING"
    ttc = float(row["ttc_s"])
    if ttc <= 1.5:
        return "critical", "TTC_LE_1_5S"
    if ttc <= 3.0:
        return "high", "TTC_LE_3S"
    if ttc <= 5.0:
        return "moderate", "TTC_LE_5S"
    return "low", "TTC_GT_5S"


def physics_predictions(row: dict[str, str], horizon_s: float, conflict_threshold_s: float = 3.0) -> dict[str, Any]:
    category, reason = current_risk_and_reason(row)
    if int(row["contact"]):
        conflict_prediction = ""
        contact_prediction = ""
        basis = "CURRENT_CONTACT_FUTURE_CENSORED"
        conflict_limit = horizon_s + conflict_threshold_s
        contact_limit = horizon_s
    elif int(row["ttc_valid"]) != 1 or row["ttc_s"] == "":
        conflict_prediction = 0
        contact_prediction = 0
        basis = "NO_VALID_POSITIVE_CLOSING_TTC"
        conflict_limit = horizon_s + conflict_threshold_s
        contact_limit = horizon_s
    else:
        ttc = float(row["ttc_s"])
        conflict_limit = horizon_s + conflict_threshold_s
        contact_limit = horizon_s
        conflict_prediction = int(ttc <= conflict_limit)
        contact_prediction = int(ttc <= contact_limit)
        basis = f"CURRENT_TTC_{ttc:.6f}_S"
    return {
        "baseline_current_risk_category": category,
        "baseline_reason_code": reason,
        "baseline_rule_basis": basis,
        "baseline_conflict_ttc_limit_s": f"{conflict_limit:.3f}",
        "baseline_contact_ttc_limit_s": f"{contact_limit:.3f}",
        "baseline_future_conflict_pred_3s": conflict_prediction,
        "baseline_future_contact_pred_3s": contact_prediction,
    }


def binary_metrics(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    tp = sum(y == 1 and p == 1 for y, p in pairs)
    fp = sum(y == 0 and p == 1 for y, p in pairs)
    fn = sum(y == 1 and p == 0 for y, p in pairs)
    tn = sum(y == 0 and p == 0 for y, p in pairs)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    fpr = fp / (fp + tn) if fp + tn else None
    return {
        "eligible_samples": len(pairs),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    horizon = float(cfg["future_horizon_s"])
    with args.input.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Input contains no labeled samples")

    predictions = [physics_predictions(row, horizon) for row in rows]
    output = [{**row, **prediction} for row, prediction in zip(rows, predictions)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "physics_baseline.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)

    primary_pairs = [
        (int(r["future_conflict_within_3s"]), int(r["baseline_future_conflict_pred_3s"]))
        for r in output
        if int(r["future_conflict_eligible"]) == 1
    ]
    contact_pairs = [
        (int(r["future_contact_within_3s"]), int(r["baseline_future_contact_pred_3s"]))
        for r in output
        if int(r["future_contact_eligible"]) == 1
    ]
    summary = {
        "baseline_version": "phase5.1",
        "sample_count": len(output),
        "rule_prediction_inputs": ["current gap", "current relative speed", "current TTC validity", "current contact"],
        "future_targets_used_to_generate_predictions": False,
        "primary_future_conflict_metrics": binary_metrics(primary_pairs),
        "secondary_future_contact_metrics": binary_metrics(contact_pairs),
        "reason_code_counts": dict(sorted(Counter(r["baseline_reason_code"] for r in output).items())),
        "current_category_counts": dict(sorted(Counter(r["baseline_current_risk_category"] for r in output).items())),
        "limitations": cfg["limitations"],
        "metric_unit": "frame-level; samples within each episode are correlated",
    }
    (args.out_dir / "baseline_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Phase 5 Physics Baseline Results", "",
        "## Primary future-conflict target", "",
        f"- Eligible frames: {summary['primary_future_conflict_metrics']['eligible_samples']}",
        f"- Confusion matrix: {summary['primary_future_conflict_metrics']['confusion_matrix']}",
        f"- Precision / recall / F1: {summary['primary_future_conflict_metrics']['precision']} / {summary['primary_future_conflict_metrics']['recall']} / {summary['primary_future_conflict_metrics']['f1']}",
        f"- False-positive rate: {summary['primary_future_conflict_metrics']['false_positive_rate']}",
        "", "## Secondary future-contact target", "",
        f"- Eligible frames: {summary['secondary_future_contact_metrics']['eligible_samples']}",
        f"- Confusion matrix: {summary['secondary_future_contact_metrics']['confusion_matrix']}",
        f"- Precision / recall / F1: {summary['secondary_future_contact_metrics']['precision']} / {summary['secondary_future_contact_metrics']['recall']} / {summary['secondary_future_contact_metrics']['f1']}",
        f"- False-positive rate: {summary['secondary_future_contact_metrics']['false_positive_rate']}",
        "", "## Trace and interpretation", "",
        f"- Reason codes: {summary['reason_code_counts']}.",
        "- Each row in `physics_baseline.csv` includes the current TTC basis, chosen category, rule reason, thresholds, and predictions.",
        "- Predictions were computed from current-row fields only; Phase 4 labels were used only for the metrics above.",
        "- Metrics are frame-level. Nearby frames from one episode are correlated and should not be treated as independent samples.",
        "", "## Limitations", "", cfg["limitations"],
        "The primary target contains a future TTC threshold event. This rule is a transparent constant-relative-speed comparator, not a calibrated model or a real-world safety claim.", "",
    ]
    (args.out_dir / "baseline_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
