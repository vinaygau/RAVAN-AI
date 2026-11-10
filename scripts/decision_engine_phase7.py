#!/usr/bin/env python3
"""Calibrate and run the transparent Phase 7 recommendation policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_phase6 import make_models

DEFAULT_CONFIG = ROOT / "config" / "phase7.json"
DEFAULT_OUTPUT = ROOT / "data" / "phase7"


def numeric(row: dict[str, str], field: str) -> float | None:
    raw = row.get(field, "")
    if raw == "":
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def calibrate_threshold(y: np.ndarray, probabilities: np.ndarray, beta: float) -> dict[str, float]:
    thresholds = sorted({0.0, 1.0, *(float(x) for x in probabilities)})
    scored = [(float(fbeta_score(y, probabilities >= t, beta=beta, zero_division=0)), t) for t in thresholds]
    score, threshold = max(scored, key=lambda pair: (pair[0], pair[1]))
    pred = probabilities >= threshold
    return {
        "threshold": float(threshold),
        "fbeta": score,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "beta": beta,
    }


def recommend(row: dict[str, str], model_probability: float | None,
              thresholds: dict[str, float], policy: dict[str, Any]) -> dict[str, str]:
    ttc = numeric(row, "ttc_s") if row.get("ttc_valid") == "1" else None
    thw = numeric(row, "thw_s") if row.get("thw_valid") == "1" else None
    contact = row.get("contact") == "1"
    speed_fields = ("ego_speed_mps", "lead_speed_mps")
    required = [numeric(row, field) for field in speed_fields]
    quality_ok = all(value is not None and value >= 0 for value in required)
    if numeric(row, "gap_m") is None or numeric(row, "relative_speed_mps") is None:
        quality_ok = False
    if contact:
        action, reason = "critical_warning", "CURRENT_CONTACT"
    elif not quality_ok:
        action, reason = "caution", "REQUIRED_INPUT_MISSING_OR_INVALID"
    elif ttc is not None and ttc <= float(policy["critical_ttc_s"]):
        action, reason = "critical_warning", "PHYSICS_TTC_CRITICAL"
    elif ttc is not None and ttc <= float(policy["high_risk_ttc_s"]):
        action, reason = "prepare_to_slow", "PHYSICS_TTC_HIGH"
    elif model_probability is not None and model_probability >= thresholds["prepare_to_slow"]:
        action, reason = "prepare_to_slow", "MODEL_SCORE_PREPARE_THRESHOLD"
    elif ttc is not None and ttc <= float(policy["moderate_risk_ttc_s"]):
        action, reason = "increase_following_gap", "PHYSICS_TTC_MODERATE"
    elif thw is not None and thw <= float(policy["short_time_headway_s"]):
        action, reason = "increase_following_gap", "PHYSICS_THW_SHORT"
    elif model_probability is not None and model_probability >= thresholds["caution"]:
        action, reason = "caution", "MODEL_SCORE_CAUTION_THRESHOLD"
    else:
        action, reason = "monitor", "NO_ACTION_THRESHOLD_MET"

    physics_high = ttc is not None and ttc <= float(policy["high_risk_ttc_s"])
    model_high = model_probability is not None and model_probability >= thresholds["caution"]
    if physics_high and not model_high:
        disagreement = "physics_high_model_low"
    elif model_high and not physics_high:
        disagreement = "model_high_physics_low"
    else:
        disagreement = "agreement_or_partial"
    return {
        "recommendation": action,
        "reason_code": reason,
        "disagreement_status": disagreement,
        "data_quality_status": "complete" if quality_ok else "incomplete_or_invalid",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    phase6_cfg = json.loads((ROOT / "data" / "phase6" / "training_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "phase6" / "split_manifest.json").read_text(encoding="utf-8"))
    validation_model_name = json.loads((ROOT / "data" / "phase6" / "validation_metrics.json").read_text(encoding="utf-8"))["selected_model"]
    model_features = list(phase6_cfg["features"])
    with (ROOT / cfg["input"]).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Calibrate on validation predictions from a model fitted only on train episodes.
    train_ids = set(manifest["episode_ids"]["train"])
    validation_ids = set(manifest["episode_ids"]["validation"])
    eligible = [r for r in rows if r["future_conflict_eligible"] == "1" and r["future_conflict_within_3s"] != ""]
    train_rows = [r for r in eligible if r["episode_id"] in train_ids]
    validation_rows = [r for r in eligible if r["episode_id"] in validation_ids]
    make_x = lambda selected: np.asarray([[float(r[f]) if r[f] != "" else np.nan for f in model_features] for r in selected], dtype=float)
    train_x, train_y = make_x(train_rows), np.asarray([int(r["future_conflict_within_3s"]) for r in train_rows])
    val_x, val_y = make_x(validation_rows), np.asarray([int(r["future_conflict_within_3s"]) for r in validation_rows])
    seed = int(phase6_cfg["split_seed"])
    calibration_model = make_models(seed, int(phase6_cfg["random_forest_estimators"]))[validation_model_name]
    calibration_model.fit(train_x, train_y)
    class_idx = list(calibration_model.classes_).index(1)
    val_probability = calibration_model.predict_proba(val_x)[:, class_idx]
    recall_cal = calibrate_threshold(val_y, val_probability, float(cfg["threshold_calibration"]["caution_fbeta"]))
    precision_cal = calibrate_threshold(val_y, val_probability, float(cfg["threshold_calibration"]["prepare_to_slow_fbeta"]))
    low, high = sorted([recall_cal["threshold"], precision_cal["threshold"]])
    thresholds = {"caution": low, "prepare_to_slow": high}
    calibration = {
        "selected_model_family": validation_model_name,
        "calibration_model_fit_episodes": len(train_ids),
        "calibration_validation_episodes": len(validation_ids),
        "validation_rows": len(validation_rows),
        "recall_weighted_f2_selection": recall_cal,
        "precision_weighted_f0_5_selection": precision_cal,
        "applied_thresholds": thresholds,
        "threshold_ordering_note": "Caution uses the lower selected threshold; prepare-to-slow uses the higher, ensuring monotonic action severity.",
        "probability_threshold_source": "validation predictions from a model trained on training episodes only",
    }

    x_all = make_x(rows)
    model = joblib.load(ROOT / cfg["model_artifact"])
    all_class_idx = list(model.classes_).index(1)
    all_probs = model.predict_proba(x_all)[:, all_class_idx]
    out_rows: list[dict[str, Any]] = []
    for row, probability in zip(rows, all_probs):
        decision = recommend(row, float(probability), thresholds, cfg["physics_policy"])
        ttc = numeric(row, "ttc_s") if row.get("ttc_valid") == "1" else None
        thw = numeric(row, "thw_s") if row.get("thw_valid") == "1" else None
        out_rows.append({
            "episode_id": row["episode_id"],
            "scenario_id": row["scenario_id"],
            "step": row["step"],
            "time_s": row["time_s"],
            "model_probability": f"{probability:.9f}",
            "current_ttc_s": "" if ttc is None else f"{ttc:.9f}",
            "current_thw_s": "" if thw is None else f"{thw:.9f}",
            "current_risk_category": row["current_risk_category"],
            "recommendation": decision["recommendation"],
            "reason_code": decision["reason_code"],
            "disagreement_status": decision["disagreement_status"],
            "data_quality_status": decision["data_quality_status"],
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "policy_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "decision_trace.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    action_counts = Counter(r["recommendation"] for r in out_rows)
    reason_counts = Counter(r["reason_code"] for r in out_rows)
    disagreement_counts = Counter(r["disagreement_status"] for r in out_rows)
    quality_counts = Counter(r["data_quality_status"] for r in out_rows)
    summary = {
        "sample_count": len(out_rows),
        "action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "disagreement_counts": dict(sorted(disagreement_counts.items())),
        "data_quality_counts": dict(sorted(quality_counts.items())),
        "thresholds": thresholds,
        "recommendations_are_advisory_only": True,
    }
    (args.out_dir / "policy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Phase 7 Decision Policy Run", "",
        f"- Samples processed: {len(out_rows)}.",
        f"- Validation-calibrated thresholds: {thresholds}.",
        f"- Recommendation counts: {dict(sorted(action_counts.items()))}.",
        f"- Disagreement counts: {dict(sorted(disagreement_counts.items()))}.",
        f"- Data quality counts: {dict(sorted(quality_counts.items()))}.",
        "- Every output row includes model probability, current TTC/THW, current risk category, recommendation, reason, disagreement, and input quality.",
        "- The output is advisory-only and makes no vehicle-control commands.",
        "", "## Limits", "", cfg["limitations"],
        "High model scores and physical indicators are related because the model target is defined from future TTC/contact. Disagreement rates should be reviewed by scenario before any user-facing integration.", "",
    ]
    (args.out_dir / "policy_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
