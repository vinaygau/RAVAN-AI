#!/usr/bin/env python3
"""Produce exact additive explanations for the Phase 6 logistic model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "phase8.json"


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -700, 700)
    return 1.0 / (1.0 + np.exp(-clipped))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out_dir = ROOT / cfg["output"]
    out_dir.mkdir(parents=True, exist_ok=True)
    phase6_cfg = json.loads((ROOT / "data" / "phase6" / "training_config.json").read_text(encoding="utf-8"))
    features = list(phase6_cfg["features"])
    predictions = read_csv(ROOT / cfg["test_predictions"])
    source_rows = read_csv(ROOT / cfg["input"])
    decision_rows = read_csv(ROOT / cfg["decision_trace"])
    phase2_metadata = json.loads((ROOT / "data" / "phase2" / "metadata.json").read_text(encoding="utf-8"))
    episode_metadata = {ep["episode_id"]: ep for ep in phase2_metadata["episodes"]}
    source_by_key = {(r["episode_id"], r["step"]): r for r in source_rows}
    decision_by_key = {(r["episode_id"], r["step"]): r for r in decision_rows}
    keys = [(r["episode_id"], r["step"]) for r in predictions]
    if any(key not in source_by_key or key not in decision_by_key for key in keys):
        raise ValueError("Test prediction rows are not aligned with Phase 4 inputs and Phase 7 traces")

    model = joblib.load(ROOT / cfg["model_artifact"])
    if "scaler" not in model.named_steps or "imputer" not in model.named_steps or "classifier" not in model.named_steps:
        raise TypeError("Phase 8 exact additive decomposition requires the logistic preprocessing pipeline")
    classifier = model.named_steps["classifier"]
    if classifier.coef_.shape[0] != 1:
        raise TypeError("Expected a binary logistic regression model")
    x = np.asarray([
        [float(source_by_key[key][f]) if source_by_key[key][f] != "" else np.nan for f in features]
        for key in keys
    ], dtype=float)
    imputed = model.named_steps["imputer"].transform(x)
    scaled = model.named_steps["scaler"].transform(imputed)
    coefficients = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])
    contributions = scaled * coefficients
    reconstructed_logit = intercept + contributions.sum(axis=1)
    reconstructed_probability = sigmoid(reconstructed_logit)
    model_probability = model.predict_proba(x)[:, list(model.classes_).index(1)]
    phase6_probability = np.asarray([float(r["probability"]) for r in predictions])
    phase7_probability = np.asarray([float(decision_by_key[key]["model_probability"]) for key in keys])
    max_reconstruction_error = float(np.max(np.abs(reconstructed_probability - model_probability)))
    max_phase6_error = float(np.max(np.abs(phase6_probability - model_probability)))
    max_phase7_error = float(np.max(np.abs(phase7_probability - model_probability)))
    tolerance = float(cfg["fidelity_tolerance_probability"])
    if max_reconstruction_error > tolerance:
        raise AssertionError(f"Additive attribution reconstruction error {max_reconstruction_error} exceeds {tolerance}")
    if max_phase6_error > 2e-6 or max_phase7_error > 2e-6:
        raise AssertionError("Explanation test rows do not reproduce Phase 6/7 probabilities")

    long_rows: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []
    index_by_key = {key: i for i, key in enumerate(keys)}
    for i, (prediction, key) in enumerate(zip(predictions, keys)):
        source = source_by_key[key]
        decision = decision_by_key[key]
        for j, feature in enumerate(features):
            raw = source[feature]
            long_rows.append({
                "episode_id": key[0],
                "scenario_id": source["scenario_id"],
                "step": key[1],
                "target": prediction["target"],
                "model_probability": prediction["probability"],
                "base_log_odds": f"{intercept:.12f}",
                "feature": feature,
                "raw_value": raw,
                "imputed_value": f"{imputed[i, j]:.12g}",
                "standardized_value": f"{scaled[i, j]:.12g}",
                "coefficient": f"{coefficients[j]:.12g}",
                "contribution_log_odds": f"{contributions[i, j]:.12g}",
                "recommendation": decision["recommendation"],
                "disagreement_status": decision["disagreement_status"],
            })

    # Representative held-out classification outcomes.
    categories = {
        "true_positive": [i for i, r in enumerate(predictions) if r["target"] == "1" and r["prediction"] == "1"],
        "true_negative": [i for i, r in enumerate(predictions) if r["target"] == "0" and r["prediction"] == "0"],
        "false_positive": [i for i, r in enumerate(predictions) if r["target"] == "0" and r["prediction"] == "1"],
        "false_negative": [i for i, r in enumerate(predictions) if r["target"] == "1" and r["prediction"] == "0"],
    }
    selected_cases: dict[str, int] = {}
    for category, indexes in categories.items():
        if not indexes:
            continue
        if category in ("true_positive", "false_positive"):
            selected = max(indexes, key=lambda i: float(predictions[i]["probability"]))
        else:
            selected = min(indexes, key=lambda i: float(predictions[i]["probability"]))
        selected_cases[category] = selected
    disagreement_candidates = [
        i for i, key in enumerate(keys)
        if decision_by_key[key]["disagreement_status"] != "agreement_or_partial"
    ]
    if disagreement_candidates:
        selected_cases["model_physics_disagreement"] = max(disagreement_candidates, key=lambda i: float(predictions[i]["probability"]))

    for category, i in selected_cases.items():
        pred, key = predictions[i], keys[i]
        src, decision = source_by_key[key], decision_by_key[key]
        episode_context = episode_metadata.get(key[0], {}).get("parameters", {})
        brake_onset = episode_context.get("lead_brake_onset_s")
        order = np.argsort(np.abs(contributions[i]))[::-1]
        top = [{
            "feature": features[j],
            "raw_value": src[features[j]],
            "contribution_log_odds": float(contributions[i, j]),
            "direction": "raises score" if contributions[i, j] > 0 else "lowers score" if contributions[i, j] < 0 else "neutral",
        } for j in order[:5]]
        per_case.append({
            "case_type": category,
            "episode_id": key[0],
            "scenario_id": src["scenario_id"],
            "step": key[1],
            "target": int(pred["target"]),
            "model_probability": float(pred["probability"]),
            "phase7_recommendation": decision["recommendation"],
            "phase7_disagreement": decision["disagreement_status"],
            "scheduled_lead_brake_onset_s": brake_onset,
            "lead_brake_time_from_sample_s": None if brake_onset is None else round(float(brake_onset) - float(src["time_s"]), 6),
            "base_log_odds": intercept,
            "reconstructed_logit": float(reconstructed_logit[i]),
            "top_absolute_contributions": top,
        })

    global_features = []
    for j, feature in enumerate(features):
        global_features.append({
            "feature": feature,
            "coefficient": float(coefficients[j]),
            "mean_absolute_test_contribution_log_odds": float(np.mean(np.abs(contributions[:, j]))),
            "mean_signed_test_contribution_log_odds": float(np.mean(contributions[:, j])),
        })
    global_features.sort(key=lambda item: item["mean_absolute_test_contribution_log_odds"], reverse=True)
    summary = {
        "method": cfg["attribution_method"],
        "model": "Phase 6 selected logistic regression",
        "held_out_test_sample_count": len(predictions),
        "features": features,
        "base_log_odds_intercept": intercept,
        "probability_fidelity": {
            "maximum_reconstruction_error": max_reconstruction_error,
            "maximum_difference_from_phase6_test_predictions": max_phase6_error,
            "maximum_difference_from_phase7_trace": max_phase7_error,
            "configured_tolerance": tolerance,
        },
        "global_feature_summary": global_features,
        "representative_examples": per_case,
        "limitations": "Correlated TTC-derived features share explanatory signal. Contributions are model-relative, not causal, and are specific to synthetic labels and data.",
    }
    with (out_dir / "test_attributions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)
    (out_dir / "explanation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Phase 8 Explanation Examples", "",
        "Method: exact additive logistic-regression contributions in standardized feature space. These are not SHAP values. Positive log-odds contributions raise the score; negative contributions lower it.",
        f"Probability recomposition maximum error: {max_reconstruction_error:.3g}; difference from Phase 6 predictions: {max_phase6_error:.3g}; difference from Phase 7 traces: {max_phase7_error:.3g}.",
        "", "## Global test-set contributions", "",
    ]
    for item in global_features[:8]:
        report.append(f"- `{item['feature']}`: mean absolute contribution {item['mean_absolute_test_contribution_log_odds']:.4f} log-odds; coefficient {item['coefficient']:.4f}.")
    report += ["", "## Representative held-out cases", ""]
    for case in per_case:
        report.append(f"### {case['case_type'].replace('_', ' ').title()}")
        report.append(f"Episode {case['episode_id']} ({case['scenario_id']}), step {case['step']}: target={case['target']}, model score={case['model_probability']:.4f}, Phase 7 action=`{case['phase7_recommendation']}`, disagreement=`{case['phase7_disagreement']}`.")
        if case["scheduled_lead_brake_onset_s"] is not None:
            delta = case["lead_brake_time_from_sample_s"]
            if delta > 0:
                report.append(f"- Scenario metadata: lead braking is scheduled at {case['scheduled_lead_brake_onset_s']:.2f} s, {delta:.2f} s after this sample; the future schedule is not a model feature.")
            else:
                report.append(f"- Scenario metadata: lead braking began at {case['scheduled_lead_brake_onset_s']:.2f} s, {-delta:.2f} s before this sample; the scripted onset time is not a model feature.")
        for attribution in case["top_absolute_contributions"]:
            report.append(f"- `{attribution['feature']}` = {attribution['raw_value'] or 'undefined'}: {attribution['contribution_log_odds']:+.4f} log-odds ({attribution['direction']}).")
        report.append("")
    report += [
        "## Limits", "",
        "These contributions exactly decompose this logistic model's log-odds relative to the zero standardized reference; they do not identify causal risk factors. TTC, gap, relative speed, THW, and DRAC are correlated. The target is itself derived from future TTC/contact in a synthetic one-dimensional simulator, so these examples explain model behavior on that operational task only.", "",
    ]
    (out_dir / "explanation_examples.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary["probability_fidelity"], indent=2))


if __name__ == "__main__":
    main()
