#!/usr/bin/env python3
"""Join held-out episodes, model scores, policy traces, and explanations for the dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "phase9" / "dashboard_data.js"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    split = json.loads((ROOT / "data" / "phase6" / "split_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "data" / "phase6" / "training_config.json").read_text(encoding="utf-8"))
    model = joblib.load(ROOT / "data" / "phase6" / "selected_model.joblib")
    features = list(config["features"])
    test_ids = set(split["episode_ids"]["test"])
    states = [r for r in read_rows(ROOT / "data" / "phase4" / "trajectories_labeled.csv") if r["episode_id"] in test_ids]
    phase6_predictions = {(r["episode_id"], r["step"]): r for r in read_rows(ROOT / "data" / "phase6" / "test_predictions.csv")}
    phase7 = {(r["episode_id"], r["step"]): r for r in read_rows(ROOT / "data" / "phase7" / "decision_trace.csv")}
    phase2 = json.loads((ROOT / "data" / "phase2" / "metadata.json").read_text(encoding="utf-8"))
    episode_meta = {r["episode_id"]: r for r in phase2["episodes"]}
    phase8 = json.loads((ROOT / "data" / "phase8" / "explanation_summary.json").read_text(encoding="utf-8"))
    coeff = model.named_steps["classifier"].coef_[0]
    intercept = float(model.named_steps["classifier"].intercept_[0])
    matrix = np.asarray([[float(r[f]) if r[f] != "" else np.nan for f in features] for r in states], dtype=float)
    imputed = model.named_steps["imputer"].transform(matrix)
    scaled = model.named_steps["scaler"].transform(imputed)
    contributions = scaled * coeff
    logits = intercept + contributions.sum(axis=1)
    probs = model.predict_proba(matrix)[:, list(model.classes_).index(1)]
    max_phase7_probability_delta = 0.0
    episodes: dict[str, dict[str, Any]] = {}
    for i, state in enumerate(states):
        key = (state["episode_id"], state["step"])
        if key not in phase7:
            raise ValueError(f"Missing Phase 7 decision trace for {key}")
        policy = phase7[key]
        phase7_delta = abs(float(policy["model_probability"]) - float(probs[i]))
        max_phase7_probability_delta = max(max_phase7_probability_delta, phase7_delta)
        if key in phase6_predictions:
            phase6_prob = float(phase6_predictions[key]["probability"])
            if abs(phase6_prob - float(probs[i])) > 2e-6:
                raise ValueError(f"Model score mismatch against Phase 6 for {key}")
        if phase7_delta > 2e-6:
            raise ValueError(f"Model score mismatch against Phase 7 for {key}")

        params = episode_meta[state["episode_id"]]["parameters"]
        if state["episode_id"] not in episodes:
            episodes[state["episode_id"]] = {
                "episodeId": state["episode_id"],
                "scenarioId": state["scenario_id"],
                "seed": state["seed"],
                "parameters": params,
                "requestedDurationS": episode_meta[state["episode_id"]]["requested_duration_s"],
                "contactSampleTimeS": episode_meta[state["episode_id"]]["first_contact_sample_time_s"],
                "samples": [],
            }
        explanation_order = np.argsort(np.abs(contributions[i]))[::-1][:6]
        explanation_features = [{
            "name": features[j],
            "rawValue": None if state[features[j]] == "" else float(state[features[j]]),
            "contribution": float(contributions[i, j]),
        } for j in explanation_order]
        target = None
        if state["future_conflict_eligible"] == "1" and state["future_conflict_within_3s"] != "":
            target = int(state["future_conflict_within_3s"])
        sample = {
            "step": int(state["step"]),
            "timeS": float(state["time_s"]),
            "egoSpeedMps": float(state["ego_speed_mps"]),
            "leadSpeedMps": float(state["lead_speed_mps"]),
            "egoAccelerationMps2": float(state["ego_acceleration_mps2"]),
            "leadAccelerationMps2": float(state["lead_acceleration_mps2"]),
            "gapM": float(state["gap_m"]),
            "relativeSpeedMps": float(state["relative_speed_mps"]),
            "ttcS": None if state["ttc_s"] == "" else float(state["ttc_s"]),
            "ttcValid": int(state["ttc_valid"]),
            "thwS": None if state["thw_s"] == "" else float(state["thw_s"]),
            "thwValid": int(state["thw_valid"]),
            "dracMps2": None if state["drac_mps2"] == "" else float(state["drac_mps2"]),
            "dracValid": int(state["drac_valid"]),
            "currentRisk": state["current_risk_category"],
            "target": target,
            "score": float(probs[i]),
            "binaryPrediction": int(float(probs[i]) >= float(config["probability_threshold"])),
            "recommendation": policy["recommendation"],
            "reason": policy["reason_code"],
            "disagreement": policy["disagreement_status"],
            "dataQuality": policy["data_quality_status"],
            "logit": float(logits[i]),
            "baseLogOdds": intercept,
            "topContributions": explanation_features,
        }
        episodes[state["episode_id"]]["samples"].append(sample)

    for episode in episodes.values():
        episode["samples"].sort(key=lambda s: s["step"])
    payload = {
        "schemaVersion": "1.0",
        "source": "Phase 2 synthetic pilot; Phase 6 held-out test episodes",
        "episodeCount": len(episodes),
        "featureCount": len(features),
        "phase8MaxReconstructionError": phase8["probability_fidelity"]["maximum_reconstruction_error"],
        "maxProbabilityDeltaFromPhase7": max_phase7_probability_delta,
        "episodes": sorted(episodes.values(), key=lambda e: e["episodeId"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("window.RAVAN_DASHBOARD_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(json.dumps({
        "episode_count": len(episodes),
        "sample_count": sum(len(e["samples"]) for e in episodes.values()),
        "max_probability_delta_from_phase7": max_phase7_probability_delta,
        "attribution_features_per_sample": 6,
        "output": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
