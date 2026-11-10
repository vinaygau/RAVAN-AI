#!/usr/bin/env python3
"""Run exploratory ablations and assemble the Phase 10 evidence report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "phase10.json"
DEFAULT_OUTPUT = ROOT / "data" / "phase10"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_report(rows: list[dict[str, str]], y: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    by_episode: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_episode[row["episode_id"]].append(i)
    positive_episodes = sum(any(y[i] == 1 for i in indexes) for indexes in by_episode.values())
    detected_episodes = sum(any(y[i] == 1 for i in indexes) and any(pred[i] == 1 for i in indexes) for indexes in by_episode.values())
    return {
        "samples": len(y),
        "positive_samples": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "event_level_recall": float(detected_episodes / positive_episodes) if positive_episodes else None,
        "positive_episodes": int(positive_episodes),
        "detected_positive_episodes": int(detected_episodes),
    }


def make_matrix(rows: list[dict[str, str]], features: list[str]) -> np.ndarray:
    return np.asarray([[float(r[f]) if r.get(f, "") != "" else np.nan for f in features] for r in rows], dtype=float)


def fit_lr(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> Pipeline:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed)),
    ])
    model.fit(x_train, y_train)
    return model


def run_feature_experiment(name: str, train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], features: list[str], seed: int) -> dict[str, Any]:
    y_train = np.asarray([int(r["future_conflict_within_3s"]) for r in train_rows])
    y_test = np.asarray([int(r["future_conflict_within_3s"]) for r in test_rows])
    model = fit_lr(make_matrix(train_rows, features), y_train, seed)
    probabilities = model.predict_proba(make_matrix(test_rows, features))[:, list(model.classes_).index(1)]
    return {"name": name, "features": features, "metrics": metric_report(test_rows, y_test, probabilities)}


def add_history(rows: list[dict[str, str]], history_s: float) -> tuple[list[dict[str, str]], list[str]]:
    by_episode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    history_fields = [
        "ego_speed_1s_ago_mps", "lead_speed_1s_ago_mps", "gap_1s_ago_m",
        "relative_speed_1s_ago_mps", "ego_acceleration_1s_ago_mps2",
        "lead_acceleration_1s_ago_mps2", "gap_change_over_1s_m", "ego_speed_change_over_1s_mps",
    ]
    result: list[dict[str, str]] = []
    for episode in by_episode.values():
        episode.sort(key=lambda r: float(r["time_s"]))
        time_map = {round(float(r["time_s"]), 6): r for r in episode}
        for row in episode:
            now = float(row["time_s"])
            prior = time_map.get(round(now - history_s, 6))
            out = dict(row)
            if prior is None:
                for field in history_fields:
                    out[field] = ""
            else:
                out["ego_speed_1s_ago_mps"] = prior["ego_speed_mps"]
                out["lead_speed_1s_ago_mps"] = prior["lead_speed_mps"]
                out["gap_1s_ago_m"] = prior["gap_m"]
                out["relative_speed_1s_ago_mps"] = prior["relative_speed_mps"]
                out["ego_acceleration_1s_ago_mps2"] = prior["ego_acceleration_mps2"]
                out["lead_acceleration_1s_ago_mps2"] = prior["lead_acceleration_mps2"]
                out["gap_change_over_1s_m"] = str(float(row["gap_m"]) - float(prior["gap_m"]))
                out["ego_speed_change_over_1s_mps"] = str(float(row["ego_speed_mps"]) - float(prior["ego_speed_mps"]))
            result.append(out)
    result.sort(key=lambda r: (r["episode_id"], int(r["step"])))
    return result, history_fields


def noisy_observation(row: dict[str, str], rng: random.Random, position_sigma: float, speed_sigma: float) -> dict[str, str]:
    out = dict(row)
    ego_x = float(row["ego_front_x_m"]) + rng.gauss(0, position_sigma)
    lead_x = float(row["lead_front_x_m"]) + rng.gauss(0, position_sigma)
    ego_v = max(0.0, float(row["ego_speed_mps"]) + rng.gauss(0, speed_sigma))
    lead_v = max(0.0, float(row["lead_speed_mps"]) + rng.gauss(0, speed_sigma))
    gap = lead_x - float(row["lead_length_m"]) - ego_x
    relative = ego_v - lead_v
    out["ego_speed_mps"] = str(ego_v)
    out["lead_speed_mps"] = str(lead_v)
    out["gap_m"] = str(gap)
    out["relative_speed_mps"] = str(relative)
    ttc_valid = gap > 0 and relative > 0
    thw_valid = gap > 0 and ego_v > 0
    drac_valid = gap > 0 and relative > 0
    out["ttc_s"] = str(gap / relative) if ttc_valid else ""
    out["ttc_valid"] = str(int(ttc_valid))
    out["thw_s"] = str(gap / ego_v) if thw_valid else ""
    out["thw_valid"] = str(int(thw_valid))
    out["drac_mps2"] = str(relative * relative / (2 * gap)) if drac_valid else ""
    out["drac_valid"] = str(int(drac_valid))
    return out


def label_sensitivity(rows: list[dict[str, str]], horizons: list[float], thresholds: list[float]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["episode_id"]].append(row)
    results = []
    for horizon in horizons:
        for threshold in thresholds:
            eligible = positives = censored = 0
            for episode in groups.values():
                episode.sort(key=lambda r: float(r["time_s"]))
                end_time = float(episode[-1]["time_s"])
                times = [float(r["time_s"]) for r in episode]
                for index, row in enumerate(episode):
                    t = times[index]
                    future = [episode[j] for j in range(index + 1, len(episode)) if times[j] <= t + horizon + 1e-9]
                    positive = any(
                        int(f["contact"]) == 1 or (
                            int(f["ttc_valid"]) == 1 and f["ttc_s"] != "" and 0 < float(f["ttc_s"]) <= threshold
                        ) for f in future
                    )
                    full = end_time >= t + horizon - 1e-9
                    if positive:
                        eligible += 1
                        positives += 1
                    elif full:
                        eligible += 1
                    else:
                        censored += 1
            results.append({
                "horizon_s": horizon,
                "future_ttc_threshold_s": threshold,
                "eligible_samples": eligible,
                "positive_samples": positives,
                "censored_samples": censored,
                "positive_prevalence": positives / eligible if eligible else None,
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    phase6_cfg = json.loads((ROOT / "data" / "phase6" / "training_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "phase6" / "split_manifest.json").read_text(encoding="utf-8"))
    rows = read_csv(ROOT / "data" / "phase4" / "trajectories_labeled.csv")
    eligible = [r for r in rows if r["future_conflict_eligible"] == "1" and r["future_conflict_within_3s"] != ""]
    trainval_ids = set(manifest["episode_ids"]["train"]) | set(manifest["episode_ids"]["validation"])
    test_ids = set(manifest["episode_ids"]["test"])
    if trainval_ids & test_ids:
        raise ValueError("Phase 6 test episodes overlap with model fitting episodes")
    train_rows = [r for r in eligible if r["episode_id"] in trainval_ids]
    test_rows = [r for r in eligible if r["episode_id"] in test_ids]
    seed = int(phase6_cfg["split_seed"])
    basic_features = list(cfg["feature_ablation_basic"])
    safety_features = list(cfg["feature_ablation_safety"])
    feature_ablation = [
        run_feature_experiment("basic_state", train_rows, test_rows, basic_features, seed),
        run_feature_experiment("safety_indicators", train_rows, test_rows, safety_features, seed),
    ]

    enriched_all, history_features = add_history(eligible, float(cfg["past_history_seconds"]))
    enriched_by_key = {(r["episode_id"], r["step"]): r for r in enriched_all}
    train_hist = [enriched_by_key[(r["episode_id"], r["step"])] for r in train_rows]
    test_hist = [enriched_by_key[(r["episode_id"], r["step"])] for r in test_rows]
    history_ablation = [
        run_feature_experiment("snapshot_only", train_rows, test_rows, safety_features, seed),
        run_feature_experiment("snapshot_plus_past_1s", train_hist, test_hist, safety_features + history_features, seed),
    ]

    noise = cfg["observation_noise"]
    clean_model = fit_lr(make_matrix(train_rows, safety_features), np.asarray([int(r["future_conflict_within_3s"]) for r in train_rows]), seed)
    y_test = np.asarray([int(r["future_conflict_within_3s"]) for r in test_rows])
    clean_test_matrix = make_matrix(test_rows, safety_features)
    clean_prob = clean_model.predict_proba(clean_test_matrix)[:, list(clean_model.classes_).index(1)]
    rng = random.Random(int(noise["seed"]))
    noisy_rows = [noisy_observation(r, rng, float(noise["position_sigma_m"]), float(noise["speed_sigma_mps"])) for r in test_rows]
    noisy_matrix = make_matrix(noisy_rows, safety_features)
    noisy_prob = clean_model.predict_proba(noisy_matrix)[:, list(clean_model.classes_).index(1)]
    missing_rng = np.random.default_rng(int(noise["seed"]) + 1)
    missing_matrix = clean_test_matrix.copy()
    missing_mask = missing_rng.random(missing_matrix.shape) < float(noise["missingness_rate"])
    missing_matrix[missing_mask] = np.nan
    missing_prob = clean_model.predict_proba(missing_matrix)[:, list(clean_model.classes_).index(1)]
    robustness = {
        "clean": metric_report(test_rows, y_test, clean_prob),
        "noisy_positions_and_speeds": metric_report(test_rows, y_test, noisy_prob),
        "random_feature_missingness": metric_report(test_rows, y_test, missing_prob),
        "noise_settings": noise,
        "missing_feature_cell_rate_realized": float(np.isnan(missing_matrix).sum() / missing_matrix.size),
        "labels_kept_from_clean_trajectory": True,
        "model_fit_on_clean_trainval_only": True,
    }

    # Descriptive decision-layer review on the same held-out test episodes.
    phase7 = {(r["episode_id"], r["step"]): r for r in read_csv(ROOT / "data" / "phase7" / "decision_trace.csv")}
    integrated_actions = Counter()
    physics_actions = Counter()
    action_positive = defaultdict(lambda: [0, 0])
    disagreements = Counter()
    for row in [r for r in rows if r["episode_id"] in test_ids]:
        key = (row["episode_id"], row["step"])
        trace = phase7[key]
        integrated = trace["recommendation"]
        ttc = float(row["ttc_s"]) if row["ttc_valid"] == "1" and row["ttc_s"] else None
        thw = float(row["thw_s"]) if row["thw_valid"] == "1" and row["thw_s"] else None
        if row["contact"] == "1" or (ttc is not None and ttc <= 1.5):
            physics = "critical_warning"
        elif ttc is not None and ttc <= 3.0:
            physics = "prepare_to_slow"
        elif (ttc is not None and ttc <= 5.0) or (thw is not None and thw <= 2.0):
            physics = "increase_following_gap"
        else:
            physics = "monitor"
        integrated_actions[integrated] += 1
        physics_actions[physics] += 1
        disagreements[trace["disagreement_status"]] += 1
        if row["future_conflict_eligible"] == "1" and row["future_conflict_within_3s"] != "":
            action_positive[integrated][1] += 1
            action_positive[integrated][0] += int(row["future_conflict_within_3s"])
    decision_review = {
        "integrated_action_counts_all_test_samples": dict(sorted(integrated_actions.items())),
        "physics_only_action_counts_all_test_samples": dict(sorted(physics_actions.items())),
        "integrated_action_positive_fraction_on_eligible_samples": {
            action: {"positive": counts[0], "eligible": counts[1], "fraction": counts[0] / counts[1] if counts[1] else None}
            for action, counts in sorted(action_positive.items())
        },
        "model_physics_disagreement_counts": dict(sorted(disagreements.items())),
        "recommendation_relevance_validated_by_humans": False,
    }

    e1 = json.loads((ROOT / "data" / "phase6" / "test_metrics.json").read_text(encoding="utf-8"))
    e4 = json.loads((ROOT / "data" / "phase6" / "robustness_metrics.json").read_text(encoding="utf-8"))
    e8 = json.loads((ROOT / "data" / "phase8" / "explanation_summary.json").read_text(encoding="utf-8"))
    sensitivity = label_sensitivity(rows, cfg["label_sensitivity"]["horizons_s"], cfg["label_sensitivity"]["future_ttc_thresholds_s"])
    phase_matrix = [
        {"phase": 1, "deliverable": "Scenario specification, parameter envelope, acceptance cases", "evidence": ["phase1/scenario_spec.md", "config/phase1.json", "tests/phase1_cases.csv"], "status": "complete"},
        {"phase": 2, "deliverable": "Synthetic generator, episode dataset, metadata, quality report", "evidence": ["scripts/generate_phase2.py", "data/phase2/trajectories.csv", "data/phase2/metadata.json", "data/phase2/quality_report.md"], "status": "complete; 140 episodes and 11,524 samples; synthetic-only"},
        {"phase": 3, "deliverable": "Preprocessing, feature dictionary, hand-calculated checks", "evidence": ["scripts/preprocess_phase3.py", "data/phase3/trajectories_features.csv", "data/phase3/feature_dictionary.md", "tests/test_phase3_features.py"], "status": "complete; 5 unit tests passed"},
        {"phase": 4, "deliverable": "Versioned target protocol, labels and audit", "evidence": ["phase4/labeling_protocol.md", "scripts/label_phase4.py", "data/phase4/trajectories_labeled.csv", "data/phase4/label_audit.md", "tests/test_phase4_labeling.py"], "status": "complete; 6 unit tests passed; surrogate labels"},
        {"phase": 5, "deliverable": "Physics baseline, rule traces and evaluation", "evidence": ["phase5/physics_baseline_protocol.md", "scripts/physics_baseline_phase5.py", "data/phase5/physics_baseline.csv", "tests/test_phase5_baseline.py"], "status": "complete; 5 unit tests passed"},
        {"phase": 6, "deliverable": "Grouped model comparison and held-out evaluation", "evidence": ["scripts/train_phase6.py", "data/phase6/selected_model.joblib", "data/phase6/test_metrics.json", "data/phase6/robustness_metrics.json"], "status": "complete; small synthetic holdouts"},
        {"phase": 7, "deliverable": "Decision policy, calibrated thresholds and traces", "evidence": ["phase7/decision_policy.md", "scripts/decision_engine_phase7.py", "data/phase7/decision_trace.csv", "tests/test_phase7_policy.py"], "status": "complete; 7 policy tests passed; advisory only"},
        {"phase": 8, "deliverable": "Faithful model explanation and examples", "evidence": ["phase8/explainability_protocol.md", "scripts/explain_phase8.py", "data/phase8/explanation_examples.md", "data/phase8/explanation_summary.json", "tests/test_phase8_explanations.py"], "status": "complete; exact logistic decomposition instead of SHAP; 2 checks passed"},
        {"phase": 9, "deliverable": "Held-out trajectory dashboard and integration checks", "evidence": ["dashboard/index.html", "scripts/build_phase9_dashboard.py", "data/phase9/dashboard_data.js", "tests/test_phase9_integration.py"], "status": "complete; 4 integration checks and JS syntax passed; browser visual review stopped by Escape"},
        {"phase": 10, "deliverable": "Evaluation report, ablations, limitations and conclusions", "evidence": ["config/phase10.json", "scripts/run_phase10.py", "data/phase10/experiment_results.json", "data/phase10/evaluation_report.md"], "status": "complete; ablations exploratory on previously examined holdout"},
    ]
    package_versions = {"python": sys.version.split()[0], "numpy": np.__version__, "scikit_learn": sklearn.__version__}
    result = {
        "phase": 10,
        "project_phase_completion_matrix": phase_matrix,
        "dataset": {"episodes": 140, "samples": len(rows), "source": "synthetic one-dimensional Phase 2 trajectories"},
        "primary_test_split": {
            "train_and_validation_episodes": len(trainval_ids),
            "test_episodes": len(test_ids),
            "test_episode_ids": sorted(test_ids),
            "test_eligible_rows": len(test_rows),
            "split_manifest": "data/phase6/split_manifest.json",
        },
        "E1_physics_vs_ml_same_test": {
            "selected_ml": e1["test_metrics"],
            "physics_rule": e1["physics_baseline_on_same_test_episodes"]["primary_future_conflict"],
        },
        "E2_feature_ablation": feature_ablation,
        "E3_temporal_history_ablation": history_ablation,
        "E4_parameter_held_out": e4,
        "E5_observation_robustness": robustness,
        "E6_decision_layer_review": decision_review,
        "E7_label_sensitivity": sensitivity,
        "E8_explanation_fidelity": e8["probability_fidelity"],
        "interpretation": "Phase 10 ablations are exploratory and reuse the Phase 6 held-out episodes after the primary result was examined. They do not support confirmatory claims.",
        "package_versions": package_versions,
        "limits": [
            "Synthetic, one-dimensional, same-lane kinematics; no CARLA or real-world data.",
            "The target is defined from future TTC/contact and current TTC-derived features are model inputs; high scores mainly show learnability of the operational label rule.",
            "Frames within an episode are correlated; test contains only 21 episodes and no confidence intervals were estimated.",
            "The strong-braking holdout has only 14 episodes. Noise and missingness are illustrative perturbations, not sensor models.",
            "Recommendation relevance has not been evaluated by drivers or domain experts; no actuation or safety effectiveness is demonstrated.",
            "Exact additive Phase 8 explanations are model-relative, not causal."
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "experiment_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    f2 = {x["name"]: x["metrics"] for x in feature_ablation}
    h3 = {x["name"]: x["metrics"] for x in history_ablation}
    report = [
        "# Phase 10 Evaluation Report", "",
        "## Scope and evaluation discipline", "",
        f"The pilot contains {len(rows)} labeled samples from 140 synthetic episodes. The primary held-out comparison uses 21 test episodes ({len(test_rows)} eligible rows). Phase 10 ablations reuse this holdout after Phase 6's primary result was already examined; they are exploratory. All preprocessing/model fitting for feature ablations uses only the existing train and validation episodes. The test episode IDs are unchanged.",
        "", "## E1 Physics baseline versus ML", "",
        f"The selected logistic model achieved test PR-AUC {e1['test_metrics']['average_precision_pr_auc']:.4f}, recall {e1['test_metrics']['recall']:.4f}, and FPR {e1['test_metrics']['false_positive_rate']:.4f}. The physics rule on the same episodes achieved PR-AUC {e1['physics_baseline_on_same_test_episodes']['primary_future_conflict']['average_precision_pr_auc']:.4f}, recall {e1['physics_baseline_on_same_test_episodes']['primary_future_conflict']['recall']:.4f}, and FPR {e1['physics_baseline_on_same_test_episodes']['primary_future_conflict']['false_positive_rate']:.4f}.",
        "", "## E2 Feature ablation", "",
        f"Basic-state features: PR-AUC {f2['basic_state']['pr_auc_average_precision']:.4f}, recall {f2['basic_state']['recall']:.4f}; full safety features: PR-AUC {f2['safety_indicators']['pr_auc_average_precision']:.4f}, recall {f2['safety_indicators']['recall']:.4f}.",
        "The basic-state and safety-feature models use the same Logistic Regression estimator, split, and threshold. See `experiment_results.json` for their complete metrics and feature lists.",
        "", "## E3 Past-only history ablation", "",
        f"Snapshot-only PR-AUC {h3['snapshot_only']['pr_auc_average_precision']:.4f}; snapshot plus 1-second lag features PR-AUC {h3['snapshot_plus_past_1s']['pr_auc_average_precision']:.4f} (absolute change {h3['snapshot_plus_past_1s']['pr_auc_average_precision']-h3['snapshot_only']['pr_auc_average_precision']:+.4f}). Lag values are sampled from t−1 s within the same episode; early samples are imputed using training data. This small change on 21 test episodes is not strong evidence that history improves generalization.",
        "", "## E4 Parameter-held-out check", "",
        f"Strong lead braking (absolute deceleration >= 5 m/s²) was excluded from fitting and selection: {e4['held_out_episode_count']} episodes, {e4['held_out_eligible_rows']} eligible rows. PR-AUC {e4['test_metrics']['average_precision_pr_auc']:.4f}; recall {e4['test_metrics']['recall']:.4f}. The small episode count limits confidence.",
        "", "## E5 Observation robustness", "",
        f"Clean held-out PR-AUC {robustness['clean']['pr_auc_average_precision']:.4f}; with independent position noise σ={noise['position_sigma_m']} m and speed noise σ={noise['speed_sigma_mps']} m/s, {robustness['noisy_positions_and_speeds']['pr_auc_average_precision']:.4f}; with an additional {noise['missingness_rate']:.0%} random feature-cell missingness, {robustness['random_feature_missingness']['pr_auc_average_precision']:.4f}. Total missing cells in that test matrix were {robustness['missing_feature_cell_rate_realized']:.1%}, including naturally undefined TTC/THW/DRAC. Labels remain from clean trajectories; these perturbations are sensitivity probes, not realistic sensor models.",
        "", "## E6 Decision-layer review", "",
        f"Integrated recommendations on held-out samples: {dict(sorted(integrated_actions.items()))}. Physics-only action counts: {dict(sorted(physics_actions.items()))}. Model/physics disagreement counts: {dict(sorted(disagreements.items()))}. These descriptive counts do not measure recommendation relevance; no human rubric or driver study was run.",
        "", "## E7 Label sensitivity", "",
        "Changing the TTC event threshold and horizon changes eligible coverage and positive prevalence. The complete 3×3 comparison is in `experiment_results.json`; these are label-distribution audits and do not retrain or re-evaluate models under each alternative target.",
        "", "## E8 Explanation review", "",
        f"The exact additive Phase 8 explanation reconstructs test probabilities within {e8['probability_fidelity']['maximum_reconstruction_error']:.3g} absolute probability error and matches the Phase 6 prediction output within {e8['probability_fidelity']['maximum_difference_from_phase6_test_predictions']:.3g}.",
        "", "## Project phase completion matrix", "",
    ]
    for phase in phase_matrix:
        evidence = ", ".join(f"`{path}`" for path in phase["evidence"])
        report.append(f"- Phase {phase['phase']}: {phase['deliverable']} — {phase['status']}. Evidence: {evidence}.")
    report += [
        "", "## Conclusions supported by this pilot", "",
        "- The end-to-end synthetic pipeline runs reproducibly from scenario generation through feature engineering, labeling, a transparent physics baseline, model scoring, recommendations, and local explanations.",
        "- On the current synthetic future-TTC/contact label, the Logistic Regression model outperformed the current-state physics rule on the same held-out episodes. This is evidence about this operational target and scenario generator only.",
        "- Feature and history ablations, observation sensitivity, and alternative-label prevalence are documented as exploratory diagnostics in `experiment_results.json`.",
        "", "## Claims not supported", "",
        "This pilot does not establish real-world collision probability, crash reduction, safety certification, driver acceptance, recommendation relevance, sensor robustness, or transfer to CARLA/road traffic. The target is generated from future TTC/contact and closely related indicators are model inputs. High scores can therefore reflect learning the target construction. The 21-episode main test and 14-episode hard-braking holdout are small, and frame-level metrics are correlated.",
        "", "## Reproduction", "",
        "Run `python scripts/run_phase10.py` from the project root. The held-out test episode IDs are preserved in `data/phase6/split_manifest.json` and copied into `experiment_results.json`; seeds, perturbation settings, feature sets, metrics, software versions, and caveats are recorded in the config/results files.", "",
    ]
    (args.out_dir / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "feature_ablation": {k: v["pr_auc_average_precision"] for k, v in f2.items()},
        "history_ablation": {k: v["pr_auc_average_precision"] for k, v in h3.items()},
        "noise_pr_auc": {k: v["pr_auc_average_precision"] for k, v in robustness.items() if isinstance(v, dict) and "pr_auc_average_precision" in v},
        "label_sensitivity_cases": len(sensitivity),
        "report": str(args.out_dir / "evaluation_report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
