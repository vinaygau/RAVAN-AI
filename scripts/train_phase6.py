#!/usr/bin/env python3
"""Train baseline classifiers with episode-grouped train/validation/test splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "phase4" / "trajectories_labeled.csv"
DEFAULT_CONFIG = ROOT / "config" / "phase6.json"
DEFAULT_OUTPUT = ROOT / "data" / "phase6"


def group_split(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    """Assign whole episodes to 70/15/15 while balancing scenario and episode outcome."""
    episodes: dict[str, dict[str, Any]] = {}
    for row in rows:
        eid = row["episode_id"]
        if eid not in episodes:
            episodes[eid] = {"scenario": row["scenario_id"], "positive": False}
        if row["future_conflict_eligible"] == "1" and row["future_conflict_within_3s"] == "1":
            episodes[eid]["positive"] = True
    strata: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for eid, meta in episodes.items():
        strata[(meta["scenario"], meta["positive"])].append(eid)
    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for key in sorted(strata, key=lambda item: (item[0], item[1])):
        members = sorted(strata[key])
        rng.shuffle(members)
        n = len(members)
        if n < 3:
            counts = {"train": n, "validation": 0, "test": 0}
        else:
            test_n = max(1, round(n * 0.15))
            val_n = max(1, round(n * 0.15))
            if test_n + val_n >= n:
                test_n, val_n = 1, 1
            counts = {"train": n - test_n - val_n, "validation": val_n, "test": test_n}
        cursor = 0
        for split in ("train", "validation", "test"):
            for eid in members[cursor:cursor + counts[split]]:
                assignments[eid] = split
            cursor += counts[split]
    if len(assignments) != len(episodes):
        raise RuntimeError("Not every episode received exactly one split assignment")
    return assignments


def make_models(seed: int, n_estimators: int) -> dict[str, Pipeline]:
    return {
        "DummyPrior": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DummyClassifier(strategy="prior")),
        ]),
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed)),
        ]),
        "DecisionTree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=10, random_state=seed)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators,
                class_weight="balanced_subsample",
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=seed,
            )),
        ]),
    }


def probabilities(model: Pipeline, x: np.ndarray) -> np.ndarray:
    class_index = list(model.classes_).index(1)
    return model.predict_proba(x)[:, class_index]


def metric_summary(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "sample_count": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_prevalence": float(y.mean()) if len(y) else None,
        "threshold": threshold,
        "confusion_matrix": {
            "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]),
        },
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "average_precision_pr_auc": float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else None,
        "brier_score": float(brier_score_loss(y, prob)),
        "false_positive_rate": float(matrix[0, 1] / matrix[0].sum()) if matrix[0].sum() else None,
    }


def episode_event_metrics(rows: list[dict[str, str]], probs: np.ndarray, threshold: float) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row["episode_id"]].append(i)
    positive_episodes = 0
    detected_episodes = 0
    negative_episodes = 0
    false_alarm_episodes = 0
    for indexes in grouped.values():
        actual = any(int(rows[i]["future_conflict_within_3s"]) == 1 for i in indexes)
        detected = any(probs[i] >= threshold for i in indexes)
        if actual:
            positive_episodes += 1
            detected_episodes += int(detected)
        else:
            negative_episodes += 1
            false_alarm_episodes += int(detected)
    return {
        "positive_episodes": positive_episodes,
        "detected_positive_episodes": detected_episodes,
        "event_level_recall": detected_episodes / positive_episodes if positive_episodes else None,
        "negative_episodes": negative_episodes,
        "negative_episodes_with_any_alarm": false_alarm_episodes,
        "negative_episode_alarm_rate": false_alarm_episodes / negative_episodes if negative_episodes else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    with args.input.open("r", newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    if not all_rows:
        raise ValueError("Input contains no labeled rows")
    features = list(cfg["features"])
    required = set(features) | {"episode_id", "scenario_id", cfg["future_target"], cfg["eligibility_column"]}
    missing = required - set(all_rows[0])
    if missing:
        raise ValueError(f"Input missing required columns: {sorted(missing)}")

    assignments = group_split(all_rows, int(cfg["split_seed"]))
    eligible = [r for r in all_rows if r[cfg["eligibility_column"]] == "1" and r[cfg["future_target"]] != ""]
    if not eligible:
        raise ValueError("No eligible labeled samples")
    x_all = np.array([[float(r[f]) if r[f] != "" else np.nan for f in features] for r in eligible], dtype=float)
    y_all = np.array([int(r[cfg["future_target"]]) for r in eligible], dtype=int)
    split_all = np.array([assignments[r["episode_id"]] for r in eligible])
    split_rows = {name: [r for r in eligible if assignments[r["episode_id"]] == name] for name in ("train", "validation", "test")}
    indexes = {name: np.where(split_all == name)[0] for name in ("train", "validation", "test")}
    if any(len(indexes[name]) == 0 for name in indexes):
        raise ValueError(f"One or more data splits contain no eligible rows: { {k: len(v) for k,v in indexes.items()} }")
    if any(len(np.unique(y_all[indexes[name]])) < 2 for name in ("train", "validation", "test")):
        raise ValueError("Each split must contain both classes for meaningful PR-AUC evaluation")

    seed = int(cfg["split_seed"])
    models = make_models(seed, int(cfg["random_forest_estimators"]))
    validation: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}
    val_idx = indexes["validation"]
    for name, model in models.items():
        model.fit(x_all[indexes["train"]], y_all[indexes["train"]])
        fitted[name] = model
        val_prob = probabilities(model, x_all[val_idx])
        validation[name] = metric_summary(y_all[val_idx], val_prob, float(cfg["probability_threshold"]))
    winner = max(
        validation,
        key=lambda name: (
            validation[name]["average_precision_pr_auc"] if validation[name]["average_precision_pr_auc"] is not None else -1,
            -validation[name]["brier_score"],
        ),
    )

    train_val_idx = np.concatenate([indexes["train"], indexes["validation"]])
    final_model = make_models(seed, int(cfg["random_forest_estimators"]))[winner]
    final_model.fit(x_all[train_val_idx], y_all[train_val_idx])
    test_idx = indexes["test"]
    test_prob = probabilities(final_model, x_all[test_idx])
    test_metrics = metric_summary(y_all[test_idx], test_prob, float(cfg["probability_threshold"]))
    test_metrics["episode_event_metrics"] = episode_event_metrics(split_rows["test"], test_prob, float(cfg["probability_threshold"]))
    physics_path = ROOT / "data" / "phase5" / "physics_baseline.csv"
    with physics_path.open("r", newline="", encoding="utf-8") as f:
        physics_rows = list(csv.DictReader(f))
    physics_by_key = {(r["episode_id"], r["step"]): r for r in physics_rows}
    primary_baseline_pairs: list[tuple[int, int]] = []
    secondary_baseline_pairs: list[tuple[int, int]] = []
    physics_test_rows: list[dict[str, str]] = []
    physics_test_preds: list[int] = []
    for row in split_rows["test"]:
        baseline_row = physics_by_key[(row["episode_id"], row["step"])]
        physics_test_rows.append(row)
        prediction = int(baseline_row["baseline_future_conflict_pred_3s"])
        physics_test_preds.append(prediction)
        primary_baseline_pairs.append((int(row[cfg["future_target"]]), prediction))
        if row["future_contact_eligible"] == "1":
            secondary_baseline_pairs.append((int(row["future_contact_within_3s"]), int(baseline_row["baseline_future_contact_pred_3s"])))
    physics_test_conflict_metrics = metric_summary(np.array([p[0] for p in primary_baseline_pairs]), np.array([p[1] for p in primary_baseline_pairs], dtype=float), 0.5)
    physics_test_conflict_metrics["episode_event_metrics"] = episode_event_metrics(physics_test_rows, np.array(physics_test_preds, dtype=float), 0.5)
    physics_test_contact_metrics = metric_summary(np.array([p[0] for p in secondary_baseline_pairs]), np.array([p[1] for p in secondary_baseline_pairs], dtype=float), 0.5)
    physics_baseline_test_metrics = {
        "same_test_episodes_as_selected_model": True,
        "primary_future_conflict": physics_test_conflict_metrics,
        "secondary_future_contact": physics_test_contact_metrics,
    }

    # A second distribution-shift check holds out all episodes with strong
    # lead braking (|deceleration| >= 5 m/s^2) from fitting and model selection.
    phase2_metadata_path = ROOT / "data" / "phase2" / "metadata.json"
    phase2_metadata = json.loads(phase2_metadata_path.read_text(encoding="utf-8"))
    robust_episode_ids = {
        ep["episode_id"] for ep in phase2_metadata["episodes"]
        if ep["parameters"].get(cfg["robustness_holdout"]["metadata_field"]) is not None
        and abs(float(ep["parameters"][cfg["robustness_holdout"]["metadata_field"]])) >= float(cfg["robustness_holdout"]["absolute_min_mps2"])
    }
    robust_test_rows = [r for r in eligible if r["episode_id"] in robust_episode_ids]
    robust_train_pool = [r for r in eligible if r["episode_id"] not in robust_episode_ids]
    if not robust_test_rows or not robust_train_pool:
        raise ValueError("Parameter-held-out robustness partition is empty")
    robust_assignments = group_split(robust_train_pool, seed + 1)
    robust_train_rows = [r for r in robust_train_pool if robust_assignments[r["episode_id"]] == "train"]
    robust_val_rows = [r for r in robust_train_pool if robust_assignments[r["episode_id"]] == "validation"]
    robust_x_train = np.array([[float(r[f]) if r[f] != "" else np.nan for f in features] for r in robust_train_rows], dtype=float)
    robust_y_train = np.array([int(r[cfg["future_target"]]) for r in robust_train_rows], dtype=int)
    robust_x_val = np.array([[float(r[f]) if r[f] != "" else np.nan for f in features] for r in robust_val_rows], dtype=float)
    robust_y_val = np.array([int(r[cfg["future_target"]]) for r in robust_val_rows], dtype=int)
    robust_candidates = make_models(seed + 1, int(cfg["random_forest_estimators"]))
    robust_val_metrics: dict[str, Any] = {}
    for name, model in robust_candidates.items():
        model.fit(robust_x_train, robust_y_train)
        robust_prob = probabilities(model, robust_x_val)
        robust_val_metrics[name] = metric_summary(robust_y_val, robust_prob, float(cfg["probability_threshold"]))
    robust_winner = max(
        robust_val_metrics,
        key=lambda name: (
            robust_val_metrics[name]["average_precision_pr_auc"] if robust_val_metrics[name]["average_precision_pr_auc"] is not None else -1,
            -robust_val_metrics[name]["brier_score"],
        ),
    )
    robust_fit_rows = robust_train_pool
    robust_fit_x = np.array([[float(r[f]) if r[f] != "" else np.nan for f in features] for r in robust_fit_rows], dtype=float)
    robust_fit_y = np.array([int(r[cfg["future_target"]]) for r in robust_fit_rows], dtype=int)
    robust_test_x = np.array([[float(r[f]) if r[f] != "" else np.nan for f in features] for r in robust_test_rows], dtype=float)
    robust_test_y = np.array([int(r[cfg["future_target"]]) for r in robust_test_rows], dtype=int)
    robust_final = make_models(seed + 1, int(cfg["random_forest_estimators"]))[robust_winner]
    robust_final.fit(robust_fit_x, robust_fit_y)
    robust_test_prob = probabilities(robust_final, robust_test_x)
    robustness_metrics = metric_summary(robust_test_y, robust_test_prob, float(cfg["probability_threshold"]))
    robustness_metrics["episode_event_metrics"] = episode_event_metrics(robust_test_rows, robust_test_prob, float(cfg["probability_threshold"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.out_dir / "selected_model.joblib")
    joblib.dump(robust_final, args.out_dir / "strong_braking_robustness_model.joblib")
    (args.out_dir / "training_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    package_versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    split_manifest = {
        "seed": seed,
        "split_fractions_requested": cfg["episode_split"],
        "split_unit": "episode_id; no episode appears in multiple partitions",
        "stratification": "scenario family and episode-level presence of a positive eligible target",
        "package_versions": package_versions,
        "episode_ids": {name: sorted(eid for eid, split in assignments.items() if split == name) for name in ("train", "validation", "test")},
        "episode_counts": {name: len([1 for split in assignments.values() if split == name]) for name in ("train", "validation", "test")},
        "eligible_row_counts": {name: len(indexes[name]) for name in indexes},
        "positive_row_counts": {name: int(y_all[indexes[name]].sum()) for name in indexes},
        "scenario_episode_counts": {
            name: dict(sorted(Counter(next(r["scenario_id"] for r in all_rows if r["episode_id"] == eid) for eid, split in assignments.items() if split == name).items()))
            for name in ("train", "validation", "test")
        },
    }
    (args.out_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    validation_report = {
        "selection_metric": cfg["selection_metric"],
        "selected_model": winner,
        "candidate_validation_metrics": validation,
        "features": features,
        "package_versions": package_versions,
        "future_labels_excluded_from_features": True,
        "split_unit": "episode_id",
    }
    (args.out_dir / "validation_metrics.json").write_text(json.dumps(validation_report, indent=2) + "\n", encoding="utf-8")
    test_report = {
        "selected_model": winner,
        "test_metrics": test_metrics,
        "physics_baseline_on_same_test_episodes": physics_baseline_test_metrics,
        "threshold": cfg["probability_threshold"],
        "final_fit_rows": int(len(train_val_idx)),
        "test_rows_evaluated_once": int(len(test_idx)),
        "features": features,
        "package_versions": package_versions,
        "limitations": cfg["limitations"],
    }
    (args.out_dir / "test_metrics.json").write_text(json.dumps(test_report, indent=2) + "\n", encoding="utf-8")
    robustness_report = {
        "held_out_parameter_range": f"absolute lead deceleration >= {cfg['robustness_holdout']['absolute_min_mps2']} m/s^2",
        "held_out_episode_count": len(robust_episode_ids),
        "held_out_eligible_rows": len(robust_test_rows),
        "held_out_episode_ids": sorted(robust_episode_ids),
        "no_held_out_episodes_used_for_fit_or_validation": True,
        "candidate_validation_metrics_on_other_parameter_ranges": robust_val_metrics,
        "selected_model": robust_winner,
        "fit_episode_count": len({r["episode_id"] for r in robust_fit_rows}),
        "test_metrics": robustness_metrics,
        "limitations": "Small synthetic parameter-held-out subset; not a real-world generalization test.",
        "package_versions": package_versions,
    }
    (args.out_dir / "robustness_metrics.json").write_text(json.dumps(robustness_report, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["episode_id", "scenario_id", "step", "time_s", "target", "probability", "prediction"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, prob in zip(test_idx, test_prob):
            row = eligible[int(idx)]
            writer.writerow({
                "episode_id": row["episode_id"],
                "scenario_id": row["scenario_id"],
                "step": row["step"],
                "time_s": row["time_s"],
                "target": y_all[idx],
                "probability": f"{prob:.9f}",
                "prediction": int(prob >= float(cfg["probability_threshold"])),
            })

    m = test_metrics
    cm = m["confusion_matrix"]
    report = [
        "# Phase 6 Model Evaluation", "",
        f"- Selected model: `{winner}` (highest validation PR-AUC).",
        f"- Episode split counts: {split_manifest['episode_counts']}; eligible frame counts: {split_manifest['eligible_row_counts']}.",
        f"- Validation selection metric: {validation_report['selection_metric']}.",
        "- Final model refit on train + validation episodes, then evaluated once on held-out test episodes.",
        f"- Test confusion matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}.",
        f"- Test precision={m['precision']:.4f}, recall={m['recall']:.4f}, F1={m['f1']:.4f}, PR-AUC={m['average_precision_pr_auc']:.4f}, Brier={m['brier_score']:.4f}, FPR={m['false_positive_rate']:.4f}.",
        f"- Test event-level recall: {m['episode_event_metrics']['event_level_recall']}; negative-episode alarm rate: {m['episode_event_metrics']['negative_episode_alarm_rate']}.",
        f"- Physics baseline on the same test episodes: PR-AUC={physics_test_conflict_metrics['average_precision_pr_auc']:.4f}, recall={physics_test_conflict_metrics['recall']:.4f}, FPR={physics_test_conflict_metrics['false_positive_rate']:.4f}.",
        f"- Strong-braking held-out check (|lead deceleration| >= {cfg['robustness_holdout']['absolute_min_mps2']} m/s^2): {robustness_metrics['average_precision_pr_auc']:.4f} PR-AUC, {robustness_metrics['recall']:.4f} recall over {len(robust_test_rows)} eligible frames from {len(robust_episode_ids)} held-out episodes; selected {robust_winner} using only other parameter ranges.",
        "- Candidate validation metrics are in `validation_metrics.json`; episode assignments and class counts are in `split_manifest.json`.",
        "- Future target columns, identifiers, time, and scenario family were excluded from model features.",
        "- The target is defined from future TTC/contact while current TTC and related physics features are inputs, so high scores mainly show learnability of this operational target.",
        "", "## Limits", "",
        cfg["limitations"],
        "Frame-level test metrics remain correlated within episodes. This pilot is small and synthetic; results describe only the generated scenario distribution and operational label rule. They do not establish real-world transfer, calibrated collision probability, or safety effectiveness.", "",
    ]
    (args.out_dir / "model_card.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"selected_model": winner, "validation": validation, "test": test_report, "robustness": robustness_report, "splits": split_manifest["episode_counts"]}, indent=2))


if __name__ == "__main__":
    main()
