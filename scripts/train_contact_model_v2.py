#!/usr/bin/env python3
"""Train and evaluate episode-split models for future simulated contact."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/contact_v2/phase4/trajectories_labeled.csv"
OUT = ROOT / "data/contact_v2/model"
FEATURES = [
    "ego_speed_mps", "lead_speed_mps", "ego_acceleration_mps2", "lead_acceleration_mps2",
    "gap_m", "relative_speed_mps", "ttc_s", "ttc_valid", "thw_s", "thw_valid",
    "drac_mps2", "drac_valid", "ego_speed_delta_1s", "lead_speed_delta_1s",
    "gap_delta_1s", "relative_speed_delta_1s",
]
SEED = 20261110


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else float("nan")


def read_dataset(path: Path):
    episodes: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            episodes[row["episode_id"]].append(row)
    records = []
    for episode_id, rows in episodes.items():
        rows.sort(key=lambda r: float(r["time_s"]))
        by_time = {round(float(r["time_s"]), 7): r for r in rows}
        for row in rows:
            if row["future_contact_eligible"] != "1":
                continue
            t = float(row["time_s"])
            past = by_time.get(round(t - 1.0, 7))
            vals = {k: number(row, k) for k in FEATURES if not k.endswith("_delta_1s")}
            if past:
                vals.update({
                    "ego_speed_delta_1s": number(row, "ego_speed_mps") - number(past, "ego_speed_mps"),
                    "lead_speed_delta_1s": number(row, "lead_speed_mps") - number(past, "lead_speed_mps"),
                    "gap_delta_1s": number(row, "gap_m") - number(past, "gap_m"),
                    "relative_speed_delta_1s": number(row, "relative_speed_mps") - number(past, "relative_speed_mps"),
                })
            else:
                vals.update({k: float("nan") for k in FEATURES if k.endswith("_delta_1s")})
            records.append({"episode_id": episode_id, "scenario_id": row["scenario_id"],
                            "x": [vals[k] for k in FEATURES], "y": int(row["future_contact_within_3s"]),
                            "time_s": t, "ttc": number(row, "ttc_s"),
                            "ttc_valid": int(row["ttc_valid"])})
    return episodes, records


def split_episodes(episodes, records):
    meta = {}
    for ep, rows in episodes.items():
        scenario = rows[0]["scenario_id"]
        contact = any(int(r["contact"]) == 1 for r in rows)
        meta[ep] = (scenario, contact)
    strata = defaultdict(list)
    for ep, key in meta.items():
        strata[key].append(ep)
    rng = random.Random(SEED)
    buckets = {"train": [], "validation": [], "calibration": [], "test": []}
    fractions = (0.60, 0.15, 0.10, 0.15)
    for _, values in sorted(strata.items()):
        rng.shuffle(values)
        n = len(values)
        sizes = [int(n * f) for f in fractions]
        for i in range(n - sum(sizes)):
            sizes[i % len(sizes)] += 1
        start = 0
        for name, size in zip(buckets, sizes):
            buckets[name].extend(values[start:start + size])
            start += size
    for values in buckets.values():
        rng.shuffle(values)
    return buckets


def arrays(records, episode_ids):
    selected = [r for r in records if r["episode_id"] in episode_ids]
    x = np.asarray([r["x"] for r in selected], dtype=float)
    y = np.asarray([r["y"] for r in selected], dtype=int)
    weights = np.asarray([1.0 for _ in selected], dtype=float)
    counts = defaultdict(int)
    for r in selected:
        counts[r["episode_id"]] += 1
    weights = np.asarray([1.0 / counts[r["episode_id"]] for r in selected])
    return selected, x, y, weights


def f2_threshold(y, p):
    best = (0.0, 0.5)
    for threshold in np.linspace(0.02, 0.98, 193):
        score = fbeta_score(y, p >= threshold, beta=2, zero_division=0)
        if score > best[0]:
            best = (score, float(threshold))
    return best[1]


def metrics(y, p, threshold):
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"rows": int(len(y)), "positive_rows": int(y.sum()),
            "average_precision_pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "threshold": float(threshold), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
            "brier_score": float(brier_score_loss(y, p)), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    episodes, records = read_dataset(args.input)
    splits = split_episodes(episodes, records)
    split_records = {k: arrays(records, set(ids)) for k, ids in splits.items()}
    models = {
        "logistic_regression": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                             LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                       RandomForestClassifier(n_estimators=350, min_samples_leaf=8,
                                                              class_weight="balanced_subsample", n_jobs=-1,
                                                              random_state=SEED)),
        "hist_gradient_boosting": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15,
                                                                               l2_regularization=2.0,
                                                                               learning_rate=0.08, random_state=SEED)),
    }
    validation_y = split_records["validation"][2]
    candidate_results = {}
    fitted = {}
    for name, model in models.items():
        _, x, y, w = split_records["train"]
        model.fit(x, y, **{"logisticregression__sample_weight": w} if name == "logistic_regression" else
                  {"randomforestclassifier__sample_weight": w} if name == "random_forest" else
                  {"histgradientboostingclassifier__sample_weight": w})
        p = model.predict_proba(split_records["validation"][1])[:, 1]
        candidate_results[name] = {"validation_average_precision": float(average_precision_score(validation_y, p)),
                                   "validation_f2_at_0_5": float(fbeta_score(validation_y, p >= 0.5, beta=2, zero_division=0))}
        fitted[name] = model
    selected_name = max(candidate_results, key=lambda n: candidate_results[n]["validation_average_precision"])
    model = fitted[selected_name]
    cal_y = split_records["calibration"][2]
    cal_raw = model.predict_proba(split_records["calibration"][1])[:, 1]
    cal_logits = np.log(np.clip(cal_raw, 1e-6, 1 - 1e-6) / np.clip(1 - cal_raw, 1e-6, 1))[:, None]
    calibrator = LogisticRegression(max_iter=1000, class_weight=None, random_state=SEED).fit(cal_logits, cal_y)
    def calibrated(raw):
        logits = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))[:, None]
        return calibrator.predict_proba(logits)[:, 1]
    val_raw = model.predict_proba(split_records["validation"][1])[:, 1]
    val_cal = calibrated(val_raw)
    threshold = f2_threshold(validation_y, val_cal)
    test_items, test_x, test_y, _ = split_records["test"]
    test_raw = model.predict_proba(test_x)[:, 1]
    test_cal = calibrated(test_raw)
    test_metrics = metrics(test_y, test_cal, threshold)
    # Transparent present-time TTC comparator, using the same test rows.
    rule_p = np.asarray([1.0 if r["ttc_valid"] and math.isfinite(r["ttc"]) and 0 < r["ttc"] <= 3 else 0.0 for r in test_items])
    rule_metrics = metrics(test_y, rule_p, 0.5)
    # Event-level sensitivity: did any prediction fire in each episode with a future event?
    by_ep = defaultdict(list)
    for r, p in zip(test_items, test_cal):
        by_ep[r["episode_id"]].append((r["y"], p))
    positive_eps = [v for v in by_ep.values() if any(y for y, _ in v)]
    event_recall = sum(any(p >= threshold for _, p in v) for v in positive_eps) / max(1, len(positive_eps))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator, "features": FEATURES,
                 "target": "future_contact_within_3s", "threshold": threshold}, args.out_dir / "contact_model.joblib")
    pred_path = args.out_dir / "test_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_id", "scenario_id", "time_s", "target_contact_within_3s", "raw_probability", "calibrated_probability", "rule_ttc_le_3s"])
        writer.writeheader()
        for r, raw, cal, rule in zip(test_items, test_raw, test_cal, rule_p):
            writer.writerow({"episode_id": r["episode_id"], "scenario_id": r["scenario_id"], "time_s": r["time_s"],
                             "target_contact_within_3s": r["y"], "raw_probability": f"{raw:.8f}",
                             "calibrated_probability": f"{cal:.8f}", "rule_ttc_le_3s": int(rule)})
    split_doc = {name: ids for name, ids in splits.items()}
    (args.out_dir / "episode_splits.json").write_text(json.dumps(split_doc, indent=2) + "\n", encoding="utf-8")
    result = {"dataset": str(args.input.relative_to(ROOT)), "seed": SEED,
              "split_episode_counts": {k: len(v) for k, v in splits.items()},
              "split_eligible_rows": {k: len(v[0]) for k, v in split_records.items()},
              "split_positive_rows": {k: int(v[2].sum()) for k, v in split_records.items()},
              "candidate_validation_results": candidate_results, "selected_model": selected_name,
              "validation_selected_f2_threshold": threshold,
              "test_contact_model_calibrated": test_metrics,
              "test_ttc_rule_baseline": rule_metrics,
              "test_event_level_recall": float(event_recall),
              "feature_names": FEATURES,
              "target_definition": "A future interpolated simulated contact within the next 3 seconds; rows with incomplete future observation and no contact are censored and excluded.",
              "limitations": ["All data and outcomes are generated by a toy simulator, not observed driving or CARLA.",
                              "The target is simulated bumper contact, not real-world crash risk.",
                              "Metrics are from one fixed synthetic episode split; external and scenario-shift validation remain necessary.",
                              "Risk estimates must not be used as a driving safety system."]}
    (args.out_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = ["# Synthetic Contact Model v2 Evaluation", "",
              "## Task and data", "",
              "Predict whether the simulator reaches bumper contact during the next 3 seconds. Training uses only current measurements and one-second past changes. Future labels and episode/scenario identifiers are not model inputs. Episodes, not rows, are separated between train, validation, calibration, and test.", "",
              f"- Source: `{args.input.relative_to(ROOT)}` ({len(episodes)} episodes; {len(records)} eligible rows).",
              f"- Contact episodes: {sum(any(int(r['contact']) for r in rs) for rs in episodes.values())}.",
              f"- Selected using validation average precision: `{selected_name}`.",
              f"- Episode split sizes: `{result['split_episode_counts']}`.", "",
              "## Locked test results", "",
              f"- Model: PR-AUC / average precision **{test_metrics['average_precision_pr_auc']:.3f}**, ROC-AUC **{test_metrics['roc_auc']:.3f}**, precision **{test_metrics['precision']:.3f}**, recall **{test_metrics['recall']:.3f}**, F2 **{test_metrics['f2']:.3f}**, Brier **{test_metrics['brier_score']:.3f}**.",
              f"- Same-test TTC rule baseline (current TTC ≤ 3 s): PR-AUC **{rule_metrics['average_precision_pr_auc']:.3f}**, recall **{rule_metrics['recall']:.3f}**, precision **{rule_metrics['precision']:.3f}**.",
              f"- Event-level recall: **{event_recall:.3f}** across {len(positive_eps)} positive test episodes.", "",
              "## Interpretation and limits", "",
              "This is a materially cleaner ML experiment than the earlier TTC-threshold target: its outcome is separately simulated bumper contact, and episode overlap is prevented across splits. It is still a prototype trained only on synthetic toy data. Good test scores show it learned this simulator, not that it predicts real crashes. Before any real safety claim, use a validated simulator such as CARLA and/or independently collected driving data, define a realistic outcome, test across unseen scenarios, and review calibration and false alarms with domain experts.", ""]
    (args.out_dir / "model_card.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
