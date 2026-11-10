#!/usr/bin/env python3
"""Create current TTC categories and future-only Phase 4 labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "phase3" / "trajectories_features.csv"
DEFAULT_CONFIG = ROOT / "config" / "phase4.json"
DEFAULT_OUTPUT = ROOT / "data" / "phase4"
EPS = 1e-9


def current_risk_category(row: dict[str, str]) -> str:
    if int(row["contact"]):
        return "critical"
    if int(row["ttc_valid"]) != 1 or row["ttc_s"] == "":
        return "low"
    ttc = float(row["ttc_s"])
    if ttc <= 1.5:
        return "critical"
    if ttc <= 3.0:
        return "high"
    if ttc <= 5.0:
        return "moderate"
    return "low"


def label_episode(rows: list[dict[str, str]], horizon_s: float) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: float(row["time_s"]))
    times = [float(row["time_s"]) for row in rows]
    episode_end = times[-1]
    labeled: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        t = times[i]
        end = t + horizon_s
        future = [rows[j] for j in range(i + 1, len(rows)) if times[j] <= end + EPS]
        conflict = any(
            int(future_row["contact"]) == 1
            or (
                int(future_row["ttc_valid"]) == 1
                and future_row["ttc_s"] != ""
                and 0.0 < float(future_row["ttc_s"]) <= 3.0
            )
            for future_row in future
        )
        contact = any(int(future_row["contact"]) == 1 for future_row in future)
        full_horizon_observed = episode_end >= end - EPS
        conflict_eligible = conflict or full_horizon_observed
        contact_eligible = contact or full_horizon_observed
        out = dict(row)
        out["current_risk_category"] = current_risk_category(row)
        out["future_conflict_within_3s"] = int(conflict) if conflict_eligible else ""
        out["future_conflict_eligible"] = int(conflict_eligible)
        out["future_contact_within_3s"] = int(contact) if contact_eligible else ""
        out["future_contact_eligible"] = int(contact_eligible)
        out["label_horizon_s"] = f"{horizon_s:.1f}"
        out["label_protocol_version"] = "phase4.1"
        labeled.append(out)
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    horizon = float(cfg["future_horizon_s"])
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.input.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        needed = {"episode_id", "scenario_id", "time_s", "contact", "ttc_s", "ttc_valid"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input missing columns required by label protocol: {sorted(missing)}")
        for row in reader:
            groups[row["episode_id"]].append(row)
    if not groups:
        raise ValueError("Input contains no samples")

    output: list[dict[str, Any]] = []
    episode_metadata: list[dict[str, Any]] = []
    for episode_id in sorted(groups):
        source_rows = sorted(groups[episode_id], key=lambda row: float(row["time_s"]))
        episode_labeled = label_episode(source_rows, horizon)
        output.extend(episode_labeled)
        ttc_events = [
            float(r["time_s"]) for r in source_rows
            if int(r["ttc_valid"]) == 1 and r["ttc_s"] != "" and 0.0 < float(r["ttc_s"]) <= 3.0
        ]
        contact_events = [float(r["time_s"]) for r in source_rows if int(r["contact"]) == 1]
        event_candidates = [(t, "future_ttc_at_or_below_3s") for t in ttc_events]
        event_candidates += [(t, "sampled_contact") for t in contact_events]
        first_event_time, first_event_source = min(event_candidates) if event_candidates else (None, None)
        episode_metadata.append({
            "episode_id": episode_id,
            "scenario_id": source_rows[0]["scenario_id"],
            "protocol_version": "phase4.1",
            "first_ttc_threshold_event_time_s": min(ttc_events) if ttc_events else None,
            "first_sampled_contact_time_s": min(contact_events) if contact_events else None,
            "first_primary_event_time_s": first_event_time,
            "first_primary_event_source": first_event_source,
            "primary_positive_rows": sum(r["future_conflict_within_3s"] == 1 for r in episode_labeled),
            "primary_negative_rows": sum(r["future_conflict_within_3s"] == 0 for r in episode_labeled),
            "primary_censored_rows": sum(r["future_conflict_eligible"] == 0 for r in episode_labeled),
            "secondary_contact_event_time_s": min(contact_events) if contact_events else None,
        })
    if not output:
        raise ValueError("Labeling produced no rows")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "trajectories_labeled.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)

    eligible_conflict = [r for r in output if r["future_conflict_eligible"] == 1]
    eligible_contact = [r for r in output if r["future_contact_eligible"] == 1]
    current_counts = Counter(r["current_risk_category"] for r in output)
    conflict_counts = Counter(str(r["future_conflict_within_3s"]) for r in eligible_conflict)
    contact_counts = Counter(str(r["future_contact_within_3s"]) for r in eligible_contact)
    valid_ttc_values = [float(r["ttc_s"]) for r in output if int(r["ttc_valid"]) == 1 and r["ttc_s"] != ""]
    boundary_audit = {
        str(boundary): {
            "exact": sum(abs(value - boundary) <= EPS for value in valid_ttc_values),
            "within_0_05_s": sum(abs(value - boundary) <= 0.05 for value in valid_ttc_values),
        }
        for boundary in (1.5, 3.0, 5.0)
    }
    by_scenario: dict[str, dict[str, int]] = {}
    for scenario in sorted({r["scenario_id"] for r in output}):
        selected = [r for r in eligible_conflict if r["scenario_id"] == scenario]
        by_scenario[scenario] = {
            "eligible": len(selected),
            "positive": sum(int(r["future_conflict_within_3s"]) for r in selected),
        }
    summary = {
        "protocol_version": "phase4.1",
        "episode_count": len(groups),
        "sample_count": len(output),
        "future_horizon_s": horizon,
        "primary_event_rule": cfg["primary_event_rule"],
        "current_risk_category_counts": dict(sorted(current_counts.items())),
        "future_conflict_eligible": len(eligible_conflict),
        "future_conflict_censored": len(output) - len(eligible_conflict),
        "future_conflict_counts": dict(sorted(conflict_counts.items())),
        "future_conflict_by_scenario": by_scenario,
        "future_contact_eligible": len(eligible_contact),
        "future_contact_censored": len(output) - len(eligible_contact),
        "future_contact_counts": dict(sorted(contact_counts.items())),
        "current_contact_samples": sum(int(r["contact"]) for r in output),
        "ttc_boundary_sample_counts": boundary_audit,
        "episode_event_metadata_count": len(episode_metadata),
        "future_values_used_only_for_labels": True,
        "future_labels_in_feature_dictionary": False,
        "limitations": cfg["limitations"],
    }
    (args.out_dir / "label_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "episode_label_metadata.json").write_text(json.dumps(episode_metadata, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Phase 4 Label Audit", "",
        f"- Protocol: `phase4.1`; future horizon: {horizon:.1f} s.",
        f"- Episodes: {len(groups)}; labeled samples: {len(output)}.",
        f"- Current risk categories: {dict(sorted(current_counts.items()))}.",
        f"- Primary future-conflict target: {dict(sorted(conflict_counts.items()))}; eligible={len(eligible_conflict)}, censored={len(output)-len(eligible_conflict)}.",
        f"- Secondary future-contact target: {dict(sorted(contact_counts.items()))}; eligible={len(eligible_contact)}, censored={len(output)-len(eligible_contact)}.",
        f"- Current contact samples: {summary['current_contact_samples']}.",
        "", "## Primary target by scenario", "",
        "| Scenario | Eligible samples | Positive samples |",
        "|---|---:|---:|",
    ]
    for scenario, values in by_scenario.items():
        report.append(f"| {scenario} | {values['eligible']} | {values['positive']} |")
    report += [
        "", "## TTC boundary review", "",
        "| TTC boundary (s) | Exact samples | Within ±0.05 s |",
        "|---:|---:|---:|",
    ]
    for boundary, counts in boundary_audit.items():
        report.append(f"| {boundary} | {counts['exact']} | {counts['within_0_05_s']} |")
    report += [
        "", "## Audit interpretation", "",
        "The primary label uses only later samples in the defined future window. A positive event is retained even if contact truncates the episode; a negative requires full horizon coverage. Rows without sufficient future observation and no observed event are censored (blank target, eligible=0). Current risk categories and future targets are distinct.",
        "", "## Limitations", "",
        cfg["limitations"],
        "The primary target contains a future TTC threshold event, while TTC is also a candidate current feature. Model results must be compared with a transparent TTC/rule baseline and described as prediction of this operational surrogate event, not independent real-world crash prediction. Thresholds require sensitivity analysis before model evaluation.",
        "",
    ]
    (args.out_dir / "label_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
