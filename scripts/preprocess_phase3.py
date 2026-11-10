#!/usr/bin/env python3
"""Validate Phase 2 episodes and recompute canonical Phase 3 features."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "phase2" / "trajectories.csv"
DEFAULT_OUTPUT = ROOT / "data" / "phase3"
PIPELINE_VERSION = "phase3.1"
TOL = 2e-5


def bumper_gap(lead_front_x_m: float, lead_length_m: float, ego_front_x_m: float) -> float:
    return lead_front_x_m - lead_length_m - ego_front_x_m


def derive_features(*, ego_front_x_m: float, lead_front_x_m: float,
                    lead_length_m: float, ego_speed_mps: float,
                    lead_speed_mps: float) -> dict[str, float | int | None]:
    """Compute instantaneous same-lane indicators from the current state only."""
    gap = bumper_gap(lead_front_x_m, lead_length_m, ego_front_x_m)
    relative_speed = ego_speed_mps - lead_speed_mps
    ttc_valid = gap > 0.0 and relative_speed > 0.0
    thw_valid = gap > 0.0 and ego_speed_mps > 0.0
    drac_valid = gap > 0.0 and relative_speed > 0.0
    return {
        "gap_m": gap,
        "relative_speed_mps": relative_speed,
        "ttc_s": gap / relative_speed if ttc_valid else None,
        "ttc_valid": int(ttc_valid),
        "thw_s": gap / ego_speed_mps if thw_valid else None,
        "thw_valid": int(thw_valid),
        "drac_mps2": relative_speed * relative_speed / (2.0 * gap) if drac_valid else None,
        "drac_valid": int(drac_valid),
    }


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Required field {key!r} is blank in episode {row.get('episode_id')}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value in {key!r} for episode {row.get('episode_id')}")
    return result


def close(a: float, b: float, tolerance: float = TOL) -> bool:
    # Phase 2 serializes positions and speeds to six decimals; division-based
    # features can amplify that quantization slightly near zero closing speed.
    return math.isclose(a, b, rel_tol=5e-5, abs_tol=tolerance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows_by_episode: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.input.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "episode_id", "scenario_id", "seed", "step", "time_s",
            "ego_front_x_m", "ego_speed_mps", "ego_acceleration_mps2",
            "lead_front_x_m", "lead_speed_mps", "lead_acceleration_mps2",
            "lead_length_m", "contact",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input is missing required columns: {sorted(missing)}")
        for row in reader:
            rows_by_episode[row["episode_id"]].append(row)

    if not rows_by_episode:
        raise ValueError("Input contains no trajectory rows")

    processed: list[dict[str, Any]] = []
    scenario_counts: Counter[str] = Counter()
    contact_count = 0
    nonmonotonic_positions = 0
    speed_bound_violations = 0
    irregular_intervals = 0
    source_feature_mismatches: Counter[str] = Counter()
    min_gap = math.inf
    max_gap = -math.inf
    dt_values: list[float] = []

    for episode_id, episode_rows in sorted(rows_by_episode.items()):
        episode_rows.sort(key=lambda row: int(row["step"]))
        scenario_counts[episode_rows[0]["scenario_id"]] += 1
        seen_contact = False
        previous: dict[str, str] | None = None
        for index, row in enumerate(episode_rows):
            if int(row["step"]) != index:
                raise ValueError(f"Non-contiguous step sequence in episode {episode_id}")
            t = as_float(row, "time_s")
            ego_x = as_float(row, "ego_front_x_m")
            ego_v = as_float(row, "ego_speed_mps")
            ego_a = as_float(row, "ego_acceleration_mps2")
            lead_x = as_float(row, "lead_front_x_m")
            lead_v = as_float(row, "lead_speed_mps")
            lead_a = as_float(row, "lead_acceleration_mps2")
            length = as_float(row, "lead_length_m")
            if ego_v < -TOL or lead_v < -TOL:
                speed_bound_violations += 1
            if previous:
                dt = t - as_float(previous, "time_s")
                dt_values.append(dt)
                is_final_contact_interval = (
                    index == len(episode_rows) - 1
                    and int(row["contact"]) == 1
                    and 0.0 < dt <= 0.1 + TOL
                )
                if not close(dt, 0.1) and not is_final_contact_interval:
                    irregular_intervals += 1
                if ego_x < as_float(previous, "ego_front_x_m") - TOL or lead_x < as_float(previous, "lead_front_x_m") - TOL:
                    nonmonotonic_positions += 1
            previous = row

            fts = derive_features(
                ego_front_x_m=ego_x,
                lead_front_x_m=lead_x,
                lead_length_m=length,
                ego_speed_mps=ego_v,
                lead_speed_mps=lead_v,
            )
            gap = float(fts["gap_m"])
            min_gap = min(min_gap, gap)
            max_gap = max(max_gap, gap)
            contact = int(gap <= 0.0)
            if int(row["contact"]) != contact:
                raise ValueError(f"Contact flag disagrees with geometry in episode {episode_id}, step {index}")
            if seen_contact:
                raise ValueError(f"Rows exist after contact in episode {episode_id}")
            if contact:
                seen_contact = True
                contact_count += 1

            # Audit the generator's stored derivations before replacing them with canonical values.
            checks = {
                "gap_m": fts["gap_m"],
                "relative_speed_mps": fts["relative_speed_mps"],
                "ttc_s": fts["ttc_s"],
                "thw_s": fts["thw_s"],
                "drac_mps2": fts["drac_mps2"],
            }
            for feature, expected in checks.items():
                observed = row.get(feature, "")
                if expected is None:
                    if observed != "":
                        source_feature_mismatches[feature] += 1
                elif observed == "" or not close(float(observed), float(expected)):
                    source_feature_mismatches[feature] += 1

            out = dict(row)
            for feature, value in fts.items():
                out[feature] = "" if value is None else (str(value) if isinstance(value, int) else f"{value:.9f}")
            out["contact"] = contact
            out["feature_pipeline_version"] = PIPELINE_VERSION
            processed.append(out)

    if speed_bound_violations:
        raise ValueError(f"Found {speed_bound_violations} negative speed samples")
    if nonmonotonic_positions:
        raise ValueError(f"Found {nonmonotonic_positions} non-monotonic position transitions")
    if irregular_intervals:
        raise ValueError(f"Found {irregular_intervals} sample intervals different from 0.1 s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_fields = list(processed[0].keys())
    with (args.out_dir / "trajectories_features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(processed)

    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "input": str(args.input.relative_to(ROOT)) if args.input.is_relative_to(ROOT) else str(args.input),
        "output": "data/phase3/trajectories_features.csv",
        "episode_count": len(rows_by_episode),
        "sample_count": len(processed),
        "scenario_episode_counts": dict(sorted(scenario_counts.items())),
        "contact_episodes": contact_count,
        "time_step_s": 0.1,
        "feature_formula_mismatches_vs_source": dict(source_feature_mismatches),
        "feature_mismatch_total": sum(source_feature_mismatches.values()),
        "source_feature_audit_tolerance": "absolute 2e-5 or relative 5e-5, accounting for Phase 2 six-decimal state serialization",
        "negative_speed_samples": speed_bound_violations,
        "nonmonotonic_position_transitions": nonmonotonic_positions,
        "irregular_time_intervals": irregular_intervals,
        "gap_m_range": [round(min_gap, 9), round(max_gap, 9)],
        "derived_features_use_future_data": False,
        "labels_generated": False,
    }
    (args.out_dir / "quality_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary["feature_mismatch_total"]:
        raise ValueError(f"Stored Phase 2 feature values failed audit: {source_feature_mismatches}")

    report = [
        "# Phase 3 Preprocessing Quality Report", "",
        f"- Pipeline version: `{PIPELINE_VERSION}`",
        f"- Episodes: {summary['episode_count']}",
        f"- Samples: {summary['sample_count']}",
        f"- Scenario episode counts: {summary['scenario_episode_counts']}",
        f"- Contact episodes: {summary['contact_episodes']}",
        f"- Gap range (m): {summary['gap_m_range'][0]} to {summary['gap_m_range'][1]}",
        f"- Stored Phase 2 derived-feature mismatches: {summary['feature_mismatch_total']}",
        f"- Negative-speed samples: {summary['negative_speed_samples']}",
        f"- Non-monotonic position transitions: {summary['nonmonotonic_position_transitions']}",
        f"- Irregular sample intervals: {summary['irregular_time_intervals']}",
        "- Feature calculations use current-row information only; no future observations or labels are used.",
        "- A shortened final interval is allowed only when Phase 2 interpolates the first contact within a time step; contact is recorded at zero bumper gap.",
        "- Phase 2 feature values are audited with the stated rounding tolerance because source positions and speeds were serialized to six decimals.",
        "- No risk or future-hazard labels are created.", "",
    ]
    (args.out_dir / "quality_report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
