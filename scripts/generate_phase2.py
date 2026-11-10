#!/usr/bin/env python3
"""Generate reproducible Phase 2 synthetic longitudinal trajectory episodes.

Uses only the Python standard library. This is a transparent toy simulator,
not a vehicle dynamics model or validated traffic safety system.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "phase2.json"
CSV_FIELDS = [
    "episode_id", "scenario_id", "seed", "step", "time_s",
    "ego_front_x_m", "ego_speed_mps", "ego_acceleration_mps2",
    "lead_front_x_m", "lead_speed_mps", "lead_acceleration_mps2",
    "lead_length_m", "gap_m", "relative_speed_mps", "ttc_s", "ttc_valid",
    "thw_s", "thw_valid", "drac_mps2", "drac_valid", "contact",
]


def uniform(rng: random.Random, bounds: list[float]) -> float:
    return rng.uniform(float(bounds[0]), float(bounds[1]))


def sample_parameters(rng: random.Random, scenario: str, cfg: dict[str, Any]) -> dict[str, Any]:
    ranges = cfg["parameter_ranges"]
    ego_v = uniform(rng, ranges["ego_initial_speed_mps"])
    lead_v = uniform(rng, ranges["lead_initial_speed_mps"])
    gap = uniform(rng, ranges["initial_gap_m"])
    lead_length = uniform(rng, ranges["lead_length_m"])
    params: dict[str, Any] = {
        "ego_initial_speed_mps": ego_v,
        "lead_initial_speed_mps": lead_v,
        "initial_gap_m": gap,
        "lead_length_m": lead_length,
        "lead_brake_onset_s": None,
        "lead_deceleration_mps2": 0.0,
        "ego_reaction_delay_s": None,
        "ego_braking_magnitude_mps2": 0.0,
    }

    if scenario == "S1":  # Equal-speed steady following.
        params["lead_initial_speed_mps"] = ego_v
    elif scenario == "S2":  # Ego closes on a steady lead.
        lead_v = uniform(rng, [5.0, max(5.0, ego_v - 0.5)])
        if lead_v >= ego_v:
            lead_v = max(0.0, ego_v - 0.5)
        params["lead_initial_speed_mps"] = lead_v
    elif scenario == "S3":  # Lead brakes; ego holds its initial speed.
        params["ego_initial_speed_mps"] = ego_v
        params["lead_initial_speed_mps"] = min(ego_v, uniform(rng, [5.0, 28.0]))
        params["lead_brake_onset_s"] = uniform(rng, ranges["lead_brake_onset_s"])
        params["lead_deceleration_mps2"] = -uniform(rng, ranges["lead_deceleration_magnitude_mps2"])
    elif scenario == "S4":  # Lead brakes; ego responds after a sampled delay.
        params["ego_initial_speed_mps"] = ego_v
        params["lead_initial_speed_mps"] = min(ego_v, uniform(rng, [5.0, 28.0]))
        params["lead_brake_onset_s"] = uniform(rng, ranges["lead_brake_onset_s"])
        params["lead_deceleration_mps2"] = -uniform(rng, ranges["lead_deceleration_magnitude_mps2"])
        params["ego_reaction_delay_s"] = uniform(rng, ranges["ego_reaction_delay_s"])
        params["ego_braking_magnitude_mps2"] = uniform(rng, ranges["ego_braking_magnitude_mps2"])
    elif scenario == "S5":  # Wide initial gap and prompt, stronger ego response.
        params["initial_gap_m"] = uniform(rng, [40.0, 60.0])
        params["ego_initial_speed_mps"] = ego_v
        params["lead_initial_speed_mps"] = ego_v
        params["lead_brake_onset_s"] = uniform(rng, [2.0, 4.0])
        params["lead_deceleration_mps2"] = -uniform(rng, [2.0, 4.0])
        params["ego_reaction_delay_s"] = 0.0
        params["ego_braking_magnitude_mps2"] = uniform(rng, [4.0, 6.0])
    elif scenario == "S6":  # Non-closing pair.
        lead_v = uniform(rng, ranges["lead_initial_speed_mps"])
        params["lead_initial_speed_mps"] = max(ego_v, lead_v)
    elif scenario == "S7":  # Low-speed pair with both vehicles braking to rest.
        speed = uniform(rng, [5.0, 10.0])
        params["ego_initial_speed_mps"] = speed
        params["lead_initial_speed_mps"] = speed
        params["initial_gap_m"] = uniform(rng, [15.0, 40.0])
        params["lead_brake_onset_s"] = uniform(rng, [1.0, 3.0])
        params["lead_deceleration_mps2"] = -uniform(rng, [3.0, 6.0])
        params["ego_reaction_delay_s"] = 0.0
        params["ego_braking_magnitude_mps2"] = abs(params["lead_deceleration_mps2"])
    else:
        raise ValueError(f"Unknown scenario family: {scenario}")

    params["ego_initial_front_x_m"] = 0.0
    params["lead_initial_front_x_m"] = params["initial_gap_m"] + lead_length
    return params


def step_vehicle(x: float, speed: float, acceleration: float, dt: float, speed_cap: float) -> tuple[float, float]:
    """Integrate constant acceleration, stopping within a step if needed."""
    acceleration = max(-8.0, min(3.0, acceleration))
    if acceleration < 0 and speed + acceleration * dt < 0:
        stop_time = -speed / acceleration if speed > 0 else 0.0
        x += speed * stop_time + 0.5 * acceleration * stop_time * stop_time
        return x, 0.0
    next_x = x + speed * dt + 0.5 * acceleration * dt * dt
    next_speed = min(speed_cap, max(0.0, speed + acceleration * dt))
    return next_x, next_speed


def valid_features(gap: float, ego_speed: float, lead_speed: float) -> dict[str, Any]:
    closing_speed = ego_speed - lead_speed
    ttc_valid = gap > 0 and closing_speed > 0
    thw_valid = gap > 0 and ego_speed > 0
    drac_valid = gap > 0 and closing_speed > 0
    return {
        "gap_m": gap,
        "relative_speed_mps": closing_speed,
        "ttc_s": gap / closing_speed if ttc_valid else "",
        "ttc_valid": int(ttc_valid),
        "thw_s": gap / ego_speed if thw_valid else "",
        "thw_valid": int(thw_valid),
        "drac_mps2": closing_speed * closing_speed / (2 * gap) if drac_valid else "",
        "drac_valid": int(drac_valid),
    }


def make_trajectory_row(episode_id: str, scenario: str, seed: int, step: int, time_s: float,
                        ego_x: float, ego_v: float, ego_a: float, lead_x: float,
                        lead_v: float, lead_a: float, lead_length: float) -> dict[str, Any]:
    gap = lead_x - lead_length - ego_x
    grid_time = round(time_s / 0.1) * 0.1
    serialized_time = f"{time_s:.1f}" if abs(time_s - grid_time) < 1e-8 else f"{time_s:.9f}"
    return {
        "episode_id": episode_id,
        "scenario_id": scenario,
        "seed": seed,
        "step": step,
        "time_s": serialized_time,
        "ego_front_x_m": f"{ego_x:.9f}",
        "ego_speed_mps": f"{ego_v:.9f}",
        "ego_acceleration_mps2": f"{ego_a:.9f}",
        "lead_front_x_m": f"{lead_x:.9f}",
        "lead_speed_mps": f"{lead_v:.9f}",
        "lead_acceleration_mps2": f"{lead_a:.9f}",
        "lead_length_m": f"{lead_length:.9f}",
        "contact": int(gap <= 0.0),
        **valid_features(gap, ego_v, lead_v),
    }


def run_episode(episode_id: str, scenario: str, seed: int, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    p = sample_parameters(rng, scenario, cfg)
    dt = float(cfg["time_step_s"])
    duration = round(uniform(rng, cfg["duration_s"]))
    nsteps = int(round(duration / dt))
    speed_cap = 35.0
    ego_x = float(p["ego_initial_front_x_m"])
    lead_x = float(p["lead_initial_front_x_m"])
    ego_v = float(p["ego_initial_speed_mps"])
    lead_v = float(p["lead_initial_speed_mps"])
    ego_a = lead_a = 0.0
    rows: list[dict[str, Any]] = []
    min_gap = math.inf
    contact_time = None
    lead_brake_start = p["lead_brake_onset_s"]
    ego_brake_start = (
        lead_brake_start + p["ego_reaction_delay_s"]
        if lead_brake_start is not None and p["ego_reaction_delay_s"] is not None
        else None
    )

    for step in range(nsteps + 1):
        t = round(step * dt, 10)
        gap = lead_x - p["lead_length_m"] - ego_x
        contact = gap <= 0
        min_gap = min(min_gap, gap)
        row = make_trajectory_row(episode_id, scenario, seed, step, t, ego_x, ego_v, ego_a, lead_x, lead_v, lead_a, float(p["lead_length_m"]))
        rows.append(row)
        if contact:
            contact_time = t
            break
        if step == nsteps:
            break

        # Piecewise-constant acceleration is selected for the interval beginning at t.
        lead_a = p["lead_deceleration_mps2"] if lead_brake_start is not None and t >= lead_brake_start else 0.0
        ego_a = -p["ego_braking_magnitude_mps2"] if ego_brake_start is not None and t >= ego_brake_start else 0.0
        next_ego_x, next_ego_v = step_vehicle(ego_x, ego_v, ego_a, dt, speed_cap)
        next_lead_x, next_lead_v = step_vehicle(lead_x, lead_v, lead_a, dt, speed_cap)
        next_gap = next_lead_x - p["lead_length_m"] - next_ego_x
        if cfg.get("continuous_contact_interpolation", False) and next_gap <= 0.0:
            lo, hi = 0.0, dt
            for _ in range(50):
                mid = (lo + hi) / 2.0
                test_ego_x, _ = step_vehicle(ego_x, ego_v, ego_a, mid, speed_cap)
                test_lead_x, _ = step_vehicle(lead_x, lead_v, lead_a, mid, speed_cap)
                test_gap = test_lead_x - p["lead_length_m"] - test_ego_x
                if test_gap > 0.0:
                    lo = mid
                else:
                    hi = mid
            contact_dt = hi
            ego_x, ego_v = step_vehicle(ego_x, ego_v, ego_a, contact_dt, speed_cap)
            lead_x, lead_v = step_vehicle(lead_x, lead_v, lead_a, contact_dt, speed_cap)
            # Keep the serialized bumper gap microscopically non-positive so
            # the geometric contact audit remains stable after CSV rounding.
            lead_x = ego_x + p["lead_length_m"] - 1e-7
            contact_time = t + contact_dt
            contact_row = make_trajectory_row(episode_id, scenario, seed, step + 1, contact_time, ego_x, ego_v, ego_a, lead_x, lead_v, lead_a, float(p["lead_length_m"]))
            contact_row["contact"] = 1
            contact_row["gap_m"] = 0.0
            contact_row["ttc_s"] = ""
            contact_row["ttc_valid"] = 0
            contact_row["thw_s"] = ""
            contact_row["thw_valid"] = 0
            contact_row["drac_mps2"] = ""
            contact_row["drac_valid"] = 0
            rows.append(contact_row)
            min_gap = min(min_gap, 0.0)
            break
        ego_x, ego_v = next_ego_x, next_ego_v
        lead_x, lead_v = next_lead_x, next_lead_v

    metadata = {
        "episode_id": episode_id,
        "scenario_id": scenario,
        "seed": seed,
        "parameters": p,
        "time_step_s": dt,
        "requested_duration_s": duration,
        "actual_duration_s": float(rows[-1]["time_s"]),
        "sample_count": len(rows),
        "contact": contact_time is not None,
        "first_contact_sample_time_s": contact_time,
        "first_contact_time_s": contact_time,
        "contact_gap_at_first_sample_m": round(float(rows[-1]["gap_m"]), 6) if contact_time is not None else None,
        "minimum_gap_m": round(min_gap, 6),
        "trace_ends_at_contact": contact_time is not None,
        "risk_labels_generated": False,
    }
    return rows, metadata


def summarize(rows: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_counts = Counter(ep["scenario_id"] for ep in episodes)
    contacts_by_scenario = Counter(ep["scenario_id"] for ep in episodes if ep["contact"])
    numeric_columns = [
        "ego_speed_mps", "lead_speed_mps", "gap_m", "relative_speed_mps", "ttc_s", "thw_s", "drac_mps2"
    ]
    ranges: dict[str, dict[str, float | int | None]] = {}
    for col in numeric_columns:
        values = [float(row[col]) for row in rows if row[col] not in ("", None)]
        ranges[col] = {
            "min": round(min(values), 6) if values else None,
            "max": round(max(values), 6) if values else None,
            "non_null_count": len(values),
        }
    return {
        "episode_count": len(episodes),
        "sample_count": len(rows),
        "episodes_per_scenario": dict(sorted(scenario_counts.items())),
        "contacts_by_scenario": dict(sorted(contacts_by_scenario.items())),
        "contact_episode_count": sum(ep["contact"] for ep in episodes),
        "episode_duration_s": {
            "min": min(ep["actual_duration_s"] for ep in episodes),
            "max": max(ep["actual_duration_s"] for ep in episodes),
        },
        "feature_ranges": ranges,
        "undefined_feature_counts": {
            "ttc": sum(not int(row["ttc_valid"]) for row in rows),
            "thw": sum(not int(row["thw_valid"]) for row in rows),
            "drac": sum(not int(row["drac_valid"]) for row in rows),
        },
        "risk_labels_generated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "phase2")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    episode_metadata: list[dict[str, Any]] = []
    per_scenario = int(cfg["episodes_per_scenario"])
    for s_idx, scenario in enumerate(cfg["scenario_families"]):
        for i in range(per_scenario):
            episode_number = s_idx * per_scenario + i + 1
            episode_id = f"{cfg.get('episode_id_prefix', 'P2')}-{episode_number:05d}"
            seed = int(cfg["base_seed"]) + episode_number
            episode_rows, metadata = run_episode(episode_id, scenario, seed, cfg)
            rows.extend(episode_rows)
            episode_metadata.append(metadata)

    with (args.out_dir / "trajectories.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = summarize(rows, episode_metadata)
    manifest = {
        "project": "Ravan AI",
        "phase": 2,
        "dataset_name": cfg.get("dataset_name", "synthetic_longitudinal_pilot_v1"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/generate_phase2.py",
        "config_path": str(args.config.relative_to(ROOT)) if args.config.is_relative_to(ROOT) else str(args.config),
        "config": cfg,
        "coordinate_convention": "front-bumper positions; positive x in direction of travel; gap = lead_front_x - lead_length - ego_front_x",
        "time_step_s": cfg["time_step_s"],
        "units": {"time": "s", "position": "m", "speed": "m/s", "acceleration": "m/s^2"},
        "seed_policy": "episode seed = base_seed + one-based episode number",
        "contact_policy": cfg["contact_handling"],
        "labels": cfg["labeling"],
        "episodes": episode_metadata,
        "quality_summary": report,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase 2 Synthetic Dataset Quality Report", "",
        f"Dataset: `{manifest['dataset_name']}`", "",
        "## Generation summary", "",
        f"- Episodes: {report['episode_count']}",
        f"- Trajectory samples: {report['sample_count']}",
        f"- Scenario families: {', '.join(f'{k} ({v})' for k, v in report['episodes_per_scenario'].items())}",
        f"- Contact episodes: {report['contact_episode_count']}",
        f"- Recorded episode duration: {report['episode_duration_s']['min']:.1f}–{report['episode_duration_s']['max']:.1f} s",
        "- Risk labels: none (intentionally not assigned in Phase 2)",
        "", "## Feature coverage", "",
        "| Feature | Minimum | Maximum | Non-null samples |", "|---|---:|---:|---:|",
    ]
    for name, stats in report["feature_ranges"].items():
        lo = "—" if stats["min"] is None else f"{stats['min']:.6f}"
        hi = "—" if stats["max"] is None else f"{stats['max']:.6f}"
        lines.append(f"| `{name}` | {lo} | {hi} | {stats['non_null_count']} |")
    lines += [
        "", "## Missing and validity handling", "",
        f"- TTC is undefined in {report['undefined_feature_counts']['ttc']} samples and stored as a blank value with `ttc_valid=0`.",
        f"- THW is undefined in {report['undefined_feature_counts']['thw']} samples and stored as a blank value with `thw_valid=0`.",
        f"- DRAC is undefined in {report['undefined_feature_counts']['drac']} samples and stored as a blank value with `drac_valid=0`.",
        ("- Contact is interpolated within a time step; the final recorded contact frame has a tiny non-positive bumper gap to remain stable after CSV rounding." if cfg.get("continuous_contact_interpolation", False) else "- Contact is detected at the first 0.1-second sample with a non-positive gap. Vehicles may overlap slightly because contact is not interpolated between frames."),
        "- No rows are emitted after the first contact sample; post-contact motion is outside this toy simulator.",
        "", "## Data dictionary", "",
        "| Column | Meaning |",
        "|---|---|",
        "| `episode_id`, `scenario_id`, `seed`, `step` | Stable episode identity, scenario family, episode seed, and sample index. |",
        "| `time_s` | Elapsed simulation time in seconds. |",
        "| `ego_front_x_m`, `lead_front_x_m` | Front-bumper positions in metres along the positive travel axis. |",
        "| `ego_speed_mps`, `lead_speed_mps` | Vehicle speeds in metres per second. |",
        "| `ego_acceleration_mps2`, `lead_acceleration_mps2` | Applied acceleration over the interval ending at this sample; zero at the initial sample. |",
        "| `lead_length_m` | Lead vehicle length used in the bumper-gap calculation. |",
        "| `gap_m` | Lead front position minus lead length minus ego front position; at interpolated contact the recorded value is approximately zero (tiny negative rounding guard). |",
        "| `relative_speed_mps` | Ego speed minus lead speed; positive means closing in this same-lane setup. |",
        "| `ttc_s`, `thw_s`, `drac_mps2` | Simplified derived indicators; blank when their validity conditions do not hold. |",
        "| `ttc_valid`, `thw_valid`, `drac_valid` | 1 when the corresponding value is defined, otherwise 0. |",
        "| `contact` | 1 at first contact; v2 interpolates within a time step and the episode ends at that row. |",
        "", "## Limitations", "",
        "This is a synthetic one-dimensional, same-lane dataset generated from scripted kinematics. It is not collected from CARLA or real vehicles, does not represent sensor noise, and does not support claims about real-world collision probability or safety. Scenario categories describe generation setups, not hazard labels. The configured ranges are provisional and should be reviewed before modeling.",
        "", "## Reproducibility", "",
        "Run `python scripts/generate_phase2.py` from the project root. The base seed, per-episode seed policy, generation configuration, episode parameters, and quality summary are recorded in `metadata.json`.", "",
    ]
    (args.out_dir / "quality_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), **report}, indent=2))


if __name__ == "__main__":
    main()
