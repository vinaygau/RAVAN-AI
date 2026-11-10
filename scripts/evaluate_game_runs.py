"""Summarize a Ravan AI game telemetry CSV using the Python standard library."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def summarize_run(path: Path) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No telemetry rows in {path}")

    risk_samples = Counter(row["risk_level"] for row in rows)
    risk_seconds: defaultdict[str, float] = defaultdict(float)
    for previous, current in zip(rows, rows[1:]):
        delta = max(0.0, float(current["session_time_s"]) - float(previous["session_time_s"]))
        risk_seconds[previous["risk_level"]] += min(delta, 1.0)

    speeds = [float(row["ego_speed_kmh"]) for row in rows if row.get("ego_speed_kmh")]
    scores = [int(row["risk_score"]) for row in rows if row.get("risk_score")]
    ttc_rows = [
        float(row["ttc_s"])
        for row in rows
        if row.get("ttc_s") not in (None, "") and float(row["ttc_s"]) <= 2.0
    ]
    run_ids = sorted({row.get("run_id", "") for row in rows if row.get("run_id")})
    legacy_scenario_ids = sorted({row.get("scenario_id", "") for row in rows if row.get("scenario_id")})
    return {
        "path": path,
        "sample_count": len(rows),
        "duration_s": float(rows[-1]["session_time_s"]) - float(rows[0]["session_time_s"]),
        "run_ids": run_ids,
        "legacy_scenario_ids": legacy_scenario_ids,
        "average_speed_kmh": sum(speeds) / len(speeds) if speeds else 0.0,
        "maximum_risk_score": max(scores, default=0),
        "risk_samples": risk_samples,
        "risk_seconds": dict(risk_seconds),
        "short_ttc_samples": len(ttc_rows),
        "collision_samples": sum(row.get("collision", "").lower() == "true" for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, help="Run CSV; omit to summarize the newest run")
    args = parser.parse_args()
    run_dir = Path(__file__).resolve().parents[1] / "data" / "runs"
    path = args.csv_path
    if path is None:
        candidates = sorted(run_dir.glob("ravan_run_*.csv"), key=lambda item: item.stat().st_mtime)
        if not candidates:
            parser.error(f"No game runs found in {run_dir}")
        path = candidates[-1]

    report = summarize_run(path)
    print(f"Run: {report['path']}")
    if report["run_ids"]:
        print(f"Run ID(s): {', '.join(report['run_ids'])}")
    elif report["legacy_scenario_ids"]:
        print(f"Legacy scenario labels: {', '.join(report['legacy_scenario_ids'])}")
    print(f"Duration: {report['duration_s']:.1f} s across {report['sample_count']} samples")
    print(f"Average ego speed: {report['average_speed_kmh']:.1f} km/h")
    print(f"Maximum risk score: {report['maximum_risk_score']}/100")
    print(f"Samples with TTC at or below 2 s: {report['short_ttc_samples']}")
    print(f"Collision samples: {report['collision_samples']}")
    print("Risk-level exposure (estimated seconds):")
    for level in ("LOW", "MODERATE", "HIGH", "CRITICAL"):
        print(f"  {level}: {report['risk_seconds'].get(level, 0.0):.1f}")


if __name__ == "__main__":
    main()
