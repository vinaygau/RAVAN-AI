"""CSV telemetry logging for educational game runs (no voice/chat capture)."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .risk import RiskAssessment
from .route_rl import LanePathPlan
from .simulation import DrivingSimulation


FIELDS = [
    "timestamp_utc",
    "session_time_s",
    "simulation_time_s",
    "run_id",
    "seed",
    "ego_lane",
    "ego_speed_kmh",
    "ego_acceleration_mps2",
    "traffic_count",
    "risk_level",
    "risk_score",
    "collision",
    "distance_m",
    "relative_speed_mps",
    "relative_acceleration_mps2",
    "ttc_s",
    "thw_s",
    "drac_mps2",
    "nearest_front_speed_kmh",
    "nearest_rear_distance_m",
    "left_lane_safe",
    "left_lane_reason",
    "left_front_gap_m",
    "left_rear_gap_m",
    "right_lane_safe",
    "right_lane_reason",
    "right_front_gap_m",
    "right_rear_gap_m",
    "rl_lane_path",
    "rl_first_action",
    "rl_predicted_cost_m",
    "recommendation",
    "nearby_vehicles_json",
]


class SimulationLogger:
    """Write one telemetry row every 0.2 seconds while the game is active."""

    def __init__(self, output_dir: Path | None = None, sample_interval_s: float = 0.2) -> None:
        root = output_dir or Path(__file__).resolve().parents[2] / "data" / "runs"
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.path = root / f"ravan_run_{timestamp}.csv"
        self.sample_interval_s = max(0.05, sample_interval_s)
        self._started = time.monotonic()
        self._last_sample: float | None = None
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()

    def log(
        self,
        simulation: DrivingSimulation,
        assessment: RiskAssessment,
        *,
        lane_plan: LanePathPlan | None = None,
        force: bool = False,
    ) -> bool:
        now = time.monotonic()
        if not force and self._last_sample is not None and now - self._last_sample < self.sample_interval_s:
            return False

        front = assessment.nearest_front or {}
        rear = assessment.nearest_rear or {}
        left = assessment.lane_change_left
        right = assessment.lane_change_right

        def rounded(value: float | None) -> float | None:
            return None if value is None else round(value, 3)

        row: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_time_s": round(now - self._started, 3),
            "simulation_time_s": round(simulation.elapsed_s, 3),
            "run_id": simulation.run_id,
            "seed": simulation.seed,
            "ego_lane": int(simulation.player.lateral_m / 3.6),
            "ego_speed_kmh": round(simulation.player.speed_mps * 3.6, 2),
            "ego_acceleration_mps2": round(simulation.player.acceleration_mps2, 3),
            "traffic_count": len(simulation.traffic),
            "risk_level": assessment.risk_level,
            "risk_score": assessment.risk_score,
            "collision": assessment.collision,
            "distance_m": rounded(assessment.distance_m),
            "relative_speed_mps": rounded(assessment.relative_speed_mps),
            "relative_acceleration_mps2": rounded(assessment.relative_acceleration_mps2),
            "ttc_s": rounded(assessment.ttc_s),
            "thw_s": rounded(assessment.thw_s),
            "drac_mps2": rounded(assessment.drac_mps2),
            "nearest_front_speed_kmh": front.get("speed_kmh"),
            "nearest_rear_distance_m": rear.get("distance_m"),
            "left_lane_safe": left.safe,
            "left_lane_reason": left.reason,
            "left_front_gap_m": rounded(left.front_gap_m),
            "left_rear_gap_m": rounded(left.rear_gap_m),
            "right_lane_safe": right.safe,
            "right_lane_reason": right.reason,
            "right_front_gap_m": rounded(right.front_gap_m),
            "right_rear_gap_m": rounded(right.rear_gap_m),
            "rl_lane_path": lane_plan.display_path if lane_plan is not None else "",
            "rl_first_action": lane_plan.first_action if lane_plan is not None else "",
            "rl_predicted_cost_m": rounded(lane_plan.predicted_cost_m) if lane_plan is not None else None,
            "recommendation": assessment.recommendation,
            "nearby_vehicles_json": json.dumps(assessment.nearby_vehicles, separators=(",", ":")),
        }
        self._writer.writerow(row)
        self._file.flush()
        self._last_sample = now
        return True

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
