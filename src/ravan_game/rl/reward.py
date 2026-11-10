"""Configurable, inspectable reward shaping for PPO experiments."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path

from ..risk import RiskAssessment
from ..simulation import DrivingSimulation, ROAD_WIDTH_M
from .actions import DriverAction


@dataclass(frozen=True)
class RewardConfig:
    progress_per_m: float = 0.12
    step_cost: float = -0.02
    collision: float = -150.0
    risk_exposure: float = -1.2
    side_vehicle_exposure: float = -0.7
    unsafe_lane_change: float = -0.6
    abrupt_control_change: float = -0.025
    unnecessary_braking: float = -0.08
    unnecessary_handbrake: float = -0.4
    boundary_proximity: float = -0.15
    safe_gap_seconds: float = 1.6
    risk_ttc_horizon_s: float = 5.0
    speed_limit_mps: float = 27.0
    speed_excess_per_mps: float = -0.025

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "RewardConfig":
        if path is None:
            path = Path(__file__).resolve().parents[3] / "config" / "rl_reward.json"
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {item.name for item in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown RL reward settings: {', '.join(sorted(unknown))}")
        return cls(**values)


def calculate_reward(
    simulation: DrivingSimulation,
    assessment: RiskAssessment,
    *,
    action: DriverAction,
    previous_position_m: float,
    previous_steering: float,
    previous_throttle: float,
    steering: float,
    throttle: float,
    collision: bool,
    config: RewardConfig,
) -> tuple[float, dict[str, float]]:
    """Return shaped reward and named components for this simulation step."""
    ego = simulation.player
    progress = max(0.0, ego.longitudinal_m - previous_position_m) * config.progress_per_m
    components = {
        "progress": progress,
        "step_cost": config.step_cost,
        "collision": config.collision if collision else 0.0,
        "risk_exposure": 0.0,
        "side_vehicle_exposure": 0.0,
        "unsafe_lane_change": 0.0,
        "abrupt_control_change": 0.0,
        "unnecessary_braking": 0.0,
        "unnecessary_handbrake": 0.0,
        "speed_excess": 0.0,
        "boundary_proximity": 0.0,
    }

    # Grade only the finite, observable lead TTC. Undefined TTC means the ego
    # is not currently closing on a same-lane lead vehicle.
    if assessment.ttc_s is not None:
        urgency = max(0.0, min(1.0, (config.risk_ttc_horizon_s - assessment.ttc_s)
                               / config.risk_ttc_horizon_s))
        components["risk_exposure"] = config.risk_exposure * urgency
    if assessment.distance_m is not None and ego.speed_mps > 0:
        desired_gap = max(5.0, ego.speed_mps * config.safe_gap_seconds)
        gap_shortfall = max(0.0, min(1.0, (desired_gap - assessment.distance_m) / desired_gap))
        components["risk_exposure"] += config.risk_exposure * gap_shortfall
    rear = assessment.nearest_rear
    rear_ttc = rear.get("ttc_s") if rear else None
    if isinstance(rear_ttc, (int, float)) and rear_ttc > 0:
        rear_urgency = max(0.0, min(1.0, (4.0 - rear_ttc) / 4.0))
        components["risk_exposure"] += config.risk_exposure * 0.75 * rear_urgency
    if any(
        vehicle.get("lane_relation") == "alongside"
        and abs(float(vehicle.get("longitudinal_offset_m", 999.0))) < 5.0
        for vehicle in assessment.nearby_vehicles
    ):
        components["side_vehicle_exposure"] = config.side_vehicle_exposure

    if action == DriverAction.STEER_LEFT and not assessment.lane_change_left.safe:
        components["unsafe_lane_change"] = config.unsafe_lane_change
    elif action == DriverAction.STEER_RIGHT and not assessment.lane_change_right.safe:
        components["unsafe_lane_change"] = config.unsafe_lane_change

    control_delta = abs(steering - previous_steering) + abs(throttle - previous_throttle)
    components["abrupt_control_change"] = config.abrupt_control_change * control_delta

    if action == DriverAction.BRAKE and assessment.risk_level == "LOW":
        components["unnecessary_braking"] = config.unnecessary_braking
    if action == DriverAction.HANDBRAKE and assessment.risk_level != "CRITICAL":
        components["unnecessary_handbrake"] = config.unnecessary_handbrake

    components["speed_excess"] = config.speed_excess_per_mps * max(
        0.0, ego.speed_mps - config.speed_limit_mps
    )

    edge_distance = min(
        ego.lateral_m - ego.width_m / 2.0,
        ROAD_WIDTH_M - ego.lateral_m - ego.width_m / 2.0,
    )
    if edge_distance < 0.65:
        components["boundary_proximity"] = config.boundary_proximity * (0.65 - edge_distance) / 0.65

    return sum(components.values()), components
