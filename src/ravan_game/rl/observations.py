"""Bounded current-state observations for the live three-lane simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from ..risk import RiskAssessment
from ..simulation import DrivingSimulation, LANE_COUNT, LANE_WIDTH_M, ROAD_WIDTH_M


MAX_GAP_M = 120.0
MAX_TTC_S = 10.0
MAX_THW_S = 10.0
MAX_DRAC_MPS2 = 20.0
MAX_RELATIVE_SPEED_MPS = 42.0
MAX_RELATIVE_ACCELERATION_MPS2 = 22.0
MAX_ACCELERATION_MPS2 = 15.0
MAX_LATERAL_SPEED_MPS = 3.2
MAX_HEADING_DEG = 45.0


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _norm(value: float | None, maximum: float) -> float:
    if value is None:
        return 0.0
    return _clip(value / maximum, 0.0, 1.0)


def _signed_norm(value: float | None, maximum: float) -> float:
    if value is None:
        return 0.0
    return _clip(value / maximum, -1.0, 1.0)


def make_observation_space() -> spaces.Dict:
    """Return the documented float32 Dict space used by the observation builder."""
    return spaces.Dict({
        "ego": spaces.Box(
            low=np.asarray([0, -1, 0, 0, 0, 0, -1, -1, -1, -1, 0], dtype=np.float32),
            high=np.ones(11, dtype=np.float32),
            dtype=np.float32,
        ),
        "front": spaces.Box(
            low=np.asarray([0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.ones(11, dtype=np.float32),
            dtype=np.float32,
        ),
        "adjacent_lanes": spaces.Box(
            low=np.tile(
                np.asarray([0, 0, 0, 0, -1, 0, 0, -1, 0], dtype=np.float32),
                (2, 1),
            ),
            high=np.ones((2, 9), dtype=np.float32),
            dtype=np.float32,
        ),
        "environment": spaces.Box(
            low=np.zeros(13, dtype=np.float32),
            high=np.ones(13, dtype=np.float32),
            dtype=np.float32,
        ),
    })


def _lane_vehicle_records(
    simulation: DrivingSimulation,
    lane: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
    ego = simulation.player
    front: dict[str, Any] | None = None
    rear: dict[str, Any] | None = None
    occupied = False

    for agent in simulation.traffic:
        other = agent.vehicle
        lanes = {other.lane_index}
        if agent.target_lane_index is not None:
            lanes.add(agent.target_lane_index)
        if lane not in lanes:
            continue

        relative_position = other.longitudinal_m - ego.longitudinal_m
        if abs(relative_position) <= 100.0:
            occupied = True

        gap = max(0.0, abs(relative_position) - (ego.length_m + other.length_m) / 2.0)
        record = {"gap_m": gap, "speed_mps": other.speed_mps}
        if relative_position >= 0:
            if front is None or gap < front["gap_m"]:
                front = record
        elif rear is None or gap < rear["gap_m"]:
            rear = record

    return front, rear, occupied


def build_observation(
    simulation: DrivingSimulation,
    assessment: RiskAssessment,
    *,
    steering_input: float = 0.0,
    throttle_input: float = 0.0,
    handbrake_input: bool = False,
    episode_step: int = 0,
    max_episode_steps: int = 1,
    max_distance_m: float | None = None,
) -> dict[str, np.ndarray]:
    """Build current-timestep observations; no future labels or state are used.

    Missing front/rear gaps use the capped value 1.0 with the corresponding
    presence flag set to 0. Undefined TTC/THW use 1.0 plus a validity flag of
    0. This separates missing/unbounded values from valid finite measurements.
    """
    ego = simulation.player
    speed_cap = max(1.0, ego.max_speed_mps)
    lane_index = max(0, min(LANE_COUNT - 1, int(ego.lateral_m / LANE_WIDTH_M)))
    lane_one_hot = [1.0 if lane_index == lane else 0.0 for lane in range(LANE_COUNT)]
    ego_observation = np.asarray([
        _norm(ego.speed_mps, speed_cap),
        _signed_norm(ego.acceleration_mps2, MAX_ACCELERATION_MPS2),
        *lane_one_hot,
        _norm(ego.lateral_m, ROAD_WIDTH_M),
        _signed_norm(ego.lateral_speed_mps, MAX_LATERAL_SPEED_MPS),
        _signed_norm(ego.heading_deg, MAX_HEADING_DEG),
        _signed_norm(steering_input, 1.0),
        _signed_norm(throttle_input, 1.0),
        float(bool(handbrake_input)),
    ], dtype=np.float32)

    front = assessment.nearest_front or {}
    front_present = bool(assessment.nearest_front)
    ttc_valid = assessment.ttc_s is not None
    thw_valid = assessment.thw_s is not None
    drac_valid = assessment.drac_mps2 is not None
    front_observation = np.asarray([
        float(front_present),
        _norm(float(front["distance_m"]) if front_present else None, MAX_GAP_M),
        _norm(float(front["speed_kmh"]) / 3.6 if front_present else None, speed_cap),
        _signed_norm(assessment.relative_speed_mps, MAX_RELATIVE_SPEED_MPS),
        _signed_norm(assessment.relative_acceleration_mps2, MAX_RELATIVE_ACCELERATION_MPS2),
        _norm(assessment.ttc_s, MAX_TTC_S) if ttc_valid else 1.0,
        float(ttc_valid),
        _norm(assessment.thw_s, MAX_THW_S) if thw_valid else 1.0,
        float(thw_valid),
        _norm(assessment.drac_mps2, MAX_DRAC_MPS2),
        float(drac_valid),
    ], dtype=np.float32)

    adjacent_rows: list[list[float]] = []
    lane_checks = (assessment.lane_change_left, assessment.lane_change_right)
    for offset, check in zip((-1, 1), lane_checks):
        lane = lane_index + offset
        exists = 0 <= lane < LANE_COUNT
        if not exists:
            adjacent_rows.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            continue

        lane_front, lane_rear, occupied = _lane_vehicle_records(simulation, lane)
        front_gap = lane_front["gap_m"] if lane_front else None
        rear_gap = lane_rear["gap_m"] if lane_rear else None
        front_relative_speed = (
            ego.speed_mps - lane_front["speed_mps"] if lane_front else None
        )
        rear_relative_speed = (
            lane_rear["speed_mps"] - ego.speed_mps if lane_rear else None
        )
        adjacent_rows.append([
            1.0,
            float(occupied),
            float(lane_front is not None),
            _norm(front_gap, MAX_GAP_M) if front_gap is not None else 1.0,
            _signed_norm(front_relative_speed, MAX_RELATIVE_SPEED_MPS),
            float(lane_rear is not None),
            _norm(rear_gap, MAX_GAP_M) if rear_gap is not None else 1.0,
            _signed_norm(rear_relative_speed, MAX_RELATIVE_SPEED_MPS),
            float(check.safe),
        ])

    density = {"Light": 0, "Moderate": 1, "Heavy": 2}[simulation.traffic_density]
    risk_index = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}[assessment.risk_level]
    visible_count = sum(
        abs(agent.vehicle.longitudinal_m - ego.longitudinal_m) <= 100.0
        for agent in simulation.traffic
    )
    max_distance = max(
        1.0,
        max_distance_m if max_distance_m is not None else speed_cap * max_episode_steps * 0.1,
    )
    left_boundary = max(0.0, ego.lateral_m - ego.width_m / 2.0)
    right_boundary = max(0.0, ROAD_WIDTH_M - ego.lateral_m - ego.width_m / 2.0)
    environment_observation = np.asarray([
        _norm(visible_count, max(1, len(simulation.traffic))),
        *[1.0 if density == value else 0.0 for value in range(3)],
        *[1.0 if risk_index == value else 0.0 for value in range(4)],
        float(assessment.collision),
        _norm(ego.longitudinal_m, max_distance),
        _norm(episode_step, max(1, max_episode_steps)),
        _norm(left_boundary, ROAD_WIDTH_M),
        _norm(right_boundary, ROAD_WIDTH_M),
    ], dtype=np.float32)

    return {
        "ego": ego_observation,
        "front": front_observation,
        "adjacent_lanes": np.asarray(adjacent_rows, dtype=np.float32),
        "environment": environment_observation,
    }
