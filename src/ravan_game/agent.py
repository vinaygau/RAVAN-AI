"""Simulation-only rule agent for optional autonomous driving."""

from __future__ import annotations

from .risk import RiskAssessment
from .simulation import DrivingSimulation, LANE_WIDTH_M


class RuleBasedDrivingAgent:
    """Conservative speed and lane rules for the simulated car.

    Lane changes use the same explainable front/rear gap checks shown in the
    dashboard. This is an educational agent, not a guarantee of safe driving.
    """

    def __init__(self, target_speed_mps: float = 18.0) -> None:
        self.target_speed_mps = float(target_speed_mps)
        self.target_lane: int | None = None

    def reset(self) -> None:
        self.target_lane = None

    def choose_action(self, simulation: DrivingSimulation, risk: RiskAssessment) -> str:
        ego = simulation.player
        if self.target_lane is not None:
            target_center = (self.target_lane + 0.5) * LANE_WIDTH_M
            error = target_center - ego.lateral_m
            if abs(error) < 0.18 and abs(ego.lateral_speed_mps) < 0.3:
                self.target_lane = None
            elif abs(error) < 0.55 and error * ego.lateral_speed_mps > 0:
                return "coast"
            else:
                return "steer_right" if error > 0 else "steer_left"

        front_gap = risk.distance_m
        ttc = risk.ttc_s
        if risk.collision or (ttc is not None and ttc <= 2.0) or (
            front_gap is not None and front_gap <= 8.0
        ):
            return "brake"

        desired_gap = max(22.0, ego.speed_mps * 1.8)
        front_is_blocking = front_gap is not None and front_gap < desired_gap
        if front_is_blocking:
            options: list[tuple[float, int]] = []
            if risk.lane_change_left.safe:
                options.append((risk.lane_change_left.front_gap_m or 120.0, ego.lane_index - 1))
            if risk.lane_change_right.safe:
                options.append((risk.lane_change_right.front_gap_m or 120.0, ego.lane_index + 1))
            if options:
                best_gap, lane = max(options)
                if best_gap > (front_gap or 0.0) + 12.0:
                    self.target_lane = lane
                    return "steer_right" if lane > ego.lane_index else "steer_left"
            return "brake"

        if ttc is not None and ttc < 4.0:
            return "brake"
        if ego.speed_mps < self.target_speed_mps - 0.8:
            return "accelerate"
        return "coast"
