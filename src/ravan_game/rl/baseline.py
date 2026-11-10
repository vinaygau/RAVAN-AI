"""Deterministic physics-rule policy used as an evaluation baseline."""

from __future__ import annotations

from ..simulation import LANE_WIDTH_M
from .actions import DriverAction


class PhysicsRulePolicy:
    """Conservative speed and gap rules using the live risk assessment.

    This policy is intentionally understandable and is not a driving-control
    system. Lane changes are suggested only after the simulator's safety check.
    """

    def __init__(self, target_speed_mps: float = 22.0) -> None:
        self.target_speed_mps = float(target_speed_mps)
        self.target_lane: int | None = None

    def reset(self) -> None:
        self.target_lane = None

    def act(self, environment) -> DriverAction:
        sim = environment.simulation
        risk = environment._assessment
        ego = sim.player
        if self.target_lane is not None:
            target_center = (self.target_lane + 0.5) * LANE_WIDTH_M
            error = target_center - ego.lateral_m
            if abs(error) < 0.18 and abs(ego.lateral_speed_mps) < 0.3:
                self.target_lane = None
            elif abs(error) < 0.55 and error * ego.lateral_speed_mps > 0:
                return DriverAction.COAST
            elif abs(error) < 0.22 and error * ego.lateral_speed_mps > 0:
                return DriverAction.STEER_RIGHT if error > 0 else DriverAction.STEER_LEFT
            else:
                return DriverAction.STEER_RIGHT if error > 0 else DriverAction.STEER_LEFT

        front_gap = risk.distance_m
        ttc = risk.ttc_s
        if risk.collision or (ttc is not None and ttc <= 1.6) or (
            front_gap is not None and front_gap <= 7.0
        ):
            return DriverAction.BRAKE

        desired_gap = max(20.0, ego.speed_mps * 1.6)
        front_is_blocking = front_gap is not None and front_gap < desired_gap
        if front_is_blocking:
            options = []
            if risk.lane_change_left.safe:
                options.append((risk.lane_change_left.front_gap_m or 120.0, ego.lane_index - 1))
            if risk.lane_change_right.safe:
                options.append((risk.lane_change_right.front_gap_m or 120.0, ego.lane_index + 1))
            if options:
                best_gap, lane = max(options)
                if best_gap > (front_gap or 0.0) + 12.0:
                    self.target_lane = lane
                    return DriverAction.STEER_RIGHT if lane > ego.lane_index else DriverAction.STEER_LEFT
            return DriverAction.BRAKE

        if ttc is not None and ttc < 4.0:
            return DriverAction.BRAKE
        if ego.speed_mps < self.target_speed_mps - 0.8:
            return DriverAction.ACCELERATE
        if ego.speed_mps > self.target_speed_mps + 1.2:
            return DriverAction.COAST
        return DriverAction.COAST
