"""Frame-rate-independent vehicle motion for the educational simulator."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class Vehicle:
    """Vehicle state in metres, metres per second, and a three-lane road frame."""

    name: str
    longitudinal_m: float
    lateral_m: float
    speed_mps: float
    max_speed_mps: float = 42.0
    length_m: float = 4.6
    width_m: float = 1.8
    acceleration_mps2: float = 0.0
    lateral_speed_mps: float = 0.0
    vehicle_id: str = ""
    lane_index: int = 1
    heading_deg: float = 0.0
    behavior_state: str = "cruise"

    def __post_init__(self) -> None:
        if not self.vehicle_id:
            self.vehicle_id = self.name
        self._update_lane_and_heading(3.6, 3)

    def _update_lane_and_heading(self, lane_width_m: float, lane_count: int) -> None:
        lane = int(self.lateral_m / lane_width_m)
        self.lane_index = max(0, min(lane_count - 1, lane))
        self.heading_deg = degrees(atan2(self.lateral_speed_mps, max(4.0, self.speed_mps)))

    def update_player(
        self,
        dt: float,
        *,
        throttle: float,
        steering: float,
        road_width_m: float,
        lane_width_m: float = 3.6,
        lane_count: int = 3,
        handbrake: bool = False,
    ) -> None:
        """Advance the ego vehicle from normalized throttle and steering input."""
        if dt <= 0:
            return

        throttle = _clamp(throttle, -1.0, 1.0)
        steering = _clamp(steering, -1.0, 1.0)
        if handbrake:
            commanded_acceleration = -15.0
        elif throttle > 0:
            commanded_acceleration = 3.5 * throttle
        elif throttle < 0:
            commanded_acceleration = 8.0 * throttle
        else:
            commanded_acceleration = -0.45 if self.speed_mps > 0 else 0.0

        old_speed = self.speed_mps
        self.speed_mps = _clamp(old_speed + commanded_acceleration * dt, 0.0, self.max_speed_mps)
        self.acceleration_mps2 = (self.speed_mps - old_speed) / dt
        self.longitudinal_m += self.speed_mps * dt

        target_lateral_speed = 3.2 * steering
        lateral_acceleration = _clamp((target_lateral_speed - self.lateral_speed_mps) * 5.0, -12.0, 12.0)
        self.lateral_speed_mps = _clamp(self.lateral_speed_mps + lateral_acceleration * dt, -3.2, 3.2)
        self.lateral_m += self.lateral_speed_mps * dt
        margin = self.width_m / 2.0
        bounded_lateral = _clamp(self.lateral_m, margin, road_width_m - margin)
        if bounded_lateral != self.lateral_m:
            self.lateral_speed_mps = 0.0
        self.lateral_m = bounded_lateral
        self._update_lane_and_heading(lane_width_m, lane_count)

    def update_traffic(
        self,
        dt: float,
        *,
        target_speed_mps: float,
        target_lateral_m: float,
        road_width_m: float,
        lane_width_m: float,
        lane_count: int,
        allow_shoulder: bool = False,
    ) -> None:
        """Move an AI traffic vehicle toward a car-following speed and lane target."""
        if dt <= 0:
            return

        old_speed = self.speed_mps
        acceleration = _clamp((target_speed_mps - old_speed) * 1.2, -6.0, 2.5)
        self.speed_mps = _clamp(old_speed + acceleration * dt, 0.0, self.max_speed_mps)
        self.acceleration_mps2 = (self.speed_mps - old_speed) / dt
        self.longitudinal_m += self.speed_mps * dt

        lateral_error = target_lateral_m - self.lateral_m
        target_lateral_speed = _clamp(lateral_error * 0.8, -1.7, 1.7)
        lateral_acceleration = _clamp((target_lateral_speed - self.lateral_speed_mps) * 2.4, -3.2, 3.2)
        self.lateral_speed_mps = _clamp(self.lateral_speed_mps + lateral_acceleration * dt, -1.7, 1.7)
        self.lateral_m += self.lateral_speed_mps * dt

        margin = self.width_m / 2.0
        lower = -margin if allow_shoulder else margin
        upper = road_width_m + margin if allow_shoulder else road_width_m - margin
        bounded_lateral = _clamp(self.lateral_m, lower, upper)
        if bounded_lateral != self.lateral_m:
            self.lateral_speed_mps = 0.0
        self.lateral_m = bounded_lateral
        self._update_lane_and_heading(lane_width_m, lane_count)

    def update_cruise(self, dt: float, *, target_speed_mps: float) -> None:
        """Compatibility helper for simple longitudinal cruise updates."""
        if dt <= 0:
            return
        old_speed = self.speed_mps
        acceleration = _clamp((target_speed_mps - old_speed) * 1.2, -2.2, 2.2)
        self.speed_mps = _clamp(old_speed + acceleration * dt, 0.0, self.max_speed_mps)
        self.acceleration_mps2 = (self.speed_mps - old_speed) / dt
        self.longitudinal_m += self.speed_mps * dt
        self._update_lane_and_heading(3.6, 3)
