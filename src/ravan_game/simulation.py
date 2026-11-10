"""Continuous player-driven traffic simulation for the Ravan AI game."""

from __future__ import annotations

from dataclasses import dataclass
from math import sin
from random import Random

from .vehicle import Vehicle


ROAD_WIDTH_M = 10.8
LANE_WIDTH_M = 3.6
LANE_COUNT = 3
TRAFFIC_TARGET_COUNT = 12


@dataclass(frozen=True)
class VehicleSnapshot:
    """Immutable state used by traffic decisions for one simulation step."""

    vehicle_id: str
    longitudinal_m: float
    lateral_m: float
    speed_mps: float
    length_m: float
    lane_index: int
    target_lane_index: int | None = None


@dataclass
class TrafficAgent:
    """An independently controlled traffic participant."""

    vehicle: Vehicle
    cruise_speed_mps: float
    speed_phase_rad: float
    speed_variation_mps: float = 0.8
    target_lane_index: int | None = None
    lane_change_cooldown_s: float = 0.0
    lane_decision_timer_s: float = 0.0
    merge_from_shoulder: bool = False

    def _gap_to(self, other: VehicleSnapshot) -> float:
        return max(0.0, abs(other.longitudinal_m - self.vehicle.longitudinal_m)
                   - (other.length_m + self.vehicle.length_m) / 2.0)

    def _lane_gaps(
        self,
        lane: int,
        vehicles: tuple[VehicleSnapshot, ...],
    ) -> tuple[VehicleSnapshot | None, float | None, VehicleSnapshot | None, float | None]:
        front: tuple[VehicleSnapshot, float] | None = None
        rear: tuple[VehicleSnapshot, float] | None = None
        for other in vehicles:
            if other.vehicle_id == self.vehicle.vehicle_id:
                continue
            if other.lane_index != lane and other.target_lane_index != lane:
                continue
            delta = other.longitudinal_m - self.vehicle.longitudinal_m
            gap = self._gap_to(other)
            if delta >= 0 and (front is None or gap < front[1]):
                front = (other, gap)
            elif delta < 0 and (rear is None or gap < rear[1]):
                rear = (other, gap)
        return (
            front[0] if front else None,
            front[1] if front else None,
            rear[0] if rear else None,
            rear[1] if rear else None,
        )

    def _lane_is_clear(
        self,
        lane: int,
        vehicles: tuple[VehicleSnapshot, ...],
    ) -> tuple[bool, float]:
        front, front_gap, rear, rear_gap = self._lane_gaps(lane, vehicles)
        front_clearance = 90.0 if front_gap is None else front_gap
        rear_clearance = 90.0 if rear_gap is None else rear_gap
        if front is not None and front_gap is not None:
            closing = self.vehicle.speed_mps - front.speed_mps
            if front_gap < max(13.0, self.vehicle.speed_mps * 0.85):
                return False, -front_gap
            if closing > 0 and front_gap / closing < 3.0:
                return False, -front_gap
        if rear is not None and rear_gap is not None:
            closing = rear.speed_mps - self.vehicle.speed_mps
            if rear_gap < max(11.0, rear.speed_mps * 0.8):
                return False, -rear_gap
            if closing > 0 and rear_gap / closing < 3.0:
                return False, -rear_gap
        ahead_speed = front.speed_mps if front is not None else self.cruise_speed_mps + 2.0
        return True, min(front_clearance, rear_clearance) + ahead_speed * 0.4

    def _consider_lane_change(
        self,
        dt: float,
        vehicles: tuple[VehicleSnapshot, ...],
        current_lane: int,
        blocked: bool,
        rng: Random,
    ) -> None:
        if self.target_lane_index is not None or self.lane_change_cooldown_s > 0:
            return
        self.lane_decision_timer_s -= dt
        if self.lane_decision_timer_s > 0:
            return
        self.lane_decision_timer_s = rng.uniform(0.65, 1.35)
        if not blocked and rng.random() >= 0.025:
            return

        options: list[tuple[float, int]] = []
        for lane in (current_lane - 1, current_lane + 1):
            if not 0 <= lane < LANE_COUNT:
                continue
            clear, space_score = self._lane_is_clear(lane, vehicles)
            if not clear:
                continue
            front, _, _, _ = self._lane_gaps(lane, vehicles)
            lane_speed = front.speed_mps if front is not None else self.cruise_speed_mps + 2.0
            options.append((min(lane_speed, self.cruise_speed_mps) + space_score * 0.015, lane))

        if not options:
            return
        options.sort(reverse=True)
        if blocked or rng.random() < 0.22:
            self.target_lane_index = options[0][1]
            self.vehicle.behavior_state = "changing_lane"

    def update(
        self,
        dt: float,
        elapsed_s: float,
        vehicles: tuple[VehicleSnapshot, ...],
        rng: Random,
    ) -> None:
        self.lane_change_cooldown_s = max(0.0, self.lane_change_cooldown_s - dt)
        current_lane = self.vehicle.lane_index
        effective_lane = self.target_lane_index if self.target_lane_index is not None else current_lane

        lead, lead_gap, _, _ = self._lane_gaps(effective_lane, vehicles)
        base_speed = self.cruise_speed_mps + self.speed_variation_mps * sin(
            elapsed_s * 0.19 + self.speed_phase_rad
        )
        target_speed = max(7.0, min(34.0, base_speed))
        desired_gap = 8.0 + self.vehicle.speed_mps * 1.0
        blocked = False
        if lead is not None and lead_gap is not None:
            blocked = lead_gap < desired_gap * 1.15 and lead.speed_mps < self.cruise_speed_mps - 1.0
            if lead_gap < desired_gap:
                following_speed = lead.speed_mps + (lead_gap - desired_gap) * 0.32
                target_speed = min(target_speed, max(0.0, following_speed))
            if lead_gap < desired_gap * 0.55:
                target_speed = min(target_speed, max(0.0, lead.speed_mps - 2.0))

        if self.target_lane_index is None:
            self._consider_lane_change(dt, vehicles, current_lane, blocked, rng)
        else:
            target_center = (self.target_lane_index + 0.5) * LANE_WIDTH_M
            if abs(self.vehicle.lateral_m - target_center) < 0.12:
                self.vehicle.lateral_m = target_center
                self.target_lane_index = None
                self.merge_from_shoulder = False
                self.lane_change_cooldown_s = rng.uniform(4.0, 8.0)
                self.vehicle.behavior_state = "cruise"

        target_lane = self.target_lane_index if self.target_lane_index is not None else current_lane
        target_lateral = (target_lane + 0.5) * LANE_WIDTH_M
        allow_shoulder = self.merge_from_shoulder and self.target_lane_index is not None
        self.vehicle.update_traffic(
            dt,
            target_speed_mps=target_speed,
            target_lateral_m=target_lateral,
            road_width_m=ROAD_WIDTH_M,
            lane_width_m=LANE_WIDTH_M,
            lane_count=LANE_COUNT,
            allow_shoulder=allow_shoulder,
        )

        if self.target_lane_index is None:
            if lead is not None and lead_gap is not None and lead_gap < desired_gap * 1.4:
                self.vehicle.behavior_state = "braking" if target_speed < self.vehicle.speed_mps - 0.6 else "following"
            else:
                self.vehicle.behavior_state = "cruise"


@dataclass
class DrivingSimulation:
    """A player car and a continuous, independently moving traffic stream."""

    seed: int | None = None
    elapsed_s: float = 0.0
    run_number: int = 0

    def __post_init__(self) -> None:
        if self.seed is None:
            self.seed = Random().randrange(1, 2**31)
        self._rng = Random(self.seed)
        self.traffic: list[TrafficAgent] = []
        self.restart()

    @property
    def run_id(self) -> str:
        return f"{self.seed}-{self.run_number}"

    def restart(self) -> None:
        self.run_number += 1
        self._rng = Random(int(self.seed) + self.run_number - 1)
        self.elapsed_s = 0.0
        center_lane = LANE_WIDTH_M * 1.5
        self.player = Vehicle(
            name="Player",
            vehicle_id="ego",
            longitudinal_m=0.0,
            lateral_m=center_lane,
            speed_mps=12.0,
            max_speed_mps=33.33,
            lane_index=1,
            behavior_state="player",
        )
        self.traffic = []
        offsets = (-100.0, -74.0, 32.0, 84.0)
        index = 0
        for lane in range(LANE_COUNT):
            for offset in offsets:
                index += 1
                position = offset + self._rng.uniform(-5.0, 5.0)
                speed = self._rng.uniform(11.0, 29.0)
                vehicle = Vehicle(
                    name=f"Traffic {index:02d}",
                    vehicle_id=f"traffic-{index:03d}",
                    longitudinal_m=position,
                    lateral_m=(lane + 0.5) * LANE_WIDTH_M,
                    speed_mps=speed,
                    lane_index=lane,
                    behavior_state="cruise",
                )
                self.traffic.append(
                    TrafficAgent(
                        vehicle=vehicle,
                        cruise_speed_mps=self._rng.uniform(12.0, 30.0),
                        speed_phase_rad=self._rng.uniform(0.0, 6.28),
                        lane_change_cooldown_s=self._rng.uniform(0.0, 4.0),
                        lane_decision_timer_s=self._rng.uniform(0.2, 1.2),
                    )
                )

    @property
    def traffic_density(self) -> str:
        visible = sum(
            abs(agent.vehicle.longitudinal_m - self.player.longitudinal_m) <= 100.0
            for agent in self.traffic
        )
        if visible <= 4:
            return "Light"
        if visible <= 8:
            return "Moderate"
        return "Heavy"

    @property
    def nearest_ahead(self) -> Vehicle | None:
        candidates = [
            agent.vehicle
            for agent in self.traffic
            if agent.vehicle.lane_index == self.player.lane_index
            and agent.vehicle.longitudinal_m > self.player.longitudinal_m
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda vehicle: vehicle.longitudinal_m)

    def _snapshots(self) -> tuple[VehicleSnapshot, ...]:
        snapshots = [
            VehicleSnapshot(
                vehicle_id=self.player.vehicle_id,
                longitudinal_m=self.player.longitudinal_m,
                lateral_m=self.player.lateral_m,
                speed_mps=self.player.speed_mps,
                length_m=self.player.length_m,
                lane_index=self.player.lane_index,
            )
        ]
        snapshots.extend(
            VehicleSnapshot(
                vehicle_id=agent.vehicle.vehicle_id,
                longitudinal_m=agent.vehicle.longitudinal_m,
                lateral_m=agent.vehicle.lateral_m,
                speed_mps=agent.vehicle.speed_mps,
                length_m=agent.vehicle.length_m,
                lane_index=agent.vehicle.lane_index,
                target_lane_index=agent.target_lane_index,
            )
            for agent in self.traffic
        )
        return tuple(snapshots)

    def _spawn_position(self, lane: int, direction: int) -> float:
        distance = self._rng.uniform(82.0, 118.0)
        candidate = self.player.longitudinal_m + distance * direction
        occupied = [
            abs(agent.vehicle.longitudinal_m - candidate)
            for agent in self.traffic
            if agent.vehicle.lane_index == lane
        ]
        if occupied and min(occupied) < 23.0:
            candidate += direction * (23.0 - min(occupied) + 20.0)
        return candidate

    def _recycle_distant_traffic(self) -> None:
        for agent in self.traffic:
            relative = agent.vehicle.longitudinal_m - self.player.longitudinal_m
            if -120.0 <= relative <= 135.0:
                continue
            lane = self._rng.randrange(LANE_COUNT)
            direction = -1 if relative > 0 else 1
            agent.vehicle.lane_index = lane
            agent.vehicle.lateral_m = (lane + 0.5) * LANE_WIDTH_M
            agent.vehicle.longitudinal_m = self._spawn_position(lane, direction)
            agent.vehicle.speed_mps = self._rng.uniform(11.0, 29.0)
            agent.vehicle.acceleration_mps2 = 0.0
            agent.vehicle.lateral_speed_mps = 0.0
            agent.vehicle.heading_deg = 0.0
            agent.vehicle.behavior_state = "cruise"
            agent.cruise_speed_mps = self._rng.uniform(12.0, 30.0)
            agent.speed_phase_rad = self._rng.uniform(0.0, 6.28)
            agent.target_lane_index = None
            agent.merge_from_shoulder = False
            # A small share of new traffic joins from the right-side merge lane.
            if self._rng.random() < 0.18:
                agent.vehicle.lateral_m = ROAD_WIDTH_M + 1.0
                agent.vehicle.lane_index = LANE_COUNT - 1
                agent.vehicle.behavior_state = "merging"
                agent.target_lane_index = LANE_COUNT - 1
                agent.merge_from_shoulder = True
            agent.lane_change_cooldown_s = self._rng.uniform(3.0, 8.0)

    def update(
        self,
        dt: float,
        *,
        throttle: float,
        steering: float,
        handbrake: bool = False,
    ) -> None:
        if dt <= 0:
            return

        self.elapsed_s += dt
        self.player.update_player(
            dt,
            throttle=throttle,
            steering=steering,
            road_width_m=ROAD_WIDTH_M,
            lane_width_m=LANE_WIDTH_M,
            lane_count=LANE_COUNT,
            handbrake=handbrake,
        )

        state = self._snapshots()
        for agent in self.traffic:
            agent.update(dt, self.elapsed_s, state, self._rng)
        self._recycle_distant_traffic()
