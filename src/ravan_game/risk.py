"""Deterministic, explainable risk metrics for the simulated road scene."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

from .simulation import DrivingSimulation, LANE_COUNT, LANE_WIDTH_M, ROAD_WIDTH_M
from .vehicle import Vehicle


def _lane_index(lateral_m: float) -> int:
    return max(0, min(LANE_COUNT - 1, int(lateral_m / LANE_WIDTH_M)))


def _bumper_gap(first: Vehicle, second: Vehicle) -> float:
    center_distance = abs(first.longitudinal_m - second.longitudinal_m)
    return max(0.0, center_distance - (first.length_m + second.length_m) / 2.0)


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None or not isfinite(value):
        return None
    return round(value, digits)


@dataclass(frozen=True)
class LaneChangeCheck:
    safe: bool
    label: str
    reason: str
    front_gap_m: float | None = None
    rear_gap_m: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RiskAssessment:
    risk_level: str
    risk_score: int
    collision: bool
    ego_speed_kmh: float
    nearest_front: dict[str, object] | None
    nearest_rear: dict[str, object] | None
    nearby_vehicles: list[dict[str, object]]
    relative_speed_mps: float | None
    relative_acceleration_mps2: float | None
    distance_m: float | None
    ttc_s: float | None
    thw_s: float | None
    drac_mps2: float | None
    lane_change_left: LaneChangeCheck
    lane_change_right: LaneChangeCheck
    main_factors: list[str]
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["ego_speed_kmh"] = round(self.ego_speed_kmh, 1)
        result["relative_speed_mps"] = _round_or_none(self.relative_speed_mps)
        result["relative_acceleration_mps2"] = _round_or_none(self.relative_acceleration_mps2)
        result["distance_m"] = _round_or_none(self.distance_m)
        result["ttc_s"] = _round_or_none(self.ttc_s)
        result["thw_s"] = _round_or_none(self.thw_s)
        result["drac_mps2"] = _round_or_none(self.drac_mps2)
        return result


def _same_lane_ttc(gap_m: float, closing_speed_mps: float) -> float | None:
    if gap_m <= 0 or closing_speed_mps <= 0:
        return None
    return gap_m / closing_speed_mps


def _lane_change_check(
    simulation: DrivingSimulation,
    direction: int,
) -> LaneChangeCheck:
    current_lane = _lane_index(simulation.player.lateral_m)
    target_lane = current_lane + direction
    side = "left" if direction < 0 else "right"
    if not 0 <= target_lane < LANE_COUNT:
        return LaneChangeCheck(False, "BLOCKED", f"There is no lane to the {side}.")

    front_gap: float | None = None
    rear_gap: float | None = None
    problems: list[str] = []

    for agent in simulation.traffic:
        other = agent.vehicle
        if _lane_index(other.lateral_m) != target_lane and agent.target_lane_index != target_lane:
            continue
        delta_y = other.longitudinal_m - simulation.player.longitudinal_m
        gap = _bumper_gap(simulation.player, other)
        if delta_y >= 0:
            front_gap = gap if front_gap is None else min(front_gap, gap)
            closing = simulation.player.speed_mps - other.speed_mps
            ttc = _same_lane_ttc(gap, closing)
            if gap < 12.0:
                problems.append(f"front gap is only {gap:.1f} m")
            elif ttc is not None and ttc < 4.0:
                problems.append(f"closing on the front vehicle in {ttc:.1f} s")
        else:
            rear_gap = gap if rear_gap is None else min(rear_gap, gap)
            closing = other.speed_mps - simulation.player.speed_mps
            ttc = _same_lane_ttc(gap, closing)
            if gap < 10.0:
                problems.append(f"rear gap is only {gap:.1f} m")
            elif ttc is not None and ttc < 4.0:
                problems.append(f"rear vehicle is closing in {ttc:.1f} s")

    if problems:
        return LaneChangeCheck(
            False,
            "WAIT",
            f"Wait: {', '.join(problems)}.",
            _round_or_none(front_gap),
            _round_or_none(rear_gap),
        )

    return LaneChangeCheck(
        True,
        "CLEAR",
        f"No close front or rear vehicle detected in the {side} lane.",
        _round_or_none(front_gap),
        _round_or_none(rear_gap),
    )


def analyze_risk(simulation: DrivingSimulation) -> RiskAssessment:
    """Calculate current kinematic risk and adjacent-lane checks."""
    ego = simulation.player
    ego_lane = _lane_index(ego.lateral_m)
    front: list[Vehicle] = []
    rear: list[Vehicle] = []
    nearby: list[dict[str, object]] = []
    collision = False
    closest_side_offset_m: float | None = None

    for agent in simulation.traffic:
        other = agent.vehicle
        delta_y = other.longitudinal_m - ego.longitudinal_m
        delta_x = other.lateral_m - ego.lateral_m
        occupied_lanes = {other.lane_index}
        if agent.target_lane_index is not None:
            occupied_lanes.add(agent.target_lane_index)
        same_lane = ego_lane in occupied_lanes
        gap = _bumper_gap(ego, other)
        closing_speed = ego.speed_mps - other.speed_mps if delta_y >= 0 else other.speed_mps - ego.speed_mps
        ttc = _same_lane_ttc(gap, closing_speed) if same_lane else None
        overlap_x = abs(delta_x) < (ego.width_m + other.width_m) / 2.0
        overlap_y = abs(delta_y) < (ego.length_m + other.length_m) / 2.0
        collision = collision or (overlap_x and overlap_y)

        if same_lane:
            (front if delta_y >= 0 else rear).append(other)
            relation = "same-lane-front" if delta_y >= 0 else "same-lane-rear"
        else:
            relation = "adjacent-lane-ahead" if delta_y >= 0 else "adjacent-lane-rear"
            if any(abs(lane - ego_lane) == 1 for lane in occupied_lanes) and abs(delta_y) <= 8.0:
                closest_side_offset_m = (
                    abs(delta_y)
                    if closest_side_offset_m is None
                    else min(closest_side_offset_m, abs(delta_y))
                )
                relation = "alongside"

        nearby.append({
            "vehicle_id": other.vehicle_id,
            "vehicle": other.name,
            "lane_relation": relation,
            "lane_index": other.lane_index,
            "heading_deg": round(other.heading_deg, 2),
            "behavior_state": other.behavior_state,
            "distance_m": round(gap, 2),
            "longitudinal_offset_m": round(delta_y, 2),
            "lateral_offset_m": round(delta_x, 2),
            "speed_kmh": round(other.speed_mps * 3.6, 1),
            "relative_speed_mps": round(ego.speed_mps - other.speed_mps, 2),
            "relative_acceleration_mps2": round(ego.acceleration_mps2 - other.acceleration_mps2, 2),
            "ttc_s": _round_or_none(ttc),
        })

    nearest_front = min(front, key=lambda vehicle: vehicle.longitudinal_m) if front else None
    nearest_rear = max(rear, key=lambda vehicle: vehicle.longitudinal_m) if rear else None
    front_gap = _bumper_gap(ego, nearest_front) if nearest_front else None
    rear_gap = _bumper_gap(ego, nearest_rear) if nearest_rear else None
    relative_speed = ego.speed_mps - nearest_front.speed_mps if nearest_front else None
    relative_acceleration = ego.acceleration_mps2 - nearest_front.acceleration_mps2 if nearest_front else None
    ttc = _same_lane_ttc(front_gap, relative_speed) if front_gap is not None and relative_speed is not None else None
    thw = front_gap / ego.speed_mps if front_gap is not None and ego.speed_mps > 0 else None
    drac = (
        relative_speed**2 / (2.0 * front_gap)
        if front_gap is not None and front_gap > 0 and relative_speed is not None and relative_speed > 0
        else None
    )

    rear_ttc = None
    if nearest_rear is not None and rear_gap is not None:
        rear_ttc = _same_lane_ttc(rear_gap, nearest_rear.speed_mps - ego.speed_mps)

    relevant_ttc = min((value for value in (ttc, rear_ttc) if value is not None), default=None)
    if collision or (relevant_ttc is not None and relevant_ttc <= 1.5):
        risk_level, risk_score = "CRITICAL", 96 if not collision else 100
    elif (relevant_ttc is not None and relevant_ttc <= 3.0) or (
        closest_side_offset_m is not None and closest_side_offset_m <= 3.0
    ):
        risk_level, risk_score = "HIGH", 76
    elif (relevant_ttc is not None and relevant_ttc <= 5.0) or (
        closest_side_offset_m is not None and closest_side_offset_m <= 8.0
    ):
        risk_level, risk_score = "MODERATE", 48
    else:
        risk_level, risk_score = "LOW", 12

    left_check = _lane_change_check(simulation, -1)
    right_check = _lane_change_check(simulation, 1)
    factors: list[str] = []
    if collision:
        factors.append("Vehicle envelopes overlap in the current sample.")
    if nearest_front is not None:
        if front_gap is not None:
            factors.append(f"Same-lane front bumper gap is {front_gap:.1f} m.")
        if relative_speed is not None and relative_speed > 0:
            factors.append(f"You are closing on the front vehicle at {relative_speed:.1f} m/s.")
        if ttc is not None:
            factors.append(f"Constant-speed TTC to the front vehicle is {ttc:.1f} s.")
    if nearest_rear is not None and rear_ttc is not None:
        factors.append(f"A same-lane rear vehicle is closing; estimated TTC is {rear_ttc:.1f} s.")
    if closest_side_offset_m is not None:
        factors.append("A vehicle is alongside in an adjacent lane.")
    if not factors:
        factors.append("No close same-lane or alongside vehicle is detected.")

    if risk_level == "CRITICAL":
        recommendation = "Brake firmly and keep your lane. Do not steer into a blocked gap."
    elif risk_level == "HIGH":
        recommendation = "Ease off the accelerator and prepare to brake. Wait for a clear lane before moving over."
    elif risk_level == "MODERATE":
        recommendation = "Reduce speed gently and increase your following distance."
    elif left_check.safe or right_check.safe:
        recommendation = "Maintain a steady speed; a checked adjacent lane is available if needed."
    else:
        recommendation = "Maintain a steady speed and keep monitoring the vehicles around you."

    def vehicle_record(vehicle: Vehicle | None, gap: float | None) -> dict[str, object] | None:
        if vehicle is None:
            return None
        is_front = vehicle.longitudinal_m >= ego.longitudinal_m
        closing = ego.speed_mps - vehicle.speed_mps if is_front else vehicle.speed_mps - ego.speed_mps
        vehicle_ttc = _same_lane_ttc(gap or 0.0, closing)
        return {
            "vehicle_id": vehicle.vehicle_id,
            "name": vehicle.name,
            "lane_index": vehicle.lane_index,
            "heading_deg": round(vehicle.heading_deg, 2),
            "behavior_state": vehicle.behavior_state,
            "speed_kmh": round(vehicle.speed_mps * 3.6, 1),
            "distance_m": _round_or_none(gap),
            "relative_speed_mps": round(ego.speed_mps - vehicle.speed_mps, 2),
            "relative_acceleration_mps2": round(ego.acceleration_mps2 - vehicle.acceleration_mps2, 2),
            "ttc_s": _round_or_none(vehicle_ttc),
        }

    return RiskAssessment(
        risk_level=risk_level,
        risk_score=risk_score,
        collision=collision,
        ego_speed_kmh=ego.speed_mps * 3.6,
        nearest_front=vehicle_record(nearest_front, front_gap),
        nearest_rear=vehicle_record(nearest_rear, rear_gap),
        nearby_vehicles=nearby,
        relative_speed_mps=relative_speed,
        relative_acceleration_mps2=relative_acceleration,
        distance_m=front_gap,
        ttc_s=ttc,
        thw_s=thw,
        drac_mps2=drac,
        lane_change_left=left_check,
        lane_change_right=right_check,
        main_factors=factors,
        recommendation=recommendation,
    )
