"""Snapshot-based Q-learning for an advisory lane path through traffic.

The planner divides the road ahead into short longitudinal segments. Each
state is a (segment, lane) pair, and each action keeps the lane or moves one
lane left/right while advancing. A route is scored by geometric distance,
lane-change effort, and a penalty for predicted traffic conflict.

This is a research/education feature. It reports a route suggestion and never
applies steering or throttle to the player vehicle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot
from random import Random

from .risk import RiskAssessment
from .simulation import DrivingSimulation, LANE_COUNT, LANE_WIDTH_M


@dataclass(frozen=True)
class LanePathPlan:
    """A Q-learning lane sequence for the next short road horizon."""

    lanes: tuple[int, ...]
    segment_distance_m: float
    horizon_distance_m: float
    predicted_cost_m: float
    predicted_risk_scores: tuple[int, ...]
    first_action: str
    rationale: str

    @staticmethod
    def lane_name(lane: int) -> str:
        return ("LEFT", "MIDDLE", "RIGHT")[max(0, min(LANE_COUNT - 1, lane))]

    @property
    def display_path(self) -> str:
        abbreviations = {0: "L", 1: "M", 2: "R"}
        return " → ".join(abbreviations[lane] for lane in self.lanes)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["lanes"] = list(self.lanes)
        result["predicted_risk_scores"] = list(self.predicted_risk_scores)
        result["lane_names"] = [self.lane_name(lane) for lane in self.lanes]
        result["display_path"] = self.display_path
        return result


def _lane_risk_score(
    simulation: DrivingSimulation,
    lane: int,
    forecast_s: float,
) -> int:
    """Estimate traffic conflict in one lane under constant-speed motion."""
    ego = simulation.player
    score = 0.0
    minimum_safe_gap = max(12.0, ego.speed_mps * 1.2)

    for agent in simulation.traffic:
        occupied_lanes = {agent.vehicle.lane_index}
        if agent.target_lane_index is not None:
            occupied_lanes.add(agent.target_lane_index)
        if lane not in occupied_lanes:
            continue

        other = agent.vehicle
        relative_position = (
            other.longitudinal_m - ego.longitudinal_m
            + (other.speed_mps - ego.speed_mps) * forecast_s
        )
        bumper_gap = max(0.0, abs(relative_position) - (ego.length_m + other.length_m) / 2.0)
        if abs(relative_position) < (ego.length_m + other.length_m) / 2.0:
            score = max(score, 100.0)
            continue

        if relative_position >= 0:
            closing_speed = ego.speed_mps - other.speed_mps
        else:
            closing_speed = other.speed_mps - ego.speed_mps

        if closing_speed > 0 and bumper_gap > 0:
            ttc = bumper_gap / closing_speed
            if ttc <= 1.5:
                score = max(score, 100.0)
            elif ttc <= 3.0:
                score = max(score, 76.0)
            elif ttc <= 5.0:
                score = max(score, 48.0)
            elif ttc <= 8.0:
                score = max(score, 24.0)

        if bumper_gap < minimum_safe_gap:
            proximity_score = 48.0 * (minimum_safe_gap - bumper_gap) / minimum_safe_gap
            score = max(score, proximity_score)

    return int(round(max(0.0, min(100.0, score))))


def plan_lane_path(
    simulation: DrivingSimulation,
    assessment: RiskAssessment,
    *,
    segment_distance_m: float = 18.0,
    segments: int = 5,
    training_episodes: int = 180,
) -> LanePathPlan:
    """Learn a low-cost lane sequence for the next 90 metres of simulated road.

    Q-learning operates on a small layered state space. Its reward is the
    negative of path distance, lane-change cost, and predicted conflict cost.
    The first lane-change action is restricted by the physics engine's
    current front/rear safety check.
    """
    segment_distance_m = max(6.0, float(segment_distance_m))
    segments = max(1, int(segments))
    current_lane = max(0, min(LANE_COUNT - 1, int(simulation.player.lateral_m / LANE_WIDTH_M)))
    horizon_distance_m = segment_distance_m * segments
    speed_mps = max(3.0, simulation.player.speed_mps)

    lane_risks: list[list[int]] = []
    for step in range(segments + 1):
        forecast_s = step * segment_distance_m / speed_mps
        lane_risks.append([
            _lane_risk_score(simulation, lane, forecast_s)
            for lane in range(LANE_COUNT)
        ])

    safe_first_lanes = {current_lane}
    if current_lane > 0 and assessment.lane_change_left.safe:
        safe_first_lanes.add(current_lane - 1)
    if current_lane < LANE_COUNT - 1 and assessment.lane_change_right.safe:
        safe_first_lanes.add(current_lane + 1)

    def actions(step: int, lane: int) -> tuple[int, ...]:
        candidates = tuple(
            candidate
            for candidate in (lane - 1, lane, lane + 1)
            if 0 <= candidate < LANE_COUNT
        )
        if step == 0:
            return tuple(candidate for candidate in candidates if candidate in safe_first_lanes)
        return candidates

    q_values: dict[tuple[int, int], dict[int, float]] = {}
    rng = Random((int(simulation.seed or 0) * 1_000_003) ^ int(simulation.elapsed_s * 2))
    alpha = 0.35
    # The layered horizon is finite and acyclic, so gamma=1 keeps the objective
    # equal to the full route cost rather than discounting later segments.
    gamma = 1.0
    risk_cost_per_point_m = 0.8
    lane_change_cost_m = 5.0

    for episode in range(max(1, int(training_episodes))):
        epsilon = max(0.04, 0.8 * (0.982 ** episode))
        lane = current_lane
        for step in range(segments):
            state = (step, lane)
            choices = actions(step, lane)
            values = q_values.setdefault(state, {choice: 0.0 for choice in choices})
            if rng.random() < epsilon:
                next_lane = rng.choice(choices)
            else:
                best_value = max(values.get(choice, 0.0) for choice in choices)
                best_choices = [choice for choice in choices if values.get(choice, 0.0) == best_value]
                next_lane = rng.choice(best_choices)

            lateral_distance = abs(next_lane - lane) * LANE_WIDTH_M
            geometric_distance = hypot(segment_distance_m, lateral_distance)
            maneuver_cost = lane_change_cost_m if next_lane != lane else 0.0
            risk_cost = lane_risks[step + 1][next_lane] * risk_cost_per_point_m
            reward = -(geometric_distance + maneuver_cost + risk_cost)

            next_step = step + 1
            if next_step >= segments:
                future_value = 0.0
            else:
                next_state = (next_step, next_lane)
                next_choices = actions(next_step, next_lane)
                next_values = q_values.setdefault(
                    next_state,
                    {choice: 0.0 for choice in next_choices},
                )
                future_value = max(next_values.get(choice, 0.0) for choice in next_choices)

            old_value = values.get(next_lane, 0.0)
            values[next_lane] = old_value + alpha * (reward + gamma * future_value - old_value)
            lane = next_lane

    def greedy_route(*, hold_first_segment: bool = False) -> tuple[list[int], list[int], float]:
        planned_lanes = [current_lane]
        planned_risks = [lane_risks[0][current_lane]]
        route_cost = 0.0
        lane = current_lane
        for step in range(segments):
            choices = actions(step, lane)
            values = q_values.get((step, lane), {})
            if hold_first_segment and step == 0:
                next_lane = lane
            else:
                best_value = max(values.get(choice, float("-inf")) for choice in choices)
                best_choices = [choice for choice in choices if values.get(choice, float("-inf")) == best_value]
                # Stable tie-breaking prefers staying in the current lane.
                next_lane = lane if lane in best_choices else min(best_choices)
            lateral_distance = abs(next_lane - lane) * LANE_WIDTH_M
            route_cost += hypot(segment_distance_m, lateral_distance)
            if next_lane != lane:
                route_cost += lane_change_cost_m
            next_risk = lane_risks[step + 1][next_lane]
            route_cost += next_risk * risk_cost_per_point_m
            planned_lanes.append(next_lane)
            planned_risks.append(next_risk)
            lane = next_lane
        return planned_lanes, planned_risks, route_cost

    route, route_risks, total_cost = greedy_route()
    if max(route_risks) >= 76:
        route, route_risks, total_cost = greedy_route(hold_first_segment=True)
        first_action = "stay_and_reduce_speed"
        rationale = "No low-risk lane path is predicted; hold the checked lane for now, reduce speed, and recheck."
    elif route[1] == current_lane:
        first_action = "stay"
        rationale = "Q-learning selected the current lane for the first segment."
    else:
        direction = "left" if route[1] < current_lane else "right"
        first_action = f"change_{direction}_when_clear"
        rationale = (
            f"Q-learning selected the {LanePathPlan.lane_name(route[1]).lower()} lane "
            "for the first segment; the physics lane check currently allows that move."
        )

    return LanePathPlan(
        lanes=tuple(route),
        segment_distance_m=round(segment_distance_m, 1),
        horizon_distance_m=round(horizon_distance_m, 1),
        predicted_cost_m=round(total_cost, 1),
        predicted_risk_scores=tuple(route_risks),
        first_action=first_action,
        rationale=rationale,
    )
