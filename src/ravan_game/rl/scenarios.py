"""Reproducible initial-state scenarios for the existing traffic simulation."""

from __future__ import annotations

from random import Random

from ..simulation import DrivingSimulation, LANE_COUNT, LANE_WIDTH_M


TRAIN_SCENARIOS = (
    "open_road",
    "safe_following",
    "close_following",
    "fast_closing",
    "lead_braking",
)
VALIDATION_SCENARIOS = ("safe_lane_change", "occupied_adjacent_lanes")
TEST_SCENARIOS = ("cut_in", "blind_spot", "dense_traffic")
ALL_SCENARIOS = TRAIN_SCENARIOS + VALIDATION_SCENARIOS + TEST_SCENARIOS


def _set_vehicle(
    simulation: DrivingSimulation,
    agent_index: int,
    *,
    lane: int,
    center_offset_m: float,
    speed_mps: float,
    cruise_speed_mps: float | None = None,
    target_lane: int | None = None,
    speed_variation_mps: float = 0.0,
) -> None:
    agent = simulation.traffic[agent_index]
    vehicle = agent.vehicle
    vehicle.lane_index = lane
    vehicle.lateral_m = (lane + 0.5) * LANE_WIDTH_M
    vehicle.longitudinal_m = simulation.player.longitudinal_m + center_offset_m
    vehicle.speed_mps = max(0.0, min(vehicle.max_speed_mps, speed_mps))
    vehicle.acceleration_mps2 = 0.0
    vehicle.lateral_speed_mps = 0.0
    vehicle.heading_deg = 0.0
    vehicle.behavior_state = "changing_lane" if target_lane is not None else "cruise"
    agent.cruise_speed_mps = cruise_speed_mps if cruise_speed_mps is not None else vehicle.speed_mps
    agent.speed_variation_mps = speed_variation_mps
    agent.target_lane_index = target_lane
    agent.lane_change_cooldown_s = 4.0
    agent.lane_decision_timer_s = 10.0
    agent.merge_from_shoulder = False


def apply_scenario(simulation: DrivingSimulation, family: str, rng: Random) -> None:
    """Place existing simulated vehicles into a scenario without new physics."""
    if family not in ALL_SCENARIOS:
        raise ValueError(f"Unknown Ravan scenario {family!r}; choose from {ALL_SCENARIOS}.")

    player = simulation.player
    player.longitudinal_m = 0.0
    player.lane_index = 1
    player.lateral_m = 1.5 * LANE_WIDTH_M
    player.speed_mps = 20.0
    player.acceleration_mps2 = 0.0
    player.lateral_speed_mps = 0.0
    player.heading_deg = 0.0

    # Spread background traffic through the same three lanes. All vehicles
    # keep using TrafficAgent.update and Vehicle.update_traffic afterward.
    offsets = (-110.0, -75.0, 92.0, 118.0)
    for index, agent in enumerate(simulation.traffic):
        lane = index // 4
        offset = offsets[index % 4] + rng.uniform(-4.0, 4.0)
        speed = rng.uniform(15.0, 27.0)
        _set_vehicle(
            simulation,
            index,
            lane=lane,
            center_offset_m=offset,
            speed_mps=speed,
            cruise_speed_mps=speed,
            speed_variation_mps=0.3,
        )
        agent.lane_change_cooldown_s = rng.uniform(1.0, 4.0)
        agent.lane_decision_timer_s = rng.uniform(2.0, 8.0)

    def center_for_gap(gap_m: float, vehicle_index: int = 4) -> float:
        vehicle = simulation.traffic[vehicle_index].vehicle
        return (player.length_m + vehicle.length_m) / 2.0 + gap_m

    if family == "open_road":
        player.speed_mps = 22.0
        for index, agent in enumerate(simulation.traffic):
            lane = index // 4
            offset = (-120.0, -105.0, 112.0, 128.0)[index % 4]
            _set_vehicle(
                simulation,
                index,
                lane=lane,
                center_offset_m=offset,
                speed_mps=22.0 + rng.uniform(-2.0, 2.0),
                speed_variation_mps=0.2,
            )

    elif family == "safe_following":
        player.speed_mps = 20.0
        _set_vehicle(simulation, 4, lane=1, center_offset_m=center_for_gap(48.0), speed_mps=19.0)

    elif family == "close_following":
        player.speed_mps = 20.0
        _set_vehicle(simulation, 4, lane=1, center_offset_m=center_for_gap(14.0), speed_mps=16.0)

    elif family == "fast_closing":
        player.speed_mps = 28.0
        _set_vehicle(simulation, 4, lane=1, center_offset_m=center_for_gap(48.0), speed_mps=15.0)

    elif family == "lead_braking":
        player.speed_mps = 24.0
        _set_vehicle(
            simulation,
            4,
            lane=1,
            center_offset_m=center_for_gap(34.0),
            speed_mps=20.0,
            cruise_speed_mps=7.0,
            speed_variation_mps=0.0,
        )

    elif family == "safe_lane_change":
        player.speed_mps = 20.0
        _set_vehicle(simulation, 4, lane=1, center_offset_m=center_for_gap(18.0), speed_mps=12.0)
        _set_vehicle(simulation, 0, lane=0, center_offset_m=55.0, speed_mps=19.0)
        _set_vehicle(simulation, 1, lane=0, center_offset_m=-65.0, speed_mps=19.0)

    elif family == "occupied_adjacent_lanes":
        player.speed_mps = 20.0
        _set_vehicle(simulation, 4, lane=1, center_offset_m=center_for_gap(28.0), speed_mps=15.0)
        _set_vehicle(simulation, 0, lane=0, center_offset_m=18.0, speed_mps=20.0)
        _set_vehicle(simulation, 1, lane=0, center_offset_m=-13.0, speed_mps=24.0)
        _set_vehicle(simulation, 8, lane=2, center_offset_m=16.0, speed_mps=20.0)
        _set_vehicle(simulation, 9, lane=2, center_offset_m=-12.0, speed_mps=23.0)

    elif family == "cut_in":
        player.speed_mps = 21.0
        _set_vehicle(simulation, 4, lane=1, center_offset_m=center_for_gap(72.0), speed_mps=20.0)
        _set_vehicle(
            simulation,
            8,
            lane=2,
            center_offset_m=center_for_gap(26.0, 8),
            speed_mps=19.0,
            target_lane=1,
        )

    elif family == "blind_spot":
        player.speed_mps = 18.0
        _set_vehicle(simulation, 4, lane=1, center_offset_m=center_for_gap(50.0), speed_mps=18.0)
        _set_vehicle(simulation, 0, lane=0, center_offset_m=-7.0, speed_mps=23.0)

    elif family == "dense_traffic":
        player.speed_mps = 18.0
        _set_vehicle(simulation, 4, lane=1, center_offset_m=center_for_gap(17.0), speed_mps=15.0)
        _set_vehicle(simulation, 5, lane=1, center_offset_m=-15.0, speed_mps=22.0)
        _set_vehicle(simulation, 0, lane=0, center_offset_m=17.0, speed_mps=18.0)
        _set_vehicle(simulation, 1, lane=0, center_offset_m=-16.0, speed_mps=21.0)
        _set_vehicle(simulation, 8, lane=2, center_offset_m=14.0, speed_mps=17.0)
        _set_vehicle(simulation, 9, lane=2, center_offset_m=-14.0, speed_mps=22.0)

    player._update_lane_and_heading(LANE_WIDTH_M, LANE_COUNT)
