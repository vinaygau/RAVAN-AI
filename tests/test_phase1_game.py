import unittest

from src.ravan_game.simulation import (
    DrivingSimulation,
    LANE_COUNT,
    LANE_WIDTH_M,
    ROAD_WIDTH_M,
    TrafficAgent,
)
from src.ravan_game.vehicle import Vehicle


class DrivingSimulationTests(unittest.TestCase):
    def test_player_accelerates_and_brakes(self):
        simulation = DrivingSimulation(seed=10)
        initial_speed = simulation.player.speed_mps

        simulation.update(0.5, throttle=1.0, steering=0.0)
        accelerated_speed = simulation.player.speed_mps
        self.assertGreater(accelerated_speed, initial_speed)

        simulation.update(0.5, throttle=-1.0, steering=0.0)
        self.assertLess(simulation.player.speed_mps, accelerated_speed)
        self.assertGreaterEqual(simulation.player.speed_mps, 0.0)

    def test_player_can_steer_between_lanes(self):
        simulation = DrivingSimulation(seed=10)
        starting_lateral_position = simulation.player.lateral_m

        for _ in range(20):
            simulation.update(0.05, throttle=0.0, steering=-1.0)

        self.assertLess(simulation.player.lateral_m, starting_lateral_position)
        self.assertGreaterEqual(simulation.player.lateral_m, simulation.player.width_m / 2.0)
        self.assertLessEqual(
            simulation.player.lateral_m,
            ROAD_WIDTH_M - simulation.player.width_m / 2.0,
        )

    def test_ai_vehicle_moves_independently(self):
        simulation = DrivingSimulation(seed=10)
        traffic_vehicle = simulation.traffic[0].vehicle
        ai_start = traffic_vehicle.longitudinal_m
        player_start = simulation.player.longitudinal_m

        for _ in range(30):
            simulation.update(0.05, throttle=0.5, steering=0.0)

        self.assertGreater(traffic_vehicle.longitudinal_m, ai_start)
        self.assertGreater(simulation.player.longitudinal_m, player_start)
        self.assertNotEqual(traffic_vehicle.longitudinal_m, simulation.player.longitudinal_m)

    def test_traffic_occupies_all_lanes_with_distinct_speeds(self):
        simulation = DrivingSimulation(seed=10)
        lanes = {agent.vehicle.lane_index for agent in simulation.traffic}
        speeds = {round(agent.vehicle.speed_mps, 1) for agent in simulation.traffic}
        ids = {agent.vehicle.vehicle_id for agent in simulation.traffic}

        self.assertEqual(len(simulation.traffic), 12)
        self.assertEqual(lanes, set(range(LANE_COUNT)))
        self.assertEqual(len(ids), len(simulation.traffic))
        self.assertGreater(len(speeds), 1)

    def test_continuous_traffic_initialization_has_no_forced_collision(self):
        from src.ravan_game.risk import analyze_risk

        self.assertFalse(analyze_risk(DrivingSimulation(seed=10)).collision)

    def test_nearest_same_lane_vehicle_is_reported(self):
        simulation = DrivingSimulation(seed=10)
        candidates = [
            agent.vehicle for agent in simulation.traffic
            if agent.vehicle.lane_index == simulation.player.lane_index
            and agent.vehicle.longitudinal_m > simulation.player.longitudinal_m
        ]
        self.assertIsNotNone(simulation.nearest_ahead)
        self.assertEqual(
            simulation.nearest_ahead.vehicle_id,
            min(candidates, key=lambda vehicle: vehicle.longitudinal_m).vehicle_id,
        )

    def test_traffic_vehicle_changes_lane_to_pass_slow_lead_when_clear(self):
        simulation = DrivingSimulation(seed=10)
        simulation.traffic = []
        mover = Vehicle(
            name="Maneuvering traffic",
            vehicle_id="traffic-test-mover",
            longitudinal_m=50.0,
            lateral_m=LANE_WIDTH_M * 0.5,
            speed_mps=18.0,
        )
        slow_lead = Vehicle(
            name="Slow lead",
            vehicle_id="traffic-test-lead",
            longitudinal_m=67.0,
            lateral_m=LANE_WIDTH_M * 0.5,
            speed_mps=5.0,
        )
        simulation.traffic = [
            TrafficAgent(mover, 23.0, 0.0, lane_decision_timer_s=0.0),
            TrafficAgent(slow_lead, 5.0, 0.0),
        ]

        simulation.update(0.05, throttle=0.0, steering=0.0)

        self.assertEqual(simulation.traffic[0].target_lane_index, 1)
        self.assertEqual(mover.behavior_state, "changing_lane")


if __name__ == "__main__":
    unittest.main()
