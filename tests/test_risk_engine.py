import unittest

from src.ravan_game.risk import analyze_risk
from src.ravan_game.simulation import DrivingSimulation, LANE_WIDTH_M, TrafficAgent
from src.ravan_game.vehicle import Vehicle


def build_simulation(ego_speed_mps, traffic=()):
    simulation = DrivingSimulation(seed=5)
    simulation.player.speed_mps = ego_speed_mps
    simulation.traffic = []
    for index, (lane, longitudinal_m, speed_mps) in enumerate(traffic):
        vehicle = Vehicle(
            name=f"Traffic {index + 1}",
            vehicle_id=f"test-{index + 1}",
            longitudinal_m=longitudinal_m,
            lateral_m=(lane + 0.5) * LANE_WIDTH_M,
            speed_mps=speed_mps,
        )
        simulation.traffic.append(TrafficAgent(vehicle, speed_mps, 0.0))
    return simulation


class RiskEngineTests(unittest.TestCase):
    def test_close_following_uses_bumper_gap_and_reports_critical_ttc(self):
        simulation = build_simulation(20.0, [(1, 14.6, 8.0)])
        assessment = analyze_risk(simulation)

        self.assertEqual(assessment.risk_level, "CRITICAL")
        self.assertAlmostEqual(assessment.distance_m, 10.0, places=1)
        self.assertAlmostEqual(assessment.ttc_s, 10.0 / 12.0, places=2)
        self.assertAlmostEqual(assessment.drac_mps2, 7.2, places=1)

    def test_safe_following_reports_undefined_ttc_when_not_closing(self):
        assessment = analyze_risk(build_simulation(14.0, [(1, 44.0, 14.0)]))

        self.assertEqual(assessment.risk_level, "LOW")
        self.assertIsNone(assessment.ttc_s)
        self.assertTrue(assessment.lane_change_left.safe)
        self.assertTrue(assessment.lane_change_right.safe)

    def test_adjacent_vehicle_is_reported_and_blocks_lane_change(self):
        simulation = build_simulation(16.0, [(1, 31.0, 16.0), (2, 1.5, 16.5)])
        assessment = analyze_risk(simulation)

        self.assertEqual(assessment.risk_level, "HIGH")
        self.assertTrue(any("alongside" in item["lane_relation"] for item in assessment.nearby_vehicles))
        self.assertFalse(assessment.lane_change_right.safe)
        self.assertIn("vehicle_id", assessment.nearby_vehicles[0])
        self.assertIn("behavior_state", assessment.nearby_vehicles[0])

    def test_high_closing_speed_produces_high_risk(self):
        assessment = analyze_risk(build_simulation(25.0, [(1, 34.6, 11.0)]))

        self.assertEqual(assessment.risk_level, "HIGH")
        self.assertGreater(assessment.relative_speed_mps, 10.0)
        self.assertLess(assessment.ttc_s, 3.0)


if __name__ == "__main__":
    unittest.main()
