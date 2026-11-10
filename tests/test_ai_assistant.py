import unittest

from src.ravan_game.ai_assistant import _verified_fallback
from src.ravan_game.risk import analyze_risk
from src.ravan_game.simulation import DrivingSimulation, LANE_WIDTH_M, TrafficAgent
from src.ravan_game.vehicle import Vehicle


class AssistantFallbackTests(unittest.TestCase):
    def setUp(self):
        simulation = DrivingSimulation(seed=5)
        simulation.player.speed_mps = 20.0
        vehicle = Vehicle(
            name="Slow lead",
            vehicle_id="test-lead",
            longitudinal_m=14.6,
            lateral_m=LANE_WIDTH_M * 1.5,
            speed_mps=8.0,
        )
        simulation.traffic = [TrafficAgent(vehicle, 8.0, 0.0)]
        self.context = analyze_risk(simulation).to_dict()

    def test_ttc_fallback_uses_risk_engine_value(self):
        reply = _verified_fallback("What is the TTC?", self.context)

        self.assertIn("0.83 seconds", reply)

    def test_lane_change_fallback_uses_computed_lane_checks(self):
        simulation = DrivingSimulation(seed=5)
        left_vehicle = Vehicle(
            name="Left lane vehicle",
            vehicle_id="test-left",
            longitudinal_m=6.0,
            lateral_m=LANE_WIDTH_M * 0.5,
            speed_mps=14.0,
        )
        simulation.traffic = [TrafficAgent(left_vehicle, 14.0, 0.0)]
        safe_context = analyze_risk(simulation).to_dict()
        reply = _verified_fallback("Is it safe to change lanes?", safe_context)

        self.assertIn("left is wait", reply)
        self.assertIn("right is clear", reply)


if __name__ == "__main__":
    unittest.main()
