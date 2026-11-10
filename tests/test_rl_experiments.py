import unittest

from src.ravan_game.rl.environment import RavanDrivingEnv
from src.ravan_game.rl.actions import DriverAction
from src.ravan_game.rl.reward import RewardConfig
from src.ravan_game.rl.shield import PhysicsSafetyShield
from src.ravan_game.rl.scenarios import TEST_SCENARIOS, TRAIN_SCENARIOS, VALIDATION_SCENARIOS


class RavanScenarioTests(unittest.TestCase):
    def test_scenarios_are_partitioned_for_training_validation_and_test(self):
        self.assertEqual(len(set(TRAIN_SCENARIOS + VALIDATION_SCENARIOS + TEST_SCENARIOS)),
                         len(TRAIN_SCENARIOS + VALIDATION_SCENARIOS + TEST_SCENARIOS))

    def test_selected_scenario_is_reported_and_seed_repeats_exact_initial_state(self):
        env = RavanDrivingEnv(scenario_families=("fast_closing",))
        try:
            first, first_info = env.reset(seed=123, options={"scenario_family": "fast_closing"})
            first_traffic = [
                (agent.vehicle.longitudinal_m, agent.vehicle.speed_mps, agent.vehicle.lane_index)
                for agent in env.simulation.traffic
            ]
            second, second_info = env.reset(seed=123, options={"scenario_family": "fast_closing"})
            second_traffic = [
                (agent.vehicle.longitudinal_m, agent.vehicle.speed_mps, agent.vehicle.lane_index)
                for agent in env.simulation.traffic
            ]
            self.assertEqual(first_info["scenario_family"], "fast_closing")
            self.assertEqual(second_info["scenario_family"], "fast_closing")
            self.assertEqual(first_traffic, second_traffic)
            for key in first:
                self.assertTrue((first[key] == second[key]).all())
            self.assertLess(first_info["risk"]["distance_m"], 60.0)
        finally:
            env.close()

    def test_scenario_options_must_be_enabled(self):
        env = RavanDrivingEnv(scenario_families=("open_road",))
        try:
            with self.assertRaises(ValueError):
                env.reset(seed=1, options={"scenario_family": "cut_in"})
        finally:
            env.close()

    def test_training_resets_select_only_training_families(self):
        env = RavanDrivingEnv(scenario_families=TRAIN_SCENARIOS)
        try:
            for seed in range(10):
                _, info = env.reset(seed=seed)
                self.assertIn(info["scenario_family"], TRAIN_SCENARIOS)
        finally:
            env.close()

    def test_reward_config_reports_distinct_speed_and_handbrake_costs(self):
        env = RavanDrivingEnv(
            scenario_families=("open_road",),
            reward_config=RewardConfig.from_json(),
        )
        try:
            env.reset(seed=18, options={"scenario_family": "open_road"})
            _, _, terminated, _, info = env.step(DriverAction.HANDBRAKE)
            self.assertFalse(terminated)
            self.assertLess(info["reward_components"]["unnecessary_handbrake"], 0.0)
            self.assertIn("risk_exposure", info["reward_components"])
            self.assertIn("speed_excess", info["reward_components"])
        finally:
            env.close()

    def test_safety_shield_rejects_steering_into_occupied_lane_and_logs_reason(self):
        env = RavanDrivingEnv(scenario_families=("occupied_adjacent_lanes",))
        try:
            env.reset(seed=6, options={"scenario_family": "occupied_adjacent_lanes"})
            action, reason = PhysicsSafetyShield().filter(env, DriverAction.STEER_LEFT)
            self.assertEqual(action, DriverAction.COAST)
            self.assertIn("safety check", reason)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
