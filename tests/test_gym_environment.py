import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from src.ravan_game.risk import analyze_risk
from src.ravan_game.rl.actions import DriverAction
from src.ravan_game.rl.environment import RavanDrivingEnv
from src.ravan_game.rl.observations import build_observation


class RavanGymEnvironmentTests(unittest.TestCase):
    def test_reset_returns_bounded_finite_structured_observation(self):
        environment = RavanDrivingEnv(max_episode_steps=20)
        try:
            observation, info = environment.reset(seed=23)

            self.assertTrue(environment.observation_space.contains(observation))
            self.assertEqual(observation["ego"].shape, (11,))
            self.assertEqual(observation["front"].shape, (11,))
            self.assertEqual(observation["adjacent_lanes"].shape, (2, 9))
            self.assertEqual(observation["environment"].shape, (13,))
            self.assertTrue(all(np.isfinite(values).all() for values in observation.values()))
            self.assertEqual(info["seed"], 23)
        finally:
            environment.close()

    def test_reset_with_same_seed_reproduces_initial_traffic(self):
        environment = RavanDrivingEnv(max_episode_steps=20)
        try:
            first_observation, _ = environment.reset(seed=101)
            first_traffic = [
                (agent.vehicle.longitudinal_m, agent.vehicle.speed_mps, agent.vehicle.lane_index)
                for agent in environment.simulation.traffic
            ]
            second_observation, _ = environment.reset(seed=101)
            second_traffic = [
                (agent.vehicle.longitudinal_m, agent.vehicle.speed_mps, agent.vehicle.lane_index)
                for agent in environment.simulation.traffic
            ]

            for key in first_observation:
                np.testing.assert_array_equal(first_observation[key], second_observation[key])
            self.assertEqual(first_traffic, second_traffic)
        finally:
            environment.close()

    def test_undefined_ttc_has_capped_value_and_validity_flag(self):
        environment = RavanDrivingEnv(max_episode_steps=20)
        try:
            environment.reset(seed=8)
            environment.simulation.traffic.clear()
            environment._assessment = analyze_risk(environment.simulation)
            observation = build_observation(
                environment.simulation,
                environment._assessment,
                max_episode_steps=environment.max_episode_steps,
            )

            self.assertEqual(observation["front"][0], 0.0)
            self.assertEqual(observation["front"][5], 1.0)
            self.assertEqual(observation["front"][6], 0.0)
            self.assertEqual(observation["front"][7], 1.0)
            self.assertEqual(observation["front"][8], 0.0)
            self.assertTrue(environment.observation_space.contains(observation))
        finally:
            environment.close()

    def test_action_uses_existing_vehicle_physics_and_returns_five_values(self):
        environment = RavanDrivingEnv(max_episode_steps=20)
        try:
            environment.reset(seed=12)
            initial_speed = environment.simulation.player.speed_mps
            result = environment.step(DriverAction.ACCELERATE)

            self.assertEqual(len(result), 5)
            observation, reward, terminated, truncated, info = result
            self.assertGreater(environment.simulation.player.speed_mps, initial_speed)
            self.assertEqual(info["action_name"], "ACCELERATE")
            self.assertIsInstance(reward, float)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(environment.observation_space.contains(observation))
        finally:
            environment.close()

    def test_step_limit_truncates_episode(self):
        environment = RavanDrivingEnv(max_episode_steps=1)
        try:
            environment.reset(seed=44)
            _, _, terminated, truncated, _ = environment.step(DriverAction.COAST)

            self.assertFalse(terminated)
            self.assertTrue(truncated)
            with self.assertRaises(RuntimeError):
                environment.step(DriverAction.COAST)
        finally:
            environment.close()

    def test_collision_terminates_episode(self):
        environment = RavanDrivingEnv(max_episode_steps=20)
        try:
            environment.reset(seed=16)
            player = environment.simulation.player
            agent = environment.simulation.traffic[0]
            agent.vehicle.longitudinal_m = player.longitudinal_m
            agent.vehicle.lateral_m = player.lateral_m
            agent.vehicle.lane_index = player.lane_index
            agent.vehicle.speed_mps = player.speed_mps
            agent.cruise_speed_mps = player.speed_mps
            agent.target_lane_index = None

            _, _, terminated, truncated, info = environment.step(DriverAction.COAST)

            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["collision"])
            self.assertEqual(info["reward_components"]["collision"], -100.0)
        finally:
            environment.close()

    def test_invalid_action_is_rejected(self):
        environment = RavanDrivingEnv(max_episode_steps=20)
        try:
            environment.reset(seed=3)
            with self.assertRaises(ValueError):
                environment.step(99)
        finally:
            environment.close()

    def test_gymnasium_environment_checker(self):
        environment = RavanDrivingEnv(max_episode_steps=80)
        try:
            check_env(environment, skip_render_check=True)
        finally:
            environment.close()

    def test_rgb_array_render_returns_hwc_frame(self):
        environment = RavanDrivingEnv(render_mode="rgb_array", max_episode_steps=5)
        try:
            environment.reset(seed=7)
            frame = environment.render()

            self.assertEqual(frame.shape, (900, 1440, 3))
            self.assertEqual(frame.dtype, np.uint8)
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()
