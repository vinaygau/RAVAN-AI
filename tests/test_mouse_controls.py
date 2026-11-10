import unittest

from src.ravan_game.controls import MouseDriveControl
from src.ravan_game.simulation import DrivingSimulation, LANE_WIDTH_M


class MouseDriveControlTests(unittest.TestCase):
    def test_tap_remains_active_after_release(self):
        control = MouseDriveControl()
        control.press("accelerate")
        self.assertEqual(control.active_action, "accelerate")

        control.release()
        self.assertEqual(control.active_action, "accelerate")
        control.advance(0.10)
        self.assertEqual(control.active_action, "accelerate")
        control.advance(0.20)
        self.assertIsNone(control.active_action)

    def test_held_steering_stays_active_until_released(self):
        control = MouseDriveControl()
        control.press("left")
        control.advance(1.0)
        self.assertEqual(control.active_action, "left")

        control.release()
        control.advance(control.steering_tap_duration_s)
        self.assertIsNone(control.active_action)

    def test_pause_clears_mouse_action(self):
        control = MouseDriveControl()
        control.press("brake")
        control.clear()
        self.assertIsNone(control.active_action)

    def test_timed_pulse_runs_without_requiring_mouse_button_release(self):
        control = MouseDriveControl()
        control.pulse("accelerate", 1.25)

        self.assertEqual(control.active_action, "accelerate")
        control.advance(1.0)
        self.assertEqual(control.active_action, "accelerate")
        control.advance(0.25)
        self.assertIsNone(control.active_action)

    def test_single_steering_tap_changes_lane(self):
        control = MouseDriveControl()
        simulation = DrivingSimulation(seed=17)
        starting_lane = int(simulation.player.lateral_m / LANE_WIDTH_M)
        control.press("right")
        control.release()

        for _ in range(24):
            action = control.active_action
            simulation.update(
                0.05,
                throttle=0.0,
                steering=1.0 if action == "right" else 0.0,
            )
            control.advance(0.05)

        self.assertGreater(int(simulation.player.lateral_m / LANE_WIDTH_M), starting_lane)


if __name__ == "__main__":
    unittest.main()
