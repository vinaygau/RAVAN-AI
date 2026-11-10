import unittest

from src.ravan_game.renderer import (
    CHAT_INPUT_RECT,
    CHAT_MIC_RECT,
    CHAT_SEND_RECT,
    CHAT_VOICE_RECT,
    MOUSE_ACCEL_RECT,
    MOUSE_BRAKE_RECT,
    MOUSE_LEFT_RECT,
    MOUSE_RIGHT_RECT,
    NAV_BUTTON_RECTS,
    NAV_BUTTONS,
    PAUSE_RECT,
    QUIT_RECT,
    RESTART_RECT,
    SETTINGS_RECT,
    MOUSE_WHEEL_RECT,
)


class DashboardHitboxTests(unittest.TestCase):
    def setUp(self):
        self.hitboxes = [
            *zip((f"nav:{name}" for name in NAV_BUTTONS), NAV_BUTTON_RECTS),
            ("chat_input", CHAT_INPUT_RECT),
            ("chat_mic", CHAT_MIC_RECT),
            ("chat_voice", CHAT_VOICE_RECT),
            ("chat_send", CHAT_SEND_RECT),
            ("steer_left", MOUSE_LEFT_RECT),
            ("steer_right", MOUSE_RIGHT_RECT),
            ("steering_wheel", MOUSE_WHEEL_RECT),
            ("brake", MOUSE_BRAKE_RECT),
            ("accelerate", MOUSE_ACCEL_RECT),
            ("pause", PAUSE_RECT),
            ("restart", RESTART_RECT),
            ("settings", SETTINGS_RECT),
            ("quit", QUIT_RECT),
        ]

    def test_every_button_center_hits_only_its_own_target(self):
        for name, rect in self.hitboxes:
            with self.subTest(button=name):
                self.assertGreater(rect.width, 0)
                self.assertGreater(rect.height, 0)
                center = rect.center
                matches = [other_name for other_name, other_rect in self.hitboxes if other_rect.collidepoint(center)]
                self.assertEqual(matches, [name])


if __name__ == "__main__":
    unittest.main()
