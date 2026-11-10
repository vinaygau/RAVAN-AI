import unittest

from src.ravan_game.voice import VoiceAssistant
from src.ravan_game.voice_commands import parse_driving_command


class VoiceAssistantTests(unittest.TestCase):
    def test_voice_can_be_muted_and_reenabled(self):
        voice = VoiceAssistant()

        self.assertTrue(voice.enabled)
        self.assertFalse(voice.toggle_voice())
        self.assertFalse(voice.enabled)
        self.assertTrue(voice.toggle_voice())
        self.assertTrue(voice.enabled)

    def test_push_to_talk_only_starts_once(self):
        voice = VoiceAssistant()
        with voice._lock:
            voice._listening = True

        self.assertFalse(voice.start_listening())

    def test_common_driving_phrases_are_parsed_as_local_commands(self):
        cases = {
            "please accelerate": "accelerate",
            "brake now": "brake",
            "change lane to the left": "steer_left",
            "turn right": "steer_right",
            "pause the simulation": "pause",
            "continue driving": "resume",
            "restart the game": "restart",
            "use the handbrake": "handbrake",
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(parse_driving_command(phrase).intent, expected)

    def test_questions_remain_normal_ai_conversation(self):
        for phrase in (
            "What is my current speed?",
            "Is it safe to turn left?",
            "Why is the collision risk high?",
            "Tell me how fast the car behind is going.",
        ):
            with self.subTest(phrase=phrase):
                self.assertIsNone(parse_driving_command(phrase))


if __name__ == "__main__":
    unittest.main()
