import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.ravan_game.learning import TrialErrorLearner


def assessment(score, ttc=2.0, level="HIGH", collision=False):
    return SimpleNamespace(
        collision=collision,
        risk_level=level,
        risk_score=score,
        nearest_front={"ttc_s": ttc},
        nearest_rear=None,
        relative_speed_mps=4.0,
        ttc_s=ttc,
        lane_change_left=SimpleNamespace(safe=True),
        lane_change_right=SimpleNamespace(safe=False),
    )


class TrialErrorLearnerTests(unittest.TestCase):
    def test_sustained_braking_outcome_is_learned_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial_error.json"
            learner = TrialErrorLearner(path)
            learner.observe(0.0, assessment(76), "brake")

            self.assertTrue(learner.observe(2.0, assessment(48, level="MODERATE"), "brake"))
            outcomes = learner.outcomes("closing_on_front")
            self.assertEqual(outcomes["brake"]["count"], 1)
            self.assertEqual(outcomes["brake"]["improved"], 1)
            self.assertTrue(path.exists())

            restored = TrialErrorLearner(path)
            self.assertEqual(restored.outcomes("closing_on_front"), outcomes)

    def test_short_control_tap_is_evaluated_after_the_following_risk_window(self):
        with tempfile.TemporaryDirectory() as directory:
            learner = TrialErrorLearner(Path(directory) / "trial_error.json")
            learner.observe(0.0, assessment(76), "brake")
            learner.observe(0.3, assessment(76), "brake")
            learner.observe(0.35, assessment(76), None)

            self.assertFalse(learner.observe(2.0, assessment(48, level="MODERATE"), None))
            self.assertTrue(learner.observe(2.4, assessment(48, level="MODERATE"), None))
            self.assertEqual(learner.outcomes("closing_on_front")["brake"]["improved"], 1)


if __name__ == "__main__":
    unittest.main()
