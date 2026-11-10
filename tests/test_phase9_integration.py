import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def dashboard_payload():
    source = (ROOT / "data/phase9/dashboard_data.js").read_text(encoding="utf-8").strip()
    prefix = "window.RAVAN_DASHBOARD_DATA = "
    if not source.startswith(prefix) or not source.endswith(";"):
        raise AssertionError("Dashboard bundle wrapper is malformed")
    return json.loads(source[len(prefix):-1])


class Phase9IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = dashboard_payload()

    def test_dashboard_contains_all_test_episodes(self):
        manifest = json.loads((ROOT / "data/phase6/split_manifest.json").read_text(encoding="utf-8"))
        actual = {episode["episodeId"] for episode in self.payload["episodes"]}
        self.assertEqual(actual, set(manifest["episode_ids"]["test"]))
        self.assertEqual(self.payload["episodeCount"], 21)

    def test_each_sample_has_the_full_review_path(self):
        for episode in self.payload["episodes"]:
            self.assertTrue(episode["samples"])
            for sample in episode["samples"]:
                self.assertIn("gapM", sample)
                self.assertIn("score", sample)
                self.assertIn("recommendation", sample)
                self.assertIn("reason", sample)
                self.assertEqual(len(sample["topContributions"]), 6)

    def test_scores_match_phase7_and_explanations_pass_fidelity_gate(self):
        self.assertLessEqual(self.payload["maxProbabilityDeltaFromPhase7"], 2e-6)
        summary = json.loads((ROOT / "data/phase8/explanation_summary.json").read_text(encoding="utf-8"))
        self.assertLessEqual(
            summary["probability_fidelity"]["maximum_reconstruction_error"],
            summary["probability_fidelity"]["configured_tolerance"],
        )

    def test_dashboard_exposes_censored_samples_separately(self):
        samples = [sample for episode in self.payload["episodes"] for sample in episode["samples"]]
        self.assertTrue(any(sample["target"] is None for sample in samples))
        self.assertTrue(any(sample["target"] in (0, 1) for sample in samples))


if __name__ == "__main__":
    unittest.main()
