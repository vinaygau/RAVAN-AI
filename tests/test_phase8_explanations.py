import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Phase8ExplanationTests(unittest.TestCase):
    def test_test_set_contributions_reconstruct_model_probability(self):
        summary = json.loads((ROOT / "data/phase8/explanation_summary.json").read_text(encoding="utf-8"))
        fidelity = summary["probability_fidelity"]
        self.assertLessEqual(fidelity["maximum_reconstruction_error"], fidelity["configured_tolerance"])
        self.assertLessEqual(fidelity["maximum_difference_from_phase6_test_predictions"], 2e-6)
        self.assertLessEqual(fidelity["maximum_difference_from_phase7_trace"], 2e-6)

    def test_summary_contains_representative_outcomes(self):
        summary = json.loads((ROOT / "data/phase8/explanation_summary.json").read_text(encoding="utf-8"))
        cases = {case["case_type"] for case in summary["representative_examples"]}
        self.assertTrue({"true_positive", "true_negative", "false_positive", "false_negative"}.issubset(cases))


if __name__ == "__main__":
    unittest.main()
