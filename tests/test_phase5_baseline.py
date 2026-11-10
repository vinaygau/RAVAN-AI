import unittest

from scripts.physics_baseline_phase5 import physics_predictions


def sample(ttc="", valid=0, contact=0):
    return {"ttc_s": str(ttc) if ttc != "" else "", "ttc_valid": str(valid), "contact": str(contact)}


class Phase5BaselineTests(unittest.TestCase):
    def test_ttc_six_seconds_predicts_future_threshold_in_three_seconds(self):
        result = physics_predictions(sample("6.0", 1), horizon_s=3.0)
        self.assertEqual(result["baseline_future_conflict_pred_3s"], 1)
        self.assertEqual(result["baseline_future_contact_pred_3s"], 0)
        self.assertEqual(result["baseline_conflict_ttc_limit_s"], "6.000")

    def test_just_above_conflict_limit_is_negative(self):
        result = physics_predictions(sample("6.001", 1), horizon_s=3.0)
        self.assertEqual(result["baseline_future_conflict_pred_3s"], 0)

    def test_ttc_three_seconds_predicts_both_conflict_and_contact(self):
        result = physics_predictions(sample("3.0", 1), horizon_s=3.0)
        self.assertEqual(result["baseline_future_conflict_pred_3s"], 1)
        self.assertEqual(result["baseline_future_contact_pred_3s"], 1)
        self.assertEqual(result["baseline_current_risk_category"], "high")

    def test_undefined_ttc_does_not_create_zero_ttc_warning(self):
        result = physics_predictions(sample(), horizon_s=3.0)
        self.assertEqual(result["baseline_future_conflict_pred_3s"], 0)
        self.assertEqual(result["baseline_future_contact_pred_3s"], 0)
        self.assertEqual(result["baseline_reason_code"], "TTC_UNDEFINED_OR_NOT_CLOSING")

    def test_current_contact_future_predictions_are_blank(self):
        result = physics_predictions(sample(contact=1), horizon_s=3.0)
        self.assertEqual(result["baseline_future_conflict_pred_3s"], "")
        self.assertEqual(result["baseline_future_contact_pred_3s"], "")
        self.assertEqual(result["baseline_reason_code"], "CURRENT_CONTACT")


if __name__ == "__main__":
    unittest.main()
