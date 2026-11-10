import unittest

from scripts.decision_engine_phase7 import recommend


THRESHOLDS = {"caution": 0.25, "prepare_to_slow": 0.75}
POLICY = {
    "critical_ttc_s": 1.5,
    "high_risk_ttc_s": 3.0,
    "moderate_risk_ttc_s": 5.0,
    "short_time_headway_s": 2.0,
}


def state(ttc="", ttc_valid="0", thw="", thw_valid="0", contact="0", ego="20", lead="20", gap="30", relative="0"):
    return {
        "ttc_s": ttc,
        "ttc_valid": ttc_valid,
        "thw_s": thw,
        "thw_valid": thw_valid,
        "contact": contact,
        "ego_speed_mps": ego,
        "lead_speed_mps": lead,
        "gap_m": gap,
        "relative_speed_mps": relative,
    }


class Phase7PolicyTests(unittest.TestCase):
    def test_contact_always_gets_critical_override(self):
        result = recommend(state(contact="1"), 0.01, THRESHOLDS, POLICY)
        self.assertEqual(result["recommendation"], "critical_warning")
        self.assertEqual(result["reason_code"], "CURRENT_CONTACT")

    def test_critical_ttc_overrides_low_model_score(self):
        result = recommend(state(ttc="1.2", ttc_valid="1"), 0.01, THRESHOLDS, POLICY)
        self.assertEqual(result["recommendation"], "critical_warning")
        self.assertEqual(result["disagreement_status"], "physics_high_model_low")

    def test_high_physics_risk_prepares_to_slow_when_model_disagrees(self):
        result = recommend(state(ttc="2.5", ttc_valid="1"), 0.10, THRESHOLDS, POLICY)
        self.assertEqual(result["recommendation"], "prepare_to_slow")
        self.assertEqual(result["disagreement_status"], "physics_high_model_low")

    def test_high_model_score_with_undefined_ttc_is_logged_and_advised(self):
        result = recommend(state(), 0.9, THRESHOLDS, POLICY)
        self.assertEqual(result["recommendation"], "prepare_to_slow")
        self.assertEqual(result["disagreement_status"], "model_high_physics_low")

    def test_moderate_ttc_recommends_larger_gap(self):
        result = recommend(state(ttc="4.0", ttc_valid="1"), 0.1, THRESHOLDS, POLICY)
        self.assertEqual(result["recommendation"], "increase_following_gap")

    def test_short_headway_recommends_larger_gap(self):
        result = recommend(state(thw="1.8", thw_valid="1"), 0.1, THRESHOLDS, POLICY)
        self.assertEqual(result["recommendation"], "increase_following_gap")

    def test_invalid_required_input_falls_back_to_caution(self):
        result = recommend(state(ego="", ttc="1.0", ttc_valid="1"), 0.99, THRESHOLDS, POLICY)
        self.assertEqual(result["recommendation"], "caution")
        self.assertEqual(result["data_quality_status"], "incomplete_or_invalid")


if __name__ == "__main__":
    unittest.main()
