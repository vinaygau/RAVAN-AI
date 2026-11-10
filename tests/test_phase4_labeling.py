import unittest

from scripts.label_phase4 import current_risk_category, label_episode


def row(t, ttc="", valid=0, contact=0):
    return {
        "episode_id": "E1",
        "scenario_id": "S1",
        "time_s": str(t),
        "ttc_s": str(ttc) if ttc != "" else "",
        "ttc_valid": str(valid),
        "contact": str(contact),
    }


class Phase4LabelTests(unittest.TestCase):
    def test_current_risk_band_boundaries(self):
        self.assertEqual(current_risk_category(row(0, "5.1", 1)), "low")
        self.assertEqual(current_risk_category(row(0, "5.0", 1)), "moderate")
        self.assertEqual(current_risk_category(row(0, "3.0", 1)), "high")
        self.assertEqual(current_risk_category(row(0, "1.5", 1)), "critical")
        self.assertEqual(current_risk_category(row(0)), "low")
        self.assertEqual(current_risk_category(row(0, contact=1)), "critical")

    def test_future_window_is_open_at_now_and_closed_at_horizon(self):
        rows = [row(0.0, "2.5", 1), row(0.1, "4.0", 1), row(3.0, "4.0", 1)]
        labels = label_episode(rows, 3.0)
        self.assertEqual(labels[0]["future_conflict_within_3s"], 0)
        self.assertEqual(labels[0]["future_conflict_eligible"], 1)

    def test_future_ttc_threshold_crossing_is_positive(self):
        rows = [row(0.0, "4.0", 1), row(1.0, "3.0", 1), row(3.0, "4.0", 1)]
        labels = label_episode(rows, 3.0)
        self.assertEqual(labels[0]["future_conflict_within_3s"], 1)
        self.assertEqual(labels[0]["future_contact_within_3s"], 0)

    def test_contact_within_window_is_positive_even_if_horizon_is_censored(self):
        rows = [row(0.0), row(0.5, contact=1)]
        labels = label_episode(rows, 3.0)
        self.assertEqual(labels[0]["future_conflict_within_3s"], 1)
        self.assertEqual(labels[0]["future_contact_within_3s"], 1)
        self.assertEqual(labels[0]["future_conflict_eligible"], 1)

    def test_incomplete_negative_horizon_is_censored(self):
        rows = [row(0.0), row(0.5)]
        labels = label_episode(rows, 3.0)
        self.assertEqual(labels[0]["future_conflict_within_3s"], "")
        self.assertEqual(labels[0]["future_conflict_eligible"], 0)

    def test_contact_at_current_time_is_not_a_future_event(self):
        rows = [row(0.0, contact=1)]
        labels = label_episode(rows, 3.0)
        self.assertEqual(labels[0]["current_risk_category"], "critical")
        self.assertEqual(labels[0]["future_contact_within_3s"], "")
        self.assertEqual(labels[0]["future_contact_eligible"], 0)


if __name__ == "__main__":
    unittest.main()
