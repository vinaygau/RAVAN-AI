import unittest

from scripts.preprocess_phase3 import bumper_gap, derive_features


class Phase3FeatureTests(unittest.TestCase):
    def test_bumper_gap_uses_front_position_and_lead_length(self):
        self.assertAlmostEqual(bumper_gap(35.0, 5.0, 0.0), 30.0)

    def test_ttc_thw_and_drac_hand_calculation(self):
        features = derive_features(
            ego_front_x_m=0.0,
            lead_front_x_m=35.0,
            lead_length_m=5.0,
            ego_speed_mps=20.0,
            lead_speed_mps=15.0,
        )
        self.assertAlmostEqual(features["gap_m"], 30.0)
        self.assertAlmostEqual(features["relative_speed_mps"], 5.0)
        self.assertAlmostEqual(features["ttc_s"], 6.0)
        self.assertAlmostEqual(features["thw_s"], 1.5)
        self.assertAlmostEqual(features["drac_mps2"], 25.0 / 60.0)
        self.assertEqual(features["ttc_valid"], 1)
        self.assertEqual(features["thw_valid"], 1)
        self.assertEqual(features["drac_valid"], 1)

    def test_non_closing_vehicle_has_undefined_ttc_and_drac(self):
        features = derive_features(
            ego_front_x_m=0.0,
            lead_front_x_m=25.0,
            lead_length_m=5.0,
            ego_speed_mps=12.0,
            lead_speed_mps=18.0,
        )
        self.assertIsNone(features["ttc_s"])
        self.assertIsNone(features["drac_mps2"])
        self.assertEqual(features["ttc_valid"], 0)
        self.assertEqual(features["drac_valid"], 0)
        self.assertAlmostEqual(features["thw_s"], 20.0 / 12.0)

    def test_zero_ego_speed_makes_thw_undefined(self):
        features = derive_features(
            ego_front_x_m=0.0,
            lead_front_x_m=15.0,
            lead_length_m=5.0,
            ego_speed_mps=0.0,
            lead_speed_mps=0.0,
        )
        self.assertIsNone(features["thw_s"])
        self.assertEqual(features["thw_valid"], 0)

    def test_contact_gap_disables_surrogate_indicators(self):
        features = derive_features(
            ego_front_x_m=10.0,
            lead_front_x_m=14.0,
            lead_length_m=4.0,
            ego_speed_mps=10.0,
            lead_speed_mps=5.0,
        )
        self.assertEqual(features["gap_m"], 0.0)
        self.assertIsNone(features["ttc_s"])
        self.assertIsNone(features["thw_s"])
        self.assertIsNone(features["drac_mps2"])


if __name__ == "__main__":
    unittest.main()
