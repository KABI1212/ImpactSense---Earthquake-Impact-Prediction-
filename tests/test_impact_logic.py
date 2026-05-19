import unittest

from impact_logic import (
    build_prediction_headline,
    categorize_depth,
    categorize_fault_proximity,
    categorize_magnitude,
    classify_risk_level,
    encode_depth_category,
    encode_fault_category,
    encode_magnitude_category,
    validate_inputs,
)


class ImpactLogicTests(unittest.TestCase):
    def test_category_encodings_match_training_order(self) -> None:
        self.assertEqual(categorize_depth(20), "Shallow")
        self.assertEqual(categorize_depth(280), "Intermediate")
        self.assertEqual(categorize_depth(500), "Deep")
        self.assertEqual(encode_depth_category("Deep"), 0)

        self.assertEqual(categorize_magnitude(3.2), "Minor")
        self.assertEqual(categorize_magnitude(4.6), "Light")
        self.assertEqual(categorize_magnitude(6.5), "Strong")
        self.assertEqual(categorize_magnitude(7.4), "Major")
        self.assertEqual(encode_magnitude_category("Major"), 1)

        self.assertEqual(categorize_fault_proximity(10), "Near")
        self.assertEqual(categorize_fault_proximity(45), "Medium")
        self.assertEqual(categorize_fault_proximity(70), "Far")
        self.assertEqual(encode_fault_category("Near"), 2)

    def test_validate_inputs_blocks_invalid_ranges(self) -> None:
        result = validate_inputs(
            {
                "magnitude": 11.2,
                "depth": -4,
                "latitude": 91,
                "longitude": 181,
                "fault_proximity": -1,
            }
        )
        self.assertGreaterEqual(len(result["errors"]), 4)

    def test_validate_inputs_warns_for_out_of_distribution_values(self) -> None:
        result = validate_inputs(
            {
                "magnitude": 2.0,
                "depth": 710.0,
                "latitude": 0.0,
                "longitude": 0.0,
                "fault_proximity": 0.0,
            },
            observed_ranges={
                "magnitude": (2.5, 9.0),
                "depth": (5.0, 700.0),
                "latitude": (-90.0, 90.0),
                "longitude": (-180.0, 180.0),
                "fault_proximity": (0.1, 100.0),
            },
        )
        self.assertEqual(result["errors"], [])
        self.assertGreaterEqual(len(result["warnings"]), 3)

    def test_classify_risk_level_uses_combined_signal(self) -> None:
        self.assertEqual(classify_risk_level(0.18, 14), "Low")
        self.assertEqual(classify_risk_level(0.36, 26), "Elevated")
        self.assertEqual(classify_risk_level(0.58, 42), "High")
        self.assertEqual(classify_risk_level(0.78, 68), "Severe")

    def test_prediction_headline_is_human_readable(self) -> None:
        headline = build_prediction_headline("High", 0.62, 57.4)
        self.assertIn("High risk outlook", headline)
        self.assertIn("62.0%", headline)
        self.assertIn("57.4/100", headline)


if __name__ == "__main__":
    unittest.main()
