import unittest

from app import ENGINEERED_FEATURE_COLUMNS, build_engineered_feature_row, predict_impact


class PredictionPathSmokeTests(unittest.TestCase):
    def test_real_prediction_path_runs_for_sample_inputs(self) -> None:
        samples = [
            {
                "magnitude": 5.5,
                "depth": 35.0,
                "latitude": 28.6139,
                "longitude": 77.2090,
                "fault_proximity": 24.0,
            },
            {
                "magnitude": 7.4,
                "depth": 18.0,
                "latitude": 34.05,
                "longitude": -118.24,
                "fault_proximity": 8.0,
            },
            {
                "magnitude": 6.2,
                "depth": 315.0,
                "latitude": -12.4,
                "longitude": 130.85,
                "fault_proximity": 42.0,
            },
        ]

        for sample in samples:
            with self.subTest(sample=sample):
                prediction = predict_impact(**sample)
                self.assertIn(prediction["impact_band"], {"Low", "Elevated", "High", "Severe"})
                self.assertGreaterEqual(prediction["impact_score"], 0)
                self.assertLessEqual(prediction["impact_score"], 100)
                self.assertGreaterEqual(prediction["high_impact_probability"], 0.0)
                self.assertLessEqual(prediction["high_impact_probability"], 1.0)
                self.assertTrue(prediction["summary"])
                self.assertTrue(prediction["reasons"])

    def test_feature_row_uses_training_feature_order(self) -> None:
        row = build_engineered_feature_row(
            {
                "magnitude": 6.1,
                "depth": 42.0,
                "latitude": 18.52,
                "longitude": 73.86,
                "fault_proximity": 24.0,
            }
        )

        self.assertEqual(list(row.columns), ENGINEERED_FEATURE_COLUMNS)
        self.assertEqual(row.shape, (1, len(ENGINEERED_FEATURE_COLUMNS)))


if __name__ == "__main__":
    unittest.main()
