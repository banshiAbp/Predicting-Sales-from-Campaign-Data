import unittest

import numpy as np
import pandas as pd

from case_study_pipeline import CampaignCleaner, parse_messy_number


class ParserTests(unittest.TestCase):
    def test_supported_formats(self):
        cases = [
            ("3,2%", 3.2),
            ("(GBP 500)", -500.0),
            ("1.234,56", 1234.56),
            ("1,234.56", 1234.56),
            ("GBP 5,000", 5000.0),
            ("125 k", 125000.0),
        ]

        for raw_value, expected in cases:
            with self.subTest(raw_value=raw_value):
                self.assertAlmostEqual(parse_messy_number(raw_value), expected)

    def test_invalid_values_become_missing(self):
        for raw_value in ["1.2.3", "unexpected", np.inf]:
            with self.subTest(raw_value=raw_value):
                self.assertTrue(np.isnan(parse_messy_number(raw_value)))

    def test_follower_scaling_is_suffix_aware(self):
        raw = pd.DataFrame(
            {
                "Followers": ["0.5k", "500"],
                "EngagementRate (%)": [2.0, 2.0],
                "AdSpend (GBP)": [1000, 1000],
                "ContentQuality": [7, 7],
            }
        )

        cleaned = CampaignCleaner()._basic_clean(raw)

        self.assertEqual(cleaned["followers"].tolist(), [500.0, 500_000.0])


if __name__ == "__main__":
    unittest.main()
