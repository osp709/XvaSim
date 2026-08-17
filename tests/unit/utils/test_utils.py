"""Tests for date conversion and utility functions."""

import datetime
import unittest

from xvasim.utils import dates_to_years


class TestDatesToYears(unittest.TestCase):
    """Unit tests for dates_to_years conversion."""

    def test_strings(self) -> None:
        val_date = "2026-01-01"
        dates = ["2026-01-01", "2026-07-02", "2027-01-01"]
        years = dates_to_years(dates, val_date)
        self.assertEqual(len(years), 3)
        self.assertAlmostEqual(float(years[0]), 0.0, places=5)
        self.assertAlmostEqual(float(years[2]), 365.0 / 365.25, places=5)

    def test_datetime_objects(self) -> None:
        val_date = datetime.date(2026, 1, 1)
        dates = [
            datetime.date(2026, 1, 1),
            datetime.datetime(2027, 1, 1, 12, 0, 0),
        ]
        years = dates_to_years(dates, val_date)
        self.assertEqual(len(years), 2)
        self.assertAlmostEqual(float(years[0]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
