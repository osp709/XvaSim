import datetime
import unittest

import numpy as np

from xvasim.utils import dates_to_years


class TestUtils(unittest.TestCase):
    def test_dates_to_years_basic(self) -> None:
        valuation_date = "2026-07-11"
        dates = ["2027-07-11", "2028-07-11"]
        tenors_yrs = dates_to_years(dates, valuation_date)
        # Expected calculation:
        # "2027-07-11" - "2026-07-11" = 365 days -> 365 / 365.25 = 0.9993155373...
        # "2028-07-11" - "2026-07-11" = 731 days -> 731 / 365.25 = 2.0013689253...
        expected = np.array([365.0 / 365.25, 731.0 / 365.25])
        np.testing.assert_allclose(tenors_yrs, expected, rtol=1e-7)

    def test_dates_to_years_with_datetime_objects(self) -> None:
        valuation_date = datetime.date(2026, 7, 11)
        dates = [datetime.date(2027, 7, 11), datetime.date(2028, 7, 11)]
        tenors_yrs = dates_to_years(dates, valuation_date)
        expected = np.array([365.0 / 365.25, 731.0 / 365.25])
        np.testing.assert_allclose(tenors_yrs, expected, rtol=1e-7)
