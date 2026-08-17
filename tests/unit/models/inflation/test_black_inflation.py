"""Tests for the Black log-normal forward CPI inflation model."""

import unittest

import numpy as np

from xvasim.models.inflation.black_inflation import (
    BlackInflationModel,
)
from xvasim.qmc import RandomSequenceType


class TestBlackInflationModel(unittest.TestCase):
    """Unit tests for BlackInflationModel analytical option pricing, zero coupon swaps, and simulation."""

    def setUp(self) -> None:
        self.tenors = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        self.nom_dfs = np.exp(-0.035 * self.tenors)
        self.real_dfs = np.exp(-0.015 * self.tenors)
        self.base_cpi = 100.0
        self.vol = 0.02

        self.model = BlackInflationModel(
            nominal_discount_curve_yrs=self.tenors,
            nominal_discount_factors=self.nom_dfs,
            real_discount_curve_yrs=self.tenors,
            real_discount_factors=self.real_dfs,
            base_cpi=self.base_cpi,
            cpi_vol_ann=self.vol,
        )

    def test_init_validation(self) -> None:
        """Invalid base_cpi or cpi_vol_ann raises ValueError."""
        with self.assertRaises(ValueError):
            BlackInflationModel(
                nominal_discount_curve_yrs=self.tenors,
                nominal_discount_factors=self.nom_dfs,
                real_discount_curve_yrs=self.tenors,
                real_discount_factors=self.real_dfs,
                base_cpi=-100.0,
            )
        with self.assertRaises(ValueError):
            BlackInflationModel(
                nominal_discount_curve_yrs=self.tenors,
                nominal_discount_factors=self.nom_dfs,
                real_discount_curve_yrs=self.tenors,
                real_discount_factors=self.real_dfs,
                cpi_vol_ann=-0.01,
            )

    def test_properties(self) -> None:
        """Verify model properties."""
        self.assertEqual(self.model.model_name, "black")
        self.assertEqual(self.model.base_cpi, 100.0)
        self.assertEqual(self.model.cpi_vol_ann, 0.02)
        np.testing.assert_array_equal(
            self.model.nominal_discount_curve_yrs, self.tenors
        )
        np.testing.assert_array_equal(self.model.nominal_discount_factors, self.nom_dfs)
        np.testing.assert_array_equal(self.model.real_discount_curve_yrs, self.tenors)
        np.testing.assert_array_equal(self.model.real_discount_factors, self.real_dfs)

    def test_forward_cpi_and_swap_rate(self) -> None:
        """Verify forward CPI and zero-coupon swap rate formulas."""
        fwd_5y = self.model.forward_cpi(5.0)
        expected_fwd = 100.0 * (np.exp(-0.015 * 5.0) / np.exp(-0.035 * 5.0))
        self.assertAlmostEqual(fwd_5y, expected_fwd, places=10)

        swap_rate_5y = self.model.zero_coupon_inflation_swap_rate(5.0)
        expected_rate = (expected_fwd / 100.0) ** (1.0 / 5.0) - 1.0
        self.assertAlmostEqual(swap_rate_5y, expected_rate, places=10)

        # Non-positive maturity swap rate
        self.assertEqual(self.model.zero_coupon_inflation_swap_rate(0.0), 0.0)

    def test_price_cpi_option_analytical(self) -> None:
        """Verify Black analytical CPI caplet / floorlet pricing and put-call parity."""
        strike_rate = 0.020
        maturity = 5.0
        notional = 1000.0

        caplet = self.model.price_consumer_price_index_option_analytical(
            strike_rate_ann=strike_rate,
            maturity_yrs=maturity,
            notional=notional,
            is_call=True,
        )
        floorlet = self.model.price_consumer_price_index_option_analytical(
            strike_rate_ann=strike_rate,
            maturity_yrs=maturity,
            notional=notional,
            is_call=False,
        )
        self.assertGreater(caplet, 0.0)
        self.assertGreater(floorlet, 0.0)

        # Zero vol limit
        zero_vol_model = BlackInflationModel(
            nominal_discount_curve_yrs=self.tenors,
            nominal_discount_factors=self.nom_dfs,
            real_discount_curve_yrs=self.tenors,
            real_discount_factors=self.real_dfs,
            base_cpi=100.0,
            cpi_vol_ann=0.0,
        )
        caplet_zero = zero_vol_model.price_consumer_price_index_option_analytical(
            strike_rate_ann=0.015,
            maturity_yrs=5.0,
            is_call=True,
        )
        floorlet_zero = zero_vol_model.price_consumer_price_index_option_analytical(
            strike_rate_ann=0.025,
            maturity_yrs=5.0,
            is_call=False,
        )
        self.assertGreater(caplet_zero, 0.0)
        self.assertGreater(floorlet_zero, 0.0)

        # Non-positive maturity option
        caplet_expired = self.model.price_consumer_price_index_option_analytical(
            strike_rate_ann=0.02,
            maturity_yrs=0.0,
            is_call=True,
        )
        self.assertEqual(caplet_expired, 0.0)

    def test_simulate_paths(self) -> None:
        """Verify Black inflation path simulation results."""
        res = self.model.simulate_paths(
            maturity_yrs=2.0,
            n_paths=50,
            n_steps=4,
            random_type=RandomSequenceType.PSEUDO,
            seed=42,
        )
        self.assertEqual(len(res.times), 5)
        self.assertEqual(res.cpi_index.shape, (50, 5))
        np.testing.assert_allclose(res.cpi_index[:, 0], 100.0)
        self.assertTrue(np.all(res.cpi_index > 0.0))


if __name__ == "__main__":
    unittest.main()
