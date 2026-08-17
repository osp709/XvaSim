"""Tests for Inflation Derivative pricing engines."""

import unittest

import numpy as np

from tests.helpers.assertions import assert_mc_within_bounds
from xvasim.models.inflation.black_inflation import BlackInflationModel
from xvasim.models.inflation.jarrow_yildirim import JarrowYildirimModel
from xvasim.models.ir.hull_white import HullWhite1FModel
from xvasim.pricing_engine import (
    OptionType,
    benchmark_price_consumer_price_index_option,
    benchmark_price_cpi_option,
    benchmark_price_zero_coupon_inflation_swap,
    price_consumer_price_index_option,
    price_cpi_option,
    price_year_on_year_inflation_swap,
    price_yoy_inflation_swap,
    price_zero_coupon_inflation_swap,
)
from xvasim.qmc import RandomSequenceType


class TestInflationPricing(unittest.TestCase):
    """Unit tests for Zero-Coupon, YoY Inflation Swaps, and CPI Options."""

    def setUp(self) -> None:
        self.tenors = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        self.nom_dfs = np.exp(-0.035 * self.tenors)
        self.real_dfs = np.exp(-0.015 * self.tenors)

        self.black_model = BlackInflationModel(
            nominal_discount_curve_yrs=self.tenors,
            nominal_discount_factors=self.nom_dfs,
            real_discount_curve_yrs=self.tenors,
            real_discount_factors=self.real_dfs,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
        )

        nom_hw = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.nom_dfs,
        )
        real_hw = HullWhite1FModel(
            a_ann=0.02,
            sigma_ann=0.008,
            discount_curve_yrs=self.tenors,
            discount_factors=self.real_dfs,
        )
        corr = np.array(
            [
                [1.0, 0.4, 0.2],
                [0.4, 1.0, -0.1],
                [0.2, -0.1, 1.0],
            ]
        )
        self.jy_model = JarrowYildirimModel(
            nominal_ir_model=nom_hw,
            real_ir_model=real_hw,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
            correlation_matrix=corr,
        )

    def test_zero_coupon_swap_pricing(self) -> None:
        """Verify MC and benchmark pricing of zero-coupon inflation swaps."""
        bm = benchmark_price_zero_coupon_inflation_swap(
            model=self.black_model,
            strike_rate_ann=0.02,
            maturity_yrs=5.0,
            notional=1000.0,
        )
        self.assertIn("price", bm)

        res = price_zero_coupon_inflation_swap(
            model=self.black_model,
            strike_rate_ann=0.02,
            maturity_yrs=5.0,
            notional=1000.0,
            n_paths=300,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        self.assertIn("price", res)
        self.assertIn("analytical_benchmark_price", res)
        assert_mc_within_bounds(
            res["price"],
            res["std_error"],
            res["analytical_benchmark_price"],
            num_std=3.5,
        )

    def test_cpi_option_pricing(self) -> None:
        """Verify MC and benchmark pricing of CPI caplets and floorlets."""
        bm_caplet = benchmark_price_consumer_price_index_option(
            model=self.black_model,
            strike_rate_ann=0.020,
            maturity_yrs=5.0,
            option_type=OptionType.CALL,
            notional=1000.0,
        )
        self.assertGreater(bm_caplet["price"], 0.0)

        bm_alias = benchmark_price_cpi_option(
            model=self.black_model,
            strike_rate_ann=0.020,
            maturity_yrs=5.0,
            option_type="call",
            notional=1000.0,
        )
        self.assertEqual(bm_caplet["price"], bm_alias["price"])

        res = price_consumer_price_index_option(
            model=self.black_model,
            strike_rate_ann=0.020,
            maturity_yrs=5.0,
            option_type=OptionType.CALL,
            notional=1000.0,
            n_paths=300,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        assert_mc_within_bounds(
            res["price"],
            res["std_error"],
            res["analytical_benchmark_price"],
            num_std=3.5,
        )

        # Alias price_cpi_option
        res_alias = price_cpi_option(
            model=self.black_model,
            strike_rate_ann=0.020,
            maturity_yrs=5.0,
            option_type="put",
            notional=1000.0,
            n_paths=300,
            seed=42,
        )
        self.assertIn("price", res_alias)

    def test_yoy_inflation_swap_pricing(self) -> None:
        """Verify Year-on-Year (YoY) inflation swap pricing."""
        res = price_year_on_year_inflation_swap(
            model=self.black_model,
            fixed_rate_ann=0.02,
            payment_times_yrs=[1.0, 2.0, 3.0],
            notional=1000.0,
            n_paths=200,
            seed=42,
        )
        self.assertIn("price", res)

        res_alias = price_yoy_inflation_swap(
            model=self.black_model,
            fixed_rate_ann=0.02,
            payment_times_yrs=[1.0, 2.0, 3.0],
            notional=1000.0,
            n_paths=200,
            seed=42,
        )
        self.assertEqual(res["price"], res_alias["price"])


if __name__ == "__main__":
    unittest.main()
