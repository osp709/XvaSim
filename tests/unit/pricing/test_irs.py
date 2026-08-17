"""Tests for single-currency Interest Rate Swap (IRS) pricing."""

import unittest

import numpy as np

from tests.helpers.assertions import assert_mc_within_bounds
from xvasim.models.ir.cir import CIRInterestRateModel
from xvasim.models.ir.hull_white import HullWhite1FModel
from xvasim.models.ir.lgm import LGMModel, LGMParams
from xvasim.models.ir.vasicek import VasicekModel
from xvasim.pricing_engine import (
    SwapLegType,
    _generate_swap_schedule,
    _parse_swap_leg_type,
    benchmark_price_interest_rate_swap,
    benchmark_price_irs,
    price_interest_rate_swap,
    price_irs,
)
from xvasim.qmc import RandomSequenceType


class TestSwapHelpers(unittest.TestCase):
    """Unit tests for internal swap helper functions."""

    def test_parse_swap_leg_type(self) -> None:
        self.assertEqual(_parse_swap_leg_type(SwapLegType.FIXED), SwapLegType.FIXED)
        self.assertEqual(
            _parse_swap_leg_type(SwapLegType.FLOATING), SwapLegType.FLOATING
        )
        self.assertEqual(_parse_swap_leg_type("fixed"), SwapLegType.FIXED)
        self.assertEqual(_parse_swap_leg_type("FIXED"), SwapLegType.FIXED)
        self.assertEqual(_parse_swap_leg_type("floating"), SwapLegType.FLOATING)
        self.assertEqual(_parse_swap_leg_type("FLOATING"), SwapLegType.FLOATING)

        with self.assertRaises(ValueError):
            _parse_swap_leg_type("invalid_type")

        with self.assertRaises(TypeError):
            _parse_swap_leg_type(123)  # type: ignore[arg-type]

    def test_generate_swap_schedule_tenor(self) -> None:
        times = _generate_swap_schedule(
            tenor_yrs=2.0, payment_times_yrs=None, pay_freq_yrs=0.5
        )
        np.testing.assert_allclose(times, [0.5, 1.0, 1.5, 2.0])

    def test_generate_swap_schedule_custom(self) -> None:
        custom = [0.25, 0.75, 1.25]
        times = _generate_swap_schedule(
            tenor_yrs=None, payment_times_yrs=custom, pay_freq_yrs=0.5
        )
        np.testing.assert_allclose(times, [0.25, 0.75, 1.25])

    def test_generate_swap_schedule_errors(self) -> None:
        with self.assertRaises(ValueError):
            _generate_swap_schedule(None, None, 0.5)
        with self.assertRaises(ValueError):
            _generate_swap_schedule(-1.0, None, 0.5)
        with self.assertRaises(ValueError):
            _generate_swap_schedule(2.0, None, -0.5)
        with self.assertRaises(ValueError):
            _generate_swap_schedule(None, [], 0.5)
        with self.assertRaises(ValueError):
            _generate_swap_schedule(None, [-0.5, 1.0], 0.5)


class TestSingleCurrencySwapPricing(unittest.TestCase):
    """Unit tests for Monte Carlo and benchmark single-currency swap pricing."""

    def setUp(self) -> None:
        self.tenors = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0])
        self.dom_dfs = np.exp(-0.03 * self.tenors)

        self.lgm_params = LGMParams(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([30.0]),
            sigma_values_ann=np.array([0.008]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.dom_dfs,
        )
        self.lgm_model = LGMModel(self.lgm_params)
        self.hw_model = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dom_dfs,
        )
        self.vasicek_model = VasicekModel(
            kappa_ann=0.15,
            theta_ann=0.03,
            sigma_ann=0.015,
            r0_ann=0.03,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dom_dfs,
        )
        self.cir_model = CIRInterestRateModel(
            kappa_ann=0.2,
            theta_ann=0.03,
            sigma_ann=0.08,
            r0_ann=0.03,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dom_dfs,
        )

    def test_benchmark_swap_pricing(self) -> None:
        """Verify analytical benchmark swap pricing."""
        bm_res = benchmark_price_interest_rate_swap(
            model=self.hw_model,
            fixed_rate_ann=0.03,
            tenor_yrs=5.0,
            is_payer=True,
            notional=100000.0,
        )
        self.assertIn("price", bm_res)
        bm_price = bm_res["price"]

        # At fair swap rate, swap price is exact 0.0
        fair_rate = bm_res["fair_swap_rate"]
        bm_at_fair = benchmark_price_interest_rate_swap(
            model=self.hw_model,
            fixed_rate_ann=fair_rate,
            tenor_yrs=5.0,
            is_payer=True,
            notional=100000.0,
        )
        self.assertAlmostEqual(bm_at_fair["price"], 0.0, places=9)

        # Receiver swap (is_payer=False) is exact negative of payer swap
        bm_receiver_res = benchmark_price_irs(
            model=self.hw_model,
            fixed_rate_ann=0.03,
            tenor_yrs=5.0,
            is_payer=False,
            notional=100000.0,
        )
        self.assertAlmostEqual(bm_price, -bm_receiver_res["price"], places=10)

    def test_mc_swap_pricing_all_models(self) -> None:
        """Verify Monte Carlo simulation pricing matches benchmark across models."""
        models = [self.lgm_model, self.hw_model, self.vasicek_model, self.cir_model]
        for mdl in models:
            res = price_interest_rate_swap(
                model=mdl,
                fixed_rate_ann=0.035,
                tenor_yrs=3.0,
                is_payer=True,
                notional=10000.0,
                n_paths=200,
                random_type=RandomSequenceType.SOBOL,
                seed=42,
            )
            self.assertIn("price", res)
            self.assertIn("std_error", res)
            self.assertIn("analytical_benchmark_price", res)
            assert_mc_within_bounds(
                res["price"],
                res["std_error"],
                res["analytical_benchmark_price"],
                num_std=3.5,
            )

    def test_alias_price_irs(self) -> None:
        """Verify price_irs alias behaves identically to price_interest_rate_swap."""
        res1 = price_irs(
            model=self.hw_model,
            fixed_rate_ann=0.03,
            tenor_yrs=2.0,
            n_paths=100,
            seed=123,
        )
        res2 = price_interest_rate_swap(
            model=self.hw_model,
            fixed_rate_ann=0.03,
            tenor_yrs=2.0,
            n_paths=100,
            seed=123,
        )
        self.assertEqual(res1["price"], res2["price"])
        self.assertEqual(
            res1["analytical_benchmark_price"], res2["analytical_benchmark_price"]
        )


if __name__ == "__main__":
    unittest.main()
