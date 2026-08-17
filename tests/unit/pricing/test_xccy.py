"""Tests for Cross-Currency Swap (XCCY) pricing engines."""

import unittest

import numpy as np

from tests.helpers.assertions import assert_mc_within_bounds
from xvasim.models.fx.two_currency import TwoCurrencyFXModel
from xvasim.models.ir.hull_white import HullWhite1FModel
from xvasim.models.ir.lgm import LGMParams
from xvasim.pricing_engine import (
    FXLGMParams,
    SwapLegType,
    benchmark_price_cross_currency_swap,
    benchmark_price_xccy_swap,
    price_cross_currency_swap,
    price_xccy_swap,
)
from xvasim.qmc import RandomSequenceType


class TestCrossCurrencySwapPricing(unittest.TestCase):
    """Unit tests for Cross-Currency Swap Monte Carlo and analytical benchmark pricing."""

    def setUp(self) -> None:
        self.tenors = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0])
        self.dom_dfs = np.exp(-0.03 * self.tenors)
        self.for_dfs = np.exp(-0.01 * self.tenors)

        self.dom_model = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dom_dfs,
        )
        self.for_model = HullWhite1FModel(
            a_ann=0.04,
            sigma_ann=0.012,
            discount_curve_yrs=self.tenors,
            discount_factors=self.for_dfs,
        )
        self.corr = np.array(
            [
                [1.0, 0.3, 0.2],
                [0.3, 1.0, -0.1],
                [0.2, -0.1, 1.0],
            ]
        )
        self.fx_model = TwoCurrencyFXModel(
            domestic_ir_model=self.dom_model,
            foreign_ir_model=self.for_model,
            spot_fx=1.15,
            fx_vol_ann=0.12,
            correlation_matrix=self.corr,
        )

    def test_benchmark_xccy_pricing_fixed_float(self) -> None:
        """Verify analytical benchmark for Fixed-Float cross-currency swap."""
        bm = benchmark_price_cross_currency_swap(
            model=self.fx_model,
            tenor_yrs=5.0,
            domestic_leg_type=SwapLegType.FIXED,
            foreign_leg_type=SwapLegType.FLOATING,
            domestic_rate_ann=0.03,
            domestic_notional=115000.0,
            foreign_notional=100000.0,
            exchange_notionals=True,
        )
        self.assertIn("price", bm)

        # Alias benchmark_price_xccy_swap
        bm_alias = benchmark_price_xccy_swap(
            model=self.fx_model,
            tenor_yrs=5.0,
            domestic_leg_type="fixed",
            foreign_leg_type="floating",
            domestic_rate_ann=0.03,
            domestic_notional=115000.0,
            foreign_notional=100000.0,
            exchange_notionals=True,
        )
        self.assertEqual(bm["price"], bm_alias["price"])

    def test_mc_xccy_pricing_fixed_float(self) -> None:
        """Verify MC cross currency swap pricing matches benchmark within statistical bounds."""
        res = price_cross_currency_swap(
            model=self.fx_model,
            tenor_yrs=3.0,
            domestic_leg_type=SwapLegType.FIXED,
            foreign_leg_type=SwapLegType.FLOATING,
            domestic_rate_ann=0.03,
            domestic_notional=115000.0,
            foreign_notional=100000.0,
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

    def test_mc_xccy_pricing_float_float(self) -> None:
        """Verify Float-Float cross currency swap with foreign FX basis spread."""
        res = price_xccy_swap(
            model=self.fx_model,
            tenor_yrs=2.0,
            domestic_leg_type="floating",
            foreign_leg_type="floating",
            foreign_spread_ann=0.0015,
            domestic_notional=115000.0,
            foreign_notional=100000.0,
            n_paths=150,
            seed=123,
        )
        self.assertIn("price", res)
        self.assertIn("analytical_benchmark_price", res)
        assert_mc_within_bounds(
            res["price"],
            res["std_error"],
            res["analytical_benchmark_price"],
            num_std=3.5,
        )

    def test_legacy_fx_lgm_params(self) -> None:
        """Verify backwards compatibility with legacy FXLGMParams."""
        dom_p = LGMParams(
            0.03, np.array([30.0]), np.array([0.01]), self.tenors, self.dom_dfs
        )
        for_p = LGMParams(
            0.04, np.array([30.0]), np.array([0.012]), self.tenors, self.for_dfs
        )
        legacy_params = FXLGMParams(
            domestic=dom_p,
            foreign=for_p,
            spot_fx=1.15,
            fx_vol_ann=0.12,
            correlation_matrix=self.corr,
        )
        res = price_cross_currency_swap(
            model=legacy_params,
            tenor_yrs=2.0,
            domestic_rate_ann=0.03,
            domestic_notional=115000.0,
            foreign_notional=100000.0,
            n_paths=100,
            seed=42,
        )
        self.assertIn("price", res)


if __name__ == "__main__":
    unittest.main()
