"""Tests for FX Forwards and Options pricing engines."""

import unittest

import numpy as np

from tests.helpers.assertions import assert_mc_within_bounds
from xvasim.models.fx.garman_kohlhagen import GarmanKohlhagenFXModel
from xvasim.models.fx.heston import HestonFXModel
from xvasim.models.fx.two_currency import TwoCurrencyFXModel
from xvasim.models.ir.hull_white import HullWhite1FModel
from xvasim.pricing_engine import (
    OptionType,
    benchmark_price_foreign_exchange_forward,
    benchmark_price_foreign_exchange_option,
    benchmark_price_fx_forward,
    benchmark_price_fx_option,
    price_foreign_exchange_forward,
    price_foreign_exchange_option,
    price_fx_forward,
    price_fx_option,
)
from xvasim.qmc import RandomSequenceType


class TestFXPricing(unittest.TestCase):
    """Unit tests for FX Forward and FX Option pricing engines."""

    def setUp(self) -> None:
        self.tenors = np.array([0.0, 1.0, 5.0])
        self.dfs_dom = np.exp(-0.03 * self.tenors)
        self.dfs_for = np.exp(-0.015 * self.tenors)

        self.gk_model = GarmanKohlhagenFXModel(
            spot_fx=1.15,
            fx_vol_ann=0.12,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.015,
        )

        dom_hw = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs_dom,
        )
        for_hw = HullWhite1FModel(
            a_ann=0.04,
            sigma_ann=0.012,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs_for,
        )
        corr = np.array([
            [1.0, 0.3, 0.2],
            [0.3, 1.0, -0.1],
            [0.2, -0.1, 1.0],
        ])
        self.two_curr_model = TwoCurrencyFXModel(
            domestic_ir_model=dom_hw,
            foreign_ir_model=for_hw,
            spot_fx=1.15,
            fx_vol_ann=0.12,
            correlation_matrix=corr,
        )

        self.heston_model = HestonFXModel(
            spot_fx=1.15,
            v_0=0.0144,
            kappa_ann=2.0,
            theta_ann=0.0144,
            sigma_v_ann=0.2,
            rho=-0.4,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.015,
        )

    def test_fx_forward_pricing(self) -> None:
        """Verify MC FX forward pricing matches benchmark."""
        res = price_foreign_exchange_forward(
            params=self.gk_model,
            strike=1.15,
            maturity_yrs=1.0,
            notional=1000.0,
            n_paths=200,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        self.assertIn("price", res)
        self.assertIn("analytical_benchmark_price", res)
        assert_mc_within_bounds(
            float(res["price"]),
            float(res["std_error"]),
            float(res["analytical_benchmark_price"]),
            num_std=3.5,
        )

        # Alias price_fx_forward
        res_alias = price_fx_forward(
            params=self.two_curr_model,
            strike=1.15,
            maturity_yrs=1.0,
            notional=1000.0,
            n_paths=200,
            seed=42,
        )
        self.assertIn("price", res_alias)

    def test_benchmark_fx_forward(self) -> None:
        """Verify benchmark_price_fx_forward calculation."""
        bm = benchmark_price_foreign_exchange_forward(
            params=self.gk_model,
            strike=1.15,
            maturity_yrs=1.0,
            notional=1000.0,
        )
        self.assertIn("price", bm)
        bm_alias = benchmark_price_fx_forward(
            params=self.gk_model,
            strike=1.15,
            maturity_yrs=1.0,
            notional=1000.0,
        )
        self.assertEqual(bm["price"], bm_alias["price"])

    def test_fx_option_pricing_gk_call_and_put(self) -> None:
        """Verify MC FX option pricing for call and put under Garman-Kohlhagen."""
        res_call = price_foreign_exchange_option(
            params=self.gk_model,
            strike=1.16,
            maturity_yrs=1.0,
            notional=1000.0,
            option_type=OptionType.CALL,
            n_paths=300,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        assert_mc_within_bounds(
            float(res_call["price"]),
            float(res_call["std_error"]),
            float(res_call["analytical_benchmark_price"]),
            num_std=3.5,
        )

        res_put = price_fx_option(
            params=self.gk_model,
            strike=1.14,
            maturity_yrs=1.0,
            notional=1000.0,
            option_type=OptionType.PUT,
            n_paths=300,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        assert_mc_within_bounds(
            float(res_put["price"]),
            float(res_put["std_error"]),
            float(res_put["analytical_benchmark_price"]),
            num_std=3.5,
        )

    def test_benchmark_fx_option(self) -> None:
        """Verify benchmark_price_fx_option analytical formula."""
        bm_call = benchmark_price_foreign_exchange_option(
            params=self.gk_model,
            strike=1.16,
            maturity_yrs=1.0,
            notional=1000.0,
            option_type="call",
        )
        self.assertGreater(bm_call["price"], 0.0)

        bm_alias = benchmark_price_fx_option(
            params=self.gk_model,
            strike=1.16,
            maturity_yrs=1.0,
            notional=1000.0,
            option_type="call",
        )
        self.assertEqual(bm_call["price"], bm_alias["price"])


if __name__ == "__main__":
    unittest.main()
