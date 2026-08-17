"""Tests for internal pricing engine helper routines, validation, and error branches."""

import typing
import unittest

import numpy as np

from xvasim.models.base import FXModel
from xvasim.models.fx.garman_kohlhagen import GarmanKohlhagenFXModel
from xvasim.models.fx.two_currency import TwoCurrencyFXModel
from xvasim.models.inflation.black_inflation import BlackInflationModel
from xvasim.models.inflation.jarrow_yildirim import JarrowYildirimModel
from xvasim.models.ir.hull_white import HullWhite1FModel
from xvasim.models.ir.lgm import LGMParams
from xvasim.pricing_engine import (
    FXLGMParams,
    OptionType,
    SwapLegType,
    _compute_h_function,
    _compute_zeta,
    _discount_path,
    _instantaneous_forward,
    benchmark_price_consumer_price_index_option,
    benchmark_price_foreign_exchange_forward,
    benchmark_price_foreign_exchange_option,
    benchmark_price_interest_rate_swap,
    benchmark_price_zero_coupon_inflation_swap,
    price_consumer_price_index_option,
    price_cross_currency_swap,
    price_foreign_exchange_forward,
    price_foreign_exchange_option,
    price_interest_rate_swap,
    price_year_on_year_inflation_swap,
    price_zero_coupon_inflation_swap,
)


class TestPricingEngineInternals(unittest.TestCase):
    """Unit tests for pricing engine edge cases, type checks, and internal calculations."""

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
        self.corr = np.array(
            [
                [1.0, 0.3, 0.2],
                [0.3, 1.0, -0.1],
                [0.2, -0.1, 1.0],
            ]
        )
        self.two_curr_model = TwoCurrencyFXModel(
            domestic_ir_model=dom_hw,
            foreign_ir_model=for_hw,
            spot_fx=1.15,
            fx_vol_ann=0.12,
            correlation_matrix=self.corr,
        )

        self.dom_lgm_params = LGMParams(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([5.0]),
            sigma_values_ann=np.array([0.01]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs_dom,
        )
        self.for_lgm_params = LGMParams(
            kappa_ann=0.04,
            sigma_grid_yrs=np.array([5.0]),
            sigma_values_ann=np.array([0.012]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs_for,
        )
        self.legacy_fx_params = FXLGMParams(
            domestic=self.dom_lgm_params,
            foreign=self.for_lgm_params,
            spot_fx=1.15,
            fx_vol_ann=0.12,
            correlation_matrix=self.corr,
        )

    def test_h_and_zeta_helpers(self) -> None:
        """Verify _compute_h_function, _compute_zeta, _instantaneous_forward, and _discount_path."""
        h_0 = _compute_h_function(0.0, 0.03)
        self.assertEqual(float(h_0), 0.0)

        h_zero_k = _compute_h_function(np.array([1.0, 2.0]), 0.0)
        np.testing.assert_allclose(h_zero_k, [1.0, 2.0])

        grid = np.array([1.0, 5.0])
        vals = np.array([0.01, 0.012])
        self.assertEqual(_compute_zeta(0.0, grid, vals, 0.03), 0.0)
        self.assertEqual(_compute_zeta(-1.0, grid, vals, 0.03), 0.0)

        # Instantaneous forward helper
        fwd = _instantaneous_forward(1.0, self.tenors, self.dfs_dom)
        self.assertAlmostEqual(fwd, 0.03, places=3)

        # Discount path helper
        times = np.array([0.0, 0.5, 1.0])
        x_state = np.zeros((10, 3))
        df_paths = _discount_path(self.dom_lgm_params, x_state, times)
        self.assertEqual(df_paths.shape, (10, 3))
        np.testing.assert_allclose(df_paths[:, 0], 1.0)

    def test_legacy_fx_lgm_forward_and_option_pricing(self) -> None:
        """Verify pricing with legacy FXLGMParams for forwards and options."""
        # Forward benchmark and MC
        bm_fwd = benchmark_price_foreign_exchange_forward(
            self.legacy_fx_params, strike=1.15, maturity_yrs=1.0, notional=1000.0
        )
        self.assertIn("price", bm_fwd)

        mc_fwd = price_foreign_exchange_forward(
            self.legacy_fx_params,
            strike=1.15,
            maturity_yrs=1.0,
            notional=1000.0,
            n_paths=50,
            seed=42,
        )
        self.assertIn("price", mc_fwd)

        # Option benchmark and MC Call
        bm_opt_call = benchmark_price_foreign_exchange_option(
            self.legacy_fx_params,
            strike=1.15,
            maturity_yrs=1.0,
            notional=1000.0,
            option_type=OptionType.CALL,
        )
        self.assertGreater(bm_opt_call["price"], 0.0)

        mc_opt_call = price_foreign_exchange_option(
            self.legacy_fx_params,
            strike=1.15,
            maturity_yrs=1.0,
            notional=1000.0,
            option_type=OptionType.CALL,
            n_paths=50,
            seed=42,
        )
        self.assertIn("price", mc_opt_call)

        # Option benchmark and MC Put
        bm_opt_put = benchmark_price_foreign_exchange_option(
            self.legacy_fx_params,
            strike=1.15,
            maturity_yrs=1.0,
            notional=1000.0,
            option_type=OptionType.PUT,
        )
        self.assertGreater(bm_opt_put["price"], 0.0)

        mc_opt_put = price_foreign_exchange_option(
            self.legacy_fx_params,
            strike=1.15,
            maturity_yrs=1.0,
            notional=1000.0,
            option_type=OptionType.PUT,
            n_paths=50,
            seed=42,
        )
        self.assertIn("price", mc_opt_put)

    def test_zero_vol_intrinsic_benchmark_options(self) -> None:
        """Verify small vol intrinsic option pricing branches."""
        zero_vol_params = FXLGMParams(
            domestic=self.dom_lgm_params,
            foreign=self.for_lgm_params,
            spot_fx=1.15,
            fx_vol_ann=0.0,
            correlation_matrix=self.corr,
        )
        res_call = benchmark_price_foreign_exchange_option(
            zero_vol_params,
            strike=1.10,
            maturity_yrs=1.0,
            notional=1000.0,
            option_type=OptionType.CALL,
        )
        self.assertGreater(res_call["price"], 0.0)

        res_put = benchmark_price_foreign_exchange_option(
            zero_vol_params,
            strike=1.20,
            maturity_yrs=1.0,
            notional=1000.0,
            option_type=OptionType.PUT,
        )
        self.assertGreater(res_put["price"], 0.0)

        # TwoCurrencyFXModel with zero vol
        zero_vol_model = TwoCurrencyFXModel(
            domestic_ir_model=self.two_curr_model.domestic_ir_model,
            foreign_ir_model=self.two_curr_model.foreign_ir_model,
            spot_fx=1.15,
            fx_vol_ann=0.0,
            correlation_matrix=self.corr,
        )
        res_call_2 = benchmark_price_foreign_exchange_option(
            zero_vol_model,
            strike=1.10,
            maturity_yrs=1.0,
            notional=1000.0,
            option_type="call",
        )
        self.assertGreater(res_call_2["price"], 0.0)

        res_put_2 = benchmark_price_foreign_exchange_option(
            zero_vol_model,
            strike=1.20,
            maturity_yrs=1.0,
            notional=1000.0,
            option_type="put",
        )
        self.assertGreater(res_put_2["price"], 0.0)

    def test_fx_forward_type_errors(self) -> None:
        """Invalid model type passed to FX forward pricer raises TypeError."""
        with self.assertRaises(TypeError):
            price_foreign_exchange_forward("invalid_model", 1.15, 1.0, notional=1000.0)  # type: ignore
        with self.assertRaises(TypeError):
            benchmark_price_foreign_exchange_forward("invalid_model", 1.15, 1.0, notional=1000.0)  # type: ignore

    def test_fx_option_option_type_validation(self) -> None:
        """Invalid option_type raises ValueError or TypeError."""
        with self.assertRaises(ValueError):
            benchmark_price_foreign_exchange_option(
                self.gk_model, 1.15, 1.0, notional=1.0, option_type="invalid"
            )
        with self.assertRaises(TypeError):
            benchmark_price_foreign_exchange_option(
                self.gk_model, 1.15, 1.0, notional=1.0, option_type=12345  # type: ignore
            )
        with self.assertRaises(TypeError):
            price_foreign_exchange_option("invalid_model", 1.15, 1.0, notional=1.0)  # type: ignore
        with self.assertRaises(ValueError):
            price_foreign_exchange_option(
                self.gk_model, 1.15, 1.0, notional=1.0, option_type="invalid"
            )
        with self.assertRaises(TypeError):
            price_foreign_exchange_option(
                self.gk_model, 1.15, 1.0, notional=1.0, option_type=12345  # type: ignore
            )

    def test_inflation_pricer_type_validation(self) -> None:
        """Invalid model type passed to inflation pricers raises TypeError."""
        with self.assertRaises(TypeError):
            benchmark_price_zero_coupon_inflation_swap("invalid_model", 0.02, 5.0)  # type: ignore
        with self.assertRaises(TypeError):
            benchmark_price_consumer_price_index_option("invalid_model", 0.02, 5.0)  # type: ignore
        with self.assertRaises(TypeError):
            price_zero_coupon_inflation_swap("invalid_model", 0.02, 5.0)  # type: ignore
        with self.assertRaises(TypeError):
            price_consumer_price_index_option("invalid_model", 0.02, 5.0)  # type: ignore
        with self.assertRaises(TypeError):
            price_year_on_year_inflation_swap(
                "invalid_model", 0.02, payment_times_yrs=[1.0, 2.0]  # type: ignore
            )

    def test_inflation_pricer_n_paths_none_and_empty_schedules(self) -> None:
        """Verify n_paths=None returns analytical benchmark and empty payment schedules."""
        black = BlackInflationModel(
            nominal_discount_curve_yrs=self.tenors,
            nominal_discount_factors=self.dfs_dom,
            real_discount_curve_yrs=self.tenors,
            real_discount_factors=self.dfs_for,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
        )
        res_zcis = price_zero_coupon_inflation_swap(
            black, strike_rate_ann=0.02, maturity_yrs=5.0, n_paths=None
        )
        self.assertIn("price", res_zcis)

        res_cpi = price_consumer_price_index_option(
            black, strike_rate_ann=0.02, maturity_yrs=5.0, n_paths=None
        )
        self.assertIn("price", res_cpi)

        res_yoy_empty = price_year_on_year_inflation_swap(
            black, fixed_rate_ann=0.02, payment_times_yrs=[]
        )
        self.assertEqual(res_yoy_empty["price"], 0.0)

    def test_jarrow_yildirim_cpi_option_benchmark(self) -> None:
        """Verify analytical CPI option benchmark for JarrowYildirimModel."""
        nom_hw = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs_dom,
        )
        real_hw = HullWhite1FModel(
            a_ann=0.02,
            sigma_ann=0.008,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs_for,
        )
        corr = np.array(
            [
                [1.0, 0.4, 0.2],
                [0.4, 1.0, -0.1],
                [0.2, -0.1, 1.0],
            ]
        )
        jy_model = JarrowYildirimModel(
            nominal_ir_model=nom_hw,
            real_ir_model=real_hw,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
            correlation_matrix=corr,
        )

        res_call = benchmark_price_consumer_price_index_option(
            model=jy_model,
            strike_rate_ann=0.02,
            maturity_yrs=5.0,
            option_type=OptionType.CALL,
            notional=1000.0,
        )
        self.assertIn("price", res_call)
        self.assertGreater(res_call["price"], 0.0)

        res_put = benchmark_price_consumer_price_index_option(
            model=jy_model,
            strike_rate_ann=0.02,
            maturity_yrs=5.0,
            option_type="put",
            notional=1000.0,
        )
        self.assertIn("price", res_put)
        self.assertGreater(res_put["price"], 0.0)

    def test_irs_with_lgm_params(self) -> None:
        """Verify IRS pricing with LGMParams dataclass directly."""
        bm = benchmark_price_interest_rate_swap(
            self.dom_lgm_params, fixed_rate_ann=0.03, tenor_yrs=3.0
        )
        self.assertIn("price", bm)

        mc = price_interest_rate_swap(
            self.dom_lgm_params, fixed_rate_ann=0.03, tenor_yrs=3.0, n_paths=50, seed=42
        )
        self.assertIn("price", mc)

    def test_xccy_fixed_fixed_branches(self) -> None:
        """Verify Fixed-Fixed XCCY and fair rate solver."""
        res_payer = price_cross_currency_swap(
            model=self.two_curr_model,
            tenor_yrs=3.0,
            domestic_leg_type=SwapLegType.FIXED,
            foreign_leg_type=SwapLegType.FIXED,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.02,
            is_domestic_payer=True,
            exchange_notionals=True,
            n_paths=50,
            seed=42,
        )
        self.assertIn("price", res_payer)
        self.assertGreater(res_payer["fair_foreign_rate"], 0.0)
        self.assertGreater(res_payer["fair_domestic_rate"], 0.0)

        res_receiver = price_cross_currency_swap(
            model=self.two_curr_model,
            tenor_yrs=3.0,
            domestic_leg_type=SwapLegType.FIXED,
            foreign_leg_type=SwapLegType.FIXED,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.02,
            is_domestic_payer=False,
            exchange_notionals=True,
            n_paths=50,
            seed=42,
        )
        self.assertIn("price", res_receiver)

    def test_xccy_floating_floating_receiver(self) -> None:
        """Verify Float-Float XCCY with is_domestic_payer=False and exchange_notionals=False."""
        res = price_cross_currency_swap(
            model=self.two_curr_model,
            tenor_yrs=2.0,
            domestic_leg_type=SwapLegType.FLOATING,
            foreign_leg_type=SwapLegType.FLOATING,
            is_domestic_payer=False,
            exchange_notionals=False,
            n_paths=50,
            seed=42,
        )
        self.assertIn("price", res)
        self.assertEqual(res["notional_exchange_pv"], 0.0)

    def test_ir_model_validation_and_receiver_irs(self) -> None:
        """Verify IRS receiver swap simulation and invalid model type handling."""
        res_receiver = price_interest_rate_swap(
            model=self.two_curr_model.domestic_ir_model,
            fixed_rate_ann=0.03,
            tenor_yrs=2.0,
            is_payer=False,
            n_paths=50,
            seed=42,
        )
        self.assertIn("price", res_receiver)

        with self.assertRaises(TypeError):
            benchmark_price_interest_rate_swap(
                "invalid_model", fixed_rate_ann=0.03, tenor_yrs=2.0  # type: ignore
            )

    def test_cpi_option_validation_and_generic_inflation_model(self) -> None:
        """Verify CPI option error handling and generic InflationModel fallback."""
        nom_hw = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs_dom,
        )
        real_hw = HullWhite1FModel(
            a_ann=0.02,
            sigma_ann=0.008,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs_for,
        )
        jy_zero_vol = JarrowYildirimModel(
            nominal_ir_model=nom_hw,
            real_ir_model=real_hw,
            base_cpi=100.0,
            cpi_vol_ann=0.0,
            correlation_matrix=np.eye(3),
        )

        with self.assertRaises(ValueError):
            benchmark_price_consumer_price_index_option(
                jy_zero_vol, 0.02, 1.0, option_type="invalid"
            )
        with self.assertRaises(TypeError):
            benchmark_price_consumer_price_index_option(
                jy_zero_vol, 0.02, 1.0, option_type=12345  # type: ignore
            )

        # Jarrow-Yildirim with zero volatility intrinsic branch
        res_jy_call = benchmark_price_consumer_price_index_option(
            jy_zero_vol, strike_rate_ann=0.01, maturity_yrs=1.0, option_type="call"
        )
        self.assertGreater(res_jy_call["price"], 0.0)

        res_jy_put = benchmark_price_consumer_price_index_option(
            jy_zero_vol, strike_rate_ann=0.05, maturity_yrs=1.0, option_type="put"
        )
        self.assertGreater(res_jy_put["price"], 0.0)

    def test_custom_fx_model_without_discount_factor(self) -> None:
        """Verify fallback when FXModel lacks domestic_discount_factor."""

        class DummyFXModel(FXModel):
            @property
            def model_name(self) -> str:
                return "dummy_fx"

            @property
            def num_factors(self) -> int:
                return 1

            @property
            def spot_fx(self) -> float:
                return 1.15

            def simulate_paths(
                self, maturity_yrs: float, n_paths: int, n_steps: int, *args: typing.Any, **kwargs: typing.Any
            ) -> tuple[np.ndarray, np.ndarray]:
                times = np.linspace(0.0, maturity_yrs, n_steps + 1)
                paths = np.full((n_paths, n_steps + 1), 1.15)
                return times, paths

        dummy = DummyFXModel()
        bm = benchmark_price_foreign_exchange_forward(
            dummy, strike=1.15, maturity_yrs=1.0, notional=1000.0
        )
        self.assertIn("price", bm)

        res = price_foreign_exchange_forward(
            dummy, strike=1.15, maturity_yrs=1.0, notional=1000.0, n_paths=20
        )
        self.assertIn("price", res)

        opt = price_foreign_exchange_option(
            dummy, strike=1.15, maturity_yrs=1.0, notional=1.0, n_paths=20
        )
        self.assertIn("price", opt)


if __name__ == "__main__":
    unittest.main()
