"""Tests for single-currency and cross-currency swap pricing engines."""

import unittest

import numpy as np

from xvasim import (
    CIRInterestRateModel,
    FXLGMParams,
    HullWhite1FModel,
    LGMModel,
    LGMParams,
    SwapLegType,
    TwoCurrencyFXModel,
    VasicekModel,
    benchmark_price_cross_currency_swap,
    benchmark_price_interest_rate_swap,
    benchmark_price_irs,
    benchmark_price_xccy_swap,
    price_cross_currency_swap,
    price_interest_rate_swap,
    price_irs,
    price_xccy_swap,
)
from xvasim.pricing_engine import _generate_swap_schedule, _parse_swap_leg_type


def _make_flat_curves(
    dom_rate: float = 0.03,
    for_rate: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tenors = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0])
    dom_dfs = np.exp(-dom_rate * tenors)
    for_dfs = np.exp(-for_rate * tenors)
    return tenors, dom_dfs, for_dfs


class TestSwapHelpers(unittest.TestCase):
    def test_parse_swap_leg_type(self) -> None:
        self.assertEqual(
            _parse_swap_leg_type(SwapLegType.FIXED), SwapLegType.FIXED
        )
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
    def setUp(self) -> None:
        self.tenors, self.dom_dfs, _ = _make_flat_curves(0.03, 0.01)
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
            sigma_ann=0.01,
            r0_ann=0.03,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dom_dfs,
        )
        self.cir_model = CIRInterestRateModel(
            kappa_ann=0.1,
            theta_ann=0.03,
            sigma_ann=0.02,
            r0_ann=0.03,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dom_dfs,
        )

    def test_par_swap_rate_is_zero_pv(self) -> None:
        """At the par swap rate, the analytical benchmark swap PV should be zero."""
        res_dummy = benchmark_price_interest_rate_swap(
            model=self.lgm_model,
            fixed_rate_ann=0.03,
            tenor_yrs=5.0,
            pay_freq_yrs=0.5,
            notional=1_000_000.0,
        )
        fair_rate = res_dummy["fair_swap_rate"]

        res_par = benchmark_price_interest_rate_swap(
            model=self.lgm_model,
            fixed_rate_ann=fair_rate,
            tenor_yrs=5.0,
            pay_freq_yrs=0.5,
            notional=1_000_000.0,
            is_payer=True,
        )
        self.assertAlmostEqual(res_par["price"], 0.0, places=6)
        self.assertAlmostEqual(
            res_par["fixed_leg_pv"], res_par["floating_leg_pv"], places=6
        )

    def test_payer_receiver_symmetry(self) -> None:
        """Payer and Receiver analytical swap prices should sum to zero."""
        res_payer = benchmark_price_interest_rate_swap(
            model=self.hw_model,
            fixed_rate_ann=0.04,
            tenor_yrs=3.0,
            pay_freq_yrs=0.5,
            notional=500_000.0,
            is_payer=True,
        )
        res_receiver = benchmark_price_interest_rate_swap(
            model=self.hw_model,
            fixed_rate_ann=0.04,
            tenor_yrs=3.0,
            pay_freq_yrs=0.5,
            notional=500_000.0,
            is_payer=False,
        )
        self.assertAlmostEqual(
            res_payer["price"] + res_receiver["price"], 0.0, places=9
        )

    def test_floating_spread_effect(self) -> None:
        """Floating spread should increase payer swap PV by N * s * Annuity."""
        spread = 0.0025  # 25 bp
        res_base = benchmark_price_interest_rate_swap(
            model=self.lgm_model,
            fixed_rate_ann=0.03,
            tenor_yrs=2.0,
            pay_freq_yrs=0.5,
            notional=100_000.0,
            spread_ann=0.0,
            is_payer=True,
        )
        res_spread = benchmark_price_interest_rate_swap(
            model=self.lgm_model,
            fixed_rate_ann=0.03,
            tenor_yrs=2.0,
            pay_freq_yrs=0.5,
            notional=100_000.0,
            spread_ann=spread,
            is_payer=True,
        )
        expected_diff = 100_000.0 * spread * res_base["annuity"]
        self.assertAlmostEqual(
            res_spread["price"] - res_base["price"], expected_diff, places=6
        )

    def test_alias_price_irs(self) -> None:
        """Alias price_irs and benchmark_price_irs match counterparts."""
        res_bench1 = benchmark_price_interest_rate_swap(
            self.lgm_model, fixed_rate_ann=0.03, tenor_yrs=1.0
        )
        res_bench2 = benchmark_price_irs(
            self.lgm_model, fixed_rate_ann=0.03, tenor_yrs=1.0
        )
        self.assertEqual(res_bench1["price"], res_bench2["price"])

        res1 = price_interest_rate_swap(
            self.lgm_model, fixed_rate_ann=0.03, tenor_yrs=1.0, seed=42
        )
        res2 = price_irs(
            self.lgm_model, fixed_rate_ann=0.03, tenor_yrs=1.0, seed=42
        )
        self.assertEqual(res1["price"], res2["price"])

    def test_accepts_lgm_params(self) -> None:
        """Should accept legacy LGMParams dataclass directly."""
        res = price_interest_rate_swap(
            self.lgm_params, fixed_rate_ann=0.03, tenor_yrs=2.0, seed=42
        )
        self.assertIn("price", res)
        self.assertIn("annuity", res)
        self.assertIn("analytical_benchmark_price", res)

    def test_various_ir_models(self) -> None:
        """All supported IR models should price IRS correctly via MC and benchmark."""
        for mod in [
            self.lgm_model,
            self.hw_model,
            self.vasicek_model,
            self.cir_model,
        ]:
            res = price_interest_rate_swap(
                mod, fixed_rate_ann=0.03, tenor_yrs=3.0, seed=42
            )
            self.assertIn("price", res)
            self.assertIn("std_error", res)
            self.assertIn("analytical_benchmark_price", res)
            self.assertGreater(res["annuity"], 0.0)

    def test_monte_carlo_matches_analytical(self) -> None:
        """MC IRS pricing should match analytical within 3 standard errors."""
        res_ana = benchmark_price_interest_rate_swap(
            model=self.hw_model,
            fixed_rate_ann=0.035,
            tenor_yrs=2.0,
            pay_freq_yrs=0.5,
            notional=100_000.0,
            is_payer=True,
        )
        res_mc = price_interest_rate_swap(
            model=self.hw_model,
            fixed_rate_ann=0.035,
            tenor_yrs=2.0,
            pay_freq_yrs=0.5,
            notional=100_000.0,
            is_payer=True,
            n_paths=30_000,
            n_steps_per_year=20,
            seed=42,
        )
        self.assertIn("std_error", res_mc)
        diff = abs(res_mc["price"] - res_ana["price"])
        self.assertLess(diff, 3.0 * res_mc["std_error"])
        # Check leg PVs
        self.assertLess(
            abs(res_mc["fixed_leg_pv"] - res_ana["fixed_leg_pv"]),
            3.0 * res_mc["std_error"],
        )
        self.assertLess(
            abs(res_mc["floating_leg_pv"] - res_ana["floating_leg_pv"]),
            3.0 * res_mc["std_error"],
        )

    def test_invalid_model_type(self) -> None:
        with self.assertRaises(TypeError):
            price_interest_rate_swap(
                "invalid_model",  # type: ignore[arg-type]
                fixed_rate_ann=0.03,
                tenor_yrs=1.0,
            )


class TestCrossCurrencySwapPricing(unittest.TestCase):
    def setUp(self) -> None:
        self.tenors, self.dom_dfs, self.for_dfs = _make_flat_curves(0.03, 0.015)
        self.dom_ir = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dom_dfs,
        )
        self.for_ir = HullWhite1FModel(
            a_ann=0.02,
            sigma_ann=0.008,
            discount_curve_yrs=self.tenors,
            discount_factors=self.for_dfs,
        )
        self.corr = np.array([
            [1.0, 0.3, -0.1],
            [0.3, 1.0, 0.2],
            [-0.1, 0.2, 1.0],
        ])
        self.fx_model = TwoCurrencyFXModel(
            domestic_ir_model=self.dom_ir,
            foreign_ir_model=self.for_ir,
            spot_fx=1.20,
            fx_vol_ann=0.10,
            correlation_matrix=self.corr,
        )
        self.dom_lgm = LGMParams(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([30.0]),
            sigma_values_ann=np.array([0.01]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.dom_dfs,
        )
        self.for_lgm = LGMParams(
            kappa_ann=0.02,
            sigma_grid_yrs=np.array([30.0]),
            sigma_values_ann=np.array([0.008]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.for_dfs,
        )
        self.fx_lgm_params = FXLGMParams(
            domestic=self.dom_lgm,
            foreign=self.for_lgm,
            spot_fx=1.20,
            fx_vol_ann=0.10,
            correlation_matrix=self.corr,
        )

    def test_fixed_vs_floating_fair_foreign_rate(self) -> None:
        """Solving for fair foreign fixed rate should make XCCY swap PV zero."""
        res_calc = benchmark_price_cross_currency_swap(
            model=self.fx_model,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.0,
            domestic_leg_type=SwapLegType.FIXED,
            foreign_leg_type=SwapLegType.FIXED,
            tenor_yrs=3.0,
            pay_freq_yrs=0.5,
            foreign_notional=100_000.0,
            is_domestic_payer=True,
            exchange_notionals=True,
        )
        fair_for_rate = res_calc["fair_foreign_rate"]
        self.assertGreater(fair_for_rate, 0.0)

        # Price with fair rate
        res_fair = benchmark_price_cross_currency_swap(
            model=self.fx_model,
            domestic_rate_ann=0.03,
            foreign_rate_ann=fair_for_rate,
            domestic_leg_type=SwapLegType.FIXED,
            foreign_leg_type=SwapLegType.FIXED,
            tenor_yrs=3.0,
            pay_freq_yrs=0.5,
            foreign_notional=100_000.0,
            is_domestic_payer=True,
            exchange_notionals=True,
        )
        self.assertAlmostEqual(res_fair["price"], 0.0, places=6)

    def test_cross_currency_basis_swap_fair_spread(self) -> None:
        """Floating-floating basis swap fair spread sets PV to zero."""
        res_calc = benchmark_price_cross_currency_swap(
            model=self.fx_model,
            domestic_spread_ann=0.0,
            foreign_spread_ann=0.0,
            domestic_leg_type=SwapLegType.FLOATING,
            foreign_leg_type=SwapLegType.FLOATING,
            tenor_yrs=5.0,
            pay_freq_yrs=0.5,
            foreign_notional=1_000_000.0,
            is_domestic_payer=True,
            exchange_notionals=True,
        )
        fair_spread = res_calc["fair_foreign_spread"]

        res_fair = benchmark_price_cross_currency_swap(
            model=self.fx_model,
            domestic_spread_ann=0.0,
            foreign_spread_ann=fair_spread,
            domestic_leg_type=SwapLegType.FLOATING,
            foreign_leg_type=SwapLegType.FLOATING,
            tenor_yrs=5.0,
            pay_freq_yrs=0.5,
            foreign_notional=1_000_000.0,
            is_domestic_payer=True,
            exchange_notionals=True,
        )
        self.assertAlmostEqual(res_fair["price"], 0.0, places=5)

    def test_payer_receiver_symmetry(self) -> None:
        """Domestic payer and Domestic receiver cross-currency swaps sum to zero."""
        res_payer = benchmark_price_cross_currency_swap(
            model=self.fx_model,
            domestic_rate_ann=0.035,
            foreign_rate_ann=0.015,
            domestic_leg_type="fixed",
            foreign_leg_type="fixed",
            tenor_yrs=2.0,
            pay_freq_yrs=0.5,
            foreign_notional=50_000.0,
            is_domestic_payer=True,
            exchange_notionals=True,
        )
        res_receiver = benchmark_price_cross_currency_swap(
            model=self.fx_model,
            domestic_rate_ann=0.035,
            foreign_rate_ann=0.015,
            domestic_leg_type="fixed",
            foreign_leg_type="fixed",
            tenor_yrs=2.0,
            pay_freq_yrs=0.5,
            foreign_notional=50_000.0,
            is_domestic_payer=False,
            exchange_notionals=True,
        )
        self.assertAlmostEqual(
            res_payer["price"] + res_receiver["price"], 0.0, places=9
        )

    def test_no_exchange_notionals(self) -> None:
        """When exchange_notionals is False, notional_exchange_pv is zero."""
        res = benchmark_price_cross_currency_swap(
            model=self.fx_model,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.01,
            domestic_leg_type=SwapLegType.FIXED,
            foreign_leg_type=SwapLegType.FIXED,
            tenor_yrs=2.0,
            exchange_notionals=False,
        )
        self.assertEqual(res["notional_exchange_pv"], 0.0)

    def test_legacy_fx_lgm_params_compatibility(self) -> None:
        """Cross-currency swap pricer works with legacy FXLGMParams."""
        res1 = benchmark_price_cross_currency_swap(
            model=self.fx_lgm_params,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.01,
            tenor_yrs=2.0,
        )
        res2 = benchmark_price_cross_currency_swap(
            model=self.fx_model,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.01,
            tenor_yrs=2.0,
        )
        self.assertAlmostEqual(res1["price"], res2["price"], places=6)

    def test_alias_price_xccy_swap(self) -> None:
        """Alias price_xccy_swap matches price_cross_currency_swap."""
        res_b1 = benchmark_price_cross_currency_swap(
            self.fx_model, domestic_rate_ann=0.03, tenor_yrs=1.0
        )
        res_b2 = benchmark_price_xccy_swap(
            self.fx_model, domestic_rate_ann=0.03, tenor_yrs=1.0
        )
        self.assertEqual(res_b1["price"], res_b2["price"])

        res1 = price_cross_currency_swap(
            self.fx_model, domestic_rate_ann=0.03, tenor_yrs=1.0, seed=42
        )
        res2 = price_xccy_swap(
            self.fx_model, domestic_rate_ann=0.03, tenor_yrs=1.0, seed=42
        )
        self.assertEqual(res1["price"], res2["price"])

    def test_cross_currency_monte_carlo_matches_analytical(self) -> None:
        """MC XCCY pricing should match analytical within 3 standard errors."""
        res_ana = benchmark_price_cross_currency_swap(
            model=self.fx_model,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.012,
            domestic_leg_type=SwapLegType.FIXED,
            foreign_leg_type=SwapLegType.FLOATING,
            tenor_yrs=2.0,
            pay_freq_yrs=0.5,
            foreign_notional=100_000.0,
            is_domestic_payer=True,
            exchange_notionals=True,
        )
        res_mc = price_cross_currency_swap(
            model=self.fx_model,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.012,
            domestic_leg_type=SwapLegType.FIXED,
            foreign_leg_type=SwapLegType.FLOATING,
            tenor_yrs=2.0,
            pay_freq_yrs=0.5,
            foreign_notional=100_000.0,
            is_domestic_payer=True,
            exchange_notionals=True,
            n_paths=30_000,
            n_steps=40,
            seed=42,
        )
        self.assertIn("std_error", res_mc)
        self.assertIn("analytical_benchmark_price", res_mc)
        diff = abs(res_mc["price"] - res_ana["price"])
        self.assertLess(diff, 3.0 * res_mc["std_error"])


if __name__ == "__main__":
    unittest.main()
