"""Unit tests for inflation rate simulation models and pricing engine."""

from __future__ import annotations

import unittest

import numpy as np

from xvasim import (
    BlackInflationModel,
    HullWhite1FModel,
    InflationModel,
    InflationSimulationResult,
    JarrowYildirimModel,
    LGMParams,
    OptionType,
    RiskFactorType,
    benchmark_price_consumer_price_index_option,
    benchmark_price_cpi_option,
    benchmark_price_zero_coupon_inflation_swap,
    create_inflation_model,
    list_available_models,
    price_consumer_price_index_option,
    price_cpi_option,
    price_year_on_year_inflation_swap,
    price_yoy_inflation_swap,
    price_zero_coupon_inflation_swap,
)


def _make_sample_curves(
    nominal_rate: float = 0.035,
    real_rate: float = 0.015,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tenors = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    nom_dfs = np.exp(-nominal_rate * tenors)
    real_dfs = np.exp(-real_rate * tenors)
    return tenors, nom_dfs, real_dfs


class TestInflationRegistry(unittest.TestCase):
    """Test registry and factory discovery for inflation models."""

    def test_list_inflation_models(self) -> None:
        models = list_available_models(RiskFactorType.INFLATION)
        self.assertIn("jarrow_yildirim", models)
        self.assertIn("jy", models)
        self.assertIn("black", models)

    def test_create_inflation_model_factory(self) -> None:
        tenors, nom_dfs, real_dfs = _make_sample_curves()
        nom_ir = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.008,
            discount_curve_yrs=tenors,
            discount_factors=nom_dfs,
        )
        real_ir = HullWhite1FModel(
            a_ann=0.02,
            sigma_ann=0.006,
            discount_curve_yrs=tenors,
            discount_factors=real_dfs,
        )
        model = create_inflation_model(
            "jarrow_yildirim",
            nominal_ir_model=nom_ir,
            real_ir_model=real_ir,
            base_cpi=105.0,
            cpi_vol_ann=0.018,
        )
        self.assertIsInstance(model, InflationModel)
        self.assertIsInstance(model, JarrowYildirimModel)
        self.assertEqual(model.base_cpi, 105.0)

    def test_create_black_model_factory(self) -> None:
        tenors, nom_dfs, real_dfs = _make_sample_curves()
        model = create_inflation_model(
            "black",
            nominal_discount_curve_yrs=tenors,
            nominal_discount_factors=nom_dfs,
            real_discount_curve_yrs=tenors,
            real_discount_factors=real_dfs,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
        )
        self.assertIsInstance(model, BlackInflationModel)


class TestJarrowYildirimModel(unittest.TestCase):
    """Test Jarrow-Yildirim inflation model simulation and properties."""

    def setUp(self) -> None:
        self.tenors, self.nom_dfs, self.real_dfs = _make_sample_curves(
            nominal_rate=0.04, real_rate=0.015
        )
        self.nom_ir = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.008,
            discount_curve_yrs=self.tenors,
            discount_factors=self.nom_dfs,
        )
        self.real_ir = HullWhite1FModel(
            a_ann=0.02,
            sigma_ann=0.006,
            discount_curve_yrs=self.tenors,
            discount_factors=self.real_dfs,
        )
        self.corr = np.array([
            [1.0, 0.4, 0.2],
            [0.4, 1.0, -0.1],
            [0.2, -0.1, 1.0],
        ])
        self.model = JarrowYildirimModel(
            nominal_ir_model=self.nom_ir,
            real_ir_model=self.real_ir,
            base_cpi=100.0,
            cpi_vol_ann=0.025,
            correlation_matrix=self.corr,
        )

    def test_invalid_parameters_raise(self) -> None:
        with self.assertRaises(ValueError):
            JarrowYildirimModel(
                nominal_ir_model=self.nom_ir,
                real_ir_model=self.real_ir,
                base_cpi=-10.0,
            )
        with self.assertRaises(ValueError):
            JarrowYildirimModel(
                nominal_ir_model=self.nom_ir,
                real_ir_model=self.real_ir,
                cpi_vol_ann=-0.01,
            )
        with self.assertRaises(ValueError):
            JarrowYildirimModel(
                nominal_ir_model=self.nom_ir,
                real_ir_model=self.real_ir,
                correlation_matrix=np.eye(2),
            )

    def test_forward_cpi_and_swap_rate(self) -> None:
        # P_nom(5) = exp(-0.04*5) = exp(-0.20)
        # P_real(5) = exp(-0.015*5) = exp(-0.075)
        # F_cpi(5) = 100 * exp(0.125) = 100 * exp(0.025 * 5)
        fwd_cpi = self.model.forward_cpi(5.0)
        expected_fwd = 100.0 * np.exp((0.04 - 0.015) * 5.0)
        self.assertAlmostEqual(fwd_cpi, expected_fwd, places=5)

        fair_swap_rate = self.model.zero_coupon_inflation_swap_rate(5.0)
        expected_rate = np.exp(0.04 - 0.015) - 1.0
        self.assertAlmostEqual(fair_swap_rate, expected_rate, places=5)

    def test_from_lgm_params_constructor(self) -> None:
        nom_lgm = LGMParams(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([30.0]),
            sigma_values_ann=np.array([0.008]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.nom_dfs,
        )
        real_lgm = LGMParams(
            kappa_ann=0.02,
            sigma_grid_yrs=np.array([30.0]),
            sigma_values_ann=np.array([0.006]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.real_dfs,
        )
        lgm_model = JarrowYildirimModel.from_lgm_params(
            nominal=nom_lgm,
            real=real_lgm,
            base_cpi=120.0,
            cpi_vol_ann=0.02,
        )
        self.assertIsInstance(lgm_model, JarrowYildirimModel)
        self.assertEqual(lgm_model.base_cpi, 120.0)

    def test_simulation_paths_structure_and_unpacking(self) -> None:
        rng = np.random.default_rng(42)
        sim_res = self.model.simulate_paths(
            maturity_yrs=2.0,
            n_paths=500,
            n_steps=24,
            rng=rng,
        )
        self.assertIsInstance(sim_res, InflationSimulationResult)
        self.assertEqual(sim_res.times.shape, (25,))
        self.assertEqual(sim_res.cpi_index.shape, (500, 25))
        self.assertEqual(sim_res.nominal_states.shape, (500, 25))
        self.assertEqual(sim_res.real_states.shape, (500, 25))
        self.assertEqual(sim_res.nominal_discount_factors.shape, (500, 25))

        # Test tuple unpacking
        times, x_nom, x_real, cpi = sim_res
        self.assertEqual(len(times), 25)
        self.assertEqual(x_nom.shape, (500, 25))
        self.assertEqual(x_real.shape, (500, 25))
        self.assertEqual(cpi.shape, (500, 25))
        self.assertTrue(np.all(cpi > 0))

    def test_mc_discounted_cpi_martingale_property(self) -> None:
        """E^{Q^n}[D_n(0, T) * I(T)] must match I(0) * P_r(0, T)."""
        maturity_yrs = 2.0
        rng = np.random.default_rng(123)
        sim_res = self.model.simulate_paths(
            maturity_yrs=maturity_yrs,
            n_paths=40_000,
            n_steps=40,
            rng=rng,
        )
        cpi_t = sim_res.cpi_index[:, -1]
        df_n_t = sim_res.nominal_discount_factors[:, -1]

        mc_discounted_cpi = float(np.mean(df_n_t * cpi_t))
        p_real_t = float(self.real_ir.interpolate_discount_factor(maturity_yrs))
        theoretical_target = self.model.base_cpi * p_real_t

        rel_error = abs(mc_discounted_cpi - theoretical_target) / theoretical_target
        self.assertLess(rel_error, 0.015)


class TestBlackInflationModel(unittest.TestCase):
    """Test Black forward CPI inflation model."""

    def setUp(self) -> None:
        self.tenors, self.nom_dfs, self.real_dfs = _make_sample_curves(
            nominal_rate=0.03, real_rate=0.01
        )
        self.model = BlackInflationModel(
            nominal_discount_curve_yrs=self.tenors,
            nominal_discount_factors=self.nom_dfs,
            real_discount_curve_yrs=self.tenors,
            real_discount_factors=self.real_dfs,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
        )

    def test_put_call_parity(self) -> None:
        maturity_yrs = 3.0
        strike_rate = 0.022  # 2.2% annual inflation
        call_pv = self.model.price_cpi_option_analytical(
            strike_rate_ann=strike_rate,
            maturity_yrs=maturity_yrs,
            notional=100_000.0,
            is_call=True,
        )
        put_pv = self.model.price_cpi_option_analytical(
            strike_rate_ann=strike_rate,
            maturity_yrs=maturity_yrs,
            notional=100_000.0,
            is_call=False,
        )
        p_nom = float(self.model.interpolate_nominal_df(maturity_yrs))
        fwd_ratio = self.model.forward_cpi(maturity_yrs) / self.model.base_cpi
        k_comp = (1.0 + strike_rate) ** maturity_yrs
        expected_diff = 100_000.0 * p_nom * (fwd_ratio - k_comp)

        self.assertAlmostEqual(call_pv - put_pv, expected_diff, places=6)

    def test_analytical_vs_mc_option_convergence(self) -> None:
        maturity_yrs = 2.0
        strike_rate = 0.02
        res_analytical = benchmark_price_cpi_option(
            self.model,
            strike_rate_ann=strike_rate,
            maturity_yrs=maturity_yrs,
            notional=10_000.0,
            option_type=OptionType.CALL,
        )
        res_mc = price_cpi_option(
            self.model,
            strike_rate_ann=strike_rate,
            maturity_yrs=maturity_yrs,
            notional=10_000.0,
            option_type=OptionType.CALL,
            n_paths=50_000,
            n_steps=20,
            seed=42,
        )
        self.assertAlmostEqual(
            res_analytical["price"], res_mc["price"], delta=3 * res_mc["std_error"]
        )

    def test_zero_coupon_inflation_swap_at_fair_rate(self) -> None:
        maturity_yrs = 5.0
        fair_rate = self.model.zero_coupon_inflation_swap_rate(maturity_yrs)

        res_analytical = benchmark_price_zero_coupon_inflation_swap(
            self.model,
            strike_rate_ann=fair_rate,
            maturity_yrs=maturity_yrs,
            notional=1_000_000.0,
            is_payer=True,
        )
        self.assertAlmostEqual(res_analytical["price"], 0.0, places=5)

    def test_zero_coupon_swap_payer_vs_receiver(self) -> None:
        maturity_yrs = 3.0
        strike_rate = 0.025
        payer = benchmark_price_zero_coupon_inflation_swap(
            self.model,
            strike_rate_ann=strike_rate,
            maturity_yrs=maturity_yrs,
            notional=100_000.0,
            is_payer=True,
        )
        receiver = benchmark_price_zero_coupon_inflation_swap(
            self.model,
            strike_rate_ann=strike_rate,
            maturity_yrs=maturity_yrs,
            notional=100_000.0,
            is_payer=False,
        )
        self.assertAlmostEqual(payer["price"] + receiver["price"], 0.0, places=6)

    def test_zero_coupon_swap_mc_pricing(self) -> None:
        maturity_yrs = 3.0
        strike_rate = 0.025
        res_mc = price_zero_coupon_inflation_swap(
            self.model,
            strike_rate_ann=strike_rate,
            maturity_yrs=maturity_yrs,
            notional=100_000.0,
            is_payer=True,
            n_paths=30_000,
            seed=42,
        )
        self.assertIn("price", res_mc)
        self.assertIn("std_error", res_mc)
        self.assertIn("analytical_benchmark_price", res_mc)
        self.assertAlmostEqual(
            res_mc["price"],
            res_mc["analytical_benchmark_price"],
            delta=3 * res_mc["std_error"],
        )

    def test_yoy_inflation_swap_pricing(self) -> None:
        pay_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = price_yoy_inflation_swap(
            self.model,
            fixed_rate_ann=0.02,
            payment_times_yrs=pay_times,
            notional=100_000.0,
            is_payer=True,
            n_paths=10_000,
            seed=42,
        )
        self.assertIn("price", res)
        self.assertIn("std_error", res)
        self.assertEqual(len(res["period_cash_flows"]), 5)

    def test_cpi_option_string_types(self) -> None:
        call_res = price_cpi_option(
            self.model,
            strike_rate_ann=0.02,
            maturity_yrs=1.0,
            notional=10_000.0,
            option_type="call",
        )
        put_res = price_cpi_option(
            self.model,
            strike_rate_ann=0.02,
            maturity_yrs=1.0,
            notional=10_000.0,
            option_type="put",
        )
        self.assertGreater(call_res["price"], 0.0)
        self.assertGreater(put_res["price"], 0.0)

        with self.assertRaises(ValueError):
            price_cpi_option(
                self.model,
                strike_rate_ann=0.02,
                maturity_yrs=1.0,
                option_type="invalid_type",
            )

    def test_fully_expanded_and_alias_equivalence(self) -> None:
        """Verify that fully expanded inflation functions match alias functions."""
        self.assertIs(price_yoy_inflation_swap, price_year_on_year_inflation_swap)
        self.assertIs(price_cpi_option, price_consumer_price_index_option)
        self.assertIs(
            benchmark_price_cpi_option,
            benchmark_price_consumer_price_index_option,
        )

        # Model method equivalence
        p1 = self.model.price_cpi_option_analytical(0.025, 2.0, 10_000.0, True)
        p2 = self.model.price_consumer_price_index_option_analytical(
            0.025, 2.0, 10_000.0, True
        )
        self.assertEqual(p1, p2)

        # Functional equivalence
        res_exp = price_consumer_price_index_option(
            self.model,
            strike_rate_ann=0.025,
            maturity_yrs=2.0,
            notional=10_000.0,
            n_paths=1000,
            seed=42,
        )
        res_alias = price_cpi_option(
            self.model,
            strike_rate_ann=0.025,
            maturity_yrs=2.0,
            notional=10_000.0,
            n_paths=1000,
            seed=42,
        )
        self.assertEqual(res_exp["price"], res_alias["price"])


if __name__ == "__main__":
    unittest.main()
