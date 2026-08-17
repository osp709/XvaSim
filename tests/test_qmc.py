"""Tests for Quasi-Monte Carlo (QMC) random sequence generation and T0 NPV fitting."""

import unittest

import numpy as np

from xvasim import (
    BlackInflationModel,
    GarmanKohlhagenFXModel,
    HullWhite1FModel,
    LGMModel,
    LGMParams,
    OptionType,
    RandomSequenceType,
    SwapLegType,
    TwoCurrencyFXModel,
    benchmark_price_interest_rate_swap,
    compare_t0_npv_fitting,
    generate_brownian_increments,
    generate_normal_draws,
    price_consumer_price_index_option,
    price_cross_currency_swap,
    price_foreign_exchange_forward,
    price_foreign_exchange_option,
    price_interest_rate_swap,
    price_zero_coupon_inflation_swap,
)
from xvasim.qmc import _parse_random_sequence_type


class TestQMCGenerators(unittest.TestCase):
    def test_parse_random_sequence_type(self) -> None:
        self.assertEqual(
            _parse_random_sequence_type(RandomSequenceType.SOBOL),
            RandomSequenceType.SOBOL,
        )
        self.assertEqual(
            _parse_random_sequence_type("sobol"), RandomSequenceType.SOBOL
        )
        self.assertEqual(
            _parse_random_sequence_type("SOBOL"), RandomSequenceType.SOBOL
        )
        self.assertEqual(
            _parse_random_sequence_type("halton"), RandomSequenceType.HALTON
        )
        self.assertEqual(
            _parse_random_sequence_type("latin_hypercube"),
            RandomSequenceType.LATIN_HYPERCUBE,
        )
        self.assertEqual(
            _parse_random_sequence_type("lhs"),
            RandomSequenceType.LATIN_HYPERCUBE,
        )
        self.assertEqual(
            _parse_random_sequence_type("pseudo"), RandomSequenceType.PSEUDO
        )
        self.assertEqual(
            _parse_random_sequence_type("normal"), RandomSequenceType.PSEUDO
        )
        self.assertEqual(
            _parse_random_sequence_type("prng"), RandomSequenceType.PSEUDO
        )

        with self.assertRaises(ValueError):
            _parse_random_sequence_type("invalid_type")

        with self.assertRaises(TypeError):
            _parse_random_sequence_type(12345)  # type: ignore[arg-type]

    def test_generate_normal_draws_shapes_and_types(self) -> None:
        for seq_type in [
            RandomSequenceType.PSEUDO,
            RandomSequenceType.SOBOL,
            RandomSequenceType.HALTON,
            RandomSequenceType.LATIN_HYPERCUBE,
        ]:
            draws = generate_normal_draws(
                n_paths=1024,
                dimension=5,
                random_type=seq_type,
                seed=42,
            )
            self.assertEqual(draws.shape, (1024, 5))
            self.assertEqual(draws.dtype, np.float64)
            self.assertFalse(np.isnan(draws).any())
            self.assertFalse(np.isinf(draws).any())

    def test_generate_normal_draws_statistics(self) -> None:
        n_paths = 65536
        for seq_type in [
            RandomSequenceType.SOBOL,
            RandomSequenceType.HALTON,
            RandomSequenceType.LATIN_HYPERCUBE,
            RandomSequenceType.PSEUDO,
        ]:
            draws = generate_normal_draws(
                n_paths=n_paths,
                dimension=3,
                random_type=seq_type,
                seed=12345,
            )
            mean = np.mean(draws, axis=0)
            std = np.std(draws, axis=0)

            # Sample mean should be close to 0 and std close to 1
            np.testing.assert_allclose(mean, np.zeros(3), atol=0.03)
            np.testing.assert_allclose(std, np.ones(3), atol=0.03)

    def test_normal_draws_qmc_variance_reduction_vs_pseudo(self) -> None:
        """Verify QMC normal draws produce lower variance than normal PRNG draws."""
        seeds = list(range(100, 115))
        dim = 3
        n_paths = 2048

        def sample_expectation(seq_type: RandomSequenceType, s: int) -> float:
            draws = generate_normal_draws(
                n_paths=n_paths,
                dimension=dim,
                random_type=seq_type,
                seed=s,
            )
            # Evaluate nonlinear payoff E[max(Z1 + 0.5*Z2, 0)]
            payoff = np.maximum(draws[:, 0] + 0.5 * draws[:, 1], 0.0)
            return float(np.mean(payoff))

        estimates_pseudo = [
            sample_expectation(RandomSequenceType.PSEUDO, s) for s in seeds
        ]
        estimates_sobol = [
            sample_expectation(RandomSequenceType.SOBOL, s) for s in seeds
        ]
        estimates_halton = [
            sample_expectation(RandomSequenceType.HALTON, s) for s in seeds
        ]

        var_pseudo = float(np.var(estimates_pseudo, ddof=1))
        var_sobol = float(np.var(estimates_sobol, ddof=1))
        var_halton = float(np.var(estimates_halton, ddof=1))

        self.assertLess(var_sobol, var_pseudo)
        self.assertLess(var_halton, var_pseudo)
        self.assertGreater(var_pseudo / var_sobol, 5.0)

    def test_generate_normal_draws_reproducibility(self) -> None:
        draws1 = generate_normal_draws(
            n_paths=512,
            dimension=4,
            random_type=RandomSequenceType.SOBOL,
            seed=999,
        )
        draws2 = generate_normal_draws(
            n_paths=512,
            dimension=4,
            random_type=RandomSequenceType.SOBOL,
            seed=999,
        )
        draws3 = generate_normal_draws(
            n_paths=512,
            dimension=4,
            random_type=RandomSequenceType.SOBOL,
            seed=1000,
        )
        np.testing.assert_allclose(draws1, draws2)
        self.assertFalse(np.allclose(draws1, draws3))

    def test_generate_normal_draws_invalid_args(self) -> None:
        with self.assertRaises(ValueError):
            generate_normal_draws(n_paths=0, dimension=5)
        with self.assertRaises(ValueError):
            generate_normal_draws(n_paths=-10, dimension=5)
        with self.assertRaises(ValueError):
            generate_normal_draws(n_paths=100, dimension=0)
        with self.assertRaises(ValueError):
            generate_normal_draws(n_paths=100, dimension=-2)

    def test_generate_brownian_increments(self) -> None:
        dt_vec = np.array([0.25, 0.25, 0.5])
        # 1 factor
        dw1 = generate_brownian_increments(
            n_paths=256,
            dt_vec=dt_vec,
            num_factors=1,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        self.assertEqual(dw1.shape, (256, 3))

        # Multi-factor
        dw3 = generate_brownian_increments(
            n_paths=256,
            dt_vec=dt_vec,
            num_factors=3,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        self.assertEqual(dw3.shape, (256, 3, 3))

        # Empty steps
        empty_dw1 = generate_brownian_increments(n_paths=10, dt_vec=np.array([]))
        self.assertEqual(empty_dw1.shape, (10, 0))
        empty_dw3 = generate_brownian_increments(
            n_paths=10, dt_vec=np.array([]), num_factors=3
        )
        self.assertEqual(empty_dw3.shape, (10, 0, 3))


class TestT0NPVFittingComparison(unittest.TestCase):
    def setUp(self) -> None:
        self.tenors = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0])
        self.dom_dfs = np.exp(-0.03 * self.tenors)
        self.for_dfs = np.exp(-0.015 * self.tenors)

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
        self.for_hw_model = HullWhite1FModel(
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
            domestic_ir_model=self.hw_model,
            foreign_ir_model=self.for_hw_model,
            spot_fx=1.25,
            fx_vol_ann=0.10,
            correlation_matrix=self.corr,
        )
        self.black_inflation = BlackInflationModel(
            nominal_discount_curve_yrs=self.tenors,
            nominal_discount_factors=self.dom_dfs,
            real_discount_curve_yrs=self.tenors,
            real_discount_factors=self.for_dfs,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
        )

    def test_irs_t0_npv_fitting_and_sobol_noise_reduction(self) -> None:
        """At par rate, analytical T0 NPV is 0. Sobol should reduce noise vs PRNG."""
        bench = benchmark_price_interest_rate_swap(
            model=self.lgm_model,
            fixed_rate_ann=0.03,
            tenor_yrs=5.0,
            pay_freq_yrs=0.5,
            notional=1_000_000.0,
        )
        fair_rate = bench["fair_swap_rate"]

        # Run comparison tool across PRNG, Sobol, Halton, Latin Hypercube
        comp_res = compare_t0_npv_fitting(
            pricer_fn=price_interest_rate_swap,
            pricer_kwargs={
                "model": self.lgm_model,
                "fixed_rate_ann": fair_rate,
                "tenor_yrs": 5.0,
                "pay_freq_yrs": 0.5,
                "notional": 1_000_000.0,
                "is_payer": True,
            },
            methods=(
                RandomSequenceType.PSEUDO,
                RandomSequenceType.SOBOL,
                RandomSequenceType.HALTON,
                RandomSequenceType.LATIN_HYPERCUBE,
            ),
            n_paths=16384,
            seeds=(11, 22, 33, 44, 55),
        )

        self.assertAlmostEqual(comp_res["benchmark_price"], 0.0, places=5)
        methods_res = comp_res["methods"]

        sobol_mae = methods_res["sobol"]["mean_absolute_error"]
        pseudo_mae = methods_res["pseudo"]["mean_absolute_error"]

        # Sobol should provide lower mean absolute error at T0
        self.assertLess(sobol_mae, pseudo_mae)
        self.assertGreater(
            methods_res["sobol"]["variance_reduction_factor"], 1.0
        )

    def test_fx_option_t0_npv_fitting_across_qmc_methods(self) -> None:
        """Price FX option under GarmanKohlhagen using Sobol, Halton, LHS, PRNG."""
        gk_model = GarmanKohlhagenFXModel(
            spot_fx=1.20,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.015,
            fx_vol_ann=0.12,
        )

        for seq_type in [
            RandomSequenceType.SOBOL,
            RandomSequenceType.HALTON,
            RandomSequenceType.LATIN_HYPERCUBE,
            RandomSequenceType.PSEUDO,
        ]:
            res = price_foreign_exchange_option(
                params=gk_model,
                strike=1.22,
                maturity_yrs=1.0,
                notional=100_000.0,
                option_type=OptionType.CALL,
                n_paths=30_000,
                n_steps=40,
                seed=42,
                random_type=seq_type,
            )
            ana_price = res["analytical_benchmark_price"]
            self.assertIsNotNone(ana_price)
            mc_price = res["price"]
            std_err = res["std_error"]

            diff = abs(float(mc_price) - float(ana_price))
            self.assertLess(diff, 3.0 * float(std_err))

    def test_xccy_swap_sobol_pricing(self) -> None:
        """Price Cross-Currency Swap using Sobol low-discrepancy sequences."""
        res_sobol = price_cross_currency_swap(
            model=self.fx_model,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.015,
            domestic_leg_type=SwapLegType.FIXED,
            foreign_leg_type=SwapLegType.FLOATING,
            tenor_yrs=3.0,
            pay_freq_yrs=0.5,
            foreign_notional=100_000.0,
            n_paths=20_000,
            n_steps=60,
            seed=42,
            random_type=RandomSequenceType.SOBOL,
        )
        self.assertIn("price", res_sobol)
        self.assertIn("std_error", res_sobol)
        diff = abs(
            res_sobol["price"] - res_sobol["analytical_benchmark_price"]
        )
        self.assertLess(diff, 3.0 * res_sobol["std_error"])

    def test_fx_forward_sobol_pricing(self) -> None:
        """Price FX Forward using Sobol low-discrepancy sequences."""
        res = price_foreign_exchange_forward(
            params=self.fx_model,
            strike=1.25,
            maturity_yrs=2.0,
            notional=100_000.0,
            n_paths=20_000,
            n_steps=50,
            seed=42,
            random_type="sobol",
        )
        diff = abs(res["price"] - res["analytical_benchmark_price"])
        self.assertLess(diff, 3.0 * res["std_error"])

    def test_inflation_options_and_swaps_qmc(self) -> None:
        """Price Zero-Coupon Inflation Swap & CPI option with Sobol & Halton."""
        # ZCIS
        res_zcis = price_zero_coupon_inflation_swap(
            model=self.black_inflation,
            strike_rate_ann=0.015,
            maturity_yrs=2.0,
            notional=10_000.0,
            n_paths=10_000,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        diff_zcis = abs(
            res_zcis["price"] - res_zcis["analytical_benchmark_price"]
        )
        self.assertLess(diff_zcis, 3.0 * res_zcis["std_error"])

        # CPI Option
        res_cpi = price_consumer_price_index_option(
            model=self.black_inflation,
            strike_rate_ann=0.015,
            maturity_yrs=2.0,
            notional=10_000.0,
            n_paths=10_000,
            random_type=RandomSequenceType.HALTON,
            seed=42,
        )
        diff_cpi = abs(
            res_cpi["price"] - res_cpi["analytical_benchmark_price"]
        )
        self.assertLess(diff_cpi, 3.0 * res_cpi["std_error"])

    def test_qmc_lower_variance_than_normal_methods(self) -> None:
        """Verify that QMC methods give lower variance than normal PRNG methods."""
        gk_model = GarmanKohlhagenFXModel(
            spot_fx=1.20,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.015,
            fx_vol_ann=0.12,
        )

        comp_res = compare_t0_npv_fitting(
            pricer_fn=price_foreign_exchange_option,
            pricer_kwargs={
                "params": gk_model,
                "strike": 1.20,
                "maturity_yrs": 1.0,
                "notional": 100_000.0,
                "option_type": OptionType.CALL,
                "n_steps": 1,
            },
            methods=(
                RandomSequenceType.PSEUDO,
                RandomSequenceType.SOBOL,
                RandomSequenceType.HALTON,
                RandomSequenceType.LATIN_HYPERCUBE,
            ),
            n_paths=4096,
            seeds=(10, 20, 30, 40, 50, 60, 70, 80),
        )

        methods_res = comp_res["methods"]
        var_pseudo = methods_res["pseudo"]["variance"]
        var_sobol = methods_res["sobol"]["variance"]
        var_halton = methods_res["halton"]["variance"]
        var_lhs = methods_res["latin_hypercube"]["variance"]

        # QMC variance should be strictly lower than normal pseudo-random method
        self.assertLess(var_sobol, var_pseudo)
        self.assertLess(var_halton, var_pseudo)
        self.assertLess(var_lhs, var_pseudo)

        # Variance reduction factor relative to normal/pseudo must be > 1
        self.assertGreater(
            methods_res["sobol"]["variance_reduction_factor"], 1.0
        )
        self.assertGreater(
            methods_res["halton"]["variance_reduction_factor"], 1.0
        )
        self.assertGreater(
            methods_res["latin_hypercube"]["variance_reduction_factor"], 1.0
        )

        # Mean absolute pricing error should also be lower for Sobol
        mae_pseudo = methods_res["pseudo"]["mean_absolute_error"]
        mae_sobol = methods_res["sobol"]["mean_absolute_error"]
        self.assertLess(mae_sobol, mae_pseudo)


if __name__ == "__main__":
    unittest.main()

