"""Statistical benchmark tests verifying Monte Carlo convergence across increasing number of paths."""

import unittest

import numpy as np
import pytest

from tests.helpers.assertions import assert_path_convergence
from tests.helpers.test_curves import (
    get_standard_credit_curve,
    get_standard_discount_curve,
    get_standard_inflation_curve,
)
from xvasim.cva_engine import compute_cva
from xvasim.models.credit.cir import CIRHazardRateModel
from xvasim.models.fx.garman_kohlhagen import GarmanKohlhagenFXModel
from xvasim.models.fx.two_currency import TwoCurrencyFXModel
from xvasim.models.inflation.black_inflation import BlackInflationModel
from xvasim.models.ir.hull_white import HullWhite1FModel
from xvasim.models.ir.lgm import LGMModel
from xvasim.pricing_engine import (
    OptionType,
    price_consumer_price_index_option,
    price_cross_currency_swap,
    price_foreign_exchange_forward,
    price_foreign_exchange_option,
    price_interest_rate_swap,
    price_zero_coupon_inflation_swap,
)
from xvasim.qmc import RandomSequenceType


@pytest.mark.benchmark
class TestPathConvergenceBenchmarks(unittest.TestCase):
    """Convergence tests verifying standard error decay and benchmark convergence as paths increase."""

    def test_irs_hull_white_path_convergence(self) -> None:
        """Verify IRS Monte Carlo standard error decay and analytical convergence with Hull-White 1F."""
        tenors, dfs = get_standard_discount_curve(flat_rate_ann=0.03)
        hw = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=tenors,
            discount_factors=dfs,
        )

        path_counts = [128, 512, 2048, 8192]
        prices: list[float] = []
        std_errors: list[float] = []
        bench_price: float | None = None

        for n in path_counts:
            res = price_interest_rate_swap(
                model=hw,
                fixed_rate_ann=0.03,
                tenor_yrs=5.0,
                is_payer=True,
                n_paths=n,
                random_type=RandomSequenceType.SOBOL,
                seed=42,
            )
            prices.append(float(res["price"]))
            std_errors.append(float(res["std_error"]))
            if bench_price is None:
                bench_price = float(res["analytical_benchmark_price"])

        assert_path_convergence(
            path_counts=path_counts,
            std_errors=std_errors,
            prices=prices,
            benchmark_price=bench_price,
            min_decay_rate=0.35,
            num_std=3.5,
        )

    def test_irs_lgm_path_convergence(self) -> None:
        """Verify IRS Monte Carlo standard error decay and analytical convergence with LGM."""
        tenors, dfs = get_standard_discount_curve(flat_rate_ann=0.035)
        lgm = LGMModel(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([30.0]),
            sigma_values_ann=np.array([0.012]),
            discount_curve_yrs=tenors,
            discount_factors=dfs,
        )

        path_counts = [128, 512, 2048, 8192]
        prices: list[float] = []
        std_errors: list[float] = []
        bench_price: float | None = None

        for n in path_counts:
            res = price_interest_rate_swap(
                model=lgm,
                fixed_rate_ann=0.035,
                tenor_yrs=5.0,
                is_payer=True,
                n_paths=n,
                random_type=RandomSequenceType.SOBOL,
                seed=101,
            )
            prices.append(float(res["price"]))
            std_errors.append(float(res["std_error"]))
            if bench_price is None:
                bench_price = float(res["analytical_benchmark_price"])

        assert_path_convergence(
            path_counts=path_counts,
            std_errors=std_errors,
            prices=prices,
            benchmark_price=bench_price,
            min_decay_rate=0.35,
            num_std=3.5,
        )

    def test_fx_option_path_convergence(self) -> None:
        """Verify European FX Option standard error decay and Garman-Kohlhagen convergence."""
        gk = GarmanKohlhagenFXModel(
            spot_fx=1.20,
            fx_vol_ann=0.15,
            domestic_rate_ann=0.04,
            foreign_rate_ann=0.02,
        )

        path_counts = [128, 512, 2048, 8192]
        prices: list[float] = []
        std_errors: list[float] = []
        bench_price: float | None = None

        for n in path_counts:
            res = price_foreign_exchange_option(
                params=gk,
                strike=1.20,
                maturity_yrs=1.0,
                notional=1.0,
                option_type=OptionType.CALL,
                n_paths=n,
                random_type=RandomSequenceType.SOBOL,
                seed=42,
            )
            prices.append(float(res["price"]))
            std_errors.append(float(res["std_error"]))
            if bench_price is None:
                bench_price = float(res["analytical_benchmark_price"])

        assert_path_convergence(
            path_counts=path_counts,
            std_errors=std_errors,
            prices=prices,
            benchmark_price=bench_price,
            min_decay_rate=0.35,
            num_std=3.5,
        )

    def test_fx_forward_path_convergence(self) -> None:
        """Verify FX Forward standard error decay and analytical benchmark convergence."""
        gk = GarmanKohlhagenFXModel(
            spot_fx=1.10,
            fx_vol_ann=0.10,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.01,
        )

        path_counts = [128, 512, 2048, 8192]
        prices: list[float] = []
        std_errors: list[float] = []
        bench_price: float | None = None

        for n in path_counts:
            res = price_foreign_exchange_forward(
                params=gk,
                strike=1.12,
                maturity_yrs=2.0,
                notional=100.0,
                n_paths=n,
                random_type=RandomSequenceType.SOBOL,
                seed=42,
            )
            prices.append(float(res["price"]))
            std_errors.append(float(res["std_error"]))
            if bench_price is None:
                bench_price = float(res["analytical_benchmark_price"])

        assert_path_convergence(
            path_counts=path_counts,
            std_errors=std_errors,
            prices=prices,
            benchmark_price=bench_price,
            min_decay_rate=0.35,
            num_std=3.5,
        )

    def test_cpi_option_path_convergence(self) -> None:
        """Verify CPI Option standard error decay and Black model closed-form convergence."""
        tenors, nom_dfs, real_dfs = get_standard_inflation_curve(
            base_cpi=100.0, expected_inflation_ann=0.02
        )
        black = BlackInflationModel(
            nominal_discount_curve_yrs=tenors,
            nominal_discount_factors=nom_dfs,
            real_discount_curve_yrs=tenors,
            real_discount_factors=real_dfs,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
        )

        path_counts = [128, 512, 2048, 8192]
        prices: list[float] = []
        std_errors: list[float] = []
        bench_price: float | None = None

        for n in path_counts:
            res = price_consumer_price_index_option(
                model=black,
                strike_rate_ann=0.020,
                maturity_yrs=3.0,
                option_type=OptionType.CALL,
                notional=1000.0,
                n_paths=n,
                random_type=RandomSequenceType.SOBOL,
                seed=42,
            )
            prices.append(float(res["price"]))
            std_errors.append(float(res["std_error"]))
            if bench_price is None:
                bench_price = float(res["analytical_benchmark_price"])

        assert_path_convergence(
            path_counts=path_counts,
            std_errors=std_errors,
            prices=prices,
            benchmark_price=bench_price,
            min_decay_rate=0.35,
            num_std=3.5,
        )

    def test_zcis_path_convergence(self) -> None:
        """Verify Zero-Coupon Inflation Swap standard error decay and analytical convergence."""
        tenors, nom_dfs, real_dfs = get_standard_inflation_curve(
            base_cpi=100.0, expected_inflation_ann=0.02
        )
        black = BlackInflationModel(
            nominal_discount_curve_yrs=tenors,
            nominal_discount_factors=nom_dfs,
            real_discount_curve_yrs=tenors,
            real_discount_factors=real_dfs,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
        )

        path_counts = [128, 512, 2048, 8192]
        prices: list[float] = []
        std_errors: list[float] = []
        bench_price: float | None = None

        for n in path_counts:
            res = price_zero_coupon_inflation_swap(
                model=black,
                strike_rate_ann=0.020,
                maturity_yrs=5.0,
                notional=1000.0,
                n_paths=n,
                random_type=RandomSequenceType.SOBOL,
                seed=42,
            )
            prices.append(float(res["price"]))
            std_errors.append(float(res["std_error"]))
            if bench_price is None:
                bench_price = float(res["analytical_benchmark_price"])

        assert_path_convergence(
            path_counts=path_counts,
            std_errors=std_errors,
            prices=prices,
            benchmark_price=bench_price,
            min_decay_rate=0.35,
            num_std=3.5,
        )

    def test_cross_currency_swap_path_convergence(self) -> None:
        """Verify Cross-Currency Swap standard error decay and benchmark convergence."""
        tenors, dom_dfs = get_standard_discount_curve(flat_rate_ann=0.03)
        _, for_dfs = get_standard_discount_curve(flat_rate_ann=0.015)

        hw_dom = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=tenors,
            discount_factors=dom_dfs,
        )
        hw_for = HullWhite1FModel(
            a_ann=0.025,
            sigma_ann=0.008,
            discount_curve_yrs=tenors,
            discount_factors=for_dfs,
        )
        corr = np.array([
            [1.0, 0.3, -0.2],
            [0.3, 1.0, 0.1],
            [-0.2, 0.1, 1.0],
        ])
        two_ccy = TwoCurrencyFXModel(
            domestic_ir_model=hw_dom,
            foreign_ir_model=hw_for,
            spot_fx=1.25,
            fx_vol_ann=0.10,
            correlation_matrix=corr,
        )

        path_counts = [128, 512, 2048, 8192]
        prices: list[float] = []
        std_errors: list[float] = []
        bench_price: float | None = None

        for n in path_counts:
            res = price_cross_currency_swap(
                model=two_ccy,
                domestic_rate_ann=0.03,
                foreign_rate_ann=0.015,
                tenor_yrs=5.0,
                domestic_notional=1.25e6,
                foreign_notional=1.0e6,
                is_domestic_payer=True,
                n_paths=n,
                random_type=RandomSequenceType.SOBOL,
                seed=42,
            )
            prices.append(float(res["price"]))
            std_errors.append(float(res["std_error"]))
            if bench_price is None:
                bench_price = float(res["analytical_benchmark_price"])

        assert_path_convergence(
            path_counts=path_counts,
            std_errors=std_errors,
            prices=prices,
            benchmark_price=bench_price,
            min_decay_rate=0.35,
            num_std=3.5,
        )

    def test_cva_path_convergence(self) -> None:
        """Verify Portfolio CVA standard error decay across increasing path counts."""
        credit_tenors, spreads = get_standard_credit_curve()
        cir = CIRHazardRateModel.calibrate_from_spreads(
            credit_spreads_ann=spreads,
            tenors_yrs=credit_tenors,
        )

        sim_times = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
        n_times = len(sim_times)
        discount_factors = np.exp(-0.03 * sim_times)
        marginal_pds = cir.marginal_pd(sim_times)
        lgd = 0.4

        path_counts = [200, 800, 3200]
        cva_vals: list[float] = []
        std_errors: list[float] = []

        for n in path_counts:
            exposures = np.maximum(
                0.0,
                100.0
                + 15.0
                * np.random.default_rng(seed=42).standard_normal((n, n_times)),
            )
            cva_val = compute_cva(
                exposure=exposures,
                marginal_pd=marginal_pds,
                discount_factor=discount_factors,
                loss_given_default=lgd,
            )
            cva_vals.append(cva_val)
            pathwise_loss = lgd * np.sum(
                exposures * marginal_pds * discount_factors, axis=1
            )
            std_err = float(np.std(pathwise_loss, ddof=1) / np.sqrt(n))
            std_errors.append(std_err)

        assert_path_convergence(
            path_counts=path_counts,
            std_errors=std_errors,
            min_decay_rate=0.35,
        )
        self.assertGreater(cva_vals[-1], 0.0)

    def test_pseudo_vs_sobol_error_decay_comparison(self) -> None:
        """Verify that Sobol sequence achieves equal or lower standard error vs Pseudo across path counts."""
        gk = GarmanKohlhagenFXModel(
            spot_fx=1.0,
            fx_vol_ann=0.20,
            domestic_rate_ann=0.05,
            foreign_rate_ann=0.02,
        )

        path_counts = [256, 1024, 4096]
        pseudo_se: list[float] = []
        sobol_se: list[float] = []

        for n in path_counts:
            res_pseudo = price_foreign_exchange_option(
                params=gk,
                strike=1.0,
                maturity_yrs=1.0,
                notional=1.0,
                option_type=OptionType.CALL,
                n_paths=n,
                random_type=RandomSequenceType.PSEUDO,
                seed=42,
            )
            res_sobol = price_foreign_exchange_option(
                params=gk,
                strike=1.0,
                maturity_yrs=1.0,
                notional=1.0,
                option_type=OptionType.CALL,
                n_paths=n,
                random_type=RandomSequenceType.SOBOL,
                seed=42,
            )
            pseudo_se.append(float(res_pseudo["std_error"]))
            sobol_se.append(float(res_sobol["std_error"]))

        assert_path_convergence(path_counts=path_counts, std_errors=pseudo_se)
        assert_path_convergence(path_counts=path_counts, std_errors=sobol_se)

    def test_assert_path_convergence_validation_errors(self) -> None:
        """Verify error handling in assert_path_convergence helper."""
        with self.assertRaises(ValueError):
            assert_path_convergence([100], [0.05])

        with self.assertRaises(ValueError):
            assert_path_convergence([500, 200], [0.05, 0.02])

        with self.assertRaises(AssertionError):
            assert_path_convergence([100, 400], [0.01, 0.05])

        with self.assertRaises(AssertionError):
            assert_path_convergence(
                [100, 10000], [0.05, 0.049], min_decay_rate=0.40
            )



if __name__ == "__main__":
    unittest.main()
