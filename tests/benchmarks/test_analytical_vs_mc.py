"""Statistical benchmark tests comparing Monte Carlo convergence vs analytical solutions."""

import unittest

import numpy as np
import pytest

from tests.helpers.assertions import assert_mc_within_bounds
from xvasim.models.fx.garman_kohlhagen import GarmanKohlhagenFXModel
from xvasim.models.inflation.black_inflation import BlackInflationModel
from xvasim.models.ir.hull_white import HullWhite1FModel
from xvasim.pricing_engine import (
    OptionType,
    price_consumer_price_index_option,
    price_foreign_exchange_option,
    price_interest_rate_swap,
    price_zero_coupon_inflation_swap,
)
from xvasim.qmc import RandomSequenceType


@pytest.mark.benchmark
class TestAnalyticalVsMonteCarloBenchmarks(unittest.TestCase):
    """Convergence benchmark tests verifying Monte Carlo convergence within 3.5 std errors."""

    def test_irs_convergence_benchmark(self) -> None:
        """Verify IRS MC convergence vs exact analytical benchmark."""
        tenors = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
        dfs = np.exp(-0.03 * tenors)
        hw = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=tenors,
            discount_factors=dfs,
        )

        res = price_interest_rate_swap(
            model=hw,
            fixed_rate_ann=0.03,
            tenor_yrs=5.0,
            is_payer=True,
            n_paths=500,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        assert_mc_within_bounds(
            res["price"],
            res["std_error"],
            res["analytical_benchmark_price"],
            num_std=3.5,
        )

    def test_fx_option_convergence_benchmark(self) -> None:
        """Verify FX option MC convergence vs Garman-Kohlhagen closed form."""
        gk = GarmanKohlhagenFXModel(
            spot_fx=1.15,
            fx_vol_ann=0.12,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.015,
        )
        res = price_foreign_exchange_option(
            params=gk,
            strike=1.15,
            maturity_yrs=1.0,
            notional=1.0,
            option_type=OptionType.CALL,
            n_paths=500,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        assert_mc_within_bounds(
            float(res["price"]),
            float(res["std_error"]),
            float(res["analytical_benchmark_price"]),
            num_std=3.5,
        )

    def test_cpi_option_convergence_benchmark(self) -> None:
        """Verify CPI option MC convergence vs Black analytical formula."""
        tenors = np.array([0.0, 1.0, 5.0])
        nom_dfs = np.exp(-0.035 * tenors)
        real_dfs = np.exp(-0.015 * tenors)
        black = BlackInflationModel(
            nominal_discount_curve_yrs=tenors,
            nominal_discount_factors=nom_dfs,
            real_discount_curve_yrs=tenors,
            real_discount_factors=real_dfs,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
        )

        res = price_consumer_price_index_option(
            model=black,
            strike_rate_ann=0.020,
            maturity_yrs=5.0,
            option_type=OptionType.CALL,
            notional=1000.0,
            n_paths=500,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        assert_mc_within_bounds(
            res["price"],
            res["std_error"],
            res["analytical_benchmark_price"],
            num_std=3.5,
        )

    def test_zcis_convergence_benchmark(self) -> None:
        """Verify Zero-Coupon Inflation Swap MC convergence vs analytical benchmark."""
        tenors = np.array([0.0, 1.0, 5.0])
        nom_dfs = np.exp(-0.035 * tenors)
        real_dfs = np.exp(-0.015 * tenors)
        black = BlackInflationModel(
            nominal_discount_curve_yrs=tenors,
            nominal_discount_factors=nom_dfs,
            real_discount_curve_yrs=tenors,
            real_discount_factors=real_dfs,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
        )

        res = price_zero_coupon_inflation_swap(
            model=black,
            strike_rate_ann=0.020,
            maturity_yrs=5.0,
            notional=1000.0,
            n_paths=500,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        assert_mc_within_bounds(
            res["price"],
            res["std_error"],
            res["analytical_benchmark_price"],
            num_std=3.5,
        )


if __name__ == "__main__":
    unittest.main()
