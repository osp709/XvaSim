"""Integration tests for joint multi-asset stochastic simulations (IR + FX + Inflation)."""

import unittest

import numpy as np
import pytest

from xvasim.models.fx.two_currency import TwoCurrencyFXModel
from xvasim.models.inflation.jarrow_yildirim import JarrowYildirimModel
from xvasim.models.ir.hull_white import HullWhite1FModel
from xvasim.models.ir.lgm import LGMModel
from xvasim.qmc import RandomSequenceType


@pytest.mark.integration
class TestCrossModelSimulation(unittest.TestCase):
    """Integration test suite for joint multi-model simulation dynamics."""

    def test_multi_currency_ir_fx_simulation(self) -> None:
        """Verify joint simulation of domestic LGM, foreign Hull-White, and Spot FX."""
        tenors = np.array([0.0, 1.0, 5.0, 10.0])
        dfs_dom = np.exp(-0.03 * tenors)
        dfs_for = np.exp(-0.015 * tenors)

        dom_lgm = LGMModel(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([10.0]),
            sigma_values_ann=np.array([0.01]),
            discount_curve_yrs=tenors,
            discount_factors=dfs_dom,
        )
        for_hw = HullWhite1FModel(
            a_ann=0.04,
            sigma_ann=0.012,
            discount_curve_yrs=tenors,
            discount_factors=dfs_for,
        )
        corr = np.array(
            [
                [1.0, 0.3, 0.2],
                [0.3, 1.0, -0.1],
                [0.2, -0.1, 1.0],
            ]
        )

        fx_model = TwoCurrencyFXModel(
            domestic_ir_model=dom_lgm,
            foreign_ir_model=for_hw,
            spot_fx=1.15,
            fx_vol_ann=0.12,
            correlation_matrix=corr,
        )

        times, x_dom, x_for, fx_spot = fx_model.simulate_paths(
            maturity_yrs=5.0,
            n_paths=100,
            n_steps=10,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )

        self.assertEqual(len(times), 11)
        self.assertEqual(x_dom.shape, (100, 11))
        self.assertEqual(x_for.shape, (100, 11))
        self.assertEqual(fx_spot.shape, (100, 11))
        self.assertTrue(np.all(fx_spot > 0.0))

    def test_two_economy_inflation_simulation(self) -> None:
        """Verify joint simulation of nominal, real rates, and CPI index."""
        tenors = np.array([0.0, 1.0, 5.0, 10.0])
        nom_dfs = np.exp(-0.035 * tenors)
        real_dfs = np.exp(-0.015 * tenors)

        nom_hw = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=tenors,
            discount_factors=nom_dfs,
        )
        real_hw = HullWhite1FModel(
            a_ann=0.02,
            sigma_ann=0.008,
            discount_curve_yrs=tenors,
            discount_factors=real_dfs,
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
            cpi_vol_ann=0.025,
            correlation_matrix=corr,
        )

        res = jy_model.simulate_paths(
            maturity_yrs=5.0,
            n_paths=100,
            n_steps=10,
            random_type=RandomSequenceType.SOBOL,
            seed=123,
        )

        self.assertEqual(res.cpi_index.shape, (100, 11))
        self.assertEqual(res.nominal_states.shape, (100, 11))
        self.assertEqual(res.real_states.shape, (100, 11))
        self.assertTrue(np.all(res.cpi_index > 0.0))


if __name__ == "__main__":
    unittest.main()
