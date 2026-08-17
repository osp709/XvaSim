"""Tests for the Two-Currency modular FX model."""

import unittest

import numpy as np

from xvasim.models.fx.two_currency import TwoCurrencyFXModel
from xvasim.models.ir.hull_white import HullWhite1FModel
from xvasim.models.ir.lgm import LGMModel, LGMParams
from xvasim.qmc import RandomSequenceType


class TestTwoCurrencyFXModel(unittest.TestCase):
    """Unit tests for TwoCurrencyFXModel initialization, validation, and multi-asset simulation."""

    def setUp(self) -> None:
        self.tenors = np.array([0.0, 1.0, 5.0, 10.0])
        self.dfs_dom = np.exp(-0.03 * self.tenors)
        self.dfs_for = np.exp(-0.02 * self.tenors)

        self.dom_model = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs_dom,
        )
        self.for_model = HullWhite1FModel(
            a_ann=0.04,
            sigma_ann=0.015,
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

        self.model = TwoCurrencyFXModel(
            domestic_ir_model=self.dom_model,
            foreign_ir_model=self.for_model,
            spot_fx=1.15,
            fx_vol_ann=0.12,
            correlation_matrix=self.corr,
        )

    def test_properties(self) -> None:
        """Verify model properties and getters."""
        self.assertEqual(self.model.model_name, "two_currency")
        self.assertEqual(self.model.spot_fx, 1.15)
        self.assertEqual(self.model.fx_vol_ann, 0.12)
        self.assertIs(self.model.domestic_ir_model, self.dom_model)
        self.assertIs(self.model.foreign_ir_model, self.for_model)
        np.testing.assert_array_equal(self.model.correlation_matrix, self.corr)

    def test_invalid_correlation_matrix_raises(self) -> None:
        """Invalid correlation matrix shape raises ValueError."""
        with self.assertRaises(ValueError):
            TwoCurrencyFXModel(
                domestic_ir_model=self.dom_model,
                foreign_ir_model=self.for_model,
                spot_fx=1.15,
                fx_vol_ann=0.12,
                correlation_matrix=np.eye(2),
            )

    def test_from_lgm_params(self) -> None:
        """Verify factory constructor from LGMParams."""
        dom_p = LGMParams(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([5.0]),
            sigma_values_ann=np.array([0.01]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs_dom,
        )
        for_p = LGMParams(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([5.0]),
            sigma_values_ann=np.array([0.012]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs_for,
        )
        mdl = TwoCurrencyFXModel.from_lgm_params(
            domestic=dom_p,
            foreign=for_p,
            spot_fx=1.20,
            fx_vol_ann=0.10,
            correlation_matrix=self.corr,
        )
        self.assertIsInstance(mdl.domestic_ir_model, LGMModel)
        self.assertIsInstance(mdl.foreign_ir_model, LGMModel)

    def test_simulate_paths_hw(self) -> None:
        """Verify path simulation with Hull-White interest rate models."""
        times, x_dom, x_for, fx_spot = self.model.simulate_paths(
            maturity_yrs=2.0,
            n_paths=50,
            n_steps=4,
            random_type=RandomSequenceType.PSEUDO,
            seed=42,
        )
        self.assertEqual(len(times), 5)
        self.assertEqual(x_dom.shape, (50, 5))
        self.assertEqual(x_for.shape, (50, 5))
        self.assertEqual(fx_spot.shape, (50, 5))
        np.testing.assert_allclose(fx_spot[:, 0], 1.15)
        self.assertTrue(np.all(fx_spot > 0.0))

    def test_simulate_paths_lgm(self) -> None:
        """Verify path simulation with LGM interest rate models."""
        dom_lgm = LGMModel(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([5.0]),
            sigma_values_ann=np.array([0.01]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs_dom,
        )
        for_lgm = LGMModel(
            kappa_ann=0.04,
            sigma_grid_yrs=np.array([5.0]),
            sigma_values_ann=np.array([0.012]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs_for,
        )
        lgm_fx = TwoCurrencyFXModel(
            domestic_ir_model=dom_lgm,
            foreign_ir_model=for_lgm,
            spot_fx=1.10,
            fx_vol_ann=0.14,
            correlation_matrix=self.corr,
        )
        _times, _x_dom, _x_for, fx_spot = lgm_fx.simulate_paths(
            maturity_yrs=1.0,
            n_paths=30,
            n_steps=4,
            seed=123,
        )
        self.assertEqual(fx_spot.shape, (30, 5))
        np.testing.assert_allclose(fx_spot[:, 0], 1.10)


if __name__ == "__main__":
    unittest.main()
