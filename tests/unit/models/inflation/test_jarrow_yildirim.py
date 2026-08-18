"""Tests for the Jarrow-Yildirim Two-Economy inflation model."""

import unittest

import numpy as np

from xvasim.models.inflation.jarrow_yildirim import (
    InflationSimulationResult,
    JarrowYildirimModel,
)
from xvasim.models.ir.hull_white import HullWhite1FModel
from xvasim.models.ir.lgm import LGMModel, LGMParams
from xvasim.qmc import RandomSequenceType


class TestJarrowYildirimModel(unittest.TestCase):
    """Unit tests for JarrowYildirimModel initialization, analytical variance, swap pricing, and simulation."""

    def setUp(self) -> None:
        self.tenors = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 30.0])
        self.nom_dfs = np.exp(-0.035 * self.tenors)
        self.real_dfs = np.exp(-0.015 * self.tenors)

        self.nom_hw = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.nom_dfs,
        )
        self.real_hw = HullWhite1FModel(
            a_ann=0.02,
            sigma_ann=0.008,
            discount_curve_yrs=self.tenors,
            discount_factors=self.real_dfs,
        )
        self.corr = np.array(
            [
                [1.0, 0.4, 0.2],
                [0.4, 1.0, -0.1],
                [0.2, -0.1, 1.0],
            ]
        )

        self.model = JarrowYildirimModel(
            nominal_ir_model=self.nom_hw,
            real_ir_model=self.real_hw,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
            correlation_matrix=self.corr,
        )

    def test_init_validation(self) -> None:
        """Invalid base_cpi, cpi_vol, or correlation matrix raises ValueError."""
        with self.assertRaises(ValueError):
            JarrowYildirimModel(
                nominal_ir_model=self.nom_hw,
                real_ir_model=self.real_hw,
                base_cpi=-10.0,
                cpi_vol_ann=0.02,
                correlation_matrix=self.corr,
            )
        with self.assertRaises(ValueError):
            JarrowYildirimModel(
                nominal_ir_model=self.nom_hw,
                real_ir_model=self.real_hw,
                base_cpi=100.0,
                cpi_vol_ann=-0.02,
                correlation_matrix=self.corr,
            )
        with self.assertRaises(ValueError):
            JarrowYildirimModel(
                nominal_ir_model=self.nom_hw,
                real_ir_model=self.real_hw,
                base_cpi=100.0,
                cpi_vol_ann=0.02,
                correlation_matrix=np.eye(2),
            )

    def test_properties(self) -> None:
        """Verify model properties and getters."""
        self.assertEqual(self.model.model_name, "jarrow_yildirim")
        self.assertEqual(self.model.base_cpi, 100.0)
        self.assertEqual(self.model.cpi_vol_ann, 0.02)
        self.assertIs(self.model.nominal_ir_model, self.nom_hw)
        self.assertIs(self.model.real_ir_model, self.real_hw)
        np.testing.assert_array_equal(self.model.correlation_matrix, self.corr)
        np.testing.assert_array_equal(
            self.model.nominal_ir_model.discount_curve_yrs, self.tenors
        )
        np.testing.assert_array_equal(
            self.model.nominal_ir_model.discount_factors, self.nom_dfs
        )
        np.testing.assert_array_equal(
            self.model.real_ir_model.discount_curve_yrs, self.tenors
        )
        np.testing.assert_array_equal(
            self.model.real_ir_model.discount_factors, self.real_dfs
        )

    def test_from_lgm_params(self) -> None:
        """Verify constructor from LGMParams, from_ir_models, and from_components."""
        nom_p = LGMParams(
            0.03, np.array([30.0]), np.array([0.01]), self.tenors, self.nom_dfs
        )
        real_p = LGMParams(
            0.02, np.array([30.0]), np.array([0.008]), self.tenors, self.real_dfs
        )
        mdl = JarrowYildirimModel.from_lgm_params(
            nominal=nom_p,
            real=real_p,
            base_cpi=105.0,
            cpi_vol_ann=0.025,
            correlation_matrix=self.corr,
        )
        self.assertIsInstance(mdl, JarrowYildirimModel)
        self.assertEqual(mdl.base_cpi, 105.0)

        # from_ir_models with LGMParams
        mdl_ir = JarrowYildirimModel.from_ir_models(
            nominal=nom_p,
            real=real_p,
            base_cpi=105.0,
            cpi_vol_ann=0.025,
            correlation_matrix=self.corr,
        )
        self.assertIsInstance(mdl_ir, JarrowYildirimModel)
        self.assertIsInstance(mdl_ir.nominal_ir_model, LGMModel)

        # from_components with InterestRateModel instances
        mdl_comp = JarrowYildirimModel.from_components(
            nominal=self.nom_hw,
            real=self.real_hw,
            base_cpi=105.0,
            cpi_vol_ann=0.025,
            correlation_matrix=self.corr,
        )
        self.assertIs(mdl_comp.nominal_ir_model, self.nom_hw)
        self.assertIs(mdl_comp.real_ir_model, self.real_hw)

    def test_forward_cpi_and_swap_rate(self) -> None:
        """Verify forward CPI and fair swap rate calculation."""
        fwd_5y = self.model.forward_cpi(5.0)
        expected_fwd = 100.0 * (np.exp(-0.015 * 5.0) / np.exp(-0.035 * 5.0))
        self.assertAlmostEqual(fwd_5y, expected_fwd, places=10)

        swap_rate_5y = self.model.zero_coupon_inflation_swap_rate(5.0)
        self.assertGreater(swap_rate_5y, 0.0)

        self.assertEqual(self.model.zero_coupon_inflation_swap_rate(0.0), 0.0)

    def test_total_variance_at_hw_and_lgm(self) -> None:
        """Verify analytical total variance integration for HW and LGM components."""
        var_5y_hw = self.model.total_variance_at(5.0)
        self.assertGreater(var_5y_hw, 0.0)

        # Zero maturity
        self.assertEqual(self.model.total_variance_at(0.0), 0.0)

        # LGM components
        nom_lgm = LGMModel(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([10.0]),
            sigma_values_ann=np.array([0.01]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.nom_dfs,
        )
        real_lgm = LGMModel(
            kappa_ann=0.02,
            sigma_grid_yrs=np.array([10.0]),
            sigma_values_ann=np.array([0.008]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.real_dfs,
        )
        lgm_jy = JarrowYildirimModel(
            nominal_ir_model=nom_lgm,
            real_ir_model=real_lgm,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
            correlation_matrix=self.corr,
        )
        var_5y_lgm = lgm_jy.total_variance_at(5.0)
        self.assertGreater(var_5y_lgm, 0.0)

    def test_simulate_paths(self) -> None:
        """Verify 3-factor path simulation under Jarrow-Yildirim."""
        res = self.model.simulate_paths(
            maturity_yrs=2.0,
            n_paths=50,
            n_steps=4,
            random_type=RandomSequenceType.PSEUDO,
            seed=42,
        )
        self.assertIsInstance(res, InflationSimulationResult)
        self.assertEqual(len(res.times), 5)
        self.assertEqual(res.nominal_states.shape, (50, 5))
        self.assertEqual(res.real_states.shape, (50, 5))
        self.assertEqual(res.cpi_index.shape, (50, 5))
        np.testing.assert_allclose(res.cpi_index[:, 0], 100.0)
        self.assertTrue(np.all(res.cpi_index > 0.0))

        # Test tuple unpacking, indexing and length
        t_arr, x_n, x_r, cpi_arr = res
        self.assertEqual(len(res), 4)
        np.testing.assert_array_equal(t_arr, res[0])
        np.testing.assert_array_equal(x_n, res[1])
        np.testing.assert_array_equal(x_r, res[2])
        np.testing.assert_array_equal(cpi_arr, res[3])

    def test_simulate_paths_with_lgm_models(self) -> None:
        """Verify simulation when nominal and real rates are LGM models."""
        nom_lgm = LGMModel(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([10.0]),
            sigma_values_ann=np.array([0.01]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.nom_dfs,
        )
        real_lgm = LGMModel(
            kappa_ann=0.02,
            sigma_grid_yrs=np.array([10.0]),
            sigma_values_ann=np.array([0.008]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.real_dfs,
        )
        lgm_jy = JarrowYildirimModel(
            nominal_ir_model=nom_lgm,
            real_ir_model=real_lgm,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
            correlation_matrix=None,  # Tests default correlation identity matrix
        )
        self.assertEqual(lgm_jy.correlation_matrix.shape, (3, 3))
        res = lgm_jy.simulate_paths(
            maturity_yrs=1.0,
            n_paths=30,
            n_steps=4,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        self.assertEqual(res.nominal_states.shape, (30, 5))
        self.assertEqual(res.real_states.shape, (30, 5))
        self.assertEqual(res.cpi_index.shape, (30, 5))


if __name__ == "__main__":
    unittest.main()
