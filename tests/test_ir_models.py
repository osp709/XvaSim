"""Tests for modular interest rate models."""

import unittest

import numpy as np

from xvasim.models.ir import (
    CIRInterestRateModel,
    HullWhite1FModel,
    HullWhite1FParams,
    LGMModel,
    LGMParams,
    VasicekModel,
)


class TestLGMModel(unittest.TestCase):
    def setUp(self) -> None:
        self.tenors = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 30.0])
        self.dfs = np.exp(-0.03 * self.tenors)
        self.lgm = LGMModel(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([30.0]),
            sigma_values_ann=np.array([0.01]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )

    def test_initialization_from_params(self) -> None:
        params = LGMParams(
            kappa_ann=0.04,
            sigma_grid_yrs=np.array([10.0]),
            sigma_values_ann=np.array([0.012]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )
        model = LGMModel(params=params)
        self.assertEqual(model.kappa_ann, 0.04)
        self.assertEqual(model.model_name, "lgm")

    def test_h_function(self) -> None:
        h0 = self.lgm.h_function(0.0)
        self.assertAlmostEqual(float(h0), 0.0)
        h1 = self.lgm.h_function(1.0)
        expected = (1.0 - np.exp(-0.03)) / 0.03
        self.assertAlmostEqual(float(h1), expected, places=10)

    def test_zero_coupon_bond(self) -> None:
        """Bond price at t=0 should match the initial discount curve."""
        state_zero = np.zeros(1)
        for t_mat in [1.0, 2.0, 5.0]:
            p = self.lgm.zero_coupon_bond(0.0, t_mat, state_zero)
            expected = float(self.lgm.interpolate_discount_factor(t_mat))
            self.assertAlmostEqual(float(p[0]), expected, places=10)

    def test_simulate_paths_shape(self) -> None:
        times = np.linspace(0.0, 5.0, 51)
        rng = np.random.default_rng(42)
        paths = self.lgm.simulate_paths(times, n_paths=100, rng=rng)
        self.assertEqual(paths.shape, (100, 51))
        np.testing.assert_allclose(paths[:, 0], 0.0)

    def test_discount_path(self) -> None:
        times = np.linspace(0.0, 5.0, 11)
        paths = np.zeros((10, 11))
        df_paths = self.lgm.discount_path(times, paths)
        self.assertEqual(df_paths.shape, (10, 11))
        np.testing.assert_allclose(df_paths[:, 0], 1.0)
        # Should be monotonically decreasing
        for i in range(10):
            self.assertTrue(np.all(np.diff(df_paths[i, :]) < 0))


class TestHullWhite1FModel(unittest.TestCase):
    def setUp(self) -> None:
        self.tenors = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
        self.dfs = np.exp(-0.03 * self.tenors)
        self.hw = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )

    def test_initialization_from_params(self) -> None:
        params = HullWhite1FParams(
            a_ann=0.05,
            sigma_ann=0.015,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )
        model = HullWhite1FModel(params=params)
        self.assertEqual(model.a_ann, 0.05)
        self.assertEqual(model.sigma_ann, 0.015)
        self.assertEqual(model.model_name, "hull_white")

    def test_b_function(self) -> None:
        b0 = self.hw.b_function(1.0, 1.0)
        self.assertAlmostEqual(b0, 0.0)
        b_val = self.hw.b_function(1.0, 2.0)
        expected = (1.0 - np.exp(-0.03 * 1.0)) / 0.03
        self.assertAlmostEqual(b_val, expected, places=10)

    def test_exact_term_structure_matching(self) -> None:
        """P(0, T) under Hull-White with x(0)=0 should match curve DFs."""
        state_zero = np.zeros(1)
        for t_mat in [0.5, 1.0, 2.0, 5.0, 10.0]:
            p = self.hw.zero_coupon_bond(0.0, t_mat, state_zero)
            expected = float(self.hw.interpolate_discount_factor(t_mat))
            self.assertAlmostEqual(float(p[0]), expected, places=9)

    def test_simulate_paths(self) -> None:
        times = np.linspace(0.0, 2.0, 21)
        rng = np.random.default_rng(123)
        paths = self.hw.simulate_paths(times, n_paths=1000, rng=rng)
        self.assertEqual(paths.shape, (1000, 21))
        # Mean of x(t) should remain close to 0
        np.testing.assert_allclose(np.mean(paths[:, -1]), 0.0, atol=0.01)

    def test_discount_path(self) -> None:
        times = np.linspace(0.0, 3.0, 31)
        paths = np.zeros((5, 31))
        dfs = self.hw.discount_path(times, paths)
        self.assertEqual(dfs.shape, (5, 31))
        np.testing.assert_allclose(dfs[:, 0], 1.0)
        for i in range(5):
            self.assertTrue(np.all(np.diff(dfs[i, :]) < 0))


class TestVasicekModel(unittest.TestCase):
    def setUp(self) -> None:
        self.vas = VasicekModel(
            kappa_ann=0.15,
            theta_ann=0.03,
            sigma_ann=0.01,
            r0_ann=0.02,
        )

    def test_initialization(self) -> None:
        self.assertEqual(self.vas.kappa_ann, 0.15)
        self.assertEqual(self.vas.theta_ann, 0.03)
        self.assertEqual(self.vas.r0_ann, 0.02)
        self.assertEqual(self.vas.model_name, "vasicek")

    def test_bond_price_at_zero(self) -> None:
        p0 = self.vas.zero_coupon_bond(0.0, 0.0, np.array([0.02]))
        self.assertAlmostEqual(float(p0[0]), 1.0)

    def test_mean_reversion(self) -> None:
        """Long simulation horizon should mean-revert toward theta."""
        times = np.linspace(0.0, 50.0, 500)
        rng = np.random.default_rng(42)
        paths = self.vas.simulate_paths(times, n_paths=5000, rng=rng)
        mean_terminal = float(np.mean(paths[:, -1]))
        self.assertAlmostEqual(mean_terminal, self.vas.theta_ann, delta=0.005)


class TestCIRInterestRateModel(unittest.TestCase):
    def setUp(self) -> None:
        self.cir = CIRInterestRateModel(
            kappa_ann=0.20,
            theta_ann=0.03,
            sigma_ann=0.05,
            r0_ann=0.025,
        )

    def test_initialization(self) -> None:
        self.assertEqual(self.cir.kappa_ann, 0.20)
        self.assertEqual(self.cir.theta_ann, 0.03)
        self.assertEqual(self.cir.model_name, "cir")

    def test_bond_price(self) -> None:
        p = self.cir.zero_coupon_bond(0.0, 1.0, np.array([0.025]))
        self.assertGreater(float(p[0]), 0.0)
        self.assertLess(float(p[0]), 1.0)

    def test_non_negativity_in_simulation(self) -> None:
        """Simulated rates must stay non-negative."""
        times = np.linspace(0.0, 5.0, 100)
        rng = np.random.default_rng(99)
        paths = self.cir.simulate_paths(times, n_paths=500, rng=rng)
        self.assertTrue(np.all(paths >= 0.0))


if __name__ == "__main__":
    unittest.main()
