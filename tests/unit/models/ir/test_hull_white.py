"""Tests for the Hull-White 1-Factor (HW1F) interest rate model."""

import unittest

import numpy as np

from xvasim.models.ir.hull_white import HullWhite1FModel, HullWhite1FParams
from xvasim.qmc import RandomSequenceType


class TestHullWhite1FModel(unittest.TestCase):
    """Unit tests for HullWhite1FModel initialization, zero coupon bond, and simulation."""

    def setUp(self) -> None:
        self.tenors = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 30.0])
        self.dfs = np.exp(-0.03 * self.tenors)
        self.model = HullWhite1FModel(
            a_ann=0.03,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )

    def test_init_with_params_dataclass(self) -> None:
        """Verify model initialization via pre-built HullWhite1FParams."""
        params = HullWhite1FParams(
            a_ann=0.04,
            sigma_ann=0.015,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )
        mdl = HullWhite1FModel(params=params)
        self.assertEqual(mdl.model_name, "hull_white")
        self.assertEqual(mdl.a_ann, 0.04)
        self.assertEqual(mdl.sigma_ann, 0.015)
        self.assertIs(mdl.params, params)
        np.testing.assert_array_equal(mdl.discount_curve_yrs, self.tenors)
        np.testing.assert_array_equal(mdl.discount_factors, self.dfs)

    def test_init_missing_arguments_raises(self) -> None:
        """Initializing without params or without discount curve raises ValueError."""
        with self.assertRaises(ValueError):
            HullWhite1FModel(a_ann=0.03)

    def test_b_function_and_zero_a_limit(self) -> None:
        """Verify B(t, T) with standard a and a=0 limit."""
        b_val = self.model.b_function(1.0, 3.0)
        expected = (1.0 - np.exp(-0.03 * 2.0)) / 0.03
        self.assertAlmostEqual(b_val, expected, places=10)

        # Zero a limit
        zero_a_model = HullWhite1FModel(
            a_ann=0.0,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )
        b_zero = zero_a_model.b_function(1.0, 3.0)
        self.assertAlmostEqual(b_zero, 2.0, places=10)

    def test_alpha_and_short_rate(self) -> None:
        """Verify deterministic drift alpha(t) and short rate r(t)."""
        alpha_t = self.model.alpha(1.0)
        self.assertGreater(alpha_t, 0.0)

        # Zero a limit for alpha
        zero_a_model = HullWhite1FModel(
            a_ann=0.0,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )
        alpha_zero = zero_a_model.alpha(1.0)
        self.assertGreater(alpha_zero, 0.0)

        # Short rate
        state = np.array([0.0, 0.01, -0.01])
        r = self.model.short_rate(1.0, state)
        np.testing.assert_allclose(r, state + alpha_t)

    def test_zero_coupon_bond(self) -> None:
        """Verify analytical zero-coupon bond pricing."""
        state = np.array([0.0, 0.01, -0.01])

        # Boundary maturity <= t
        zcb_expired = self.model.zero_coupon_bond(2.0, 2.0, state)
        np.testing.assert_allclose(zcb_expired, np.ones(3), rtol=1e-12)

        # Non-zero a
        zcb = self.model.zero_coupon_bond(1.0, 5.0, state)
        self.assertTrue(np.all(zcb > 0.0))
        self.assertTrue(np.all(zcb < 1.0))

        # Zero a model zero coupon bond
        zero_a_model = HullWhite1FModel(
            a_ann=0.0,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )
        zcb_zero_a = zero_a_model.zero_coupon_bond(1.0, 5.0, state)
        self.assertTrue(np.all(zcb_zero_a > 0.0))

    def test_discount_path(self) -> None:
        """Verify path-wise discount factor computation."""
        times = np.array([0.0, 0.5, 1.0, 2.0])
        n_paths = 10
        state_paths = np.zeros((n_paths, len(times)))
        disc_paths = self.model.discount_path(times, state_paths)
        self.assertEqual(disc_paths.shape, (n_paths, len(times)))
        np.testing.assert_allclose(disc_paths[:, 0], 1.0)
        for step in range(len(times) - 1):
            self.assertTrue(np.all(disc_paths[:, step + 1] < disc_paths[:, step]))

    def test_simulate_paths(self) -> None:
        """Verify Monte Carlo simulation with pseudo, custom dw, and zero a."""
        times = np.array([0.0, 0.5, 1.0, 2.0])
        n_paths = 50

        paths_pseudo = self.model.simulate_paths(
            times=times,
            n_paths=n_paths,
            random_type=RandomSequenceType.PSEUDO,
            seed=42,
        )
        self.assertEqual(paths_pseudo.shape, (n_paths, len(times)))
        np.testing.assert_allclose(paths_pseudo[:, 0], 0.0)

        # Custom increments
        dw = np.random.default_rng(123).normal(0.0, 0.5, size=(n_paths, len(times) - 1))
        paths_custom = self.model.simulate_paths(
            times=times,
            n_paths=n_paths,
            dw=dw,
        )
        self.assertEqual(paths_custom.shape, (n_paths, len(times)))

        # Zero a simulation
        zero_a_model = HullWhite1FModel(
            a_ann=0.0,
            sigma_ann=0.01,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )
        paths_zero_a = zero_a_model.simulate_paths(
            times=times,
            n_paths=n_paths,
            seed=42,
        )
        self.assertEqual(paths_zero_a.shape, (n_paths, len(times)))


if __name__ == "__main__":
    unittest.main()
