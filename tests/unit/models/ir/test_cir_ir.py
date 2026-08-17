"""Tests for the Cox-Ingersoll-Ross (CIR) interest rate model."""

import unittest

import numpy as np

from xvasim.models.ir.cir import CIRInterestRateModel, CIRInterestRateParams
from xvasim.qmc import RandomSequenceType


class TestCIRInterestRateModel(unittest.TestCase):
    """Unit tests for CIRInterestRateModel initialization, bond pricing, non-negativity, and simulation."""

    def setUp(self) -> None:
        self.model = CIRInterestRateModel(
            kappa_ann=0.20,
            theta_ann=0.03,
            sigma_ann=0.08,
            r0_ann=0.025,
        )

    def test_init_with_params_dataclass(self) -> None:
        """Verify model initialization via pre-built CIRInterestRateParams."""
        params = CIRInterestRateParams(
            kappa_ann=0.30,
            theta_ann=0.04,
            sigma_ann=0.09,
            r0_ann=0.035,
        )
        mdl = CIRInterestRateModel(params=params)
        self.assertEqual(mdl.model_name, "cir")
        self.assertEqual(mdl.kappa_ann, 0.30)
        self.assertEqual(mdl.theta_ann, 0.04)
        self.assertEqual(mdl.sigma_ann, 0.09)
        self.assertEqual(mdl.r0_ann, 0.035)
        self.assertIs(mdl.params, params)
        self.assertGreater(len(mdl.discount_curve_yrs), 0)
        self.assertGreater(len(mdl.discount_factors), 0)

    def test_init_with_explicit_discount_curve(self) -> None:
        """Verify initializing with user-specified discount curve."""
        curve_yrs = np.array([0.0, 1.0, 5.0])
        curve_dfs = np.array([1.0, 0.97, 0.85])
        mdl = CIRInterestRateModel(
            kappa_ann=0.20,
            theta_ann=0.03,
            sigma_ann=0.08,
            r0_ann=0.025,
            discount_curve_yrs=curve_yrs,
            discount_factors=curve_dfs,
        )
        np.testing.assert_array_equal(mdl.discount_curve_yrs, curve_yrs)
        np.testing.assert_array_equal(mdl.discount_factors, curve_dfs)

    def test_analytical_bond_pricing(self) -> None:
        """Verify analytical CIR zero-coupon bond pricing."""
        zcb_1y = self.model.zero_coupon_bond(0.0, 1.0, np.array([0.025]))
        self.assertGreater(float(zcb_1y[0]), 0.0)
        self.assertLess(float(zcb_1y[0]), 1.0)

        # Non-negative clamping on state
        zcb_neg_state = self.model.zero_coupon_bond(0.0, 1.0, np.array([-0.01]))
        zcb_zero_state = self.model.zero_coupon_bond(0.0, 1.0, np.array([0.0]))
        np.testing.assert_allclose(zcb_neg_state, zcb_zero_state)

    def test_short_rate_and_discount_path(self) -> None:
        """Verify non-negative short rate clamping and discount path calculation."""
        state = np.array([-0.02, 0.0, 0.03])
        r = self.model.short_rate(1.0, state)
        np.testing.assert_allclose(r, np.array([0.0, 0.0, 0.03]))

        times = np.array([0.0, 0.5, 1.0])
        state_paths = np.ones((5, 3)) * 0.03
        disc = self.model.discount_path(times, state_paths)
        self.assertEqual(disc.shape, (5, 3))
        np.testing.assert_allclose(disc[:, 0], 1.0)
        for step in range(len(times) - 1):
            self.assertTrue(np.all(disc[:, step + 1] <= disc[:, step]))

    def test_simulate_paths(self) -> None:
        """Verify path simulation non-negativity and custom increments."""
        times = np.array([0.0, 0.5, 1.0, 2.0])
        n_paths = 100

        paths_pseudo = self.model.simulate_paths(
            times=times,
            n_paths=n_paths,
            random_type=RandomSequenceType.PSEUDO,
            seed=42,
        )
        self.assertEqual(paths_pseudo.shape, (n_paths, len(times)))
        np.testing.assert_allclose(paths_pseudo[:, 0], 0.025)
        self.assertTrue(np.all(paths_pseudo >= 0.0))

        # Custom increments
        dw = np.random.default_rng(123).normal(0.0, 0.5, size=(n_paths, len(times) - 1))
        paths_custom = self.model.simulate_paths(
            times=times,
            n_paths=n_paths,
            dw=dw,
        )
        self.assertEqual(paths_custom.shape, (n_paths, len(times)))
        self.assertTrue(np.all(paths_custom >= 0.0))


if __name__ == "__main__":
    unittest.main()
