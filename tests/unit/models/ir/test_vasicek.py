"""Tests for the Vasicek interest rate model."""

import unittest

import numpy as np

from xvasim.models.ir.vasicek import VasicekModel, VasicekParams
from xvasim.qmc import RandomSequenceType


class TestVasicekModel(unittest.TestCase):
    """Unit tests for VasicekModel initialization, bond pricing, and path simulation."""

    def setUp(self) -> None:
        self.model = VasicekModel(
            kappa_ann=0.15,
            theta_ann=0.03,
            sigma_ann=0.015,
            r0_ann=0.025,
        )

    def test_init_with_params_dataclass(self) -> None:
        """Verify model initialization via pre-built VasicekParams."""
        params = VasicekParams(
            kappa_ann=0.20,
            theta_ann=0.04,
            sigma_ann=0.02,
            r0_ann=0.03,
        )
        mdl = VasicekModel(params=params)
        self.assertEqual(mdl.model_name, "vasicek")
        self.assertEqual(mdl.kappa_ann, 0.20)
        self.assertEqual(mdl.theta_ann, 0.04)
        self.assertEqual(mdl.sigma_ann, 0.02)
        self.assertEqual(mdl.r0_ann, 0.03)
        self.assertIs(mdl.params, params)
        self.assertGreater(len(mdl.discount_curve_yrs), 0)
        self.assertGreater(len(mdl.discount_factors), 0)

    def test_init_with_explicit_discount_curve(self) -> None:
        """Verify initializing with user-specified discount curve."""
        curve_yrs = np.array([0.0, 1.0, 5.0])
        curve_dfs = np.array([1.0, 0.97, 0.85])
        mdl = VasicekModel(
            kappa_ann=0.15,
            theta_ann=0.03,
            sigma_ann=0.015,
            r0_ann=0.025,
            discount_curve_yrs=curve_yrs,
            discount_factors=curve_dfs,
        )
        np.testing.assert_array_equal(mdl.discount_curve_yrs, curve_yrs)
        np.testing.assert_array_equal(mdl.discount_factors, curve_dfs)

    def test_analytical_bond_pricing(self) -> None:
        """Verify analytical bond price formula for standard and zero-kappa cases."""
        zcb_1y = self.model.zero_coupon_bond(0.0, 1.0, np.array([0.025]))
        self.assertGreater(float(zcb_1y[0]), 0.0)
        self.assertLess(float(zcb_1y[0]), 1.0)

        # Zero kappa limit
        zero_kappa_model = VasicekModel(
            kappa_ann=0.0,
            theta_ann=0.03,
            sigma_ann=0.015,
            r0_ann=0.025,
        )
        zcb_zero_kappa = zero_kappa_model.zero_coupon_bond(0.0, 1.0, np.array([0.025]))
        self.assertGreater(float(zcb_zero_kappa[0]), 0.0)

    def test_short_rate_and_discount_path(self) -> None:
        """Verify short rate and path-wise discount factor computation."""
        state = np.array([0.02, 0.03, 0.04])
        r = self.model.short_rate(1.0, state)
        np.testing.assert_array_equal(r, state)

        times = np.array([0.0, 0.5, 1.0])
        state_paths = np.ones((5, 3)) * 0.03
        disc = self.model.discount_path(times, state_paths)
        self.assertEqual(disc.shape, (5, 3))
        np.testing.assert_allclose(disc[:, 0], 1.0)
        np.testing.assert_allclose(disc[:, 2], np.exp(-0.03 * 1.0), rtol=1e-10)

    def test_simulate_paths(self) -> None:
        """Verify Monte Carlo simulation with pseudo, custom dw, and zero kappa."""
        times = np.array([0.0, 0.5, 1.0, 2.0])
        n_paths = 50

        paths_pseudo = self.model.simulate_paths(
            times=times,
            n_paths=n_paths,
            random_type=RandomSequenceType.PSEUDO,
            seed=42,
        )
        self.assertEqual(paths_pseudo.shape, (n_paths, len(times)))
        np.testing.assert_allclose(paths_pseudo[:, 0], 0.025)

        # Custom increments
        dw = np.random.default_rng(123).normal(0.0, 0.5, size=(n_paths, len(times) - 1))
        paths_custom = self.model.simulate_paths(
            times=times,
            n_paths=n_paths,
            dw=dw,
        )
        self.assertEqual(paths_custom.shape, (n_paths, len(times)))

        # Zero kappa simulation
        zero_kappa = VasicekModel(kappa_ann=0.0, theta_ann=0.03, r0_ann=0.025)
        paths_zero = zero_kappa.simulate_paths(
            times=times,
            n_paths=n_paths,
            seed=42,
        )
        self.assertEqual(paths_zero.shape, (n_paths, len(times)))


if __name__ == "__main__":
    unittest.main()
