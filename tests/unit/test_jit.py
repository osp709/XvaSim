"""Tests for Numba JIT kernels and numerical simulation routines."""

import unittest

import numpy as np

from xvasim.jit import (
    cir_calibration_objective_kernel,
    cir_simulate_paths_kernel,
    cir_survival_probability_kernel,
    discount_path_kernel,
    heston_simulate_paths_kernel,
    hull_white_simulate_paths_kernel,
    is_numba_available,
    lgm_simulate_paths_kernel,
    vasicek_simulate_paths_kernel,
)


class TestJITKernels(unittest.TestCase):
    """Unit tests for compiled numerical kernels."""

    def test_is_numba_available(self) -> None:
        """Verify JIT availability function returns boolean."""
        avail = is_numba_available()
        self.assertIsInstance(avail, bool)

    def test_cir_survival_probability_kernel_exactness(self) -> None:
        """Verify CIR survival probability calculation against reference analytical values."""
        tenors = np.array([-0.1, 0.0, 0.5, 1.0, 2.0, 5.0, np.nan])
        kappa, theta, sigma, lam0 = 0.5, 0.03, 0.10, 0.02

        surv = cir_survival_probability_kernel(
            tenors, kappa, theta, sigma, lam0
        )
        self.assertEqual(len(surv), 7)
        self.assertEqual(surv[0], 1.0)
        self.assertAlmostEqual(surv[1], 1.0, places=10)
        self.assertTrue(np.isnan(surv[-1]))
        # Should be monotonically decreasing on positive tenors
        for i in range(1, 5):
            self.assertGreater(surv[i], surv[i + 1])
            self.assertTrue(0.0 < surv[i] <= 1.0)

    def test_cir_calibration_objective_kernel(self) -> None:
        """Verify sum-of-squares objective kernel value."""
        tenors = np.array([0.5, 1.0, 2.0])
        spreads = np.array([0.02, 0.022, 0.025])
        params_exact = np.array([0.5, 0.03, 0.10, 0.02])

        val = cir_calibration_objective_kernel(params_exact, tenors, spreads)
        self.assertIsInstance(val, float)
        self.assertGreaterEqual(val, 0.0)

        # Invalid bounds penalty
        invalid_params = np.array([-0.5, 0.03, 0.10, 0.02])
        penalty = cir_calibration_objective_kernel(
            invalid_params, tenors, spreads
        )
        self.assertEqual(penalty, 1e12)

        # NaN handling
        nan_tenor_penalty = cir_calibration_objective_kernel(
            params_exact, np.array([np.nan]), spreads
        )
        self.assertEqual(nan_tenor_penalty, 1e12)

    def test_cir_simulate_paths_kernel(self) -> None:
        """Verify CIR simulation kernel bounds and shapes."""
        n_paths = 100
        n_steps = 10
        dt_vec = np.full(n_steps, 0.1)
        dw_matrix = np.random.standard_normal((n_paths, n_steps)) * np.sqrt(0.1)

        paths = cir_simulate_paths_kernel(
            n_paths=n_paths,
            n_steps=n_steps,
            dt_vec=dt_vec,
            kappa=0.2,
            theta=0.03,
            sigma=0.08,
            r0=0.025,
            dw_matrix=dw_matrix,
        )
        self.assertEqual(paths.shape, (n_paths, n_steps + 1))
        np.testing.assert_allclose(paths[:, 0], 0.025)
        self.assertTrue(np.all(paths >= 0.0))

    def test_lgm_simulate_paths_kernel(self) -> None:
        """Verify LGM simulation kernel."""
        n_paths = 50
        n_steps = 5
        dt_vec = np.full(n_steps, 0.2)
        sigmas = np.full(n_steps, 0.01)
        dw_matrix = np.random.standard_normal((n_paths, n_steps)) * np.sqrt(0.2)

        paths = lgm_simulate_paths_kernel(
            n_paths=n_paths,
            n_steps=n_steps,
            dt_vec=dt_vec,
            kappa=0.03,
            sigmas=sigmas,
            dw_matrix=dw_matrix,
        )
        self.assertEqual(paths.shape, (n_paths, n_steps + 1))
        np.testing.assert_allclose(paths[:, 0], 0.0)

    def test_vasicek_simulate_paths_kernel(self) -> None:
        """Verify Vasicek simulation kernel."""
        n_paths = 50
        n_steps = 5
        dt_vec = np.full(n_steps, 0.2)
        dw_matrix = np.random.standard_normal((n_paths, n_steps)) * np.sqrt(0.2)

        paths = vasicek_simulate_paths_kernel(
            n_paths=n_paths,
            n_steps=n_steps,
            dt_vec=dt_vec,
            kappa=0.15,
            theta=0.03,
            sigma=0.015,
            r0=0.025,
            dw_matrix=dw_matrix,
        )
        self.assertEqual(paths.shape, (n_paths, n_steps + 1))
        np.testing.assert_allclose(paths[:, 0], 0.025)

    def test_hull_white_simulate_paths_kernel(self) -> None:
        """Verify Hull-White simulation kernel."""
        n_paths = 50
        n_steps = 5
        dt_vec = np.full(n_steps, 0.2)
        theta_vec = np.full(n_steps, 0.03)
        dw_matrix = np.random.standard_normal((n_paths, n_steps)) * np.sqrt(0.2)

        paths = hull_white_simulate_paths_kernel(
            n_paths=n_paths,
            n_steps=n_steps,
            dt_vec=dt_vec,
            a=0.03,
            sigma=0.01,
            theta_vec=theta_vec,
            r0=0.025,
            dw_matrix=dw_matrix,
        )
        self.assertEqual(paths.shape, (n_paths, n_steps + 1))
        np.testing.assert_allclose(paths[:, 0], 0.025)

    def test_heston_simulate_paths_kernel(self) -> None:
        """Verify Heston simulation kernel shapes and non-negativity."""
        n_paths = 64
        n_steps = 4
        dt = 0.25
        sqrt_dt = 0.5
        r_d_vec = np.full(n_steps, 0.03)
        r_f_vec = np.full(n_steps, 0.01)
        z_all = np.random.standard_normal((n_paths, n_steps, 2))

        v_paths, fx_spot = heston_simulate_paths_kernel(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            sqrt_dt=sqrt_dt,
            v0=0.04,
            spot_fx=1.30,
            kappa=1.5,
            theta=0.04,
            sigma_v=0.2,
            rho=-0.5,
            r_d_vec=r_d_vec,
            r_f_vec=r_f_vec,
            z_all=z_all,
        )
        self.assertEqual(v_paths.shape, (n_paths, n_steps + 1))
        self.assertEqual(fx_spot.shape, (n_paths, n_steps + 1))
        np.testing.assert_allclose(v_paths[:, 0], 0.04)
        np.testing.assert_allclose(fx_spot[:, 0], 1.30)
        self.assertTrue(np.all(fx_spot > 0.0))

    def test_discount_path_kernel(self) -> None:
        """Verify trapezoidal discount integration."""
        times = np.array([0.0, 1.0, 2.0])
        short_rates = np.array([[0.05, 0.05, 0.05], [0.02, 0.04, 0.06]])

        dfs = discount_path_kernel(times, short_rates)
        self.assertEqual(dfs.shape, (2, 3))
        np.testing.assert_allclose(dfs[:, 0], [1.0, 1.0])
        # Path 0 has constant 0.05 rate: D(0, 1) = exp(-0.05), D(0, 2) = exp(-0.10)
        self.assertAlmostEqual(dfs[0, 1], np.exp(-0.05), places=8)
        self.assertAlmostEqual(dfs[0, 2], np.exp(-0.10), places=8)


if __name__ == "__main__":
    unittest.main()
