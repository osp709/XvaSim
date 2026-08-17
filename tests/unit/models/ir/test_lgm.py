"""Tests for the Linear Gauss-Markov (LGM) 1-factor interest rate model."""

import unittest

import numpy as np

from xvasim.models.ir.lgm import (
    LGMModel,
    LGMParams,
    _lgm_swaption_price_normal_helper,
)
from xvasim.qmc import RandomSequenceType


class TestLGMModel(unittest.TestCase):
    """Unit tests for LGMModel initialization, analytical formulas, and calibration."""

    def setUp(self) -> None:
        self.tenors = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 30.0])
        self.dfs = np.exp(-0.03 * self.tenors)
        self.sigma_grid = np.array([1.0, 3.0, 5.0, 10.0])
        self.sigma_vals = np.array([0.010, 0.012, 0.011, 0.013])
        self.kappa = 0.03

        self.model = LGMModel(
            kappa_ann=self.kappa,
            sigma_grid_yrs=self.sigma_grid,
            sigma_values_ann=self.sigma_vals,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )

    def test_init_with_params_dataclass(self) -> None:
        """Verify model initialization via pre-built LGMParams."""
        params = LGMParams(
            kappa_ann=0.04,
            sigma_grid_yrs=self.sigma_grid,
            sigma_values_ann=self.sigma_vals,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )
        mdl = LGMModel(params=params)
        self.assertEqual(mdl.model_name, "lgm")
        self.assertEqual(mdl.kappa_ann, 0.04)
        self.assertIs(mdl.params, params)
        np.testing.assert_array_equal(mdl.sigma_grid_yrs, self.sigma_grid)
        np.testing.assert_array_equal(mdl.sigma_values_ann, self.sigma_vals)
        np.testing.assert_array_equal(mdl.discount_curve_yrs, self.tenors)
        np.testing.assert_array_equal(mdl.discount_factors, self.dfs)

    def test_init_missing_arguments_raises(self) -> None:
        """Initializing without params or without all required arrays raises ValueError."""
        with self.assertRaises(ValueError):
            LGMModel(kappa_ann=0.03)

        with self.assertRaises(ValueError):
            LGMModel(
                kappa_ann=0.03,
                sigma_grid_yrs=self.sigma_grid,
                sigma_values_ann=self.sigma_vals,
                # missing discount curve
            )

    def test_h_function(self) -> None:
        """Verify H(t) calculation for non-zero kappa and zero kappa limit."""
        t_arr = np.array([0.0, 1.0, 5.0])
        h_vals = self.model.h_function(t_arr)
        expected = (1.0 - np.exp(-self.kappa * t_arr)) / self.kappa
        np.testing.assert_allclose(h_vals, expected, rtol=1e-12)

        # Zero kappa limit
        zero_kappa_model = LGMModel(
            kappa_ann=0.0,
            sigma_grid_yrs=self.sigma_grid,
            sigma_values_ann=self.sigma_vals,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )
        h_zero = zero_kappa_model.h_function(t_arr)
        np.testing.assert_allclose(h_zero, t_arr, rtol=1e-12)

    def test_zeta_function(self) -> None:
        """Verify accumulated variance zeta(t) across piecewise segments."""
        self.assertEqual(self.model.zeta(0.0), 0.0)
        self.assertEqual(self.model.zeta(-1.0), 0.0)

        # Intermediate time within first bucket
        z1 = self.model.zeta(0.5)
        self.assertGreater(z1, 0.0)

        # Time beyond first bucket
        z2 = self.model.zeta(2.0)
        self.assertGreater(z2, z1)

        # Time beyond all breakpoints
        z_large = self.model.zeta(15.0)
        self.assertGreater(z_large, z2)

        # Zero kappa model zeta
        zero_kappa_model = LGMModel(
            kappa_ann=0.0,
            sigma_grid_yrs=self.sigma_grid,
            sigma_values_ann=self.sigma_vals,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )
        z_zero_1 = zero_kappa_model.zeta(0.5)
        expected_z_zero_1 = (0.010**2) * 0.5
        self.assertAlmostEqual(z_zero_1, expected_z_zero_1, places=10)

        # Grid with 0.0 at start (ds <= 0 branch)
        grid_zero_start = np.array([0.0, 1.0, 5.0])
        vals_zero_start = np.array([0.01, 0.01, 0.012])
        model_zero_grid = LGMModel(
            kappa_ann=0.03,
            sigma_grid_yrs=grid_zero_start,
            sigma_values_ann=vals_zero_start,
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )
        self.assertGreater(model_zero_grid.zeta(2.0), 0.0)

    def test_sigma_at(self) -> None:
        """Verify piecewise constant volatility lookup."""
        self.assertEqual(self.model.sigma_at(0.5), 0.010)
        self.assertEqual(self.model.sigma_at(1.5), 0.012)
        self.assertEqual(self.model.sigma_at(4.0), 0.011)
        self.assertEqual(self.model.sigma_at(20.0), 0.013)

    def test_short_rate_and_zero_coupon_bond(self) -> None:
        """Verify short rate and zero-coupon bond pricing."""
        state = np.array([0.0, 0.01, -0.01])
        r = self.model.short_rate(1.0, state)
        self.assertEqual(len(r), 3)

        # Zero coupon bond at maturity <= t returns 1
        zcb_expired = self.model.zero_coupon_bond(2.0, 2.0, state)
        np.testing.assert_allclose(zcb_expired, np.ones(3), rtol=1e-12)
        zcb_past = self.model.zero_coupon_bond(3.0, 2.0, state)
        np.testing.assert_allclose(zcb_past, np.ones(3), rtol=1e-12)

        # Zero coupon bond for maturity > t
        zcb = self.model.zero_coupon_bond(1.0, 5.0, state)
        self.assertEqual(len(zcb), 3)
        self.assertTrue(np.all(zcb > 0.0))
        self.assertTrue(np.all(zcb < 1.0))

    def test_discount_path(self) -> None:
        """Verify path-wise discount factor computation."""
        times = np.array([0.0, 0.5, 1.0, 2.0])
        n_paths = 10
        state_paths = np.zeros((n_paths, len(times)))
        disc_paths = self.model.discount_path(times, state_paths)
        self.assertEqual(disc_paths.shape, (n_paths, len(times)))
        np.testing.assert_allclose(disc_paths[:, 0], 1.0)
        # Should be monotonically decreasing on flat positive curve with zero state
        for step in range(len(times) - 1):
            self.assertTrue(np.all(disc_paths[:, step + 1] < disc_paths[:, step]))

    def test_simulate_paths(self) -> None:
        """Verify Monte Carlo path simulation with pseudo and custom increments."""
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

    def test_calibrate_to_swaptions(self) -> None:
        """Verify calibration of LGM model to swaption normal volatilities."""
        expiries = np.array([1.0, 2.0, 5.0])
        tenors = np.array([5.0, 5.0, 5.0])
        mkt_vols = np.array([0.0080, 0.0085, 0.0090])
        fixed_rates = np.array([0.03, 0.03, 0.03])

        calibrated = LGMModel.calibrate_to_swaptions(
            swaption_expiries_yrs=expiries,
            swap_tenors_yrs=tenors,
            market_normal_vols_ann=mkt_vols,
            curve_yrs=self.tenors,
            curve_dfs=self.dfs,
            fixed_rates_ann=fixed_rates,
            kappa_ann=0.02,
        )

        self.assertIsInstance(calibrated, LGMModel)
        self.assertEqual(len(calibrated.sigma_values_ann), 3)
        self.assertTrue(np.all(calibrated.sigma_values_ann > 0.0))

    def test_swaption_price_helper_zero_kappa(self) -> None:
        """Verify Bachelier swaption helper function with zero kappa."""
        model_p, mkt_p = _lgm_swaption_price_normal_helper(
            expiry_yrs=1.0,
            swap_tenor_yrs=5.0,
            market_normal_vol_ann=0.008,
            kappa=0.0,
            sigma_grid_yrs=np.array([1.0, 5.0]),
            sigma_values_ann=np.array([0.01, 0.01]),
            curve_yrs=self.tenors,
            curve_dfs=self.dfs,
            fixed_rate_ann=0.03,
        )
        self.assertGreater(model_p, 0.0)
        self.assertGreater(mkt_p, 0.0)

    def test_calibration_failure(self) -> None:
        """Calibration with unmatchable market conditions raises RuntimeError."""
        expiries = np.array([1.0])
        tenors = np.array([5.0])
        # Impossibly high normal vol with narrow bounds raises RuntimeError
        mkt_vols = np.array([100.0])
        fixed_rates = np.array([0.03])

        with self.assertRaises(RuntimeError):
            LGMModel.calibrate_to_swaptions(
                swaption_expiries_yrs=expiries,
                swap_tenors_yrs=tenors,
                market_normal_vols_ann=mkt_vols,
                curve_yrs=self.tenors,
                curve_dfs=self.dfs,
                fixed_rates_ann=fixed_rates,
            )


if __name__ == "__main__":
    unittest.main()
