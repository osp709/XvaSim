"""Tests for LGM swaption calibration engine."""

import unittest

import numpy as np

from xvasim.models.ir.lgm import LGMModel
from xvasim.pricing_engine import (
    LGMParams,
    _compute_zeta,
    _interpolate_discount_factor,
    _lgm_swaption_price_normal,
    _swaption_price_normal,
    calibrate_ir_model_to_swaptions,
    calibrate_lgm_to_swaptions,
)


class TestLGMCalibrationEngine(unittest.TestCase):
    """Unit tests for LGM swaption pricing and bootstrap calibration in pricing_engine."""

    def setUp(self) -> None:
        self.tenors = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
        self.dfs = np.exp(-0.03 * self.tenors)

    def test_lgm_zeta_and_interp_df(self) -> None:
        """Verify helper _compute_zeta and _interpolate_discount_factor."""
        grid = np.array([1.0, 5.0])
        vals = np.array([0.01, 0.012])
        self.assertEqual(_compute_zeta(0.0, grid, vals, 0.03), 0.0)
        self.assertEqual(_compute_zeta(-0.5, grid, vals, 0.03), 0.0)
        self.assertGreater(_compute_zeta(2.0, grid, vals, 0.03), 0.0)
        self.assertGreater(_compute_zeta(2.0, grid, vals, 0.0), 0.0)

        df_interp = _interpolate_discount_factor(1.0, self.tenors, self.dfs)
        self.assertAlmostEqual(float(df_interp), np.exp(-0.03 * 1.0), places=10)

    def test_lgm_swaption_price_normal_zero_kappa(self) -> None:
        """Verify swaption pricing helper with zero kappa."""
        grid = np.array([1.0, 5.0])
        vals = np.array([0.01, 0.01])
        model_p, mkt_p = _lgm_swaption_price_normal(
            expiry_yrs=1.0,
            swap_tenor_yrs=5.0,
            market_normal_vol_ann=0.008,
            kappa=0.0,
            sigma_grid_yrs=grid,
            sigma_values_ann=vals,
            curve_yrs=self.tenors,
            curve_dfs=self.dfs,
            fixed_rate_ann=0.03,
        )
        self.assertGreater(model_p, 0.0)
        self.assertGreater(mkt_p, 0.0)

        # Test _swaption_price_normal parity
        model_p2, mkt_p2 = _swaption_price_normal(
            expiry_yrs=1.0,
            swap_tenor_yrs=5.0,
            market_normal_vol_ann=0.008,
            kappa=0.0,
            sigma_grid_yrs=grid,
            sigma_values_ann=vals,
            curve_yrs=self.tenors,
            curve_dfs=self.dfs,
            fixed_rate_ann=0.03,
        )
        self.assertEqual(model_p, model_p2)
        self.assertEqual(mkt_p, mkt_p2)

    def test_lgm_model_swaption_price_normal(self) -> None:
        """Verify LGMModel.swaption_price_normal and analytical_swaption_price methods."""
        lgm = LGMModel(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([1.0, 5.0]),
            sigma_values_ann=np.array([0.01, 0.012]),
            discount_curve_yrs=self.tenors,
            discount_factors=self.dfs,
        )
        mdl_p, mkt_p = lgm.swaption_price_normal(
            expiry_yrs=1.0,
            swap_tenor_yrs=5.0,
            market_normal_vol_ann=0.008,
            fixed_rate_ann=0.03,
        )
        self.assertGreater(mdl_p, 0.0)
        self.assertGreater(mkt_p, 0.0)

        mdl_p_alias, mkt_p_alias = lgm.analytical_swaption_price(
            expiry_yrs=1.0,
            swap_tenor_yrs=5.0,
            market_normal_vol_ann=0.008,
            fixed_rate_ann=0.03,
        )
        self.assertEqual(mdl_p, mdl_p_alias)
        self.assertEqual(mkt_p, mkt_p_alias)

    def test_calibrate_ir_model_to_swaptions(self) -> None:
        """Verify calibrate_ir_model_to_swaptions and calibrate_lgm_to_swaptions."""
        expiries = np.array([1.0, 2.0, 5.0])
        tenors = np.array([5.0, 5.0, 5.0])
        mkt_vols = np.array([0.0080, 0.0085, 0.0090])
        fixed_rates = np.array([0.03, 0.03, 0.03])

        params = calibrate_ir_model_to_swaptions(
            swaption_expiries_yrs=expiries,
            swap_tenors_yrs=tenors,
            market_normal_vols_ann=mkt_vols,
            curve_yrs=self.tenors,
            curve_dfs=self.dfs,
            fixed_rates_ann=fixed_rates,
            kappa_ann=0.03,
            model_type="lgm",
        )
        self.assertIsInstance(params, LGMParams)
        self.assertEqual(len(params.sigma_values_ann), 3)
        self.assertTrue(np.all(params.sigma_values_ann > 0.0))

        # Backward compatibility alias
        params_alias = calibrate_lgm_to_swaptions(
            swaption_expiries_yrs=expiries,
            swap_tenors_yrs=tenors,
            market_normal_vols_ann=mkt_vols,
            curve_yrs=self.tenors,
            curve_dfs=self.dfs,
            fixed_rates_ann=fixed_rates,
            kappa_ann=0.03,
        )
        np.testing.assert_allclose(params.sigma_values_ann, params_alias.sigma_values_ann)

    def test_calibrate_ir_model_invalid_model_type(self) -> None:
        """Invalid model_type raises ValueError."""
        with self.assertRaises(ValueError):
            calibrate_ir_model_to_swaptions(
                swaption_expiries_yrs=np.array([1.0]),
                swap_tenors_yrs=np.array([5.0]),
                market_normal_vols_ann=np.array([0.008]),
                curve_yrs=self.tenors,
                curve_dfs=self.dfs,
                fixed_rates_ann=np.array([0.03]),
                model_type="vasicek",
            )

    def test_calibrate_lgm_failure_raises(self) -> None:
        """Calibration failure on impossible vol raises RuntimeError."""
        expiries = np.array([1.0])
        tenors = np.array([5.0])
        mkt_vols = np.array([100.0])
        fixed_rates = np.array([0.03])

        with self.assertRaises(RuntimeError):
            calibrate_lgm_to_swaptions(
                swaption_expiries_yrs=expiries,
                swap_tenors_yrs=tenors,
                market_normal_vols_ann=mkt_vols,
                curve_yrs=self.tenors,
                curve_dfs=self.dfs,
                fixed_rates_ann=fixed_rates,
            )


if __name__ == "__main__":
    unittest.main()
