"""Tests for the Heston stochastic volatility FX model."""

import unittest

import numpy as np

from tests.helpers.assertions import assert_put_call_parity
from xvasim.models.fx.heston import HestonFXModel, HestonFXParams
from xvasim.qmc import RandomSequenceType


class TestHestonFXModel(unittest.TestCase):
    """Unit tests for HestonFXModel initialization, semi-analytical pricing, and simulation."""

    def setUp(self) -> None:
        self.model = HestonFXModel(
            spot_fx=1.15,
            v_0=0.04,
            kappa_ann=2.0,
            theta_ann=0.04,
            sigma_v_ann=0.3,
            rho=-0.5,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.015,
        )

    def test_init_validation(self) -> None:
        """Invalid Heston parameters raise ValueError."""
        with self.assertRaises(ValueError):
            HestonFXModel(
                spot_fx=-1.0,
                v_0=0.04,
                kappa_ann=2.0,
                theta_ann=0.04,
                sigma_v_ann=0.3,
                rho=0.0,
            )
        with self.assertRaises(ValueError):
            HestonFXModel(
                spot_fx=1.15,
                v_0=-0.01,
                kappa_ann=2.0,
                theta_ann=0.04,
                sigma_v_ann=0.3,
                rho=0.0,
            )
        with self.assertRaises(ValueError):
            HestonFXModel(
                spot_fx=1.15,
                v_0=0.04,
                kappa_ann=-1.0,
                theta_ann=0.04,
                sigma_v_ann=0.3,
                rho=0.0,
            )
        with self.assertRaises(ValueError):
            HestonFXModel(
                spot_fx=1.15,
                v_0=0.04,
                kappa_ann=2.0,
                theta_ann=-0.04,
                sigma_v_ann=0.3,
                rho=0.0,
            )
        with self.assertRaises(ValueError):
            HestonFXModel(
                spot_fx=1.15,
                v_0=0.04,
                kappa_ann=2.0,
                theta_ann=0.04,
                sigma_v_ann=-0.1,
                rho=0.0,
            )
        with self.assertRaises(ValueError):
            HestonFXModel(
                spot_fx=1.15,
                v_0=0.04,
                kappa_ann=2.0,
                theta_ann=0.04,
                sigma_v_ann=0.3,
                rho=1.5,
            )

    def test_from_params_and_properties(self) -> None:
        """Verify factory constructor and property accessors."""
        params = HestonFXParams(
            spot_fx=1.20,
            v_0=0.05,
            kappa_ann=1.5,
            theta_ann=0.05,
            sigma_v_ann=0.2,
            rho=-0.3,
            domestic_rate_ann=0.04,
            foreign_rate_ann=0.02,
        )
        mdl = HestonFXModel.from_params(params)
        self.assertEqual(mdl.model_name, "heston")
        self.assertEqual(mdl.num_factors, 2)
        self.assertEqual(mdl.spot_fx, 1.20)
        self.assertEqual(mdl.v_0, 0.05)
        self.assertEqual(mdl.kappa_ann, 1.5)
        self.assertEqual(mdl.theta_ann, 0.05)
        self.assertEqual(mdl.sigma_v_ann, 0.2)
        self.assertEqual(mdl.rho, -0.3)
        self.assertEqual(mdl.domestic_rate_ann, 0.04)
        self.assertEqual(mdl.foreign_rate_ann, 0.02)
        self.assertTrue(mdl.is_feller_satisfied)

    def test_feller_condition(self) -> None:
        """Verify Feller condition check."""
        # 2 * 0.5 * 0.01 = 0.01 < 0.5^2 = 0.25 (not satisfied)
        non_feller = HestonFXModel(
            spot_fx=1.15,
            v_0=0.04,
            kappa_ann=0.5,
            theta_ann=0.01,
            sigma_v_ann=0.5,
            rho=-0.2,
        )
        self.assertFalse(non_feller.is_feller_satisfied)

    def test_discount_curves_and_forward(self) -> None:
        """Verify discount curves and forward rate calculations."""
        tenors = np.array([0.0, 1.0, 5.0])
        dfs_d = np.array([1.0, 0.96, 0.80])
        dfs_f = np.array([1.0, 0.98, 0.88])
        mdl = HestonFXModel(
            spot_fx=1.10,
            v_0=0.04,
            kappa_ann=2.0,
            theta_ann=0.04,
            sigma_v_ann=0.3,
            rho=-0.5,
            discount_curve_domestic_yrs=tenors,
            discount_factors_domestic=dfs_d,
            discount_curve_foreign_yrs=tenors,
            discount_factors_foreign=dfs_f,
        )
        self.assertAlmostEqual(
            float(mdl.domestic_discount_factor(1.0)), 0.96, places=10
        )
        self.assertAlmostEqual(float(mdl.foreign_discount_factor(1.0)), 0.98, places=10)
        self.assertAlmostEqual(mdl.forward_rate(1.0), 1.10 * 0.98 / 0.96, places=10)

    def test_semi_analytical_option_pricing_and_parity(self) -> None:
        """Verify Fourier semi-analytical option pricing and put-call parity."""
        strike = 1.15
        maturity = 1.0
        notional = 1000.0

        call_p = self.model.closed_form_option_price(
            strike, maturity, option_type="call", notional=notional
        )
        put_p = self.model.closed_form_option_price(
            strike, maturity, option_type="put", notional=notional
        )
        self.assertGreater(call_p, 0.0)
        self.assertGreater(put_p, 0.0)

        fwd = self.model.forward_rate(maturity)
        df_d = float(self.model.domestic_discount_factor(maturity))
        assert_put_call_parity(
            call_price=call_p / notional,
            put_price=put_p / notional,
            forward_price=fwd,
            discount_factor=df_d,
            strike=strike,
            tolerance=1e-6,
        )

    def test_invalid_option_inputs_raise(self) -> None:
        """Negative strike or maturity raises ValueError."""
        with self.assertRaises(ValueError):
            self.model.closed_form_option_price(strike=-1.0, maturity_yrs=1.0)
        with self.assertRaises(ValueError):
            self.model.closed_form_option_price(strike=1.15, maturity_yrs=-0.5)

    def test_simulate_paths(self) -> None:
        """Verify Full Truncation Heston simulation paths."""
        times, v_paths, _, fx_spot = self.model.simulate_paths(
            maturity_yrs=1.0,
            n_paths=50,
            n_steps=4,
            random_type=RandomSequenceType.PSEUDO,
            seed=42,
        )
        self.assertEqual(len(times), 5)
        self.assertEqual(v_paths.shape, (50, 5))
        self.assertEqual(fx_spot.shape, (50, 5))
        np.testing.assert_allclose(v_paths[:, 0], 0.04)
        np.testing.assert_allclose(fx_spot[:, 0], 1.15)

    def test_characteristic_function(self) -> None:
        """Verify Heston characteristic function calculation for j=1 and j=2."""
        cf1 = self.model._characteristic_function(u=1.0, tau=1.0, j=1)
        cf2 = self.model._characteristic_function(u=1.0, tau=1.0, j=2)
        self.assertIsInstance(cf1, complex)
        self.assertIsInstance(cf2, complex)
        self.assertAlmostEqual(
            abs(self.model._characteristic_function(u=0.0, tau=1.0, j=2)), 1.0, places=5
        )


if __name__ == "__main__":
    unittest.main()
