"""Tests for modular FX models: Garman-Kohlhagen and Heston stochastic volatility."""

import unittest

import numpy as np

from xvasim import (
    GarmanKohlhagenFXModel,
    GarmanKohlhagenParams,
    HestonFXModel,
    HestonFXParams,
    OptionType,
    create_fx_model,
    list_available_models,
    price_fx_forward,
    price_fx_option,
)
from xvasim.models.base import RiskFactorType


class TestGarmanKohlhagenFXModel(unittest.TestCase):
    def setUp(self) -> None:
        self.spot = 1.25
        self.vol = 0.12
        self.r_d = 0.04
        self.r_f = 0.02
        self.model = GarmanKohlhagenFXModel(
            spot_fx=self.spot,
            fx_vol_ann=self.vol,
            domestic_rate_ann=self.r_d,
            foreign_rate_ann=self.r_f,
        )

    def test_properties(self) -> None:
        """Verify model properties and risk factor type."""
        self.assertEqual(self.model.model_name, "garman_kohlhagen")
        self.assertEqual(self.model.risk_factor_type, RiskFactorType.FX)
        self.assertEqual(self.model.num_factors, 1)
        self.assertEqual(self.model.spot_fx, self.spot)
        self.assertEqual(self.model.fx_vol_ann, self.vol)
        self.assertEqual(self.model.domestic_rate_ann, self.r_d)
        self.assertEqual(self.model.foreign_rate_ann, self.r_f)

    def test_invalid_parameters_raise(self) -> None:
        """Verify constructor validation for invalid parameters."""
        with self.assertRaises(ValueError):
            GarmanKohlhagenFXModel(spot_fx=-1.0, fx_vol_ann=0.1)
        with self.assertRaises(ValueError):
            GarmanKohlhagenFXModel(spot_fx=1.0, fx_vol_ann=-0.1)

    def test_from_params(self) -> None:
        """Test constructing from GarmanKohlhagenParams dataclass."""
        params = GarmanKohlhagenParams(
            spot_fx=1.10,
            fx_vol_ann=0.15,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.01,
        )
        model = GarmanKohlhagenFXModel.from_params(params)
        self.assertEqual(model.spot_fx, 1.10)
        self.assertEqual(model.fx_vol_ann, 0.15)

    def test_forward_rate(self) -> None:
        """Verify forward rate F(0, T) = S_0 * exp((r_d - r_f) * T)."""
        t = 2.0
        expected_fwd = self.spot * np.exp((self.r_d - self.r_f) * t)
        self.assertAlmostEqual(self.model.forward_rate(t), expected_fwd, places=10)

    def test_discount_factors_with_curve(self) -> None:
        """Test discount factor interpolation with explicit term structures."""
        tenors = np.array([0.0, 1.0, 5.0])
        dom_dfs = np.array([1.0, 0.96, 0.80])
        for_dfs = np.array([1.0, 0.98, 0.90])
        model_curve = GarmanKohlhagenFXModel(
            spot_fx=1.20,
            fx_vol_ann=0.10,
            discount_curve_domestic_yrs=tenors,
            discount_factors_domestic=dom_dfs,
            discount_curve_foreign_yrs=tenors,
            discount_factors_foreign=for_dfs,
        )
        self.assertAlmostEqual(
            float(model_curve.domestic_discount_factor(1.0)), 0.96, places=6
        )
        self.assertAlmostEqual(
            float(model_curve.foreign_discount_factor(1.0)), 0.98, places=6
        )

    def test_put_call_parity(self) -> None:
        """Verify Garman-Kohlhagen analytical prices satisfy Put-Call parity."""
        strike = 1.28
        maturity = 1.5
        call = self.model.closed_form_option_price(strike, maturity, "call")
        put = self.model.closed_form_option_price(strike, maturity, "put")

        df_d = np.exp(-self.r_d * maturity)
        df_f = np.exp(-self.r_f * maturity)
        parity_rhs = self.spot * df_f - strike * df_d

        self.assertAlmostEqual(call - put, parity_rhs, places=10)

    def test_mc_simulation_matches_closed_form(self) -> None:
        """Verify MC option pricing matches Garman-Kohlhagen analytical price."""
        strike = 1.25
        maturity = 1.0
        exact_call = self.model.closed_form_option_price(strike, maturity, "call")

        res = price_fx_option(
            self.model,
            strike=strike,
            maturity_yrs=maturity,
            notional=1.0,
            option_type=OptionType.CALL,
            n_paths=100_000,
            n_steps=50,
            seed=42,
        )
        mc_price = float(res["price"])
        std_error = float(res["std_error"])

        # Within 3 standard errors
        self.assertAlmostEqual(mc_price, exact_call, delta=3.0 * std_error)

    def test_price_fx_forward_matches_theoretical(self) -> None:
        """Verify forward pricing engine matches theoretical forward value."""
        strike = 1.20
        maturity = 1.0
        notional = 10_000.0

        fwd_res = price_fx_forward(
            self.model,
            strike=strike,
            maturity_yrs=maturity,
            notional=notional,
            n_paths=100_000,
            n_steps=20,
            seed=42,
        )
        fwd_rate = self.model.forward_rate(maturity)
        df_d = float(self.model.domestic_discount_factor(maturity))
        exact_fwd_val = notional * (fwd_rate - strike) * df_d

        self.assertAlmostEqual(
            float(fwd_res["price"]),
            exact_fwd_val,
            delta=3.0 * float(fwd_res["std_error"]),
        )


class TestHestonFXModel(unittest.TestCase):
    def setUp(self) -> None:
        self.spot = 1.15
        self.v_0 = 0.04
        self.kappa = 2.0
        self.theta = 0.04
        self.sigma_v = 0.3
        self.rho = -0.6
        self.r_d = 0.03
        self.r_f = 0.01

        self.model = HestonFXModel(
            spot_fx=self.spot,
            v_0=self.v_0,
            kappa_ann=self.kappa,
            theta_ann=self.theta,
            sigma_v_ann=self.sigma_v,
            rho=self.rho,
            domestic_rate_ann=self.r_d,
            foreign_rate_ann=self.r_f,
        )

    def test_properties_and_feller(self) -> None:
        """Verify model attributes, num_factors, and Feller condition."""
        self.assertEqual(self.model.model_name, "heston")
        self.assertEqual(self.model.risk_factor_type, RiskFactorType.FX)
        self.assertEqual(self.model.num_factors, 2)
        self.assertEqual(self.model.spot_fx, self.spot)
        self.assertEqual(self.model.v_0, self.v_0)
        self.assertEqual(self.model.kappa_ann, self.kappa)
        self.assertEqual(self.model.theta_ann, self.theta)
        self.assertEqual(self.model.sigma_v_ann, self.sigma_v)
        self.assertEqual(self.model.rho, self.rho)
        # 2 * 2.0 * 0.04 = 0.16 > 0.3^2 = 0.09 -> Feller condition satisfied
        self.assertTrue(self.model.is_feller_satisfied)

    def test_invalid_parameters_raise(self) -> None:
        """Verify parameter bounds checks."""
        with self.assertRaises(ValueError):
            HestonFXModel(
                spot_fx=1.0,
                v_0=-0.01,
                kappa_ann=1.0,
                theta_ann=0.04,
                sigma_v_ann=0.1,
                rho=0.0,
            )
        with self.assertRaises(ValueError):
            HestonFXModel(
                spot_fx=1.0,
                v_0=0.04,
                kappa_ann=0.0,
                theta_ann=0.04,
                sigma_v_ann=0.1,
                rho=0.0,
            )
        with self.assertRaises(ValueError):
            HestonFXModel(
                spot_fx=1.0,
                v_0=0.04,
                kappa_ann=1.0,
                theta_ann=0.04,
                sigma_v_ann=0.1,
                rho=1.5,
            )

    def test_from_params(self) -> None:
        """Test constructor from HestonFXParams dataclass."""
        params = HestonFXParams(
            spot_fx=1.20,
            v_0=0.02,
            kappa_ann=1.5,
            theta_ann=0.02,
            sigma_v_ann=0.2,
            rho=-0.3,
        )
        model = HestonFXModel.from_params(params)
        self.assertEqual(model.spot_fx, 1.20)
        self.assertEqual(model.v_0, 0.02)

    def test_heston_put_call_parity(self) -> None:
        """Verify semi-analytical Heston prices satisfy Put-Call parity."""
        strike = 1.18
        maturity = 1.0
        call = self.model.closed_form_option_price(strike, maturity, "call")
        put = self.model.closed_form_option_price(strike, maturity, "put")

        df_d = np.exp(-self.r_d * maturity)
        df_f = np.exp(-self.r_f * maturity)
        parity_rhs = self.spot * df_f - strike * df_d

        self.assertAlmostEqual(call - put, parity_rhs, places=6)

    def test_heston_mc_vs_semi_analytical(self) -> None:
        """MC option price matches Heston semi-analytical price."""
        strike = 1.15
        maturity = 0.5
        exact_call = self.model.closed_form_option_price(strike, maturity, "call")

        res = price_fx_option(
            self.model,
            strike=strike,
            maturity_yrs=maturity,
            notional=1.0,
            option_type=OptionType.CALL,
            n_paths=100_000,
            n_steps=100,
            seed=42,
        )
        mc_price = float(res["price"])
        std_error = float(res["std_error"])

        self.assertAlmostEqual(mc_price, exact_call, delta=3.0 * std_error)

    def test_heston_limit_to_black_scholes(self) -> None:
        """In the limit sigma_v -> 0, Heston prices converge to Garman-Kohlhagen."""
        vol = 0.20
        var = vol**2
        heston_det = HestonFXModel(
            spot_fx=1.0,
            v_0=var,
            kappa_ann=1.0,
            theta_ann=var,
            sigma_v_ann=0.0001,
            rho=0.0,
            domestic_rate_ann=0.05,
            foreign_rate_ann=0.02,
        )
        gk = GarmanKohlhagenFXModel(
            spot_fx=1.0,
            fx_vol_ann=vol,
            domestic_rate_ann=0.05,
            foreign_rate_ann=0.02,
        )

        h_call = heston_det.closed_form_option_price(strike=1.05, maturity_yrs=1.0)
        gk_call = gk.closed_form_option_price(strike=1.05, maturity_yrs=1.0)

        self.assertAlmostEqual(h_call, gk_call, places=4)


class TestFXRegistryAndFactory(unittest.TestCase):
    def test_registered_models(self) -> None:
        """Verify registered FX models list includes new models."""
        fx_models = list_available_models(RiskFactorType.FX)
        self.assertIn("garman_kohlhagen", fx_models)
        self.assertIn("black_scholes", fx_models)
        self.assertIn("gbm", fx_models)
        self.assertIn("heston", fx_models)
        self.assertIn("heston_fx", fx_models)
        self.assertIn("two_currency", fx_models)

    def test_create_fx_model_factory(self) -> None:
        """Test creating new FX models via create_fx_model factory."""
        gk = create_fx_model(
            "garman_kohlhagen",
            spot_fx=1.30,
            fx_vol_ann=0.11,
            domestic_rate_ann=0.04,
            foreign_rate_ann=0.01,
        )
        self.assertIsInstance(gk, GarmanKohlhagenFXModel)
        self.assertEqual(gk.spot_fx, 1.30)

        bs = create_fx_model(
            "black_scholes",
            spot_fx=1.30,
            fx_vol_ann=0.11,
        )
        self.assertIsInstance(bs, GarmanKohlhagenFXModel)

        heston = create_fx_model(
            "heston",
            spot_fx=1.20,
            v_0=0.03,
            kappa_ann=1.5,
            theta_ann=0.03,
            sigma_v_ann=0.2,
            rho=-0.5,
        )
        self.assertIsInstance(heston, HestonFXModel)
        self.assertEqual(heston.spot_fx, 1.20)


if __name__ == "__main__":
    unittest.main()
