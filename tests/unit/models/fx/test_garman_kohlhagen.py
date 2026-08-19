"""Tests for the Garman-Kohlhagen (Black-Scholes) FX model."""

import unittest

import numpy as np

from tests.helpers.assertions import assert_put_call_parity
from xvasim.models.fx.garman_kohlhagen import (
    GarmanKohlhagenFXModel,
    GarmanKohlhagenParams,
)
from xvasim.pricing_engine import OptionType
from xvasim.qmc import RandomSequenceType


class TestGarmanKohlhagenFXModel(unittest.TestCase):
    """Unit tests for GarmanKohlhagenFXModel analytical option pricing, properties, and simulation."""

    def setUp(self) -> None:
        self.model = GarmanKohlhagenFXModel(
            spot_fx=1.15,
            fx_vol_ann=0.12,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.015,
        )

    def test_init_validation(self) -> None:
        """Invalid spot or volatility raises ValueError."""
        with self.assertRaises(ValueError):
            GarmanKohlhagenFXModel(spot_fx=-1.0, fx_vol_ann=0.10)

        with self.assertRaises(ValueError):
            GarmanKohlhagenFXModel(spot_fx=1.15, fx_vol_ann=-0.05)

        with self.assertRaises(ValueError):
            GarmanKohlhagenFXModel(params=GarmanKohlhagenParams(spot_fx=-1.0, fx_vol_ann=0.10))

        with self.assertRaises(ValueError):
            GarmanKohlhagenFXModel(params=GarmanKohlhagenParams(spot_fx=1.0, fx_vol_ann=-0.10))

    def test_from_params(self) -> None:
        """Verify initialization via GarmanKohlhagenParams."""
        params = GarmanKohlhagenParams(
            spot_fx=1.20,
            fx_vol_ann=0.15,
            domestic_rate_ann=0.04,
            foreign_rate_ann=0.02,
        )
        mdl = GarmanKohlhagenFXModel.from_params(params)
        self.assertEqual(mdl.model_name, "garman_kohlhagen")
        self.assertEqual(mdl.spot_fx, 1.20)
        self.assertEqual(mdl.fx_vol_ann, 0.15)
        self.assertEqual(mdl.domestic_rate_ann, 0.04)
        self.assertEqual(mdl.foreign_rate_ann, 0.02)
        self.assertIs(mdl.params, params)

        mdl_direct = GarmanKohlhagenFXModel(params=params)
        self.assertEqual(mdl_direct.spot_fx, 1.20)
        self.assertIs(mdl_direct.params, params)

    def test_curve_discount_factors_and_forward(self) -> None:
        """Verify discount factors and forward rates with discrete curves."""
        tenors = np.array([0.0, 1.0, 5.0])
        dfs_d = np.array([1.0, 0.96, 0.80])
        dfs_f = np.array([1.0, 0.98, 0.88])

        mdl = GarmanKohlhagenFXModel(
            spot_fx=1.10,
            fx_vol_ann=0.10,
            discount_curve_domestic_yrs=tenors,
            discount_factors_domestic=dfs_d,
            discount_curve_foreign_yrs=tenors,
            discount_factors_foreign=dfs_f,
        )
        df_d_1 = mdl.domestic_discount_factor(1.0)
        df_f_1 = mdl.foreign_discount_factor(1.0)
        np.testing.assert_allclose(df_d_1, 0.96)
        np.testing.assert_allclose(df_f_1, 0.98)

        fwd_1 = mdl.forward_rate(1.0)
        expected_fwd = 1.10 * (0.98 / 0.96)
        self.assertAlmostEqual(fwd_1, expected_fwd, places=10)

    def test_closed_form_option_pricing_and_parity(self) -> None:
        """Verify call and put prices satisfy European put-call parity."""
        strike = 1.16
        maturity = 1.5
        notional = 1000.0

        call_price = self.model.closed_form_option_price(
            strike=strike,
            maturity_yrs=maturity,
            option_type=OptionType.CALL,
            notional=notional,
        )
        put_price = self.model.closed_form_option_price(
            strike=strike,
            maturity_yrs=maturity,
            option_type="put",
            notional=notional,
        )
        self.assertGreater(call_price, 0.0)
        self.assertGreater(put_price, 0.0)

        fwd = self.model.forward_rate(maturity)
        df_d = float(self.model.domestic_discount_factor(maturity))
        assert_put_call_parity(
            call_price=call_price / notional,
            put_price=put_price / notional,
            forward_price=fwd,
            discount_factor=df_d,
            strike=strike,
            tolerance=1e-10,
        )

    def test_zero_vol_option_pricing(self) -> None:
        """Verify zero volatility option price matches intrinsic value."""
        zero_vol_model = GarmanKohlhagenFXModel(
            spot_fx=1.15,
            fx_vol_ann=0.0,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.015,
        )
        fwd = zero_vol_model.forward_rate(1.0)
        df_d = float(zero_vol_model.domestic_discount_factor(1.0))

        call_itm = zero_vol_model.closed_form_option_price(
            strike=fwd - 0.05, maturity_yrs=1.0, option_type="call"
        )
        call_otm = zero_vol_model.closed_form_option_price(
            strike=fwd + 0.05, maturity_yrs=1.0, option_type="call"
        )
        put_itm = zero_vol_model.closed_form_option_price(
            strike=fwd + 0.05, maturity_yrs=1.0, option_type="put"
        )

        self.assertAlmostEqual(call_itm, df_d * 0.05, places=10)
        self.assertEqual(call_otm, 0.0)
        self.assertAlmostEqual(put_itm, df_d * 0.05, places=10)

    def test_invalid_option_inputs_raise(self) -> None:
        """Negative strike or maturity raises ValueError."""
        with self.assertRaises(ValueError):
            self.model.closed_form_option_price(strike=-1.0, maturity_yrs=1.0)
        with self.assertRaises(ValueError):
            self.model.closed_form_option_price(strike=1.15, maturity_yrs=-0.5)

    def test_simulate_paths(self) -> None:
        """Verify path simulation shapes and positive spot rates."""
        times, _x_dom, _x_for, fx_spot = self.model.simulate_paths(
            maturity_yrs=1.0,
            n_paths=50,
            n_steps=4,
            random_type=RandomSequenceType.PSEUDO,
            seed=42,
        )
        self.assertEqual(len(times), 5)
        self.assertEqual(fx_spot.shape, (50, 5))
        np.testing.assert_allclose(fx_spot[:, 0], 1.15)
        self.assertTrue(np.all(fx_spot > 0.0))


if __name__ == "__main__":
    unittest.main()
