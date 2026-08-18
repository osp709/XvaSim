"""Tests for base classes and abstractions in xvasim.models.base."""

import typing
import unittest

import numpy as np

from xvasim.models.base import (
    CreditModel,
    FXModel,
    InflationModel,
    InterestRateModel,
    RiskFactorType,
    StochasticModel,
)


class DummyStochasticModel(StochasticModel):
    @property
    def risk_factor_type(self) -> RiskFactorType:
        return RiskFactorType.EQUITY

    @property
    def model_name(self) -> str:
        return "dummy_equity"


class DummyIRModel(InterestRateModel):
    @property
    def model_name(self) -> str:
        return "dummy_ir"

    @property
    def discount_curve_yrs(self) -> np.ndarray:
        return np.array([0.0, 1.0, 5.0])

    @property
    def discount_factors(self) -> np.ndarray:
        return np.array([1.0, 0.97, 0.85])

    def short_rate(self, t: float, state: np.ndarray) -> np.ndarray:
        return np.asarray(state)

    def zero_coupon_bond(
        self, t: float, maturity_yrs: float, state: np.ndarray
    ) -> np.ndarray:
        return np.ones_like(state)

    def discount_path(self, times: np.ndarray, state_paths: np.ndarray) -> np.ndarray:
        return np.ones_like(state_paths)

    def simulate_paths(
        self, times: np.ndarray, n_paths: int, *args: typing.Any, **kwargs: typing.Any
    ) -> np.ndarray:
        return np.zeros((n_paths, len(times)))


class DummyCreditModel(CreditModel):
    @property
    def model_name(self) -> str:
        return "dummy_credit"

    def survival_probability(self, tenors_yrs: np.ndarray) -> np.ndarray:
        return np.exp(-0.02 * tenors_yrs)


class DummyFXModel(FXModel):
    @property
    def model_name(self) -> str:
        return "dummy_fx"

    @property
    def spot_fx(self) -> float:
        return 1.25

    def simulate_paths(
        self, maturity_yrs: float, n_paths: int, n_steps: int, *args: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        return None


class DummyInflationModel(InflationModel):
    @property
    def model_name(self) -> str:
        return "dummy_inflation"

    @property
    def base_cpi(self) -> float:
        return 100.0

    def forward_cpi(self, maturity_yrs: float) -> float:
        return 100.0 * (1.0 + 0.02 * maturity_yrs)

    def zero_coupon_inflation_swap_rate(self, maturity_yrs: float) -> float:
        return 0.02

    def simulate_paths(
        self, maturity_yrs: float, n_paths: int, n_steps: int, *args: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        return None


class TestBaseClasses(unittest.TestCase):
    """Unit tests verifying base class methods, properties, and default implementations."""

    def test_risk_factor_types(self) -> None:
        """Verify RiskFactorType enum members and string conversions."""
        self.assertEqual(RiskFactorType.INTEREST_RATE.value, "interest_rate")
        self.assertEqual(RiskFactorType.FX.value, "fx")
        self.assertEqual(RiskFactorType.CREDIT.value, "credit")
        self.assertEqual(RiskFactorType.INFLATION.value, "inflation")
        self.assertEqual(RiskFactorType.EQUITY.value, "equity")
        self.assertEqual(RiskFactorType.COMMODITY.value, "commodity")

    def test_stochastic_model_defaults(self) -> None:
        """Verify StochasticModel base properties."""
        mdl = DummyStochasticModel()
        self.assertEqual(mdl.risk_factor_type, RiskFactorType.EQUITY)
        self.assertEqual(mdl.model_name, "dummy_equity")
        self.assertEqual(mdl.num_factors, 1)

    def test_ir_model_base_methods(self) -> None:
        """Verify InterestRateModel base interpolation and instantaneous forward calculation."""
        mdl = DummyIRModel()
        self.assertEqual(mdl.risk_factor_type, RiskFactorType.INTEREST_RATE)
        df_interp = mdl.interpolate_discount_factor(1.0)
        self.assertAlmostEqual(float(df_interp), 0.97, places=10)

        fwd = mdl.instantaneous_forward(1.0)
        self.assertGreater(fwd, 0.0)

        with self.assertRaises(NotImplementedError):
            mdl.swaption_price_normal(1.0, 5.0, 0.008, 0.03)

        with self.assertRaises(NotImplementedError):
            mdl.analytical_swaption_price(1.0, 5.0, 0.008, 0.03)

    def test_credit_model_base_methods(self) -> None:
        """Verify CreditModel marginal_pd default calculation from survival probabilities."""
        mdl = DummyCreditModel()
        self.assertEqual(mdl.risk_factor_type, RiskFactorType.CREDIT)

        tenors = np.array([1.0, 2.0, 3.0])
        mpd = mdl.marginal_pd(tenors)
        self.assertEqual(len(mpd), 3)
        self.assertTrue(np.all(mpd > 0.0))
        self.assertAlmostEqual(
            float(np.sum(mpd)), float(1.0 - np.exp(-0.02 * 3.0)), places=10
        )

        with self.assertRaises(NotImplementedError):
            DummyCreditModel.calibrate_from_spreads(np.array([0.02]), np.array([1.0]))

    def test_fx_and_inflation_base_types(self) -> None:
        """Verify FXModel and InflationModel risk_factor_type properties."""
        fx = DummyFXModel()
        self.assertEqual(fx.risk_factor_type, RiskFactorType.FX)
        self.assertEqual(fx.spot_fx, 1.25)

        inf = DummyInflationModel()
        self.assertEqual(inf.risk_factor_type, RiskFactorType.INFLATION)
        self.assertEqual(inf.base_cpi, 100.0)


if __name__ == "__main__":
    unittest.main()
