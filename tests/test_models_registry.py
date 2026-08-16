"""Tests for the stochastic model registry and factory functions."""

import unittest

import numpy as np

from xvasim.models import (
    CIRHazardRateModel,
    CIRInterestRateModel,
    HullWhite1FModel,
    InterestRateModel,
    LGMModel,
    ModelRegistry,
    RiskFactorType,
    TwoCurrencyFXModel,
    VasicekModel,
    create_credit_model,
    create_fx_model,
    create_ir_model,
    list_available_models,
)


class TestModelRegistry(unittest.TestCase):
    def test_registered_ir_models(self) -> None:
        """Verify default interest rate models are registered."""
        ir_models = list_available_models(RiskFactorType.INTEREST_RATE)
        self.assertIn("lgm", ir_models)
        self.assertIn("hull_white", ir_models)
        self.assertIn("vasicek", ir_models)
        self.assertIn("cir", ir_models)

    def test_registered_credit_models(self) -> None:
        """Verify default credit models are registered."""
        credit_models = list_available_models(RiskFactorType.CREDIT)
        self.assertIn("cir", credit_models)

    def test_registered_fx_models(self) -> None:
        """Verify default FX models are registered."""
        fx_models = list_available_models(RiskFactorType.FX)
        self.assertIn("two_currency", fx_models)

    def test_list_all_models(self) -> None:
        """Verify list_available_models with None returns all model names."""
        all_models = list_available_models()
        self.assertIn("lgm", all_models)
        self.assertIn("hull_white", all_models)
        self.assertIn("cir", all_models)
        self.assertIn("two_currency", all_models)

    def test_create_ir_model_factory(self) -> None:
        """Test creating IR models via factory helper create_ir_model."""
        tenors = np.array([0.0, 1.0, 5.0, 10.0])
        dfs = np.exp(-0.03 * tenors)

        hw = create_ir_model(
            "hull_white",
            a_ann=0.04,
            sigma_ann=0.012,
            discount_curve_yrs=tenors,
            discount_factors=dfs,
        )
        self.assertIsInstance(hw, HullWhite1FModel)
        assert isinstance(hw, HullWhite1FModel)
        self.assertEqual(hw.a_ann, 0.04)

        lgm = create_ir_model(
            "lgm",
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([10.0]),
            sigma_values_ann=np.array([0.01]),
            discount_curve_yrs=tenors,
            discount_factors=dfs,
        )
        self.assertIsInstance(lgm, LGMModel)
        assert isinstance(lgm, LGMModel)
        self.assertEqual(lgm.kappa_ann, 0.03)

        vas = create_ir_model("vasicek", kappa_ann=0.2, theta_ann=0.04)
        self.assertIsInstance(vas, VasicekModel)

        cir = create_ir_model("cir", kappa_ann=0.3, theta_ann=0.035)
        self.assertIsInstance(cir, CIRInterestRateModel)

    def test_create_credit_model_factory(self) -> None:
        """Test creating Credit models via factory helper create_credit_model."""
        credit = create_credit_model(
            "cir",
            kappa_ann=0.4,
            theta_ann=0.025,
            sigma_ann=0.08,
            lambda_0_ann=0.015,
        )
        self.assertIsInstance(credit, CIRHazardRateModel)
        assert isinstance(credit, CIRHazardRateModel)
        self.assertEqual(credit.kappa_ann, 0.4)

    def test_create_fx_model_factory(self) -> None:
        """Test creating FX models via factory helper create_fx_model."""
        tenors = np.array([0.0, 1.0, 5.0])
        dfs = np.exp(-0.03 * tenors)
        dom = HullWhite1FModel(
            a_ann=0.03, sigma_ann=0.01, discount_curve_yrs=tenors, discount_factors=dfs
        )
        foreign = HullWhite1FModel(
            a_ann=0.03, sigma_ann=0.01, discount_curve_yrs=tenors, discount_factors=dfs
        )
        corr = np.eye(3)

        fx_model = create_fx_model(
            "two_currency",
            domestic_ir_model=dom,
            foreign_ir_model=foreign,
            spot_fx=1.15,
            fx_vol_ann=0.12,
            correlation_matrix=corr,
        )
        self.assertIsInstance(fx_model, TwoCurrencyFXModel)
        self.assertEqual(fx_model.spot_fx, 1.15)

    def test_unregistered_model_raises_key_error(self) -> None:
        """Querying an unknown model should raise KeyError."""
        with self.assertRaises(KeyError):
            ModelRegistry.get(RiskFactorType.INTEREST_RATE, "non_existent_model")

    def test_custom_model_registration(self) -> None:
        """Users should be able to register custom models dynamically."""

        @ModelRegistry.register("interest_rate", "custom_dummy_model")
        class CustomDummyModel(InterestRateModel):
            @property
            def model_name(self) -> str:
                return "custom_dummy_model"

            @property
            def discount_curve_yrs(self) -> np.ndarray:
                return np.array([0.0, 1.0])

            @property
            def discount_factors(self) -> np.ndarray:
                return np.array([1.0, 0.97])

            def short_rate(self, t: float, state: np.ndarray) -> np.ndarray:
                return np.asarray(state)

            def zero_coupon_bond(
                self, t: float, maturity_yrs: float, state: np.ndarray
            ) -> np.ndarray:
                return np.ones_like(state)

            def discount_path(
                self, times: np.ndarray, state_paths: np.ndarray
            ) -> np.ndarray:
                return np.ones_like(state_paths)

            def simulate_paths(
                self,
                times: np.ndarray,
                n_paths: int,
                rng: np.random.Generator,
                dw: np.ndarray | None = None,
            ) -> np.ndarray:
                return np.zeros((n_paths, len(times)))

        self.assertIn(
            "custom_dummy_model",
            list_available_models(RiskFactorType.INTEREST_RATE),
        )
        created = create_ir_model("custom_dummy_model")
        self.assertIsInstance(created, CustomDummyModel)


if __name__ == "__main__":
    unittest.main()
