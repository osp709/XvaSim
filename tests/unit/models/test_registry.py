"""Tests for the stochastic model registry and factory functions."""

import typing
import unittest

import numpy as np

from xvasim.models.base import InterestRateModel, RiskFactorType
from xvasim.models.credit.cir import CIRHazardRateModel
from xvasim.models.fx.two_currency import TwoCurrencyFXModel
from xvasim.models.inflation.black_inflation import BlackInflationModel
from xvasim.models.ir.cir import CIRInterestRateModel
from xvasim.models.ir.hull_white import HullWhite1FModel
from xvasim.models.ir.lgm import LGMModel
from xvasim.models.ir.vasicek import VasicekModel
from xvasim.models.registry import (
    ModelRegistry,
    create_credit_model,
    create_fx_model,
    create_inflation_model,
    create_ir_model,
    list_available_models,
)


class TestModelRegistry(unittest.TestCase):
    """Unit tests for ModelRegistry and factory functions."""

    def test_registered_ir_models(self) -> None:
        """Verify default interest rate models are registered."""
        ir_models = list_available_models(RiskFactorType.INTEREST_RATE)
        self.assertIn("lgm", ir_models)
        self.assertIn("hull_white", ir_models)
        self.assertIn("hull_white_1f", ir_models)
        self.assertIn("vasicek", ir_models)
        self.assertIn("cir", ir_models)
        self.assertIn("cir_ir", ir_models)
        self.assertIn("cox_ingersoll_ross", ir_models)

    def test_registered_credit_models(self) -> None:
        """Verify default credit models are registered."""
        credit_models = list_available_models(RiskFactorType.CREDIT)
        self.assertIn("cir", credit_models)
        self.assertIn("cir_hazard_rate", credit_models)
        self.assertIn("cox_ingersoll_ross", credit_models)

    def test_registered_fx_models(self) -> None:
        """Verify default FX models are registered."""
        fx_models = list_available_models(RiskFactorType.FX)
        self.assertIn("two_currency", fx_models)
        self.assertIn("garman_kohlhagen", fx_models)
        self.assertIn("heston", fx_models)

    def test_registered_inflation_models(self) -> None:
        """Verify default inflation models are registered."""
        inf_models = list_available_models(RiskFactorType.INFLATION)
        self.assertIn("jarrow_yildirim", inf_models)
        self.assertIn("black", inf_models)
        self.assertIn("black_inflation", inf_models)

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

        lgm = create_ir_model(
            "lgm",
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([10.0]),
            sigma_values_ann=np.array([0.01]),
            discount_curve_yrs=tenors,
            discount_factors=dfs,
        )
        self.assertIsInstance(lgm, LGMModel)

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

    def test_create_inflation_model_factory(self) -> None:
        """Test creating Inflation models via factory helper create_inflation_model."""
        tenors = np.array([0.0, 1.0, 5.0])
        nom_dfs = np.exp(-0.035 * tenors)
        real_dfs = np.exp(-0.015 * tenors)

        inf_model = create_inflation_model(
            "black",
            nominal_discount_curve_yrs=tenors,
            nominal_discount_factors=nom_dfs,
            real_discount_curve_yrs=tenors,
            real_discount_factors=real_dfs,
            base_cpi=100.0,
            cpi_vol_ann=0.02,
        )
        self.assertIsInstance(inf_model, BlackInflationModel)

    def test_unregistered_model_raises_key_error(self) -> None:
        """Querying an unknown model should raise KeyError."""
        with self.assertRaises(KeyError):
            ModelRegistry.get(RiskFactorType.INTEREST_RATE, "non_existent_model")

    def test_custom_model_registration_and_direct_register(self) -> None:
        """Users should be able to register custom models dynamically."""

        class CustomDirectModel(InterestRateModel):
            @property
            def model_name(self) -> str:
                return "custom_direct_model"

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
                self, times: np.ndarray, n_paths: int, *args: typing.Any, **kwargs: typing.Any
            ) -> np.ndarray:
                return np.zeros((n_paths, len(times)))

        ModelRegistry.register(
            "interest_rate", "custom_direct_model", CustomDirectModel
        )
        self.assertIn(
            "custom_direct_model", list_available_models(RiskFactorType.INTEREST_RATE)
        )
        created = create_ir_model("custom_direct_model")
        self.assertIsInstance(created, CustomDirectModel)

    def test_factory_type_error_checks(self) -> None:
        """Verify TypeError is raised if factory creates instance not matching base type."""

        class NotAnIRModel:
            pass

        class NotACreditModel:
            pass

        class NotAnFXModel:
            pass

        class NotAnInflationModel:
            pass

        ModelRegistry.register("interest_rate", "dummy_ir", NotAnIRModel)  # type: ignore
        with self.assertRaises(TypeError):
            create_ir_model("dummy_ir")

        ModelRegistry.register("credit", "dummy_credit", NotACreditModel)  # type: ignore
        with self.assertRaises(TypeError):
            create_credit_model("dummy_credit")

        ModelRegistry.register("fx", "dummy_fx", NotAnFXModel)  # type: ignore
        with self.assertRaises(TypeError):
            create_fx_model("dummy_fx")

        ModelRegistry.register("inflation", "dummy_inflation", NotAnInflationModel)  # type: ignore
        with self.assertRaises(TypeError):
            create_inflation_model("dummy_inflation")

    def test_model_registry_clear(self) -> None:
        """Verify ModelRegistry.clear empties registry and can be restored."""
        snapshot = dict(ModelRegistry._registry)
        try:
            ModelRegistry.clear()
            self.assertEqual(len(ModelRegistry.list_models()), 0)
        finally:
            ModelRegistry._registry.clear()
            ModelRegistry._registry.update(snapshot)

        self.assertGreater(len(ModelRegistry.list_models()), 0)


if __name__ == "__main__":
    unittest.main()
