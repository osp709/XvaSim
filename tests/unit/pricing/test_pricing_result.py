"""Tests for PricingResult container and pricer parameter flexibility."""

import unittest

import numpy as np

from xvasim.models.fx.garman_kohlhagen import (
    GarmanKohlhagenFXModel,
    GarmanKohlhagenParams,
)
from xvasim.pricing_engine import (
    OptionType,
    PricingResult,
    benchmark_price_foreign_exchange_forward,
    benchmark_price_foreign_exchange_option,
    price_foreign_exchange_forward,
    price_foreign_exchange_option,
)


class TestPricingResult(unittest.TestCase):
    """Unit tests for PricingResult dictionary and attribute interface."""

    def test_pricing_result_dict_and_attr_access(self) -> None:
        """PricingResult should support both dict indexing and attribute access."""
        res = PricingResult(
            price=12.345,
            std_error=0.012,
            analytical_benchmark_price=12.340,
            fair_swap_rate=0.035,
        )

        # Dict indexing
        self.assertEqual(res["price"], 12.345)
        self.assertEqual(res["std_error"], 0.012)
        self.assertEqual(res["analytical_benchmark_price"], 12.340)
        self.assertEqual(res["fair_swap_rate"], 0.035)

        # Attribute access
        self.assertEqual(res.price, 12.345)
        self.assertEqual(res.std_error, 0.012)
        self.assertEqual(res.analytical_benchmark_price, 12.340)
        self.assertEqual(res.fair_swap_rate, 0.035)

        # Attribute mutation & deletion
        res.custom_metric = 42.0
        self.assertEqual(res["custom_metric"], 42.0)
        self.assertEqual(res.custom_metric, 42.0)
        del res.custom_metric
        self.assertNotIn("custom_metric", res)

    def test_pricing_result_missing_attribute(self) -> None:
        """Accessing or deleting a non-existent attribute should raise AttributeError."""
        res = PricingResult(price=5.0)
        with self.assertRaises(AttributeError):
            _ = res.non_existent_key

        with self.assertRaises(AttributeError):
            del res.non_existent_key

        self.assertIsNone(res.std_error)
        self.assertIsNone(res.analytical_benchmark_price)

    def test_pricing_result_repr(self) -> None:
        """PricingResult string representation should be formatted cleanly."""
        res = PricingResult(
            price=12.345678,
            std_error=0.001,
            arr=np.zeros((10, 2)),
            tag="USD/EUR",
        )
        repr_str = repr(res)
        self.assertIn("PricingResult(", repr_str)
        self.assertIn("price=12.3457", repr_str)
        self.assertIn("ndarray(shape=(10, 2))", repr_str)
        self.assertIn("tag='USD/EUR'", repr_str)


class TestPricerParameterFlexibility(unittest.TestCase):
    """Verify that pricing functions accept both model= and params= keyword args."""

    def setUp(self) -> None:
        self.model = GarmanKohlhagenFXModel(
            spot_fx=1.20,
            fx_vol_ann=0.15,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.01,
        )
        self.params = GarmanKohlhagenParams(
            spot_fx=1.20,
            fx_vol_ann=0.15,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.01,
        )

    def test_fx_forward_with_model_and_params(self) -> None:
        """Test FX forward pricers accept model=, params=, or positional model."""
        res1 = benchmark_price_foreign_exchange_forward(
            model=self.model, strike=1.20, maturity_yrs=1.0, notional=100.0
        )
        res2 = benchmark_price_foreign_exchange_forward(
            params=self.params, strike=1.20, maturity_yrs=1.0, notional=100.0
        )
        self.assertIsInstance(res1, PricingResult)
        self.assertAlmostEqual(res1.price, res2.price, places=6)

        mc_res1 = price_foreign_exchange_forward(
            model=self.model,
            strike=1.20,
            maturity_yrs=1.0,
            notional=100.0,
            n_paths=1000,
        )
        mc_res2 = price_foreign_exchange_forward(
            params=self.params,
            strike=1.20,
            maturity_yrs=1.0,
            notional=100.0,
            n_paths=1000,
        )
        self.assertIsInstance(mc_res1, PricingResult)
        self.assertIsInstance(mc_res2, PricingResult)

    def test_fx_option_with_model_and_params(self) -> None:
        """Test FX option pricers accept model=, params=, or positional model."""
        res1 = benchmark_price_foreign_exchange_option(
            model=self.model,
            strike=1.20,
            maturity_yrs=1.0,
            notional=100.0,
            option_type=OptionType.CALL,
        )
        res2 = benchmark_price_foreign_exchange_option(
            params=self.params,
            strike=1.20,
            maturity_yrs=1.0,
            notional=100.0,
            option_type="call",
        )
        self.assertIsInstance(res1, PricingResult)
        self.assertAlmostEqual(res1.price, res2.price, places=6)

        mc_res = price_foreign_exchange_option(
            model=self.model,
            strike=1.20,
            maturity_yrs=1.0,
            notional=100.0,
            option_type=OptionType.CALL,
            n_paths=1000,
        )
        self.assertIsInstance(mc_res, PricingResult)


if __name__ == "__main__":
    unittest.main()
