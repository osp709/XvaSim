"""Integration test: FX portfolio (netting set) -> exposure -> total XVA.

Combines the modular-model FX/IR simulation, the portfolio / netting-set
exposure layer, and the CVA/DVA/FVA/KVA/MVA aggregation pipeline into a
single end-to-end scenario.
"""

import unittest

import numpy as np
import pytest

from tests.helpers.test_curves import get_standard_credit_curve
from xvasim import (
    FXEuropeanOptionTrade,
    FXForwardTrade,
    HullWhite1FModel,
    LGMModel,
    OptionType,
    Portfolio,
    RandomSequenceType,
    TwoCurrencyFXModel,
    compute_portfolio_exposure,
    compute_portfolio_xva,
    simulate_portfolio_exposure,
)


@pytest.mark.integration
class TestPortfolioXvaPipeline(unittest.TestCase):
    """End-to-end portfolio -> exposure -> total XVA pipeline test."""

    def _build_two_currency_model(self) -> TwoCurrencyFXModel:
        tenors = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0])
        dom_dfs = np.exp(-0.03 * tenors)
        for_dfs = np.exp(-0.02 * tenors)
        domestic = LGMModel(
            kappa_ann=0.03,
            sigma_grid_yrs=np.array([1.0, 3.0, 5.0, 10.0, 30.0]),
            sigma_values_ann=np.array([0.010, 0.011, 0.012, 0.013, 0.012]),
            discount_curve_yrs=tenors,
            discount_factors=dom_dfs,
        )
        foreign = HullWhite1FModel(
            a_ann=0.04,
            sigma_ann=0.015,
            discount_curve_yrs=tenors,
            discount_factors=for_dfs,
        )
        return TwoCurrencyFXModel.from_ir_models(
            domestic=domestic,
            foreign=foreign,
            spot_fx=1.20,
            fx_vol_ann=0.15,
            correlation_matrix=np.array(
                [
                    [1.0, 0.3, 0.2],
                    [0.3, 1.0, -0.1],
                    [0.2, -0.1, 1.0],
                ]
            ),
        )

    def _build_portfolio(self) -> Portfolio:
        return Portfolio(
            portfolio_id="fx_netting_set",
            trades=[
                FXForwardTrade("fwd_long", 1_000_000.0, 1.22, 3.0),
                FXForwardTrade("fwd_short", -300_000.0, 1.19, 2.0),
                FXEuropeanOptionTrade("call", 500_000.0, 1.18, 2.0, "call"),
                FXEuropeanOptionTrade(
                    "put", 500_000.0, 1.24, 4.0, OptionType.PUT
                ),
            ],
        )

    def test_two_currency_exposure_shapes(self) -> None:
        model = self._build_two_currency_model()
        portfolio = self._build_portfolio()
        result = simulate_portfolio_exposure(
            portfolio=portfolio,
            fx_model=model,
            maturity_yrs=5.0,
            n_steps=10,
            n_paths=128,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        self.assertEqual(result.exposure.shape, (128, 11))
        self.assertEqual(result.net_mtm.shape, (128, 11))
        self.assertEqual(set(result.trade_mtm), contraction_target := {
            "fwd_long",
            "fwd_short",
            "call",
            "put",
        })
        self.assertEqual(contraction_target, set(portfolio.trade_ids))
        self.assertTrue(np.all(result.exposure >= 0.0))

    def test_netting_set_offsets_opposing_legs(self) -> None:
        model = self._build_two_currency_model()
        flat = FXForwardTrade("flat1", 100_000.0, 1.20, 3.0)
        flat_net = FXForwardTrade("flat2", -100_000.0, 1.20, 3.0)
        portfolio = Portfolio(trades=[flat, flat_net], netting=True)
        sim = simulate_portfolio_exposure(portfolio, model, 3.0, 6, 64, seed=7)
        np.testing.assert_allclose(sim.net_mtm, np.zeros((64, 7)), atol=1e-9)
        np.testing.assert_allclose(sim.exposure, np.zeros((64, 7)), atol=1e-9)

    def test_full_xva_ledger_end_to_end(self) -> None:
        model = self._build_two_currency_model()
        portfolio = self._build_portfolio()
        tenors, spreads = get_standard_credit_curve()

        result = compute_portfolio_xva(
            portfolio=portfolio,
            fx_model=model,
            maturity_yrs=5.0,
            n_steps=10,
            counterparty_credit_spreads_ann=spreads,
            own_credit_spreads_ann=spreads * 0.6,
            credit_tenors_yrs=tenors,
            n_paths=256,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        self.assertGreater(result["cva"], 0.0)
        self.assertGreaterEqual(result["dva"], 0.0)
        self.assertGreater(result["kva"], 0.0)
        self.assertGreater(result["total_xva"], 0.0)
        self.assertAlmostEqual(
            result["total_xva"],
            result["cva"] - result["dva"] + result["fva"]
            + result["kva"] + result["mva"],
            places=9,
        )
        self.assertEqual(result["exposure"].shape, (256, 11))
        self.assertEqual(result["expected_exposure"].shape, (10,))

    def test_chunked_equals_full_aggregation(self) -> None:
        model = self._build_two_currency_model()
        portfolio = self._build_portfolio()
        tenors, spreads = get_standard_credit_curve()

        full = compute_portfolio_xva(
            portfolio, model, 5.0, 10, spreads, spreads * 0.6,
            credit_tenors_yrs=tenors, n_paths=256, seed=1,
        )
        chunked = compute_portfolio_xva(
            portfolio, model, 5.0, 10, spreads, spreads * 0.6,
            credit_tenors_yrs=tenors, n_paths=256, seed=1, chunk_size=64,
        )
        self.assertAlmostEqual(chunked["cva"], full["cva"], places=10)
        self.assertAlmostEqual(chunked["dva"], full["dva"], places=10)
        self.assertAlmostEqual(chunked["total_xva"], full["total_xva"], places=10)

    def test_exposure_from_pre_built_market_simulation(self) -> None:
        model = self._build_two_currency_model()
        portfolio = self._build_portfolio()

        from xvasim import simulate_market

        sim = simulate_market(
            fx_model=model,
            maturity_yrs=5.0,
            n_steps=10,
            n_paths=64,
            random_type=RandomSequenceType.SOBOL,
            seed=5,
        )
        result = compute_portfolio_exposure(portfolio, sim)
        self.assertEqual(result.trade_mtm["fwd_short"].shape, (64, 11))
        self.assertEqual(result.fx_spot_paths.shape, (64, 11))


if __name__ == "__main__":
    unittest.main()
