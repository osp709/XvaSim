"""Unit tests for the portfolio / netting-set exposure layer."""

import typing
import unittest

import numpy as np
import pytest

from tests.helpers.test_curves import (
    get_standard_credit_curve,
    get_standard_discount_curve,
)
from xvasim import (
    CIRHazardRateModel,
    FXEuropeanOptionTrade,
    FXForwardTrade,
    GarmanKohlhagenFXModel,
    HestonFXModel,
    HullWhite1FModel,
    LGMModel,
    MarketSimulation,
    OptionType,
    Portfolio,
    RandomSequenceType,
    TwoCurrencyFXModel,
    benchmark_price_foreign_exchange_forward,
    compute_portfolio_exposure,
    compute_portfolio_xva,
    compute_total_xva,
    simulate_market,
    simulate_portfolio_exposure,
)
from xvasim.portfolio import PortfolioExposureResult


def _gk_model() -> GarmanKohlhagenFXModel:
    """Deterministic-curve Garman-Kohlhagen model for exact analytical checks."""
    return GarmanKohlhagenFXModel(
        spot_fx=1.20,
        fx_vol_ann=0.15,
        domestic_rate_ann=0.03,
        foreign_rate_ann=0.02,
    )


def _two_currency_model() -> TwoCurrencyFXModel:
    """Stochastic-rates two-currency model with LGM/HW components."""
    tenors, dom_dfs = get_standard_discount_curve()
    foreign_dfs = np.exp(-0.020 * tenors)
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
        discount_factors=foreign_dfs,
    )
    return TwoCurrencyFXModel(
        domestic_ir_model=domestic,
        foreign_ir_model=foreign,
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


@pytest.mark.unit
class TestMarketSimulation(unittest.TestCase):
    """Tests for :func:`simulate_market` and :class:`MarketSimulation`."""

    def test_simulation_grid_and_shapes(self) -> None:
        model = _gk_model()
        sim = simulate_market(
            fx_model=model,
            maturity_yrs=3.0,
            n_steps=6,
            n_paths=128,
            random_type=RandomSequenceType.SOBOL,
            seed=42,
        )
        self.assertIsInstance(sim, object)
        expected_times = np.linspace(0.0, 3.0, 7)
        np.testing.assert_allclose(sim.times_yrs, expected_times)
        self.assertEqual(sim.fx_spot_paths.shape, (128, 7))
        self.assertEqual(sim.fx_spot_paths[0, 0], 1.20)
        self.assertEqual(sim.n_paths, 128)

    def test_deterministic_discount_paths(self) -> None:
        model = _gk_model()
        sim = simulate_market(model, 3.0, 6, 64, seed=1)
        expected = np.tile(
            np.exp(-0.03 * sim.times_yrs), (64, 1)
        )
        np.testing.assert_allclose(sim.domestic_df_paths, expected, rtol=1e-12)
        self.assertEqual(sim.domestic_state_paths.shape, (64, 7))

    def test_stochastic_discount_paths_two_currency(self) -> None:
        model = _two_currency_model()
        sim = simulate_market(model, 3.0, 6, 128, seed=7)
        dom_df_t0 = sim.domestic_df_paths[:, 0]
        np.testing.assert_allclose(dom_df_t0, np.ones(128))
        self.assertGreater(float(np.mean(sim.domestic_df_paths[:, -1])), 0.0)
        self.assertLess(float(np.mean(sim.domestic_df_paths[:, -1])), 1.0)

    def test_invalid_simulation_parameters(self) -> None:
        model = _gk_model()
        with self.assertRaises(ValueError):
            simulate_market(model, 0.0, 6, 64)
        with self.assertRaises(ValueError):
            simulate_market(model, 3.0, 0, 64)
        with self.assertRaises(ValueError):
            simulate_market(model, 3.0, 6, 0)

    def test_market_simulation_shape_validation(self) -> None:
        model = _gk_model()
        sim = simulate_market(model, 3.0, 6, 64, seed=1)
        with self.assertRaises(ValueError):
            MarketSimulation(
                times_yrs=sim.times_yrs,
                n_paths=64,
                fx_model=model,
                fx_spot_paths=np.zeros((63, 7)),
                domestic_state_paths=sim.domestic_state_paths,
                foreign_state_paths=sim.foreign_state_paths,
                domestic_df_paths=sim.domestic_df_paths,
                foreign_df_paths=sim.foreign_df_paths,
            )


@pytest.mark.unit
class TestFXForwardTrade(unittest.TestCase):
    """Tests for :class:`FXForwardTrade` conditional valuation."""

    def test_t0_value_matches_cip_benchmark(self) -> None:
        model = _gk_model()
        notional = 100_000.0
        strike = 1.22
        maturity = 2.0
        trade = FXForwardTrade(
            trade_id="fwd1",
            notional=notional,
            strike_fx=strike,
            maturity_yrs=maturity,
        )
        sim = simulate_market(model, maturity, 8, 256, seed=3)

        mtm = trade.value_paths(sim)
        self.assertEqual(mtm.shape, (256, 9))

        bench = benchmark_price_foreign_exchange_forward(
            model=model, strike=strike, maturity_yrs=maturity, notional=notional
        )
        np.testing.assert_allclose(mtm[:, 0], bench["price"], rtol=1e-10)

        # Mid-grid conditional value: N * (S_t * P_f(t,T) - K * P_d(t,T))
        for j in (2, 4):
            t = sim.times_yrs[j]
            p_d = np.exp(-0.03 * (maturity - t))
            p_f = np.exp(-0.02 * (maturity - t))
            expected = notional * (sim.fx_spot_paths[:, j] * p_f - strike * p_d)
            np.testing.assert_allclose(mtm[:, j], expected, rtol=1e-10)

    def test_forward_value_at_and_after_maturity(self) -> None:
        model = _gk_model()
        trade = FXForwardTrade(
            trade_id="fwd1",
            notional=10.0,
            strike_fx=1.20,
            maturity_yrs=2.0,
        )
        sim = simulate_market(model, 3.0, 6, 64, seed=5)
        mtm = trade.value_paths(sim)
        last_col = sim.times_yrs[-1]
        self.assertGreater(last_col, 2.0)
        self.assertTrue(np.all(mtm[:, -1] == 0.0))
        maturity_idx = int(np.where(sim.times_yrs == 2.0)[0][0])
        expected_payoff = 10.0 * (sim.fx_spot_paths[:, maturity_idx] - 1.20)
        np.testing.assert_allclose(mtm[:, maturity_idx], expected_payoff, rtol=1e-10)

    def test_two_currency_forward_t0_value(self) -> None:
        model = _two_currency_model()
        notional = 50_000.0
        strike = 1.25
        maturity = 3.0
        trade = FXForwardTrade("fwd2", notional, strike, maturity)
        sim = simulate_market(model, maturity, 6, 128, seed=11)
        mtm = trade.value_paths(sim)

        tenors, dom_dfs = get_standard_discount_curve()
        p_d = float(np.exp(np.interp(maturity, tenors, np.log(dom_dfs))))
        p_f = float(np.exp(np.interp(maturity, tenors, np.log(np.exp(-0.02 * tenors)))))
        expected_t0 = notional * (model.spot_fx * p_f - strike * p_d)
        np.testing.assert_allclose(mtm[:, 0], expected_t0, rtol=1e-8)

    def test_invalid_trade_parameters(self) -> None:
        with self.assertRaises(ValueError):
            FXForwardTrade("", 1.0, 1.2, 1.0)
        with self.assertRaises(ValueError):
            FXForwardTrade("id", 1.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            FXForwardTrade("id", 1.0, 1.2, 0.0)


@pytest.mark.unit
class TestFXEuropeanOptionTrade(unittest.TestCase):
    """Tests for :class:`FXEuropeanOptionTrade` conditional Black-76 valuation."""

    def test_t0_value_matches_gk_closed_form(self) -> None:
        model = _gk_model()
        notional = 1_000.0
        strike = 1.18
        maturity = 1.5
        call = FXEuropeanOptionTrade(
            "call1", notional, strike, maturity, OptionType.CALL
        )
        put = FXEuropeanOptionTrade(
            "put1", notional, strike, maturity, OptionType.PUT
        )
        sim = simulate_market(model, maturity, 6, 256, seed=13)

        call_mtm = call.value_paths(sim)
        put_mtm = put.value_paths(sim)

        gk_call = model.closed_form_option_price(
            strike, maturity, "call", notional
        )
        gk_put = model.closed_form_option_price(strike, maturity, "put", notional)
        np.testing.assert_allclose(call_mtm[:, 0], gk_call, rtol=1e-10)
        np.testing.assert_allclose(put_mtm[:, 0], gk_put, rtol=1e-10)

    def test_put_call_parity_along_grid(self) -> None:
        model = _gk_model()
        notional = 1_000.0
        strike = 1.21
        maturity = 2.0
        call = FXEuropeanOptionTrade("call", notional, strike, maturity, "call")
        put = FXEuropeanOptionTrade("put", notional, strike, maturity, "put")
        sim = simulate_market(model, maturity, 8, 128, seed=17)

        call_mtm = call.value_paths(sim)
        put_mtm = put.value_paths(sim)
        fwd = FXForwardTrade("fwd", notional, strike, maturity)
        fwd_mtm = fwd.value_paths(sim)

        np.testing.assert_allclose(
            call_mtm - put_mtm, fwd_mtm, rtol=1e-10
        )

    def test_option_intrinsic_at_maturity(self) -> None:
        model = _gk_model()
        maturity = 2.0
        call = FXEuropeanOptionTrade("call", 10.0, 1.20, maturity, "call")
        sim = simulate_market(model, maturity, 4, 64, seed=19)
        mtm = call.value_paths(sim)
        last = np.where(sim.times_yrs == maturity)[0][0]
        expected = 10.0 * np.maximum(sim.fx_spot_paths[:, last] - 1.20, 0.0)
        np.testing.assert_allclose(mtm[:, last], expected, rtol=1e-10)

    def test_heston_option_trade_raises(self) -> None:
        heston = HestonFXModel(
            spot_fx=1.20,
            v_0=0.04,
            kappa_ann=2.0,
            theta_ann=0.04,
            sigma_v_ann=0.20,
            rho=-0.5,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.02,
        )
        portfolio = Portfolio(trades=[FXEuropeanOptionTrade("c", 1.0, 1.2, 1.0)])
        sim = simulate_market(heston, 1.5, 3, 32, seed=3)
        with self.assertRaises(NotImplementedError):
            compute_portfolio_exposure(portfolio, sim)


@pytest.mark.unit
class TestPortfolioAggregation(unittest.TestCase):
    """Tests for :class:`Portfolio` validation and exposure aggregation."""

    def test_unique_trade_ids_required(self) -> None:
        t1 = FXForwardTrade("dup", 1.0, 1.20, 1.0)
        t2 = FXForwardTrade("dup", 1.0, 1.22, 1.0)
        with self.assertRaises(ValueError):
            Portfolio(trades=[t1, t2])

    def test_non_trade_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Portfolio(trades=["not-a-trade"])  # type: ignore[list-item]

    def test_empty_portfolio_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Portfolio(portfolio_id="")

    def test_netting_offsets_positions(self) -> None:
        notional = 100_000.0
        long_fwd = FXForwardTrade("long", notional, 1.18, 2.0)
        short_fwd = FXForwardTrade("short", -notional, 1.18, 2.0)
        portfolio = Portfolio(trades=[long_fwd, short_fwd], netting=True)

        model = _gk_model()
        sim = simulate_market(model, 2.0, 4, 64, seed=23)
        result = compute_portfolio_exposure(portfolio, sim)
        self.assertIsInstance(result, PortfolioExposureResult)

        np.testing.assert_allclose(result.net_mtm, np.zeros((64, 5)), atol=1e-9)
        np.testing.assert_allclose(
            result.exposure, np.zeros((64, 5)), atol=1e-9
        )
        self.assertEqual(result.n_paths, 64)

    def test_no_netting_sums_positive_marks(self) -> None:
        model = _gk_model()
        notional = 100_000.0
        long_fwd = FXForwardTrade("long", notional, 1.18, 2.0)
        short_fwd = FXForwardTrade("short", -notional, 1.18, 2.0)
        portfolio = Portfolio(trades=[long_fwd, short_fwd], netting=False)

        sim = simulate_market(model, 2.0, 4, 64, seed=29)
        result = compute_portfolio_exposure(portfolio, sim)

        desired = np.mean(
            np.maximum(result.trade_mtm["long"], 0.0)
            + np.maximum(result.trade_mtm["short"], 0.0),
            axis=0,
        )
        np.testing.assert_allclose(
            np.mean(result.exposure, axis=0), desired, rtol=1e-10
        )

    def test_trade_horizon_validation(self) -> None:
        model = _gk_model()
        portfolio = Portfolio(trades=[FXForwardTrade("f", 1.0, 1.2, 4.0)])
        sim = simulate_market(model, 2.0, 4, 8, seed=31)
        with self.assertRaises(ValueError):
            compute_portfolio_exposure(portfolio, sim)

    def test_simulate_portfolio_exposure_wrapper(self) -> None:
        model = _gk_model()
        portfolio = Portfolio(
            trades=[
                FXForwardTrade("fwd", 10_000.0, 1.20, 2.0),
                FXEuropeanOptionTrade("opt", 5_000.0, 1.20, 2.0, "call"),
            ]
        )
        result = simulate_portfolio_exposure(
            portfolio, model, 2.0, 4, 64, seed=37
        )
        self.assertEqual(set(result.trade_mtm), {"fwd", "opt"})
        self.assertEqual(result.n_paths, 64)
        self.assertEqual(result.trade_mtm["fwd"].shape, (64, 5))


@pytest.mark.unit
class TestComputePortfolioXva(unittest.TestCase):
    """End-to-end portfolio -> exposure -> total XVA ledger."""

    def _run_xva(self, **kwargs: typing.Any):
        model = _two_currency_model()
        ccy_tenors, ccy_spreads = get_standard_credit_curve()
        own_spreads = ccy_spreads * 0.6
        portfolio = Portfolio(
            trades=[
                FXForwardTrade("fwd", 1_000_000.0, 1.22, 3.0),
                FXEuropeanOptionTrade("call", 500_000.0, 1.18, 2.0, "call"),
                FXEuropeanOptionTrade("put", 500_000.0, 1.24, 4.0, "put"),
            ]
        )
        return compute_portfolio_xva(
            portfolio=portfolio,
            fx_model=model,
            maturity_yrs=5.0,
            n_steps=10,
            counterparty_credit_spreads_ann=ccy_spreads,
            own_credit_spreads_ann=own_spreads,
            credit_tenors_yrs=ccy_tenors,
            n_paths=512,
            random_type=RandomSequenceType.SOBOL,
            seed=41,
            **kwargs,
        )

    def test_xva_ledger_keys_and_positivity(self) -> None:
        result = self._run_xva()
        for key in (
            "cva",
            "dva",
            "fca",
            "fba",
            "fva",
            "kva",
            "mva",
            "total_xva",
        ):
            self.assertIn(key, result)
            self.assertIsInstance(result[key], float)

        self.assertGreater(result["cva"], 0.0)
        self.assertGreaterEqual(result["dva"], 0.0)
        self.assertGreater(result["kva"], 0.0)
        self.assertGreater(result["total_xva"], 0.0)
        self.assertAlmostEqual(result["fva"], result["fca"] + result["fba"], places=10)

    def test_xva_matches_manual_aggregation(self) -> None:
        result = self._run_xva()
        exposure = result["exposure"][:, 1:]
        df = result["domestic_df_paths"][:, 1:]
        times = result["times_yrs"]
        np.testing.assert_allclose(times, np.linspace(0.0, 5.0, 11))
        ccy_tenors, ccy_spreads = get_standard_credit_curve()
        cp_pd = CIRHazardRateModel.calibrate_from_spreads(
            ccy_spreads, ccy_tenors
        ).marginal_pd(times[1:])
        own_pd = CIRHazardRateModel.calibrate_from_spreads(
            ccy_spreads * 0.6, ccy_tenors
        ).marginal_pd(times[1:])
        n_paths = result["exposure"].shape[0]
        manual = compute_total_xva(
            exposure=exposure,
            time_steps_yrs=np.diff(times),
            discount_factor=df,
            counterparty_marginal_pd=np.tile(cp_pd, (n_paths, 1)),
            own_marginal_pd=np.tile(own_pd, (n_paths, 1)),
            counterparty_lgd=0.60,
            own_lgd=0.60,
        )
        self.assertAlmostEqual(result["cva"], manual["cva"], places=6)
        self.assertAlmostEqual(result["dva"], manual["dva"], places=6)
        self.assertAlmostEqual(result["total_xva"], manual["total_xva"], places=6)

    def test_exposure_profile_keys_present(self) -> None:
        result = self._run_xva()
        for key in (
            "expected_exposure",
            "epe",
            "max_pfe",
            "pfe_profiles",
            "pfe_99",
        ):
            self.assertIn(key, result)
        self.assertGreater(result["epe"], 0.0)
        self.assertGreaterEqual(result["max_pfe"], result["epe"])

    def test_chunked_aggregation_matches_full(self) -> None:
        full = self._run_xva()
        chunked = self._run_xva(chunk_size=64)
        self.assertAlmostEqual(chunked["cva"], full["cva"], places=10)
        self.assertAlmostEqual(chunked["total_xva"], full["total_xva"], places=10)

    def test_custom_credit_tenors_csv_style(self) -> None:
        result = self._run_xva()
        self.assertIsInstance(result["total_xva"], float)

    def test_mismatched_credit_spreads_raises(self) -> None:
        model = _two_currency_model()
        portfolio = Portfolio(trades=[FXForwardTrade("f", 1.0, 1.2, 2.0)])
        with self.assertRaises(ValueError):
            compute_portfolio_xva(
                portfolio=portfolio,
                fx_model=model,
                maturity_yrs=2.0,
                n_steps=4,
                counterparty_credit_spreads_ann=np.array([0.01, 0.02]),
                own_credit_spreads_ann=np.array([0.01, 0.02]),
            )


if __name__ == "__main__":
    unittest.main()
