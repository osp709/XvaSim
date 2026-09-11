"""Automatic-differentiation Greeks vs analytical and finite-difference benchmarks."""

import math
import unittest

import pytest

from xvasim import (
    FXEuropeanOptionTrade,
    GarmanKohlhagenFXModel,
    OptionType,
    benchmark_price_foreign_exchange_option,
    compute_greeks,
)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _black76_analytics(
    spot: float,
    strike: float,
    maturity_yrs: float,
    vol: float,
    rd: float,
    rf: float,
    notional: float,
) -> dict[str, float]:
    """Closed-form Black-76 call greeks independent of the AD engine."""
    df_d = math.exp(-rd * maturity_yrs)
    df_f = math.exp(-rf * maturity_yrs)
    fwd = spot * df_f / df_d
    sqt = math.sqrt(maturity_yrs)
    d1 = (math.log(fwd / strike) + 0.5 * vol * vol * maturity_yrs) / (vol * sqt)
    d2 = d1 - vol * sqt
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    price = notional * df_d * (fwd * _norm_cdf(d1) - strike * _norm_cdf(d2))
    delta = notional * df_f * _norm_cdf(d1)
    gamma = notional * df_f * df_f / df_d * pdf / (fwd * vol * sqt)
    vega = notional * df_d * fwd * pdf * sqt
    rho_domestic = notional * maturity_yrs * df_d * strike * _norm_cdf(d2)
    rho_foreign = -notional * maturity_yrs * df_d * fwd * _norm_cdf(d1)
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "rho_domestic": rho_domestic,
        "rho_foreign": rho_foreign,
    }


@pytest.mark.benchmark
class TestAutodiffGreeksConvergence(unittest.TestCase):
    """Verify that AD Greeks agree with closed-form Black-76 and shrink-step FD."""

    def _call_trade(self) -> FXEuropeanOptionTrade:
        return FXEuropeanOptionTrade(
            trade_id="opt",
            notional=1_000_000.0,
            strike_fx=1.20,
            maturity_yrs=1.0,
            option_type=OptionType.CALL,
        )

    def test_ad_greeks_match_black76_analytics(self) -> None:
        model = GarmanKohlhagenFXModel(
            spot_fx=1.20,
            fx_vol_ann=0.15,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.01,
        )
        res = compute_greeks(self._call_trade(), model)
        exact = _black76_analytics(
            spot=1.20,
            strike=1.20,
            maturity_yrs=1.0,
            vol=0.15,
            rd=0.03,
            rf=0.01,
            notional=1_000_000.0,
        )
        for name in ("price", "delta", "gamma", "vega", "rho_domestic", "rho_foreign"):
            with self.subTest(name=name):
                self.assertTrue(
                    math.isclose(float(res[name]), exact[name], rel_tol=1e-9, abs_tol=1e-6),
                    f"AD {name}={float(res[name]):.9f} != analytic {exact[name]:.9f}",
                )

    def test_delta_and_vega_finite_difference_convergence(self) -> None:
        model = GarmanKohlhagenFXModel(
            spot_fx=1.20,
            fx_vol_ann=0.15,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.01,
        )
        res = compute_greeks(self._call_trade(), model)
        self.assertIsNotNone(res.delta)
        self.assertIsNotNone(res.vega)
        delta_ad: float = res.delta
        vega_ad: float = res.vega

        def bench(s: float, v: float) -> float:
            up = GarmanKohlhagenFXModel(
                spot_fx=s,
                fx_vol_ann=v,
                domestic_rate_ann=0.03,
                foreign_rate_ann=0.01,
            )
            return float(
                benchmark_price_foreign_exchange_option(
                    model=up,
                    strike=1.20,
                    maturity_yrs=1.0,
                    notional=1_000_000.0,
                    option_type="call",
                ).price
            )

        prev_delta_err: float | None = None
        prev_vega_err: float | None = None
        for h in (1e-3, 1e-4, 1e-5, 1e-6):
            delta_fd = (bench(1.20 + h, 0.15) - bench(1.20 - h, 0.15)) / (2 * h)
            vega_fd = (bench(1.20, 0.15 + h) - bench(1.20, 0.15 - h)) / (2 * h)
            delta_err = abs(delta_ad - delta_fd)
            vega_err = abs(vega_ad - vega_fd)
            if prev_delta_err is not None and prev_vega_err is not None:
                self.assertLess(delta_err, prev_delta_err * 5 + 1e-3)
                self.assertLess(vega_err, prev_vega_err * 5 + 1e-3)
            prev_delta_err = delta_err
            prev_vega_err = vega_err

    def test_gamma_finite_difference_converges_from_above(self) -> None:
        """AD gamma must be closer to the second-difference limit than FD truncation."""
        res = compute_greeks(self._call_trade(), _gk())
        self.assertIsNotNone(res.gamma)
        gamma_ad: float = res.gamma
        coarse = _central_gamma(1e-2)
        fine = _central_gamma(1e-4)
        self.assertLess(abs(gamma_ad - fine), abs(gamma_ad - coarse))


def _gk() -> GarmanKohlhagenFXModel:
    return GarmanKohlhagenFXModel(
        spot_fx=1.20,
        fx_vol_ann=0.15,
        domestic_rate_ann=0.03,
        foreign_rate_ann=0.01,
    )


def _central_gamma(h: float) -> float:
    def c(s: float) -> float:
        f = s * math.exp(0.02)
        d1 = (math.log(f / 1.20) + 0.5 * 0.0225) / 0.15
        d2 = d1 - 0.15
        return 1_000_000.0 * math.exp(-0.03) * (f * _norm_cdf(d1) - 1.20 * _norm_cdf(d2))

    return (c(1.20 + h) - 2 * c(1.20) + c(1.20 - h)) / (h * h)
