"""Unit tests for the automatic-differentiation (dual-number) Greeks engine."""

import math
import typing
import unittest

import numpy as np
import pytest

from xvasim import (
    FXEuropeanOptionTrade,
    FXForwardTrade,
    GarmanKohlhagenFXModel,
    GreeksResult,
    HestonFXModel,
    HullWhite1FModel,
    LGMModel,
    OptionType,
    TwoCurrencyFXModel,
    benchmark_price_foreign_exchange_forward,
    benchmark_price_foreign_exchange_option,
    compute_greeks,
)
from xvasim.greeks import (
    Dual,
    Dual2,
    dual_erf,
    dual_exp,
    dual_log,
    dual_mean,
    dual_norm_cdf,
    dual_relu,
    dual_sqrt,
)


def _as_dual(value: Dual | Dual2 | np.ndarray) -> Dual:
    return typing.cast(Dual, value)


def _as_dual2(value: Dual | Dual2 | np.ndarray) -> Dual2:
    return typing.cast(Dual2, value)


def _as_array(value: Dual | Dual2 | np.ndarray) -> np.ndarray:
    return typing.cast(np.ndarray, value)


def _val(res: dict[str, typing.Any], name: str) -> float:
    return float(res[name])


def _flat_gk_model() -> GarmanKohlhagenFXModel:
    """Flat-rate Garman-Kohlhagen model with all four sensitivity seeds."""
    return GarmanKohlhagenFXModel(
        spot_fx=1.20,
        fx_vol_ann=0.15,
        domestic_rate_ann=0.03,
        foreign_rate_ann=0.01,
    )


def _two_currency_model() -> TwoCurrencyFXModel:
    """Stochastic-rates two-currency model (no scalar-rate rho seeds)."""
    tenors = np.array([0.0, 1.0, 2.0, 3.0])
    domestic = LGMModel(
        kappa_ann=0.03,
        sigma_grid_yrs=np.array([1.0, 3.0]),
        sigma_values_ann=np.array([0.010, 0.011]),
        discount_curve_yrs=tenors,
        discount_factors=np.exp(-0.03 * tenors),
    )
    foreign = HullWhite1FModel(
        a_ann=0.04,
        sigma_ann=0.015,
        discount_curve_yrs=tenors,
        discount_factors=np.exp(-0.01 * tenors),
    )
    return TwoCurrencyFXModel(
        domestic_ir_model=domestic,
        foreign_ir_model=foreign,
        spot_fx=1.20,
        fx_vol_ann=0.15,
        correlation_matrix=np.eye(3),
    )


def _forward_trade() -> FXForwardTrade:
    return FXForwardTrade(
        trade_id="fwd-1",
        notional=1_000_000.0,
        strike_fx=1.21,
        maturity_yrs=1.0,
    )


def _option_trade(option_type: OptionType | str = OptionType.CALL) -> FXEuropeanOptionTrade:
    return FXEuropeanOptionTrade(
        trade_id="opt-1",
        notional=1_000_000.0,
        strike_fx=1.20,
        maturity_yrs=1.0,
        option_type=option_type,
    )


def _benchmark_forward_price(model: typing.Any, trade: FXForwardTrade) -> float:
    return float(
        benchmark_price_foreign_exchange_forward(
            model=model,
            strike=trade.strike_fx,
            maturity_yrs=trade.maturity_yrs,
            notional=trade.notional,
        ).price
    )


def _benchmark_option_price(
    model: typing.Any,
    trade: FXEuropeanOptionTrade,
) -> float:
    return float(
        benchmark_price_foreign_exchange_option(
            model=model,
            strike=trade.strike_fx,
            maturity_yrs=trade.maturity_yrs,
            notional=trade.notional,
            option_type=trade.option_type,
        ).price
    )


class TestFirstOrderAdEngine(unittest.TestCase):
    def test_product_rule_two_seeds(self) -> None:
        x = Dual(3.0, np.array([1.0, 0.0]))
        y = Dual(2.0, np.array([0.0, 1.0]))
        out = x * y
        self.assertAlmostEqual(float(out.v), 6.0, places=12)
        np.testing.assert_allclose(out.g, [2.0, 3.0], rtol=1e-12)

    def test_polynomial_chain_rule(self) -> None:
        t = Dual(2.0, np.array([1.0]))
        out = t**3 + dual_exp(t) + dual_log(t)
        expected = 3 * 4.0 + math.exp(2.0) + 0.5
        self.assertAlmostEqual(float(out.g[0]), expected, places=9)

    def test_division_quotient_rule(self) -> None:
        x = Dual(4.0, np.array([1.0]))
        y = Dual(2.0, np.array([1.0]))
        out = x / y
        self.assertAlmostEqual(float(out.v), 2.0, places=12)
        self.assertAlmostEqual(float(out.g[0]), -0.5, places=9)

    def test_sqrt_derivative(self) -> None:
        x = Dual(9.0, np.array([1.0]))
        out = _as_dual(dual_sqrt(x))
        self.assertAlmostEqual(float(out.v), 3.0, places=12)
        self.assertAlmostEqual(float(out.g[0]), 1.0 / 6.0, places=9)

    def test_norm_cdf_derivative_is_pdf(self) -> None:
        z = 0.7
        out = _as_dual(dual_norm_cdf(Dual(z, np.array([1.0]))))
        val = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        self.assertAlmostEqual(float(out.v), val, places=12)
        self.assertAlmostEqual(float(out.g[0]), pdf, places=10)

    def test_erf_derivative(self) -> None:
        x0 = 0.4
        out = _as_dual(dual_erf(Dual(x0, np.array([1.0]))))
        phi = 2.0 / math.sqrt(math.pi) * math.exp(-x0 * x0)
        self.assertAlmostEqual(float(out.v), math.erf(x0), places=12)
        self.assertAlmostEqual(float(out.g[0]), phi, places=10)

    def test_relu_subgradient_vectorized(self) -> None:
        d = Dual(np.array([-1.0, 2.0]), np.array([[1.0, 2.0], [3.0, 4.0]]))
        out = _as_dual(dual_relu(d))
        np.testing.assert_allclose(out.v, [0.0, 2.0])
        np.testing.assert_allclose(out.g, [[0.0, 0.0], [3.0, 4.0]])

    def test_mean_reduces_path_axis(self) -> None:
        d = Dual(np.array([1.0, 3.0]), np.array([[1.0, 0.0], [0.0, 2.0]]))
        out = _as_dual(dual_mean(d, axis=0))
        self.assertAlmostEqual(float(out.v), 2.0, places=12)
        np.testing.assert_allclose(out.g, [0.5, 1.0], rtol=1e-12)

    def test_scalar_seed_broadcasts_over_draw_array(self) -> None:
        vol = Dual(0.2, np.array([1.0, 0.0]))
        z = np.array([-1.0, 0.5, 2.0])
        out = vol * z
        self.assertEqual(out.v.shape, (3,))
        self.assertEqual(out.g.shape, (3, 2))
        np.testing.assert_allclose(out.v, 0.2 * z)
        np.testing.assert_allclose(out.g[:, 0], z)
        np.testing.assert_allclose(out.g[:, 1], 0.0)

    def test_plain_inputs_pass_through(self) -> None:
        np.testing.assert_allclose(
            _as_array(dual_exp(np.array([0.0, 1.0]))),
            np.exp(np.array([0.0, 1.0])),
        )
        self.assertAlmostEqual(float(_as_array(dual_norm_cdf(0.0))), 0.5, places=12)
        np.testing.assert_allclose(_as_array(dual_relu(np.array([-1.0, 1.0]))), [0.0, 1.0])

    def test_relu_rejects_second_order(self) -> None:
        x = Dual2(1.0, np.array([1.0]), np.array([0.0]))
        with self.assertRaises(TypeError):
            dual_relu(x)

    def test_dual_exponent_rejected(self) -> None:
        x = Dual(2.0, np.array([1.0]))
        with self.assertRaises(TypeError):
            x ** (Dual(3.0, np.array([1.0])))


class TestSecondOrderAdEngine(unittest.TestCase):
    def test_cube_second_derivative(self) -> None:
        x = Dual2(2.0, np.array([1.0]), np.array([0.0]))
        out = x**3
        self.assertAlmostEqual(float(out.v), 8.0, places=12)
        self.assertAlmostEqual(float(out.g[0]), 12.0, places=12)
        self.assertAlmostEqual(float(out.h[0]), 12.0, places=12)

    def test_product_hessian_diagonal(self) -> None:
        x = Dual2(3.0, np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        y = Dual2(2.0, np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        out = x * y
        np.testing.assert_allclose(out.g, [2.0, 3.0], rtol=1e-12)
        np.testing.assert_allclose(out.h, [0.0, 0.0], atol=1e-12)

    def test_exp_second_derivative(self) -> None:
        x = Dual2(1.3, np.array([1.0]), np.array([0.0]))
        out = _as_dual2(dual_exp(x))
        self.assertAlmostEqual(float(out.h[0]), math.exp(1.3), places=9)

    def test_log_second_derivative(self) -> None:
        x = Dual2(1.1, np.array([1.0]), np.array([0.0]))
        out = _as_dual2(dual_log(x) + 2 * x)
        self.assertAlmostEqual(float(out.h[0]), -1.0 / 1.21, places=9)

    def test_norm_cdf_gamma_is_negative_z_pdf(self) -> None:
        z = 0.7
        x = Dual2(z, np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        out = _as_dual2(dual_norm_cdf(x))
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        np.testing.assert_allclose(out.g, [pdf, 0.0], rtol=1e-10)
        np.testing.assert_allclose(out.h, [-z * pdf, 0.0], rtol=1e-10)

    def test_erf_second_derivative(self) -> None:
        x0 = 0.4
        x = Dual2(x0, np.array([1.0]), np.array([0.0]))
        out = _as_dual2(dual_erf(x))
        phi = 2.0 / math.sqrt(math.pi) * math.exp(-x0 * x0)
        self.assertAlmostEqual(float(out.h[0]), -2.0 * x0 * phi, places=9)

    def test_second_order_pathwise_shapes(self) -> None:
        d = Dual2(np.array([1.0, 2.0]), np.array([[1.0], [2.0]]), np.array([[0.0], [0.0]]))
        out = d * d
        self.assertEqual(out.g.shape, (2, 1))
        np.testing.assert_allclose(out.v, [1.0, 4.0])
        np.testing.assert_allclose(out.g.flatten(), [2.0, 8.0])
        np.testing.assert_allclose(out.h.flatten(), [2.0, 8.0])


class TestComputeGreeksErrors(unittest.TestCase):
    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_greeks(_forward_trade(), _flat_gk_model(), method="bump_and_reval")

    def test_unsupported_trade_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            compute_greeks(typing.cast(typing.Any, "not-a-trade"), _flat_gk_model())

    def test_model_without_spot_fx_raises(self) -> None:
        class NoSpotModel:
            model_name = "no-spot"

        model = typing.cast(typing.Any, NoSpotModel())
        with self.assertRaises(ValueError):
            compute_greeks(_forward_trade(), model)

    def test_heston_option_closed_form_raises(self) -> None:
        heston = HestonFXModel(
            spot_fx=1.20,
            v_0=0.0225,
            kappa_ann=1.0,
            theta_ann=0.0225,
            sigma_v_ann=0.20,
            rho=0.0,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.01,
        )
        with self.assertRaises(NotImplementedError):
            compute_greeks(_option_trade(), heston)

    def test_monte_carlo_requires_gk_model(self) -> None:
        with self.assertRaises(NotImplementedError):
            compute_greeks(
                _forward_trade(),
                _two_currency_model(),
                method="monte_carlo",
            )


class TestClosedFormForwardGreeks(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _flat_gk_model()
        self.trade = _forward_trade()

    def test_price_matches_benchmark(self) -> None:
        res = compute_greeks(self.trade, self.model)
        self.assertAlmostEqual(
            res.price,
            _benchmark_forward_price(self.model, self.trade),
            places=6,
        )

    def test_delta_is_notional_times_foreign_df(self) -> None:
        res = compute_greeks(self.trade, self.model)
        expected = self.trade.notional * float(
            self.model.foreign_discount_factor(self.trade.maturity_yrs)
        )
        self.assertAlmostEqual(_val(res, "delta"), expected, places=3)

    def test_gamma_and_vega_are_zero(self) -> None:
        res = compute_greeks(self.trade, self.model)
        self.assertAlmostEqual(_val(res, "gamma"), 0.0, places=9)
        self.assertAlmostEqual(_val(res, "vega"), 0.0, places=9)

    def test_rho_matches_rate_finite_difference(self) -> None:
        res = compute_greeks(self.trade, self.model)
        h = 1e-5

        def price(rd: float, rf: float) -> float:
            bumped = GarmanKohlhagenFXModel(
                spot_fx=1.20,
                fx_vol_ann=0.15,
                domestic_rate_ann=rd,
                foreign_rate_ann=rf,
            )
            return _benchmark_forward_price(bumped, self.trade)

        rho_d_fd = (price(0.03 + h, 0.01) - price(0.03 - h, 0.01)) / (2 * h)
        rho_f_fd = (price(0.03, 0.01 + h) - price(0.03, 0.01 - h)) / (2 * h)
        self.assertAlmostEqual(_val(res, "rho_domestic"), rho_d_fd, places=1)
        self.assertAlmostEqual(_val(res, "rho_foreign"), rho_f_fd, places=1)


class TestClosedFormOptionGreeks(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _flat_gk_model()
        self.call = _option_trade(OptionType.CALL)
        self.put = _option_trade(OptionType.PUT)

    def assert_greeks_close(self, actual: float, expected: float) -> None:
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=1e-4, abs_tol=1e-6),
            f"{actual} != {expected}",
        )

    def _fd_greeks(self, trade: FXEuropeanOptionTrade) -> dict[str, float]:
        h = 1e-5
        spot, vol, rd, rf = 1.20, 0.15, 0.03, 0.01

        def price(s: float, v: float, rd_: float, rf_: float) -> float:
            model = GarmanKohlhagenFXModel(
                spot_fx=s,
                fx_vol_ann=v,
                domestic_rate_ann=rd_,
                foreign_rate_ann=rf_,
            )
            return _benchmark_option_price(model, trade)

        delta = (price(spot + h, vol, rd, rf) - price(spot - h, vol, rd, rf)) / (2 * h)
        gamma = (
            price(spot + h, vol, rd, rf)
            - 2 * price(spot, vol, rd, rf)
            + price(spot - h, vol, rd, rf)
        ) / h**2
        vega = (price(spot, vol + h, rd, rf) - price(spot, vol - h, rd, rf)) / (2 * h)
        rho_d = (price(spot, vol, rd + h, rf) - price(spot, vol, rd - h, rf)) / (2 * h)
        rho_f = (price(spot, vol, rd, rf + h) - price(spot, vol, rd, rf - h)) / (2 * h)
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "rho_domestic": rho_d,
            "rho_foreign": rho_f,
        }

    def test_call_greeks_match_finite_differences(self) -> None:
        res = compute_greeks(self.call, self.model)
        fd = self._fd_greeks(self.call)
        self.assertAlmostEqual(
            res.price,
            _benchmark_option_price(self.model, self.call),
            places=6,
        )
        for name in ("delta", "gamma", "vega", "rho_domestic", "rho_foreign"):
            with self.subTest(name=name):
                self.assert_greeks_close(float(res[name]), fd[name])

    def test_put_greeks_match_finite_differences(self) -> None:
        res = compute_greeks(self.put, self.model)
        fd = self._fd_greeks(self.put)
        for name in ("delta", "gamma", "vega"):
            with self.subTest(name=name):
                self.assert_greeks_close(float(res[name]), fd[name])
        self.assertLess(_val(res, "delta"), 0.0)

    def test_closed_form_alias_is_autodiff(self) -> None:
        ad = compute_greeks(self.call, self.model, method="autodiff")
        cf = compute_greeks(self.call, self.model, method="closed_form")
        self.assertAlmostEqual(ad.price, cf.price, places=12)
        self.assertAlmostEqual(_val(ad, "delta"), _val(cf, "delta"), places=12)
        self.assertEqual(ad.method, "autodiff")


class TestCurveModelAndTwoCurrencyGreeks(unittest.TestCase):
    def setUp(self) -> None:
        self.fwd = _forward_trade()

    def test_gk_with_curves_excludes_rho(self) -> None:
        tenors = np.array([0.0, 1.0])
        model = GarmanKohlhagenFXModel(
            spot_fx=1.20,
            fx_vol_ann=0.15,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.01,
            discount_curve_domestic_yrs=tenors,
            discount_factors_domestic=np.exp(-0.03 * tenors),
            discount_curve_foreign_yrs=tenors,
            discount_factors_foreign=np.exp(-0.01 * tenors),
        )
        res = compute_greeks(self.fwd, model)
        expected_delta = self.fwd.notional * float(
            model.foreign_discount_factor(self.fwd.maturity_yrs)
        )
        self.assertAlmostEqual(_val(res, "delta"), expected_delta, places=3)
        self.assertIsNone(res.rho_domestic)
        self.assertIsNone(res.rho_foreign)
        self.assertNotIn("domestic_rate_ann", res.available_parameters)

    def test_two_currency_forward_has_delta_gamma_vega_no_rho(self) -> None:
        model = _two_currency_model()
        res = compute_greeks(self.fwd, model)
        self.assertAlmostEqual(
            res.price,
            _benchmark_forward_price(model, self.fwd),
            places=6,
        )
        self.assertAlmostEqual(_val(res, "gamma"), 0.0, places=9)
        self.assertAlmostEqual(_val(res, "vega"), 0.0, places=9)
        self.assertIsNone(res.rho_domestic)
        self.assertIsNone(res.rho_foreign)

    def test_two_currency_option_matches_benchmark_price(self) -> None:
        model = _two_currency_model()
        call = _option_trade(OptionType.CALL)
        res = compute_greeks(call, model)
        self.assertAlmostEqual(
            res.price,
            _benchmark_option_price(model, call),
            places=6,
        )
        self.assertIsNotNone(res.gamma)
        self.assertIsNone(res.rho_domestic)
        self.assertIn("spot_fx", res.available_parameters)
        self.assertIn("fx_vol_ann", res.available_parameters)

    def test_heston_forward_has_no_vega(self) -> None:
        heston = HestonFXModel(
            spot_fx=1.20,
            v_0=0.0225,
            kappa_ann=1.0,
            theta_ann=0.0225,
            sigma_v_ann=0.20,
            rho=0.0,
            domestic_rate_ann=0.03,
            foreign_rate_ann=0.01,
        )
        res = compute_greeks(self.fwd, heston)
        self.assertAlmostEqual(
            res.price,
            _benchmark_forward_price(heston, self.fwd),
            places=6,
        )
        self.assertIsNone(res.vega)
        self.assertIsNotNone(res.rho_domestic)
        self.assertIsNotNone(res.rho_foreign)


class TestMonteCarloGreeks(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _flat_gk_model()
        self.fwd = _forward_trade()

    def test_forward_mc_matches_closed_form(self) -> None:
        cf = compute_greeks(self.fwd, self.model)
        mc = compute_greeks(
            self.fwd,
            self.model,
            method="monte_carlo",
            n_paths=16_384,
            random_type="sobol",
            seed=42,
            scramble=True,
        )
        self.assertEqual(mc.method, "monte_carlo")
        self.assertIsNotNone(mc.std_error)
        self.assertAlmostEqual(mc.price / cf.price, 1.0, delta=5e-3)
        self.assertAlmostEqual(_val(mc, "delta") / _val(cf, "delta"), 1.0, delta=5e-3)
        self.assertAlmostEqual(_val(mc, "gamma"), 0.0, places=6)
        self.assertLess(abs(_val(mc, "vega")), 50.0)

    def test_option_mc_delta_vega_rho_and_degenerate_gamma(self) -> None:
        call = _option_trade(OptionType.CALL)
        cf = compute_greeks(call, self.model)
        mc = compute_greeks(
            call,
            self.model,
            method="monte_carlo",
            n_paths=16_384,
            random_type="sobol",
            seed=42,
            scramble=True,
        )
        self.assertAlmostEqual(mc.price / cf.price, 1.0, delta=5e-3)
        self.assertAlmostEqual(_val(mc, "delta") / _val(cf, "delta"), 1.0, delta=5e-3)
        self.assertAlmostEqual(_val(mc, "vega") / _val(cf, "vega"), 1.0, delta=5e-3)
        self.assertAlmostEqual(_val(mc, "rho_domestic") / _val(cf, "rho_domestic"), 1.0, delta=5e-3)
        self.assertAlmostEqual(_val(mc, "rho_foreign") / _val(cf, "rho_foreign"), 1.0, delta=5e-3)
        self.assertIsNone(mc.gamma)


class TestGreeksResultContainer(unittest.TestCase):
    def test_dict_and_attribute_access_agree(self) -> None:
        res = compute_greeks(_forward_trade(), _flat_gk_model())
        self.assertIsInstance(res, GreeksResult)
        self.assertEqual(res["delta"], res.delta)
        self.assertEqual(res["vega"], res.vega)
        self.assertEqual(res["rho_domestic"], res.rho_domestic)
        res.extra = 123.0
        self.assertEqual(res["extra"], 123.0)
        del res.extra

    def test_missing_attribute_raises(self) -> None:
        res = compute_greeks(_forward_trade(), _flat_gk_model())
        with self.assertRaises(AttributeError):
            _ = res.not_a_real_field

    def test_option_type_recorded(self) -> None:
        call = compute_greeks(_option_trade("call"), _flat_gk_model())
        self.assertTrue(str(call.option_type).endswith("call"))
        put = compute_greeks(_option_trade("put"), _flat_gk_model())
        self.assertTrue(str(put.option_type).endswith("put"))


class TestComputeGreeksMontecarloParamValidation(unittest.TestCase):
    @pytest.mark.slow
    def test_large_path_count_is_reproducible(self) -> None:
        fwd = _forward_trade()
        g1 = compute_greeks(
            fwd,
            _flat_gk_model(),
            method="monte_carlo",
            n_paths=65_536,
            random_type="sobol",
            seed=7,
        )
        g2 = compute_greeks(
            fwd,
            _flat_gk_model(),
            method="monte_carlo",
            n_paths=65_536,
            random_type="sobol",
            seed=7,
        )
        self.assertAlmostEqual(g1.price, g2.price, places=10)
        self.assertAlmostEqual(_val(g1, "delta"), _val(g2, "delta"), places=8)


class TestAdEdgeCases(unittest.TestCase):
    """Covers reverse operators and plain-array branches not exercised elsewhere."""

    def test_dual_reverse_operators(self) -> None:
        d = Dual(3.0, np.array([1.0]))
        self.assertAlmostEqual(float((2.0 + d).v), 5.0)
        self.assertAlmostEqual(float((5.0 - d).v), 2.0)
        self.assertAlmostEqual(float((2.0 * d).v), 6.0)
        self.assertAlmostEqual(float((6.0 / d).v), 2.0)

    def test_dual_neg(self) -> None:
        d = Dual(3.0, np.array([1.0]))
        out = -d
        self.assertAlmostEqual(float(out.v), -3.0)
        self.assertAlmostEqual(float(out.g[0]), -1.0)

    def test_dual2_reverse_operators(self) -> None:
        d = Dual2(2.0, np.array([1.0]), np.array([0.0]))
        self.assertAlmostEqual(float((3.0 + d).v), 5.0)
        self.assertAlmostEqual(float((5.0 - d).v), 3.0)
        self.assertAlmostEqual(float((3.0 * d).v), 6.0)
        self.assertAlmostEqual(float((6.0 / d).v), 3.0)

    def test_dual2_add_plain_array(self) -> None:
        d = Dual2(np.array([1.0, 2.0]), np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        out = d + 5.0
        np.testing.assert_allclose(out.v, [6.0, 7.0])

    def test_dual2_sub(self) -> None:
        d = Dual2(4.0, np.array([1.0]), np.array([2.0]))
        out = d - 1.0
        self.assertAlmostEqual(float(out.v), 3.0)
        self.assertAlmostEqual(float(out.g[0]), 1.0)

    def test_dual_pow_scalar(self) -> None:
        d = Dual(3.0, np.array([1.0]))
        out = d ** 2
        self.assertAlmostEqual(float(out.v), 9.0)
        self.assertAlmostEqual(float(out.g[0]), 6.0)

    def test_plain_array_pow(self) -> None:
        arr = np.array([2.0, 3.0])
        out = _as_array(dual_exp(arr))
        np.testing.assert_allclose(out, np.exp(arr))
        out2 = _as_array(dual_log(arr))
        np.testing.assert_allclose(out2, np.log(arr))
        out3 = _as_array(dual_erf(arr))
        np.testing.assert_allclose(out3, np.vectorize(math.erf)(arr))
