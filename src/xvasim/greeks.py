"""Automatic-differentiation Greeks for FX derivatives.

This module provides a dependency-free forward-mode automatic-differentiation
(AD) engine built on dual numbers (:class:`Dual` for first-order sensitivities
and :class:`Dual2` for second-order / Gamma) together with :func:`compute_greeks`,
which differentiates the analytical closed-form FX forward / Black-76 option
prices, or a pathwise Monte Carlo payoff evaluated on frozen quasi-random draws.

Sensitivities are labelled consistently with the library units and naming
conventions:

- ``delta`` / ``gamma``: first / second derivative of the trade price with
  respect to ``spot_fx``;
- ``vega``: derivative with respect to ``fx_vol_ann``;
- ``rho_domestic`` / ``rho_foreign``: derivatives with respect to the constant
  ``domestic_rate_ann`` / ``foreign_rate_ann`` model parameters.

Parameters that a model does not expose (e.g. scalar rates on
:class:`~xvasim.models.fx.TwoCurrencyFXModel`, or ``fx_vol_ann`` on
:class:`~xvasim.models.fx.HestonFXModel`) are reported as ``None`` and listed
in ``available_parameters``.
"""

from __future__ import annotations

import math
import typing

import numpy as np
from scipy.special import erf

from .models.base import FXModel
from .models.fx.garman_kohlhagen import GarmanKohlhagenFXModel
from .portfolio import FXEuropeanOptionTrade, FXForwardTrade, _option_is_call
from .qmc import RandomSequenceType, generate_normal_draws

if typing.TYPE_CHECKING:
    from typing import Any

__all__ = ["Dual", "Dual2", "GreeksResult", "compute_greeks"]


# ---------------------------------------------------------------------------
# Forward-mode automatic differentiation engine (dual numbers)
# ---------------------------------------------------------------------------


def _bx(v: np.ndarray) -> np.ndarray:
    """Expand a dual value so it broadcasts against the trailing seed axis."""
    return v if v.ndim == 0 else v[..., np.newaxis]


class Dual:
    """First-order forward-mode dual number: value plus gradient vector.

    Attributes:
        v: Primal value (0-d or 1-d array).
        g: Gradient vector; shape ``v.shape + (n_seeds,)``.
    """

    __slots__ = ("g", "v")
    __array_priority__: typing.ClassVar[int] = 1000

    def __init__(self, v: np.ndarray | float, g: np.ndarray | float) -> None:
        self.v = np.asarray(v, dtype=np.float64)
        self.g = np.asarray(g, dtype=np.float64)

    def __add__(self, other: Any) -> Dual:
        return typing.cast(Dual, _add(self, other))

    def __radd__(self, other: Any) -> Dual:
        return typing.cast(Dual, _add(other, self))

    def __sub__(self, other: Any) -> Dual:
        return typing.cast(Dual, _add(self, _neg(other)))

    def __rsub__(self, other: Any) -> Dual:
        return typing.cast(Dual, _add(other, _neg(self)))

    def __mul__(self, other: Any) -> Dual:
        return typing.cast(Dual, _mul(self, other))

    def __rmul__(self, other: Any) -> Dual:
        return typing.cast(Dual, _mul(other, self))

    def __truediv__(self, other: Any) -> Dual:
        return typing.cast(Dual, _div(self, other))

    def __rtruediv__(self, other: Any) -> Dual:
        return typing.cast(Dual, _div(other, self))

    def __pow__(self, other: Any) -> Dual:
        return typing.cast(Dual, _pow(self, other))

    def __neg__(self) -> Dual:
        return typing.cast(Dual, _mul(self, -1.0))


class Dual2:
    """Second-order forward-mode dual number for gamma computation.

    Attributes:
        v: Primal value (0-d or 1-d array).
        g: First derivative vector; shape ``v.shape + (n_seeds,)``.
        h: Second derivative diagonal (``d2f/dx_i^2``); shape ``v.shape + (n_seeds,)``.
    """

    __slots__ = ("g", "h", "v")
    __array_priority__: typing.ClassVar[int] = 1001

    def __init__(
        self,
        v: np.ndarray | float,
        g: np.ndarray | float,
        h: np.ndarray | float,
    ) -> None:
        self.v = np.asarray(v, dtype=np.float64)
        self.g = np.asarray(g, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)

    def __add__(self, other: Any) -> Dual2:
        return typing.cast(Dual2, _add(self, other))

    def __radd__(self, other: Any) -> Dual2:
        return typing.cast(Dual2, _add(other, self))

    def __sub__(self, other: Any) -> Dual2:
        return typing.cast(Dual2, _add(self, _neg(other)))

    def __rsub__(self, other: Any) -> Dual2:
        return typing.cast(Dual2, _add(other, _neg(self)))

    def __mul__(self, other: Any) -> Dual2:
        return typing.cast(Dual2, _mul(self, other))

    def __rmul__(self, other: Any) -> Dual2:
        return typing.cast(Dual2, _mul(other, self))

    def __truediv__(self, other: Any) -> Dual2:
        return typing.cast(Dual2, _div(self, other))

    def __rtruediv__(self, other: Any) -> Dual2:
        return typing.cast(Dual2, _div(other, self))

    def __pow__(self, other: Any) -> Dual2:
        return typing.cast(Dual2, _pow(self, other))

    def __neg__(self) -> Dual2:
        return typing.cast(Dual2, _mul(self, -1.0))


def _to_d1(x: Any) -> Dual | np.ndarray:
    return x if isinstance(x, Dual) else np.asarray(x, dtype=np.float64)


def _to_d2(x: Any) -> Dual2 | np.ndarray:
    if isinstance(x, Dual2):
        return x
    if isinstance(x, Dual):
        return Dual2(x.v, x.g, np.zeros_like(x.g))
    return np.asarray(x, dtype=np.float64)


def _add(a: Any, b: Any) -> Dual | Dual2 | np.ndarray:
    if isinstance(a, Dual2) or isinstance(b, Dual2):
        return _add2(_to_d2(a), _to_d2(b))
    if isinstance(a, Dual) or isinstance(b, Dual):
        return _add1(_to_d1(a), _to_d1(b))
    return np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)


def _add1(a: Dual | np.ndarray, b: Dual | np.ndarray) -> Dual | np.ndarray:
    a_d = isinstance(a, Dual)
    b_d = isinstance(b, Dual)
    g: np.ndarray
    if a_d and b_d:
        g = a.g + b.g
    elif a_d:
        assert isinstance(a, Dual)
        g = a.g
    else:
        g = typing.cast(Dual, b).g
    v = (a.v if a_d else np.asarray(a)) + (b.v if b_d else np.asarray(b))
    return Dual(np.asarray(v, dtype=np.float64), np.asarray(g, dtype=np.float64))


def _add2(a: Dual2 | np.ndarray, b: Dual2 | np.ndarray) -> Dual2 | np.ndarray:
    a_d = isinstance(a, Dual2)
    b_d = isinstance(b, Dual2)
    if a_d and b_d:
        return Dual2(a.v + b.v, a.g + b.g, a.h + b.h)
    if a_d:
        assert isinstance(a, Dual2)
        bv = np.asarray(b)
        return Dual2(a.v + bv, a.g, a.h)
    av = np.asarray(a)
    b_d2 = typing.cast(Dual2, b)
    return Dual2(av + b_d2.v, b_d2.g, b_d2.h)


def _mul(a: Any, b: Any) -> Dual | Dual2 | np.ndarray:
    if isinstance(a, Dual2) or isinstance(b, Dual2):
        return _mul2(_to_d2(a), _to_d2(b))
    if isinstance(a, Dual) or isinstance(b, Dual):
        return _mul1(_to_d1(a), _to_d1(b))
    return np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)


def _mul1(a: Dual | np.ndarray, b: Dual | np.ndarray) -> Dual | np.ndarray:
    a_d = isinstance(a, Dual)
    b_d = isinstance(b, Dual)
    if a_d and b_d:
        v = a.v * b.v
        g = a.g * _bx(b.v) + b.g * _bx(a.v)
    elif a_d:
        assert isinstance(a, Dual)
        bv = np.asarray(b)
        v = a.v * bv
        g = a.g * _bx(bv)
    else:
        av = np.asarray(a)
        v = av * typing.cast(Dual, b).v
        g = _bx(av) * typing.cast(Dual, b).g
    return Dual(np.asarray(v, dtype=np.float64), np.asarray(g, dtype=np.float64))


def _mul2(a: Dual2 | np.ndarray, b: Dual2 | np.ndarray) -> Dual2 | np.ndarray:
    a_d = isinstance(a, Dual2)
    b_d = isinstance(b, Dual2)
    if a_d and b_d:
        v = a.v * b.v
        g = a.g * _bx(b.v) + b.g * _bx(a.v)
        h = a.h * _bx(b.v) + 2.0 * a.g * b.g + b.h * _bx(a.v)
    elif a_d:
        assert isinstance(a, Dual2)
        bv = np.asarray(b)
        v = a.v * bv
        g = a.g * _bx(bv)
        h = a.h * _bx(bv)
    else:
        av = np.asarray(a)
        v = av * typing.cast(Dual2, b).v
        g = _bx(av) * typing.cast(Dual2, b).g
        h = _bx(av) * typing.cast(Dual2, b).h
    return Dual2(
        np.asarray(v, dtype=np.float64),
        np.asarray(g, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
    )


def _neg(a: Any) -> Dual | Dual2 | np.ndarray:
    return _mul(a, -1.0)


def _sub(a: Any, b: Any) -> Dual | Dual2 | np.ndarray:
    return _add(a, _neg(b))


def _recip1(x: Dual) -> Dual:
    g = x.g * _bx(-1.0 / (x.v * x.v))
    return Dual(
        np.asarray(1.0 / x.v, dtype=np.float64),
        np.asarray(g, dtype=np.float64),
    )


def _recip2(x: Dual2) -> Dual2:
    f1 = -1.0 / (x.v * x.v)
    f2 = 2.0 / (x.v * x.v * x.v)
    g = x.g * _bx(f1)
    h = x.h * _bx(f1) + (x.g * x.g) * _bx(f2)
    return Dual2(np.asarray(1.0 / x.v, dtype=np.float64), g, h)


def _div(a: Any, b: Any) -> Dual | Dual2 | np.ndarray:
    if isinstance(b, Dual2):
        return _mul(a, _recip2(b))
    if isinstance(b, Dual):
        return _mul(a, _recip1(b))
    return _mul(a, np.asarray(1.0 / np.asarray(b, dtype=np.float64), dtype=np.float64))


def _pow(x: Any, p: Any) -> Dual | Dual2 | np.ndarray:
    if isinstance(p, (Dual, Dual2)):
        msg = "dual-typed exponents are not supported"
        raise TypeError(msg)
    exponent = float(p)
    if isinstance(x, Dual2):
        f1 = exponent * x.v ** (exponent - 1.0)
        f2 = exponent * (exponent - 1.0) * x.v ** (exponent - 2.0)
        g = x.g * _bx(f1)
        h = x.h * _bx(f1) + (x.g * x.g) * _bx(f2)
        return Dual2(np.asarray(x.v**exponent, dtype=np.float64), g, h)
    if isinstance(x, Dual):
        f1 = exponent * x.v ** (exponent - 1.0)
        return Dual(
            np.asarray(x.v**exponent, dtype=np.float64),
            np.asarray(x.g * _bx(f1), dtype=np.float64),
        )
    return np.asarray(np.asarray(x, dtype=np.float64) ** exponent, dtype=np.float64)


def dual_exp(x: Any) -> Dual | Dual2 | np.ndarray:
    xv = np.asarray(x.v if isinstance(x, (Dual, Dual2)) else x, dtype=np.float64)
    e = np.exp(xv)
    if isinstance(x, Dual2):
        g = _bx(e) * x.g
        h = _bx(e) * (x.h + x.g * x.g)
        return Dual2(np.asarray(e, dtype=np.float64), g, h)
    if isinstance(x, Dual):
        return Dual(
            np.asarray(e, dtype=np.float64),
            np.asarray(_bx(e) * x.g, dtype=np.float64),
        )
    return np.asarray(e, dtype=np.float64)


def dual_log(x: Any) -> Dual | Dual2 | np.ndarray:
    xv = np.asarray(x.v if isinstance(x, (Dual, Dual2)) else x, dtype=np.float64)
    inv = 1.0 / xv
    if isinstance(x, Dual2):
        g = _bx(inv) * x.g
        h = _bx(inv) * x.h + _bx(-1.0 / (xv * xv)) * (x.g * x.g)
        return Dual2(np.asarray(np.log(xv), dtype=np.float64), g, h)
    if isinstance(x, Dual):
        return Dual(
            np.asarray(np.log(xv), dtype=np.float64),
            np.asarray(_bx(inv) * x.g, dtype=np.float64),
        )
    return np.asarray(np.log(xv), dtype=np.float64)


def dual_sqrt(x: Any) -> Dual | Dual2 | np.ndarray:
    return _pow(x, 0.5)


def dual_erf(x: Any) -> Dual | Dual2 | np.ndarray:
    xv = np.asarray(x.v if isinstance(x, (Dual, Dual2)) else x, dtype=np.float64)
    phi = (2.0 / math.sqrt(math.pi)) * np.exp(-xv * xv)
    val = erf(xv)
    if isinstance(x, Dual2):
        g = _bx(phi) * x.g
        h = _bx(phi) * x.h + _bx(-2.0 * xv * phi) * (x.g * x.g)
        return Dual2(np.asarray(val, dtype=np.float64), g, h)
    if isinstance(x, Dual):
        return Dual(
            np.asarray(val, dtype=np.float64),
            np.asarray(_bx(phi) * x.g, dtype=np.float64),
        )
    return np.asarray(val, dtype=np.float64)


def dual_norm_cdf(x: Any) -> Dual | Dual2 | np.ndarray:
    xv = np.asarray(x.v if isinstance(x, (Dual, Dual2)) else x, dtype=np.float64)
    pdf = np.exp(-0.5 * xv * xv) / math.sqrt(2.0 * math.pi)
    val = 0.5 * (1.0 + erf(xv / math.sqrt(2.0)))
    if isinstance(x, Dual2):
        g = _bx(pdf) * x.g
        h = _bx(pdf) * x.h + _bx(-xv * pdf) * (x.g * x.g)
        return Dual2(np.asarray(val, dtype=np.float64), g, h)
    if isinstance(x, Dual):
        return Dual(
            np.asarray(val, dtype=np.float64),
            np.asarray(_bx(pdf) * x.g, dtype=np.float64),
        )
    return np.asarray(val, dtype=np.float64)


def dual_relu(x: Any) -> Dual | np.ndarray:
    if isinstance(x, Dual2):
        msg = "relu is only supported for first-order Dual numbers"
        raise TypeError(msg)
    if isinstance(x, Dual):
        mask = x.v > 0.0
        return Dual(
            np.asarray(np.maximum(x.v, 0.0), dtype=np.float64),
            np.asarray(_bx(mask) * x.g, dtype=np.float64),
        )
    return np.asarray(np.maximum(x, 0.0), dtype=np.float64)


def dual_mean(x: Any, axis: int = 0) -> Dual | Dual2 | np.ndarray:
    if isinstance(x, Dual2):
        if x.v.ndim == 0:
            return x
        return Dual2(
            np.asarray(np.mean(x.v, axis=axis), dtype=np.float64),
            np.asarray(np.mean(x.g, axis=axis), dtype=np.float64),
            np.asarray(np.mean(x.h, axis=axis), dtype=np.float64),
        )
    if isinstance(x, Dual):
        if x.v.ndim == 0:
            return x
        return Dual(
            np.asarray(np.mean(x.v, axis=axis), dtype=np.float64),
            np.asarray(np.mean(x.g, axis=axis), dtype=np.float64),
        )
    return np.asarray(np.mean(x, axis=axis), dtype=np.float64)


# ---------------------------------------------------------------------------
# Price functions
# ---------------------------------------------------------------------------


def _df_domestic(
    params: dict[str, Any],
    fx_model: FXModel,
    maturity_yrs: float,
) -> Any:
    if "domestic_rate_ann" in params:
        return dual_exp(_mul(_neg(params["domestic_rate_ann"]), maturity_yrs))
    return np.asarray(fx_model.domestic_discount_factor(maturity_yrs), dtype=np.float64)


def _df_foreign(
    params: dict[str, Any],
    fx_model: FXModel,
    maturity_yrs: float,
) -> Any:
    if "foreign_rate_ann" in params:
        return dual_exp(_mul(_neg(params["foreign_rate_ann"]), maturity_yrs))
    return np.asarray(fx_model.foreign_discount_factor(maturity_yrs), dtype=np.float64)


def _closed_form_forward_price(
    trade: FXForwardTrade,
    fx_model: FXModel,
) -> typing.Callable[[dict[str, Any]], Dual | Dual2 | np.ndarray]:
    """Price ``N × (F(0, T) - K) × P_d(0, T)`` with differentiated inputs."""
    maturity = trade.maturity_yrs
    strike = trade.strike_fx
    notional = trade.notional

    def price_fn(params: dict[str, Any]) -> Dual | Dual2 | np.ndarray:
        spot = params["spot_fx"]
        df_d = _df_domestic(params, fx_model, maturity)
        df_f = _df_foreign(params, fx_model, maturity)
        fwd = _div(_mul(spot, df_f), df_d)
        return _mul(_mul(notional, _add(fwd, _neg(strike))), df_d)

    return price_fn


def _closed_form_option_price(
    trade: FXEuropeanOptionTrade,
    fx_model: FXModel,
    is_call: bool,
) -> typing.Callable[[dict[str, Any]], Dual | Dual2 | np.ndarray]:
    """Price a Black-76 FX option with differentiated inputs.

    ``C = N·P_d·(F·N(d1) - K·N(d2))`` and ``P = N·P_d·(K·N(-d2) - F·N(-d1))``.
    """
    maturity = trade.maturity_yrs
    strike = trade.strike_fx
    notional = trade.notional
    sqt = math.sqrt(maturity)

    def price_fn(params: dict[str, Any]) -> Dual | Dual2 | np.ndarray:
        spot = params["spot_fx"]
        vol = params["fx_vol_ann"]
        df_d = _df_domestic(params, fx_model, maturity)
        df_f = _df_foreign(params, fx_model, maturity)
        fwd = _div(_mul(spot, df_f), df_d)
        vterm = _mul(vol, sqt)
        d1 = _div(
            _add(
                dual_log(_div(fwd, strike)),
                _mul(_mul(_mul(0.5, vol), vol), maturity),
            ),
            vterm,
        )
        d2 = _add(d1, _neg(vterm))
        if is_call:
            pv = _sub(_mul(fwd, dual_norm_cdf(d1)), _mul(strike, dual_norm_cdf(d2)))
        else:
            pv = _sub(
                _mul(strike, dual_norm_cdf(_neg(d2))),
                _mul(fwd, dual_norm_cdf(_neg(d1))),
            )
        return _mul(notional, _mul(df_d, pv))

    return price_fn


def _monte_carlo_price(
    trade: FXForwardTrade | FXEuropeanOptionTrade,
    draws: np.ndarray,
    is_call: bool | None,
    order: int,
) -> typing.Callable[[dict[str, Any]], Dual | Dual2 | np.ndarray]:
    """Pathwise price on frozen Gaussian draws (terminal GBM discretisation).

    ``is_call`` is None for forwards (linear payoff), True/False for options
    (payoff differentiated with a relu subgradient).
    """
    maturity = trade.maturity_yrs
    strike = trade.strike_fx
    notional = trade.notional
    sqt = math.sqrt(maturity)

    def price_fn(params: dict[str, Any]) -> Dual | Dual2 | np.ndarray:
        spot = params["spot_fx"]
        vol = params["fx_vol_ann"]
        rd = params["domestic_rate_ann"]
        rf = params["foreign_rate_ann"]
        drift = _add(_sub(rd, rf), _neg(_mul(_mul(0.5, vol), vol)))
        s_term = _add(_mul(drift, maturity), _mul(_mul(vol, sqt), draws))
        s_t = _mul(spot, dual_exp(s_term))
        if is_call is None:
            payoff = _mul(notional, _sub(s_t, strike))
        elif is_call:
            payoff = _mul(notional, dual_relu(_sub(s_t, strike)))
        else:
            payoff = _mul(notional, dual_relu(_sub(strike, s_t)))
        pv = _mul(dual_exp(_neg(_mul(rd, maturity))), payoff)
        return dual_mean(pv, axis=0)

    return price_fn


def _monte_carlo_std_error(
    trade: FXForwardTrade | FXEuropeanOptionTrade,
    fx_model: GarmanKohlhagenFXModel,
    draws: np.ndarray,
    is_call: bool | None,
) -> float:
    """Standard error of the pathwise Monte Carlo price estimate."""
    maturity = trade.maturity_yrs
    strike = trade.strike_fx
    notional = trade.notional
    spot = float(fx_model.spot_fx)
    vol = float(fx_model.fx_vol_ann)
    rd = float(fx_model.domestic_rate_ann)
    rf = float(fx_model.foreign_rate_ann)
    sqt = math.sqrt(maturity)
    drift = rd - rf - 0.5 * vol * vol
    s_t = spot * np.exp(drift * maturity + vol * sqt * draws)
    if is_call is None:
        payoff = notional * (s_t - strike)
    elif is_call:
        payoff = notional * np.maximum(s_t - strike, 0.0)
    else:
        payoff = notional * np.maximum(strike - s_t, 0.0)
    pv = np.exp(-rd * maturity) * payoff
    return float(np.std(pv) / math.sqrt(float(len(draws))))


# ---------------------------------------------------------------------------
# Greeks driver
# ---------------------------------------------------------------------------


def _model_uses_curve_discounting(fx_model: FXModel) -> bool:
    """Whether a model discounts with configured curves instead of flat rates."""
    params_obj = getattr(fx_model, "_params", None)
    if params_obj is None:
        return False
    for name in ("discount_curve_domestic_yrs", "discount_curve_foreign_yrs"):
        if getattr(params_obj, name, None) is not None:
            return True
    return False


def _model_seed_params(
    fx_model: FXModel,
    *,
    include_rate_seeds: bool,
) -> dict[str, float]:
    """Collect the scalar sensitivity-seed parameters a model exposes.

    Rate seeds are only differentiated when the model discounts with flat
    constant rates (no configured discount curves); curve-based models expose
    no scalar ``rho``.
    """
    out: dict[str, float] = {}
    curve_mode = _model_uses_curve_discounting(fx_model)
    for name in ("spot_fx", "fx_vol_ann", "domestic_rate_ann", "foreign_rate_ann"):
        if (
            name in ("domestic_rate_ann", "foreign_rate_ann")
            and curve_mode
            and not include_rate_seeds
        ):
            continue
        if hasattr(fx_model, name):
            value = float(getattr(fx_model, name))
            if math.isfinite(value):
                out[name] = value
    return out


def _run_forward_ad(
    price_fn: typing.Callable[[dict[str, Any]], Dual | Dual2 | np.ndarray],
    params: dict[str, float],
    seeds: tuple[str, ...],
    order: int,
) -> tuple[float, dict[str, float], dict[str, float] | None]:
    """Evaluate a differentiated price function and extract sensitivities."""
    ad_params: dict[str, Any] = dict(params)
    for i, name in enumerate(seeds):
        base = np.asarray(params[name], dtype=np.float64)
        unit = np.zeros(len(seeds), dtype=np.float64)
        unit[i] = 1.0
        if order == 2:
            ad_params[name] = Dual2(base, unit, np.zeros_like(unit))
        else:
            ad_params[name] = Dual(base, unit)
    out = price_fn(ad_params)
    if not isinstance(out, (Dual, Dual2)):
        msg = "price function did not return a dual number"
        raise TypeError(msg)
    price = float(np.asarray(out.v))
    grads = {name: float(out.g[i]) for i, name in enumerate(seeds)}
    hess: dict[str, float] | None = None
    if order == 2 and isinstance(out, Dual2):
        hess = {name: float(out.h[i]) for i, name in enumerate(seeds)}
    return price, grads, hess


def compute_greeks(
    trade: FXForwardTrade | FXEuropeanOptionTrade,
    fx_model: FXModel,
    *,
    method: str = "autodiff",
    n_paths: int = 10_000,
    random_type: RandomSequenceType | str = RandomSequenceType.SOBOL,
    seed: int | None = 42,
    scramble: bool = True,
    rng: np.random.Generator | None = None,
) -> GreeksResult:
    r"""Compute FX derivative Greeks by forward-mode automatic differentiation.

    Supports two :meth:`method` strategies:

    - ``"autodiff"`` (alias ``"closed_form"``): differentiates the analytical
      closed-form price (the FX forward formula, or Black-76 for options)
      through the separable dual numbers, giving exact delta / gamma / vega /
      rho. Options require a model exposing ``fx_vol_ann``
      (Garman-Kohlhagen or Two-Currency); stochastic volatility models raise
      :class:`NotImplementedError`.
    - ``"monte_carlo"``: differentiates a pathwise Monte Carlo payoff evaluated
      on frozen quasi-random draws. Only the terminal GBM
      :class:`~xvasim.models.fx.GarmanKohlhagenFXModel` discretisation is
      supported; option gamma is left ``None`` because the pathwise relu
      payoff is piecewise-affine and its second derivative is zero almost
      everywhere (degenerate estimator).

    Args:
        trade: An FX forward or European option trade.
        fx_model: FX market model providing ``spot_fx``, optional scalar
            ``fx_vol_ann`` / ``domestic_rate_ann`` / ``foreign_rate_ann``.
        method: ``"autodiff"`` / ``"closed_form"`` or ``"monte_carlo"``.
        n_paths: Number of simulation paths for ``"monte_carlo"``.
        random_type: QMC / PRNG sequence type for path generation.
        seed: Random seed for reproducible draws.
        scramble: Owen scrambling for Sobol/Halton sequences.
        rng: Optional pre-configured NumPy generator.

    Returns:
        A :class:`GreeksResult` mapping with ``price``, ``delta``, ``gamma``,
        ``vega``, ``rho_domestic``, ``rho_foreign``, ``method``,
        ``model_name``, ``parameters`` and ``available_parameters``.

    Raises:
        ValueError: On an unknown method, unsupported trade type or a model
            without a ``spot_fx`` attribute.
        NotImplementedError: For option pricing under a model without
            ``fx_vol_ann``, or pathwise MC under a non-GBM model.
    """
    method_l = method.strip().lower()
    if method_l not in {"autodiff", "closed_form", "monte_carlo"}:
        msg = f"unknown Greeks method {method!r}; expected 'autodiff' or 'monte_carlo'"
        raise ValueError(msg)

    if isinstance(trade, FXEuropeanOptionTrade):
        is_call = _option_is_call(trade.option_type)
    elif isinstance(trade, FXForwardTrade):
        is_call = None
    else:
        msg = (
            "trade must be an FXForwardTrade or FXEuropeanOptionTrade, "
            f"got {type(trade).__name__}"
        )
        raise TypeError(msg)

    is_mc = method_l == "monte_carlo"

    params = _model_seed_params(fx_model, include_rate_seeds=is_mc)
    if "spot_fx" not in params:
        msg = f"{fx_model.model_name} does not expose a spot_fx sensitivity seed"
        raise ValueError(msg)
    seeds = tuple(params)

    if is_mc and not isinstance(fx_model, GarmanKohlhagenFXModel):
        msg = (
            "pathwise (monte_carlo) Greeks require the terminal-GBM "
            f"GarmanKohlhagenFXModel, got {fx_model.model_name}"
        )
        raise NotImplementedError(msg)
    if not is_mc and is_call is not None and "fx_vol_ann" not in params:
        msg = (
            "closed-form option Greeks require a model exposing fx_vol_ann "
            f"(e.g. GarmanKohlhagenFXModel); {fx_model.model_name} does not"
        )
        raise NotImplementedError(msg)

    draws: np.ndarray | None = None
    std_error: float | None = None
    if is_mc:
        draws = generate_normal_draws(
            n_paths=n_paths,
            dimension=1,
            random_type=random_type,
            seed=seed,
            scramble=scramble,
            rng=rng,
            use_cache=True,
        )[:, 0]

    order = 1 if (is_mc and is_call is not None) else 2
    price_fn: typing.Callable[[dict[str, Any]], Dual | Dual2 | np.ndarray]
    if is_mc:
        # is_mc guarantees a Garman-Kohlhagen model (validated above).
        assert draws is not None
        gk_model = typing.cast(GarmanKohlhagenFXModel, fx_model)
        std_error = _monte_carlo_std_error(trade, gk_model, draws, is_call)
        price_fn = _monte_carlo_price(trade, draws, is_call, order)
    elif is_call is None:
        price_fn = _closed_form_forward_price(
            typing.cast(FXForwardTrade, trade),
            fx_model,
        )
    else:
        price_fn = _closed_form_option_price(
            typing.cast(FXEuropeanOptionTrade, trade),
            fx_model,
            is_call,
        )

    price, grads, hess = _run_forward_ad(price_fn, params, seeds, order)

    gamma: float | None
    if is_mc and is_call is not None:
        gamma = None
    elif hess is not None:
        gamma = hess.get("spot_fx")
    else:
        gamma = None

    res: dict[str, Any] = {
        "price": price,
        "delta": grads.get("spot_fx"),
        "gamma": gamma,
        "vega": grads.get("fx_vol_ann"),
        "rho_domestic": grads.get("domestic_rate_ann"),
        "rho_foreign": grads.get("foreign_rate_ann"),
        "method": "monte_carlo" if is_mc else "autodiff",
        "model_name": fx_model.model_name,
        "parameters": dict(params),
        "available_parameters": list(seeds),
        "option_type": (
            getattr(trade, "option_type", None) if is_call is not None else None
        ),
    }
    if std_error is not None:
        res["std_error"] = std_error
    return GreeksResult(res)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class GreeksResult(dict[str, typing.Any]):
    """Greeks container supporting dict and attribute access.

    Supports:
    - Standard dict indexing: ``res["delta"]``, ``res["price"]``
    - Direct attribute access: ``res.delta``, ``res.vega``
    - ``None`` values for sensitivities a model cannot express (e.g. rho on a
      Two-Currency model, or gamma for pathwise option Monte Carlo).
    """

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)

    def __getattr__(self, name: str) -> typing.Any:
        if name in self:
            return self[name]
        msg = f"'{self.__class__.__name__}' object has no attribute {name!r}"
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: typing.Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        if name in self:
            del self[name]
        else:
            msg = f"'{self.__class__.__name__}' object has no attribute {name!r}"
            raise AttributeError(msg)

    @property
    def price(self) -> float:
        """The differentiated trade price."""
        return float(self["price"])

    @property
    def delta(self) -> float | None:
        """Sensitivity of the price to ``spot_fx``."""
        val = self.get("delta")
        return float(val) if val is not None else None

    @property
    def gamma(self) -> float | None:
        """Second-order sensitivity of the price to ``spot_fx``."""
        val = self.get("gamma")
        return float(val) if val is not None else None

    @property
    def vega(self) -> float | None:
        """Sensitivity of the price to ``fx_vol_ann``."""
        val = self.get("vega")
        return float(val) if val is not None else None

    @property
    def rho_domestic(self) -> float | None:
        """Sensitivity of the price to ``domestic_rate_ann``."""
        val = self.get("rho_domestic")
        return float(val) if val is not None else None

    @property
    def rho_foreign(self) -> float | None:
        """Sensitivity of the price to ``foreign_rate_ann``."""
        val = self.get("rho_foreign")
        return float(val) if val is not None else None
