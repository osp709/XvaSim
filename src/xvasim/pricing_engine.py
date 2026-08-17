"""LGM-based and modular Monte Carlo pricing engine for derivatives.

This module implements analytical and Monte Carlo derivative pricing for:
- Foreign exchange forwards and European currency options.
- Zero-coupon and year-on-year inflation swaps, inflation options / caps / floors.
- Single-currency fixed-for-floating Interest Rate Swaps (IRS).
- Multi-currency / Cross-Currency Swaps (XCCY) supporting fixed-for-floating,
  fixed-for-fixed, and floating-for-floating (basis) structures.

Public API
----------
- :class:`LGMParams` — single-currency calibrated LGM parameters.
- :class:`FXLGMParams` — two-currency + FX spot model parameters.
- :class:`OptionType` — enumeration of supported option types.
- :class:`SwapLegType` — enumeration of supported swap leg types.
- :func:`calibrate_lgm_to_swaptions` — calibrate ``σ(t)`` to swaptions.
- :func:`price_foreign_exchange_forward` (alias ``price_fx_forward``) —
  MC price a currency forward.
- :func:`price_foreign_exchange_option` (alias ``price_fx_option``) —
  MC price a European currency option.
- :func:`price_interest_rate_swap` (alias ``price_irs``) —
  price single-currency IRS.
- :func:`price_cross_currency_swap` (alias ``price_xccy_swap``) —
  price cross-currency swap.
- :func:`price_zero_coupon_inflation_swap` —
  price zero-coupon inflation swaps.
- :func:`price_year_on_year_inflation_swap` (alias
  ``price_yoy_inflation_swap``) — MC price year-on-year inflation swaps.
- :func:`price_consumer_price_index_option` (alias ``price_cpi_option``) —
  price European CPI index options / inflation caps & floors.

Units & Conventions
-------------------
- Time / tenor in **years** (suffix ``_yrs``).
- Rates / vols as **annualised decimals** (suffix ``_ann``).
"""

from __future__ import annotations

import dataclasses
import enum
import typing

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .models.base import FXModel, InflationModel, InterestRateModel
from .models.fx.two_currency import TwoCurrencyFXModel
from .models.inflation.black_inflation import BlackInflationModel
from .models.inflation.jarrow_yildirim import JarrowYildirimModel
from .models.ir.lgm import LGMModel, LGMParams
from .qmc import RandomSequenceType, generate_normal_draws

__all__ = [
    "FXLGMParams",
    "LGMParams",
    "OptionType",
    "SwapLegType",
    "benchmark_price_consumer_price_index_option",
    "benchmark_price_cpi_option",
    "benchmark_price_cross_currency_swap",
    "benchmark_price_foreign_exchange_forward",
    "benchmark_price_foreign_exchange_option",
    "benchmark_price_fx_forward",
    "benchmark_price_fx_option",
    "benchmark_price_interest_rate_swap",
    "benchmark_price_irs",
    "benchmark_price_xccy_swap",
    "benchmark_price_zero_coupon_inflation_swap",
    "calibrate_lgm_to_swaptions",
    "price_consumer_price_index_option",
    "price_cpi_option",
    "price_cross_currency_swap",
    "price_foreign_exchange_forward",
    "price_foreign_exchange_option",
    "price_fx_forward",
    "price_fx_option",
    "price_interest_rate_swap",
    "price_irs",
    "price_xccy_swap",
    "price_year_on_year_inflation_swap",
    "price_yoy_inflation_swap",
    "price_zero_coupon_inflation_swap",
]


@dataclasses.dataclass(frozen=True)
class FXLGMParams:
    """Two-currency LGM model parameters for FX derivative pricing.

    Attributes:
        domestic: LGM parameters for the domestic (numeraire) currency.
        foreign: LGM parameters for the foreign currency.
        spot_fx: Current spot FX rate (units of domestic per 1 foreign).
        fx_vol_ann: Annualised log-normal volatility of the FX spot rate.
        correlation_matrix: 3×3 correlation matrix ordered as
            ``[domestic_rate, foreign_rate, fx_spot]``.
    """

    domestic: LGMParams
    foreign: LGMParams
    spot_fx: float
    fx_vol_ann: float
    correlation_matrix: np.ndarray


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class OptionType(enum.Enum):
    """Supported FX option types for :func:`price_fx_option`.

    Members:
        CALL: European call — payoff ``N × max(S(T) − K, 0)``.
        PUT: European put — payoff ``N × max(K − S(T), 0)``.
    """

    CALL = "call"
    PUT = "put"


class SwapLegType(enum.Enum):
    """Supported leg types for interest rate and cross-currency swaps.

    Members:
        FIXED: Fixed-rate coupon leg.
        FLOATING: Floating-rate coupon leg (e.g. forward Ibor/Libor/SOFR).
    """

    FIXED = "fixed"
    FLOATING = "floating"


# ---------------------------------------------------------------------------
# LGM helper functions
# ---------------------------------------------------------------------------


def _compute_h_function(t: np.ndarray | float, kappa: float) -> np.ndarray:
    r"""Compute the LGM *H*-function.

    .. math::
        H(t) = \frac{1 - e^{-\kappa\,t}}{\kappa}

    For ``κ ≈ 0`` the limit ``H(t) → t`` is used to avoid division by zero.

    Args:
        t: Time(s) in years (scalar or array).
        kappa: Mean-reversion speed (annualised).

    Returns:
        Array of ``H`` values with the same shape as *t*.
    """
    t = np.asarray(t, dtype=np.float64)
    if abs(kappa) < 1e-12:
        return t.copy()
    return (1.0 - np.exp(-kappa * t)) / kappa


def _compute_zeta(
    t: float,
    sigma_grid_yrs: np.ndarray,
    sigma_values_ann: np.ndarray,
    kappa: float,
) -> float:
    r"""Compute the LGM accumulated variance ζ(t).

    .. math::
        \zeta(t) = \int_0^t \sigma(s)^2\, e^{-2\kappa(t-s)}\, ds

    evaluated analytically on each piecewise-constant segment.

    Args:
        t: Evaluation time (years, scalar).
        sigma_grid_yrs: Breakpoints of the piecewise-constant σ function.
        sigma_values_ann: Volatility values on each segment.
        kappa: Mean-reversion speed.

    Returns:
        ζ(t) as a float.
    """
    if t <= 0.0:
        return 0.0

    zeta = 0.0
    s_start = 0.0
    for i in range(len(sigma_grid_yrs)):
        s_end = min(float(sigma_grid_yrs[i]), t)
        if s_start >= t:
            break
        sig = float(sigma_values_ann[i])
        ds = s_end - s_start
        if ds <= 0.0:
            s_start = s_end
            continue

        if abs(kappa) < 1e-12:
            zeta += sig * sig * ds
        else:
            zeta += (sig * sig / (2.0 * kappa)) * (
                np.exp(-2.0 * kappa * (t - s_end))
                - np.exp(-2.0 * kappa * (t - s_start))
            )
        s_start = s_end

    return float(zeta)


def _interpolate_discount_factor(
    t: np.ndarray | float,
    curve_yrs: np.ndarray,
    curve_dfs: np.ndarray,
) -> np.ndarray:
    """Log-linearly interpolate (and flat-extrapolate) a discount curve.

    Args:
        t: Query time(s) in years.
        curve_yrs: Tenor pillar points of the curve (years).
        curve_dfs: Corresponding discount factors.

    Returns:
        Interpolated discount factor(s) with the same shape as *t*.
    """
    t = np.asarray(t, dtype=np.float64)
    log_dfs = np.log(np.maximum(curve_dfs, 1e-18))
    interp_log = np.interp(t, curve_yrs, log_dfs)
    return np.exp(interp_log)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# LGM swaption calibration
# ---------------------------------------------------------------------------


def _lgm_swaption_price_normal(
    expiry_yrs: float,
    swap_tenor_yrs: float,
    market_normal_vol_ann: float,
    kappa: float,
    sigma_grid_yrs: np.ndarray,
    sigma_values_ann: np.ndarray,
    curve_yrs: np.ndarray,
    curve_dfs: np.ndarray,
    fixed_rate_ann: float,
    pay_freq_yrs: float = 0.5,
) -> tuple[float, float]:
    """Compute the LGM model swaption price and the market swaption price.

    Both are expressed as *normal (Bachelier)* prices. The LGM normal vol
    for a co-terminal swaption is approximated by the *annuity-weighted*
    H-function dispersion:

    .. math::
        V_{\\text{lgm}}^2 = \\frac{\\zeta(T_0)}{T_0}
            \\left(\\frac{\\sum_i \\tau_i P(0,T_i) H(T_i)}{A_0}\\right)^2

    where *A₀* is the forward annuity and *τᵢ* are the year-fractions of the
    underlying swap payments.

    Args:
        expiry_yrs: Swaption expiry in years.
        swap_tenor_yrs: Underlying swap tenor in years.
        market_normal_vol_ann: Market-observed normal (Bachelier) volatility
            (annualised, e.g. 0.0050 for 50 bp/yr).
        kappa: LGM mean reversion.
        sigma_grid_yrs: Current volatility breakpoints.
        sigma_values_ann: Current piecewise-constant volatilities.
        curve_yrs: Discount curve tenors.
        curve_dfs: Discount curve discount factors.
        fixed_rate_ann: Fixed rate of the underlying swap (annualised).
        pay_freq_yrs: Payment frequency of the fixed leg in years
            (default 0.5 = semi-annual).

    Returns:
        ``(model_price, market_price)`` — both expressed as receiver
        swaption price (per unit notional) under Bachelier's formula.
    """
    n_periods = max(1, round(swap_tenor_yrs / pay_freq_yrs))
    actual_freq = swap_tenor_yrs / n_periods
    payment_times = np.array(
        [expiry_yrs + actual_freq * (k + 1) for k in range(n_periods)]
    )

    dfs = _interpolate_discount_factor(payment_times, curve_yrs, curve_dfs)
    annuity = float(np.sum(actual_freq * dfs))
    h_vals = _compute_h_function(payment_times, kappa)

    zeta = _compute_zeta(expiry_yrs, sigma_grid_yrs, sigma_values_ann, kappa)
    h_expiry = float(_compute_h_function(expiry_yrs, kappa))
    delta_h = h_vals - h_expiry

    weighted_delta_h = float(np.sum(actual_freq * dfs * delta_h)) / annuity

    model_normal_var = zeta * weighted_delta_h**2
    model_normal_vol = np.sqrt(max(model_normal_var / expiry_yrs, 0.0))

    sqrt_t = np.sqrt(expiry_yrs)
    phi_0 = 1.0 / np.sqrt(2.0 * np.pi)

    model_price = annuity * model_normal_vol * sqrt_t * phi_0
    market_price = annuity * market_normal_vol_ann * sqrt_t * phi_0

    return float(model_price), float(market_price)


def calibrate_lgm_to_swaptions(
    swaption_expiries_yrs: np.ndarray,
    swap_tenors_yrs: np.ndarray,
    market_normal_vols_ann: np.ndarray,
    curve_yrs: np.ndarray,
    curve_dfs: np.ndarray,
    fixed_rates_ann: np.ndarray,
    kappa_ann: float = 0.03,
    pay_freq_yrs: float = 0.5,
) -> LGMParams:
    """Calibrate an LGM volatility function to a set of swaptions.

    The calibration proceeds *bootstrapping style*: for each swaption
    (ordered by expiry), a root-finding step determines the piecewise-
    constant volatility on the interval ending at that expiry, so that
    the model price matches the market price.

    Args:
        swaption_expiries_yrs: 1-D array of swaption expiry times (years),
            sorted ascending.
        swap_tenors_yrs: 1-D array of underlying swap tenors (years),
            one per swaption.
        market_normal_vols_ann: 1-D array of market normal (Bachelier)
            volatilities (annualised).
        curve_yrs: Discount-curve tenors (years).
        curve_dfs: Discount factors on the curve.
        fixed_rates_ann: 1-D array of fixed rates (annualised) of each
            underlying swap (usually set to the ATM forward swap rate).
        kappa_ann: Mean-reversion speed (annualised). Default 0.03.
        pay_freq_yrs: Fixed-leg payment frequency in years. Default 0.5.

    Returns:
        A fully populated :class:`LGMParams` instance.

    Raises:
        RuntimeError: If any single-expiry root-finding fails.
    """
    n = len(swaption_expiries_yrs)
    sigma_grid = np.array(swaption_expiries_yrs, dtype=np.float64)
    sigma_vals = np.zeros(n, dtype=np.float64)

    for i in range(n):
        partial_grid = sigma_grid[: i + 1]

        def _price_diff(
            sigma_i: float,
            idx: int = i,
            grid: np.ndarray = partial_grid,
        ) -> float:
            partial_vals = sigma_vals[: idx + 1].copy()
            partial_vals[idx] = sigma_i
            model_p, market_p = _lgm_swaption_price_normal(
                expiry_yrs=float(swaption_expiries_yrs[idx]),
                swap_tenor_yrs=float(swap_tenors_yrs[idx]),
                market_normal_vol_ann=float(market_normal_vols_ann[idx]),
                kappa=kappa_ann,
                sigma_grid_yrs=grid,
                sigma_values_ann=partial_vals,
                curve_yrs=curve_yrs,
                curve_dfs=curve_dfs,
                fixed_rate_ann=float(fixed_rates_ann[idx]),
                pay_freq_yrs=pay_freq_yrs,
            )
            return model_p - market_p

        try:
            sigma_vals[i] = brentq(_price_diff, 1e-6, 2.0, xtol=1e-12)
        except ValueError as exc:
            msg = (
                f"LGM calibration failed at expiry "
                f"{swaption_expiries_yrs[i]:.4f}y: {exc}"
            )
            raise RuntimeError(msg) from exc

    return LGMParams(
        kappa_ann=kappa_ann,
        sigma_grid_yrs=sigma_grid,
        sigma_values_ann=sigma_vals,
        discount_curve_yrs=np.array(curve_yrs, dtype=np.float64),
        discount_factors=np.array(curve_dfs, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------


def _simulate_fx_paths(
    params: FXLGMParams,
    maturity_yrs: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator | None = None,
    random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
    seed: int | None = None,
    scramble: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate FX and short-rate state paths under the domestic measure.

    Three correlated Brownian drivers are evolved on a uniform time grid:
      1. Domestic LGM state ``x_d(t)``
      2. Foreign LGM state ``x_f(t)``
      3. Log-FX process ``ln S(t)``

    Args:
        params: Two-currency model parameters.
        maturity_yrs: Simulation horizon (years).
        n_paths: Number of Monte Carlo paths.
        n_steps: Number of time steps.
        rng: Optional NumPy random Generator for reproducibility.
        random_type: Sequence type (:class:`RandomSequenceType` or str).
        seed: Optional random seed.
        scramble: If True, scrambles QMC sequences.

    Returns:
        ``(times, x_dom, x_for, fx_spot)`` — all arrays of shape
        ``(n_paths, n_steps + 1)`` except *times* which is
        ``(n_steps + 1,)``.
    """
    dt = maturity_yrs / n_steps
    sqrt_dt = np.sqrt(dt)
    times = np.linspace(0.0, maturity_yrs, n_steps + 1)

    chol = np.linalg.cholesky(params.correlation_matrix)

    x_dom = np.zeros((n_paths, n_steps + 1))
    x_for = np.zeros((n_paths, n_steps + 1))
    ln_fx = np.zeros((n_paths, n_steps + 1))
    ln_fx[:, 0] = np.log(params.spot_fx)

    kd = params.domestic.kappa_ann
    kf = params.foreign.kappa_ann
    vol_fx = params.fx_vol_ann

    z_all = generate_normal_draws(
        n_paths=n_paths,
        dimension=3 * n_steps,
        random_type=random_type,
        seed=seed,
        scramble=scramble,
        rng=rng,
    ).reshape(n_paths, n_steps, 3)

    for step in range(n_steps):
        t = times[step]

        sig_d = float(
            params.domestic.sigma_values_ann[
                min(
                    int(
                        np.searchsorted(params.domestic.sigma_grid_yrs, t, side="right")
                    ),
                    len(params.domestic.sigma_values_ann) - 1,
                )
            ]
        )
        sig_f = float(
            params.foreign.sigma_values_ann[
                min(
                    int(
                        np.searchsorted(params.foreign.sigma_grid_yrs, t, side="right")
                    ),
                    len(params.foreign.sigma_values_ann) - 1,
                )
            ]
        )

        z_indep = z_all[:, step, :]
        z_corr = z_indep @ chol.T

        dw_d = z_corr[:, 0] * sqrt_dt
        dw_f = z_corr[:, 1] * sqrt_dt
        dw_fx = z_corr[:, 2] * sqrt_dt

        x_dom[:, step + 1] = x_dom[:, step] - kd * x_dom[:, step] * dt + sig_d * dw_d

        rho_f_fx = params.correlation_matrix[1, 2]
        x_for[:, step + 1] = (
            x_for[:, step]
            - kf * x_for[:, step] * dt
            - sig_f * rho_f_fx * vol_fx * dt
            + sig_f * dw_f
        )

        df_d_t = float(
            _interpolate_discount_factor(
                t, params.domestic.discount_curve_yrs, params.domestic.discount_factors
            )
        )
        df_d_t1 = float(
            _interpolate_discount_factor(
                t + dt,
                params.domestic.discount_curve_yrs,
                params.domestic.discount_factors,
            )
        )
        df_f_t = float(
            _interpolate_discount_factor(
                t, params.foreign.discount_curve_yrs, params.foreign.discount_factors
            )
        )
        df_f_t1 = float(
            _interpolate_discount_factor(
                t + dt,
                params.foreign.discount_curve_yrs,
                params.foreign.discount_factors,
            )
        )

        fwd_d = -np.log(df_d_t1 / df_d_t) / dt
        fwd_f = -np.log(df_f_t1 / df_f_t) / dt

        h_prime_d = np.exp(-kd * t)
        h_prime_f = np.exp(-kf * t)

        zeta_d = _compute_zeta(
            t,
            params.domestic.sigma_grid_yrs,
            params.domestic.sigma_values_ann,
            kd,
        )
        zeta_f = _compute_zeta(
            t,
            params.foreign.sigma_grid_yrs,
            params.foreign.sigma_values_ann,
            kf,
        )

        r_d = fwd_d + h_prime_d * x_dom[:, step] + 0.5 * h_prime_d**2 * zeta_d
        r_f = fwd_f + h_prime_f * x_for[:, step] + 0.5 * h_prime_f**2 * zeta_f

        drift_fx = r_d - r_f - 0.5 * vol_fx**2
        ln_fx[:, step + 1] = ln_fx[:, step] + drift_fx * dt + vol_fx * dw_fx

    fx_spot = np.exp(ln_fx)
    return times, x_dom, x_for, fx_spot


def _discount_path(
    lgm: LGMParams,
    x_state: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Compute path-wise discount factors D(0, tᵢ) under LGM.

    Args:
        lgm: Single-currency LGM parameters.
        x_state: State variable paths, shape ``(n_paths, n_steps + 1)``.
        times: Time grid, shape ``(n_steps + 1,)``.

    Returns:
        Array of discount factors ``D(0, tᵢ)`` with same shape as *x_state*.
    """
    n_paths, n_times = x_state.shape
    kappa = lgm.kappa_ann

    short_rates = np.zeros_like(x_state)
    for j in range(n_times):
        t = times[j]
        h_prime = np.exp(-kappa * t)
        fwd = _instantaneous_forward(t, lgm.discount_curve_yrs, lgm.discount_factors)
        zeta = _compute_zeta(t, lgm.sigma_grid_yrs, lgm.sigma_values_ann, kappa)
        short_rates[:, j] = fwd + h_prime * x_state[:, j] + 0.5 * h_prime**2 * zeta

    dt_vec = np.diff(times)
    cum_integral = np.zeros((n_paths, n_times))
    for j in range(1, n_times):
        cum_integral[:, j] = cum_integral[:, j - 1] + 0.5 * dt_vec[j - 1] * (
            short_rates[:, j - 1] + short_rates[:, j]
        )

    return np.exp(-cum_integral)


def _instantaneous_forward(
    t: float,
    curve_yrs: np.ndarray,
    curve_dfs: np.ndarray,
) -> float:
    """Approximate the instantaneous forward rate f(0, t) from the curve.

    Uses a finite-difference bump of 1 day (≈ 1/365.25 yr).
    """
    bump = 1.0 / 365.25
    t_lo = max(t - bump / 2.0, 0.0)
    t_hi = t + bump / 2.0
    df_lo = float(_interpolate_discount_factor(t_lo, curve_yrs, curve_dfs))
    df_hi = float(_interpolate_discount_factor(t_hi, curve_yrs, curve_dfs))
    return float(-np.log(df_hi / max(df_lo, 1e-18)) / (t_hi - t_lo))


# ---------------------------------------------------------------------------
# Public pricing functions: Foreign Exchange (FX)
# ---------------------------------------------------------------------------


def benchmark_price_foreign_exchange_forward(
    params: FXLGMParams | FXModel,
    strike: float,
    maturity_yrs: float,
    notional: float,
) -> dict[str, float]:
    """Analytical benchmark pricing for currency forwards via Covered Interest Parity.

    Args:
        params: Two-currency model parameters, either :class:`FXLGMParams` or
            a modular :class:`~xvasim.models.base.FXModel`.
        strike: Forward strike (domestic per foreign).
        maturity_yrs: Maturity in years.
        notional: Notional amount in foreign currency.

    Returns:
        Dictionary with ``"price"``, ``"forward_fx"``, ``"domestic_df"``,
        and ``"foreign_df"``.
    """
    if isinstance(params, FXLGMParams):
        df_d = float(
            _interpolate_discount_factor(
                maturity_yrs,
                params.domestic.discount_curve_yrs,
                params.domestic.discount_factors,
            )
        )
        df_f = float(
            _interpolate_discount_factor(
                maturity_yrs,
                params.foreign.discount_curve_yrs,
                params.foreign.discount_factors,
            )
        )
        spot = float(params.spot_fx)
    elif isinstance(params, TwoCurrencyFXModel):
        df_d = float(
            params.domestic_ir_model.interpolate_discount_factor(maturity_yrs)
        )
        df_f = float(
            params.foreign_ir_model.interpolate_discount_factor(maturity_yrs)
        )
        spot = float(params.spot_fx)
    elif isinstance(params, FXModel):
        if hasattr(params, "domestic_discount_factor") and hasattr(
            params, "foreign_discount_factor"
        ):
            df_d = float(params.domestic_discount_factor(maturity_yrs))
            df_f = float(params.foreign_discount_factor(maturity_yrs))
        else:
            df_d = 1.0
            df_f = 1.0
        spot = float(params.spot_fx) if hasattr(params, "spot_fx") else 1.0
    else:
        msg = (
            f"params must be FXLGMParams or FXModel, "
            f"got {type(params).__name__}"
        )
        raise TypeError(msg)

    fwd_fx = spot * (df_f / max(df_d, 1e-18))
    price = notional * (fwd_fx - strike) * df_d
    return {
        "price": float(price),
        "forward_fx": float(fwd_fx),
        "domestic_df": float(df_d),
        "foreign_df": float(df_f),
    }


# Convenience alias for foreign exchange forward benchmark pricing
benchmark_price_fx_forward = benchmark_price_foreign_exchange_forward


def price_foreign_exchange_forward(
    params: FXLGMParams | FXModel,
    strike: float,
    maturity_yrs: float,
    notional: float,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int | None = 42,
    random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
    scramble: bool = True,
) -> dict[str, float | np.ndarray]:
    """Price a currency forward via Monte Carlo under LGM or modular FX models.

    The forward buyer receives ``N × (S(T) − K)`` at maturity *T*, discounted
    to today using the domestic bank account.

    Args:
        params: Two-currency model parameters, either :class:`FXLGMParams` or
            a modular :class:`~xvasim.models.base.FXModel`.
        strike: Forward strike (domestic per foreign).
        maturity_yrs: Maturity in years.
        notional: Notional amount in foreign currency.
        n_paths: Number of Monte Carlo paths.
        n_steps: Number of simulation time steps.
        seed: Random seed (``None`` for non-deterministic).
        random_type: Sequence type (:class:`RandomSequenceType` or str).
        scramble: If True, applies scrambling to QMC sequences.

    Returns:
        Dictionary with keys:

        - ``"price"`` — MC forward value (domestic currency).
        - ``"std_error"`` — standard error of the MC estimate.
        - ``"analytical_benchmark_price"`` — exact analytical benchmark price.
        - ``"fx_terminal"`` — 1-D array of terminal FX rates.
    """
    if isinstance(params, FXLGMParams):
        times, x_dom, _x_for, fx_paths = _simulate_fx_paths(
            params,
            maturity_yrs,
            n_paths,
            n_steps,
            random_type=random_type,
            seed=seed,
            scramble=scramble,
        )
        s_t = fx_paths[:, -1]
        dom_df = _discount_path(params.domestic, x_dom, times)
        df_t = dom_df[:, -1]
    elif isinstance(params, TwoCurrencyFXModel):
        times, x_dom, _x_for, fx_paths = params.simulate_paths(
            maturity_yrs,
            n_paths,
            n_steps,
            random_type=random_type,
            seed=seed,
            scramble=scramble,
        )
        s_t = fx_paths[:, -1]
        dom_df = params.domestic_ir_model.discount_path(times, x_dom)
        df_t = dom_df[:, -1]
    elif isinstance(params, FXModel):
        sim_res = params.simulate_paths(
            maturity_yrs,
            n_paths,
            n_steps,
            random_type=random_type,
            seed=seed,
            scramble=scramble,
        )
        fx_paths = sim_res[-1]
        s_t = fx_paths[:, -1]
        if hasattr(params, "domestic_discount_factor"):
            df_val = float(params.domestic_discount_factor(maturity_yrs))
            df_t = np.full(n_paths, df_val, dtype=np.float64)
        else:
            df_t = np.ones(n_paths, dtype=np.float64)
    else:
        msg = (
            f"params must be FXLGMParams or FXModel, "
            f"got {type(params).__name__}"
        )
        raise TypeError(msg)

    payoff = notional * (s_t - strike) * df_t
    price = float(np.mean(payoff))
    std_error = float(np.std(payoff) / np.sqrt(n_paths))

    ana_res = benchmark_price_foreign_exchange_forward(
        params=params,
        strike=strike,
        maturity_yrs=maturity_yrs,
        notional=notional,
    )

    return {
        "price": price,
        "std_error": std_error,
        "analytical_benchmark_price": ana_res["price"],
        "fx_terminal": s_t,
    }


# Convenience alias for foreign exchange forward pricing
price_fx_forward = price_foreign_exchange_forward


def benchmark_price_foreign_exchange_option(
    params: FXLGMParams | FXModel,
    strike: float,
    maturity_yrs: float,
    notional: float,
    option_type: OptionType | str = OptionType.CALL,
) -> dict[str, float]:
    r"""Analytical benchmark pricing for European currency options.

    Args:
        params: Two-currency model parameters, either :class:`FXLGMParams` or
            a modular :class:`~xvasim.models.base.FXModel`.
        strike: Option strike (domestic per foreign).
        maturity_yrs: Expiry in years.
        notional: Notional in foreign currency.
        option_type: :class:`OptionType` member or string (``'call'`` / ``'put'``).

    Returns:
        Dictionary containing benchmark ``"price"`` and diagnostics.
    """
    if isinstance(option_type, str):
        try:
            resolved = OptionType(option_type.strip().lower())
        except ValueError:
            msg = (
                f"option_type must be OptionType.CALL, OptionType.PUT, "
                f"'call', or 'put', got {option_type!r}"
            )
            raise ValueError(msg) from None
    elif isinstance(option_type, OptionType):
        resolved = option_type
    else:
        msg = (
            f"option_type must be an OptionType or str, "
            f"got {type(option_type).__name__}"
        )
        raise TypeError(msg)

    is_call = resolved is OptionType.CALL

    if hasattr(params, "price_option_analytical"):
        res = params.price_option_analytical(
            strike=strike,
            maturity_yrs=maturity_yrs,
            notional=notional,
            option_type=resolved,
        )
        if isinstance(res, dict):
            return {k: float(v) for k, v in res.items()}
        return {"price": float(res)}

    if hasattr(params, "closed_form_option_price"):
        opt_str = (
            resolved.value
            if isinstance(resolved, OptionType)
            else str(resolved).lower()
        )
        res_val = params.closed_form_option_price(
            strike=strike,
            maturity_yrs=maturity_yrs,
            option_type=opt_str,
            notional=notional,
        )
        return {"price": float(res_val)}

    if isinstance(params, (FXLGMParams, TwoCurrencyFXModel)):
        if isinstance(params, FXLGMParams):
            df_d = float(
                _interpolate_discount_factor(
                    maturity_yrs,
                    params.domestic.discount_curve_yrs,
                    params.domestic.discount_factors,
                )
            )
            df_f = float(
                _interpolate_discount_factor(
                    maturity_yrs,
                    params.foreign.discount_curve_yrs,
                    params.foreign.discount_factors,
                )
            )
            spot = float(params.spot_fx)
            vol = float(params.fx_vol_ann)
        else:
            df_d = float(
                params.domestic_ir_model.interpolate_discount_factor(
                    maturity_yrs
                )
            )
            df_f = float(
                params.foreign_ir_model.interpolate_discount_factor(
                    maturity_yrs
                )
            )
            spot = float(params.spot_fx)
            vol = float(params.fx_vol_ann)

        fwd_fx = spot * (df_f / max(df_d, 1e-18))
        total_std = vol * np.sqrt(maturity_yrs)
        if total_std < 1e-12:
            intrinsic = (
                max(fwd_fx - strike, 0.0)
                if is_call
                else max(strike - fwd_fx, 0.0)
            )
            return {
                "price": float(notional * df_d * intrinsic),
                "forward_fx": float(fwd_fx),
            }

        d1 = (
            np.log(fwd_fx / strike) + 0.5 * vol * vol * maturity_yrs
        ) / total_std
        d2 = d1 - total_std
        if is_call:
            pv = (
                notional
                * df_d
                * (fwd_fx * float(norm.cdf(d1)) - strike * float(norm.cdf(d2)))
            )
        else:
            pv = (
                notional
                * df_d
                * (
                    strike * float(norm.cdf(-d2))
                    - fwd_fx * float(norm.cdf(-d1))
                )
            )

        return {"price": float(pv), "forward_fx": float(fwd_fx)}

    msg = f"Analytical benchmark not supported for {type(params).__name__}"
    raise TypeError(msg)


# Convenience alias for foreign exchange option benchmark pricing
benchmark_price_fx_option = benchmark_price_foreign_exchange_option


def price_foreign_exchange_option(
    params: FXLGMParams | FXModel,
    strike: float,
    maturity_yrs: float,
    notional: float,
    option_type: OptionType | str = OptionType.CALL,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int | None = 42,
    random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
    scramble: bool = True,
) -> dict[str, float | np.ndarray]:
    r"""Price a European currency option via Monte Carlo under LGM or modular FX models.

    Payoffs:

    - **Call**: :math:`N \times \max(S(T) - K,\; 0)`
    - **Put**: :math:`N \times \max(K - S(T),\; 0)`

    Args:
        params: Two-currency model parameters, either :class:`FXLGMParams` or
            a modular :class:`~xvasim.models.base.FXModel`.
        strike: Option strike (domestic per foreign).
        maturity_yrs: Expiry in years.
        notional: Notional in foreign currency.
        option_type: :class:`OptionType` member or equivalent string
            (``"call"`` / ``"put"``). Strings are accepted for
            backwards compatibility and are resolved case-insensitively.
        n_paths: Number of Monte Carlo paths.
        n_steps: Number of simulation time steps.
        seed: Random seed.
        random_type: Random sequence type (:class:`RandomSequenceType` or str).
        scramble: If True, applies scrambling to QMC sequences.

    Returns:
        Dictionary with keys:

        - ``"price"`` — MC option premium (domestic currency).
        - ``"std_error"`` — standard error of the MC estimate.
        - ``"analytical_benchmark_price"`` — exact analytical benchmark price.
        - ``"fx_terminal"`` — 1-D array of terminal FX rates.

    Raises:
        ValueError: If *option_type* is not a valid
            :class:`OptionType` member or recognised string.
    """
    if isinstance(option_type, str):
        try:
            resolved = OptionType(option_type.strip().lower())
        except ValueError:
            msg = (
                f"option_type must be OptionType.CALL, OptionType.PUT, "
                f"'call', or 'put', got {option_type!r}"
            )
            raise ValueError(msg) from None
    elif isinstance(option_type, OptionType):
        resolved = option_type
    else:
        msg = (
            f"option_type must be an OptionType or str, "
            f"got {type(option_type).__name__}"
        )
        raise TypeError(msg)

    if isinstance(params, FXLGMParams):
        times, x_dom, _x_for, fx_paths = _simulate_fx_paths(
            params,
            maturity_yrs,
            n_paths,
            n_steps,
            random_type=random_type,
            seed=seed,
            scramble=scramble,
        )
        s_t = fx_paths[:, -1]
        dom_df = _discount_path(params.domestic, x_dom, times)
        df_t = dom_df[:, -1]
    elif isinstance(params, TwoCurrencyFXModel):
        times, x_dom, _x_for, fx_paths = params.simulate_paths(
            maturity_yrs,
            n_paths,
            n_steps,
            random_type=random_type,
            seed=seed,
            scramble=scramble,
        )
        s_t = fx_paths[:, -1]
        dom_df = params.domestic_ir_model.discount_path(times, x_dom)
        df_t = dom_df[:, -1]
    elif isinstance(params, FXModel):
        sim_res = params.simulate_paths(
            maturity_yrs,
            n_paths,
            n_steps,
            random_type=random_type,
            seed=seed,
            scramble=scramble,
        )
        fx_paths = sim_res[-1]
        s_t = fx_paths[:, -1]
        if hasattr(params, "domestic_discount_factor"):
            df_val = float(params.domestic_discount_factor(maturity_yrs))
            df_t = np.full(n_paths, df_val, dtype=np.float64)
        else:
            df_t = np.ones(n_paths, dtype=np.float64)
    else:
        msg = (
            f"params must be FXLGMParams or FXModel, "
            f"got {type(params).__name__}"
        )
        raise TypeError(msg)

    if resolved is OptionType.CALL:
        intrinsic = np.maximum(s_t - strike, 0.0)
    else:
        intrinsic = np.maximum(strike - s_t, 0.0)

    payoff = notional * intrinsic * df_t
    price = float(np.mean(payoff))
    std_error = float(np.std(payoff) / np.sqrt(n_paths))

    try:
        ana_res = benchmark_price_foreign_exchange_option(
            params=params,
            strike=strike,
            maturity_yrs=maturity_yrs,
            notional=notional,
            option_type=resolved,
        )
        ana_price: float | None = ana_res["price"]
    except Exception:
        ana_price = None

    res_dict: dict[str, float | np.ndarray] = {
        "price": price,
        "std_error": std_error,
        "fx_terminal": s_t,
    }
    if ana_price is not None:
        res_dict["analytical_benchmark_price"] = ana_price

    return res_dict


# Convenience alias for foreign exchange option pricing
price_fx_option = price_foreign_exchange_option


# ---------------------------------------------------------------------------
# Inflation pricing functions
# ---------------------------------------------------------------------------


def benchmark_price_zero_coupon_inflation_swap(
    model: InflationModel,
    strike_rate_ann: float,
    maturity_yrs: float,
    notional: float = 1.0,
    is_payer: bool = True,
) -> dict[str, float]:
    """Analytical benchmark pricing for a Zero-Coupon Inflation Swap (ZCIS).

    Args:
        model: An instantiated :class:`~xvasim.models.base.InflationModel`.
        strike_rate_ann: Annualised fixed swap rate :math:`K` (e.g. 0.025 for 2.5%).
        maturity_yrs: Swap maturity in years.
        notional: Trade notional amount.
        is_payer: If True, pays fixed and receives inflation; if False, receives fixed.

    Returns:
        Dictionary with ``"price"``, ``"fair_swap_rate"``, ``"forward_cpi"``.
    """
    if not isinstance(model, InflationModel):
        msg = f"model must be an InflationModel, got {type(model).__name__}"
        raise TypeError(msg)

    fair_swap_rate = model.zero_coupon_inflation_swap_rate(maturity_yrs)
    forward_cpi_val = model.forward_cpi(maturity_yrs)
    k_comp = (1.0 + strike_rate_ann) ** maturity_yrs - 1.0

    if hasattr(model, "interpolate_nominal_df"):
        p_nom = float(model.interpolate_nominal_df(maturity_yrs))
    elif hasattr(model, "nominal_ir_model"):
        p_nom = float(
            model.nominal_ir_model.interpolate_discount_factor(maturity_yrs)
        )
    else:
        p_nom = 1.0

    fwd_float_comp = (forward_cpi_val / model.base_cpi) - 1.0
    unit_net = (
        (fwd_float_comp - k_comp) if is_payer else (k_comp - fwd_float_comp)
    )
    price = notional * p_nom * unit_net
    return {
        "price": float(price),
        "fair_swap_rate": float(fair_swap_rate),
        "forward_cpi": float(forward_cpi_val),
    }


def price_zero_coupon_inflation_swap(
    model: InflationModel,
    strike_rate_ann: float,
    maturity_yrs: float,
    notional: float = 1.0,
    is_payer: bool = True,
    n_paths: int | None = 10_000,
    n_steps: int = 50,
    seed: int | None = None,
    random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
    scramble: bool = True,
) -> dict[str, float]:
    """Price a Zero-Coupon Inflation Swap (ZCIS) via Monte Carlo simulation.

    In a zero-coupon inflation swap with maturity :math:`T`:
    - Inflation leg pays :math:`N \\times \\left(\\frac{I(T)}{I(0)} - 1\\right)`.
    - Fixed leg pays :math:`N \\times \\left((1 + K)^T - 1\\right)`.
    - Payer swap pays fixed rate :math:`K` and receives floating inflation.

    Args:
        model: An instantiated :class:`~xvasim.models.base.InflationModel`.
        strike_rate_ann: Annualised fixed swap rate :math:`K` (e.g. 0.025 for 2.5%).
        maturity_yrs: Swap maturity in years.
        notional: Trade notional amount.
        is_payer: If True, pays fixed and receives inflation; if False, receives fixed.
        n_paths: Number of Monte Carlo paths (default: 10,000). If None, computes
            analytical closed form benchmark.
        n_steps: Number of simulation steps (if Monte Carlo).
        seed: Optional random seed.
        random_type: Random sequence generator type (:class:`RandomSequenceType`
            or str).
        scramble: If True, applies scrambling to QMC sequences.

    Returns:
        Dictionary with ``price``, ``fair_swap_rate``, ``forward_cpi``,
        ``analytical_benchmark_price``, and optionally ``std_error`` (for Monte Carlo).
    """
    if not isinstance(model, InflationModel):
        msg = f"model must be an InflationModel, got {type(model).__name__}"
        raise TypeError(msg)

    ana_res = benchmark_price_zero_coupon_inflation_swap(
        model=model,
        strike_rate_ann=strike_rate_ann,
        maturity_yrs=maturity_yrs,
        notional=notional,
        is_payer=is_payer,
    )

    if n_paths is None:
        return ana_res

    # Monte Carlo pricing
    sim_res = model.simulate_paths(
        maturity_yrs=maturity_yrs,
        n_paths=n_paths,
        n_steps=n_steps,
        random_type=random_type,
        seed=seed,
        scramble=scramble,
    )

    cpi_t = sim_res.cpi_index[:, -1]
    df_t = sim_res.nominal_discount_factors[:, -1]

    k_comp = (1.0 + strike_rate_ann) ** maturity_yrs - 1.0
    float_payoff = (cpi_t / model.base_cpi) - 1.0
    net_payoff = (
        (float_payoff - k_comp) if is_payer else (k_comp - float_payoff)
    )
    pv_paths = notional * net_payoff * df_t

    price = float(np.mean(pv_paths))
    std_error = float(np.std(pv_paths) / np.sqrt(n_paths))

    return {
        "price": price,
        "std_error": std_error,
        "analytical_benchmark_price": ana_res["price"],
        "fair_swap_rate": ana_res["fair_swap_rate"],
        "forward_cpi": ana_res["forward_cpi"],
    }


def price_year_on_year_inflation_swap(
    model: InflationModel,
    fixed_rate_ann: float,
    payment_times_yrs: np.ndarray | list[float],
    notional: float = 1.0,
    is_payer: bool = True,
    n_paths: int = 10_000,
    n_steps_per_year: int = 12,
    seed: int | None = None,
    random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
    scramble: bool = True,
) -> dict[str, typing.Any]:
    """Price a Year-on-Year (YoY) Inflation Swap via Monte Carlo simulation.

    For each period :math:`[t_{i-1}, t_i]`:
    - Floating rate: :math:`\\frac{I(t_i) - I(t_{i-1})}{I(t_{i-1})}`.
    - Fixed rate: :math:`K \\times (t_i - t_{i-1})`.

    Args:
        model: An instantiated :class:`~xvasim.models.base.InflationModel`.
        fixed_rate_ann: Annualised fixed swap coupon rate (e.g. 0.025 for 2.5%).
        payment_times_yrs: Tenors / payment times in years.
        notional: Trade notional amount.
        is_payer: If True, pays fixed and receives inflation; if False, receives fixed.
        n_paths: Number of Monte Carlo paths.
        n_steps_per_year: Number of simulation steps per year.
        seed: Optional random seed.
        random_type: Random sequence generator type (:class:`RandomSequenceType`
            or str).
        scramble: If True, applies scrambling to QMC sequences.

    Returns:
        Dictionary with ``price``, ``std_error``, and ``period_cash_flows``.
    """
    if not isinstance(model, InflationModel):
        msg = f"model must be an InflationModel, got {type(model).__name__}"
        raise TypeError(msg)

    pay_times = np.sort(np.asarray(payment_times_yrs, dtype=np.float64))
    if len(pay_times) == 0:
        return {"price": 0.0, "std_error": 0.0, "period_cash_flows": []}

    maturity_yrs = float(pay_times[-1])
    n_steps = max(int(np.ceil(maturity_yrs * n_steps_per_year)), len(pay_times))

    sim_res = model.simulate_paths(
        maturity_yrs=maturity_yrs,
        n_paths=n_paths,
        n_steps=n_steps,
        random_type=random_type,
        seed=seed,
        scramble=scramble,
    )

    times = sim_res.times
    cpi_paths = sim_res.cpi_index
    df_paths = sim_res.nominal_discount_factors

    # Map payment times to nearest simulation grid index
    time_indices = [int(np.argmin(np.abs(times - t_val))) for t_val in pay_times]

    total_pv_paths = np.zeros(n_paths, dtype=np.float64)
    period_pvs: list[float] = []

    prev_idx = 0
    prev_t = 0.0
    for curr_idx, curr_t in zip(time_indices, pay_times, strict=True):
        dt_period = curr_t - prev_t
        cpi_prev = cpi_paths[:, prev_idx]
        cpi_curr = cpi_paths[:, curr_idx]
        df_curr = df_paths[:, curr_idx]

        float_return = (cpi_curr - cpi_prev) / np.maximum(cpi_prev, 1e-18)
        fixed_return = fixed_rate_ann * dt_period

        net_return = (
            (float_return - fixed_return)
            if is_payer
            else (fixed_return - float_return)
        )
        period_pv = notional * net_return * df_curr
        total_pv_paths += period_pv
        period_pvs.append(float(np.mean(period_pv)))

        prev_idx = curr_idx
        prev_t = curr_t

    price = float(np.mean(total_pv_paths))
    std_error = float(np.std(total_pv_paths) / np.sqrt(n_paths))

    return {
        "price": price,
        "std_error": std_error,
        "period_cash_flows": period_pvs,
    }


# Convenience alias for year-on-year inflation swap pricing
price_yoy_inflation_swap = price_year_on_year_inflation_swap


def benchmark_price_consumer_price_index_option(
    model: InflationModel,
    strike_rate_ann: float,
    maturity_yrs: float,
    notional: float = 1.0,
    option_type: OptionType | str = OptionType.CALL,
) -> dict[str, float]:
    """Analytical benchmark pricing for a European Zero-Coupon CPI Option.

    Args:
        model: An instantiated :class:`~xvasim.models.base.InflationModel`.
        strike_rate_ann: Annualised strike inflation rate :math:`K`.
        maturity_yrs: Option expiry in years.
        notional: Trade notional amount.
        option_type: :class:`OptionType` or string (``'call'`` / ``'put'``).

    Returns:
        Dictionary with ``"price"`` and ``"forward_cpi"``.
    """
    if not isinstance(model, InflationModel):
        msg = f"model must be an InflationModel, got {type(model).__name__}"
        raise TypeError(msg)

    if isinstance(option_type, str):
        try:
            resolved = OptionType(option_type.strip().lower())
        except ValueError:
            msg = (
                f"option_type must be OptionType.CALL, OptionType.PUT, "
                f"'call', or 'put', got {option_type!r}"
            )
            raise ValueError(msg) from None
    elif isinstance(option_type, OptionType):
        resolved = option_type
    else:
        msg = (
            f"option_type must be an OptionType or str, "
            f"got {type(option_type).__name__}"
        )
        raise TypeError(msg)

    is_call = resolved is OptionType.CALL
    k_comp = (1.0 + strike_rate_ann) ** maturity_yrs
    forward_cpi_val = model.forward_cpi(maturity_yrs)
    fwd_ratio = forward_cpi_val / model.base_cpi

    if isinstance(model, BlackInflationModel):
        price_val = model.price_consumer_price_index_option_analytical(
            strike_rate_ann=strike_rate_ann,
            maturity_yrs=maturity_yrs,
            notional=notional,
            is_call=is_call,
        )
        return {"price": price_val, "forward_cpi": float(forward_cpi_val)}

    if isinstance(model, JarrowYildirimModel):
        p_nom = float(
            model.nominal_ir_model.interpolate_discount_factor(maturity_yrs)
        )
        tot_var = model.total_variance_at(maturity_yrs)
        total_std = np.sqrt(max(tot_var, 1e-16))

        if total_std < 1e-12:
            intrinsic = (
                max(fwd_ratio - k_comp, 0.0)
                if is_call
                else max(k_comp - fwd_ratio, 0.0)
            )
            return {
                "price": float(notional * p_nom * intrinsic),
                "forward_cpi": float(forward_cpi_val),
            }

        d1 = (np.log(fwd_ratio / k_comp) + 0.5 * tot_var) / total_std
        d2 = d1 - total_std

        if is_call:
            pv = p_nom * (
                fwd_ratio * float(norm.cdf(d1))
                - k_comp * float(norm.cdf(d2))
            )
        else:
            pv = p_nom * (
                k_comp * float(norm.cdf(-d2))
                - fwd_ratio * float(norm.cdf(-d1))
            )
        return {
            "price": float(notional * pv),
            "forward_cpi": float(forward_cpi_val),
        }

    p_nom = (
        float(model.interpolate_nominal_df(maturity_yrs))
        if hasattr(model, "interpolate_nominal_df")
        else 1.0
    )
    intrinsic = (
        max(fwd_ratio - k_comp, 0.0) if is_call else max(k_comp - fwd_ratio, 0.0)
    )
    return {
        "price": float(notional * p_nom * intrinsic),
        "forward_cpi": float(forward_cpi_val),
    }


# Convenience alias for CPI option benchmark pricing
benchmark_price_cpi_option = benchmark_price_consumer_price_index_option


def price_consumer_price_index_option(
    model: InflationModel,
    strike_rate_ann: float,
    maturity_yrs: float,
    notional: float = 1.0,
    option_type: OptionType | str = OptionType.CALL,
    n_paths: int | None = 10_000,
    n_steps: int = 50,
    seed: int | None = None,
    random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
    scramble: bool = True,
) -> dict[str, float]:
    """Price a European Zero-Coupon CPI option via Monte Carlo.

    Payoff at maturity :math:`T`:
    - Call (Caplet): :math:`N \\times \\max(I(T)/I(0) - (1+K)^T, 0)`
    - Put (Floorlet): :math:`N \\times \\max((1+K)^T - I(T)/I(0), 0)`

    Args:
        model: An instantiated :class:`~xvasim.models.base.InflationModel`.
        strike_rate_ann: Annualised strike inflation rate :math:`K`.
        maturity_yrs: Option expiry in years.
        notional: Trade notional amount.
        option_type: :class:`OptionType` or string (``'call'`` / ``'put'``).
        n_paths: Number of Monte Carlo paths (default: 10,000). If None, computes
            analytical closed-form benchmark.
        n_steps: Number of simulation steps (for Monte Carlo).
        seed: Optional random seed.
        random_type: Random sequence generator type (:class:`RandomSequenceType`
            or str).
        scramble: If True, applies scrambling to QMC sequences.

    Returns:
        Dictionary containing ``price``, ``std_error`` (for MC),
        ``analytical_benchmark_price``, and ``forward_cpi``.
    """
    if not isinstance(model, InflationModel):
        msg = f"model must be an InflationModel, got {type(model).__name__}"
        raise TypeError(msg)

    ana_res = benchmark_price_consumer_price_index_option(
        model=model,
        strike_rate_ann=strike_rate_ann,
        maturity_yrs=maturity_yrs,
        notional=notional,
        option_type=option_type,
    )

    if n_paths is None:
        return ana_res

    if isinstance(option_type, str):
        resolved = OptionType(option_type.strip().lower())
    else:
        resolved = option_type

    is_call = resolved is OptionType.CALL
    k_comp = (1.0 + strike_rate_ann) ** maturity_yrs

    sim_res = model.simulate_paths(
        maturity_yrs=maturity_yrs,
        n_paths=n_paths,
        n_steps=n_steps,
        random_type=random_type,
        seed=seed,
        scramble=scramble,
    )

    cpi_t = sim_res.cpi_index[:, -1]
    df_t = sim_res.nominal_discount_factors[:, -1]
    ratio_t = cpi_t / model.base_cpi

    if is_call:
        intrinsic_paths = np.maximum(ratio_t - k_comp, 0.0)
    else:
        intrinsic_paths = np.maximum(k_comp - ratio_t, 0.0)

    payoff = notional * intrinsic_paths * df_t
    price_mc = float(np.mean(payoff))
    std_err = float(np.std(payoff) / np.sqrt(n_paths))

    return {
        "price": price_mc,
        "std_error": std_err,
        "analytical_benchmark_price": ana_res["price"],
        "forward_cpi": ana_res["forward_cpi"],
    }


# Convenience alias for CPI option pricing
price_cpi_option = price_consumer_price_index_option


# ---------------------------------------------------------------------------
# Interest Rate Swap & Cross-Currency Swap Pricing
# ---------------------------------------------------------------------------


def _parse_swap_leg_type(
    leg_type: SwapLegType | str, name: str = "leg_type"
) -> SwapLegType:
    """Parse and validate a SwapLegType enum or string representation."""
    if isinstance(leg_type, str):
        try:
            return SwapLegType(leg_type.strip().lower())
        except ValueError:
            msg = (
                f"{name} must be SwapLegType.FIXED, SwapLegType.FLOATING, "
                f"'fixed', or 'floating', got {leg_type!r}"
            )
            raise ValueError(msg) from None
    elif isinstance(leg_type, SwapLegType):
        return leg_type
    else:
        msg = (
            f"{name} must be a SwapLegType or str, "
            f"got {type(leg_type).__name__}"
        )
        raise TypeError(msg)


def _generate_swap_schedule(
    tenor_yrs: float | None,
    payment_times_yrs: np.ndarray | list[float] | None,
    pay_freq_yrs: float,
) -> np.ndarray:
    """Generate or validate sorted payment times in years."""
    if payment_times_yrs is not None:
        pay_times = np.sort(np.asarray(payment_times_yrs, dtype=np.float64))
        if len(pay_times) == 0:
            raise ValueError("payment_times_yrs must not be empty.")
        if float(pay_times[0]) <= 0.0:
            raise ValueError("Payment times must be strictly positive.")
        return pay_times
    elif tenor_yrs is not None:
        if tenor_yrs <= 0.0:
            raise ValueError("tenor_yrs must be strictly positive.")
        if pay_freq_yrs <= 0.0:
            raise ValueError("pay_freq_yrs must be strictly positive.")
        n_periods = max(1, int(np.round(tenor_yrs / pay_freq_yrs)))
        dt = tenor_yrs / n_periods
        return np.array(
            [(k + 1) * dt for k in range(n_periods)], dtype=np.float64
        )
    else:
        raise ValueError("Must provide either tenor_yrs or payment_times_yrs.")


def _get_ir_model(model: InterestRateModel | LGMParams) -> InterestRateModel:
    """Wrap legacy LGMParams in LGMModel if necessary."""
    if isinstance(model, InterestRateModel):
        return model
    elif isinstance(model, LGMParams):
        return LGMModel(model)
    else:
        msg = (
            f"model must be an InterestRateModel or LGMParams, "
            f"got {type(model).__name__}"
        )
        raise TypeError(msg)


def benchmark_price_interest_rate_swap(
    model: InterestRateModel | LGMParams,
    fixed_rate_ann: float,
    tenor_yrs: float | None = None,
    payment_times_yrs: np.ndarray | list[float] | None = None,
    pay_freq_yrs: float = 0.5,
    notional: float = 1.0,
    spread_ann: float = 0.0,
    is_payer: bool = True,
) -> dict[str, typing.Any]:
    r"""Analytical closed-form benchmark pricing for a single-currency IRS.

    Args:
        model: An instantiated :class:`~xvasim.models.base.InterestRateModel`
            or :class:`~xvasim.models.ir.lgm.LGMParams`.
        fixed_rate_ann: Annualised fixed coupon rate :math:`K` (e.g. 0.03 for 3%).
        tenor_yrs: Total swap tenor in years (used if *payment_times_yrs* is None).
        payment_times_yrs: Custom array or list of payment dates in years.
        pay_freq_yrs: Coupon payment frequency in years (default: 0.5 = semi-annual).
        notional: Swap notional amount (default: 1.0).
        spread_ann: Annualised spread added to the floating rate (default: 0.0).
        is_payer: True for Payer IRS, False for Receiver IRS.

    Returns:
        Dictionary containing analytical ``price``, ``fixed_leg_pv``,
        ``floating_leg_pv``, ``fair_swap_rate``, ``annuity``, and ``period_cash_flows``.
    """
    return price_interest_rate_swap(
        model=model,
        fixed_rate_ann=fixed_rate_ann,
        tenor_yrs=tenor_yrs,
        payment_times_yrs=payment_times_yrs,
        pay_freq_yrs=pay_freq_yrs,
        notional=notional,
        spread_ann=spread_ann,
        is_payer=is_payer,
        n_paths=None,
    )


# Convenience alias for benchmark interest rate swaps
benchmark_price_irs = benchmark_price_interest_rate_swap


def price_interest_rate_swap(
    model: InterestRateModel | LGMParams,
    fixed_rate_ann: float,
    tenor_yrs: float | None = None,
    payment_times_yrs: np.ndarray | list[float] | None = None,
    pay_freq_yrs: float = 0.5,
    notional: float = 1.0,
    spread_ann: float = 0.0,
    is_payer: bool = True,
    n_paths: int | None = 10_000,
    n_steps_per_year: int = 20,
    seed: int | None = None,
    random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
    scramble: bool = True,
) -> dict[str, typing.Any]:
    r"""Price a single-currency vanilla Interest Rate Swap (IRS).

    A vanilla IRS exchanges fixed-rate coupon payments for floating-rate
    coupon payments (e.g. forward Ibor / Libor / SOFR) on a series of
    payment dates :math:`T_1, T_2, \dots, T_n`.

    - **Payer Swap** (``is_payer=True``): Pays fixed rate :math:`K`, receives
      floating rate :math:`L(T_{i-1}, T_i) + s`.
    - **Receiver Swap** (``is_payer=False``): Receives fixed rate :math:`K`,
      pays floating rate :math:`L(T_{i-1}, T_i) + s`.

    Args:
        model: An instantiated :class:`~xvasim.models.base.InterestRateModel`
            or :class:`~xvasim.models.ir.lgm.LGMParams`.
        fixed_rate_ann: Annualised fixed coupon rate :math:`K` (e.g. 0.03 for 3%).
        tenor_yrs: Total swap tenor in years (used if *payment_times_yrs* is None).
        payment_times_yrs: Custom array or list of payment dates in years.
        pay_freq_yrs: Coupon payment frequency in years (default: 0.5 = semi-annual).
        notional: Swap notional amount (default: 1.0).
        spread_ann: Annualised spread added to the floating rate (default: 0.0).
        is_payer: True for Payer IRS, False for Receiver IRS.
        n_paths: Number of Monte Carlo simulation paths (default: 10,000).
            If None, computes the exact analytical closed-form price from
            the discount curve.
        n_steps_per_year: Number of simulation steps per year (for Monte Carlo).
        seed: Random seed for Monte Carlo simulation.
        random_type: Random sequence generator type (:class:`RandomSequenceType`
            or str).
        scramble: If True, applies scrambling to QMC sequences.

    Returns:
        Dictionary containing:

        - ``"price"`` — Net present value (PV) of the swap.
        - ``"fixed_leg_pv"`` — Present value of the fixed leg.
        - ``"floating_leg_pv"`` — Present value of the floating leg.
        - ``"fair_swap_rate"`` — Par/fair annualised swap rate :math:`S_0`.
        - ``"annuity"`` — Forward annuity / PV01 :math:`A_0 = \sum_i \tau_i P(0, T_i)`.
        - ``"std_error"`` — Monte Carlo standard error (if ``n_paths`` is provided).
        - ``"analytical_benchmark_price"`` — Exact analytical benchmark price.
        - ``"period_cash_flows"`` — List of detailed period cash flow dictionaries.
    """
    ir_model = _get_ir_model(model)
    pay_times = _generate_swap_schedule(
        tenor_yrs, payment_times_yrs, pay_freq_yrs
    )
    n_periods = len(pay_times)

    # Period boundaries: T_0 = 0.0, T_1, ..., T_n
    t_starts = np.insert(pay_times[:-1], 0, 0.0)
    t_ends = pay_times
    tau_vec = t_ends - t_starts

    # Analytical values from discount curve
    df_ends = np.array(
        [float(ir_model.interpolate_discount_factor(t)) for t in t_ends],
        dtype=np.float64,
    )
    df_starts = np.array(
        [float(ir_model.interpolate_discount_factor(t)) for t in t_starts],
        dtype=np.float64,
    )

    annuity = float(np.sum(tau_vec * df_ends))
    p_0 = float(df_starts[0])
    p_n = float(df_ends[-1])

    # Fair swap rate S_0 = (P(0, T_0) - P(0, T_n)) / Annuity
    fair_swap_rate = float((p_0 - p_n) / max(annuity, 1e-18))

    fixed_leg_pv_analytical = float(notional * fixed_rate_ann * annuity)
    floating_leg_pv_analytical = float(
        notional * ((p_0 - p_n) + spread_ann * annuity)
    )

    # Analytical period cash flows
    period_cfs: list[dict[str, float]] = []
    for i in range(n_periods):
        t_s = float(t_starts[i])
        t_e = float(t_ends[i])
        tau = float(tau_vec[i])
        df_e = float(df_ends[i])
        df_s = float(df_starts[i])

        fwd_rate = float((df_s / max(df_e, 1e-18) - 1.0) / tau)
        fixed_cf = notional * fixed_rate_ann * tau
        float_cf = notional * (fwd_rate + spread_ann) * tau
        net_cf = (float_cf - fixed_cf) if is_payer else (fixed_cf - float_cf)

        period_cfs.append({
            "start_time_yrs": t_s,
            "end_time_yrs": t_e,
            "year_fraction_yrs": tau,
            "discount_factor": df_e,
            "forward_rate_ann": fwd_rate,
            "fixed_payment": fixed_cf,
            "floating_payment": float_cf,
            "net_payment": net_cf,
            "discounted_net_pv": net_cf * df_e,
        })

    price_analytical = (
        floating_leg_pv_analytical - fixed_leg_pv_analytical
        if is_payer
        else fixed_leg_pv_analytical - floating_leg_pv_analytical
    )

    if n_paths is None:
        # Closed-form pricing
        return {
            "price": price_analytical,
            "fixed_leg_pv": fixed_leg_pv_analytical,
            "floating_leg_pv": floating_leg_pv_analytical,
            "fair_swap_rate": fair_swap_rate,
            "annuity": annuity,
            "period_cash_flows": period_cfs,
        }

    # Monte Carlo simulation
    grid_points: list[float] = [0.0]
    for i in range(n_periods):
        t_s = float(t_starts[i])
        t_e = float(t_ends[i])
        n_sub = max(1, int(np.ceil((t_e - t_s) * n_steps_per_year)))
        sub_grid = np.linspace(t_s, t_e, n_sub + 1)
        grid_points.extend(sub_grid[1:].tolist())

    sim_times = np.array(
        sorted(set(np.round(grid_points, decimals=8))), dtype=np.float64
    )

    idx_starts = [int(np.argmin(np.abs(sim_times - t))) for t in t_starts]
    idx_ends = [int(np.argmin(np.abs(sim_times - t))) for t in t_ends]

    x_paths = ir_model.simulate_paths(
        sim_times,
        n_paths,
        random_type=random_type,
        seed=seed,
        scramble=scramble,
    )
    df_paths = ir_model.discount_path(sim_times, x_paths)

    total_pv_paths = np.zeros(n_paths, dtype=np.float64)
    fixed_pv_paths = np.zeros(n_paths, dtype=np.float64)
    float_pv_paths = np.zeros(n_paths, dtype=np.float64)

    for i in range(n_periods):
        tau = float(tau_vec[i])
        idx_s = idx_starts[i]
        idx_e = idx_ends[i]
        t_s = float(t_starts[i])
        t_e = float(t_ends[i])

        if t_s == 0.0:
            df_init = float(ir_model.interpolate_discount_factor(t_e))
            l_rates = np.full(
                n_paths, (1.0 / df_init - 1.0) / tau, dtype=np.float64
            )
        else:
            state_at_reset = x_paths[:, idx_s]
            p_reset_end = ir_model.zero_coupon_bond(t_s, t_e, state_at_reset)
            l_rates = (1.0 / np.maximum(p_reset_end, 1e-18) - 1.0) / tau

        cf_fixed = notional * fixed_rate_ann * tau
        cf_float = notional * (l_rates + spread_ann) * tau
        df_at_pay = df_paths[:, idx_e]

        fixed_pv_paths += cf_fixed * df_at_pay
        float_pv_paths += cf_float * df_at_pay

        if is_payer:
            total_pv_paths += (cf_float - cf_fixed) * df_at_pay
        else:
            total_pv_paths += (cf_fixed - cf_float) * df_at_pay

    price_mc = float(np.mean(total_pv_paths))
    std_error = float(np.std(total_pv_paths) / np.sqrt(n_paths))
    fixed_leg_pv_mc = float(np.mean(fixed_pv_paths))
    float_leg_pv_mc = float(np.mean(float_pv_paths))

    return {
        "price": price_mc,
        "std_error": std_error,
        "fixed_leg_pv": fixed_leg_pv_mc,
        "floating_leg_pv": float_leg_pv_mc,
        "analytical_benchmark_price": price_analytical,
        "fair_swap_rate": fair_swap_rate,
        "annuity": annuity,
        "period_cash_flows": period_cfs,
    }


# Convenience alias for interest rate swaps
price_irs = price_interest_rate_swap


def benchmark_price_cross_currency_swap(
    model: TwoCurrencyFXModel | FXLGMParams | FXModel,
    domestic_rate_ann: float = 0.0,
    foreign_rate_ann: float = 0.0,
    domestic_spread_ann: float = 0.0,
    foreign_spread_ann: float = 0.0,
    domestic_leg_type: SwapLegType | str = SwapLegType.FIXED,
    foreign_leg_type: SwapLegType | str = SwapLegType.FLOATING,
    tenor_yrs: float | None = None,
    payment_times_yrs: np.ndarray | list[float] | None = None,
    pay_freq_yrs: float = 0.5,
    foreign_notional: float = 1.0,
    domestic_notional: float | None = None,
    is_domestic_payer: bool = True,
    exchange_notionals: bool = True,
) -> dict[str, typing.Any]:
    r"""Exact analytical closed-form benchmark pricing for Cross-Currency Swaps.

    Args:
        model: A :class:`~xvasim.models.fx.TwoCurrencyFXModel`, legacy
            :class:`FXLGMParams`, or a modular :class:`~xvasim.models.base.FXModel`.
        domestic_rate_ann: Annualised fixed coupon on domestic leg (if fixed).
        foreign_rate_ann: Annualised fixed coupon on foreign leg (if fixed).
        domestic_spread_ann: Annualised spread on domestic floating rate (if floating).
        foreign_spread_ann: Annualised spread on foreign floating rate (if floating).
        domestic_leg_type: Leg type for domestic currency (:class:`SwapLegType`
            or ``"fixed"`` / ``"floating"``).
        foreign_leg_type: Leg type for foreign currency (:class:`SwapLegType`
            or ``"fixed"`` / ``"floating"``).
        tenor_yrs: Total swap tenor in years (used if *payment_times_yrs* is None).
        payment_times_yrs: Custom array or list of payment dates in years.
        pay_freq_yrs: Payment frequency in years (default: 0.5 = semi-annual).
        foreign_notional: Foreign currency notional amount (default: 1.0).
        domestic_notional: Domestic currency notional amount. If None, defaults
            to ``foreign_notional * spot_fx`` (at-the-money notional matching).
        is_domestic_payer: True if paying domestic leg and receiving foreign leg;
            False if receiving domestic leg and paying foreign leg.
        exchange_notionals: If True, includes principal notional exchanges at
            inception (:math:`t=0`) and maturity (:math:`t=T`).

    Returns:
        Dictionary with analytical benchmark results.
    """
    return price_cross_currency_swap(
        model=model,
        domestic_rate_ann=domestic_rate_ann,
        foreign_rate_ann=foreign_rate_ann,
        domestic_spread_ann=domestic_spread_ann,
        foreign_spread_ann=foreign_spread_ann,
        domestic_leg_type=domestic_leg_type,
        foreign_leg_type=foreign_leg_type,
        tenor_yrs=tenor_yrs,
        payment_times_yrs=payment_times_yrs,
        pay_freq_yrs=pay_freq_yrs,
        foreign_notional=foreign_notional,
        domestic_notional=domestic_notional,
        is_domestic_payer=is_domestic_payer,
        exchange_notionals=exchange_notionals,
        n_paths=None,
    )


# Convenience alias for cross-currency swap analytical benchmark
benchmark_price_xccy_swap = benchmark_price_cross_currency_swap


def price_cross_currency_swap(
    model: TwoCurrencyFXModel | FXLGMParams | FXModel,
    domestic_rate_ann: float = 0.0,
    foreign_rate_ann: float = 0.0,
    domestic_spread_ann: float = 0.0,
    foreign_spread_ann: float = 0.0,
    domestic_leg_type: SwapLegType | str = SwapLegType.FIXED,
    foreign_leg_type: SwapLegType | str = SwapLegType.FLOATING,
    tenor_yrs: float | None = None,
    payment_times_yrs: np.ndarray | list[float] | None = None,
    pay_freq_yrs: float = 0.5,
    foreign_notional: float = 1.0,
    domestic_notional: float | None = None,
    is_domestic_payer: bool = True,
    exchange_notionals: bool = True,
    n_paths: int | None = 10_000,
    n_steps: int = 100,
    seed: int | None = None,
    random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
    scramble: bool = True,
) -> dict[str, typing.Any]:
    r"""Price a Cross-Currency Swap (XCCY) supporting multi-currency rate dynamics.

    A cross-currency swap exchanges interest cash flows and principal notionals
    between two different currencies: **Domestic** (numeraire currency) and
    **Foreign**. Supported leg types on either side include:
    - **Fixed vs. Floating** (e.g. Fixed domestic coupon vs Floating foreign rate)
    - **Fixed vs. Fixed** (Fixed domestic coupon vs Fixed foreign coupon)
    - **Floating vs. Floating** (Cross-currency basis swap)

    Args:
        model: A :class:`~xvasim.models.fx.TwoCurrencyFXModel`, legacy
            :class:`FXLGMParams`, or a modular :class:`~xvasim.models.base.FXModel`.
        domestic_rate_ann: Annualised fixed coupon on domestic leg (if fixed).
        foreign_rate_ann: Annualised fixed coupon on foreign leg (if fixed).
        domestic_spread_ann: Annualised spread on domestic floating rate (if floating).
        foreign_spread_ann: Annualised spread on foreign floating rate (if floating).
        domestic_leg_type: Leg type for domestic currency (:class:`SwapLegType`
            or ``"fixed"`` / ``"floating"``).
        foreign_leg_type: Leg type for foreign currency (:class:`SwapLegType`
            or ``"fixed"`` / ``"floating"``).
        tenor_yrs: Total swap tenor in years (used if *payment_times_yrs* is None).
        payment_times_yrs: Custom array or list of payment dates in years.
        pay_freq_yrs: Payment frequency in years (default: 0.5 = semi-annual).
        foreign_notional: Foreign currency notional amount (default: 1.0).
        domestic_notional: Domestic currency notional amount. If None, defaults
            to ``foreign_notional * spot_fx`` (at-the-money notional matching).
        is_domestic_payer: True if paying domestic leg and receiving foreign leg;
            False if receiving domestic leg and paying foreign leg.
        exchange_notionals: If True, includes principal notional exchanges at
            inception (:math:`t=0`) and maturity (:math:`t=T`).
        n_paths: Number of Monte Carlo simulation paths (default: 10,000).
            If None, computes the exact analytical closed-form price.
        n_steps: Number of simulation steps (for Monte Carlo).
        seed: Random seed for Monte Carlo simulation.
        random_type: Random sequence generator type (:class:`RandomSequenceType`
            or str).
        scramble: If True, applies scrambling to QMC sequences.

    Returns:
        Dictionary containing:

        - ``"price"`` — Net present value in domestic currency.
        - ``"domestic_leg_pv"`` — PV of domestic leg in domestic currency.
        - ``"foreign_leg_pv_domestic"`` — PV of foreign leg (domestic currency).
        - ``"foreign_leg_pv_foreign"`` — PV of foreign leg (foreign currency).
        - ``"notional_exchange_pv"`` — PV of notional exchanges (domestic currency).
        - ``"fair_foreign_rate"`` — Fair fixed foreign rate.
        - ``"fair_foreign_spread"`` — Fair foreign basis spread.
        - ``"fair_domestic_rate"`` — Fair fixed domestic rate.
        - ``"fair_domestic_spread"`` — Fair domestic spread.
        - ``"annuity_domestic"`` — Domestic forward annuity / PV01.
        - ``"annuity_foreign"`` — Foreign forward annuity / PV01.
        - ``"std_error"`` — Monte Carlo standard error (if ``n_paths`` is provided).
        - ``"analytical_benchmark_price"`` — Exact analytical benchmark price.
    """
    if isinstance(model, FXLGMParams):
        fx_model = TwoCurrencyFXModel.from_lgm_params(
            domestic=model.domestic,
            foreign=model.foreign,
            spot_fx=model.spot_fx,
            fx_vol_ann=model.fx_vol_ann,
            correlation_matrix=model.correlation_matrix,
        )
    elif isinstance(model, TwoCurrencyFXModel):
        fx_model = model
    elif isinstance(model, FXModel) and hasattr(model, "domestic_ir_model"):
        fx_model = typing.cast(TwoCurrencyFXModel, model)
    else:
        msg = (
            f"model must be TwoCurrencyFXModel, FXLGMParams, or FXModel "
            f"with domestic/foreign IR models, got {type(model).__name__}"
        )
        raise TypeError(msg)

    dom_ir = fx_model.domestic_ir_model
    for_ir = fx_model.foreign_ir_model
    spot_fx = fx_model.spot_fx

    dom_type = _parse_swap_leg_type(domestic_leg_type, "domestic_leg_type")
    for_type = _parse_swap_leg_type(foreign_leg_type, "foreign_leg_type")

    for_notional = float(foreign_notional)
    dom_notional = (
        float(domestic_notional)
        if domestic_notional is not None
        else for_notional * spot_fx
    )

    pay_times = _generate_swap_schedule(
        tenor_yrs, payment_times_yrs, pay_freq_yrs
    )
    n_periods = len(pay_times)

    t_starts = np.insert(pay_times[:-1], 0, 0.0)
    t_ends = pay_times
    tau_vec = t_ends - t_starts

    df_d_ends = np.array(
        [float(dom_ir.interpolate_discount_factor(t)) for t in t_ends],
        dtype=np.float64,
    )
    df_d_starts = np.array(
        [float(dom_ir.interpolate_discount_factor(t)) for t in t_starts],
        dtype=np.float64,
    )
    df_f_ends = np.array(
        [float(for_ir.interpolate_discount_factor(t)) for t in t_ends],
        dtype=np.float64,
    )
    df_f_starts = np.array(
        [float(for_ir.interpolate_discount_factor(t)) for t in t_starts],
        dtype=np.float64,
    )

    annuity_dom = float(np.sum(tau_vec * df_d_ends))
    annuity_for = float(np.sum(tau_vec * df_f_ends))

    # Domestic leg PV
    if dom_type is SwapLegType.FIXED:
        dom_leg_pv = dom_notional * domestic_rate_ann * annuity_dom
    else:
        dom_leg_pv = dom_notional * (
            (df_d_starts[0] - df_d_ends[-1])
            + domestic_spread_ann * annuity_dom
        )

    # Foreign leg PV (in foreign currency)
    if for_type is SwapLegType.FIXED:
        for_leg_pv_foreign = for_notional * foreign_rate_ann * annuity_for
    else:
        for_leg_pv_foreign = for_notional * (
            (df_f_starts[0] - df_f_ends[-1]) + foreign_spread_ann * annuity_for
        )

    for_leg_pv_domestic = spot_fx * for_leg_pv_foreign

    # Notional exchanges PV (in domestic currency)
    if exchange_notionals:
        init_exchange_pv = -dom_notional + spot_fx * for_notional
        final_exchange_pv = (
            dom_notional * df_d_ends[-1] - spot_fx * for_notional * df_f_ends[-1]
        )
        notional_exchange_pv = init_exchange_pv + final_exchange_pv
    else:
        notional_exchange_pv = 0.0

    # Net swap price (analytical)
    if is_domestic_payer:
        price_analytical = (
            for_leg_pv_domestic - dom_leg_pv + notional_exchange_pv
        )
    else:
        price_analytical = (
            dom_leg_pv - for_leg_pv_domestic - notional_exchange_pv
        )

    # Fair rate / spread solving (analytical)
    target_for_pv_dom = dom_leg_pv - notional_exchange_pv
    target_for_pv_for = target_for_pv_dom / max(spot_fx, 1e-18)

    if for_type is SwapLegType.FIXED:
        fair_foreign_rate = float(
            target_for_pv_for / (for_notional * max(annuity_for, 1e-18))
        )
        fair_foreign_spread = 0.0
    else:
        fair_foreign_rate = 0.0
        fair_foreign_spread = float(
            (
                target_for_pv_for / for_notional
                - (df_f_starts[0] - df_f_ends[-1])
            )
            / max(annuity_for, 1e-18)
        )

    target_dom_pv = for_leg_pv_domestic + notional_exchange_pv
    if dom_type is SwapLegType.FIXED:
        fair_domestic_rate = float(
            target_dom_pv / (dom_notional * max(annuity_dom, 1e-18))
        )
        fair_domestic_spread = 0.0
    else:
        fair_domestic_rate = 0.0
        fair_domestic_spread = float(
            (target_dom_pv / dom_notional - (df_d_starts[0] - df_d_ends[-1]))
            / max(annuity_dom, 1e-18)
        )

    base_result: dict[str, typing.Any] = {
        "price": price_analytical,
        "domestic_leg_pv": dom_leg_pv,
        "foreign_leg_pv_domestic": for_leg_pv_domestic,
        "foreign_leg_pv_foreign": for_leg_pv_foreign,
        "notional_exchange_pv": notional_exchange_pv,
        "fair_foreign_rate": fair_foreign_rate,
        "fair_foreign_spread": fair_foreign_spread,
        "fair_domestic_rate": fair_domestic_rate,
        "fair_domestic_spread": fair_domestic_spread,
        "annuity_domestic": annuity_dom,
        "annuity_foreign": annuity_for,
    }

    if n_paths is None:
        return base_result

    # Monte Carlo simulation
    maturity_yrs = float(t_ends[-1])
    n_steps_actual = max(n_steps, n_periods * 5)

    sim_times, x_dom, x_for, fx_spot = fx_model.simulate_paths(
        maturity_yrs=maturity_yrs,
        n_paths=n_paths,
        n_steps=n_steps_actual,
        random_type=random_type,
        seed=seed,
        scramble=scramble,
    )

    df_d_paths = dom_ir.discount_path(sim_times, x_dom)

    idx_starts = [int(np.argmin(np.abs(sim_times - t))) for t in t_starts]
    idx_ends = [int(np.argmin(np.abs(sim_times - t))) for t in t_ends]

    pv_d_paths = np.zeros(n_paths, dtype=np.float64)
    pv_f_dom_paths = np.zeros(n_paths, dtype=np.float64)
    pv_f_for_paths = np.zeros(n_paths, dtype=np.float64)

    for i in range(n_periods):
        tau = float(tau_vec[i])
        idx_s = idx_starts[i]
        idx_e = idx_ends[i]
        t_s = float(t_starts[i])
        t_e = float(t_ends[i])

        # Domestic coupon
        if dom_type is SwapLegType.FIXED:
            cf_d = np.full(
                n_paths, dom_notional * domestic_rate_ann * tau, dtype=np.float64
            )
        else:
            if t_s == 0.0:
                df_d_0_e = float(dom_ir.interpolate_discount_factor(t_e))
                l_d = np.full(
                    n_paths, (1.0 / df_d_0_e - 1.0) / tau, dtype=np.float64
                )
            else:
                p_d_mat = dom_ir.zero_coupon_bond(t_s, t_e, x_dom[:, idx_s])
                l_d = (1.0 / np.maximum(p_d_mat, 1e-18) - 1.0) / tau
            cf_d = dom_notional * (l_d + domestic_spread_ann) * tau

        # Foreign coupon
        if for_type is SwapLegType.FIXED:
            cf_f = np.full(
                n_paths, foreign_notional * foreign_rate_ann * tau, dtype=np.float64
            )
        else:
            if t_s == 0.0:
                df_f_0_e = float(for_ir.interpolate_discount_factor(t_e))
                l_f = np.full(
                    n_paths, (1.0 / df_f_0_e - 1.0) / tau, dtype=np.float64
                )
            else:
                p_f_mat = for_ir.zero_coupon_bond(t_s, t_e, x_for[:, idx_s])
                l_f = (1.0 / np.maximum(p_f_mat, 1e-18) - 1.0) / tau
            cf_f = foreign_notional * (l_f + foreign_spread_ann) * tau

        cf_f_dom = cf_f * fx_spot[:, idx_e]
        df_d_pay = df_d_paths[:, idx_e]

        pv_d_paths += cf_d * df_d_pay
        pv_f_dom_paths += cf_f_dom * df_d_pay
        pv_f_for_paths += cf_f * float(df_f_ends[i])

    if exchange_notionals:
        idx_mat = idx_ends[-1]
        init_notional_dom = -dom_notional + spot_fx * for_notional
        final_notional_dom_paths = (
            dom_notional - fx_spot[:, idx_mat] * for_notional
        ) * df_d_paths[:, idx_mat]
        notional_pv_paths = init_notional_dom + final_notional_dom_paths
    else:
        notional_pv_paths = np.zeros(n_paths, dtype=np.float64)

    if is_domestic_payer:
        total_pv_mc = (pv_f_dom_paths - pv_d_paths) + notional_pv_paths
    else:
        total_pv_mc = (pv_d_paths - pv_f_dom_paths) - notional_pv_paths

    price_mc = float(np.mean(total_pv_mc))
    std_error = float(np.std(total_pv_mc) / np.sqrt(n_paths))

    base_result["price"] = price_mc
    base_result["std_error"] = std_error
    base_result["domestic_leg_pv"] = float(np.mean(pv_d_paths))
    base_result["foreign_leg_pv_domestic"] = float(np.mean(pv_f_dom_paths))
    base_result["foreign_leg_pv_foreign"] = float(np.mean(pv_f_for_paths))
    base_result["notional_exchange_pv"] = float(np.mean(notional_pv_paths))
    base_result["analytical_benchmark_price"] = price_analytical

    return base_result


# Convenience alias for cross-currency swaps
price_xccy_swap = price_cross_currency_swap


