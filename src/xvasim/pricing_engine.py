"""LGM-based Monte Carlo pricing engine for currency forwards and options.

This module implements a two-currency Linear Gauss-Markov (LGM) model for
pricing FX derivatives via Monte Carlo simulation.  The LGM model drives
interest-rate dynamics through a Gaussian state variable ``x(t)`` whose
volatility function ``σ(t)`` is calibrated to swaption market data supplied
by the caller.

Public API
----------
- :class:`LGMParams`  — single-currency calibrated LGM parameters.
- :class:`FXLGMParams` — two-currency + FX spot model parameters.
- :class:`OptionType`  — enumeration of supported option types.
- :func:`calibrate_lgm_to_swaptions` — calibrate ``σ(t)`` to swaptions.
- :func:`price_fx_forward` — MC price a currency forward.
- :func:`price_fx_option`  — MC price a European currency option.

Units & Conventions
-------------------
- Time / tenor in **years** (suffix ``_yrs``).
- Rates / vols as **annualised decimals** (suffix ``_ann``).
"""

from __future__ import annotations

import dataclasses
import enum

import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LGMParams:
    """Calibrated parameters for a single-currency LGM model.

    Attributes:
        kappa_ann: Mean-reversion speed (annualised, e.g. 0.03).
        sigma_grid_yrs: 1-D array of *breakpoint* times (years) for the
            piecewise-constant volatility function.  Must be sorted in
            ascending order with strictly positive entries.
        sigma_values_ann: 1-D array of piecewise-constant volatility values
            (annualised).  ``sigma_values_ann[i]`` applies on the interval
            ``(sigma_grid_yrs[i-1], sigma_grid_yrs[i]]`` (with
            ``sigma_grid_yrs[-1] = 0`` implied for the first bucket).
            Must have the same length as *sigma_grid_yrs*.
        discount_curve_yrs: 1-D array of tenors (years) defining the
            risk-free discount curve.  Must be sorted ascending and start
            at or above 0.
        discount_factors: 1-D array of discount factors corresponding to
            *discount_curve_yrs*.  Same length as *discount_curve_yrs*.
    """

    kappa_ann: float
    sigma_grid_yrs: np.ndarray
    sigma_values_ann: np.ndarray
    discount_curve_yrs: np.ndarray
    discount_factors: np.ndarray


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
        PUT:  European put  — payoff ``N × max(K − S(T), 0)``.
    """

    CALL = "call"
    PUT = "put"


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
            # Limit κ → 0: integral becomes σ² · ds
            zeta += sig * sig * ds
        else:
            # ∫_{s_start}^{s_end} σ² exp(-2κ(t-s)) ds
            #   = σ²/(2κ) [exp(-2κ(t-s_end)) - exp(-2κ(t-s_start))]  — wrong sign
            #   corrected:
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

    Both are expressed as *normal (Bachelier)* prices.  The LGM normal vol
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
    # Build swap schedule
    n_periods = max(1, round(swap_tenor_yrs / pay_freq_yrs))
    actual_freq = swap_tenor_yrs / n_periods
    payment_times = np.array(
        [expiry_yrs + actual_freq * (k + 1) for k in range(n_periods)]
    )

    # Discount factors at payment dates
    dfs = _interpolate_discount_factor(payment_times, curve_yrs, curve_dfs)
    float(_interpolate_discount_factor(expiry_yrs, curve_yrs, curve_dfs))

    # Forward annuity (PV01)
    annuity = float(np.sum(actual_freq * dfs))

    # H-function at payment dates
    h_vals = _compute_h_function(payment_times, kappa)

    # LGM swaption variance  ≈  ζ(T₀) × (dS/dx)² where dS/dx is the
    # swap-rate sensitivity approximated via H-weighted annuity
    zeta = _compute_zeta(expiry_yrs, sigma_grid_yrs, sigma_values_ann, kappa)

    # dP/dx at each payment date (bond sensitivity to x)
    h_expiry = float(_compute_h_function(expiry_yrs, kappa))
    delta_h = h_vals - h_expiry  # H(Tᵢ) - H(T₀)

    # Swap rate sensitivity to x:  dS/dx ≈ (P(0,T₀) - P(0,T_n) -
    #       fixed_rate × Σ τᵢ P(0,Tᵢ) × δHᵢ) / A₀
    # Simplified (assuming ATM, P(0,T₀) - P(0,Tn) ≈ S × A₀):
    #   dS/dx ≈ -1/A₀ × Σ τᵢ P(0,Tᵢ) δHᵢ × fixed_rate
    #         + (P(0,T₀)δH_0 )/A₀   (≈ 0 since δH₀=0)
    # A good first-order approximation for the *normal vol*:
    weighted_delta_h = float(np.sum(actual_freq * dfs * delta_h)) / annuity

    model_normal_var = zeta * weighted_delta_h**2  # total variance
    model_normal_vol = np.sqrt(max(model_normal_var / expiry_yrs, 0.0))

    # Bachelier price = A₀ × σ_n × √T₀ × φ(0)   [ATM, d=0 ⇒ φ(0) ≈ 0.3989]
    sqrt_t = np.sqrt(expiry_yrs)
    phi_0 = 1.0 / np.sqrt(2.0 * np.pi)  # ≈ 0.3989

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
        kappa_ann: Mean-reversion speed (annualised).  Default 0.03.
        pay_freq_yrs: Fixed-leg payment frequency in years.  Default 0.5.

    Returns:
        A fully populated :class:`LGMParams` instance.

    Raises:
        RuntimeError: If any single-expiry root-finding fails.
    """
    n = len(swaption_expiries_yrs)
    sigma_grid = np.array(swaption_expiries_yrs, dtype=np.float64)
    sigma_vals = np.zeros(n, dtype=np.float64)

    for i in range(n):
        # Build partial σ grid up to (and including) bucket i
        partial_grid = sigma_grid[: i + 1]

        def _price_diff(sigma_i: float, idx: int = i) -> float:
            partial_vals = sigma_vals[: idx + 1].copy()
            partial_vals[idx] = sigma_i
            model_p, market_p = _lgm_swaption_price_normal(
                expiry_yrs=float(swaption_expiries_yrs[idx]),
                swap_tenor_yrs=float(swap_tenors_yrs[idx]),
                market_normal_vol_ann=float(market_normal_vols_ann[idx]),
                kappa=kappa_ann,
                sigma_grid_yrs=partial_grid,
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
    rng: np.random.Generator,
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
        rng: NumPy random Generator for reproducibility.

    Returns:
        ``(times, x_dom, x_for, fx_spot)`` — all arrays of shape
        ``(n_paths, n_steps + 1)`` except *times* which is
        ``(n_steps + 1,)``.
    """
    dt = maturity_yrs / n_steps
    sqrt_dt = np.sqrt(dt)
    times = np.linspace(0.0, maturity_yrs, n_steps + 1)

    # Cholesky decomposition of the correlation matrix
    chol = np.linalg.cholesky(params.correlation_matrix)

    # Pre-allocate state arrays
    x_dom = np.zeros((n_paths, n_steps + 1))
    x_for = np.zeros((n_paths, n_steps + 1))
    ln_fx = np.zeros((n_paths, n_steps + 1))
    ln_fx[:, 0] = np.log(params.spot_fx)

    kd = params.domestic.kappa_ann
    kf = params.foreign.kappa_ann
    vol_fx = params.fx_vol_ann

    for step in range(n_steps):
        t = times[step]

        # Piecewise-constant σ at time t for each currency
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

        # Correlated normal increments  (n_paths × 3)
        z_indep = rng.standard_normal((n_paths, 3))
        z_corr = z_indep @ chol.T  # (n_paths, 3)

        dw_d = z_corr[:, 0] * sqrt_dt
        dw_f = z_corr[:, 1] * sqrt_dt
        dw_fx = z_corr[:, 2] * sqrt_dt

        # Domestic LGM state:  dx_d = -κ_d x_d dt + σ_d dW_d
        x_dom[:, step + 1] = x_dom[:, step] - kd * x_dom[:, step] * dt + sig_d * dw_d

        # Foreign LGM state:  dx_f = (-κ_f x_f - σ_f ρ_{f,fx} vol_fx) dt
        #                              + σ_f dW_f
        # The quanto drift adjustment  -σ_f ρ_{f,fx} vol_fx  arises because
        # we are simulating the foreign rate under the *domestic* measure.
        rho_f_fx = params.correlation_matrix[1, 2]
        x_for[:, step + 1] = (
            x_for[:, step]
            - kf * x_for[:, step] * dt
            - sig_f * rho_f_fx * vol_fx * dt
            + sig_f * dw_f
        )

        # Log-FX:  d(ln S) = (r_d - r_f - ½ vol_fx²) dt + vol_fx dW_fx
        # Under LGM the instantaneous short rate is:
        #   r(t) = f(0,t) + H'(t) x(t) + ½ H'(t)² ζ(t)
        # For the *drift* of ln(S) we use the difference of instantaneous
        # forward rates plus the x-dependent terms.  Because the forwards
        # are already embedded in the discount curve, we use the simpler
        # discretisation that relies on the ratio of discount factors:
        #   E[S(T)] = S(0) P_f(0,T) / P_d(0,T)
        # and ensure the log-FX drift is consistent:
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

        # Instantaneous forward rates from the curve
        fwd_d = -np.log(df_d_t1 / df_d_t) / dt
        fwd_f = -np.log(df_f_t1 / df_f_t) / dt

        # H'(t) = exp(-κ t)
        h_prime_d = np.exp(-kd * t)
        h_prime_f = np.exp(-kf * t)

        # Stochastic short-rate corrections
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

    Uses the relationship:
        D(0, t) = P(0, t) × exp(-H(t) x(t) - ½ H(t)² ζ(t))

    but for discounting cash-flows we need cumulative discounting from 0
    to t.  We use the *bank-account numeraire* built from instantaneous
    short rates:
        B(t) = exp(∫₀ᵗ r(s) ds)

    approximated via the trapezoidal rule on the simulated short-rate path.

    Args:
        lgm: Single-currency LGM parameters.
        x_state: State variable paths, shape ``(n_paths, n_steps + 1)``.
        times: Time grid, shape ``(n_steps + 1,)``.

    Returns:
        Array of discount factors ``D(0, tᵢ)`` with same shape as *x_state*.
    """
    n_paths, n_times = x_state.shape
    kappa = lgm.kappa_ann

    # Short rates at each grid point
    short_rates = np.zeros_like(x_state)
    for j in range(n_times):
        t = times[j]
        h_prime = np.exp(-kappa * t)
        fwd = _instantaneous_forward(t, lgm.discount_curve_yrs, lgm.discount_factors)
        zeta = _compute_zeta(t, lgm.sigma_grid_yrs, lgm.sigma_values_ann, kappa)
        short_rates[:, j] = fwd + h_prime * x_state[:, j] + 0.5 * h_prime**2 * zeta

    # Cumulative integral via trapezoidal rule
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
# Public pricing functions
# ---------------------------------------------------------------------------


def price_fx_forward(
    params: FXLGMParams,
    strike: float,
    maturity_yrs: float,
    notional: float,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int | None = 42,
) -> dict[str, float | np.ndarray]:
    """Price a currency forward via Monte Carlo under the LGM model.

    The forward buyer receives ``N × (S(T) − K)`` at maturity *T*, discounted
    to today using the domestic bank account.

    Args:
        params: Two-currency LGM + FX model parameters.
        strike: Forward strike (domestic per foreign).
        maturity_yrs: Maturity in years.
        notional: Notional amount in foreign currency.
        n_paths: Number of Monte Carlo paths.
        n_steps: Number of simulation time steps.
        seed: Random seed (``None`` for non-deterministic).

    Returns:
        Dictionary with keys:

        - ``"price"`` — MC forward value (domestic currency).
        - ``"std_error"`` — standard error of the MC estimate.
        - ``"fx_terminal"`` — 1-D array of terminal FX rates.
    """
    rng = np.random.default_rng(seed)
    times, x_dom, _x_for, fx_paths = _simulate_fx_paths(
        params, maturity_yrs, n_paths, n_steps, rng
    )

    # Terminal FX rate
    s_t = fx_paths[:, -1]

    # Domestic discount factor along each path
    dom_df = _discount_path(params.domestic, x_dom, times)
    df_t = dom_df[:, -1]

    # Discounted payoff
    payoff = notional * (s_t - strike) * df_t
    price = float(np.mean(payoff))
    std_error = float(np.std(payoff) / np.sqrt(n_paths))

    return {"price": price, "std_error": std_error, "fx_terminal": s_t}


def price_fx_option(
    params: FXLGMParams,
    strike: float,
    maturity_yrs: float,
    notional: float,
    option_type: OptionType | str = OptionType.CALL,
    n_paths: int = 100_000,
    n_steps: int = 100,
    seed: int | None = 42,
) -> dict[str, float | np.ndarray]:
    r"""Price a European currency option via Monte Carlo under the LGM model.

    Payoffs:

    - **Call**: :math:`N \times \max(S(T) - K,\; 0)`
    - **Put**:  :math:`N \times \max(K - S(T),\; 0)`

    Args:
        params: Two-currency LGM + FX model parameters.
        strike: Option strike (domestic per foreign).
        maturity_yrs: Expiry in years.
        notional: Notional in foreign currency.
        option_type: :class:`OptionType` member or equivalent string
            (``"call"`` / ``"put"``).  Strings are accepted for
            backwards compatibility and are resolved case-insensitively.
        n_paths: Number of Monte Carlo paths.
        n_steps: Number of simulation time steps.
        seed: Random seed.

    Returns:
        Dictionary with keys:

        - ``"price"`` — MC option premium (domestic currency).
        - ``"std_error"`` — standard error of the MC estimate.
        - ``"fx_terminal"`` — 1-D array of terminal FX rates.

    Raises:
        ValueError: If *option_type* is not a valid
            :class:`OptionType` member or recognised string.
    """
    # Normalise to OptionType enum
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

    rng = np.random.default_rng(seed)
    times, x_dom, _x_for, fx_paths = _simulate_fx_paths(
        params, maturity_yrs, n_paths, n_steps, rng
    )

    s_t = fx_paths[:, -1]

    dom_df = _discount_path(params.domestic, x_dom, times)
    df_t = dom_df[:, -1]

    if resolved is OptionType.CALL:
        intrinsic = np.maximum(s_t - strike, 0.0)
    else:
        intrinsic = np.maximum(strike - s_t, 0.0)

    payoff = notional * intrinsic * df_t
    price = float(np.mean(payoff))
    std_error = float(np.std(payoff) / np.sqrt(n_paths))

    return {"price": price, "std_error": std_error, "fx_terminal": s_t}
