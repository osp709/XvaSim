"""Numba JIT-compiled numeric kernels for Monte Carlo simulation and calibration.

This module provides high-performance JIT-compiled kernels (compiled via Numba
with ``fastmath=True``) for:
- CIR survival probability and L-BFGS-B calibration objective evaluation
- Monte Carlo path evolution across Interest Rate and FX models
- Trapezoidal numerical integration for path-wise discount factors

Public API
----------
- :func:`cir_calibration_objective_kernel` — compiled CIR calibration objective.
- :func:`cir_simulate_paths_kernel` — compiled CIR short rate path stepping.
- :func:`cir_survival_probability_kernel` — compiled CIR survival probabilities.
- :func:`discount_path_kernel` — compiled cumulative discount factor calculation.
- :func:`heston_simulate_paths_kernel` — compiled Full Truncation Heston stepping.
- :func:`hull_white_simulate_paths_kernel` — compiled Hull-White 1F path stepping.
- :func:`is_numba_available` — check whether Numba JIT is active.
- :func:`lgm_simulate_paths_kernel` — compiled LGM state path stepping.
- :func:`vasicek_simulate_paths_kernel` — compiled Vasicek short rate path stepping.
"""

from __future__ import annotations

import typing

import numpy as np

__all__ = [
    "cir_calibration_objective_kernel",
    "cir_simulate_paths_kernel",
    "cir_survival_probability_kernel",
    "discount_path_kernel",
    "heston_simulate_paths_kernel",
    "hull_white_simulate_paths_kernel",
    "is_numba_available",
    "lgm_simulate_paths_kernel",
    "vasicek_simulate_paths_kernel",
]

# ---------------------------------------------------------------------------
# Numba JIT compilation wrapper with graceful fallback
# ---------------------------------------------------------------------------

try:
    import numba  # type: ignore[import-not-found]

    HAS_NUMBA = True
    njit = numba.njit(fastmath=True, nogil=True)
    njit_safe = numba.njit(fastmath=False, nogil=True)
except Exception:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

    njit_safe = njit


def is_numba_available() -> bool:
    """Return True if Numba JIT compilation is installed and active."""
    return HAS_NUMBA


# ---------------------------------------------------------------------------
# JIT-compiled kernels
# ---------------------------------------------------------------------------


@njit_safe
def cir_survival_probability_kernel(
    tenors_yrs: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    lam0: float,
) -> np.ndarray:
    """JIT-compiled CIR closed-form survival probability calculation.

    Args:
        tenors_yrs: 1-D array of tenor pillar points (years).
        kappa: Speed of mean-reversion.
        theta: Long-term mean hazard rate.
        sigma: Volatility of hazard rate.
        lam0: Initial hazard rate.

    Returns:
        1-D array of survival probabilities.
    """
    n = len(tenors_yrs)
    out = np.empty(n, dtype=np.float64)
    sig_sq = max(sigma * sigma, 1e-14)
    gamma = np.sqrt(kappa * kappa + 2.0 * sig_sq)
    power = (2.0 * kappa * theta) / sig_sq
    two_gamma = 2.0 * gamma
    kappa_plus_gamma = kappa + gamma

    for i in range(n):
        t = tenors_yrs[i]
        if np.isnan(t) or t != t:
            out[i] = np.nan
            continue
        if t <= 0.0:
            out[i] = 1.0
            continue
        exp_gamma_t = np.exp(gamma * t)
        denom = kappa_plus_gamma * (exp_gamma_t - 1.0) + two_gamma
        if denom <= 0.0 or np.isnan(denom) or denom != denom:
            out[i] = 0.0
            continue
        num_a = two_gamma * np.exp(kappa_plus_gamma * t * 0.5)
        a_t = (num_a / denom) ** power
        b_t = 2.0 * (exp_gamma_t - 1.0) / denom
        out[i] = a_t * np.exp(-b_t * lam0)

    return out


@njit_safe
def cir_calibration_objective_kernel(
    params_vec: np.ndarray,
    tenors_yrs: np.ndarray,
    credit_spreads_ann: np.ndarray,
) -> float:
    """JIT-compiled sum-of-squares objective function for CIR credit calibration.

    Args:
        params_vec: 1-D array ``[kappa, theta, sigma, lambda_0]``.
        tenors_yrs: 1-D array of tenors in years.
        credit_spreads_ann: 1-D array of market credit spreads.

    Returns:
        Sum of squared pricing errors.
    """
    kappa = params_vec[0]
    theta = params_vec[1]
    sigma = params_vec[2]
    lam0 = params_vec[3]

    if kappa <= 0.0 or theta <= 0.0 or sigma <= 0.0 or lam0 < 0.0:
        return 1e12

    surv_prob = cir_survival_probability_kernel(
        tenors_yrs, kappa, theta, sigma, lam0
    )
    n = len(tenors_yrs)
    sse = 0.0
    for i in range(n):
        t = tenors_yrs[i]
        if np.isnan(t) or np.isnan(credit_spreads_ann[i]) or np.isnan(surv_prob[i]):
            return 1e12
        p = max(surv_prob[i], 1e-15)
        t_pos = max(t, 1e-10)
        model_spread = -np.log(p) / t_pos
        diff = model_spread - credit_spreads_ann[i]
        sse += diff * diff

    return sse


@njit
def cir_simulate_paths_kernel(
    n_paths: int,
    n_steps: int,
    dt_vec: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    r0: float,
    dw_matrix: np.ndarray,
) -> np.ndarray:
    """JIT-compiled CIR short-rate Monte Carlo path stepping with full truncation.

    Args:
        n_paths: Number of simulation paths.
        n_steps: Number of discrete time steps.
        dt_vec: 1-D array of step increments (length n_steps).
        kappa: Mean reversion speed.
        theta: Long-term mean rate.
        sigma: Volatility parameter.
        r0: Initial short rate.
        dw_matrix: Brownian increments array of shape ``(n_paths, n_steps)``.

    Returns:
        Array of short rate paths of shape ``(n_paths, n_steps + 1)``.
    """
    r_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    r_paths[:, 0] = r0

    for step in range(n_steps):
        dt = dt_vec[step]
        for p in range(n_paths):
            r_curr = r_paths[p, step]
            r_pos = r_curr if r_curr > 0.0 else 0.0
            drift = kappa * (theta - r_pos) * dt
            diffusion = sigma * np.sqrt(r_pos) * dw_matrix[p, step]
            r_next = r_curr + drift + diffusion
            r_paths[p, step + 1] = r_next if r_next > 0.0 else 0.0

    return r_paths


@njit
def lgm_simulate_paths_kernel(
    n_paths: int,
    n_steps: int,
    dt_vec: np.ndarray,
    kappa: float,
    sigmas: np.ndarray,
    dw_matrix: np.ndarray,
) -> np.ndarray:
    """JIT-compiled Linear Gauss-Markov state path stepping.

    Args:
        n_paths: Number of simulation paths.
        n_steps: Number of discrete time steps.
        dt_vec: 1-D array of time steps.
        kappa: Mean reversion speed.
        sigmas: 1-D array of volatility values at each step.
        dw_matrix: Brownian increments array of shape ``(n_paths, n_steps)``.

    Returns:
        Array of state paths of shape ``(n_paths, n_steps + 1)``.
    """
    x_paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)

    for step in range(n_steps):
        dt = dt_vec[step]
        sig = sigmas[step]
        decay = 1.0 - kappa * dt
        for p in range(n_paths):
            x_paths[p, step + 1] = (
                x_paths[p, step] * decay + sig * dw_matrix[p, step]
            )

    return x_paths


@njit
def vasicek_simulate_paths_kernel(
    n_paths: int,
    n_steps: int,
    dt_vec: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    r0: float,
    dw_matrix: np.ndarray,
) -> np.ndarray:
    """JIT-compiled Vasicek short rate path stepping.

    Args:
        n_paths: Number of simulation paths.
        n_steps: Number of discrete time steps.
        dt_vec: 1-D array of time steps.
        kappa: Mean reversion speed.
        theta: Long term mean rate.
        sigma: Rate volatility.
        r0: Initial rate.
        dw_matrix: Brownian increments array of shape ``(n_paths, n_steps)``.

    Returns:
        Array of short rate paths of shape ``(n_paths, n_steps + 1)``.
    """
    r_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    r_paths[:, 0] = r0

    for step in range(n_steps):
        dt = dt_vec[step]
        for p in range(n_paths):
            r_curr = r_paths[p, step]
            drift = kappa * (theta - r_curr) * dt
            diffusion = sigma * dw_matrix[p, step]
            r_paths[p, step + 1] = r_curr + drift + diffusion

    return r_paths


@njit
def hull_white_simulate_paths_kernel(
    n_paths: int,
    n_steps: int,
    dt_vec: np.ndarray,
    a: float,
    sigma: float,
    theta_vec: np.ndarray,
    r0: float,
    dw_matrix: np.ndarray,
) -> np.ndarray:
    """JIT-compiled Hull-White 1F short rate path stepping.

    Args:
        n_paths: Number of simulation paths.
        n_steps: Number of discrete time steps.
        dt_vec: 1-D array of time steps.
        a: Mean reversion speed.
        sigma: Rate volatility.
        theta_vec: 1-D array of calibrated theta(t) values.
        r0: Initial rate.
        dw_matrix: Brownian increments array of shape ``(n_paths, n_steps)``.

    Returns:
        Array of short rate paths of shape ``(n_paths, n_steps + 1)``.
    """
    r_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    r_paths[:, 0] = r0

    for step in range(n_steps):
        dt = dt_vec[step]
        theta_t = theta_vec[step]
        for p in range(n_paths):
            r_curr = r_paths[p, step]
            drift = (theta_t - a * r_curr) * dt
            diffusion = sigma * dw_matrix[p, step]
            r_paths[p, step + 1] = r_curr + drift + diffusion

    return r_paths


@njit
def heston_simulate_paths_kernel(
    n_paths: int,
    n_steps: int,
    dt: float,
    sqrt_dt: float,
    v0: float,
    spot_fx: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    r_d_vec: np.ndarray,
    r_f_vec: np.ndarray,
    z_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """JIT-compiled Full Truncation Heston variance and spot FX simulation.

    Args:
        n_paths: Number of Monte Carlo paths.
        n_steps: Number of time steps.
        dt: Discrete time increment.
        sqrt_dt: Square root of dt.
        v0: Initial variance.
        spot_fx: Initial FX spot rate.
        kappa: Variance mean reversion speed.
        theta: Long term variance mean.
        sigma_v: Vol-of-vol.
        rho: Spot-variance correlation.
        r_d_vec: Forward domestic interest rate at each step.
        r_f_vec: Forward foreign interest rate at each step.
        z_all: Array of shape ``(n_paths, n_steps, 2)`` with independent normal draws.

    Returns:
        Tuple of ``(v_paths, fx_spot)`` arrays of shape ``(n_paths, n_steps + 1)``.
    """
    v_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    ln_fx = np.empty((n_paths, n_steps + 1), dtype=np.float64)

    v_paths[:, 0] = v0
    ln_fx[:, 0] = np.log(spot_fx)

    sqrt_one_minus_rho2 = np.sqrt(max(0.0, 1.0 - rho * rho))

    for step in range(n_steps):
        r_d = r_d_vec[step]
        r_f = r_f_vec[step]

        for p in range(n_paths):
            z1 = z_all[p, step, 0]
            z2 = z_all[p, step, 1]

            dw_v = z1 * sqrt_dt
            dw_s = (rho * z1 + sqrt_one_minus_rho2 * z2) * sqrt_dt

            v_curr = v_paths[p, step]
            v_pos = v_curr if v_curr > 0.0 else 0.0
            sqrt_v_pos = np.sqrt(v_pos)

            # Variance evolution (Lord et al. Full Truncation)
            v_next = (
                v_curr
                + kappa * (theta - v_pos) * dt
                + sigma_v * sqrt_v_pos * dw_v
            )
            v_paths[p, step + 1] = v_next

            # Spot evolution
            drift_s = (r_d - r_f) - 0.5 * v_pos
            ln_fx[p, step + 1] = ln_fx[p, step] + drift_s * dt + sqrt_v_pos * dw_s

    fx_spot = np.exp(ln_fx)
    return v_paths, fx_spot


@njit
def discount_path_kernel(
    times: np.ndarray,
    short_rates: np.ndarray,
) -> np.ndarray:
    """JIT-compiled trapezoidal integration for cumulative discount factors.

    Args:
        times: 1-D array of grid times of shape ``(n_steps + 1,)``.
        short_rates: 2-D array of short rates of shape ``(n_paths, n_steps + 1)``.

    Returns:
        2-D array of discount factors of shape ``(n_paths, n_steps + 1)``.
    """
    n_paths, n_times = short_rates.shape
    dfs = np.empty((n_paths, n_times), dtype=np.float64)
    dfs[:, 0] = 1.0

    cum_integral = np.zeros(n_paths, dtype=np.float64)

    for j in range(1, n_times):
        dt = times[j] - times[j - 1]
        half_dt = 0.5 * dt
        for p in range(n_paths):
            r_prev = short_rates[p, j - 1]
            r_curr = short_rates[p, j]
            cum_integral[p] += half_dt * (r_prev + r_curr)
            dfs[p, j] = np.exp(-cum_integral[p])

    return dfs
