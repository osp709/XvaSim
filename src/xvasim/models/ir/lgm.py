"""Linear Gauss-Markov (LGM) interest rate model implementation.

This module implements the 1-factor Linear Gauss-Markov model as an
:class:`~xvasim.models.base.InterestRateModel`. The model dynamics are
driven by a Gaussian state variable x(t) whose volatility function σ(t) is
piecewise constant and calibrated to swaption market data.

Public API
----------
- :class:`LGMModel` — object-oriented LGM interest rate model.
- :class:`LGMParams` — calibrated LGM parameters dataclass (re-exported).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import brentq

from ...jit import discount_path_kernel, lgm_simulate_paths_kernel
from ...qmc import RandomSequenceType, generate_brownian_increments
from ..base import InterestRateModel
from ..registry import ModelRegistry


@dataclasses.dataclass(frozen=True)
class LGMParams:
    """Calibrated parameters for a single-currency LGM model.

    Attributes:
        kappa_ann: Mean-reversion speed (annualised, e.g. 0.03).
        sigma_grid_yrs: 1-D array of *breakpoint* times (years) for the
            piecewise-constant volatility function. Must be sorted in
            ascending order with strictly positive entries.
        sigma_values_ann: 1-D array of piecewise-constant volatility values
            (annualised). ``sigma_values_ann[i]`` applies on the interval
            ``(sigma_grid_yrs[i-1], sigma_grid_yrs[i]]`` (with
            ``sigma_grid_yrs[-1] = 0`` implied for the first bucket).
            Must have the same length as *sigma_grid_yrs*.
        discount_curve_yrs: 1-D array of tenors (years) defining the
            risk-free discount curve. Must be sorted ascending and start
            at or above 0.
        discount_factors: 1-D array of discount factors corresponding to
            *discount_curve_yrs*. Same length as *discount_curve_yrs*.
    """

    kappa_ann: float
    sigma_grid_yrs: np.ndarray
    sigma_values_ann: np.ndarray
    discount_curve_yrs: np.ndarray
    discount_factors: np.ndarray


@ModelRegistry.register("interest_rate", "lgm")
@ModelRegistry.register("interest_rate", "linear_gauss_markov")
class LGMModel(InterestRateModel):
    """Linear Gauss-Markov (LGM) 1-factor interest rate model."""

    def __init__(
        self,
        params: LGMParams | None = None,
        *,
        kappa_ann: float = 0.03,
        sigma_grid_yrs: np.ndarray | None = None,
        sigma_values_ann: np.ndarray | None = None,
        discount_curve_yrs: np.ndarray | None = None,
        discount_factors: np.ndarray | None = None,
    ) -> None:
        """Initialize an LGM interest rate model.

        Can be initialized either from a pre-built :class:`LGMParams` instance
        or with individual keyword arguments.
        """
        if params is not None:
            self._params = params
        else:
            if (
                sigma_grid_yrs is None
                or sigma_values_ann is None
                or discount_curve_yrs is None
                or discount_factors is None
            ):
                msg = (
                    "Must supply either an LGMParams instance or all of "
                    "sigma_grid_yrs, sigma_values_ann, discount_curve_yrs, "
                    "discount_factors"
                )
                raise ValueError(msg)
            self._params = LGMParams(
                kappa_ann=kappa_ann,
                sigma_grid_yrs=np.asarray(sigma_grid_yrs, dtype=np.float64),
                sigma_values_ann=np.asarray(sigma_values_ann, dtype=np.float64),
                discount_curve_yrs=np.asarray(discount_curve_yrs, dtype=np.float64),
                discount_factors=np.asarray(discount_factors, dtype=np.float64),
            )

    @property
    def model_name(self) -> str:
        """Returns 'lgm'."""
        return "lgm"

    @property
    def params(self) -> LGMParams:
        """The underlying calibrated :class:`LGMParams`."""
        return self._params

    @property
    def kappa_ann(self) -> float:
        """Mean-reversion speed (annualised)."""
        return self._params.kappa_ann

    @property
    def sigma_grid_yrs(self) -> np.ndarray:
        """Volatility breakpoint grid (years)."""
        return self._params.sigma_grid_yrs

    @property
    def sigma_values_ann(self) -> np.ndarray:
        """Piecewise-constant volatility values."""
        return self._params.sigma_values_ann

    @property
    def discount_curve_yrs(self) -> np.ndarray:
        """Discount curve tenors in years."""
        return self._params.discount_curve_yrs

    @property
    def discount_factors(self) -> np.ndarray:
        """Discount factors along the curve."""
        return self._params.discount_factors

    def h_function(self, t: np.ndarray | float) -> np.ndarray:
        r"""Compute the LGM H-function: :math:`H(t) = (1 - e^{-\kappa t}) / \kappa`."""
        t_arr = np.asarray(t, dtype=np.float64)
        if abs(self.kappa_ann) < 1e-12:
            return t_arr.copy()
        return (1.0 - np.exp(-self.kappa_ann * t_arr)) / self.kappa_ann

    def zeta(self, t: float) -> float:
        r"""Compute the accumulated variance :math:`\zeta(t)`."""
        if t <= 0.0:
            return 0.0

        zeta_val = 0.0
        s_start = 0.0
        for i in range(len(self.sigma_grid_yrs)):
            s_end = min(float(self.sigma_grid_yrs[i]), t)
            if s_start >= t:
                break
            sig = float(self.sigma_values_ann[i])
            ds = s_end - s_start
            if ds <= 0.0:
                s_start = s_end
                continue

            if abs(self.kappa_ann) < 1e-12:
                zeta_val += sig * sig * ds
            else:
                zeta_val += (sig * sig / (2.0 * self.kappa_ann)) * (
                    np.exp(-2.0 * self.kappa_ann * (t - s_end))
                    - np.exp(-2.0 * self.kappa_ann * (t - s_start))
                )
            s_start = s_end

        return float(zeta_val)

    def sigma_at(self, t: float) -> float:
        """Evaluate the piecewise-constant volatility σ(t) at time t."""
        idx = min(
            int(np.searchsorted(self.sigma_grid_yrs, t, side="right")),
            len(self.sigma_values_ann) - 1,
        )
        return float(self.sigma_values_ann[idx])

    def short_rate(self, t: float, state: np.ndarray) -> np.ndarray:
        r"""Compute the instantaneous short rate r(t) given state variable x(t).

        .. math::
            r(t) = f(0, t) + e^{-\kappa t} x(t)
            + \frac{1}{2} e^{-2\kappa t} \zeta(t)
        """
        fwd = self.instantaneous_forward(t)
        h_prime = np.exp(-self.kappa_ann * t)
        zeta_val = self.zeta(t)
        state_arr = np.asarray(state, dtype=np.float64)
        res = fwd + h_prime * state_arr + 0.5 * (h_prime**2) * zeta_val
        return np.asarray(res, dtype=np.float64)

    def zero_coupon_bond(
        self,
        t: float,
        maturity_yrs: float,
        state: np.ndarray,
    ) -> np.ndarray:
        r"""Compute zero-coupon bond price P(t, T) given state variable x(t).

        .. math::
            P(t, T) = \frac{P(0, T)}{P(0, t)} \exp\left(
                -(H(T) - H(t)) x(t) - \frac{1}{2}(H(T)^2 - H(t)^2)\zeta(t)
            \right)
        """
        if maturity_yrs <= t:
            return np.ones_like(state, dtype=np.float64)

        p_0_t = float(self.interpolate_discount_factor(t))
        p_0_mat = float(self.interpolate_discount_factor(maturity_yrs))
        h_t = float(self.h_function(t))
        h_mat = float(self.h_function(maturity_yrs))
        zeta_t = self.zeta(t)

        delta_h = h_mat - h_t
        exponent = -delta_h * state - 0.5 * (h_mat**2 - h_t**2) * zeta_t
        return (p_0_mat / max(p_0_t, 1e-18)) * np.exp(exponent)

    def discount_path(
        self,
        times: np.ndarray,
        state_paths: np.ndarray,
    ) -> np.ndarray:
        """Compute path-wise discount factors D(0, t_i) using the bank account."""
        times_arr = np.asarray(times, dtype=np.float64)
        paths_arr = np.asarray(state_paths, dtype=np.float64)
        n_paths, n_times = paths_arr.shape
        short_rates = np.empty((n_paths, n_times), dtype=np.float64)
        for j in range(n_times):
            t = times_arr[j]
            short_rates[:, j] = self.short_rate(t, paths_arr[:, j])

        return discount_path_kernel(times_arr, short_rates)

    def simulate_paths(
        self,
        times: np.ndarray,
        n_paths: int,
        rng: np.random.Generator | None = None,
        dw: np.ndarray | None = None,
        random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
        seed: int | None = None,
        scramble: bool = True,
    ) -> np.ndarray:
        """Simulate state variable x(t) paths: dx = -kappa * x * dt + sigma(t) * dW."""
        times_arr = np.asarray(times, dtype=np.float64)
        n_steps = len(times_arr) - 1
        dt_vec = np.diff(times_arr)

        if dw is None:
            dw_matrix = generate_brownian_increments(
                n_paths=n_paths,
                dt_vec=dt_vec,
                num_factors=1,
                random_type=random_type,
                seed=seed,
                scramble=scramble,
                rng=rng,
            )
        else:
            dw_matrix = np.asarray(dw, dtype=np.float64)

        sigmas = np.array(
            [self.sigma_at(times_arr[s]) for s in range(n_steps)],
            dtype=np.float64,
        )
        return lgm_simulate_paths_kernel(
            n_paths=n_paths,
            n_steps=n_steps,
            dt_vec=dt_vec,
            kappa=self.kappa_ann,
            sigmas=sigmas,
            dw_matrix=dw_matrix,
        )

    @classmethod
    def calibrate_to_swaptions(
        cls,
        swaption_expiries_yrs: np.ndarray,
        swap_tenors_yrs: np.ndarray,
        market_normal_vols_ann: np.ndarray,
        curve_yrs: np.ndarray,
        curve_dfs: np.ndarray,
        fixed_rates_ann: np.ndarray,
        kappa_ann: float = 0.03,
        pay_freq_yrs: float = 0.5,
    ) -> LGMModel:
        """Calibrate an LGM model instance to swaption normal volatilities."""
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
                mdl_p, mkt_p = _lgm_swaption_price_normal_helper(
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
                return mdl_p - mkt_p

            try:
                sigma_vals[i] = brentq(_price_diff, 1e-6, 2.0, xtol=1e-12)
            except ValueError as exc:
                msg = (
                    f"LGM calibration failed at expiry "
                    f"{swaption_expiries_yrs[i]:.4f}y: {exc}"
                )
                raise RuntimeError(msg) from exc

        params = LGMParams(
            kappa_ann=kappa_ann,
            sigma_grid_yrs=sigma_grid,
            sigma_values_ann=sigma_vals,
            discount_curve_yrs=np.array(curve_yrs, dtype=np.float64),
            discount_factors=np.array(curve_dfs, dtype=np.float64),
        )
        return cls(params=params)


def _lgm_swaption_price_normal_helper(
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
    """Helper to compute model and market Bachelier swaption prices."""
    n_periods = max(1, round(swap_tenor_yrs / pay_freq_yrs))
    actual_freq = swap_tenor_yrs / n_periods
    payment_times = np.array(
        [expiry_yrs + actual_freq * (k + 1) for k in range(n_periods)]
    )

    # Log-linear discount factors
    log_dfs = np.log(np.maximum(curve_dfs, 1e-18))
    interp_log = np.interp(payment_times, curve_yrs, log_dfs)
    dfs = np.exp(interp_log)

    annuity = float(np.sum(actual_freq * dfs))

    # H-function at payments
    if abs(kappa) < 1e-12:
        h_vals = payment_times.copy()
        h_expiry = expiry_yrs
    else:
        h_vals = (1.0 - np.exp(-kappa * payment_times)) / kappa
        h_expiry = (1.0 - np.exp(-kappa * expiry_yrs)) / kappa

    delta_h = h_vals - h_expiry

    # zeta(T0)
    zeta = 0.0
    s_start = 0.0
    for i in range(len(sigma_grid_yrs)):
        s_end = min(float(sigma_grid_yrs[i]), expiry_yrs)
        if s_start >= expiry_yrs:
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
                np.exp(-2.0 * kappa * (expiry_yrs - s_end))
                - np.exp(-2.0 * kappa * (expiry_yrs - s_start))
            )
        s_start = s_end

    weighted_delta_h = float(np.sum(actual_freq * dfs * delta_h)) / annuity
    model_normal_var = zeta * weighted_delta_h**2
    model_normal_vol = np.sqrt(max(model_normal_var / expiry_yrs, 0.0))

    sqrt_t = np.sqrt(expiry_yrs)
    phi_0 = 1.0 / np.sqrt(2.0 * np.pi)

    model_price = annuity * model_normal_vol * sqrt_t * phi_0
    market_price = annuity * market_normal_vol_ann * sqrt_t * phi_0

    return float(model_price), float(market_price)
