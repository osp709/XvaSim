r"""Hull-White 1-Factor (HW1F) interest rate model implementation.

This module implements the classic 1-Factor Hull-White short-rate model:

.. math::
    dr(t) = (\theta(t) - a r(t))\,dt + \sigma\,dW(t)

with exact calibration to the initial discount curve :math:`P(0, t)`:

.. math::
    \theta(t) = \frac{\partial f(0, t)}{\partial t} + a f(0, t)
    + \frac{\sigma^2}{2a}\left(1 - e^{-2at}\right)

Public API
----------
- :class:`HullWhite1FParams` — Hull-White 1-Factor parameter dataclass.
- :class:`HullWhite1FModel` — object-oriented HW1F model.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ...jit import discount_path_kernel
from ...qmc import RandomSequenceType, generate_brownian_increments
from ..base import InterestRateModel
from ..registry import ModelRegistry


@dataclasses.dataclass(frozen=True)
class HullWhite1FParams:
    """Parameters for a 1-Factor Hull-White interest rate model.

    Attributes:
        a_ann: Mean-reversion speed (annualised decimal, e.g. 0.03).
        sigma_ann: Short-rate volatility (annualised decimal, e.g. 0.01 for 100 bp).
        discount_curve_yrs: 1-D array of tenors (years) defining the discount curve.
        discount_factors: 1-D array of discount factors corresponding to
            *discount_curve_yrs*.
    """

    a_ann: float
    sigma_ann: float
    discount_curve_yrs: np.ndarray
    discount_factors: np.ndarray


@ModelRegistry.register("interest_rate", "hull_white")
@ModelRegistry.register("interest_rate", "hull_white_1f")
@ModelRegistry.register("interest_rate", "hw1f")
class HullWhite1FModel(InterestRateModel):
    """Hull-White 1-Factor (HW1F) interest rate model."""

    def __init__(
        self,
        params: HullWhite1FParams | None = None,
        *,
        a_ann: float = 0.03,
        sigma_ann: float = 0.01,
        discount_curve_yrs: np.ndarray | None = None,
        discount_factors: np.ndarray | None = None,
    ) -> None:
        """Initialize a Hull-White 1-Factor interest rate model."""
        if params is not None:
            self._params = params
        else:
            if discount_curve_yrs is None or discount_factors is None:
                msg = (
                    "Must supply either a HullWhite1FParams instance or both "
                    "discount_curve_yrs and discount_factors"
                )
                raise ValueError(msg)
            self._params = HullWhite1FParams(
                a_ann=a_ann,
                sigma_ann=sigma_ann,
                discount_curve_yrs=np.asarray(discount_curve_yrs, dtype=np.float64),
                discount_factors=np.asarray(discount_factors, dtype=np.float64),
            )

    @property
    def model_name(self) -> str:
        """Returns 'hull_white'."""
        return "hull_white"

    @property
    def params(self) -> HullWhite1FParams:
        """The underlying :class:`HullWhite1FParams`."""
        return self._params

    @property
    def a_ann(self) -> float:
        """Mean-reversion speed a."""
        return self._params.a_ann

    @property
    def sigma_ann(self) -> float:
        """Short-rate volatility σ."""
        return self._params.sigma_ann

    @property
    def discount_curve_yrs(self) -> np.ndarray:
        """Discount curve tenors in years."""
        return self._params.discount_curve_yrs

    @property
    def discount_factors(self) -> np.ndarray:
        """Discount factors along the curve."""
        return self._params.discount_factors

    def b_function(self, t: float, maturity_yrs: float) -> float:
        r"""Compute the Hull-White B(t, T) function.

        .. math::
            B(t, T) = \frac{1 - e^{-a(T-t)}}{a}
        """
        tau = max(maturity_yrs - t, 0.0)
        if abs(self.a_ann) < 1e-12:
            return float(tau)
        return float((1.0 - np.exp(-self.a_ann * tau)) / self.a_ann)

    def alpha(self, t: float) -> float:
        r"""Compute the deterministic drift correction :math:`\alpha(t)`.

        .. math::
            \alpha(t) = f(0, t) + \frac{\sigma^2}{2a^2}(1 - e^{-at})^2
        """
        fwd = self.instantaneous_forward(t)
        if abs(self.a_ann) < 1e-12:
            return float(fwd + 0.5 * (self.sigma_ann * t) ** 2)
        exp_at = np.exp(-self.a_ann * t)
        correction = (self.sigma_ann**2 / (2.0 * self.a_ann**2)) * ((1.0 - exp_at) ** 2)
        return float(fwd + correction)

    def short_rate(self, t: float, state: np.ndarray) -> np.ndarray:
        r"""Compute the instantaneous short rate :math:`r(t) = x(t) + \alpha(t)`."""
        state_arr = np.asarray(state, dtype=np.float64)
        return state_arr + self.alpha(t)

    def zero_coupon_bond(
        self,
        t: float,
        maturity_yrs: float,
        state: np.ndarray,
    ) -> np.ndarray:
        r"""Compute zero-coupon bond price P(t, T) in the Hull-White model.

        .. math::
            P(t, T) = A(t, T)\,\exp\bigl(-B(t, T)\,r(t)\bigr)
        """
        if maturity_yrs <= t:
            return np.ones_like(state, dtype=np.float64)

        b_t_t = self.b_function(t, maturity_yrs)
        p_0_t = float(self.interpolate_discount_factor(t))
        p_0_t_mat = float(self.interpolate_discount_factor(maturity_yrs))
        fwd_0_t = self.instantaneous_forward(t)

        a = self.a_ann
        sig = self.sigma_ann
        if abs(a) < 1e-12:
            var_term = 0.5 * sig**2 * t * b_t_t**2
        else:
            var_term = (sig**2 / (4.0 * a)) * (1.0 - np.exp(-2.0 * a * t)) * (b_t_t**2)

        ln_a_t_t = np.log(p_0_t_mat / max(p_0_t, 1e-18)) + b_t_t * fwd_0_t - var_term
        a_t_t = np.exp(ln_a_t_t)

        r_t = self.short_rate(t, state)
        return a_t_t * np.exp(-b_t_t * r_t)  # type: ignore[no-any-return]

    def discount_path(
        self,
        times: np.ndarray,
        state_paths: np.ndarray,
    ) -> np.ndarray:
        """Compute path-wise discount factors D(0, t_i) using simulated short rates."""
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
        """Simulate Hull-White state variable x(t) paths: dx = -a x dt + sigma dW."""
        times_arr = np.asarray(times, dtype=np.float64)
        n_steps = len(times_arr) - 1
        dt_vec = np.diff(times_arr)
        x_paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)

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

        for step in range(n_steps):
            dt = dt_vec[step]
            # Exact Ornstein-Uhlenbeck transition:
            if abs(self.a_ann) < 1e-12:
                decay = 1.0
            else:
                decay = np.exp(-self.a_ann * dt)
            x_paths[:, step + 1] = (
                decay * x_paths[:, step] + self.sigma_ann * dw_matrix[:, step]
            )

        return x_paths
