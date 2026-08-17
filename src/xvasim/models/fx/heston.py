"""Heston stochastic volatility model for foreign exchange (FX) simulation.

This module implements the Heston (1993) stochastic volatility model tailored
for cross-currency FX spot dynamics under the domestic risk-neutral measure.

Public API
----------
- :class:`HestonFXParams` — parameter container for the Heston FX model.
- :class:`HestonFXModel` — Heston FX stochastic volatility market model class.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.integrate import quad

from ...jit import heston_simulate_paths_kernel
from ...qmc import RandomSequenceType, generate_normal_draws
from ..base import FXModel
from ..registry import ModelRegistry


@dataclasses.dataclass(frozen=True)
class HestonFXParams:
    """Parameters for the Heston stochastic volatility FX model.

    Attributes:
        spot_fx: Current spot FX rate (units of domestic per 1 foreign).
        v_0: Initial variance of the FX rate process (v_0 >= 0).
        kappa_ann: Mean-reversion speed of the variance process (kappa > 0).
        theta_ann: Long-term mean variance (theta > 0).
        sigma_v_ann: Volatility of the variance process (vol-of-vol, sigma_v >= 0).
        rho: Correlation between spot FX and variance Brownian motions (-1 <= rho <= 1).
        domestic_rate_ann: Constant annualised domestic risk-free rate.
        foreign_rate_ann: Constant annualised foreign risk-free rate.
        discount_curve_domestic_yrs: Optional domestic curve tenors.
        discount_factors_domestic: Optional domestic discount factors.
        discount_curve_foreign_yrs: Optional foreign curve tenors.
        discount_factors_foreign: Optional foreign discount factors.
    """

    spot_fx: float
    v_0: float
    kappa_ann: float
    theta_ann: float
    sigma_v_ann: float
    rho: float
    domestic_rate_ann: float = 0.0
    foreign_rate_ann: float = 0.0
    discount_curve_domestic_yrs: np.ndarray | None = None
    discount_factors_domestic: np.ndarray | None = None
    discount_curve_foreign_yrs: np.ndarray | None = None
    discount_factors_foreign: np.ndarray | None = None


@ModelRegistry.register("fx", "heston")
@ModelRegistry.register("fx", "heston_fx")
class HestonFXModel(FXModel):
    """Heston stochastic volatility FX market model.

    Under the domestic risk-neutral measure, the spot FX and instantaneous
    variance dynamics are:

    .. math::
        \\frac{dS(t)}{S(t)} = (r_d(t) - r_f(t))\\,dt + \\sqrt{v(t)}\\,dW_S(t)

    .. math::
        dv(t) = \\kappa (\\theta - v(t))\\,dt + \\sigma_v \\sqrt{v(t)}\\,dW_v(t)

    where :math:`d\\langle W_S, W_v \\rangle_t = \\rho\\,dt`.
    """

    def __init__(
        self,
        spot_fx: float,
        v_0: float,
        kappa_ann: float,
        theta_ann: float,
        sigma_v_ann: float,
        rho: float,
        domestic_rate_ann: float = 0.0,
        foreign_rate_ann: float = 0.0,
        discount_curve_domestic_yrs: np.ndarray | None = None,
        discount_factors_domestic: np.ndarray | None = None,
        discount_curve_foreign_yrs: np.ndarray | None = None,
        discount_factors_foreign: np.ndarray | None = None,
    ) -> None:
        """Initialize the Heston FX stochastic volatility model.

        Args:
            spot_fx: Current spot FX rate (units of domestic per 1 foreign).
            v_0: Initial variance (v_0 >= 0).
            kappa_ann: Mean-reversion speed of variance (kappa > 0).
            theta_ann: Long-term variance mean (theta > 0).
            sigma_v_ann: Volatility of variance (vol-of-vol >= 0).
            rho: Correlation between spot and variance Brownian motions
                (-1 <= rho <= 1).
            domestic_rate_ann: Constant domestic interest rate (annualised).
            foreign_rate_ann: Constant foreign interest rate (annualised).
            discount_curve_domestic_yrs: Optional domestic curve tenors.
            discount_factors_domestic: Optional domestic discount factors.
            discount_curve_foreign_yrs: Optional foreign curve tenors.
            discount_factors_foreign: Optional foreign discount factors.
        """
        if spot_fx <= 0.0:
            msg = f"spot_fx must be strictly positive, got {spot_fx}"
            raise ValueError(msg)
        if v_0 < 0.0:
            msg = f"v_0 must be non-negative, got {v_0}"
            raise ValueError(msg)
        if kappa_ann <= 0.0:
            msg = f"kappa_ann must be strictly positive, got {kappa_ann}"
            raise ValueError(msg)
        if theta_ann < 0.0:
            msg = f"theta_ann must be non-negative, got {theta_ann}"
            raise ValueError(msg)
        if sigma_v_ann < 0.0:
            msg = f"sigma_v_ann must be non-negative, got {sigma_v_ann}"
            raise ValueError(msg)
        if not (-1.0 <= rho <= 1.0):
            msg = f"rho must be in [-1, 1], got {rho}"
            raise ValueError(msg)

        self._spot_fx = float(spot_fx)
        self._v_0 = float(v_0)
        self._kappa_ann = float(kappa_ann)
        self._theta_ann = float(theta_ann)
        self._sigma_v_ann = float(sigma_v_ann)
        self._rho = float(rho)
        self._domestic_rate_ann = float(domestic_rate_ann)
        self._foreign_rate_ann = float(foreign_rate_ann)

        if discount_curve_domestic_yrs is not None:
            self._dom_curve_yrs: np.ndarray | None = np.asarray(
                discount_curve_domestic_yrs, dtype=np.float64
            )
            self._dom_dfs: np.ndarray | None = np.asarray(
                discount_factors_domestic, dtype=np.float64
            )
        else:
            self._dom_curve_yrs = None
            self._dom_dfs = None

        if discount_curve_foreign_yrs is not None:
            self._for_curve_yrs: np.ndarray | None = np.asarray(
                discount_curve_foreign_yrs, dtype=np.float64
            )
            self._for_dfs: np.ndarray | None = np.asarray(
                discount_factors_foreign, dtype=np.float64
            )
        else:
            self._for_curve_yrs = None
            self._for_dfs = None

    @classmethod
    def from_params(cls, params: HestonFXParams) -> HestonFXModel:
        """Construct a HestonFXModel from a parameters object."""
        return cls(
            spot_fx=params.spot_fx,
            v_0=params.v_0,
            kappa_ann=params.kappa_ann,
            theta_ann=params.theta_ann,
            sigma_v_ann=params.sigma_v_ann,
            rho=params.rho,
            domestic_rate_ann=params.domestic_rate_ann,
            foreign_rate_ann=params.foreign_rate_ann,
            discount_curve_domestic_yrs=params.discount_curve_domestic_yrs,
            discount_factors_domestic=params.discount_factors_domestic,
            discount_curve_foreign_yrs=params.discount_curve_foreign_yrs,
            discount_factors_foreign=params.discount_factors_foreign,
        )

    @property
    def model_name(self) -> str:
        """Returns 'heston'."""
        return "heston"

    @property
    def num_factors(self) -> int:
        """Returns 2 (spot Brownian factor and variance Brownian factor)."""
        return 2

    @property
    def spot_fx(self) -> float:
        """Current spot FX rate."""
        return self._spot_fx

    @property
    def v_0(self) -> float:
        """Initial variance."""
        return self._v_0

    @property
    def kappa_ann(self) -> float:
        """Mean-reversion speed of variance."""
        return self._kappa_ann

    @property
    def theta_ann(self) -> float:
        """Long-term mean variance."""
        return self._theta_ann

    @property
    def sigma_v_ann(self) -> float:
        """Volatility of variance (vol-of-vol)."""
        return self._sigma_v_ann

    @property
    def rho(self) -> float:
        """Correlation between spot and variance innovations."""
        return self._rho

    @property
    def domestic_rate_ann(self) -> float:
        """Constant annualised domestic interest rate."""
        return self._domestic_rate_ann

    @property
    def foreign_rate_ann(self) -> float:
        """Constant annualised foreign interest rate."""
        return self._foreign_rate_ann

    @property
    def is_feller_satisfied(self) -> bool:
        """Check whether the Feller condition 2*kappa*theta > sigma_v^2 holds.

        When satisfied, the variance process is strictly positive almost surely.
        """
        return 2.0 * self._kappa_ann * self._theta_ann > self._sigma_v_ann**2

    def domestic_discount_factor(self, t: float | np.ndarray) -> np.ndarray:
        """Evaluate the domestic discount factor P_d(0, t).

        Args:
            t: Tenor(s) in years.

        Returns:
            Discount factor(s).
        """
        t_arr = np.asarray(t, dtype=np.float64)
        if self._dom_curve_yrs is not None and self._dom_dfs is not None:
            log_dfs = np.log(np.maximum(self._dom_dfs, 1e-18))
            interp_log = np.interp(t_arr, self._dom_curve_yrs, log_dfs)
            return np.asarray(np.exp(interp_log), dtype=np.float64)
        return np.asarray(np.exp(-self._domestic_rate_ann * t_arr), dtype=np.float64)

    def foreign_discount_factor(self, t: float | np.ndarray) -> np.ndarray:
        """Evaluate the foreign discount factor P_f(0, t).

        Args:
            t: Tenor(s) in years.

        Returns:
            Discount factor(s).
        """
        t_arr = np.asarray(t, dtype=np.float64)
        if self._for_curve_yrs is not None and self._for_dfs is not None:
            log_dfs = np.log(np.maximum(self._for_dfs, 1e-18))
            interp_log = np.interp(t_arr, self._for_curve_yrs, log_dfs)
            return np.asarray(np.exp(interp_log), dtype=np.float64)
        return np.asarray(np.exp(-self._foreign_rate_ann * t_arr), dtype=np.float64)

    def forward_rate(self, maturity_yrs: float) -> float:
        """Compute the theoretical FX forward rate F(0, T).

        Args:
            maturity_yrs: Forward maturity in years.

        Returns:
            FX forward rate.
        """
        df_d = float(self.domestic_discount_factor(maturity_yrs))
        df_f = float(self.foreign_discount_factor(maturity_yrs))
        return float(self._spot_fx * df_f / max(df_d, 1e-18))

    def _characteristic_function(
        self,
        u: float,
        tau: float,
        j: int,
    ) -> complex:
        """Evaluate the Heston characteristic function f_j(u) using Albrecher et al.

        Args:
            u: Integration frequency variable.
            tau: Time to expiry in years.
            j: 1 for asset-measure characteristic function, 2 for domestic measure.

        Returns:
            Complex characteristic value.
        """
        kappa = self._kappa_ann
        theta = self._theta_ann
        sigma_v = max(self._sigma_v_ann, 1e-7)
        rho = self._rho
        v0 = self._v_0
        s0 = self._spot_fx

        df_d = float(self.domestic_discount_factor(tau))
        df_f = float(self.foreign_discount_factor(tau))
        r_d = -np.log(max(df_d, 1e-18)) / tau
        r_f = -np.log(max(df_f, 1e-18)) / tau

        if j == 1:
            u_j = 0.5
            b_j = kappa - rho * sigma_v
        else:
            u_j = -0.5
            b_j = kappa

        i = 1j
        d = np.sqrt(
            (rho * sigma_v * u * i - b_j) ** 2 - sigma_v**2 * (2 * u_j * u * i - u**2)
        )
        g = (b_j - rho * sigma_v * u * i - d) / (b_j - rho * sigma_v * u * i + d)

        # Little Heston Trap stable formulation
        c = (r_d - r_f) * u * i * tau + (kappa * theta / (sigma_v**2)) * (
            (b_j - rho * sigma_v * u * i - d) * tau
            - 2.0 * np.log((1.0 - g * np.exp(-d * tau)) / (1.0 - g))
        )
        d_val = (
            (b_j - rho * sigma_v * u * i - d)
            / (sigma_v**2)
            * ((1.0 - np.exp(-d * tau)) / (1.0 - g * np.exp(-d * tau)))
        )

        val: complex = complex(np.exp(c + d_val * v0 + i * u * np.log(s0)))
        return val

    def closed_form_option_price(
        self,
        strike: float,
        maturity_yrs: float,
        option_type: str = "call",
        notional: float = 1.0,
    ) -> float:
        """Compute European option price via Heston semi-analytical integration.

        Args:
            strike: Option strike (domestic per foreign).
            maturity_yrs: Option maturity in years.
            option_type: 'call' or 'put'.
            notional: Option notional in foreign currency units.

        Returns:
            Option premium in domestic currency.
        """
        if strike <= 0.0:
            msg = f"strike must be strictly positive, got {strike}"
            raise ValueError(msg)
        if maturity_yrs <= 0.0:
            msg = f"maturity_yrs must be strictly positive, got {maturity_yrs}"
            raise ValueError(msg)

        ln_k = np.log(strike)
        tau = maturity_yrs

        def integrand_1(u: float) -> float:
            if u < 1e-12:
                return 0.0
            cf = self._characteristic_function(u, tau, j=1)
            val = (np.exp(-1j * u * ln_k) * cf) / (1j * u)
            return float(val.real)

        def integrand_2(u: float) -> float:
            if u < 1e-12:
                return 0.0
            cf = self._characteristic_function(u, tau, j=2)
            val = (np.exp(-1j * u * ln_k) * cf) / (1j * u)
            return float(val.real)

        int_1, _ = quad(integrand_1, 0.0, 100.0, limit=100)
        int_2, _ = quad(integrand_2, 0.0, 100.0, limit=100)

        p1 = 0.5 + (1.0 / np.pi) * int_1
        p2 = 0.5 + (1.0 / np.pi) * int_2

        # Bound probabilities
        p1 = max(0.0, min(1.0, p1))
        p2 = max(0.0, min(1.0, p2))

        df_d = float(self.domestic_discount_factor(tau))
        df_f = float(self.foreign_discount_factor(tau))

        call_price = self._spot_fx * df_f * p1 - strike * df_d * p2
        call_price = max(0.0, call_price)

        is_call = option_type.strip().lower() == "call"
        if is_call:
            return float(notional * call_price)

        # Put via Put-Call Parity
        put_price = call_price - self._spot_fx * df_f + strike * df_d
        return float(notional * max(0.0, put_price))

    def simulate_paths(
        self,
        maturity_yrs: float,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator | None = None,
        random_type: RandomSequenceType | str = RandomSequenceType.PSEUDO,
        seed: int | None = None,
        scramble: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate joint variance and spot FX paths under the domestic measure.

        Employs the Full Truncation Euler-Maruyama scheme (Lord et al. 2010).

        Args:
            maturity_yrs: Simulation horizon in years.
            n_paths: Number of Monte Carlo paths.
            n_steps: Number of time steps.
            rng: Optional NumPy random Generator.
            random_type: Random sequence generator type (:class:`RandomSequenceType`
                or str).
            seed: Optional random seed.
            scramble: If True, scrambles QMC sequences.

        Returns:
            ``(times, v_paths, x_for, fx_spot)`` — simulated time grid, variance
            paths, dummy state array, and FX spot paths.
        """
        dt = maturity_yrs / n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, maturity_yrs, n_steps + 1)
        x_dummy = np.zeros((n_paths, n_steps + 1), dtype=np.float64)

        z_all = generate_normal_draws(
            n_paths=n_paths,
            dimension=2 * n_steps,
            random_type=random_type,
            seed=seed,
            scramble=scramble,
            rng=rng,
        ).reshape(n_paths, n_steps, 2)

        # Precompute forward rates for each step
        r_d_vec = np.empty(n_steps, dtype=np.float64)
        r_f_vec = np.empty(n_steps, dtype=np.float64)
        for step in range(n_steps):
            t = times[step]
            t_next = times[step + 1]
            df_d_t = float(self.domestic_discount_factor(t))
            df_d_next = float(self.domestic_discount_factor(t_next))
            df_f_t = float(self.foreign_discount_factor(t))
            df_f_next = float(self.foreign_discount_factor(t_next))
            r_d_vec[step] = -np.log(df_d_next / max(df_d_t, 1e-18)) / dt
            r_f_vec[step] = -np.log(df_f_next / max(df_f_t, 1e-18)) / dt

        v_paths, fx_spot = heston_simulate_paths_kernel(
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            sqrt_dt=sqrt_dt,
            v0=self._v_0,
            spot_fx=self._spot_fx,
            kappa=self._kappa_ann,
            theta=self._theta_ann,
            sigma_v=self._sigma_v_ann,
            rho=self._rho,
            r_d_vec=r_d_vec,
            r_f_vec=r_f_vec,
            z_all=z_all,
        )

        return times, v_paths, x_dummy, fx_spot
