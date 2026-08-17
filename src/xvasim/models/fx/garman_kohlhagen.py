"""Garman-Kohlhagen (Black-Scholes) foreign exchange market model.

This module implements the classic Garman-Kohlhagen (1983) model for foreign
exchange spot dynamics with continuous domestic and foreign interest rates
(or discount curves) and constant log-normal FX volatility.

Public API
----------
- :class:`GarmanKohlhagenParams` — parameter container for Garman-Kohlhagen model.
- :class:`GarmanKohlhagenFXModel` — Garman-Kohlhagen FX market model class.
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
from scipy.stats import norm

from ...qmc import RandomSequenceType, generate_brownian_increments
from ..base import FXModel
from ..registry import ModelRegistry


@dataclasses.dataclass(frozen=True)
class GarmanKohlhagenParams:
    """Parameters for the Garman-Kohlhagen FX model.

    Attributes:
        spot_fx: Current spot FX rate (units of domestic per 1 foreign).
        fx_vol_ann: Annualised FX volatility (e.g. 0.10 for 10%).
        domestic_rate_ann: Constant annualised domestic risk-free rate.
        foreign_rate_ann: Constant annualised foreign risk-free rate.
        discount_curve_domestic_yrs: Optional domestic curve tenors.
        discount_factors_domestic: Optional domestic discount factors.
        discount_curve_foreign_yrs: Optional foreign curve tenors.
        discount_factors_foreign: Optional foreign discount factors.
    """

    spot_fx: float
    fx_vol_ann: float
    domestic_rate_ann: float = 0.0
    foreign_rate_ann: float = 0.0
    discount_curve_domestic_yrs: np.ndarray | None = None
    discount_factors_domestic: np.ndarray | None = None
    discount_curve_foreign_yrs: np.ndarray | None = None
    discount_factors_foreign: np.ndarray | None = None


@ModelRegistry.register("fx", "garman_kohlhagen")
@ModelRegistry.register("fx", "black_scholes")
@ModelRegistry.register("fx", "gbm")
class GarmanKohlhagenFXModel(FXModel):
    """Garman-Kohlhagen (Black-Scholes) FX market model.

    Under the domestic risk-neutral measure, the spot FX rate follows:

    .. math::
        \\frac{dS(t)}{S(t)} = (r_d(t) - r_f(t))\\,dt + \\sigma_{\\text{fx}}\\,dW(t)
    """

    def __init__(
        self,
        spot_fx: float,
        fx_vol_ann: float,
        domestic_rate_ann: float = 0.0,
        foreign_rate_ann: float = 0.0,
        discount_curve_domestic_yrs: np.ndarray | None = None,
        discount_factors_domestic: np.ndarray | None = None,
        discount_curve_foreign_yrs: np.ndarray | None = None,
        discount_factors_foreign: np.ndarray | None = None,
    ) -> None:
        """Initialize the Garman-Kohlhagen FX model.

        Args:
            spot_fx: Current spot FX rate (units of domestic per 1 foreign).
            fx_vol_ann: Annualised log-normal FX volatility.
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
        if fx_vol_ann < 0.0:
            msg = f"fx_vol_ann must be non-negative, got {fx_vol_ann}"
            raise ValueError(msg)

        self._spot_fx = float(spot_fx)
        self._fx_vol_ann = float(fx_vol_ann)
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
    def from_params(cls, params: GarmanKohlhagenParams) -> GarmanKohlhagenFXModel:
        """Construct a GarmanKohlhagenFXModel from a parameters object."""
        return cls(
            spot_fx=params.spot_fx,
            fx_vol_ann=params.fx_vol_ann,
            domestic_rate_ann=params.domestic_rate_ann,
            foreign_rate_ann=params.foreign_rate_ann,
            discount_curve_domestic_yrs=params.discount_curve_domestic_yrs,
            discount_factors_domestic=params.discount_factors_domestic,
            discount_curve_foreign_yrs=params.discount_curve_foreign_yrs,
            discount_factors_foreign=params.discount_factors_foreign,
        )

    @property
    def model_name(self) -> str:
        """Returns 'garman_kohlhagen'."""
        return "garman_kohlhagen"

    @property
    def spot_fx(self) -> float:
        """Current spot FX rate."""
        return self._spot_fx

    @property
    def fx_vol_ann(self) -> float:
        """Annualised FX volatility."""
        return self._fx_vol_ann

    @property
    def domestic_rate_ann(self) -> float:
        """Constant annualised domestic interest rate."""
        return self._domestic_rate_ann

    @property
    def foreign_rate_ann(self) -> float:
        """Constant annualised foreign interest rate."""
        return self._foreign_rate_ann

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

        .. math::
            F(0, T) = S_0 \\frac{P_f(0, T)}{P_d(0, T)}

        Args:
            maturity_yrs: Forward maturity in years.

        Returns:
            FX forward rate.
        """
        df_d = float(self.domestic_discount_factor(maturity_yrs))
        df_f = float(self.foreign_discount_factor(maturity_yrs))
        return float(self._spot_fx * df_f / max(df_d, 1e-18))

    def closed_form_option_price(
        self,
        strike: float,
        maturity_yrs: float,
        option_type: typing.Any = "call",
        notional: float = 1.0,
    ) -> float:
        r"""Compute the closed-form Garman-Kohlhagen European option price.

        .. math::
            d_1 = \frac{\ln(S_0 / K) + (r_d - r_f + \sigma^2/2) T}{\sigma \sqrt{T}}
            d_2 = d_1 - \sigma \sqrt{T}
            C = e^{-r_d T} \left[ F N(d_1) - K N(d_2) \right]
            P = e^{-r_d T} \left[ K N(-d_2) - F N(-d_1) \right]

        Args:
            strike: Option strike (domestic per foreign).
            maturity_yrs: Option maturity in years.
            option_type: 'call' or 'put' or OptionType enum.
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

        if isinstance(option_type, str):
            is_call = option_type.strip().lower() == "call"
        else:
            opt_val = getattr(option_type, "value", str(option_type)).lower()
            is_call = opt_val.endswith("call")
        df_d = float(self.domestic_discount_factor(maturity_yrs))
        fwd = self.forward_rate(maturity_yrs)

        vol = self._fx_vol_ann
        sigma_sqrt_t = vol * np.sqrt(maturity_yrs)

        if sigma_sqrt_t < 1e-12:
            intrinsic = max(fwd - strike, 0.0) if is_call else max(strike - fwd, 0.0)
            return float(notional * df_d * intrinsic)

        d1 = (np.log(fwd / strike) + 0.5 * vol**2 * maturity_yrs) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t

        if is_call:
            price = df_d * (fwd * norm.cdf(d1) - strike * norm.cdf(d2))
        else:
            price = df_d * (strike * norm.cdf(-d2) - fwd * norm.cdf(-d1))

        return float(notional * price)

    # Convenience alias for backwards compatibility & uniform interface
    price_option_analytical = closed_form_option_price

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
        """Simulate joint state and FX spot paths under the domestic measure.

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
            ``(times, x_dom, x_for, fx_spot)`` — simulated time grid and paths.
        """
        dt = maturity_yrs / n_steps
        times = np.linspace(0.0, maturity_yrs, n_steps + 1)
        dt_vec = np.diff(times)

        x_dom = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        x_for = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        ln_fx = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        ln_fx[:, 0] = np.log(self._spot_fx)

        vol = self._fx_vol_ann

        dw_matrix = generate_brownian_increments(
            n_paths=n_paths,
            dt_vec=dt_vec,
            num_factors=1,
            random_type=random_type,
            seed=seed,
            scramble=scramble,
            rng=rng,
        )

        for step in range(n_steps):
            t = times[step]
            t_next = times[step + 1]

            df_d_t = float(self.domestic_discount_factor(t))
            df_d_next = float(self.domestic_discount_factor(t_next))
            df_f_t = float(self.foreign_discount_factor(t))
            df_f_next = float(self.foreign_discount_factor(t_next))

            r_d = -np.log(df_d_next / max(df_d_t, 1e-18)) / dt
            r_f = -np.log(df_f_next / max(df_f_t, 1e-18)) / dt

            drift = (r_d - r_f) - 0.5 * vol**2
            dw = dw_matrix[:, step]
            ln_fx[:, step + 1] = ln_fx[:, step] + drift * dt + vol * dw

        fx_spot = np.exp(ln_fx)
        return times, x_dom, x_for, fx_spot
