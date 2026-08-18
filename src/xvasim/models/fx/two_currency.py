"""Two-currency FX market model with modular interest rate dynamics.

This module implements a cross-currency market model driven by two modular
:class:`~xvasim.models.base.InterestRateModel` instances (domestic and foreign)
and a log-normal spot FX process with full 3×3 correlation structure.

Public API
----------
- :class:`TwoCurrencyFXModel` — modular two-currency FX model.
"""

from __future__ import annotations

import numpy as np

from ...qmc import RandomSequenceType, generate_normal_draws
from ..base import FXModel, InterestRateModel
from ..ir.lgm import LGMModel, LGMParams
from ..registry import ModelRegistry


@ModelRegistry.register("fx", "two_currency")
@ModelRegistry.register("fx", "cross_currency")
class TwoCurrencyFXModel(FXModel):
    """Two-currency FX market model with pluggable interest rate models."""

    def __init__(
        self,
        domestic_ir_model: InterestRateModel | LGMParams,
        foreign_ir_model: InterestRateModel | LGMParams,
        spot_fx: float,
        fx_vol_ann: float,
        correlation_matrix: np.ndarray,
    ) -> None:
        """Initialize a two-currency FX model.

        Args:
            domestic_ir_model: Interest rate model for domestic (numeraire) currency.
            foreign_ir_model: Interest rate model for foreign currency.
            spot_fx: Current spot FX rate (units of domestic per 1 foreign).
            fx_vol_ann: Annualised log-normal FX spot volatility.
            correlation_matrix: 3×3 correlation matrix ordered as
                ``[domestic_rate, foreign_rate, fx_spot]``.
        """
        if isinstance(domestic_ir_model, LGMParams):
            self._domestic: InterestRateModel = LGMModel(params=domestic_ir_model)
        else:
            self._domestic = domestic_ir_model

        if isinstance(foreign_ir_model, LGMParams):
            self._foreign: InterestRateModel = LGMModel(params=foreign_ir_model)
        else:
            self._foreign = foreign_ir_model

        self._spot_fx = float(spot_fx)
        self._fx_vol_ann = float(fx_vol_ann)
        self._correlation_matrix = np.asarray(correlation_matrix, dtype=np.float64)

        if self._correlation_matrix.shape != (3, 3):
            msg = (
                f"correlation_matrix must be 3x3, got {self._correlation_matrix.shape}"
            )
            raise ValueError(msg)

    @property
    def model_name(self) -> str:
        """Returns 'two_currency'."""
        return "two_currency"

    @property
    def domestic_ir_model(self) -> InterestRateModel:
        """The domestic interest rate model."""
        return self._domestic

    @property
    def foreign_ir_model(self) -> InterestRateModel:
        """The foreign interest rate model."""
        return self._foreign

    @property
    def spot_fx(self) -> float:
        """Current spot FX rate."""
        return self._spot_fx

    @property
    def fx_vol_ann(self) -> float:
        """FX volatility."""
        return self._fx_vol_ann

    @property
    def correlation_matrix(self) -> np.ndarray:
        """3×3 correlation matrix."""
        return self._correlation_matrix

    @classmethod
    def from_ir_models(
        cls,
        domestic: InterestRateModel | LGMParams,
        foreign: InterestRateModel | LGMParams,
        spot_fx: float,
        fx_vol_ann: float,
        correlation_matrix: np.ndarray,
    ) -> TwoCurrencyFXModel:
        """Construct TwoCurrencyFXModel from model or LGMParams instances.

        Args:
            domestic: Domestic interest rate model or :class:`LGMParams`.
            foreign: Foreign interest rate model or :class:`LGMParams`.
            spot_fx: Current spot FX rate.
            fx_vol_ann: Annualised FX volatility.
            correlation_matrix: 3x3 correlation matrix.

        Returns:
            A new :class:`TwoCurrencyFXModel` instance.
        """
        dom_model = LGMModel(domestic) if isinstance(domestic, LGMParams) else domestic
        for_model = LGMModel(foreign) if isinstance(foreign, LGMParams) else foreign
        return cls(
            domestic_ir_model=dom_model,
            foreign_ir_model=for_model,
            spot_fx=spot_fx,
            fx_vol_ann=fx_vol_ann,
            correlation_matrix=correlation_matrix,
        )

    from_components = from_ir_models

    @classmethod
    def from_lgm_params(
        cls,
        domestic: LGMParams,
        foreign: LGMParams,
        spot_fx: float,
        fx_vol_ann: float,
        correlation_matrix: np.ndarray,
    ) -> TwoCurrencyFXModel:
        """Construct a TwoCurrencyFXModel from LGMParams instances."""
        return cls.from_ir_models(
            domestic=domestic,
            foreign=foreign,
            spot_fx=spot_fx,
            fx_vol_ann=fx_vol_ann,
            correlation_matrix=correlation_matrix,
        )

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
            random_type: Random sequence type (:class:`RandomSequenceType` or str).
            seed: Optional random seed.
            scramble: If True, scrambles QMC sequences.

        Returns:
            ``(times, x_dom, x_for, fx_spot)`` — arrays of simulated paths.
        """
        dt = maturity_yrs / n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, maturity_yrs, n_steps + 1)

        chol = np.linalg.cholesky(self._correlation_matrix)

        x_dom = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        x_for = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        ln_fx = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        ln_fx[:, 0] = np.log(self._spot_fx)

        rho_f_fx = self._correlation_matrix[1, 2]
        vol_fx = self._fx_vol_ann

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

            z_indep = z_all[:, step, :]
            z_corr = z_indep @ chol.T

            dw_d = z_corr[:, 0] * sqrt_dt
            dw_f = z_corr[:, 1] * sqrt_dt
            dw_fx = z_corr[:, 2] * sqrt_dt

            # Domestic state evolution
            if isinstance(self._domestic, LGMModel):
                sig_d = self._domestic.sigma_at(t)
                kd = self._domestic.kappa_ann
                x_dom[:, step + 1] = (
                    x_dom[:, step] - kd * x_dom[:, step] * dt + sig_d * dw_d
                )
            else:
                # General diffusion step
                sim_step = self._domestic.simulate_paths(
                    np.array([t, t + dt]),
                    n_paths,
                    rng,
                    dw=dw_d.reshape(-1, 1),
                )
                x_dom[:, step + 1] = sim_step[:, -1]

            # Foreign state evolution with quanto drift adjustment
            if isinstance(self._foreign, LGMModel):
                sig_f = self._foreign.sigma_at(t)
                kf = self._foreign.kappa_ann
                quanto_drift = -sig_f * rho_f_fx * vol_fx
                x_for[:, step + 1] = (
                    x_for[:, step]
                    - kf * x_for[:, step] * dt
                    + quanto_drift * dt
                    + sig_f * dw_f
                )
            else:
                # General quanto adjustment on foreign diffusion
                sim_step = self._foreign.simulate_paths(
                    np.array([t, t + dt]),
                    n_paths,
                    rng,
                    dw=dw_f.reshape(-1, 1),
                )
                x_for[:, step + 1] = sim_step[:, -1]

            # Short rates at step t
            # For exact forward consistency with discrete steps:
            df_d_t = float(self._domestic.interpolate_discount_factor(t))
            df_d_t1 = float(self._domestic.interpolate_discount_factor(t + dt))
            df_f_t = float(self._foreign.interpolate_discount_factor(t))
            df_f_t1 = float(self._foreign.interpolate_discount_factor(t + dt))

            fwd_d = -np.log(df_d_t1 / max(df_d_t, 1e-18)) / dt
            fwd_f = -np.log(df_f_t1 / max(df_f_t, 1e-18)) / dt

            if isinstance(self._domestic, LGMModel):
                kd = self._domestic.kappa_ann
                h_prime_d = np.exp(-kd * t)
                zeta_d = self._domestic.zeta(t)
                r_d = fwd_d + h_prime_d * x_dom[:, step] + 0.5 * h_prime_d**2 * zeta_d
            else:
                r_d = self._domestic.short_rate(t, x_dom[:, step])

            if isinstance(self._foreign, LGMModel):
                kf = self._foreign.kappa_ann
                h_prime_f = np.exp(-kf * t)
                zeta_f = self._foreign.zeta(t)
                r_f = fwd_f + h_prime_f * x_for[:, step] + 0.5 * h_prime_f**2 * zeta_f
            else:
                r_f = self._foreign.short_rate(t, x_for[:, step])

            drift_fx = r_d - r_f - 0.5 * vol_fx**2
            ln_fx[:, step + 1] = ln_fx[:, step] + drift_fx * dt + vol_fx * dw_fx

        fx_spot = np.exp(ln_fx)
        return times, x_dom, x_for, fx_spot
