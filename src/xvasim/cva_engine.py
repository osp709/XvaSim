"""CIR-based credit model calibration and CVA computation engine.

This module implements the **Cox-Ingersoll-Ross (CIR)** hazard-rate model
for computing survival probabilities and Credit Valuation Adjustment (CVA).
The CIR model parameters are calibrated to observed market credit spreads,
and the resulting survival curve drives marginal default-probability
estimation for path-wise CVA aggregation.

Public API
----------
- :class:`CIRParams`  — calibrated CIR model parameters.
- :func:`compute_marginal_pd` — marginal default probabilities from spreads.
- :func:`compute_cva` — path-wise CVA aggregation.

Units & Conventions
-------------------
- Time / tenor in **years** (suffix ``_yrs``).
- Rates / spreads as **annualised decimals** (suffix ``_ann``).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CIRParams:
    """Calibrated parameters for the Cox-Ingersoll-Ross hazard-rate model.

    The CIR process for the default intensity is:

    .. math::
        d\\lambda_t = \\kappa_{\\text{ann}}
        (\\theta_{\\text{ann}} - \\lambda_t)\\,dt
        + \\sigma_{\\text{ann}}\\sqrt{\\lambda_t}\\,dW_t

    Attributes:
        kappa_ann: Speed of mean reversion (annualised, e.g. 0.5).
        theta_ann: Long-term mean hazard rate (annualised decimal,
            e.g. 0.03 for 3 % per annum).
        sigma_ann: Volatility of the hazard-rate process (annualised,
            e.g. 0.10).
        lambda_0_ann: Initial hazard rate at time 0 (annualised decimal,
            e.g. 0.02).
    """

    kappa_ann: float
    theta_ann: float
    sigma_ann: float
    lambda_0_ann: float


# ---------------------------------------------------------------------------
# Survival probability
# ---------------------------------------------------------------------------


def _cir_survival_probability(
    tenors_yrs: np.ndarray,
    params: CIRParams,
) -> np.ndarray:
    r"""Compute survival probabilities using the CIR closed-form solution.

    .. math::
        P_{\text{surv}}(0, t) = A(t)\,e^{-B(t)\,\lambda_{0,\text{ann}}}

    where :math:`\gamma = \sqrt{\kappa_{\text{ann}}^2
    + 2\,\sigma_{\text{ann}}^2}` and:

    .. math::
        A(t) = \left[\frac{2\gamma\,e^{(\kappa_{\text{ann}}+\gamma)\,t/2}}
        {(\kappa_{\text{ann}}+\gamma)(e^{\gamma t}-1)+2\gamma}
        \right]^{\frac{2\kappa_{\text{ann}}\theta_{\text{ann}}}
        {\sigma_{\text{ann}}^2}}

    .. math::
        B(t) = \frac{2(e^{\gamma t}-1)}
        {(\kappa_{\text{ann}}+\gamma)(e^{\gamma t}-1)+2\gamma}

    Args:
        tenors_yrs: 1-D array of time points (in years).
        params: Calibrated :class:`CIRParams` instance.

    Returns:
        1-D array of survival probabilities at each tenor.
    """
    kappa = params.kappa_ann
    theta = params.theta_ann
    sigma = params.sigma_ann
    lam0 = params.lambda_0_ann

    gamma = np.sqrt(kappa**2 + 2 * sigma**2)
    exp_gamma_t = np.exp(gamma * tenors_yrs)
    denominator = (kappa + gamma) * (exp_gamma_t - 1) + 2 * gamma

    power = (2 * kappa * theta) / (sigma**2)
    numerator_a = 2 * gamma * np.exp((kappa + gamma) * tenors_yrs / 2)
    a_t = (numerator_a / denominator) ** power
    b_t = 2 * (exp_gamma_t - 1) / denominator

    return a_t * np.exp(-b_t * lam0)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _calibrate_cir(
    credit_spreads_ann: np.ndarray,
    tenors_yrs: np.ndarray,
) -> CIRParams:
    r"""Calibrate CIR model parameters to market credit spreads.

    Minimises the sum of squared errors between model-implied credit
    spreads and the observed market credit spreads using **L-BFGS-B**:

    .. math::
        \min_{\kappa,\,\theta,\,\sigma,\,\lambda_0}
        \sum_{k=1}^{M}\bigl(S_{\text{model}}(t_k)
        - S_{\text{market}}(t_k)\bigr)^2

    where the model-implied spread at tenor *t* is:

    .. math::
        S_{\text{model}}(t) = -\frac{\ln P_{\text{surv}}(0,t)}{t}

    Args:
        credit_spreads_ann: 1-D array of market credit spreads at each
            tenor (annualised decimals, e.g. 0.02 for 2.0 % p.a.).
        tenors_yrs: 1-D array of time points (years) corresponding to
            the credit spreads.

    Returns:
        A :class:`CIRParams` instance with calibrated parameters.

    Raises:
        RuntimeError: If the optimisation fails to converge.
    """

    def objective(params_vec: np.ndarray) -> float:
        cir = CIRParams(*params_vec)
        surv_prob = _cir_survival_probability(tenors_yrs, cir)
        model_spreads_ann = -np.log(np.maximum(surv_prob, 1e-15)) / np.maximum(
            tenors_yrs, 1e-10
        )
        return float(np.sum((model_spreads_ann - credit_spreads_ann) ** 2))

    x0 = [
        0.1,
        float(np.mean(credit_spreads_ann)),
        0.05,
        float(credit_spreads_ann[0]),
    ]
    bounds = [
        (1e-4, 5.0),
        (1e-4, 2.0),
        (1e-4, 1.0),
        (1e-4, 2.0),
    ]

    result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
    if not result.success:
        msg = f"CIR calibration failed: {result.message}"
        raise RuntimeError(msg)

    return CIRParams(*result.x)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_marginal_pd(
    credit_spreads_ann: np.ndarray,
    tenors_yrs: np.ndarray,
) -> np.ndarray:
    r"""Compute marginal default probabilities using a CIR model.

    Calibrates a Cox-Ingersoll-Ross hazard-rate model to the provided
    market credit spreads, computes cumulative default probabilities
    from the calibrated survival curve, and returns the marginal default
    probability for each interval :math:`[t_{i-1},\, t_i]`.

    .. math::
        \text{Marginal PD}_i = F(t_i) - F(t_{i-1})
        \quad\text{where}\quad F(t) = 1 - P_{\text{surv}}(0, t)

    Args:
        credit_spreads_ann: 1-D array of market credit spreads at each
            tenor (annualised decimals, e.g. 0.02 for 2.0 % p.a.).
        tenors_yrs: 1-D array of time points (years) at which to
            evaluate the default probabilities.

    Returns:
        1-D array of marginal default probabilities at each tenor.
    """
    params = _calibrate_cir(credit_spreads_ann, tenors_yrs)
    survival_probability = _cir_survival_probability(tenors_yrs, params)
    cumulative_pd = 1.0 - survival_probability
    marginal_pd = np.diff(cumulative_pd, prepend=0.0)
    return marginal_pd


def compute_cva(
    exposure: np.ndarray,
    marginal_pd: np.ndarray,
    discount_factor: np.ndarray,
    loss_given_default: float,
) -> float:
    r"""Calculate the Credit Valuation Adjustment (CVA) of a counterparty.

    .. math::
        \text{CVA} = \text{LGD} \times \frac{1}{N_{\text{paths}}}
        \sum_{i=1}^{N_{\text{paths}}}
        \sum_{j=1}^{N_{\text{dates}}}
        E_{i,j}\;\Delta\text{PD}_{i,j}\;D_{i,j}

    Args:
        exposure: 2-D array of shape ``(n_paths, n_dates)`` containing
            portfolio exposure values.
        marginal_pd: 2-D array of shape ``(n_paths, n_dates)`` containing
            marginal default probabilities for each period (dimensionless).
        discount_factor: 2-D array of shape ``(n_paths, n_dates)`` containing
            risk-free discount factors.
        loss_given_default: Loss given default (decimal, e.g. 0.60 for 60 %).

    Returns:
        The average CVA value across all paths.
    """
    path_cva = np.sum(
        exposure * marginal_pd * discount_factor * loss_given_default,
        axis=1,
        keepdims=True,
    )
    return float(np.mean(path_cva))
