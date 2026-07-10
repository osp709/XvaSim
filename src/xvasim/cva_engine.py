import typing

import numpy as np
from scipy.optimize import minimize


def compute_cva(
    exposure: np.ndarray,
    marginal_pd: np.ndarray,
    discount_factor: np.ndarray,
    loss_given_default: float,
) -> float:
    """Calculate the Credit Valuation Adjustment (CVA) of a counterparty.

    Args:
        exposure: 2D numpy array of shape (n_paths, n_dates) containing the
            portfolio exposure values.
        marginal_pd: 2D numpy array of shape (n_paths, n_dates) containing the
            marginal default probabilities for each period (unit: dimensionless
            probability).
        discount_factor: 2D numpy array of shape (n_paths, n_dates) containing the
            risk-free discount factors.
        loss_given_default: A floating point number representing the loss
            given default (LGD, in decimal, e.g., 0.60 for 60% loss).

    Returns:
        The average CVA value across all paths.
    """
    path_cva = np.sum(
        exposure * marginal_pd * discount_factor * loss_given_default,
        axis=1,
        keepdims=True,
    )
    return float(np.mean(path_cva))


def _cir_survival_probability(
    tenors_yrs: np.ndarray,
    kappa_ann: float,
    theta_ann: float,
    sigma_ann: float,
    lambda_0_ann: float,
) -> np.ndarray:
    """Compute survival probabilities using the CIR closed-form solution.

    The CIR process for the hazard rate is:
        d(lambda) = kappa_ann * (theta_ann - lambda) * dt
                    + sigma_ann * sqrt(lambda) * dW(t)

    The survival probability P(0, t) = A(t) * exp(-B(t) * lambda_0_ann)
    where A and B are determined by the CIR model parameters.

    Args:
        tenors_yrs: 1D array of time points (in years).
        kappa_ann: Speed of mean reversion (annualized rate, e.g., 0.5).
        theta_ann: Long-term mean hazard rate (annualized rate in decimal,
            e.g., 0.03).
        sigma_ann: Volatility of the hazard rate process (annualized,
            e.g., 0.10).
        lambda_0_ann: Initial hazard rate at time 0 (annualized rate in
            decimal, e.g., 0.02).

    Returns:
        1D array of survival probabilities at each tenor.
    """
    gamma = np.sqrt(kappa_ann**2 + 2 * sigma_ann**2)
    exp_gamma_t = np.exp(gamma * tenors_yrs)
    denominator = (kappa_ann + gamma) * (exp_gamma_t - 1) + 2 * gamma

    power = (2 * kappa_ann * theta_ann) / (sigma_ann**2)
    numerator_a = 2 * gamma * np.exp((kappa_ann + gamma) * tenors_yrs / 2)
    a_t = (numerator_a / denominator) ** power
    b_t = 2 * (exp_gamma_t - 1) / denominator

    return typing.cast(np.ndarray, a_t * np.exp(-b_t * lambda_0_ann))


def _calibrate_cir(
    credit_spreads_ann: np.ndarray,
    tenors_yrs: np.ndarray,
) -> tuple[float, float, float, float]:
    """Calibrate CIR model parameters to market credit spreads.

    Minimises the sum of squared errors between model-implied credit
    spreads and the observed market credit spreads using L-BFGS-B.

    The model-implied spread at tenor t is:
        S_model(t) = -ln(P_surv(t)) / t

    Args:
        credit_spreads_ann: 1D array of market credit spreads at each
            tenor (annualized rates in decimal, e.g., 0.02 for 2.0% per
            annum).
        tenors_yrs: 1D array of time points (in years) corresponding to
            the credit spreads.

    Returns:
        A tuple (kappa_ann, theta_ann, sigma_ann, lambda_0_ann) of
        calibrated CIR model parameters.

    Raises:
        RuntimeError: If the optimisation fails to converge.
    """

    def objective(params: np.ndarray) -> float:
        kappa_ann, theta_ann, sigma_ann, lambda_0_ann = params
        surv_prob = _cir_survival_probability(
            tenors_yrs, kappa_ann, theta_ann, sigma_ann, lambda_0_ann
        )
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

    kappa_ann, theta_ann, sigma_ann, lambda_0_ann = result.x
    return float(kappa_ann), float(theta_ann), float(sigma_ann), float(lambda_0_ann)


def compute_marginal_pd(
    credit_spreads_ann: np.ndarray,
    tenors_yrs: np.ndarray,
) -> np.ndarray:
    """Compute marginal default probabilities using a CIR model.

    Calibrates a Cox-Ingersoll-Ross hazard rate model to the provided
    market credit spreads, computes cumulative default probabilities
    from the calibrated survival curve, and returns marginal default
    probabilities at each tenor.

    Args:
        credit_spreads_ann: 1D array of market credit spreads at each
            tenor (annualized rates in decimal, e.g., 0.02 for 2.0% per
            annum).
        tenors_yrs: 1D array of time points (in years) at which to
            evaluate the default probabilities. These represent
            the t in P(t).

    Returns:
        1D array of marginal default probabilities at each tenor.
    """
    kappa_ann, theta_ann, sigma_ann, lambda_0_ann = _calibrate_cir(
        credit_spreads_ann, tenors_yrs
    )
    survival_probability = _cir_survival_probability(
        tenors_yrs, kappa_ann, theta_ann, sigma_ann, lambda_0_ann
    )
    cumulative_pd = 1.0 - survival_probability
    marginal_pd = np.diff(cumulative_pd, prepend=0.0)
    return marginal_pd
