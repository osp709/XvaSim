"""Modular credit model calibration and CVA computation engine.

This module implements Credit Valuation Adjustment (CVA) aggregation and credit
spread calibration. It supports pluggable credit / hazard-rate models
(:class:`~xvasim.models.base.CreditModel`, such as
:class:`~xvasim.models.credit.CIRHazardRateModel`) and provides path-wise Monte
Carlo CVA calculation with memory chunking and Numexpr acceleration.

Public API
----------
- :class:`CIRParams` — calibrated CIR model parameters (re-exported).
- :func:`compute_cva` — path-wise CVA aggregation with numexpr and chunking.
- :func:`compute_cva_chunked` — generator/iterable streaming CVA aggregation.
- :func:`compute_exposure_profile` — counterparty EE, EPE, and PFE metrics.
- :func:`compute_marginal_pd` — marginal default probabilities from spreads.

Units & Conventions
-------------------
- Time / tenor in **years** (suffix ``_yrs``).
- Rates / spreads as **annualised decimals** (suffix ``_ann``).
"""

from __future__ import annotations

import typing

import numpy as np

from .models.base import CreditModel
from .models.credit.cir import CIRHazardRateModel, CIRParams

try:
    import numexpr as _ne  # type: ignore[import-not-found]

    HAS_NUMEXPR = True
except Exception:  # pragma: no cover
    HAS_NUMEXPR = False
    _ne = None

__all__ = [
    "CIRParams",
    "compute_cva",
    "compute_cva_chunked",
    "compute_exposure_profile",
    "compute_marginal_pd",
]


# ---------------------------------------------------------------------------
# Credit model survival probability & Calibration helpers
# ---------------------------------------------------------------------------


def _credit_model_survival_probability(
    tenors_yrs: np.ndarray,
    params: CIRParams | CreditModel,
) -> np.ndarray:
    r"""Compute survival probabilities using a CreditModel or CIRParams instance.

    .. math::
        P_{\text{surv}}(0, t) = A(t)\,e^{-B(t)\,\lambda_{0,\text{ann}}}

    Args:
        tenors_yrs: 1-D array of time points (in years).
        params: Calibrated :class:`CreditModel` or :class:`CIRParams` instance.

    Returns:
        1-D array of survival probabilities at each tenor.
    """
    if isinstance(params, CreditModel):
        return params.survival_probability(tenors_yrs)
    return CIRHazardRateModel(params).survival_probability(tenors_yrs)


_cir_survival_probability = _credit_model_survival_probability


def _calibrate_credit_model(
    credit_spreads_ann: np.ndarray,
    tenors_yrs: np.ndarray,
    model_type: str = "cir",
) -> CIRParams:
    r"""Calibrate a credit model to market credit spreads.

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
        model_type: Category of credit model to calibrate (default: ``"cir"``).

    Returns:
        A calibrated model parameters instance (e.g. :class:`CIRParams`).

    Raises:
        ValueError: If an unsupported *model_type* is specified.
        RuntimeError: If the optimisation fails to converge.
    """
    model_key = model_type.strip().lower()
    if model_key not in ("cir", "cox_ingersoll_ross"):
        msg = (
            f"Credit calibration currently supports 'cir', "
            f"got model_type='{model_type}'"
        )
        raise ValueError(msg)

    return CIRHazardRateModel.calibrate_from_spreads(
        credit_spreads_ann, tenors_yrs
    ).params


_calibrate_cir = _calibrate_credit_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_marginal_pd(
    credit_spreads_ann: np.ndarray,
    tenors_yrs: np.ndarray,
    model: CreditModel | None = None,
) -> np.ndarray:
    r"""Compute marginal default probabilities using a CIR or modular credit model.

    Calibrates a hazard-rate model to the provided market credit spreads,
    computes cumulative default probabilities from the calibrated survival
    curve, and returns the marginal default probability for each interval
    :math:`[t_{i-1},\, t_i]`.

    .. math::
        \text{Marginal PD}_i = F(t_i) - F(t_{i-1})
        \quad\text{where}\quad F(t) = 1 - P_{\text{surv}}(0, t)

    Args:
        credit_spreads_ann: 1-D array of market credit spreads at each
            tenor (annualised decimals, e.g. 0.02 for 2.0 % p.a.).
        tenors_yrs: 1-D array of time points (years) at which to
            evaluate the default probabilities.
        model: Optional pre-calibrated or custom
            :class:`~xvasim.models.base.CreditModel`. If None (default),
            calibrates a :class:`~xvasim.models.credit.CIRHazardRateModel`.

    Returns:
        1-D array of marginal default probabilities at each tenor.
    """
    if model is not None:
        return model.marginal_pd(tenors_yrs)

    credit_model = CIRHazardRateModel.calibrate_from_spreads(
        credit_spreads_ann, tenors_yrs
    )
    return credit_model.marginal_pd(tenors_yrs)


def compute_cva(
    exposure: np.ndarray,
    marginal_pd: np.ndarray,
    discount_factor: np.ndarray,
    loss_given_default: float,
    chunk_size: int | None = None,
    use_numexpr: bool = True,
) -> float:
    r"""Calculate the Credit Valuation Adjustment (CVA) of a counterparty.

    .. math::
        \text{CVA} = \text{LGD} \times \frac{1}{N_{\text{paths}}}
        \sum_{i=1}^{N_{\text{paths}}}
        \sum_{j=1}^{N_{\text{dates}}}
        E_{i,j}\;\Delta\text{PD}_{i,j}\;D_{i,j}

    Supports memory-efficient evaluation via chunking and **Numexpr** acceleration
    to avoid large intermediate array allocations for massive simulation grids.

    Args:
        exposure: 2-D array of shape ``(n_paths, n_dates)`` containing
            portfolio exposure values.
        marginal_pd: Array of shape ``(n_paths, n_dates)`` or ``(n_dates,)`` containing
            marginal default probabilities for each period (dimensionless).
        discount_factor: Array of shape ``(n_paths, n_dates)`` or ``(n_dates,)``
            containing risk-free discount factors.
        loss_given_default: Loss given default (decimal, e.g. 0.60 for 60 %).
        chunk_size: Optional integer batch size for chunked evaluation over paths.
        use_numexpr: If True and Numexpr is available, accelerates element-wise
            reduction while eliminating peak memory allocations.

    Returns:
        The average CVA value across all paths.
    """
    exp_arr = np.asarray(exposure, dtype=np.float64)
    n_paths = exp_arr.shape[0]

    if n_paths == 0:
        return 0.0

    pd_arr = np.asarray(marginal_pd, dtype=np.float64)
    df_arr = np.asarray(discount_factor, dtype=np.float64)
    lgd = float(loss_given_default)

    # Chunked evaluation
    if chunk_size is not None and chunk_size > 0 and chunk_size < n_paths:
        total_sum = 0.0
        for start_idx in range(0, n_paths, chunk_size):
            end_idx = min(start_idx + chunk_size, n_paths)
            exp_chunk = exp_arr[start_idx:end_idx]
            pd_chunk = pd_arr[start_idx:end_idx] if pd_arr.ndim == 2 else pd_arr
            df_chunk = df_arr[start_idx:end_idx] if df_arr.ndim == 2 else df_arr

            if use_numexpr and HAS_NUMEXPR and _ne is not None:
                chunk_val = float(
                    _ne.evaluate("sum(exp_chunk * pd_chunk * df_chunk * lgd)")
                )
            else:
                chunk_val = float(np.sum(exp_chunk * pd_chunk * df_chunk * lgd))
            total_sum += chunk_val

        return float(total_sum / n_paths)

    # Full array evaluation
    if use_numexpr and HAS_NUMEXPR and _ne is not None:
        total_val = float(_ne.evaluate("sum(exp_arr * pd_arr * df_arr * lgd)"))
        return float(total_val / n_paths)

    path_cva = np.sum(
        exp_arr * pd_arr * df_arr * lgd,
        axis=1,
        keepdims=True,
    )
    return float(np.mean(path_cva))


def compute_cva_chunked(
    exposure_chunks: typing.Iterable[np.ndarray],
    marginal_pd: np.ndarray,
    discount_factor: np.ndarray,
    loss_given_default: float,
    use_numexpr: bool = True,
) -> float:
    """Calculate CVA over a stream or generator of exposure matrix chunks.

    Memory-efficient aggregator for massive portfolios where the complete exposure
    matrix exceeds system RAM.

    Args:
        exposure_chunks: Iterable/generator yielding 2-D arrays of shape
            ``(chunk_paths, n_dates)``.
        marginal_pd: Array of shape ``(n_dates,)`` or matching 2-D shape.
        discount_factor: Array of shape ``(n_dates,)`` or matching 2-D shape.
        loss_given_default: Loss given default (e.g. 0.60).
        use_numexpr: If True and Numexpr is available, accelerates evaluation.

    Returns:
        The average CVA value across all aggregated paths.
    """
    total_weighted_cva = 0.0
    total_paths = 0

    pd_arr = np.asarray(marginal_pd, dtype=np.float64)
    df_arr = np.asarray(discount_factor, dtype=np.float64)
    lgd = float(loss_given_default)

    offset = 0
    for chunk in exposure_chunks:
        exp_chunk = np.asarray(chunk, dtype=np.float64)
        c_paths = exp_chunk.shape[0]
        if c_paths == 0:
            continue

        pd_chunk = (
            pd_arr[offset : offset + c_paths] if pd_arr.ndim == 2 else pd_arr
        )
        df_chunk = (
            df_arr[offset : offset + c_paths] if df_arr.ndim == 2 else df_arr
        )
        offset += c_paths

        if use_numexpr and HAS_NUMEXPR and _ne is not None:
            chunk_sum = float(
                _ne.evaluate("sum(exp_chunk * pd_chunk * df_chunk * lgd)")
            )
        else:
            chunk_sum = float(np.sum(exp_chunk * pd_chunk * df_chunk * lgd))

        total_weighted_cva += chunk_sum
        total_paths += c_paths

    if total_paths == 0:
        return 0.0

    return float(total_weighted_cva / total_paths)


def compute_exposure_profile(
    exposure: np.ndarray,
    percentiles: typing.Sequence[float] = (95.0, 97.5, 99.0),
) -> dict[str, typing.Any]:
    r"""Compute counterparty exposure profiles: Expected Exposure (EE), EPE, and PFE.

    Given a simulated portfolio exposure matrix :math:`E_{i, j}` of shape
    ``(n_paths, n_dates)``:
    - **Expected Exposure (EE)**: :math:`EE(t_j) = \frac{1}{N}\sum_{i=1}^N E_{i, j}`
    - **Expected Positive Exposure (EPE)**: Average Expected Exposure across time.
    - **Potential Future Exposure (PFE)**: Path-wise quantiles at given
      confidence level(s).
    - **Max PFE**: Peak value of the PFE profile.

    Args:
        exposure: 2-D array of shape ``(n_paths, n_dates)`` containing
            portfolio exposure paths.
        percentiles: Sequence of percentile confidence levels
            (default: (95.0, 97.5, 99.0)).

    Returns:
        Dictionary containing:
        - ``"expected_exposure"`` (and ``"ee"``): 1-D array of shape ``(n_dates,)``.
        - ``"expected_positive_exposure"`` (and ``"epe"``): scalar float.
        - ``"max_pfe"``: scalar float (peak across highest computed percentile).
        - ``"pfe_profiles"``: dict mapping float percentile level to 1-D PFE curve.
        - Convenience percentile keys (e.g. ``"pfe_95"``, ``"pfe_99"``).
    """
    exp_arr = np.asarray(exposure, dtype=np.float64)
    if exp_arr.ndim == 1:
        exp_arr = exp_arr.reshape(1, -1)

    n_paths, n_dates = exp_arr.shape
    if n_paths == 0 or n_dates == 0:
        return {
            "expected_exposure": np.array([], dtype=np.float64),
            "ee": np.array([], dtype=np.float64),
            "expected_positive_exposure": 0.0,
            "epe": 0.0,
            "max_pfe": 0.0,
            "pfe_profiles": {},
        }

    ee = np.mean(exp_arr, axis=0)
    epe = float(np.mean(ee))

    pfe_profiles: dict[float, np.ndarray] = {}
    result: dict[str, typing.Any] = {
        "expected_exposure": ee,
        "ee": ee,
        "expected_positive_exposure": epe,
        "epe": epe,
        "pfe_profiles": pfe_profiles,
    }

    max_pfe_val = 0.0
    for p in percentiles:
        p_val = float(p)
        pfe_curve = np.percentile(exp_arr, p_val, axis=0)
        pfe_profiles[p_val] = pfe_curve
        max_pfe_val = max(max_pfe_val, float(np.max(pfe_curve)))
        result[f"pfe_{p_val}"] = pfe_curve
        if p_val.is_integer():
            result[f"pfe_{int(p_val)}"] = pfe_curve
        clean_key = f"pfe_{str(p_val).replace('.', '_').rstrip('_0')}"
        result[clean_key] = pfe_curve

    result["max_pfe"] = max_pfe_val
    return result
