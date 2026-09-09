"""Modular credit model calibration and XVA computation engine.

This module implements Credit Valuation Adjustment (CVA), Debit Valuation
Adjustment (DVA), Funding Valuation Adjustment (FVA), Capital Valuation
Adjustment (KVA), and Marginal Funding Valuation Adjustment (MVA)
aggregation along with credit spread calibration.  It supports pluggable
credit / hazard-rate models (:class:`~xvasim.models.base.CreditModel`,
such as :class:`~xvasim.models.credit.CIRHazardRateModel`) and provides
path-wise Monte Carlo XVA calculation with memory chunking and Numexpr
acceleration.

Public API
----------
- :class:`CIRHazardRateParams` — calibrated CIR model parameters (re-exported).
- :func:`compute_cva` — path-wise CVA aggregation with numexpr and chunking.
- :func:`compute_cva_chunked` — generator/iterable streaming CVA aggregation.
- :func:`compute_dva` — Debit Valuation Adjustment (own-default benefit).
- :func:`compute_fva` — Funding Valuation Adjustment, decomposed into the
  funding cost (FCA) and symmetric funding benefit (FBA) legs.
- :func:`compute_kva` — Capital Valuation Adjustment (regulatory capital cost).
- :func:`compute_mva` — Marginal Funding Valuation Adjustment (IM funding cost).
- :func:`compute_total_xva` — single-pass CVA+DVA+FVA+KVA+MVA aggregator.
- :func:`compute_exposure_profile` — counterparty EE, EPE, NE, and PFE metrics.
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
from .models.credit.cir import CIRHazardRateModel, CIRHazardRateParams

try:
    import numexpr as _ne  # type: ignore[import-not-found]

    HAS_NUMEXPR = True
except Exception:  # pragma: no cover
    HAS_NUMEXPR = False
    _ne = None

__all__ = [
    "CIRHazardRateParams",
    "compute_cva",
    "compute_cva_chunked",
    "compute_dva",
    "compute_exposure_profile",
    "compute_fva",
    "compute_kva",
    "compute_marginal_pd",
    "compute_mva",
    "compute_total_xva",
]


# ---------------------------------------------------------------------------
# Credit model survival probability & Calibration helpers
# ---------------------------------------------------------------------------


def _credit_model_survival_probability(
    tenors_yrs: np.ndarray,
    params: CIRHazardRateParams | CreditModel,
) -> np.ndarray:
    r"""Compute survival probabilities using a CreditModel or
    CIRHazardRateParams instance.

    .. math::
        P_{\text{surv}}(0, t) = A(t)\,e^{-B(t)\,\lambda_{0,\text{ann}}}

    Args:
        tenors_yrs: 1-D array of time points (in years).
        params: Calibrated :class:`CreditModel` or
            :class:`CIRHazardRateParams` instance.

    Returns:
        1-D array of survival probabilities at each tenor.
    """
    if isinstance(params, CreditModel):
        return params.survival_probability(tenors_yrs)
    return CIRHazardRateModel(params).survival_probability(tenors_yrs)


def _calibrate_credit_model(
    credit_spreads_ann: np.ndarray,
    tenors_yrs: np.ndarray,
    model_type: str = "cir",
) -> CIRHazardRateParams:
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
        A calibrated model parameters instance (e.g. :class:`CIRHazardRateParams`).

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
    lgd = loss_given_default

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
    lgd = loss_given_default

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


# ---------------------------------------------------------------------------
# DVA — Debit Valuation Adjustment
# ---------------------------------------------------------------------------


def compute_dva(
    exposure: np.ndarray,
    marginal_pd: np.ndarray,
    discount_factor: np.ndarray,
    loss_given_default: float,
    chunk_size: int | None = None,
    use_numexpr: bool = True,
) -> float:
    r"""Calculate the Debit Valuation Adjustment (DVA).

    DVA captures the reduction in a firm's liabilities due to its own
    default risk. It is the mirror image of CVA applied to negative
    (counterparty-beneficial) exposures using the firm's own credit curve:

    .. math::
        \text{DVA} = \text{LGD}_{\text{own}}
        \times \frac{1}{N_{\text{paths}}}
        \sum_{i=1}^{N_{\text{paths}}}
        \sum_{j=1}^{N_{\text{dates}}}
        \text{NE}_{i,j}\;\Delta\text{PD}_{i,j}\;D_{i,j}

    where :math:`\text{NE}_{i,j} = \max(-E_{i,j},\, 0)` is the negative
    exposure (benefit to the firm upon its own default).

    Args:
        exposure: 2-D array of shape ``(n_paths, n_dates)`` containing
            portfolio exposure values (positive = credit exposure to
            counterparty, negative = liability).
        marginal_pd: Array of shape ``(n_paths, n_dates)`` or ``(n_dates,)``
            containing the firm's own marginal default probabilities.
        discount_factor: Array of shape ``(n_paths, n_dates)`` or ``(n_dates,)``
            containing risk-free discount factors.
        loss_given_default: Firm's own loss given default (decimal).
        chunk_size: Optional batch size for memory-efficient chunked evaluation.
        use_numexpr: If True and Numexpr is available, accelerates evaluation.

    Returns:
        The average DVA value across all paths.
    """
    exp_arr = np.asarray(exposure, dtype=np.float64)
    n_paths = exp_arr.shape[0]

    if n_paths == 0:
        return 0.0

    ne_arr = np.maximum(-exp_arr, 0.0)
    pd_arr = np.asarray(marginal_pd, dtype=np.float64)
    df_arr = np.asarray(discount_factor, dtype=np.float64)
    lgd = loss_given_default

    if chunk_size is not None and chunk_size > 0 and chunk_size < n_paths:
        total_sum = 0.0
        for start_idx in range(0, n_paths, chunk_size):
            end_idx = min(start_idx + chunk_size, n_paths)
            ne_chunk = ne_arr[start_idx:end_idx]
            pd_chunk = pd_arr[start_idx:end_idx] if pd_arr.ndim == 2 else pd_arr
            df_chunk = df_arr[start_idx:end_idx] if df_arr.ndim == 2 else df_arr

            if use_numexpr and HAS_NUMEXPR and _ne is not None:
                chunk_val = float(
                    _ne.evaluate("sum(ne_chunk * pd_chunk * df_chunk * lgd)")
                )
            else:
                chunk_val = float(np.sum(ne_chunk * pd_chunk * df_chunk * lgd))
            total_sum += chunk_val

        return float(total_sum / n_paths)

    if use_numexpr and HAS_NUMEXPR and _ne is not None:
        total_val = float(_ne.evaluate("sum(ne_arr * pd_arr * df_arr * lgd)"))
        return float(total_val / n_paths)

    path_dva = np.sum(ne_arr * pd_arr * df_arr * lgd, axis=1, keepdims=True)
    return float(np.mean(path_dva))


# ---------------------------------------------------------------------------
# FVA — Funding Valuation Adjustment
# ---------------------------------------------------------------------------


def compute_fva(
    exposure: np.ndarray,
    time_steps_yrs: np.ndarray,
    discount_factor: np.ndarray,
    funding_spread_borrow_ann: float,
    funding_spread_deposit_ann: float = 0.0,
    chunk_size: int | None = None,
    use_numexpr: bool = True,
) -> dict[str, float]:
    r"""Calculate the Funding Valuation Adjustment (FVA) and its symmetric legs.

    FVA captures the cost (or benefit) of funding derivatives positions
    through the balance sheet. Positive exposures fund the buying of the
    receivable and incur borrowing costs; negative exposures represent
    funds we hold, which earn deposit benefits. The two symmetric
    components are decomposed into:

    - **FCA** (Funding Cost Adjustment): cost of funding positive exposure
      (EE) at the borrowing spread.
    - **FBA** (Funding Benefit Adjustment): benefit earned on negative
      exposure (ENE) at the deposit spread.

    .. math::
        \text{FVA} = \text{FCA} + \text{FBA} = \frac{1}{N_{\text{paths}}}
        \sum_{i=1}^{N_{\text{paths}}}
        \sum_{j=1}^{N_{\text{dates}}}
        \bigl[\text{EE}_{i,j}\;f_{\text{borrow}}
        + \text{NE}_{i,j}\;f_{\text{deposit}}\bigr]
        D_{i,j}\;\Delta t_j

    where :math:`\text{EE}_{i,j} = \max(E_{i,j},\, 0)` and
    :math:`\text{NE}_{i,j} = \max(-E_{i,j},\, 0)`.

    Args:
        exposure: 2-D array of shape ``(n_paths, n_dates)`` containing
            portfolio exposure values.
        time_steps_yrs: 1-D array of shape ``(n_dates,)`` containing the
            time-step sizes in years (Δt) for each evaluation date.
        discount_factor: Array of shape ``(n_paths, n_dates)`` or ``(n_dates,)``
            containing risk-free discount factors.
        funding_spread_borrow_ann: Annualised borrowing spread (decimal,
            e.g. 0.005 for 50 bps).
        funding_spread_deposit_ann: Annualised deposit spread (decimal).
            Default 0.0 (unsecured deposit rate = risk-free).
        chunk_size: Optional batch size for memory-efficient chunked evaluation.
        use_numexpr: If True and Numexpr is available, accelerates evaluation.

    Returns:
        Dictionary with:
        - ``"fca"``: Funding Cost Adjustment (positive-exposure leg).
        - ``"fba"``: Funding Benefit Adjustment (negative-exposure leg).
        - ``"fva"``: Net FVA = FCA + FBA.
    """
    exp_arr = np.asarray(exposure, dtype=np.float64)
    n_paths = exp_arr.shape[0]

    if n_paths == 0:
        return {"fca": 0.0, "fba": 0.0, "fva": 0.0}

    dt_arr = np.asarray(time_steps_yrs, dtype=np.float64)
    df_arr = np.asarray(discount_factor, dtype=np.float64)
    f_borrow = funding_spread_borrow_ann
    f_deposit = funding_spread_deposit_ann

    pos_exp = np.maximum(exp_arr, 0.0)
    neg_exp = np.maximum(-exp_arr, 0.0)

    # Element-wise funding legs: EE * f_borrow * df * dt and NE * f_deposit * df * dt
    if use_numexpr and HAS_NUMEXPR and _ne is not None:
        fca_expr = _ne.evaluate("pos_exp * f_borrow * df_arr * dt_arr")
        fba_expr = _ne.evaluate("neg_exp * f_deposit * df_arr * dt_arr")
    else:
        fca_expr = pos_exp * f_borrow * df_arr * dt_arr
        fba_expr = neg_exp * f_deposit * df_arr * dt_arr

    def _leg_average(leg_expr: np.ndarray) -> float:
        if chunk_size is not None and chunk_size > 0 and chunk_size < n_paths:
            leg_sum = 0.0
            for start_idx in range(0, n_paths, chunk_size):
                end_idx = min(start_idx + chunk_size, n_paths)
                leg_sum += float(np.sum(leg_expr[start_idx:end_idx]))
            return leg_sum / n_paths
        return float(np.sum(leg_expr) / n_paths)

    fca_val = _leg_average(fca_expr)
    fba_val = _leg_average(fba_expr)
    fva_val = fca_val + fba_val
    return {"fca": fca_val, "fba": fba_val, "fva": fva_val}


# ---------------------------------------------------------------------------
# KVA — Capital Valuation Adjustment
# ---------------------------------------------------------------------------


def compute_kva(
    exposure: np.ndarray,
    time_steps_yrs: np.ndarray,
    discount_factor: np.ndarray,
    capital_charge_ann: float,
    regulatory_lgd: float = 0.75,
    chunk_size: int | None = None,
    use_numexpr: bool = True,
) -> float:
    r"""Calculate the Capital Valuation Adjustment (KVA).

    KVA estimates the lifetime cost of holding regulatory capital against
    CVA risk. Under Basel III / SA-CVA, the capital charge is proportional
    to the time-averaged expected positive exposure:

    .. math::
        \text{KVA} = c_{\text{reg}} \times \text{LGD}_{\text{reg}}
        \times \frac{1}{N_{\text{paths}}}
        \sum_{i=1}^{N_{\text{paths}}}
        \sum_{j=1}^{N_{\text{dates}}}
        \text{EPE}_{i,j}\;D_{i,j}\;\Delta t_j

    where :math:`\text{EPE}_{i,j} = \max(E_{i,j},\, 0)`.

    Args:
        exposure: 2-D array of shape ``(n_paths, n_dates)`` containing
            portfolio exposure values.
        time_steps_yrs: 1-D array of shape ``(n_dates,)`` containing the
            time-step sizes in years (Δt).
        discount_factor: Array of shape ``(n_paths, n_dates)`` or ``(n_dates,)``
            containing risk-free discount factors.
        capital_charge_ann: Annualised regulatory capital charge rate
            (decimal, e.g. 0.08 for 8 %).
        regulatory_lgd: Regulatory loss given default (decimal, default 0.75
            under Basel III SA-CVA).
        chunk_size: Optional batch size for memory-efficient chunked evaluation.
        use_numexpr: If True and Numexpr is available, accelerates evaluation.

    Returns:
        The average KVA value across all paths.
    """
    exp_arr = np.asarray(exposure, dtype=np.float64)
    n_paths = exp_arr.shape[0]

    if n_paths == 0:
        return 0.0

    dt_arr = np.asarray(time_steps_yrs, dtype=np.float64)
    df_arr = np.asarray(discount_factor, dtype=np.float64)
    c_reg = capital_charge_ann
    lgd_reg = regulatory_lgd

    pos_exp = np.maximum(exp_arr, 0.0)

    if use_numexpr and HAS_NUMEXPR and _ne is not None:
        base_expr = _ne.evaluate("pos_exp * df_arr * dt_arr * c_reg * lgd_reg")
    else:
        base_expr = pos_exp * df_arr * dt_arr * c_reg * lgd_reg

    if chunk_size is not None and chunk_size > 0 and chunk_size < n_paths:
        total_sum = 0.0
        for start_idx in range(0, n_paths, chunk_size):
            end_idx = min(start_idx + chunk_size, n_paths)
            total_sum += float(np.sum(base_expr[start_idx:end_idx]))
        return float(total_sum / n_paths)

    return float(np.sum(base_expr) / n_paths)


# ---------------------------------------------------------------------------
# MVA — Marginal Funding Valuation Adjustment
# ---------------------------------------------------------------------------


def compute_mva(
    exposure: np.ndarray,
    time_steps_yrs: np.ndarray,
    discount_factor: np.ndarray,
    funding_spread_borrow_ann: float,
    im_scaling: float = 1.0,
    im_percentile: float = 99.0,
    chunk_size: int | None = None,
    use_numexpr: bool = True,
) -> float:
    r"""Calculate the Marginal Funding Valuation Adjustment (MVA).

    MVA estimates the lifetime cost of funding initial margin (IM)
    posted against derivative positions. Initial margin is modelled as a
    linear scaling of the Potential Future Exposure (PFE) at a given
    confidence level:

    .. math::
        \text{IM}(t_j) = \alpha \times \text{PFE}_{p}(t_j)

    where :math:`\alpha` is the IM scaling factor and
    :math:`\text{PFE}_{p}` is the *p*-th percentile exposure profile.
    The MVA is then:

    .. math::
        \text{MVA} = f_{\text{borrow}} \times \frac{1}{N_{\text{paths}}}
        \sum_{i=1}^{N_{\text{paths}}}
        \sum_{j=1}^{N_{\text{dates}}}
        \text{IM}_{i,j}\;D_{i,j}\;\Delta t_j

    Args:
        exposure: 2-D array of shape ``(n_paths, n_dates)`` containing
            portfolio exposure values.
        time_steps_yrs: 1-D array of shape ``(n_dates,)`` containing the
            time-step sizes in years (Δt).
        discount_factor: Array of shape ``(n_paths, n_dates)`` or ``(n_dates,)``
            containing risk-free discount factors.
        funding_spread_borrow_ann: Annualised borrowing spread for IM funding
            (decimal).
        im_scaling: Scaling factor α for IM computation (default 1.0).
        im_percentile: Percentile confidence level for PFE-based IM
            (default 99.0).
        chunk_size: Optional batch size for memory-efficient chunked evaluation.
        use_numexpr: If True and Numexpr is available, accelerates evaluation.

    Returns:
        The average MVA value across all paths.
    """
    exp_arr = np.asarray(exposure, dtype=np.float64)
    n_paths = exp_arr.shape[0]

    if n_paths == 0:
        return 0.0

    dt_arr = np.asarray(time_steps_yrs, dtype=np.float64)
    df_arr = np.asarray(discount_factor, dtype=np.float64)
    f_borrow = funding_spread_borrow_ann

    # Compute time-profile PFE and broadcast to (n_paths, n_dates)
    pfe_curve = np.percentile(exp_arr, im_percentile, axis=0)
    im_profile = im_scaling * pfe_curve
    im_matrix = np.tile(im_profile, (n_paths, 1))

    if use_numexpr and HAS_NUMEXPR and _ne is not None:
        base_expr = _ne.evaluate("im_matrix * df_arr * dt_arr * f_borrow")
    else:
        base_expr = im_matrix * df_arr * dt_arr * f_borrow

    if chunk_size is not None and chunk_size > 0 and chunk_size < n_paths:
        total_sum = 0.0
        for start_idx in range(0, n_paths, chunk_size):
            end_idx = min(start_idx + chunk_size, n_paths)
            total_sum += float(np.sum(base_expr[start_idx:end_idx]))
        return float(total_sum / n_paths)

    return float(np.sum(base_expr) / n_paths)


# ---------------------------------------------------------------------------
# Total XVA — Single-pass aggregator
# ---------------------------------------------------------------------------


def compute_total_xva(
    exposure: np.ndarray,
    time_steps_yrs: np.ndarray,
    discount_factor: np.ndarray,
    counterparty_marginal_pd: np.ndarray,
    own_marginal_pd: np.ndarray,
    counterparty_lgd: float = 0.60,
    own_lgd: float = 0.60,
    funding_spread_borrow_ann: float = 0.005,
    funding_spread_deposit_ann: float = 0.0,
    capital_charge_ann: float = 0.08,
    regulatory_lgd: float = 0.75,
    im_scaling: float = 1.0,
    im_percentile: float = 99.0,
    chunk_size: int | None = None,
    use_numexpr: bool = True,
) -> dict[str, float]:
    r"""Compute CVA, DVA, FVA, KVA, and MVA in a single pass over the
    exposure matrix.

    This is significantly more efficient than calling each XVA function
    separately because the exposure splitting (positive/negative), masking,
    and time-step alignment are performed once.

    Args:
        exposure: 2-D array of shape ``(n_paths, n_dates)`` containing
            portfolio exposure values.
        time_steps_yrs: 1-D array of shape ``(n_dates,)`` containing the
            time-step sizes in years (Δt).
        discount_factor: Array of shape ``(n_paths, n_dates)`` or ``(n_dates,)``
            containing risk-free discount factors.
        counterparty_marginal_pd: Marginal default probabilities for the
            counterparty, shape ``(n_paths, n_dates)`` or ``(n_dates,)``.
        own_marginal_pd: Marginal default probabilities for the firm,
            shape ``(n_paths, n_dates)`` or ``(n_dates,)``.
        counterparty_lgd: Counterparty loss given default (decimal).
        own_lgd: Firm's own loss given default (decimal).
        funding_spread_borrow_ann: Annualised borrowing spread (decimal).
        funding_spread_deposit_ann: Annualised deposit spread (decimal).
        capital_charge_ann: Annualised regulatory capital charge rate (decimal).
        regulatory_lgd: Regulatory loss given default (decimal).
        im_scaling: Scaling factor for initial margin (default 1.0).
        im_percentile: Percentile confidence level for PFE-based IM (default 99.0).
        chunk_size: Optional batch size for memory-efficient chunked evaluation.
        use_numexpr: If True and Numexpr is available, accelerates evaluation.

    Returns:
        Dictionary with keys ``"cva"``, ``"dva"``, ``"fca"``, ``"fba"``,
        ``"fva"``, ``"kva"``, ``"mva"``, and ``"total_xva"``
        (= CVA − DVA + FVA + KVA + MVA).
    """
    exp_arr = np.asarray(exposure, dtype=np.float64)
    n_paths = exp_arr.shape[0]

    if n_paths == 0:
        return {
            "cva": 0.0,
            "dva": 0.0,
            "fca": 0.0,
            "fba": 0.0,
            "fva": 0.0,
            "kva": 0.0,
            "mva": 0.0,
            "total_xva": 0.0,
        }

    dt_arr = np.asarray(time_steps_yrs, dtype=np.float64)
    df_arr = np.asarray(discount_factor, dtype=np.float64)
    cp_pd = np.asarray(counterparty_marginal_pd, dtype=np.float64)
    own_pd = np.asarray(own_marginal_pd, dtype=np.float64)

    pos_exp = np.maximum(exp_arr, 0.0)
    neg_exp = np.maximum(-exp_arr, 0.0)

    # PFE curve for MVA
    pfe_curve = np.percentile(exp_arr, im_percentile, axis=0)
    im_matrix = np.tile(im_scaling * pfe_curve, (n_paths, 1))

    # Accumulators
    cva_sum = 0.0
    dva_sum = 0.0
    fca_sum = 0.0
    fba_sum = 0.0
    kva_sum = 0.0
    mva_sum = 0.0

    cp_lgd = counterparty_lgd
    o_lgd = own_lgd
    f_borrow = funding_spread_borrow_ann
    f_deposit = funding_spread_deposit_ann
    c_reg = capital_charge_ann
    lgd_reg = regulatory_lgd

    if use_numexpr and HAS_NUMEXPR and _ne is not None:
        for start_idx in range(0, n_paths, chunk_size or n_paths):
            end_idx = min(start_idx + (chunk_size or n_paths), n_paths)
            p_chunk = pos_exp[start_idx:end_idx]
            n_chunk = neg_exp[start_idx:end_idx]
            cp_pd_chunk = cp_pd[start_idx:end_idx] if cp_pd.ndim == 2 else cp_pd
            own_pd_chunk = own_pd[start_idx:end_idx] if own_pd.ndim == 2 else own_pd
            df_chunk = df_arr[start_idx:end_idx] if df_arr.ndim == 2 else df_arr
            im_chunk = im_matrix[start_idx:end_idx]

            cva_sum += float(
                _ne.evaluate(
                    "sum(p_chunk * cp_pd_chunk * df_chunk * cp_lgd)"
                )
            )
            dva_sum += float(
                _ne.evaluate(
                    "sum(n_chunk * own_pd_chunk * df_chunk * o_lgd)"
                )
            )
            fca_sum += float(
                _ne.evaluate(
                    "sum(p_chunk * f_borrow * df_chunk * dt_arr)"
                )
            )
            fba_sum += float(
                _ne.evaluate(
                    "sum(n_chunk * f_deposit * df_chunk * dt_arr)"
                )
            )
            kva_sum += float(
                _ne.evaluate(
                    "sum(p_chunk * df_chunk * dt_arr * c_reg * lgd_reg)"
                )
            )
            mva_sum += float(
                _ne.evaluate(
                    "sum(im_chunk * df_chunk * dt_arr * f_borrow)"
                )
            )

    else:
        for start_idx in range(0, n_paths, chunk_size or n_paths):
            end_idx = min(start_idx + (chunk_size or n_paths), n_paths)
            p_chunk = pos_exp[start_idx:end_idx]
            n_chunk = neg_exp[start_idx:end_idx]
            cp_pd_chunk = cp_pd[start_idx:end_idx] if cp_pd.ndim == 2 else cp_pd
            own_pd_chunk = own_pd[start_idx:end_idx] if own_pd.ndim == 2 else own_pd
            df_chunk = df_arr[start_idx:end_idx] if df_arr.ndim == 2 else df_arr
            im_chunk = im_matrix[start_idx:end_idx]

            cva_sum += float(np.sum(
                p_chunk * cp_pd_chunk * df_chunk * cp_lgd
            ))
            dva_sum += float(np.sum(
                n_chunk * own_pd_chunk * df_chunk * o_lgd
            ))
            fca_sum += float(np.sum(
                p_chunk * f_borrow * df_chunk * dt_arr
            ))
            fba_sum += float(np.sum(
                n_chunk * f_deposit * df_chunk * dt_arr
            ))
            kva_sum += float(np.sum(
                p_chunk * df_chunk * dt_arr * c_reg * lgd_reg
            ))
            mva_sum += float(np.sum(
                im_chunk * df_chunk * dt_arr * f_borrow
            ))

    inv_n = 1.0 / n_paths
    cva_val = cva_sum * inv_n
    dva_val = dva_sum * inv_n
    fca_val = fca_sum * inv_n
    fba_val = fba_sum * inv_n
    fva_val = fca_val + fba_val
    kva_val = kva_sum * inv_n
    mva_val = mva_sum * inv_n
    total = cva_val - dva_val + fva_val + kva_val + mva_val

    return {
        "cva": cva_val,
        "dva": dva_val,
        "fca": fca_val,
        "fba": fba_val,
        "fva": fva_val,
        "kva": kva_val,
        "mva": mva_val,
        "total_xva": total,
    }


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

    neg_exp = np.maximum(-exp_arr, 0.0)
    ene = np.mean(neg_exp, axis=0)
    ene_scalar = float(np.mean(ene))

    pfe_profiles: dict[float, np.ndarray] = {}
    result: dict[str, typing.Any] = {
        "expected_exposure": ee,
        "ee": ee,
        "expected_positive_exposure": epe,
        "epe": epe,
        "expected_negative_exposure": ene,
        "ene": ene,
        "expected_negative_exposure_scalar": ene_scalar,
        "ene_scalar": ene_scalar,
        "funding_exposure": ee - ene,
        "pfe_profiles": pfe_profiles,
    }

    max_pfe_val = 0.0
    for p in percentiles:
        p_val = p
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
