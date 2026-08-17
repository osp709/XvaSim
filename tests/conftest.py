"""Pytest configuration, shared fixtures, and hooks for XvaSim test suite."""

import numpy as np
import pytest

from xvasim.models import (
    HullWhite1FModel,
    LGMModel,
    TwoCurrencyFXModel,
)


@pytest.fixture
def flat_discount_curve() -> tuple[np.ndarray, np.ndarray]:
    """Provide a standard flat 3% discount curve."""
    tenors = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    dfs = np.exp(-0.03 * tenors)
    return tenors, dfs


@pytest.fixture
def sample_hw_model(
    flat_discount_curve: tuple[np.ndarray, np.ndarray],
) -> HullWhite1FModel:
    """Provide an initialized 1-factor Hull-White model."""
    tenors, dfs = flat_discount_curve
    return HullWhite1FModel(
        a_ann=0.03,
        sigma_ann=0.012,
        discount_curve_yrs=tenors,
        discount_factors=dfs,
    )


@pytest.fixture
def sample_lgm_model(flat_discount_curve: tuple[np.ndarray, np.ndarray]) -> LGMModel:
    """Provide an initialized LGM interest rate model."""
    tenors, dfs = flat_discount_curve
    return LGMModel(
        kappa_ann=0.03,
        sigma_grid_yrs=np.array([1.0, 3.0, 5.0, 10.0, 30.0]),
        sigma_values_ann=np.array([0.010, 0.011, 0.012, 0.013, 0.012]),
        discount_curve_yrs=tenors,
        discount_factors=dfs,
    )


@pytest.fixture
def sample_two_currency_fx(
    sample_hw_model: HullWhite1FModel,
    flat_discount_curve: tuple[np.ndarray, np.ndarray],
) -> TwoCurrencyFXModel:
    """Provide a two-currency FX model."""
    tenors, _ = flat_discount_curve
    foreign_model = HullWhite1FModel(
        a_ann=0.04,
        sigma_ann=0.015,
        discount_curve_yrs=tenors,
        discount_factors=np.exp(-0.02 * tenors),
    )
    corr = np.array(
        [
            [1.0, 0.3, 0.2],
            [0.3, 1.0, -0.1],
            [0.2, -0.1, 1.0],
        ]
    )
    return TwoCurrencyFXModel(
        domestic_ir_model=sample_hw_model,
        foreign_ir_model=foreign_model,
        spot_fx=1.20,
        fx_vol_ann=0.15,
        correlation_matrix=corr,
    )
