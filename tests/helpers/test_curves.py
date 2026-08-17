"""Standardized market curves for testing interest rates, credit, and inflation."""

import numpy as np


def get_standard_discount_curve(
    flat_rate_ann: float = 0.03,
    max_tenor_yrs: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return standard test discount curve (tenors and discount factors)."""
    tenors = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
    tenors = tenors[tenors <= max_tenor_yrs]
    dfs = np.exp(-flat_rate_ann * tenors)
    return tenors, dfs


def get_inverted_discount_curve() -> tuple[np.ndarray, np.ndarray]:
    """Return inverted test discount curve (higher short rates, lower long rates)."""
    tenors = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    rates = np.array([0.05, 0.048, 0.045, 0.040, 0.035, 0.030, 0.028])
    dfs = np.exp(-rates * tenors)
    return tenors, dfs


def get_humped_discount_curve() -> tuple[np.ndarray, np.ndarray]:
    """Return humped test discount curve."""
    tenors = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0])
    rates = 0.02 + 0.025 * tenors / (1.0 + 0.1 * tenors**2)
    dfs = np.exp(-rates * tenors)
    return tenors, dfs


def get_standard_credit_curve() -> tuple[np.ndarray, np.ndarray]:
    """Return standard test credit spread curve."""
    tenors = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    spreads = np.array([0.010, 0.012, 0.015, 0.018, 0.022, 0.025, 0.028])
    return tenors, spreads


def get_standard_inflation_curve(
    base_cpi: float = 100.0,
    expected_inflation_ann: float = 0.025,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return test nominal curve, real curve, and base CPI."""
    tenors = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0])
    nom_dfs = np.exp(-0.035 * tenors)
    real_dfs = np.exp(-(0.035 - expected_inflation_ann) * tenors)
    return tenors, nom_dfs, real_dfs
