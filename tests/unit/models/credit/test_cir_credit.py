"""Tests for the CIR credit and hazard rate model."""

import unittest

import numpy as np

from xvasim.models.credit.cir import CIRHazardRateModel, CIRParams


class TestCIRHazardRateModel(unittest.TestCase):
    """Unit tests for CIRHazardRateModel initialization, survival probabilities, and calibration."""

    def setUp(self) -> None:
        self.model = CIRHazardRateModel(
            kappa_ann=0.5,
            theta_ann=0.03,
            sigma_ann=0.10,
            lambda_0_ann=0.02,
        )

    def test_init_with_params_dataclass(self) -> None:
        """Verify model initialization via pre-built CIRParams."""
        params = CIRParams(
            kappa_ann=0.4,
            theta_ann=0.025,
            sigma_ann=0.08,
            lambda_0_ann=0.015,
        )
        mdl = CIRHazardRateModel(params=params)
        self.assertEqual(mdl.model_name, "cir")
        self.assertEqual(mdl.kappa_ann, 0.4)
        self.assertEqual(mdl.theta_ann, 0.025)
        self.assertEqual(mdl.sigma_ann, 0.08)
        self.assertEqual(mdl.lambda_0_ann, 0.015)
        self.assertIs(mdl.params, params)

    def test_survival_probability_properties(self) -> None:
        """Verify survival probability properties: P(0)=1, monotonically decreasing in (0, 1]."""
        tenors = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
        surv = self.model.survival_probability(tenors)

        self.assertAlmostEqual(float(surv[0]), 1.0, places=10)
        self.assertTrue(np.all(surv > 0.0))
        self.assertTrue(np.all(surv <= 1.0))
        for i in range(len(surv) - 1):
            self.assertGreater(surv[i], surv[i + 1])

    def test_calibrate_from_spreads(self) -> None:
        """Verify CIR credit model calibration from market spread curves."""
        tenors = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        spreads = np.array([0.010, 0.015, 0.020, 0.025, 0.030])

        calibrated = CIRHazardRateModel.calibrate_from_spreads(
            credit_spreads_ann=spreads,
            tenors_yrs=tenors,
        )
        self.assertIsInstance(calibrated, CIRHazardRateModel)
        self.assertGreater(calibrated.kappa_ann, 0.0)
        self.assertGreater(calibrated.theta_ann, 0.0)
        self.assertGreater(calibrated.sigma_ann, 0.0)
        self.assertGreater(calibrated.lambda_0_ann, 0.0)

    def test_calibration_failure_raises_runtime_error(self) -> None:
        """Calibration failure on NaN spreads raises RuntimeError."""
        tenors = np.array([np.nan, np.nan])
        spreads = np.array([0.01, 0.02])
        with self.assertRaises(RuntimeError):
            CIRHazardRateModel.calibrate_from_spreads(
                credit_spreads_ann=spreads,
                tenors_yrs=tenors,
            )


if __name__ == "__main__":
    unittest.main()
