"""Tests for anomaly detection."""

import numpy as np
import pytest
from datetime import datetime, timedelta

from analytics.anomaly_detector import AnomalyDetector, AnomalyResult, AnomalyReport
from analytics.feature_extractor import SurfaceFeatures


class TestAnomalyDetector:
    """Tests for anomaly detection."""

    @pytest.fixture
    def detector(self):
        return AnomalyDetector(zscore_threshold=2.0, min_observations=10)

    @pytest.fixture
    def sample_features(self):
        return SurfaceFeatures(
            timestamp=datetime.now(),
            ticker="TEST",
            spot=100.0,
            atm_vol_30d=0.20,
            atm_vol_60d=0.21,
            atm_vol_90d=0.22,
            skew_25d_30d=0.04,
            butterfly_30d=0.01,
            term_slope=0.05,
        )

    def test_analyze_no_history(self, detector, sample_features):
        """Test analysis with no historical data."""
        report = detector.analyze(sample_features)

        assert isinstance(report, AnomalyReport)
        assert report.ticker == "TEST"
        # Should not flag anomalies without history
        assert not report.has_anomalies

    def test_analyze_with_history(self, detector, sample_features):
        """Test analysis with sufficient history."""
        ticker = "TEST"

        # Load history with current value being an outlier
        np.random.seed(42)
        timestamps = [datetime.now() - timedelta(days=i) for i in range(100)]
        values = np.random.normal(0.15, 0.02, 100)  # Mean 15%, current is 20%

        detector.load_history(ticker, "atm_vol_30d", timestamps, values.tolist())

        report = detector.analyze(sample_features)

        # Should detect anomaly (20% is >2 std above 15%)
        atm_anomaly = next(
            (a for a in report.anomalies if a.feature_name == "atm_vol_30d"),
            None
        )
        assert atm_anomaly is not None
        assert atm_anomaly.is_anomaly
        assert atm_anomaly.direction == "high"

    def test_zscore_calculation(self, detector):
        """Test z-score calculation."""
        historical = np.array([10, 12, 11, 9, 10, 11, 10, 12, 11, 10])
        current = 15  # Well above mean

        result = detector._analyze_feature("test_feature", current, historical)

        # Mean ≈ 10.6, std ≈ 1
        assert result.z_score > 2
        assert result.is_anomaly
        assert result.direction == "high"

    def test_percentile_calculation(self, detector):
        """Test percentile calculation."""
        historical = np.arange(100)  # 0-99
        current = 50

        result = detector._analyze_feature("test_feature", current, historical)

        # 50 is at 50th percentile
        assert np.isclose(result.percentile, 50, atol=2)

    def test_low_anomaly_detection(self, detector, sample_features):
        """Test detection of low anomalies."""
        ticker = "TEST"

        # Load history with current being low outlier
        timestamps = [datetime.now() - timedelta(days=i) for i in range(100)]
        values = np.random.normal(0.30, 0.02, 100)  # Mean 30%, current is 20%

        detector.load_history(ticker, "atm_vol_30d", timestamps, values.tolist())

        report = detector.analyze(sample_features)

        atm_anomaly = next(
            (a for a in report.anomalies if a.feature_name == "atm_vol_30d"),
            None
        )
        assert atm_anomaly is not None
        assert atm_anomaly.is_anomaly
        assert atm_anomaly.direction == "low"


class TestAnomalyResult:
    """Tests for AnomalyResult."""

    def test_to_dict(self):
        """Test serialization."""
        result = AnomalyResult(
            feature_name="atm_vol_30d",
            current_value=0.25,
            historical_mean=0.20,
            historical_std=0.02,
            z_score=2.5,
            percentile=95.0,
            is_anomaly=True,
            direction="high",
        )

        d = result.to_dict()
        assert d["feature_name"] == "atm_vol_30d"
        assert d["z_score"] == 2.5
        assert d["is_anomaly"] is True

    def test_from_dict(self):
        """Test deserialization."""
        d = {
            "feature_name": "skew_25d_30d",
            "current_value": 0.05,
            "historical_mean": 0.04,
            "historical_std": 0.005,
            "z_score": 2.0,
            "percentile": 97.7,
            "is_anomaly": True,
            "direction": "high",
        }

        result = AnomalyResult.from_dict(d)
        assert result.feature_name == "skew_25d_30d"
        assert result.z_score == 2.0


class TestAnomalyReport:
    """Tests for AnomalyReport."""

    def test_has_anomalies(self):
        """Test has_anomalies property."""
        # Report with anomaly
        report_with = AnomalyReport(
            timestamp=datetime.now(),
            ticker="TEST",
            anomalies=[
                AnomalyResult("f1", 1, 0.5, 0.1, 5.0, 99, True, "high"),
            ],
            summary={},
        )
        assert report_with.has_anomalies

        # Report without anomaly
        report_without = AnomalyReport(
            timestamp=datetime.now(),
            ticker="TEST",
            anomalies=[
                AnomalyResult("f1", 0.5, 0.5, 0.1, 0.0, 50, False, "normal"),
            ],
            summary={},
        )
        assert not report_without.has_anomalies

    def test_anomaly_count(self):
        """Test anomaly counting."""
        report = AnomalyReport(
            timestamp=datetime.now(),
            ticker="TEST",
            anomalies=[
                AnomalyResult("f1", 1, 0.5, 0.1, 5.0, 99, True, "high"),
                AnomalyResult("f2", 0.5, 0.5, 0.1, 0.0, 50, False, "normal"),
                AnomalyResult("f3", 0.1, 0.5, 0.1, -4.0, 1, True, "low"),
            ],
            summary={},
        )

        assert report.anomaly_count == 2


class TestFeatureDistribution:
    """Tests for distribution statistics."""

    @pytest.fixture
    def detector_with_data(self):
        detector = AnomalyDetector()
        np.random.seed(42)

        timestamps = [datetime.now() - timedelta(days=i) for i in range(252)]
        values = np.random.normal(0.20, 0.03, 252)

        detector.load_history("TEST", "atm_vol_30d", timestamps, values.tolist())
        return detector

    def test_get_distribution(self, detector_with_data):
        """Test distribution statistics."""
        dist = detector_with_data.get_feature_distribution("TEST", "atm_vol_30d")

        assert "mean" in dist
        assert "std" in dist
        assert "min" in dist
        assert "max" in dist
        assert dist["n_observations"] == 252

    def test_get_zscore_bands(self, detector_with_data):
        """Test z-score band calculation."""
        bands = detector_with_data.get_zscore_bands("TEST", "atm_vol_30d")

        assert "mean" in bands
        assert "plus_2_sigma" in bands
        assert "minus_2_sigma" in bands

        # Bands should be symmetric around mean
        assert np.isclose(
            bands["plus_2_sigma"] - bands["mean"],
            bands["mean"] - bands["minus_2_sigma"],
            rtol=1e-6
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
