"""Statistical anomaly detection for volatility surface features.

Detects anomalies by comparing current feature values against
a 252-day rolling historical distribution. Uses z-scores and
percentile ranks to identify unusual market conditions.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from analytics.feature_extractor import SurfaceFeatures

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single feature.

    Attributes:
        feature_name: Name of the feature analyzed
        current_value: Current feature value
        historical_mean: Mean of historical distribution
        historical_std: Standard deviation of historical distribution
        z_score: Number of standard deviations from mean
        percentile: Percentile rank (0-100)
        is_anomaly: Whether this is flagged as an anomaly
        direction: 'high', 'low', or 'normal'
    """

    feature_name: str
    current_value: float
    historical_mean: float
    historical_std: float
    z_score: float
    percentile: float
    is_anomaly: bool
    direction: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "current_value": self.current_value,
            "historical_mean": self.historical_mean,
            "historical_std": self.historical_std,
            "z_score": self.z_score,
            "percentile": self.percentile,
            "is_anomaly": self.is_anomaly,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnomalyResult":
        """Create from dictionary."""
        return cls(
            feature_name=data["feature_name"],
            current_value=data["current_value"],
            historical_mean=data["historical_mean"],
            historical_std=data["historical_std"],
            z_score=data["z_score"],
            percentile=data["percentile"],
            is_anomaly=data["is_anomaly"],
            direction=data["direction"],
        )


@dataclass
class AnomalyReport:
    """Complete anomaly report for a ticker.

    Attributes:
        timestamp: When analysis was performed
        ticker: Underlying symbol
        anomalies: List of anomaly results
        summary: Summary statistics
    """

    timestamp: datetime
    ticker: str
    anomalies: list[AnomalyResult]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "ticker": self.ticker,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "summary": self.summary,
        }

    @property
    def has_anomalies(self) -> bool:
        """Check if any anomalies were detected."""
        return any(a.is_anomaly for a in self.anomalies)

    @property
    def anomaly_count(self) -> int:
        """Count of anomalies detected."""
        return sum(1 for a in self.anomalies if a.is_anomaly)


class AnomalyDetector:
    """Detects statistical anomalies in volatility surface features.

    Maintains a rolling history of features and compares current values
    against the historical distribution. Anomalies are flagged when
    z-scores exceed a threshold (default: 2.0 standard deviations).

    Example:
        >>> detector = AnomalyDetector(zscore_threshold=2.0)
        >>> detector.add_observation(features)  # Add historical data
        >>> report = detector.analyze(current_features)
        >>> for anomaly in report.anomalies:
        ...     if anomaly.is_anomaly:
        ...         print(f"{anomaly.feature_name}: z={anomaly.z_score:.2f}")
    """

    # Features to analyze
    FEATURE_NAMES = [
        "atm_vol_30d",
        "atm_vol_60d",
        "atm_vol_90d",
        "skew_25d_30d",
        "skew_25d_60d",
        "skew_25d_90d",
        "butterfly_30d",
        "butterfly_60d",
        "butterfly_90d",
        "term_slope",
    ]

    def __init__(
        self,
        zscore_threshold: float = 2.0,
        lookback_days: int = 252,
        min_observations: int = 20,
    ):
        """Initialize anomaly detector.

        Args:
            zscore_threshold: Z-score threshold for flagging anomalies
            lookback_days: Number of days for rolling history
            min_observations: Minimum observations required for analysis
        """
        self.zscore_threshold = zscore_threshold
        self.lookback_days = lookback_days
        self.min_observations = min_observations

        # Historical data storage: {ticker: {feature: [values]}}
        self._history: dict[str, dict[str, list[tuple[datetime, float]]]] = {}

    def add_observation(self, features: SurfaceFeatures) -> None:
        """Add a feature observation to historical data.

        Args:
            features: SurfaceFeatures to add to history
        """
        ticker = features.ticker

        if ticker not in self._history:
            self._history[ticker] = {name: [] for name in self.FEATURE_NAMES}

        for name in self.FEATURE_NAMES:
            value = getattr(features, name, None)
            if value is not None and not np.isnan(value):
                self._history[ticker][name].append((features.timestamp, value))

        # Prune old observations
        self._prune_history(ticker)

    def _prune_history(self, ticker: str) -> None:
        """Remove observations older than lookback period."""
        if ticker not in self._history:
            return

        cutoff = datetime.now().timestamp() - (self.lookback_days * 24 * 60 * 60)

        for name in self.FEATURE_NAMES:
            self._history[ticker][name] = [
                (ts, val)
                for ts, val in self._history[ticker][name]
                if ts.timestamp() > cutoff
            ]

    def load_history(
        self,
        ticker: str,
        feature_name: str,
        timestamps: list[datetime],
        values: list[float],
    ) -> None:
        """Load historical data directly.

        Args:
            ticker: Underlying symbol
            feature_name: Feature name
            timestamps: List of observation timestamps
            values: List of feature values
        """
        if ticker not in self._history:
            self._history[ticker] = {name: [] for name in self.FEATURE_NAMES}

        if feature_name not in self.FEATURE_NAMES:
            logger.warning(f"Unknown feature: {feature_name}")
            return

        self._history[ticker][feature_name] = list(zip(timestamps, values))
        self._prune_history(ticker)

    def get_historical_values(
        self, ticker: str, feature_name: str
    ) -> NDArray[np.float64]:
        """Get historical values for a feature.

        Args:
            ticker: Underlying symbol
            feature_name: Feature name

        Returns:
            Array of historical values
        """
        if ticker not in self._history:
            return np.array([])

        if feature_name not in self._history[ticker]:
            return np.array([])

        values = [v for _, v in self._history[ticker][feature_name]]
        return np.array(values)

    def _analyze_feature(
        self,
        feature_name: str,
        current_value: float,
        historical_values: NDArray[np.float64],
    ) -> AnomalyResult:
        """Analyze a single feature for anomalies.

        Args:
            feature_name: Name of the feature
            current_value: Current feature value
            historical_values: Historical values array

        Returns:
            AnomalyResult for this feature
        """
        if len(historical_values) < self.min_observations:
            # Insufficient history - cannot determine anomaly
            return AnomalyResult(
                feature_name=feature_name,
                current_value=current_value,
                historical_mean=np.nan,
                historical_std=np.nan,
                z_score=0.0,
                percentile=50.0,
                is_anomaly=False,
                direction="normal",
            )

        mean = float(np.mean(historical_values))
        std = float(np.std(historical_values, ddof=1))

        # Avoid division by zero
        if std < 1e-10:
            z_score = 0.0
        else:
            z_score = (current_value - mean) / std

        # Percentile rank
        percentile = float(np.sum(historical_values < current_value) / len(historical_values) * 100)

        # Determine direction and anomaly status
        is_anomaly = abs(z_score) > self.zscore_threshold

        if z_score > self.zscore_threshold:
            direction = "high"
        elif z_score < -self.zscore_threshold:
            direction = "low"
        else:
            direction = "normal"

        return AnomalyResult(
            feature_name=feature_name,
            current_value=current_value,
            historical_mean=mean,
            historical_std=std,
            z_score=z_score,
            percentile=percentile,
            is_anomaly=is_anomaly,
            direction=direction,
        )

    def analyze(self, features: SurfaceFeatures) -> AnomalyReport:
        """Analyze features for anomalies against historical distribution.

        Args:
            features: Current SurfaceFeatures to analyze

        Returns:
            AnomalyReport with all analysis results

        Example:
            >>> detector = AnomalyDetector()
            >>> # Load historical data...
            >>> report = detector.analyze(current_features)
            >>> print(f"Anomalies detected: {report.anomaly_count}")
        """
        ticker = features.ticker
        anomalies = []

        for name in self.FEATURE_NAMES:
            current = getattr(features, name, None)
            if current is None or np.isnan(current):
                continue

            historical = self.get_historical_values(ticker, name)
            result = self._analyze_feature(name, current, historical)
            anomalies.append(result)

        # Summary statistics
        anomaly_list = [a for a in anomalies if a.is_anomaly]
        high_anomalies = [a for a in anomaly_list if a.direction == "high"]
        low_anomalies = [a for a in anomaly_list if a.direction == "low"]

        summary = {
            "total_features_analyzed": len(anomalies),
            "total_anomalies": len(anomaly_list),
            "high_anomalies": len(high_anomalies),
            "low_anomalies": len(low_anomalies),
            "most_extreme_zscore": max((abs(a.z_score) for a in anomalies), default=0),
            "anomaly_features": [a.feature_name for a in anomaly_list],
        }

        logger.info(
            f"Anomaly analysis for {ticker}: {len(anomaly_list)} anomalies detected "
            f"({len(high_anomalies)} high, {len(low_anomalies)} low)"
        )

        return AnomalyReport(
            timestamp=datetime.now(),
            ticker=ticker,
            anomalies=anomalies,
            summary=summary,
        )

    def get_feature_distribution(
        self,
        ticker: str,
        feature_name: str,
        n_bins: int = 50,
    ) -> dict[str, Any]:
        """Get histogram distribution for a feature.

        Args:
            ticker: Underlying symbol
            feature_name: Feature name
            n_bins: Number of histogram bins

        Returns:
            Dictionary with histogram data
        """
        values = self.get_historical_values(ticker, feature_name)

        if len(values) == 0:
            return {"bins": [], "counts": [], "mean": None, "std": None}

        counts, bin_edges = np.histogram(values, bins=n_bins)

        return {
            "bin_edges": bin_edges.tolist(),
            "counts": counts.tolist(),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n_observations": len(values),
        }

    def get_zscore_bands(
        self,
        ticker: str,
        feature_name: str,
    ) -> dict[str, float]:
        """Get z-score bands for a feature.

        Returns mean and ±1σ, ±2σ, ±3σ levels.

        Args:
            ticker: Underlying symbol
            feature_name: Feature name

        Returns:
            Dictionary with band levels
        """
        values = self.get_historical_values(ticker, feature_name)

        if len(values) < self.min_observations:
            return {}

        mean = float(np.mean(values))
        std = float(np.std(values))

        return {
            "mean": mean,
            "std": std,
            "plus_1_sigma": mean + std,
            "minus_1_sigma": mean - std,
            "plus_2_sigma": mean + 2 * std,
            "minus_2_sigma": mean - 2 * std,
            "plus_3_sigma": mean + 3 * std,
            "minus_3_sigma": mean - 3 * std,
        }
