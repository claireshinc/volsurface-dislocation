"""Anomaly detection API endpoints.

Provides endpoints for:
- Running anomaly detection
- Getting anomaly history
- Generating trade ideas
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from analytics.svi_fitter import SVIFitter
from analytics.feature_extractor import FeatureExtractor
from analytics.anomaly_detector import AnomalyDetector
from analytics.trade_ideas import TradeIdeaGenerator
from data.collector import OptionsDataCollector, create_synthetic_chain

logger = logging.getLogger(__name__)
router = APIRouter()

# Global detector instance (in production, use dependency injection)
_detector = AnomalyDetector()


def _get_detector() -> AnomalyDetector:
    """Get the global anomaly detector."""
    return _detector


@router.get("/current/{ticker}")
async def get_current_anomalies(
    ticker: str,
    zscore_threshold: float = Query(
        default=2.0, ge=1.0, le=5.0, description="Z-score threshold"
    ),
    generate_ideas: bool = Query(
        default=True, description="Generate trade ideas"
    ),
) -> dict[str, Any]:
    """Run anomaly detection on current surface.

    Fetches current options data, fits surface, extracts features,
    and compares against historical distribution.

    Args:
        ticker: Underlying symbol
        zscore_threshold: Z-score threshold for anomaly flagging
        generate_ideas: Whether to generate trade suggestions

    Returns:
        Anomaly report with optional trade ideas
    """
    ticker = ticker.upper()
    logger.info(f"Running anomaly detection for {ticker}")

    # Get current surface data
    try:
        collector = OptionsDataCollector()
        chain = collector.get_chain(ticker)
    except Exception as e:
        logger.warning(f"Could not fetch real data: {e}")
        chain = create_synthetic_chain(ticker)

    # Fit SVI
    fitter = SVIFitter()
    params_list = []

    for expiry in chain.expiries[:5]:
        try:
            k, w, T = collector.prepare_for_svi_fitting(chain, expiry)
            params = fitter.fit(k, w, T)
            params_list.append(params)
        except Exception as e:
            logger.warning(f"Could not fit {expiry}: {e}")

    if not params_list:
        raise HTTPException(400, "Could not fit any expiries")

    # Extract features
    extractor = FeatureExtractor(chain.spot_price)
    features = extractor.extract(params_list, ticker)

    # Run anomaly detection
    detector = _get_detector()
    detector.zscore_threshold = zscore_threshold
    report = detector.analyze(features)

    result = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "spot_price": chain.spot_price,
        "features": features.to_dict(),
        "report": report.to_dict(),
        "has_anomalies": report.has_anomalies,
        "anomaly_count": report.anomaly_count,
    }

    # Generate trade ideas if requested
    if generate_ideas and report.has_anomalies:
        generator = TradeIdeaGenerator(min_zscore=zscore_threshold)
        ideas = generator.generate(report)
        result["trade_ideas"] = [idea.to_dict() for idea in ideas]
        result["trade_summary"] = generator.summarize_ideas(ideas)

    return result


@router.post("/load-history/{ticker}")
async def load_synthetic_history(
    ticker: str,
    days: int = Query(default=252, ge=30, le=500, description="Days of history"),
) -> dict[str, Any]:
    """Load synthetic historical data for anomaly detection.

    This creates simulated historical feature data for testing.
    In production, this would be replaced with real historical data.

    Args:
        ticker: Underlying symbol
        days: Number of days of synthetic history

    Returns:
        Confirmation of loaded data
    """
    import numpy as np
    from datetime import timedelta

    ticker = ticker.upper()
    detector = _get_detector()

    # Generate synthetic historical data
    np.random.seed(42)  # Reproducible

    base_values = {
        "atm_vol_30d": 0.18,
        "atm_vol_60d": 0.19,
        "atm_vol_90d": 0.20,
        "skew_25d_30d": 0.04,
        "skew_25d_60d": 0.035,
        "skew_25d_90d": 0.03,
        "butterfly_30d": 0.01,
        "butterfly_60d": 0.012,
        "butterfly_90d": 0.015,
        "term_slope": 0.05,
    }

    volatility_of_features = {
        "atm_vol_30d": 0.03,
        "atm_vol_60d": 0.025,
        "atm_vol_90d": 0.02,
        "skew_25d_30d": 0.015,
        "skew_25d_60d": 0.012,
        "skew_25d_90d": 0.01,
        "butterfly_30d": 0.005,
        "butterfly_60d": 0.004,
        "butterfly_90d": 0.003,
        "term_slope": 0.03,
    }

    now = datetime.now()
    loaded_counts = {}

    for feature_name, base in base_values.items():
        vol = volatility_of_features[feature_name]
        timestamps = []
        values = []

        for i in range(days):
            ts = now - timedelta(days=days - i)
            # Add some mean reversion and noise
            noise = np.random.normal(0, vol)
            value = base + noise
            timestamps.append(ts)
            values.append(value)

        detector.load_history(ticker, feature_name, timestamps, values)
        loaded_counts[feature_name] = len(values)

    return {
        "ticker": ticker,
        "days_loaded": days,
        "features_loaded": loaded_counts,
        "message": f"Loaded {days} days of synthetic history for {ticker}",
    }


@router.get("/history/{ticker}")
async def get_anomaly_history(
    ticker: str,
    feature: str = Query(default="atm_vol_30d", description="Feature to analyze"),
) -> dict[str, Any]:
    """Get historical distribution for a feature.

    Args:
        ticker: Underlying symbol
        feature: Feature name to get history for

    Returns:
        Historical distribution statistics
    """
    ticker = ticker.upper()
    detector = _get_detector()

    distribution = detector.get_feature_distribution(ticker, feature)
    bands = detector.get_zscore_bands(ticker, feature)

    return {
        "ticker": ticker,
        "feature": feature,
        "distribution": distribution,
        "zscore_bands": bands,
    }


@router.get("/trade-ideas/{ticker}")
async def get_trade_ideas(
    ticker: str,
    confidence: str = Query(
        default="all",
        description="Filter by confidence: all, high, medium, low"
    ),
) -> dict[str, Any]:
    """Get trade ideas for current anomalies.

    Args:
        ticker: Underlying symbol
        confidence: Confidence level filter

    Returns:
        List of trade ideas
    """
    # First run anomaly detection
    result = await get_current_anomalies(ticker, generate_ideas=True)

    ideas = result.get("trade_ideas", [])

    # Filter by confidence if requested
    if confidence != "all":
        ideas = [i for i in ideas if i["confidence"] == confidence]

    return {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "total_ideas": len(ideas),
        "trade_ideas": ideas,
        "summary": result.get("trade_summary", {}),
    }


@router.get("/compare/{ticker}")
async def compare_features(
    ticker: str,
    feature: str = Query(default="atm_vol_30d", description="Feature to compare"),
) -> dict[str, Any]:
    """Compare current feature value to historical distribution.

    Args:
        ticker: Underlying symbol
        feature: Feature name

    Returns:
        Comparison showing current vs historical
    """
    ticker = ticker.upper()

    # Get current features
    try:
        collector = OptionsDataCollector()
        chain = collector.get_chain(ticker)
    except Exception:
        chain = create_synthetic_chain(ticker)

    fitter = SVIFitter()
    params_list = []

    for expiry in chain.expiries[:5]:
        try:
            k, w, T = collector.prepare_for_svi_fitting(chain, expiry)
            params_list.append(fitter.fit(k, w, T))
        except Exception:
            pass

    if not params_list:
        raise HTTPException(400, "Could not fit surface")

    extractor = FeatureExtractor(chain.spot_price)
    features = extractor.extract(params_list, ticker)

    current_value = getattr(features, feature, None)
    if current_value is None:
        raise HTTPException(400, f"Feature {feature} not available")

    # Get historical data
    detector = _get_detector()
    historical = detector.get_historical_values(ticker, feature)
    bands = detector.get_zscore_bands(ticker, feature)

    if len(historical) == 0:
        return {
            "ticker": ticker,
            "feature": feature,
            "current_value": current_value,
            "message": "No historical data. Use POST /load-history to add synthetic history.",
        }

    import numpy as np

    z_score = (current_value - bands["mean"]) / bands["std"] if bands.get("std", 0) > 0 else 0
    percentile = float(np.sum(historical < current_value) / len(historical) * 100)

    return {
        "ticker": ticker,
        "feature": feature,
        "current_value": current_value,
        "historical": {
            "mean": bands.get("mean"),
            "std": bands.get("std"),
            "min": float(np.min(historical)),
            "max": float(np.max(historical)),
            "n_observations": len(historical),
        },
        "analysis": {
            "z_score": z_score,
            "percentile": percentile,
            "is_anomaly": abs(z_score) > 2.0,
            "direction": "high" if z_score > 2 else "low" if z_score < -2 else "normal",
        },
        "bands": bands,
    }
