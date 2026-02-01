"""Analytics modules for volatility surface fitting and analysis.

This package provides quantitative analytics using QuantLib:

QuantLib-Based Modules:
- quantlib_utils: Core QuantLib setup, processes, and engines
- black_scholes: Option pricing and Greeks using QuantLib
- quantlib_surface: Volatility surface construction with QuantLib

Custom Implementations:
- svi_fitter: SVI model fitting (QuantLib doesn't include SVI)
- feature_extractor: Extract surface features
- anomaly_detector: Statistical anomaly detection
- trade_ideas: Generate trade suggestions

QuantLib Architecture Overview:
==============================

    Market Data (Quotes, Handles)
            ↓
    Term Structures (Yield, Vol)
            ↓
    Stochastic Process (BSM, Heston)
            ↓
    Pricing Engine (Analytic, FD, MC)
            ↓
    Instrument (Option, Swap)
            ↓
    NPV + Greeks

Example:
    >>> from analytics import BlackScholes, SVIFitter, VolatilitySurface
    >>>
    >>> # Price an option with QuantLib
    >>> bs = BlackScholes(spot=100, rate=0.05)
    >>> price = bs.price(strike=100, expiry=0.25, vol=0.2, option_type="call")
    >>>
    >>> # Fit SVI model
    >>> fitter = SVIFitter()
    >>> params = fitter.fit(log_moneyness, total_variance, expiry_years=0.25)
    >>>
    >>> # Build vol surface with QuantLib
    >>> surface = VolatilitySurface(spot=100)
    >>> surface.build_from_market_data(strikes, expiries, vols)
"""

from analytics.svi_fitter import SVIFitter, SVIParams
from analytics.black_scholes import BlackScholes, Greeks, AmericanOptionPricer
from analytics.feature_extractor import FeatureExtractor, SurfaceFeatures
from analytics.anomaly_detector import AnomalyDetector, AnomalyResult
from analytics.trade_ideas import TradeIdeaGenerator, TradeIdea
from analytics.quantlib_utils import (
    QuantLibSetup,
    MarketData,
    PricingEngineFactory,
    create_quote,
    create_flat_yield_curve,
    create_flat_vol_surface,
    create_black_scholes_process,
    create_vanilla_option,
    create_american_option,
)
from analytics.quantlib_surface import (
    VolatilitySurface,
    VolatilitySurfaceAnalytics,
    VolSurfacePoint,
)

__all__ = [
    # SVI (custom implementation)
    "SVIFitter",
    "SVIParams",
    # Black-Scholes (QuantLib)
    "BlackScholes",
    "Greeks",
    "AmericanOptionPricer",
    # Feature extraction
    "FeatureExtractor",
    "SurfaceFeatures",
    # Anomaly detection
    "AnomalyDetector",
    "AnomalyResult",
    # Trade ideas
    "TradeIdeaGenerator",
    "TradeIdea",
    # QuantLib utilities
    "QuantLibSetup",
    "MarketData",
    "PricingEngineFactory",
    "create_quote",
    "create_flat_yield_curve",
    "create_flat_vol_surface",
    "create_black_scholes_process",
    "create_vanilla_option",
    "create_american_option",
    # Volatility surface (QuantLib)
    "VolatilitySurface",
    "VolatilitySurfaceAnalytics",
    "VolSurfacePoint",
]
