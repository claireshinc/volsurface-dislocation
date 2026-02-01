"""Surface-related API endpoints.

Provides endpoints for:
- Fetching and fitting volatility surfaces
- Retrieving historical surface data
- Getting surface features
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from analytics.svi_fitter import SVIFitter, SVIParams
from analytics.black_scholes import BlackScholes
from analytics.feature_extractor import FeatureExtractor
from data.collector import OptionsDataCollector, create_synthetic_chain
from data.schemas import (
    SurfaceRequest,
    SurfaceResponse,
    SurfaceFeaturesSchema,
    SVIParamsSchema,
    PriceRequest,
    GreeksSchema,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/current/{ticker}")
async def get_current_surface(
    ticker: str,
    rate: float = Query(default=0.05, description="Risk-free rate"),
    dividend_yield: float = Query(default=0.0, description="Dividend yield"),
    use_synthetic: bool = Query(default=False, description="Use synthetic data"),
) -> dict[str, Any]:
    """Get current volatility surface for a ticker.

    Fetches options chain, fits SVI model, and extracts features.

    Args:
        ticker: Underlying symbol (e.g., SPY, AAPL)
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        use_synthetic: Force use of synthetic data (for testing)

    Returns:
        Fitted surface with SVI parameters and features
    """
    ticker = ticker.upper()
    logger.info(f"Fetching current surface for {ticker}")

    # Get options chain
    if use_synthetic:
        chain = create_synthetic_chain(ticker)
    else:
        try:
            collector = OptionsDataCollector()
            chain = collector.get_chain(ticker)
        except Exception as e:
            logger.warning(f"Could not fetch real data: {e}, using synthetic")
            chain = create_synthetic_chain(ticker)

    # Fit SVI to each expiry
    fitter = SVIFitter()
    collector = OptionsDataCollector()
    params_list: list[SVIParams] = []
    params_by_expiry: dict[int, dict] = {}

    for expiry in chain.expiries[:6]:  # Limit expiries
        try:
            k, w, T = collector.prepare_for_svi_fitting(
                chain, expiry, rate, dividend_yield
            )
            params = fitter.fit(k, w, T)
            params_list.append(params)

            days = int(T * 365)
            params_by_expiry[days] = params.to_dict()
        except Exception as e:
            logger.warning(f"Could not fit expiry {expiry}: {e}")

    if not params_list:
        raise HTTPException(status_code=400, detail="Could not fit any expiries")

    # Extract features
    extractor = FeatureExtractor(chain.spot_price, rate, dividend_yield)
    features = extractor.extract(params_list, ticker)

    # Get smile data at standard deltas
    smile_data = {}
    for params in params_list:
        days = int(params.expiry_years * 365)
        try:
            smile_data[days] = extractor.extract_smile_points(params)
        except Exception as e:
            logger.warning(f"Could not extract smile for {days}d: {e}")

    return {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "spot_price": chain.spot_price,
        "features": features.to_dict(),
        "params_by_expiry": params_by_expiry,
        "smile_data": smile_data,
        "expiries": [e.isoformat() for e in chain.expiries],
    }


@router.get("/history/{ticker}")
async def get_surface_history(
    ticker: str,
    days: int = Query(default=30, ge=1, le=365, description="Days of history"),
) -> dict[str, Any]:
    """Get historical surface features for a ticker.

    Args:
        ticker: Underlying symbol
        days: Number of days of history to retrieve

    Returns:
        Historical feature time series
    """
    ticker = ticker.upper()

    # In a production system, this would query the database
    # For now, return a message indicating this requires historical data
    return {
        "ticker": ticker,
        "message": "Historical data requires database population. Run the collector to build history.",
        "requested_days": days,
        "tip": "Use POST /api/v1/surface/collect/{ticker} to start collecting data",
    }


@router.get("/smile/{ticker}")
async def get_smile(
    ticker: str,
    days: int = Query(default=30, description="Target expiry in days"),
    n_points: int = Query(default=21, ge=5, le=101, description="Number of points"),
) -> dict[str, Any]:
    """Get volatility smile for a specific tenor.

    Args:
        ticker: Underlying symbol
        days: Target expiry in days
        n_points: Number of strike points to return

    Returns:
        Volatility smile data
    """
    ticker = ticker.upper()

    # Get current surface
    try:
        collector = OptionsDataCollector()
        chain = collector.get_chain(ticker)
    except Exception:
        chain = create_synthetic_chain(ticker)

    # Find closest expiry
    import numpy as np
    from datetime import date

    target_date = date.fromordinal(date.today().toordinal() + days)
    expiries = chain.expiries
    closest = min(expiries, key=lambda e: abs((e - target_date).days))

    # Fit SVI
    fitter = SVIFitter()
    k, w, T = collector.prepare_for_svi_fitting(chain, closest)
    params = fitter.fit(k, w, T)

    # Generate smile points
    k_range = np.linspace(-0.3, 0.3, n_points)
    ivs = SVIFitter.svi_implied_vol(
        k_range, params.a, params.b, params.rho, params.m, params.sigma, T
    )

    # Convert to strikes
    forward = chain.spot_price * np.exp(0.05 * T)  # Simple forward calc
    strikes = forward * np.exp(k_range)

    return {
        "ticker": ticker,
        "expiry_days": int(T * 365),
        "expiry_date": closest.isoformat(),
        "spot": chain.spot_price,
        "forward": float(forward),
        "svi_params": params.to_dict(),
        "smile": {
            "log_moneyness": k_range.tolist(),
            "strikes": strikes.tolist(),
            "implied_vol": ivs.tolist(),
        },
    }


@router.post("/price")
async def calculate_price(request: PriceRequest) -> dict[str, Any]:
    """Calculate Black-Scholes option price and Greeks.

    Args:
        request: Pricing request with spot, strike, expiry, vol, etc.

    Returns:
        Option price and all Greeks
    """
    bs = BlackScholes(
        spot=request.spot,
        rate=request.rate,
        dividend_yield=request.dividend_yield,
    )

    price = bs.price(
        strike=request.strike,
        expiry=request.expiry_years,
        vol=request.vol,
        option_type=request.option_type,
    )

    greeks = bs.greeks(
        strike=request.strike,
        expiry=request.expiry_years,
        vol=request.vol,
        option_type=request.option_type,
    )

    return {
        "price": price,
        "greeks": greeks.to_dict(),
        "inputs": {
            "spot": request.spot,
            "strike": request.strike,
            "expiry_years": request.expiry_years,
            "vol": request.vol,
            "rate": request.rate,
            "dividend_yield": request.dividend_yield,
            "option_type": request.option_type,
        },
    }


@router.get("/term-structure/{ticker}")
async def get_term_structure(ticker: str) -> dict[str, Any]:
    """Get ATM volatility term structure.

    Args:
        ticker: Underlying symbol

    Returns:
        ATM vol at each available expiry
    """
    ticker = ticker.upper()

    try:
        collector = OptionsDataCollector()
        chain = collector.get_chain(ticker)
    except Exception:
        chain = create_synthetic_chain(ticker)

    fitter = SVIFitter()
    term_structure = []

    for expiry in chain.expiries[:8]:
        try:
            k, w, T = collector.prepare_for_svi_fitting(chain, expiry)
            params = fitter.fit(k, w, T)
            atm_vol = SVIFitter.svi_implied_vol(
                np.array([0.0]),
                params.a,
                params.b,
                params.rho,
                params.m,
                params.sigma,
                T,
            )[0]

            term_structure.append({
                "expiry_date": expiry.isoformat(),
                "days": int(T * 365),
                "atm_vol": float(atm_vol),
                "total_variance": float(atm_vol**2 * T),
            })
        except Exception as e:
            logger.warning(f"Could not fit {expiry}: {e}")

    # Calculate forward vols
    import numpy as np

    for i in range(1, len(term_structure)):
        prev = term_structure[i - 1]
        curr = term_structure[i]

        t1 = prev["days"] / 365
        t2 = curr["days"] / 365
        var1 = prev["atm_vol"] ** 2 * t1
        var2 = curr["atm_vol"] ** 2 * t2

        fwd_var = (var2 - var1) / (t2 - t1)
        fwd_vol = np.sqrt(max(0, fwd_var))

        curr["forward_vol"] = float(fwd_vol)

    return {
        "ticker": ticker,
        "spot": chain.spot_price,
        "timestamp": datetime.now().isoformat(),
        "term_structure": term_structure,
    }


# Need to import numpy for term structure calculation
import numpy as np
