"""Extract features from volatility surfaces for analysis.

Key features extracted:
- ATM vol: At-the-money implied volatility at standard tenors
- 25Δ Skew: σ(25Δ put) - σ(25Δ call), measures asymmetry
- Butterfly: 0.5*(σ(25Δ put) + σ(25Δ call)) - σ(ATM), measures curvature
- Term slope: (σ_90d - σ_30d) / σ_30d, measures term structure steepness
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from analytics.svi_fitter import SVIFitter, SVIParams, interpolate_svi_surface
from analytics.black_scholes import BlackScholes

logger = logging.getLogger(__name__)


@dataclass
class SurfaceFeatures:
    """Container for extracted volatility surface features.

    Attributes:
        timestamp: When features were extracted
        ticker: Underlying symbol
        spot: Current spot price
        atm_vol_30d: ATM volatility at 30 days
        atm_vol_60d: ATM volatility at 60 days
        atm_vol_90d: ATM volatility at 90 days
        skew_25d_30d: 25-delta skew at 30 days
        skew_25d_60d: 25-delta skew at 60 days
        skew_25d_90d: 25-delta skew at 90 days
        butterfly_30d: Butterfly spread at 30 days
        butterfly_60d: Butterfly spread at 60 days
        butterfly_90d: Butterfly spread at 90 days
        term_slope: Term structure slope (90d vs 30d)
        forward_vol_30_60: Forward vol from 30d to 60d
        forward_vol_60_90: Forward vol from 60d to 90d
    """

    timestamp: datetime
    ticker: str
    spot: float

    # ATM volatilities
    atm_vol_30d: float | None = None
    atm_vol_60d: float | None = None
    atm_vol_90d: float | None = None

    # Skew (25-delta put vol - 25-delta call vol)
    skew_25d_30d: float | None = None
    skew_25d_60d: float | None = None
    skew_25d_90d: float | None = None

    # Butterfly (wing vol premium over ATM)
    butterfly_30d: float | None = None
    butterfly_60d: float | None = None
    butterfly_90d: float | None = None

    # Term structure
    term_slope: float | None = None
    forward_vol_30_60: float | None = None
    forward_vol_60_90: float | None = None

    # Raw SVI parameters for reference
    svi_params: dict[int, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "ticker": self.ticker,
            "spot": self.spot,
            "atm_vol_30d": self.atm_vol_30d,
            "atm_vol_60d": self.atm_vol_60d,
            "atm_vol_90d": self.atm_vol_90d,
            "skew_25d_30d": self.skew_25d_30d,
            "skew_25d_60d": self.skew_25d_60d,
            "skew_25d_90d": self.skew_25d_90d,
            "butterfly_30d": self.butterfly_30d,
            "butterfly_60d": self.butterfly_60d,
            "butterfly_90d": self.butterfly_90d,
            "term_slope": self.term_slope,
            "forward_vol_30_60": self.forward_vol_30_60,
            "forward_vol_60_90": self.forward_vol_60_90,
            "svi_params": self.svi_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SurfaceFeatures":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            ticker=data["ticker"],
            spot=data["spot"],
            atm_vol_30d=data.get("atm_vol_30d"),
            atm_vol_60d=data.get("atm_vol_60d"),
            atm_vol_90d=data.get("atm_vol_90d"),
            skew_25d_30d=data.get("skew_25d_30d"),
            skew_25d_60d=data.get("skew_25d_60d"),
            skew_25d_90d=data.get("skew_25d_90d"),
            butterfly_30d=data.get("butterfly_30d"),
            butterfly_60d=data.get("butterfly_60d"),
            butterfly_90d=data.get("butterfly_90d"),
            term_slope=data.get("term_slope"),
            forward_vol_30_60=data.get("forward_vol_30_60"),
            forward_vol_60_90=data.get("forward_vol_60_90"),
            svi_params=data.get("svi_params", {}),
        )


class FeatureExtractor:
    """Extracts standardized features from fitted volatility surfaces.

    Features are extracted at standard tenors (30d, 60d, 90d) using
    interpolation of SVI parameters if exact expiries aren't available.

    Example:
        >>> extractor = FeatureExtractor(spot=450, rate=0.05)
        >>> features = extractor.extract(svi_params_list, "SPY")
        >>> print(f"30d ATM vol: {features.atm_vol_30d:.2%}")
        >>> print(f"30d skew: {features.skew_25d_30d:.2%}")
    """

    # Standard tenors in days
    STANDARD_TENORS = [30, 60, 90]
    # Days per year for conversion
    DAYS_PER_YEAR = 365

    def __init__(
        self,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
    ):
        """Initialize feature extractor.

        Args:
            spot: Current spot price
            rate: Risk-free interest rate
            dividend_yield: Continuous dividend yield
        """
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield
        self.bs = BlackScholes(spot, rate, dividend_yield)

    def _get_vol_at_delta(
        self,
        params: SVIParams,
        delta: float,
        option_type: str,
    ) -> float:
        """Get implied vol at a specific delta.

        Uses iterative search to find the strike corresponding to delta,
        then evaluates the SVI surface at that log-moneyness.

        Args:
            params: SVI parameters for this expiry
            delta: Target delta (e.g., 0.25)
            option_type: 'call' or 'put'

        Returns:
            Implied volatility at the target delta
        """
        T = params.expiry_years

        # Initial guess using ATM vol
        atm_vol = SVIFitter.svi_implied_vol(
            np.array([0.0]), params.a, params.b, params.rho, params.m, params.sigma, T
        )[0]

        # Find strike at target delta
        strike = self.bs.strike_from_delta(delta, T, atm_vol, option_type)
        forward = self.bs.forward_price(T)
        log_moneyness = np.log(strike / forward)

        # Get vol at this log-moneyness
        vol = SVIFitter.svi_implied_vol(
            np.array([log_moneyness]),
            params.a,
            params.b,
            params.rho,
            params.m,
            params.sigma,
            T,
        )[0]

        # Iterate once more for refinement
        strike = self.bs.strike_from_delta(delta, T, vol, option_type)
        log_moneyness = np.log(strike / forward)

        return float(
            SVIFitter.svi_implied_vol(
                np.array([log_moneyness]),
                params.a,
                params.b,
                params.rho,
                params.m,
                params.sigma,
                T,
            )[0]
        )

    def _get_atm_vol(self, params: SVIParams) -> float:
        """Get ATM implied volatility (at k=0)."""
        return float(
            SVIFitter.svi_implied_vol(
                np.array([0.0]),
                params.a,
                params.b,
                params.rho,
                params.m,
                params.sigma,
                params.expiry_years,
            )[0]
        )

    def _compute_skew(self, params: SVIParams) -> float:
        """Compute 25-delta skew: σ(25Δ put) - σ(25Δ call)."""
        put_vol = self._get_vol_at_delta(params, 0.25, "put")
        call_vol = self._get_vol_at_delta(params, 0.25, "call")
        return put_vol - call_vol

    def _compute_butterfly(self, params: SVIParams) -> float:
        """Compute butterfly: 0.5*(σ(25Δ put) + σ(25Δ call)) - σ(ATM)."""
        put_vol = self._get_vol_at_delta(params, 0.25, "put")
        call_vol = self._get_vol_at_delta(params, 0.25, "call")
        atm_vol = self._get_atm_vol(params)
        return 0.5 * (put_vol + call_vol) - atm_vol

    def _compute_forward_vol(
        self,
        atm_vol_near: float,
        atm_vol_far: float,
        t_near: float,
        t_far: float,
    ) -> float:
        """Compute forward volatility between two tenors.

        Forward variance: var_fwd = (var_far * t_far - var_near * t_near) / (t_far - t_near)
        Forward vol: σ_fwd = sqrt(var_fwd)

        Args:
            atm_vol_near: ATM vol at near tenor
            atm_vol_far: ATM vol at far tenor
            t_near: Near tenor in years
            t_far: Far tenor in years

        Returns:
            Forward volatility
        """
        var_near = atm_vol_near**2 * t_near
        var_far = atm_vol_far**2 * t_far
        var_fwd = (var_far - var_near) / (t_far - t_near)

        if var_fwd < 0:
            logger.warning("Negative forward variance detected, using 0")
            return 0.0

        return np.sqrt(var_fwd)

    def _interpolate_params_to_tenor(
        self,
        params_list: list[SVIParams],
        target_days: int,
    ) -> SVIParams | None:
        """Interpolate SVI parameters to a target tenor.

        Args:
            params_list: List of fitted SVI parameters
            target_days: Target tenor in days

        Returns:
            Interpolated SVIParams or None if not possible
        """
        if len(params_list) < 1:
            return None

        target_years = target_days / self.DAYS_PER_YEAR
        expiries = np.array([p.expiry_years for p in params_list])

        # Find closest expiry
        idx = np.argmin(np.abs(expiries - target_years))
        closest = params_list[idx]

        # If very close (within 7 days), use directly
        if np.abs(expiries[idx] - target_years) < 7 / self.DAYS_PER_YEAR:
            return SVIParams(
                a=closest.a,
                b=closest.b,
                rho=closest.rho,
                m=closest.m,
                sigma=closest.sigma,
                expiry_years=target_years,
            )

        # Otherwise interpolate if we have bracketing expiries
        if len(params_list) < 2:
            return SVIParams(
                a=closest.a,
                b=closest.b,
                rho=closest.rho,
                m=closest.m,
                sigma=closest.sigma,
                expiry_years=target_years,
            )

        sorted_params = sorted(params_list, key=lambda p: p.expiry_years)
        sorted_expiries = [p.expiry_years for p in sorted_params]

        # Find bracketing expiries
        for i in range(len(sorted_expiries) - 1):
            if sorted_expiries[i] <= target_years <= sorted_expiries[i + 1]:
                t1, t2 = sorted_expiries[i], sorted_expiries[i + 1]
                p1, p2 = sorted_params[i], sorted_params[i + 1]
                w = (target_years - t1) / (t2 - t1)

                return SVIParams(
                    a=p1.a + w * (p2.a - p1.a),
                    b=p1.b + w * (p2.b - p1.b),
                    rho=p1.rho + w * (p2.rho - p1.rho),
                    m=p1.m + w * (p2.m - p1.m),
                    sigma=p1.sigma + w * (p2.sigma - p1.sigma),
                    expiry_years=target_years,
                )

        # Extrapolate from nearest
        return SVIParams(
            a=closest.a,
            b=closest.b,
            rho=closest.rho,
            m=closest.m,
            sigma=closest.sigma,
            expiry_years=target_years,
        )

    def extract(
        self,
        params_list: list[SVIParams],
        ticker: str,
        timestamp: datetime | None = None,
    ) -> SurfaceFeatures:
        """Extract all features from fitted SVI parameters.

        Args:
            params_list: List of fitted SVI parameters at various expiries
            ticker: Underlying symbol
            timestamp: Observation timestamp (defaults to now)

        Returns:
            SurfaceFeatures with all extracted features

        Example:
            >>> extractor = FeatureExtractor(spot=450)
            >>> features = extractor.extract(params_list, "SPY")
        """
        if timestamp is None:
            timestamp = datetime.now()

        features = SurfaceFeatures(
            timestamp=timestamp,
            ticker=ticker,
            spot=self.spot,
        )

        if not params_list:
            logger.warning("No SVI parameters provided")
            return features

        # Extract features at each standard tenor
        tenors = {30: "30d", 60: "60d", 90: "90d"}
        atm_vols: dict[int, float] = {}

        for days, suffix in tenors.items():
            params = self._interpolate_params_to_tenor(params_list, days)
            if params is None:
                continue

            # ATM vol
            atm_vol = self._get_atm_vol(params)
            atm_vols[days] = atm_vol
            setattr(features, f"atm_vol_{suffix}", atm_vol)

            # Skew
            try:
                skew = self._compute_skew(params)
                setattr(features, f"skew_25d_{suffix}", skew)
            except Exception as e:
                logger.warning(f"Could not compute skew at {days}d: {e}")

            # Butterfly
            try:
                butterfly = self._compute_butterfly(params)
                setattr(features, f"butterfly_{suffix}", butterfly)
            except Exception as e:
                logger.warning(f"Could not compute butterfly at {days}d: {e}")

            # Store SVI params
            features.svi_params[days] = params.to_dict()

        # Term structure slope: (σ_90d - σ_30d) / σ_30d
        if 30 in atm_vols and 90 in atm_vols and atm_vols[30] > 0:
            features.term_slope = (atm_vols[90] - atm_vols[30]) / atm_vols[30]

        # Forward volatilities
        if 30 in atm_vols and 60 in atm_vols:
            features.forward_vol_30_60 = self._compute_forward_vol(
                atm_vols[30], atm_vols[60], 30 / 365, 60 / 365
            )

        if 60 in atm_vols and 90 in atm_vols:
            features.forward_vol_60_90 = self._compute_forward_vol(
                atm_vols[60], atm_vols[90], 60 / 365, 90 / 365
            )

        logger.info(
            f"Extracted features for {ticker}: ATM30={features.atm_vol_30d:.2%}, "
            f"skew30={features.skew_25d_30d:.4f}, butterfly30={features.butterfly_30d:.4f}"
            if features.atm_vol_30d
            else f"Extracted features for {ticker}"
        )

        return features

    def extract_smile_points(
        self,
        params: SVIParams,
        deltas: list[float] | None = None,
    ) -> dict[str, float]:
        """Extract volatility at standard delta points.

        Args:
            params: SVI parameters for an expiry
            deltas: List of deltas to extract (default: standard grid)

        Returns:
            Dictionary mapping delta labels to implied vols

        Example:
            >>> extractor = FeatureExtractor(spot=100)
            >>> smile = extractor.extract_smile_points(params)
            >>> print(smile["25D Put"])
        """
        if deltas is None:
            deltas = [0.10, 0.25, 0.50]  # 10D, 25D, ATM

        result = {}

        for d in deltas:
            if d == 0.50:
                # ATM
                result["ATM"] = self._get_atm_vol(params)
            else:
                # Put and call at this delta
                label_put = f"{int(d*100)}D Put"
                label_call = f"{int(d*100)}D Call"
                result[label_put] = self._get_vol_at_delta(params, d, "put")
                result[label_call] = self._get_vol_at_delta(params, d, "call")

        return result
