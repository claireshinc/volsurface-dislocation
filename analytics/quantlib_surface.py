"""QuantLib-based volatility surface construction and interpolation.

This module provides volatility surface handling using QuantLib's
sophisticated interpolation and surface construction capabilities.

QuantLib Volatility Surface Hierarchy:
=====================================

BlackVolTermStructure (abstract base)
├── BlackConstantVol (flat vol)
├── BlackVarianceSurface (interpolated market data)
├── LocalVolSurface (Dupire local vol)
└── HestonBlackVolSurface (from Heston model)

Key Concepts:
- BlackVarianceSurface: Interpolates variance (not vol) for smoothness
- Interpolation methods: Bilinear, Bicubic, etc.
- Extrapolation: Constant or linear outside grid
- Calendar and day count conventions matter

Example:
    >>> surface = VolatilitySurface(spot=100, rate=0.05)
    >>> surface.build_from_market_data(strikes, expiries, vols)
    >>> vol = surface.get_vol(strike=105, expiry=0.25)
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import QuantLib as ql
import pandas as pd

from analytics.quantlib_utils import QuantLibSetup

logger = logging.getLogger(__name__)


@dataclass
class VolSurfacePoint:
    """A single point on the volatility surface.

    Attributes:
        strike: Strike price
        expiry: Expiry date
        expiry_years: Time to expiry in years
        implied_vol: Implied volatility
        forward: Forward price at this expiry
        log_moneyness: log(K/F)
    """

    strike: float
    expiry: date
    expiry_years: float
    implied_vol: float
    forward: float = 0.0
    log_moneyness: float = 0.0


class VolatilitySurface:
    """QuantLib-based implied volatility surface.

    Constructs a BlackVarianceSurface from market data and provides
    interpolation of implied volatility at any strike/expiry point.

    QuantLib Interpolation Methods:
    - Bilinear: Linear in both dimensions (default)
    - Bicubic: Cubic spline in both dimensions (smoother)

    The surface interpolates VARIANCE (σ²×T) not volatility directly.
    This ensures calendar arbitrage-free interpolation.

    Example:
        >>> surface = VolatilitySurface(spot=100, rate=0.05)
        >>>
        >>> # Build from market data
        >>> strikes = [90, 95, 100, 105, 110]
        >>> expiries = [date(2024, 4, 1), date(2024, 7, 1)]
        >>> vols = [[0.22, 0.20, 0.18, 0.19, 0.21],
        ...         [0.21, 0.19, 0.17, 0.18, 0.20]]
        >>> surface.build_from_market_data(strikes, expiries, vols)
        >>>
        >>> # Get interpolated vol
        >>> vol = surface.get_vol(strike=102, expiry_years=0.35)
    """

    def __init__(
        self,
        spot: float,
        rate: float = 0.05,
        dividend_yield: float = 0.0,
    ):
        """Initialize volatility surface.

        Args:
            spot: Current spot price
            rate: Risk-free interest rate
            dividend_yield: Continuous dividend yield
        """
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield

        self._setup = QuantLibSetup()
        self._eval_date = ql.Settings.instance().evaluationDate
        self._calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        self._day_count = ql.Actual365Fixed()

        self._ql_surface: Optional[ql.BlackVarianceSurface] = None
        self._surface_handle: Optional[ql.BlackVolTermStructureHandle] = None

        # Store market data
        self._strikes: List[float] = []
        self._expiry_dates: List[ql.Date] = []
        self._vol_matrix: Optional[ql.Matrix] = None

    def _to_ql_date(self, dt: date) -> ql.Date:
        """Convert Python date to QuantLib Date."""
        return ql.Date(dt.day, dt.month, dt.year)

    def _year_fraction(self, expiry_date: ql.Date) -> float:
        """Calculate year fraction from evaluation date to expiry."""
        return self._day_count.yearFraction(self._eval_date, expiry_date)

    def forward_price(self, expiry_years: float) -> float:
        """Calculate forward price at given expiry.

        F = S × exp((r - q) × T)

        Args:
            expiry_years: Time to expiry in years

        Returns:
            Forward price
        """
        return self.spot * np.exp((self.rate - self.dividend_yield) * expiry_years)

    def build_from_market_data(
        self,
        strikes: List[float],
        expiry_dates: List[date] | List[float],
        vol_matrix: List[List[float]],
        interpolation: str = "bilinear",
    ) -> None:
        """Build volatility surface from market data.

        Creates a QuantLib BlackVarianceSurface which:
        1. Stores the market vol grid
        2. Interpolates variance between grid points
        3. Extrapolates outside the grid (constant)

        Args:
            strikes: List of strike prices (must be sorted ascending)
            expiry_dates: List of expiry dates OR year fractions (must be sorted)
            vol_matrix: 2D list of implied vols [expiry][strike]
            interpolation: "bilinear" or "bicubic"

        Example:
            >>> # Using dates
            >>> surface.build_from_market_data(
            ...     strikes=[90, 100, 110],
            ...     expiry_dates=[date(2024, 4, 1), date(2024, 7, 1)],
            ...     vol_matrix=[[0.22, 0.18, 0.21], [0.21, 0.17, 0.20]]
            ... )
            >>> # OR using year fractions
            >>> surface.build_from_market_data(
            ...     strikes=[90, 100, 110],
            ...     expiry_dates=[0.25, 0.5],  # years
            ...     vol_matrix=[[0.22, 0.18, 0.21], [0.21, 0.17, 0.20]]
            ... )
        """
        # Store data
        self._strikes = sorted(strikes)

        # Handle both date objects and year fractions
        sorted_expiries = sorted(expiry_dates)
        if isinstance(sorted_expiries[0], (int, float)):
            # Convert year fractions to dates from evaluation date
            from datetime import timedelta
            eval_py_date = date(
                self._eval_date.year(),
                self._eval_date.month(),
                self._eval_date.dayOfMonth()
            )
            self._expiry_dates = [
                ql.Date(
                    (eval_py_date + timedelta(days=int(yf * 365))).day,
                    (eval_py_date + timedelta(days=int(yf * 365))).month,
                    (eval_py_date + timedelta(days=int(yf * 365))).year
                )
                for yf in sorted_expiries
            ]
        else:
            self._expiry_dates = [self._to_ql_date(d) for d in sorted_expiries]

        # Create QuantLib matrix (rows=strikes, cols=expiries)
        # Note: QuantLib expects Matrix(rows, cols) with vol[strike][expiry]
        n_strikes = len(strikes)
        n_expiries = len(expiry_dates)

        self._vol_matrix = ql.Matrix(n_strikes, n_expiries)

        for i, strike in enumerate(self._strikes):
            for j, _ in enumerate(self._expiry_dates):
                # vol_matrix is [expiry][strike], QuantLib wants [strike][expiry]
                self._vol_matrix[i][j] = vol_matrix[j][i]

        # Create the BlackVarianceSurface
        self._ql_surface = ql.BlackVarianceSurface(
            self._eval_date,
            self._calendar,
            self._expiry_dates,
            self._strikes,
            self._vol_matrix,
            self._day_count,
        )

        # Set extrapolation
        self._ql_surface.enableExtrapolation()

        # Create handle for use in pricing engines
        self._surface_handle = ql.BlackVolTermStructureHandle(self._ql_surface)

        logger.info(
            f"Built vol surface: {n_strikes} strikes × {n_expiries} expiries"
        )

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        strike_col: str = "strike",
        expiry_col: str = "expiration",
        vol_col: str = "implied_volatility",
    ) -> None:
        """Build surface from a DataFrame of options data.

        Pivots the DataFrame to create the strike×expiry grid.

        Args:
            df: DataFrame with options data
            strike_col: Column name for strikes
            expiry_col: Column name for expiry dates
            vol_col: Column name for implied volatilities
        """
        # Pivot to get vol matrix
        pivot = df.pivot_table(
            values=vol_col,
            index=strike_col,
            columns=expiry_col,
            aggfunc="mean",
        )

        strikes = pivot.index.tolist()
        expiries = [
            d.date() if hasattr(d, 'date') else d
            for d in pivot.columns.tolist()
        ]

        # Convert to list of lists [expiry][strike]
        vol_matrix = []
        for expiry in pivot.columns:
            vol_matrix.append(pivot[expiry].tolist())

        self.build_from_market_data(strikes, expiries, vol_matrix)

    def get_vol(
        self,
        strike: float,
        expiry_years: float = None,
        expiry_date: date = None,
    ) -> float:
        """Get interpolated implied volatility.

        QuantLib's BlackVarianceSurface interpolates in variance space
        (σ²×T) to avoid calendar arbitrage, then converts back to vol.

        Args:
            strike: Strike price
            expiry_years: Time to expiry in years (provide this OR expiry_date)
            expiry_date: Expiry date (provide this OR expiry_years)

        Returns:
            Interpolated implied volatility

        Raises:
            ValueError: If surface not built or invalid inputs
        """
        if self._ql_surface is None:
            raise ValueError("Surface not built. Call build_from_market_data first.")

        if expiry_date is not None:
            ql_date = self._to_ql_date(expiry_date)
            expiry_years = self._year_fraction(ql_date)
        elif expiry_years is None:
            raise ValueError("Provide either expiry_years or expiry_date")

        return self._ql_surface.blackVol(expiry_years, strike)

    def get_variance(
        self,
        strike: float,
        expiry_years: float,
    ) -> float:
        """Get interpolated total variance (σ²×T).

        Total variance is what QuantLib actually interpolates.
        This is more fundamental than implied vol.

        Args:
            strike: Strike price
            expiry_years: Time to expiry in years

        Returns:
            Total variance
        """
        if self._ql_surface is None:
            raise ValueError("Surface not built.")

        return self._ql_surface.blackVariance(expiry_years, strike)

    def get_local_vol(
        self,
        strike: float,
        expiry_years: float,
    ) -> float:
        """Get Dupire local volatility.

        Local volatility is derived from the implied vol surface using
        Dupire's formula. It represents the instantaneous volatility
        at a specific spot and time.

        Dupire's formula:
        σ_local² = (∂w/∂T) / (1 - (y/w)(∂w/∂y) + 0.25(-0.25 - 1/w + y²/w²)(∂w/∂y)² + 0.5(∂²w/∂y²))

        where w = σ²T (total variance), y = log(K/F)

        Args:
            strike: Strike price
            expiry_years: Time to expiry in years

        Returns:
            Local volatility

        Note:
            This is a simplified numerical approximation.
            For production, use QuantLib's LocalVolSurface.
        """
        if self._ql_surface is None:
            raise ValueError("Surface not built.")

        # Numerical derivatives
        dt = 0.01
        dk = self.spot * 0.01

        w = self.get_variance(strike, expiry_years)
        w_up_t = self.get_variance(strike, expiry_years + dt)
        w_up_k = self.get_variance(strike + dk, expiry_years)
        w_down_k = self.get_variance(strike - dk, expiry_years)

        dw_dt = (w_up_t - w) / dt
        dw_dk = (w_up_k - w_down_k) / (2 * dk)
        d2w_dk2 = (w_up_k - 2 * w + w_down_k) / (dk ** 2)

        # Simplified Dupire (ignoring some terms for stability)
        local_var = dw_dt / (1 + strike * d2w_dk2 * 0.5)
        local_var = max(local_var, 1e-6)  # Ensure positive

        return np.sqrt(local_var)

    def get_smile(
        self,
        expiry_years: float,
        n_points: int = 21,
        moneyness_range: Tuple[float, float] = (0.8, 1.2),
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get volatility smile at a specific expiry.

        Args:
            expiry_years: Time to expiry in years
            n_points: Number of strike points
            moneyness_range: (min, max) as fraction of forward

        Returns:
            Tuple of (strikes, implied_vols)
        """
        forward = self.forward_price(expiry_years)

        strikes = np.linspace(
            forward * moneyness_range[0],
            forward * moneyness_range[1],
            n_points,
        )

        vols = np.array([self.get_vol(k, expiry_years) for k in strikes])

        return strikes, vols

    def get_term_structure(
        self,
        strike: float = None,
        delta: float = None,
        expiry_years_range: Tuple[float, float] = (0.05, 1.0),
        n_points: int = 20,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get ATM volatility term structure.

        Args:
            strike: Fixed strike (if None, uses ATM)
            delta: Fixed delta (alternative to strike)
            expiry_years_range: (min, max) expiry in years
            n_points: Number of expiry points

        Returns:
            Tuple of (expiry_years, implied_vols)
        """
        expiries = np.linspace(
            expiry_years_range[0],
            expiry_years_range[1],
            n_points,
        )

        vols = []
        for T in expiries:
            if strike is not None:
                k = strike
            else:
                # ATM forward
                k = self.forward_price(T)

            vols.append(self.get_vol(k, T))

        return expiries, np.array(vols)

    def get_handle(self) -> ql.BlackVolTermStructureHandle:
        """Get QuantLib handle for use in pricing engines.

        The handle can be passed to QuantLib pricing engines:

            >>> process = ql.BlackScholesMertonProcess(
            ...     spot_handle, div_handle, rate_handle,
            ...     surface.get_handle()
            ... )

        Returns:
            BlackVolTermStructureHandle
        """
        if self._surface_handle is None:
            raise ValueError("Surface not built.")
        return self._surface_handle

    def to_matrix(
        self,
        strikes: List[float] = None,
        expiry_years: List[float] = None,
    ) -> pd.DataFrame:
        """Export surface as a DataFrame matrix.

        Args:
            strikes: Strike points (default: stored strikes)
            expiry_years: Expiry points (default: stored expiries)

        Returns:
            DataFrame with strikes as index, expiries as columns
        """
        if strikes is None:
            strikes = self._strikes

        if expiry_years is None:
            expiry_years = [
                self._year_fraction(d) for d in self._expiry_dates
            ]

        data = {}
        for T in expiry_years:
            col = []
            for k in strikes:
                try:
                    col.append(self.get_vol(k, T))
                except:
                    col.append(np.nan)
            data[f"{T:.3f}y"] = col

        return pd.DataFrame(data, index=strikes)


class VolatilitySurfaceAnalytics:
    """Analytics computed from a volatility surface.

    Provides higher-level analytics that can be derived from
    the implied volatility surface.
    """

    def __init__(self, surface: VolatilitySurface):
        """Initialize with a volatility surface.

        Args:
            surface: Built VolatilitySurface object
        """
        self.surface = surface

    def atm_vol(self, expiry_years: float) -> float:
        """Get ATM implied volatility.

        ATM is defined as the forward strike.

        Args:
            expiry_years: Time to expiry in years

        Returns:
            ATM implied volatility
        """
        forward = self.surface.forward_price(expiry_years)
        return self.surface.get_vol(forward, expiry_years)

    def skew(
        self,
        expiry_years: float,
        delta: float = 0.25,
    ) -> float:
        """Calculate volatility skew.

        Skew = σ(25Δ put) - σ(25Δ call)

        Positive skew means puts are more expensive (typical for equities).

        Args:
            expiry_years: Time to expiry in years
            delta: Delta level (default 0.25)

        Returns:
            Skew value
        """
        from analytics.black_scholes import BlackScholes

        atm_vol = self.atm_vol(expiry_years)

        bs = BlackScholes(
            self.surface.spot,
            self.surface.rate,
            self.surface.dividend_yield,
        )

        # Find 25-delta strikes
        k_put = bs.strike_from_delta(delta, expiry_years, atm_vol, "put")
        k_call = bs.strike_from_delta(delta, expiry_years, atm_vol, "call")

        vol_put = self.surface.get_vol(k_put, expiry_years)
        vol_call = self.surface.get_vol(k_call, expiry_years)

        return vol_put - vol_call

    def butterfly(
        self,
        expiry_years: float,
        delta: float = 0.25,
    ) -> float:
        """Calculate butterfly spread.

        Butterfly = 0.5 × (σ_put + σ_call) - σ_ATM

        Measures smile curvature / convexity.

        Args:
            expiry_years: Time to expiry in years
            delta: Delta level (default 0.25)

        Returns:
            Butterfly value
        """
        from analytics.black_scholes import BlackScholes

        atm_vol = self.atm_vol(expiry_years)

        bs = BlackScholes(
            self.surface.spot,
            self.surface.rate,
            self.surface.dividend_yield,
        )

        k_put = bs.strike_from_delta(delta, expiry_years, atm_vol, "put")
        k_call = bs.strike_from_delta(delta, expiry_years, atm_vol, "call")

        vol_put = self.surface.get_vol(k_put, expiry_years)
        vol_call = self.surface.get_vol(k_call, expiry_years)

        return 0.5 * (vol_put + vol_call) - atm_vol

    def skew_25d(
        self,
        expiry_years: float,
        spot: float = None,
        rate: float = None,
    ) -> float:
        """Calculate 25-delta skew (convenience alias for skew).

        Args:
            expiry_years: Time to expiry in years
            spot: Spot price (ignored, uses surface spot)
            rate: Rate (ignored, uses surface rate)

        Returns:
            25-delta skew value
        """
        return self.skew(expiry_years, delta=0.25)

    def risk_reversal(
        self,
        expiry_years: float,
        delta: float = 0.25,
    ) -> float:
        """Calculate risk reversal.

        Risk Reversal = σ(Δ call) - σ(Δ put)

        Note: This is the negative of skew.

        Args:
            expiry_years: Time to expiry in years
            delta: Delta level

        Returns:
            Risk reversal value
        """
        return -self.skew(expiry_years, delta)

    def term_slope(
        self,
        short_expiry: float = 30 / 365,
        long_expiry: float = 90 / 365,
    ) -> float:
        """Calculate term structure slope.

        Slope = (σ_long - σ_short) / σ_short

        Positive slope = contango (normal)
        Negative slope = backwardation (inverted)

        Args:
            short_expiry: Short-term expiry in years
            long_expiry: Long-term expiry in years

        Returns:
            Term slope as fraction
        """
        vol_short = self.atm_vol(short_expiry)
        vol_long = self.atm_vol(long_expiry)

        if vol_short <= 0:
            return 0.0

        return (vol_long - vol_short) / vol_short

    def forward_vol(
        self,
        t1: float,
        t2: float,
        strike: float = None,
    ) -> float:
        """Calculate forward implied volatility.

        Forward vol from t1 to t2:
        σ_fwd² = (σ₂²×t₂ - σ₁²×t₁) / (t₂ - t₁)

        Args:
            t1: Start time in years
            t2: End time in years
            strike: Strike (default: ATM forward)

        Returns:
            Forward volatility
        """
        if strike is None:
            strike = self.surface.forward_price((t1 + t2) / 2)

        var1 = self.surface.get_variance(strike, t1)
        var2 = self.surface.get_variance(strike, t2)

        fwd_var = (var2 - var1) / (t2 - t1)

        if fwd_var < 0:
            logger.warning(f"Negative forward variance between {t1} and {t2}")
            return 0.0

        return np.sqrt(fwd_var)
