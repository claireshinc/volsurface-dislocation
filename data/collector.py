"""Options chain data collection using OpenBB.

Supports multiple free data providers:
- CBOE (Chicago Board Options Exchange) - primary
- yfinance - fallback
"""

import logging
from dataclasses import dataclass
from datetime import datetime, date
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class OptionsChain:
    """Processed options chain data.

    Attributes:
        ticker: Underlying symbol
        timestamp: When data was collected
        spot_price: Current underlying price
        expiries: Available expiration dates
        chain_data: DataFrame with full chain data
        calls: DataFrame filtered to calls
        puts: DataFrame filtered to puts
    """

    ticker: str
    timestamp: datetime
    spot_price: float
    expiries: list[date]
    chain_data: pd.DataFrame
    calls: pd.DataFrame
    puts: pd.DataFrame

    def get_expiry_chain(self, expiry: date) -> pd.DataFrame:
        """Get options for a specific expiry."""
        return self.chain_data[self.chain_data["expiration"] == expiry]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Convert DataFrame to records with proper type handling
        chain_records = []
        for _, row in self.chain_data.iterrows():
            record = {}
            for col, val in row.items():
                if pd.isna(val):
                    record[col] = None
                elif hasattr(val, 'isoformat'):
                    record[col] = val.isoformat() if hasattr(val, 'isoformat') else str(val)
                elif isinstance(val, (np.integer, np.floating)):
                    record[col] = float(val) if isinstance(val, np.floating) else int(val)
                else:
                    record[col] = val
            chain_records.append(record)

        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "spot_price": self.spot_price,
            "expiries": [e.isoformat() for e in self.expiries],
            "chain_data": chain_records,
        }


class OptionsDataCollector:
    """Collects options chain data using OpenBB.

    Uses free data providers (CBOE, yfinance) to fetch options chains.
    Handles data cleaning and standardization across providers.

    Example:
        >>> collector = OptionsDataCollector()
        >>> chain = collector.get_chain("SPY")
        >>> print(f"Spot: ${chain.spot_price:.2f}")
        >>> print(f"Expiries: {len(chain.expiries)}")
    """

    # Column name mappings for standardization
    COLUMN_MAPPINGS = {
        "strike": ["strike", "strike_price", "strikePrice"],
        "expiration": ["expiration", "expiry", "expirationDate"],
        "bid": ["bid", "bidPrice"],
        "ask": ["ask", "askPrice"],
        "last": ["last", "lastPrice", "lastTradePrice"],
        "volume": ["volume", "totalVolume"],
        "open_interest": ["open_interest", "openInterest", "oi"],
        "implied_volatility": ["implied_volatility", "impliedVolatility", "iv"],
        "delta": ["delta"],
        "gamma": ["gamma"],
        "theta": ["theta"],
        "vega": ["vega"],
        "option_type": ["option_type", "optionType", "type", "contractType"],
    }

    def __init__(
        self,
        provider: str = "cboe",
        fallback_provider: str = "yfinance",
    ):
        """Initialize data collector.

        Args:
            provider: Primary data provider
            fallback_provider: Fallback if primary fails
        """
        self.provider = provider
        self.fallback_provider = fallback_provider
        self._obb = None

    def _get_obb(self):
        """Lazy load OpenBB."""
        if self._obb is None:
            try:
                from openbb import obb

                obb.user.preferences.output_type = "dataframe"
                self._obb = obb
            except ImportError:
                logger.warning("OpenBB not installed, will use yfinance directly")
                self._obb = False  # Mark as unavailable
        return self._obb

    def _get_chain_yfinance(self, ticker: str) -> tuple[pd.DataFrame, float]:
        """Fetch options chain directly from yfinance.

        Args:
            ticker: Stock symbol

        Returns:
            Tuple of (chain DataFrame, spot price)
        """
        import yfinance as yf

        stock = yf.Ticker(ticker)

        # Get spot price
        info = stock.info
        spot = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose", 0)

        # Get all expiration dates
        expirations = stock.options

        if not expirations:
            raise ValueError(f"No options data available for {ticker}")

        all_options = []

        for exp_date in expirations[:8]:  # Limit to first 8 expiries
            try:
                opt = stock.option_chain(exp_date)

                # Process calls
                calls = opt.calls.copy()
                calls["option_type"] = "call"
                calls["expiration"] = pd.to_datetime(exp_date).date()

                # Process puts
                puts = opt.puts.copy()
                puts["option_type"] = "put"
                puts["expiration"] = pd.to_datetime(exp_date).date()

                all_options.append(calls)
                all_options.append(puts)

            except Exception as e:
                logger.warning(f"Could not fetch {ticker} {exp_date}: {e}")

        if not all_options:
            raise ValueError(f"Could not fetch any options for {ticker}")

        df = pd.concat(all_options, ignore_index=True)

        # Rename columns to standard names
        column_map = {
            "impliedVolatility": "implied_volatility",
            "openInterest": "open_interest",
            "lastPrice": "last",
        }
        df.rename(columns=column_map, inplace=True)

        return df, spot

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across providers."""
        df = df.copy()

        # Rename columns to standard names
        for standard_name, variants in self.COLUMN_MAPPINGS.items():
            for variant in variants:
                if variant in df.columns and standard_name not in df.columns:
                    df.rename(columns={variant: standard_name}, inplace=True)
                    break

        return df

    def _clean_chain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate options chain data."""
        df = self._standardize_columns(df)

        # Ensure required columns exist
        required = ["strike", "expiration", "option_type"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Convert expiration to date if needed
        if df["expiration"].dtype == "object":
            df["expiration"] = pd.to_datetime(df["expiration"]).dt.date

        # Standardize option_type
        df["option_type"] = df["option_type"].str.lower()
        df["option_type"] = df["option_type"].replace({
            "c": "call",
            "p": "put",
        })

        # Calculate mid price if not present
        if "mid" not in df.columns and "bid" in df.columns and "ask" in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2

        # Filter out zero/negative values
        if "bid" in df.columns:
            df = df[df["bid"] >= 0]
        if "ask" in df.columns:
            df = df[df["ask"] > 0]
        if "implied_volatility" in df.columns:
            df = df[(df["implied_volatility"] > 0) & (df["implied_volatility"] < 5)]

        return df.reset_index(drop=True)

    def get_spot_price(self, ticker: str) -> float:
        """Get current spot price for ticker.

        Args:
            ticker: Stock symbol

        Returns:
            Current price
        """
        # Try yfinance directly first (more reliable)
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info
            spot = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
            if spot:
                return float(spot)
        except Exception as e:
            logger.warning(f"Error getting spot price via yfinance: {e}")

        # Try OpenBB if available
        try:
            obb = self._get_obb()
            if obb:
                quote = obb.equity.price.quote(ticker, provider="yfinance")
                if isinstance(quote, pd.DataFrame) and len(quote) > 0:
                    for col in ["last_price", "regularMarketPrice", "price", "close"]:
                        if col in quote.columns:
                            return float(quote[col].iloc[0])
        except Exception as e:
            logger.warning(f"Error getting spot price via OpenBB: {e}")

        return 0.0

    def get_chain(
        self,
        ticker: str,
        min_volume: int = 0,
        min_oi: int = 0,
    ) -> OptionsChain:
        """Fetch options chain for a ticker.

        Args:
            ticker: Stock symbol
            min_volume: Minimum volume filter
            min_oi: Minimum open interest filter

        Returns:
            OptionsChain object with processed data

        Example:
            >>> collector = OptionsDataCollector()
            >>> chain = collector.get_chain("SPY", min_oi=100)
        """
        logger.info(f"Fetching options chain for {ticker}")

        obb = self._get_obb()
        df = None
        spot = None

        # Try yfinance directly first (most reliable)
        try:
            logger.info(f"Fetching {ticker} via yfinance")
            df, spot = self._get_chain_yfinance(ticker)
        except Exception as e:
            logger.warning(f"yfinance failed: {e}")

            # Try OpenBB if available
            if obb:
                try:
                    logger.info(f"Trying OpenBB provider: {self.provider}")
                    result = obb.derivatives.options.chains(ticker, provider=self.provider)
                    if isinstance(result, pd.DataFrame):
                        df = result
                    else:
                        df = result.to_df()
                except Exception as e2:
                    logger.warning(f"OpenBB {self.provider} failed: {e2}")

                    # Try fallback provider
                    if self.fallback_provider:
                        try:
                            result = obb.derivatives.options.chains(
                                ticker, provider=self.fallback_provider
                            )
                            if isinstance(result, pd.DataFrame):
                                df = result
                            else:
                                df = result.to_df()
                        except Exception as e3:
                            logger.error(f"All providers failed for {ticker}")

        if df is None or len(df) == 0:
            raise ValueError(f"No options data found for {ticker}")

        # Clean and standardize
        df = self._clean_chain(df)

        # Apply filters
        if min_volume > 0 and "volume" in df.columns:
            df = df[df["volume"] >= min_volume]

        if min_oi > 0 and "open_interest" in df.columns:
            df = df[df["open_interest"] >= min_oi]

        # Get spot price (may already have it from yfinance)
        if spot is None or spot == 0:
            spot = self.get_spot_price(ticker)
        if spot == 0.0 and "strike" in df.columns:
            # Estimate spot from ATM options
            spot = df["strike"].median()

        # Split calls and puts
        calls = df[df["option_type"] == "call"].copy()
        puts = df[df["option_type"] == "put"].copy()

        # Get unique expiries
        expiries = sorted(df["expiration"].unique())

        chain = OptionsChain(
            ticker=ticker,
            timestamp=datetime.now(),
            spot_price=spot,
            expiries=expiries,
            chain_data=df,
            calls=calls,
            puts=puts,
        )

        logger.info(
            f"Fetched {len(df)} options for {ticker}: "
            f"{len(calls)} calls, {len(puts)} puts, "
            f"{len(expiries)} expiries"
        )

        return chain

    def get_chain_for_expiry(
        self,
        ticker: str,
        expiry: date,
    ) -> pd.DataFrame:
        """Get options chain for a specific expiry.

        Args:
            ticker: Stock symbol
            expiry: Target expiration date

        Returns:
            DataFrame with options for that expiry
        """
        chain = self.get_chain(ticker)
        return chain.get_expiry_chain(expiry)

    def calculate_log_moneyness(
        self,
        strikes: NDArray[np.float64],
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
        expiry_years: float = 0.0,
    ) -> NDArray[np.float64]:
        """Calculate log-moneyness for strikes.

        Log-moneyness: k = log(K/F) where F = S*exp((r-q)*T)

        Args:
            strikes: Strike prices
            spot: Spot price
            rate: Risk-free rate
            dividend_yield: Dividend yield
            expiry_years: Time to expiry in years

        Returns:
            Array of log-moneyness values
        """
        forward = spot * np.exp((rate - dividend_yield) * expiry_years)
        return np.log(strikes / forward)

    def prepare_for_svi_fitting(
        self,
        chain: OptionsChain,
        expiry: date,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Prepare chain data for SVI fitting.

        Args:
            chain: OptionsChain object
            expiry: Expiration date to fit
            rate: Risk-free rate
            dividend_yield: Dividend yield

        Returns:
            Tuple of (log_moneyness, total_variance, expiry_years)
        """
        df = chain.get_expiry_chain(expiry)

        if len(df) == 0:
            raise ValueError(f"No data for expiry {expiry}")

        # Calculate time to expiry
        today = date.today()
        days_to_expiry = (expiry - today).days
        expiry_years = max(days_to_expiry / 365, 1 / 365)  # Minimum 1 day

        # Get mid IV if available, otherwise use implied_volatility
        if "mid" in df.columns and "implied_volatility" not in df.columns:
            # Need to calculate IV from prices - use mid prices
            logger.warning("IV not in chain data, using mid prices")

        iv_col = "implied_volatility" if "implied_volatility" in df.columns else None
        if iv_col is None:
            raise ValueError("No implied volatility data in chain")

        # Filter valid data
        valid = df[iv_col].notna() & (df[iv_col] > 0) & (df[iv_col] < 3)
        df = df[valid]

        if len(df) < 5:
            raise ValueError(f"Insufficient valid data points for expiry {expiry}")

        strikes = df["strike"].values
        ivs = df[iv_col].values

        # Calculate log-moneyness
        log_moneyness = self.calculate_log_moneyness(
            strikes, chain.spot_price, rate, dividend_yield, expiry_years
        )

        # Convert IV to total variance: w = σ² * T
        total_variance = ivs**2 * expiry_years

        return log_moneyness, total_variance, expiry_years


def create_synthetic_chain(
    ticker: str = "SYNTH",
    spot: float = 100.0,
    atm_vol: float = 0.20,
    skew: float = -0.05,
    expiry_days: list[int] | None = None,
    n_strikes: int = 21,
) -> OptionsChain:
    """Create synthetic options chain for testing.

    Generates realistic options data using a simple skew model:
    σ(k) = atm_vol + skew * k

    Args:
        ticker: Symbol name
        spot: Spot price
        atm_vol: ATM implied volatility
        skew: Skew coefficient
        expiry_days: List of days to expiry
        n_strikes: Number of strikes per expiry

    Returns:
        OptionsChain with synthetic data
    """
    if expiry_days is None:
        expiry_days = [7, 14, 30, 60, 90, 180]

    today = date.today()
    all_data = []

    for days in expiry_days:
        expiry = date.fromordinal(today.toordinal() + days)
        T = days / 365

        # Generate strikes around ATM
        strikes = np.linspace(spot * 0.8, spot * 1.2, n_strikes)
        log_moneyness = np.log(strikes / spot)

        # Simple skew model
        ivs = atm_vol + skew * log_moneyness

        for i, strike in enumerate(strikes):
            iv = ivs[i]

            # Create call and put
            for opt_type in ["call", "put"]:
                # Simple mid approximation
                mid = max(0.01, iv * spot * np.sqrt(T) * 0.4)

                all_data.append({
                    "strike": strike,
                    "expiration": expiry,
                    "option_type": opt_type,
                    "bid": mid * 0.95,
                    "ask": mid * 1.05,
                    "mid": mid,
                    "implied_volatility": iv,
                    "volume": int(np.random.exponential(1000)),
                    "open_interest": int(np.random.exponential(5000)),
                    "delta": 0.5 if strike == spot else None,
                    "gamma": None,
                    "theta": None,
                    "vega": None,
                })

    df = pd.DataFrame(all_data)
    calls = df[df["option_type"] == "call"].copy()
    puts = df[df["option_type"] == "put"].copy()
    expiries = sorted(df["expiration"].unique())

    return OptionsChain(
        ticker=ticker,
        timestamp=datetime.now(),
        spot_price=spot,
        expiries=expiries,
        chain_data=df,
        calls=calls,
        puts=puts,
    )
