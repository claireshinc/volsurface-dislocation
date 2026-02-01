"""Black-Scholes option pricing and Greeks.

Implements the standard Black-Scholes-Merton model for European options:
    Call = S*N(d1) - K*exp(-r*T)*N(d2)
    Put  = K*exp(-r*T)*N(-d2) - S*N(-d1)

Where:
    d1 = [ln(S/K) + (r + σ²/2)*T] / (σ*√T)
    d2 = d1 - σ*√T
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


class OptionType(str, Enum):
    """Option type enumeration."""

    CALL = "call"
    PUT = "put"


@dataclass
class Greeks:
    """Option Greeks container.

    Attributes:
        delta: Rate of change of option price with spot
        gamma: Rate of change of delta with spot
        vega: Sensitivity to volatility (per 1% move)
        theta: Time decay (per day)
        rho: Sensitivity to interest rate (per 1% move)
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


class BlackScholes:
    """Black-Scholes option pricing and Greeks calculator.

    Example:
        >>> bs = BlackScholes(spot=100, rate=0.05)
        >>> price = bs.price(strike=100, expiry=0.25, vol=0.2, option_type="call")
        >>> greeks = bs.greeks(strike=100, expiry=0.25, vol=0.2, option_type="call")
        >>> print(f"Call price: ${price:.2f}, Delta: {greeks.delta:.3f}")
    """

    def __init__(
        self,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
    ):
        """Initialize Black-Scholes calculator.

        Args:
            spot: Current spot price
            rate: Risk-free interest rate (annualized, continuous)
            dividend_yield: Continuous dividend yield
        """
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield

    def _d1_d2(
        self,
        strike: float,
        expiry: float,
        vol: float,
    ) -> tuple[float, float]:
        """Calculate d1 and d2 parameters.

        d1 = [ln(S/K) + (r - q + σ²/2)*T] / (σ*√T)
        d2 = d1 - σ*√T

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility (annualized)

        Returns:
            Tuple of (d1, d2)
        """
        if expiry <= 0:
            raise ValueError("Expiry must be positive")
        if vol <= 0:
            raise ValueError("Volatility must be positive")

        sqrt_t = np.sqrt(expiry)
        r_adj = self.rate - self.dividend_yield

        d1 = (np.log(self.spot / strike) + (r_adj + vol**2 / 2) * expiry) / (
            vol * sqrt_t
        )
        d2 = d1 - vol * sqrt_t

        return d1, d2

    def price(
        self,
        strike: float,
        expiry: float,
        vol: float,
        option_type: Literal["call", "put"] | OptionType = OptionType.CALL,
    ) -> float:
        """Calculate option price using Black-Scholes formula.

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility (annualized)
            option_type: 'call' or 'put'

        Returns:
            Option price

        Example:
            >>> bs = BlackScholes(spot=100, rate=0.05)
            >>> call_price = bs.price(100, 0.25, 0.2, "call")
        """
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())

        d1, d2 = self._d1_d2(strike, expiry, vol)
        discount = np.exp(-self.rate * expiry)
        forward_discount = np.exp(-self.dividend_yield * expiry)

        if option_type == OptionType.CALL:
            return (
                self.spot * forward_discount * norm.cdf(d1)
                - strike * discount * norm.cdf(d2)
            )
        else:
            return (
                strike * discount * norm.cdf(-d2)
                - self.spot * forward_discount * norm.cdf(-d1)
            )

    def delta(
        self,
        strike: float,
        expiry: float,
        vol: float,
        option_type: Literal["call", "put"] | OptionType = OptionType.CALL,
    ) -> float:
        """Calculate option delta.

        Delta = ∂V/∂S
        Call delta = exp(-q*T) * N(d1)
        Put delta = -exp(-q*T) * N(-d1)

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility
            option_type: 'call' or 'put'

        Returns:
            Delta value
        """
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())

        d1, _ = self._d1_d2(strike, expiry, vol)
        forward_discount = np.exp(-self.dividend_yield * expiry)

        if option_type == OptionType.CALL:
            return forward_discount * norm.cdf(d1)
        else:
            return -forward_discount * norm.cdf(-d1)

    def gamma(
        self,
        strike: float,
        expiry: float,
        vol: float,
    ) -> float:
        """Calculate option gamma.

        Gamma = ∂²V/∂S² = N'(d1) / (S * σ * √T)

        Same for both calls and puts.

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility

        Returns:
            Gamma value
        """
        d1, _ = self._d1_d2(strike, expiry, vol)
        forward_discount = np.exp(-self.dividend_yield * expiry)

        return (
            forward_discount
            * norm.pdf(d1)
            / (self.spot * vol * np.sqrt(expiry))
        )

    def vega(
        self,
        strike: float,
        expiry: float,
        vol: float,
    ) -> float:
        """Calculate option vega (sensitivity to volatility).

        Vega = S * N'(d1) * √T * exp(-q*T)

        Returns vega per 1% move in vol (divided by 100).

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility

        Returns:
            Vega value (per 1% vol move)
        """
        d1, _ = self._d1_d2(strike, expiry, vol)
        forward_discount = np.exp(-self.dividend_yield * expiry)

        # Vega per 1% move
        return (
            self.spot * forward_discount * norm.pdf(d1) * np.sqrt(expiry) / 100
        )

    def theta(
        self,
        strike: float,
        expiry: float,
        vol: float,
        option_type: Literal["call", "put"] | OptionType = OptionType.CALL,
    ) -> float:
        """Calculate option theta (time decay).

        Returns theta per calendar day (divided by 365).

        Theta = -S*N'(d1)*σ*exp(-q*T)/(2√T) - r*K*exp(-r*T)*N(d2) + q*S*exp(-q*T)*N(d1)
            (for calls, sign adjustments for puts)

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility
            option_type: 'call' or 'put'

        Returns:
            Theta value (per calendar day)
        """
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())

        d1, d2 = self._d1_d2(strike, expiry, vol)
        sqrt_t = np.sqrt(expiry)
        forward_discount = np.exp(-self.dividend_yield * expiry)
        discount = np.exp(-self.rate * expiry)

        # Common term
        term1 = (
            -self.spot * forward_discount * norm.pdf(d1) * vol / (2 * sqrt_t)
        )

        if option_type == OptionType.CALL:
            theta_annual = (
                term1
                - self.rate * strike * discount * norm.cdf(d2)
                + self.dividend_yield * self.spot * forward_discount * norm.cdf(d1)
            )
        else:
            theta_annual = (
                term1
                + self.rate * strike * discount * norm.cdf(-d2)
                - self.dividend_yield * self.spot * forward_discount * norm.cdf(-d1)
            )

        # Convert to per-day
        return theta_annual / 365

    def rho(
        self,
        strike: float,
        expiry: float,
        vol: float,
        option_type: Literal["call", "put"] | OptionType = OptionType.CALL,
    ) -> float:
        """Calculate option rho (sensitivity to interest rate).

        Returns rho per 1% move in rate (divided by 100).

        Call rho = K * T * exp(-r*T) * N(d2)
        Put rho = -K * T * exp(-r*T) * N(-d2)

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility
            option_type: 'call' or 'put'

        Returns:
            Rho value (per 1% rate move)
        """
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())

        _, d2 = self._d1_d2(strike, expiry, vol)
        discount = np.exp(-self.rate * expiry)

        if option_type == OptionType.CALL:
            return strike * expiry * discount * norm.cdf(d2) / 100
        else:
            return -strike * expiry * discount * norm.cdf(-d2) / 100

    def greeks(
        self,
        strike: float,
        expiry: float,
        vol: float,
        option_type: Literal["call", "put"] | OptionType = OptionType.CALL,
    ) -> Greeks:
        """Calculate all Greeks for an option.

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility
            option_type: 'call' or 'put'

        Returns:
            Greeks object with all sensitivities

        Example:
            >>> bs = BlackScholes(spot=100, rate=0.05)
            >>> g = bs.greeks(100, 0.25, 0.2, "call")
            >>> print(f"Delta: {g.delta:.3f}, Gamma: {g.gamma:.4f}")
        """
        return Greeks(
            delta=self.delta(strike, expiry, vol, option_type),
            gamma=self.gamma(strike, expiry, vol),
            vega=self.vega(strike, expiry, vol),
            theta=self.theta(strike, expiry, vol, option_type),
            rho=self.rho(strike, expiry, vol, option_type),
        )

    def implied_volatility(
        self,
        market_price: float,
        strike: float,
        expiry: float,
        option_type: Literal["call", "put"] | OptionType = OptionType.CALL,
        vol_bounds: tuple[float, float] = (0.001, 5.0),
    ) -> float:
        """Calculate implied volatility from market price.

        Uses Brent's method to solve for vol such that
        BS_price(vol) = market_price.

        Args:
            market_price: Observed market price
            strike: Strike price
            expiry: Time to expiry in years
            option_type: 'call' or 'put'
            vol_bounds: Search bounds for volatility

        Returns:
            Implied volatility

        Raises:
            ValueError: If no solution found in bounds

        Example:
            >>> bs = BlackScholes(spot=100, rate=0.05)
            >>> iv = bs.implied_volatility(5.50, 100, 0.25, "call")
        """
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())

        def objective(vol: float) -> float:
            return self.price(strike, expiry, vol, option_type) - market_price

        try:
            iv = brentq(objective, vol_bounds[0], vol_bounds[1])
            return float(iv)
        except ValueError:
            raise ValueError(
                f"Could not find IV in range {vol_bounds} for price {market_price}"
            )

    def forward_price(self, expiry: float) -> float:
        """Calculate forward price.

        F = S * exp((r - q) * T)

        Args:
            expiry: Time to expiry in years

        Returns:
            Forward price
        """
        return self.spot * np.exp((self.rate - self.dividend_yield) * expiry)

    def strike_from_delta(
        self,
        delta: float,
        expiry: float,
        vol: float,
        option_type: Literal["call", "put"] | OptionType = OptionType.CALL,
    ) -> float:
        """Find strike corresponding to a given delta.

        Args:
            delta: Target delta (e.g., 0.25 for 25-delta)
            expiry: Time to expiry in years
            vol: Implied volatility
            option_type: 'call' or 'put'

        Returns:
            Strike price corresponding to the delta

        Example:
            >>> bs = BlackScholes(spot=100, rate=0.05)
            >>> k25d = bs.strike_from_delta(0.25, 0.25, 0.2, "put")
        """
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())

        forward_discount = np.exp(-self.dividend_yield * expiry)
        sqrt_t = np.sqrt(expiry)

        if option_type == OptionType.CALL:
            # For call: delta = exp(-q*T) * N(d1)
            # So N(d1) = delta / exp(-q*T)
            # d1 = N_inv(delta / exp(-q*T))
            d1 = norm.ppf(delta / forward_discount)
        else:
            # For put: delta = -exp(-q*T) * N(-d1)
            # So N(-d1) = -delta / exp(-q*T)
            # -d1 = N_inv(-delta / exp(-q*T))
            d1 = -norm.ppf(-delta / forward_discount)

        # K = S * exp(-(d1*σ√T - (r-q+σ²/2)*T))
        forward = self.forward_price(expiry)
        strike = forward * np.exp(-(d1 * vol * sqrt_t - vol**2 * expiry / 2))

        return strike


def vectorized_price(
    spots: NDArray[np.float64],
    strikes: NDArray[np.float64],
    expiries: NDArray[np.float64],
    vols: NDArray[np.float64],
    rates: NDArray[np.float64],
    option_types: NDArray[np.bool_],  # True for call, False for put
) -> NDArray[np.float64]:
    """Vectorized Black-Scholes pricing.

    Args:
        spots: Spot prices
        strikes: Strike prices
        expiries: Times to expiry
        vols: Implied volatilities
        rates: Risk-free rates
        option_types: True for call, False for put

    Returns:
        Option prices
    """
    sqrt_t = np.sqrt(expiries)
    d1 = (np.log(spots / strikes) + (rates + vols**2 / 2) * expiries) / (vols * sqrt_t)
    d2 = d1 - vols * sqrt_t

    discount = np.exp(-rates * expiries)

    call_prices = spots * norm.cdf(d1) - strikes * discount * norm.cdf(d2)
    put_prices = strikes * discount * norm.cdf(-d2) - spots * norm.cdf(-d1)

    return np.where(option_types, call_prices, put_prices)
