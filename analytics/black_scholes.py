"""Black-Scholes option pricing and Greeks using QuantLib.

This module provides option pricing using QuantLib's robust implementation.
QuantLib handles all the numerical details and edge cases properly.

QuantLib Pricing Flow:
=====================

1. Create MARKET DATA (Quotes)
   └── SimpleQuote for spot, vol, rate

2. Build TERM STRUCTURES
   ├── YieldTermStructure (interest rates)
   └── BlackVolTermStructure (volatility)

3. Create STOCHASTIC PROCESS
   └── BlackScholesMertonProcess (GBM dynamics)

4. Create INSTRUMENT
   └── VanillaOption (payoff + exercise)

5. Attach PRICING ENGINE
   └── AnalyticEuropeanEngine (closed-form BS)

6. PRICE and compute GREEKS
   └── option.NPV(), option.delta(), etc.

Example:
    >>> bs = BlackScholes(spot=100, rate=0.05)
    >>> price = bs.price(strike=100, expiry=0.25, vol=0.2, option_type="call")
    >>> greeks = bs.greeks(strike=100, expiry=0.25, vol=0.2, option_type="call")
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray
import QuantLib as ql

from analytics.quantlib_utils import (
    QuantLibSetup,
    create_black_scholes_process,
    create_vanilla_option,
    create_american_option,
    PricingEngineFactory,
)

logger = logging.getLogger(__name__)


class OptionType(str, Enum):
    """Option type enumeration."""

    CALL = "call"
    PUT = "put"


@dataclass
class Greeks:
    """Option Greeks container.

    Greeks measure option price sensitivity to various factors.
    QuantLib computes these via finite differencing on the pricing model.

    Attributes:
        delta: ∂V/∂S - Sensitivity to underlying price
        gamma: ∂²V/∂S² - Rate of change of delta
        vega: ∂V/∂σ - Sensitivity to volatility (per 1% move)
        theta: ∂V/∂t - Time decay (per day)
        rho: ∂V/∂r - Sensitivity to interest rate (per 1% move)
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
    """Black-Scholes option pricing using QuantLib.

    This class wraps QuantLib's European option pricing functionality
    with a simple interface. It handles all the QuantLib setup internally.

    QuantLib Components Used:
    - BlackScholesMertonProcess: Models underlying dynamics
    - AnalyticEuropeanEngine: Closed-form BS pricing
    - VanillaOption: European option instrument

    Example:
        >>> bs = BlackScholes(spot=100, rate=0.05, dividend_yield=0.02)
        >>>
        >>> # Price a call option
        >>> price = bs.price(strike=105, expiry=0.25, vol=0.2, option_type="call")
        >>> print(f"Call price: ${price:.2f}")
        >>>
        >>> # Get all Greeks
        >>> greeks = bs.greeks(strike=105, expiry=0.25, vol=0.2, option_type="call")
        >>> print(f"Delta: {greeks.delta:.3f}, Gamma: {greeks.gamma:.4f}")
    """

    def __init__(
        self,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
    ):
        """Initialize Black-Scholes calculator.

        Sets up the QuantLib environment and stores market parameters.

        Args:
            spot: Current spot price of the underlying
            rate: Risk-free interest rate (annualized, continuous)
            dividend_yield: Continuous dividend yield
        """
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield

        # Initialize QuantLib setup
        self._setup = QuantLibSetup()
        self._eval_date = ql.Settings.instance().evaluationDate

    def _create_option_and_engine(
        self,
        strike: float,
        expiry: float,
        vol: float,
        option_type: str,
    ) -> ql.VanillaOption:
        """Create a priced QuantLib option.

        This internal method:
        1. Creates the BSM process with current market data
        2. Creates the vanilla option instrument
        3. Attaches the analytic pricing engine
        4. Returns the option ready for pricing

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility
            option_type: "call" or "put"

        Returns:
            QuantLib VanillaOption with engine attached
        """
        # Convert expiry to QuantLib date
        expiry_days = int(expiry * 365)
        expiry_date = self._eval_date + ql.Period(expiry_days, ql.Days)

        # Create the stochastic process
        process, _ = create_black_scholes_process(
            spot=self.spot,
            rate=self.rate,
            dividend_yield=self.dividend_yield,
            volatility=vol,
            eval_date=self._eval_date,
        )

        # Create the option
        option = create_vanilla_option(strike, expiry_date, option_type)

        # Attach the analytic engine
        engine = ql.AnalyticEuropeanEngine(process)
        option.setPricingEngine(engine)

        return option

    def price(
        self,
        strike: float,
        expiry: float,
        vol: float,
        option_type: Literal["call", "put"] | OptionType = OptionType.CALL,
    ) -> float:
        """Calculate option price using QuantLib.

        Uses the AnalyticEuropeanEngine which implements the
        closed-form Black-Scholes-Merton formula.

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility (annualized)
            option_type: 'call' or 'put'

        Returns:
            Option price (NPV - Net Present Value)

        Example:
            >>> bs = BlackScholes(spot=100, rate=0.05)
            >>> call_price = bs.price(100, 0.25, 0.2, "call")
            >>> put_price = bs.price(100, 0.25, 0.2, "put")
        """
        if isinstance(option_type, OptionType):
            option_type = option_type.value

        option = self._create_option_and_engine(strike, expiry, vol, option_type)
        return option.NPV()

    def delta(
        self,
        strike: float,
        expiry: float,
        vol: float,
        option_type: Literal["call", "put"] | OptionType = OptionType.CALL,
    ) -> float:
        """Calculate option delta using QuantLib.

        Delta = ∂V/∂S

        QuantLib computes delta by differentiating the pricing formula
        with respect to the spot price.

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility
            option_type: 'call' or 'put'

        Returns:
            Delta value (call: 0 to 1, put: -1 to 0)
        """
        if isinstance(option_type, OptionType):
            option_type = option_type.value

        option = self._create_option_and_engine(strike, expiry, vol, option_type)
        return option.delta()

    def gamma(
        self,
        strike: float,
        expiry: float,
        vol: float,
    ) -> float:
        """Calculate option gamma using QuantLib.

        Gamma = ∂²V/∂S² = ∂Δ/∂S

        Gamma is the same for calls and puts with same strike/expiry.

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility

        Returns:
            Gamma value (always positive)
        """
        option = self._create_option_and_engine(strike, expiry, vol, "call")
        return option.gamma()

    def vega(
        self,
        strike: float,
        expiry: float,
        vol: float,
    ) -> float:
        """Calculate option vega using QuantLib.

        Vega = ∂V/∂σ

        Returns vega per 1% move in volatility (divided by 100).
        Same for calls and puts.

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility

        Returns:
            Vega value (per 1% vol move)
        """
        option = self._create_option_and_engine(strike, expiry, vol, "call")
        # QuantLib returns vega per 1.0 vol change, we want per 0.01
        return option.vega() / 100

    def theta(
        self,
        strike: float,
        expiry: float,
        vol: float,
        option_type: Literal["call", "put"] | OptionType = OptionType.CALL,
    ) -> float:
        """Calculate option theta using QuantLib.

        Theta = ∂V/∂t

        Returns theta per calendar day (divided by 365).

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility
            option_type: 'call' or 'put'

        Returns:
            Theta value (per calendar day, usually negative)
        """
        if isinstance(option_type, OptionType):
            option_type = option_type.value

        option = self._create_option_and_engine(strike, expiry, vol, option_type)
        # QuantLib returns theta per year, convert to per day
        return option.theta() / 365

    def rho(
        self,
        strike: float,
        expiry: float,
        vol: float,
        option_type: Literal["call", "put"] | OptionType = OptionType.CALL,
    ) -> float:
        """Calculate option rho using QuantLib.

        Rho = ∂V/∂r

        Returns rho per 1% move in rate (divided by 100).

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility
            option_type: 'call' or 'put'

        Returns:
            Rho value (per 1% rate move)
        """
        if isinstance(option_type, OptionType):
            option_type = option_type.value

        option = self._create_option_and_engine(strike, expiry, vol, option_type)
        # QuantLib returns rho per 1.0 rate change, we want per 0.01
        return option.rho() / 100

    def greeks(
        self,
        strike: float,
        expiry: float,
        vol: float,
        option_type: Literal["call", "put"] | OptionType = OptionType.CALL,
    ) -> Greeks:
        """Calculate all Greeks for an option.

        Efficiently computes all Greeks in one call by reusing
        the same QuantLib option object.

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
            >>> print(f"Delta: {g.delta:.3f}")
            >>> print(f"Gamma: {g.gamma:.4f}")
            >>> print(f"Vega:  {g.vega:.2f}")
            >>> print(f"Theta: {g.theta:.3f}")
            >>> print(f"Rho:   {g.rho:.3f}")
        """
        if isinstance(option_type, OptionType):
            option_type = option_type.value

        option = self._create_option_and_engine(strike, expiry, vol, option_type)

        return Greeks(
            delta=option.delta(),
            gamma=option.gamma(),
            vega=option.vega() / 100,  # Per 1% vol
            theta=option.theta() / 365,  # Per day
            rho=option.rho() / 100,  # Per 1% rate
        )

    def implied_volatility(
        self,
        market_price: float,
        strike: float,
        expiry: float,
        option_type: Literal["call", "put"] | OptionType = OptionType.CALL,
        vol_bounds: tuple[float, float] = (0.001, 5.0),
    ) -> float:
        """Calculate implied volatility from market price using QuantLib.

        Uses QuantLib's built-in IV solver which employs Brent's method
        to find the volatility that matches the market price.

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
            >>> price = 5.50
            >>> iv = bs.implied_volatility(price, 100, 0.25, "call")
            >>> print(f"Implied Vol: {iv:.2%}")
        """
        if isinstance(option_type, OptionType):
            option_type = option_type.value

        # Convert expiry to QuantLib date
        expiry_days = int(expiry * 365)
        expiry_date = self._eval_date + ql.Period(expiry_days, ql.Days)

        # Create the option
        option = create_vanilla_option(strike, expiry_date, option_type)

        # Create a process with dummy vol (will be solved for)
        process, _ = create_black_scholes_process(
            spot=self.spot,
            rate=self.rate,
            dividend_yield=self.dividend_yield,
            volatility=0.20,  # Initial guess
            eval_date=self._eval_date,
        )

        # Use QuantLib's implied volatility solver
        engine = ql.AnalyticEuropeanEngine(process)
        option.setPricingEngine(engine)

        try:
            iv = option.impliedVolatility(
                market_price,
                process,
                accuracy=1e-6,
                maxEvaluations=100,
                minVol=vol_bounds[0],
                maxVol=vol_bounds[1],
            )
            return iv
        except RuntimeError as e:
            raise ValueError(
                f"Could not find IV in range {vol_bounds} for price {market_price}: {e}"
            )

    def forward_price(self, expiry: float) -> float:
        """Calculate forward price.

        F = S × exp((r - q) × T)

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

        Uses numerical search to find the strike where
        the option delta equals the target.

        Args:
            delta: Target delta (e.g., 0.25 for 25-delta)
            expiry: Time to expiry in years
            vol: Implied volatility
            option_type: 'call' or 'put'

        Returns:
            Strike price corresponding to the delta

        Example:
            >>> bs = BlackScholes(spot=100, rate=0.05)
            >>> k25d_put = bs.strike_from_delta(0.25, 0.25, 0.2, "put")
            >>> k25d_call = bs.strike_from_delta(0.25, 0.25, 0.2, "call")
        """
        if isinstance(option_type, OptionType):
            option_type = option_type.value

        from scipy.optimize import brentq

        def delta_diff(strike: float) -> float:
            calc_delta = self.delta(strike, expiry, vol, option_type)
            if option_type == "put":
                return abs(calc_delta) - delta
            return calc_delta - delta

        # Search bounds based on option type
        forward = self.forward_price(expiry)

        if option_type == "call":
            # Call delta decreases with strike, search above spot for low delta
            low_strike = forward * 0.5
            high_strike = forward * 2.0
        else:
            # Put delta (absolute) decreases with strike, search below spot for low delta
            low_strike = forward * 0.5
            high_strike = forward * 1.5

        try:
            strike = brentq(delta_diff, low_strike, high_strike)
            return strike
        except ValueError:
            # Fallback to simple approximation
            logger.warning(f"Could not find exact strike for {delta} delta {option_type}")
            from scipy.stats import norm
            sqrt_t = np.sqrt(expiry)

            if option_type == "call":
                d1 = norm.ppf(delta * np.exp(self.dividend_yield * expiry))
            else:
                d1 = -norm.ppf(delta * np.exp(self.dividend_yield * expiry))

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

    Prices multiple options efficiently using QuantLib.

    Note: For very large arrays, consider using numpy-based
    formulas directly as QuantLib has Python overhead per option.

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
    prices = np.zeros(len(spots))

    for i in range(len(spots)):
        bs = BlackScholes(spot=spots[i], rate=rates[i])
        opt_type = "call" if option_types[i] else "put"
        prices[i] = bs.price(strikes[i], expiries[i], vols[i], opt_type)

    return prices


class AmericanOptionPricer:
    """Price American options using QuantLib finite difference methods.

    American options can be exercised at any time before expiry.
    They require numerical methods since no closed-form solution exists.

    QuantLib Methods Used:
    - FdBlackScholesVanillaEngine: Finite difference PDE solver
    - BinomialVanillaEngine: Cox-Ross-Rubinstein tree

    Example:
        >>> pricer = AmericanOptionPricer(spot=100, rate=0.05)
        >>> price = pricer.price(strike=100, expiry=0.25, vol=0.2, option_type="put")
    """

    def __init__(
        self,
        spot: float,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
    ):
        """Initialize American option pricer.

        Args:
            spot: Current spot price
            rate: Risk-free rate
            dividend_yield: Dividend yield
        """
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield
        self._eval_date = ql.Settings.instance().evaluationDate

    def price(
        self,
        strike: float,
        expiry: float,
        vol: float,
        option_type: str = "call",
        method: str = "fd",
        steps: int = 100,
    ) -> float:
        """Price an American option.

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility
            option_type: 'call' or 'put'
            method: 'fd' (finite difference) or 'tree' (binomial)
            steps: Number of time steps

        Returns:
            Option price
        """
        expiry_days = int(expiry * 365)
        expiry_date = self._eval_date + ql.Period(expiry_days, ql.Days)

        # Create process
        process, _ = create_black_scholes_process(
            spot=self.spot,
            rate=self.rate,
            dividend_yield=self.dividend_yield,
            volatility=vol,
            eval_date=self._eval_date,
        )

        # Create American option
        option = create_american_option(strike, expiry_date, option_type)

        # Create engine
        factory = PricingEngineFactory(process)

        if method == "fd":
            engine = factory.finite_difference(time_steps=steps, grid_points=steps)
        else:
            engine = factory.binomial_tree(steps=steps)

        option.setPricingEngine(engine)

        return option.NPV()

    def greeks(
        self,
        strike: float,
        expiry: float,
        vol: float,
        option_type: str = "call",
        method: str = "fd",
        steps: int = 100,
    ) -> Greeks:
        """Calculate Greeks for an American option.

        Uses finite differencing to compute sensitivities since
        analytical Greeks aren't available for American options.

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility
            option_type: 'call' or 'put'
            method: 'fd' or 'binomial'
            steps: Number of steps for numerical method

        Returns:
            Greeks object with delta, gamma, vega, theta, rho
        """
        expiry_days = int(expiry * 365)
        expiry_date = self._eval_date + ql.Period(expiry_days, ql.Days)

        # Create process
        process, _ = create_black_scholes_process(
            spot=self.spot,
            rate=self.rate,
            dividend_yield=self.dividend_yield,
            volatility=vol,
            eval_date=self._eval_date,
        )

        # Create American option
        option = create_american_option(strike, expiry_date, option_type)

        # Create engine
        factory = PricingEngineFactory(process)
        if method == "fd":
            engine = factory.finite_difference(time_steps=steps, grid_points=steps)
        else:
            engine = factory.binomial_tree(steps=steps)

        option.setPricingEngine(engine)

        # Compute Greeks via finite differencing
        h_spot = self.spot * 0.01  # 1% bump for delta/gamma
        h_vol = 0.01  # 1% bump for vega
        h_time = 1 / 365  # 1 day for theta
        h_rate = 0.0001  # 1bp for rho

        base_price = option.NPV()

        # Delta and Gamma via spot bump
        pricer_up = AmericanOptionPricer(self.spot + h_spot, self.rate, self.dividend_yield)
        pricer_down = AmericanOptionPricer(self.spot - h_spot, self.rate, self.dividend_yield)
        price_up = pricer_up.price(strike, expiry, vol, option_type, method, steps)
        price_down = pricer_down.price(strike, expiry, vol, option_type, method, steps)
        delta = (price_up - price_down) / (2 * h_spot)
        gamma = (price_up - 2 * base_price + price_down) / (h_spot ** 2)

        # Vega via vol bump
        price_vol_up = self.price(strike, expiry, vol + h_vol, option_type, method, steps)
        vega = (price_vol_up - base_price) / (h_vol * 100)  # Per 1% vol

        # Theta via time decay
        if expiry > h_time:
            price_theta = self.price(strike, expiry - h_time, vol, option_type, method, steps)
            theta = (price_theta - base_price)  # Already per day
        else:
            theta = -base_price / expiry * h_time

        # Rho via rate bump
        pricer_rho = AmericanOptionPricer(self.spot, self.rate + h_rate, self.dividend_yield)
        price_rho_up = pricer_rho.price(strike, expiry, vol, option_type, method, steps)
        rho = (price_rho_up - base_price) / (h_rate * 100)  # Per 1%

        return Greeks(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
        )

    def early_exercise_premium(
        self,
        strike: float,
        expiry: float,
        vol: float,
        option_type: str = "call",
    ) -> float:
        """Calculate the early exercise premium.

        Premium = American Price - European Price

        This represents the value of being able to exercise early.
        For calls on non-dividend paying stocks, this is typically zero.

        Args:
            strike: Strike price
            expiry: Time to expiry in years
            vol: Implied volatility
            option_type: 'call' or 'put'

        Returns:
            Early exercise premium
        """
        american_price = self.price(strike, expiry, vol, option_type)

        bs = BlackScholes(self.spot, self.rate, self.dividend_yield)
        european_price = bs.price(strike, expiry, vol, option_type)

        return american_price - european_price
