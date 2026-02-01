"""QuantLib utility functions and shared components.

This module provides common QuantLib setup and helper functions used
across the analytics modules. It handles:
- Date management and calendar setup
- Quote and Handle construction
- Term structure building
- Common type conversions

QuantLib Architecture Overview:
==============================

QuantLib uses a layered architecture:

1. MARKET DATA LAYER (Quotes, Handles)
   - SimpleQuote: Observable market values (spot, vol, rate)
   - Handle: Smart pointers that allow updates to propagate

2. TERM STRUCTURE LAYER
   - YieldTermStructure: Interest rate curves
   - BlackVolTermStructure: Volatility surfaces

3. PROCESS LAYER
   - StochasticProcess: Defines underlying dynamics
   - BlackScholesMertonProcess: GBM with dividends

4. PRICING ENGINE LAYER
   - AnalyticEuropeanEngine: Closed-form solutions
   - FdBlackScholesVanillaEngine: Finite difference
   - MCEuropeanEngine: Monte Carlo

5. INSTRUMENT LAYER
   - VanillaOption, BarrierOption, etc.
   - Instruments use engines to compute NPV and Greeks
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Tuple
from enum import Enum

import QuantLib as ql

logger = logging.getLogger(__name__)


class DayCountConvention(Enum):
    """Common day count conventions."""

    ACTUAL_365 = "Actual365Fixed"
    ACTUAL_360 = "Actual360"
    THIRTY_360 = "Thirty360"
    ACT_ACT = "ActualActual"


@dataclass
class MarketData:
    """Container for market data quotes.

    QuantLib uses Quotes and Handles for market data:
    - Quote: An observable value that can change
    - Handle: A pointer to a Quote that propagates changes

    This allows repricing when market data updates.

    Attributes:
        spot: Current underlying price
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Implied volatility (can be scalar or surface)
    """

    spot: float
    rate: float = 0.05
    dividend_yield: float = 0.0
    volatility: float = 0.20


class QuantLibSetup:
    """Manages QuantLib global settings and common setup.

    QuantLib requires careful management of:
    - Evaluation date (the "today" for all calculations)
    - Calendar (business day conventions)
    - Day count conventions

    Example:
        >>> setup = QuantLibSetup()
        >>> setup.set_evaluation_date(date(2024, 1, 15))
        >>> engine = setup.create_bs_engine(spot=100, rate=0.05, vol=0.20)
    """

    def __init__(
        self,
        calendar: ql.Calendar = None,
        day_count: ql.DayCounter = None,
    ):
        """Initialize QuantLib setup.

        Args:
            calendar: Business calendar (default: US NYSE)
            day_count: Day count convention (default: Actual/365)
        """
        self.calendar = calendar or ql.UnitedStates(ql.UnitedStates.NYSE)
        self.day_count = day_count or ql.Actual365Fixed()

        # Set evaluation date to today
        self._eval_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = self._eval_date

    @property
    def evaluation_date(self) -> ql.Date:
        """Get current evaluation date."""
        return self._eval_date

    def set_evaluation_date(self, dt: date | datetime | ql.Date) -> None:
        """Set the evaluation date for all QuantLib calculations.

        The evaluation date is "today" for pricing purposes.
        All term structures and option expiries are relative to this date.

        Args:
            dt: The evaluation date
        """
        if isinstance(dt, datetime):
            dt = dt.date()
        if isinstance(dt, date):
            dt = ql.Date(dt.day, dt.month, dt.year)

        self._eval_date = dt
        ql.Settings.instance().evaluationDate = dt
        logger.debug(f"Evaluation date set to {dt}")

    def to_ql_date(self, dt: date | datetime | ql.Date) -> ql.Date:
        """Convert Python date to QuantLib Date.

        Args:
            dt: Date to convert

        Returns:
            QuantLib Date object
        """
        if isinstance(dt, ql.Date):
            return dt
        if isinstance(dt, datetime):
            dt = dt.date()
        return ql.Date(dt.day, dt.month, dt.year)

    def from_ql_date(self, ql_date: ql.Date) -> date:
        """Convert QuantLib Date to Python date.

        Args:
            ql_date: QuantLib Date to convert

        Returns:
            Python date object
        """
        return date(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())

    def date_from_tenor(self, tenor_days: int) -> ql.Date:
        """Get a QuantLib Date from days from evaluation date.

        Args:
            tenor_days: Number of days from evaluation date

        Returns:
            QuantLib Date
        """
        return self._eval_date + ql.Period(tenor_days, ql.Days)

    def year_fraction(self, start: ql.Date, end: ql.Date) -> float:
        """Calculate year fraction between two dates.

        Uses the configured day count convention.

        Args:
            start: Start date
            end: End date

        Returns:
            Year fraction as float
        """
        return self.day_count.yearFraction(start, end)


def create_quote(value: float) -> Tuple[ql.SimpleQuote, ql.QuoteHandle]:
    """Create a QuantLib Quote and its Handle.

    Quotes are observable values in QuantLib. When a Quote changes,
    all dependent calculations automatically update.

    The Handle is a smart pointer that allows the Quote to be
    passed around while maintaining the update chain.

    Args:
        value: The initial value

    Returns:
        Tuple of (SimpleQuote, QuoteHandle)

    Example:
        >>> quote, handle = create_quote(100.0)
        >>> # Later, update the value:
        >>> quote.setValue(101.0)
        >>> # All calculations using 'handle' auto-update
    """
    quote = ql.SimpleQuote(value)
    handle = ql.QuoteHandle(quote)
    return quote, handle


def create_flat_yield_curve(
    rate: float,
    eval_date: ql.Date,
    day_count: ql.DayCounter = None,
) -> ql.YieldTermStructureHandle:
    """Create a flat yield term structure.

    A YieldTermStructure represents interest rates over time.
    This creates the simplest case: a flat (constant) rate.

    For more complex curves, use:
    - ZeroCurve: From zero rates at specific tenors
    - ForwardCurve: From forward rates
    - FittedBondDiscountCurve: Fitted to bond prices

    Args:
        rate: Constant interest rate
        eval_date: Evaluation date
        day_count: Day count convention

    Returns:
        Handle to the yield term structure

    Example:
        >>> curve = create_flat_yield_curve(0.05, ql.Date.todaysDate())
        >>> # Get discount factor for 1 year
        >>> df = curve.discount(1.0)
    """
    day_count = day_count or ql.Actual365Fixed()

    flat_curve = ql.FlatForward(eval_date, rate, day_count)
    return ql.YieldTermStructureHandle(flat_curve)


def create_flat_vol_surface(
    volatility: float,
    eval_date: ql.Date,
    calendar: ql.Calendar = None,
    day_count: ql.DayCounter = None,
) -> ql.BlackVolTermStructureHandle:
    """Create a flat (constant) volatility surface.

    A BlackVolTermStructure represents implied volatility over
    strikes and expiries. This creates the simplest case.

    For real vol surfaces, use:
    - BlackVarianceSurface: Interpolated from market data
    - LocalVolSurface: Local volatility (Dupire)
    - HestonBlackVolSurface: From Heston model

    Args:
        volatility: Constant implied volatility
        eval_date: Evaluation date
        calendar: Business calendar
        day_count: Day count convention

    Returns:
        Handle to the volatility surface

    Example:
        >>> vol_surface = create_flat_vol_surface(0.20, ql.Date.todaysDate())
        >>> # Get vol for 1 year expiry
        >>> vol = vol_surface.blackVol(1.0, 100.0)
    """
    calendar = calendar or ql.NullCalendar()
    day_count = day_count or ql.Actual365Fixed()

    flat_vol = ql.BlackConstantVol(eval_date, calendar, volatility, day_count)
    return ql.BlackVolTermStructureHandle(flat_vol)


def create_black_scholes_process(
    spot: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    eval_date: ql.Date,
) -> Tuple[ql.BlackScholesMertonProcess, dict]:
    """Create a Black-Scholes-Merton process.

    The BSM process models the underlying asset dynamics:
        dS = (r - q) * S * dt + σ * S * dW

    Where:
        S = spot price
        r = risk-free rate
        q = dividend yield
        σ = volatility
        W = Brownian motion

    This process is used by pricing engines to value derivatives.

    Args:
        spot: Current underlying price
        rate: Risk-free interest rate
        dividend_yield: Continuous dividend yield
        volatility: Implied volatility
        eval_date: Evaluation date

    Returns:
        Tuple of (BSM process, dict of quotes for updating)

    Example:
        >>> process, quotes = create_black_scholes_process(
        ...     spot=100, rate=0.05, dividend_yield=0.02,
        ...     volatility=0.20, eval_date=ql.Date.todaysDate()
        ... )
        >>> # Update spot price later:
        >>> quotes['spot'].setValue(105.0)
    """
    # Create quotes (allows updates)
    spot_quote = ql.SimpleQuote(spot)

    # Create handles
    spot_handle = ql.QuoteHandle(spot_quote)

    # Term structures
    rate_curve = create_flat_yield_curve(rate, eval_date)
    div_curve = create_flat_yield_curve(dividend_yield, eval_date)
    vol_surface = create_flat_vol_surface(volatility, eval_date)

    # Create the process
    process = ql.BlackScholesMertonProcess(
        spot_handle,
        div_curve,
        rate_curve,
        vol_surface,
    )

    quotes = {'spot': spot_quote}

    return process, quotes


def create_vanilla_option(
    strike: float,
    expiry_date: ql.Date,
    option_type: str = "call",
) -> ql.VanillaOption:
    """Create a European vanilla option.

    A VanillaOption in QuantLib consists of:
    - Payoff: Defines the option type and strike
    - Exercise: When the option can be exercised

    For other option types:
    - BarrierOption: Knock-in/knock-out barriers
    - AsianOption: Average price options
    - LookbackOption: Path-dependent on extremes

    Args:
        strike: Strike price
        expiry_date: Expiration date
        option_type: "call" or "put"

    Returns:
        QuantLib VanillaOption object

    Example:
        >>> option = create_vanilla_option(100.0, expiry, "call")
        >>> option.setPricingEngine(engine)
        >>> price = option.NPV()
    """
    ql_type = ql.Option.Call if option_type.lower() == "call" else ql.Option.Put

    payoff = ql.PlainVanillaPayoff(ql_type, strike)
    exercise = ql.EuropeanExercise(expiry_date)

    return ql.VanillaOption(payoff, exercise)


def create_american_option(
    strike: float,
    expiry_date: ql.Date,
    option_type: str = "call",
) -> ql.VanillaOption:
    """Create an American vanilla option.

    American options can be exercised at any time before expiry.
    They require numerical methods (finite difference, binomial tree)
    rather than closed-form solutions.

    Args:
        strike: Strike price
        expiry_date: Expiration date
        option_type: "call" or "put"

    Returns:
        QuantLib VanillaOption with American exercise
    """
    ql_type = ql.Option.Call if option_type.lower() == "call" else ql.Option.Put

    payoff = ql.PlainVanillaPayoff(ql_type, strike)
    exercise = ql.AmericanExercise(ql.Settings.instance().evaluationDate, expiry_date)

    return ql.VanillaOption(payoff, exercise)


class PricingEngineFactory:
    """Factory for creating QuantLib pricing engines.

    QuantLib separates the instrument from the pricing method.
    An Engine encapsulates the numerical method used to price.

    Engine Types:
    - Analytic: Closed-form solutions (fastest, when available)
    - Finite Difference: PDE solvers (American options, barriers)
    - Monte Carlo: Simulation (path-dependent, complex payoffs)
    - Binomial Tree: Lattice methods (American options)

    Example:
        >>> factory = PricingEngineFactory(process)
        >>> engine = factory.analytic_european()
        >>> option.setPricingEngine(engine)
    """

    def __init__(self, process: ql.BlackScholesMertonProcess):
        """Initialize with a stochastic process.

        Args:
            process: The underlying stochastic process
        """
        self.process = process

    def analytic_european(self) -> ql.PricingEngine:
        """Create analytic European engine (Black-Scholes formula).

        This is the fastest method for European options.
        Uses the closed-form Black-Scholes-Merton formula.

        Returns:
            AnalyticEuropeanEngine
        """
        return ql.AnalyticEuropeanEngine(self.process)

    def finite_difference(
        self,
        time_steps: int = 100,
        grid_points: int = 100,
    ) -> ql.PricingEngine:
        """Create finite difference engine.

        Solves the Black-Scholes PDE numerically.
        Required for American options and some exotics.

        Args:
            time_steps: Number of time steps in the grid
            grid_points: Number of spot price grid points

        Returns:
            FdBlackScholesVanillaEngine
        """
        return ql.FdBlackScholesVanillaEngine(
            self.process,
            time_steps,
            grid_points
        )

    def monte_carlo(
        self,
        num_paths: int = 10000,
        seed: int = 42,
    ) -> ql.PricingEngine:
        """Create Monte Carlo engine.

        Simulates price paths and averages payoffs.
        Most flexible but slowest method.

        Args:
            num_paths: Number of simulation paths
            seed: Random seed for reproducibility

        Returns:
            MCEuropeanEngine
        """
        return ql.MCEuropeanEngine(
            self.process,
            "PseudoRandom",
            timeSteps=1,
            requiredSamples=num_paths,
            seed=seed,
        )

    def binomial_tree(
        self,
        steps: int = 100,
        tree_type: str = "crr",
    ) -> ql.PricingEngine:
        """Create binomial tree engine.

        Builds a recombining tree of possible prices.
        Good for American options.

        Tree types:
        - "crr": Cox-Ross-Rubinstein
        - "jr": Jarrow-Rudd
        - "eqp": Equal probabilities

        Args:
            steps: Number of tree steps
            tree_type: Type of binomial tree

        Returns:
            BinomialVanillaEngine
        """
        tree_map = {
            "crr": "CoxRossRubinstein",
            "jr": "JarrowRudd",
            "eqp": "EqualProbabilities",
        }
        tree_name = tree_map.get(tree_type.lower(), "CoxRossRubinstein")

        return ql.BinomialVanillaEngine(self.process, tree_name, steps)
