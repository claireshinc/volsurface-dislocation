"""SVI (Stochastic Volatility Inspired) model for volatility surface fitting.

The SVI parameterization models total variance w(k) as a function of log-moneyness k:
    w(k) = a + b * [ρ*(k-m) + sqrt((k-m)² + σ²)]

Where:
    - k = log(K/F) is log-moneyness (K=strike, F=forward)
    - a = level of variance (vertical shift)
    - b = slope magnitude (controls wing steepness)
    - ρ = rotation parameter (-1 to 1, controls skew direction)
    - m = translation (horizontal shift of the smile)
    - σ = smoothness (controls ATM curvature)
"""

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, minimize

logger = logging.getLogger(__name__)


@dataclass
class SVIParams:
    """SVI model parameters.

    Attributes:
        a: Level of variance (vertical shift)
        b: Slope magnitude (wing steepness)
        rho: Rotation parameter (-1 to 1, skew direction)
        m: Translation (horizontal shift)
        sigma: Smoothness (ATM curvature)
        expiry_years: Time to expiry in years
        fit_error: Root mean squared error of the fit
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    expiry_years: float
    fit_error: float | None = None

    def to_dict(self) -> dict:
        """Convert parameters to dictionary."""
        return {
            "a": self.a,
            "b": self.b,
            "rho": self.rho,
            "m": self.m,
            "sigma": self.sigma,
            "expiry_years": self.expiry_years,
            "fit_error": self.fit_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SVIParams":
        """Create SVIParams from dictionary."""
        return cls(
            a=data["a"],
            b=data["b"],
            rho=data["rho"],
            m=data["m"],
            sigma=data["sigma"],
            expiry_years=data["expiry_years"],
            fit_error=data.get("fit_error"),
        )


class SVIFitter:
    """Fits the SVI model to market implied volatility data.

    Uses a two-stage optimization:
    1. Global search with differential evolution
    2. Local refinement with L-BFGS-B

    Example:
        >>> fitter = SVIFitter()
        >>> log_moneyness = np.array([-0.1, -0.05, 0, 0.05, 0.1])
        >>> total_variance = np.array([0.06, 0.045, 0.04, 0.045, 0.055])
        >>> params = fitter.fit(log_moneyness, total_variance, expiry_years=0.25)
        >>> print(f"ATM variance: {params.a:.4f}")
    """

    # Parameter bounds: (a, b, rho, m, sigma)
    DEFAULT_BOUNDS = [
        (-0.5, 0.5),   # a: variance level
        (0.01, 1.0),   # b: slope (must be positive)
        (-0.99, 0.99), # rho: rotation
        (-0.5, 0.5),   # m: translation
        (0.01, 1.0),   # sigma: smoothness (must be positive)
    ]

    def __init__(
        self,
        bounds: list[tuple[float, float]] | None = None,
        max_global_iter: int = 100,
        enforce_arbitrage_free: bool = True,
    ):
        """Initialize the SVI fitter.

        Args:
            bounds: Parameter bounds [(a_min, a_max), ...]. Uses defaults if None.
            max_global_iter: Maximum iterations for differential evolution.
            enforce_arbitrage_free: Whether to enforce no-arbitrage constraints.
        """
        self.bounds = bounds or self.DEFAULT_BOUNDS
        self.max_global_iter = max_global_iter
        self.enforce_arbitrage_free = enforce_arbitrage_free

    @staticmethod
    def svi_total_variance(
        k: NDArray[np.float64],
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float,
    ) -> NDArray[np.float64]:
        """Calculate SVI total variance for given log-moneyness.

        The SVI formula:
            w(k) = a + b * [ρ*(k-m) + sqrt((k-m)² + σ²)]

        Args:
            k: Log-moneyness values, k = log(K/F)
            a: Level of variance
            b: Slope magnitude
            rho: Rotation parameter
            m: Translation
            sigma: Smoothness

        Returns:
            Total variance w(k) for each input k

        Example:
            >>> k = np.array([-0.1, 0, 0.1])
            >>> w = SVIFitter.svi_total_variance(k, 0.04, 0.1, -0.3, 0, 0.1)
        """
        k_shifted = k - m
        return a + b * (rho * k_shifted + np.sqrt(k_shifted**2 + sigma**2))

    @staticmethod
    def svi_implied_vol(
        k: NDArray[np.float64],
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float,
        expiry_years: float,
    ) -> NDArray[np.float64]:
        """Calculate implied volatility from SVI parameters.

        Implied volatility is sqrt(total_variance / T).

        Args:
            k: Log-moneyness values
            a, b, rho, m, sigma: SVI parameters
            expiry_years: Time to expiry in years

        Returns:
            Implied volatility for each k
        """
        w = SVIFitter.svi_total_variance(k, a, b, rho, m, sigma)
        # Ensure non-negative variance before sqrt
        w = np.maximum(w, 1e-10)
        return np.sqrt(w / expiry_years)

    def _check_arbitrage_free(
        self, a: float, b: float, rho: float, sigma: float
    ) -> bool:
        """Check if parameters satisfy no-arbitrage constraint.

        The constraint is: a + b*σ*sqrt(1-ρ²) >= 0

        This ensures the minimum of the SVI curve is non-negative,
        preventing negative variance.
        """
        min_variance = a + b * sigma * np.sqrt(1 - rho**2)
        return min_variance >= 0

    def _objective(
        self,
        params: NDArray[np.float64],
        k: NDArray[np.float64],
        target_variance: NDArray[np.float64],
        weights: NDArray[np.float64] | None,
    ) -> float:
        """Calculate weighted mean squared error objective.

        Args:
            params: SVI parameters [a, b, rho, m, sigma]
            k: Log-moneyness values
            target_variance: Market total variance values
            weights: Optional weights for each observation

        Returns:
            Weighted mean squared error (with penalty if arbitrage-violating)
        """
        a, b, rho, m, sigma = params

        # Arbitrage penalty
        if self.enforce_arbitrage_free:
            if not self._check_arbitrage_free(a, b, rho, sigma):
                return 1e10  # Large penalty

        model_variance = self.svi_total_variance(k, a, b, rho, m, sigma)

        # Ensure model variance is positive
        if np.any(model_variance < 0):
            return 1e10

        errors = (model_variance - target_variance) ** 2

        if weights is not None:
            return float(np.sum(weights * errors) / np.sum(weights))
        return float(np.mean(errors))

    def fit(
        self,
        log_moneyness: NDArray[np.float64],
        total_variance: NDArray[np.float64],
        expiry_years: float,
        weights: NDArray[np.float64] | None = None,
        initial_guess: NDArray[np.float64] | None = None,
    ) -> SVIParams:
        """Fit SVI model to market data.

        Uses two-stage optimization:
        1. Differential evolution for global search
        2. L-BFGS-B for local refinement

        Args:
            log_moneyness: Log-moneyness values k = log(K/F)
            total_variance: Market total variance w = σ²*T
            expiry_years: Time to expiry in years
            weights: Optional weights for observations (e.g., by vega)
            initial_guess: Optional initial parameter guess [a, b, rho, m, sigma]

        Returns:
            SVIParams with fitted parameters and fit error

        Example:
            >>> fitter = SVIFitter()
            >>> k = np.array([-0.2, -0.1, 0, 0.1, 0.2])
            >>> w = np.array([0.08, 0.05, 0.04, 0.05, 0.07])
            >>> params = fitter.fit(k, w, expiry_years=0.25)
        """
        if len(log_moneyness) != len(total_variance):
            raise ValueError("log_moneyness and total_variance must have same length")

        if len(log_moneyness) < 5:
            raise ValueError("Need at least 5 data points to fit SVI model")

        # Convert to numpy arrays
        k = np.asarray(log_moneyness, dtype=np.float64)
        w = np.asarray(total_variance, dtype=np.float64)

        # Stage 1: Global optimization with differential evolution
        logger.debug("Starting global optimization with differential evolution")

        objective: Callable[[NDArray[np.float64]], float] = (
            lambda p: self._objective(p, k, w, weights)
        )

        global_result = differential_evolution(
            objective,
            bounds=self.bounds,
            maxiter=self.max_global_iter,
            seed=42,
            polish=False,
            disp=False,
        )

        # Stage 2: Local refinement with L-BFGS-B
        logger.debug("Starting local refinement with L-BFGS-B")

        x0 = initial_guess if initial_guess is not None else global_result.x

        local_result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=self.bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        # Use best result
        if local_result.fun < global_result.fun:
            best_params = local_result.x
            best_error = local_result.fun
        else:
            best_params = global_result.x
            best_error = global_result.fun

        a, b, rho, m, sigma = best_params
        rmse = np.sqrt(best_error)

        logger.info(
            f"SVI fit complete: a={a:.4f}, b={b:.4f}, rho={rho:.4f}, "
            f"m={m:.4f}, sigma={sigma:.4f}, RMSE={rmse:.6f}"
        )

        return SVIParams(
            a=float(a),
            b=float(b),
            rho=float(rho),
            m=float(m),
            sigma=float(sigma),
            expiry_years=expiry_years,
            fit_error=float(rmse),
        )

    def fit_surface(
        self,
        log_moneyness: NDArray[np.float64],
        implied_vols: NDArray[np.float64],
        expiry_years: NDArray[np.float64],
    ) -> list[SVIParams]:
        """Fit SVI model to multiple expiries.

        Args:
            log_moneyness: 2D array of log-moneyness [n_expiries, n_strikes]
            implied_vols: 2D array of implied vols [n_expiries, n_strikes]
            expiry_years: 1D array of expiry times [n_expiries]

        Returns:
            List of SVIParams, one per expiry
        """
        if log_moneyness.shape != implied_vols.shape:
            raise ValueError("log_moneyness and implied_vols must have same shape")

        if len(expiry_years) != log_moneyness.shape[0]:
            raise ValueError("expiry_years length must match first dimension")

        results = []
        for i, T in enumerate(expiry_years):
            k = log_moneyness[i]
            iv = implied_vols[i]

            # Filter out NaN/invalid values
            valid = ~np.isnan(k) & ~np.isnan(iv) & (iv > 0)
            if np.sum(valid) < 5:
                logger.warning(f"Skipping expiry {T:.3f}y: insufficient data points")
                continue

            # Convert IV to total variance
            total_variance = iv[valid] ** 2 * T

            params = self.fit(k[valid], total_variance, T)
            results.append(params)

        return results


def interpolate_svi_surface(
    params_list: list[SVIParams],
    target_expiry: float,
    log_moneyness: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate implied volatility at arbitrary expiry.

    Uses linear interpolation of SVI parameters in the time dimension.

    Args:
        params_list: List of fitted SVI parameters at various expiries
        target_expiry: Target expiry time in years
        log_moneyness: Log-moneyness values for which to compute IV

    Returns:
        Interpolated implied volatility at target expiry
    """
    if len(params_list) < 2:
        raise ValueError("Need at least 2 expiries for interpolation")

    expiries = np.array([p.expiry_years for p in params_list])
    sorted_idx = np.argsort(expiries)
    expiries = expiries[sorted_idx]
    params_list = [params_list[i] for i in sorted_idx]

    # Find bracketing expiries
    if target_expiry <= expiries[0]:
        p = params_list[0]
        return SVIFitter.svi_implied_vol(
            log_moneyness, p.a, p.b, p.rho, p.m, p.sigma, target_expiry
        )
    if target_expiry >= expiries[-1]:
        p = params_list[-1]
        return SVIFitter.svi_implied_vol(
            log_moneyness, p.a, p.b, p.rho, p.m, p.sigma, target_expiry
        )

    # Linear interpolation
    idx = np.searchsorted(expiries, target_expiry)
    t1, t2 = expiries[idx - 1], expiries[idx]
    p1, p2 = params_list[idx - 1], params_list[idx]

    # Interpolation weight
    w = (target_expiry - t1) / (t2 - t1)

    # Interpolate each parameter
    a = p1.a + w * (p2.a - p1.a)
    b = p1.b + w * (p2.b - p1.b)
    rho = p1.rho + w * (p2.rho - p1.rho)
    m = p1.m + w * (p2.m - p1.m)
    sigma = p1.sigma + w * (p2.sigma - p1.sigma)

    return SVIFitter.svi_implied_vol(
        log_moneyness, a, b, rho, m, sigma, target_expiry
    )
