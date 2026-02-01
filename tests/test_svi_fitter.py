"""Tests for SVI model fitting.

Tests cover:
- Basic SVI formula computation
- Parameter fitting with synthetic data
- Arbitrage constraint enforcement
- Surface interpolation
"""

import numpy as np
import pytest

from analytics.svi_fitter import SVIFitter, SVIParams, interpolate_svi_surface


class TestSVIFormula:
    """Tests for SVI total variance formula."""

    def test_svi_total_variance_at_atm(self):
        """Test SVI formula at ATM (k=0)."""
        k = np.array([0.0])
        # At k=0, m=0: w(0) = a + b * sqrt(σ²) = a + b*σ
        w = SVIFitter.svi_total_variance(k, a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1)
        expected = 0.04 + 0.1 * 0.1  # a + b*sigma when rho=0, k=m=0
        assert np.isclose(w[0], expected, rtol=1e-6)

    def test_svi_symmetry_with_zero_rho(self):
        """Test that smile is symmetric when rho=0."""
        k = np.array([-0.1, 0.0, 0.1])
        w = SVIFitter.svi_total_variance(k, a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1)
        # With rho=0 and m=0, w(-k) = w(k)
        assert np.isclose(w[0], w[2], rtol=1e-6)

    def test_svi_skew_with_negative_rho(self):
        """Test that negative rho produces downward skew (put wing higher)."""
        k = np.array([-0.1, 0.0, 0.1])
        w = SVIFitter.svi_total_variance(k, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        # Negative rho: put wing (k<0) should have higher vol than call wing
        assert w[0] > w[2]

    def test_svi_implied_vol_positive(self):
        """Test that implied vol is always positive."""
        k = np.linspace(-0.5, 0.5, 21)
        iv = SVIFitter.svi_implied_vol(k, a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1, expiry_years=0.25)
        assert np.all(iv > 0)


class TestSVIFitting:
    """Tests for SVI model fitting."""

    @pytest.fixture
    def synthetic_smile_data(self):
        """Generate synthetic smile data for testing."""
        # True parameters
        true_a = 0.04
        true_b = 0.1
        true_rho = -0.25
        true_m = 0.01
        true_sigma = 0.1
        T = 0.25

        # Generate data points
        np.random.seed(42)
        k = np.linspace(-0.2, 0.2, 15)
        w_true = SVIFitter.svi_total_variance(k, true_a, true_b, true_rho, true_m, true_sigma)

        # Add small noise
        noise = np.random.normal(0, 0.001, len(k))
        w_noisy = w_true + noise

        return {
            "k": k,
            "w": w_noisy,
            "T": T,
            "true_params": (true_a, true_b, true_rho, true_m, true_sigma),
        }

    def test_fit_recovers_parameters(self, synthetic_smile_data):
        """Test that fitting recovers true parameters from synthetic data."""
        fitter = SVIFitter()
        params = fitter.fit(
            synthetic_smile_data["k"],
            synthetic_smile_data["w"],
            synthetic_smile_data["T"],
        )

        true_a, true_b, true_rho, true_m, true_sigma = synthetic_smile_data["true_params"]

        # Should recover parameters within reasonable tolerance
        assert np.isclose(params.a, true_a, atol=0.01)
        assert np.isclose(params.b, true_b, atol=0.02)
        assert np.isclose(params.rho, true_rho, atol=0.1)
        assert np.isclose(params.m, true_m, atol=0.02)
        assert np.isclose(params.sigma, true_sigma, atol=0.02)

    def test_fit_error_small(self, synthetic_smile_data):
        """Test that fit error is small for good data."""
        fitter = SVIFitter()
        params = fitter.fit(
            synthetic_smile_data["k"],
            synthetic_smile_data["w"],
            synthetic_smile_data["T"],
        )

        # RMSE should be small
        assert params.fit_error < 0.01

    def test_fit_insufficient_data(self):
        """Test that fitting fails with insufficient data."""
        fitter = SVIFitter()

        with pytest.raises(ValueError, match="at least 5 data points"):
            fitter.fit(
                np.array([0.0, 0.1]),
                np.array([0.04, 0.05]),
                expiry_years=0.25,
            )

    def test_arbitrage_free_constraint(self):
        """Test that no-arbitrage constraint is enforced."""
        fitter = SVIFitter(enforce_arbitrage_free=True)

        # Generate data that might produce arbitrage-violating params
        k = np.linspace(-0.3, 0.3, 15)
        w = np.array([0.1, 0.08, 0.06, 0.05, 0.045, 0.04, 0.038, 0.04,
                      0.045, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15])

        params = fitter.fit(k, w, expiry_years=0.25)

        # Check no-arbitrage: a + b*σ*sqrt(1-ρ²) >= 0
        min_variance = params.a + params.b * params.sigma * np.sqrt(1 - params.rho**2)
        assert min_variance >= -1e-6  # Allow small numerical error


class TestSVIParams:
    """Tests for SVIParams dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        params = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.01, sigma=0.1, expiry_years=0.25)
        d = params.to_dict()

        assert d["a"] == 0.04
        assert d["b"] == 0.1
        assert d["rho"] == -0.3
        assert d["expiry_years"] == 0.25

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {"a": 0.04, "b": 0.1, "rho": -0.3, "m": 0.01, "sigma": 0.1, "expiry_years": 0.25}
        params = SVIParams.from_dict(d)

        assert params.a == 0.04
        assert params.rho == -0.3


class TestSurfaceInterpolation:
    """Tests for surface interpolation."""

    @pytest.fixture
    def params_list(self):
        """Create list of SVI params at different expiries."""
        return [
            SVIParams(a=0.03, b=0.08, rho=-0.2, m=0.0, sigma=0.08, expiry_years=0.08),  # ~30d
            SVIParams(a=0.035, b=0.09, rho=-0.22, m=0.0, sigma=0.09, expiry_years=0.16),  # ~60d
            SVIParams(a=0.04, b=0.1, rho=-0.25, m=0.0, sigma=0.1, expiry_years=0.25),  # ~90d
        ]

    def test_interpolate_between_expiries(self, params_list):
        """Test interpolation between two expiries."""
        k = np.array([0.0])
        target_expiry = 0.12  # Between 30d and 60d

        iv = interpolate_svi_surface(params_list, target_expiry, k)

        # Should be between the two neighboring expiry ATM vols
        iv_30d = SVIFitter.svi_implied_vol(k, 0.03, 0.08, -0.2, 0.0, 0.08, 0.08)[0]
        iv_60d = SVIFitter.svi_implied_vol(k, 0.035, 0.09, -0.22, 0.0, 0.09, 0.16)[0]

        assert iv_30d < iv[0] < iv_60d or iv_60d < iv[0] < iv_30d

    def test_extrapolate_before_first(self, params_list):
        """Test extrapolation before first expiry."""
        k = np.array([0.0])
        iv = interpolate_svi_surface(params_list, 0.05, k)  # Before 30d

        # Should use first expiry params
        assert iv[0] > 0

    def test_insufficient_params(self):
        """Test error with insufficient params."""
        params = [SVIParams(a=0.04, b=0.1, rho=-0.25, m=0.0, sigma=0.1, expiry_years=0.25)]

        with pytest.raises(ValueError, match="at least 2 expiries"):
            interpolate_svi_surface(params, 0.15, np.array([0.0]))


class TestWeightedFitting:
    """Tests for weighted SVI fitting."""

    def test_weighted_fit_respects_weights(self):
        """Test that weights affect the fit."""
        fitter = SVIFitter()

        # Create data with outlier
        k = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
        w = np.array([0.06, 0.05, 0.04, 0.05, 0.10])  # Last point is outlier

        # Fit without weights
        params_unweighted = fitter.fit(k, w, 0.25)

        # Fit with low weight on outlier
        weights = np.array([1.0, 1.0, 1.0, 1.0, 0.1])
        params_weighted = fitter.fit(k, w, 0.25, weights=weights)

        # Weighted fit should have lower error on non-outlier points
        w_pred_weighted = SVIFitter.svi_total_variance(
            k[:4], params_weighted.a, params_weighted.b,
            params_weighted.rho, params_weighted.m, params_weighted.sigma
        )
        w_pred_unweighted = SVIFitter.svi_total_variance(
            k[:4], params_unweighted.a, params_unweighted.b,
            params_unweighted.rho, params_unweighted.m, params_unweighted.sigma
        )

        error_weighted = np.mean((w[:4] - w_pred_weighted)**2)
        error_unweighted = np.mean((w[:4] - w_pred_unweighted)**2)

        assert error_weighted <= error_unweighted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
