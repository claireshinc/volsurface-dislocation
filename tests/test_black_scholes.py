"""Tests for Black-Scholes pricing and Greeks."""

import numpy as np
import pytest

from analytics.black_scholes import BlackScholes, Greeks, OptionType, vectorized_price


class TestBlackScholesPrice:
    """Tests for BS pricing."""

    @pytest.fixture
    def bs(self):
        """Standard BS calculator."""
        return BlackScholes(spot=100, rate=0.05)

    def test_call_price_atm(self, bs):
        """Test ATM call price is reasonable."""
        price = bs.price(strike=100, expiry=0.25, vol=0.2, option_type="call")
        # ATM call with 25% vol, 3 month expiry should be roughly $4-5
        assert 3 < price < 7

    def test_put_price_atm(self, bs):
        """Test ATM put price is reasonable."""
        price = bs.price(strike=100, expiry=0.25, vol=0.2, option_type="put")
        # ATM put should be slightly less than call due to positive rates
        call_price = bs.price(strike=100, expiry=0.25, vol=0.2, option_type="call")
        assert price < call_price

    def test_put_call_parity(self, bs):
        """Test put-call parity: C - P = S - K*exp(-r*T)."""
        K, T, vol = 100, 0.25, 0.2
        call = bs.price(K, T, vol, "call")
        put = bs.price(K, T, vol, "put")

        expected_diff = bs.spot - K * np.exp(-bs.rate * T)
        assert np.isclose(call - put, expected_diff, rtol=1e-6)

    def test_call_approaches_intrinsic_itm(self, bs):
        """Deep ITM call should approach intrinsic value."""
        # Deep ITM call (S=100, K=50)
        price = bs.price(strike=50, expiry=0.01, vol=0.2, option_type="call")
        intrinsic = max(100 - 50, 0)
        assert np.isclose(price, intrinsic, atol=1)

    def test_put_approaches_zero_far_otm(self, bs):
        """Far OTM put should be nearly worthless."""
        price = bs.price(strike=50, expiry=0.1, vol=0.2, option_type="put")
        assert price < 0.01


class TestBlackScholesGreeks:
    """Tests for BS Greeks."""

    @pytest.fixture
    def bs(self):
        return BlackScholes(spot=100, rate=0.05)

    def test_call_delta_range(self, bs):
        """Call delta should be between 0 and 1."""
        for strike in [80, 100, 120]:
            delta = bs.delta(strike, 0.25, 0.2, "call")
            assert 0 <= delta <= 1

    def test_put_delta_range(self, bs):
        """Put delta should be between -1 and 0."""
        for strike in [80, 100, 120]:
            delta = bs.delta(strike, 0.25, 0.2, "put")
            assert -1 <= delta <= 0

    def test_atm_delta_approximately_half(self, bs):
        """ATM call delta should be approximately 0.5."""
        delta = bs.delta(strike=100, expiry=0.25, vol=0.2, option_type="call")
        # Slightly above 0.5 due to drift
        assert 0.45 < delta < 0.65

    def test_gamma_positive(self, bs):
        """Gamma should always be positive."""
        for strike in [80, 100, 120]:
            gamma = bs.gamma(strike, 0.25, 0.2)
            assert gamma > 0

    def test_gamma_maximum_at_atm(self, bs):
        """Gamma should be maximized at ATM."""
        gamma_itm = bs.gamma(80, 0.25, 0.2)
        gamma_atm = bs.gamma(100, 0.25, 0.2)
        gamma_otm = bs.gamma(120, 0.25, 0.2)

        assert gamma_atm > gamma_itm
        assert gamma_atm > gamma_otm

    def test_vega_positive(self, bs):
        """Vega should always be positive."""
        for strike in [80, 100, 120]:
            vega = bs.vega(strike, 0.25, 0.2)
            assert vega > 0

    def test_theta_negative_for_long_options(self, bs):
        """Theta should generally be negative for long options."""
        # ATM options have significant time decay
        theta_call = bs.theta(100, 0.25, 0.2, "call")
        theta_put = bs.theta(100, 0.25, 0.2, "put")

        # Note: deep ITM puts can have positive theta due to interest
        assert theta_call < 0

    def test_greeks_object(self, bs):
        """Test Greeks object creation and serialization."""
        greeks = bs.greeks(100, 0.25, 0.2, "call")

        assert isinstance(greeks, Greeks)
        assert greeks.delta > 0
        assert greeks.gamma > 0
        assert greeks.vega > 0

        d = greeks.to_dict()
        assert "delta" in d
        assert "gamma" in d


class TestImpliedVolatility:
    """Tests for IV calculation."""

    @pytest.fixture
    def bs(self):
        return BlackScholes(spot=100, rate=0.05)

    def test_iv_recovery(self, bs):
        """Test that IV calculation recovers input vol."""
        original_vol = 0.25
        price = bs.price(100, 0.25, original_vol, "call")

        recovered_vol = bs.implied_volatility(price, 100, 0.25, "call")
        assert np.isclose(recovered_vol, original_vol, rtol=1e-4)

    def test_iv_put(self, bs):
        """Test IV calculation for puts."""
        original_vol = 0.3
        price = bs.price(100, 0.25, original_vol, "put")

        recovered_vol = bs.implied_volatility(price, 100, 0.25, "put")
        assert np.isclose(recovered_vol, original_vol, rtol=1e-4)

    def test_iv_no_solution(self, bs):
        """Test error when no IV solution exists."""
        # Price too high for any reasonable vol
        with pytest.raises(ValueError, match="Could not find IV"):
            bs.implied_volatility(200, 100, 0.25, "call")


class TestStrikeFromDelta:
    """Tests for strike calculation from delta."""

    @pytest.fixture
    def bs(self):
        return BlackScholes(spot=100, rate=0.05)

    def test_25_delta_put(self, bs):
        """Test 25-delta put strike calculation."""
        strike = bs.strike_from_delta(0.25, 0.25, 0.2, "put")

        # 25-delta put should have strike below spot
        assert strike < 100

        # Verify delta
        delta = bs.delta(strike, 0.25, 0.2, "put")
        assert np.isclose(abs(delta), 0.25, atol=0.01)

    def test_25_delta_call(self, bs):
        """Test 25-delta call strike calculation."""
        strike = bs.strike_from_delta(0.25, 0.25, 0.2, "call")

        # 25-delta call should have strike above spot
        assert strike > 100

        # Verify delta
        delta = bs.delta(strike, 0.25, 0.2, "call")
        assert np.isclose(delta, 0.25, atol=0.01)

    def test_50_delta_near_atm(self, bs):
        """Test 50-delta should be near ATM."""
        strike = bs.strike_from_delta(0.50, 0.25, 0.2, "call")

        # Should be close to spot
        assert np.isclose(strike, 100, atol=5)


class TestVectorizedPrice:
    """Tests for vectorized pricing."""

    def test_vectorized_matches_scalar(self):
        """Test vectorized pricing matches scalar results."""
        spots = np.array([100, 100, 100])
        strikes = np.array([95, 100, 105])
        expiries = np.array([0.25, 0.25, 0.25])
        vols = np.array([0.2, 0.2, 0.2])
        rates = np.array([0.05, 0.05, 0.05])
        is_call = np.array([True, True, True])

        vec_prices = vectorized_price(spots, strikes, expiries, vols, rates, is_call)

        bs = BlackScholes(100, 0.05)
        for i, strike in enumerate(strikes):
            scalar_price = bs.price(strike, 0.25, 0.2, "call")
            assert np.isclose(vec_prices[i], scalar_price, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
