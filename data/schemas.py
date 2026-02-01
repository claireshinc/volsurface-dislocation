"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class OptionsChainSchema(BaseModel):
    """Schema for options chain data."""

    ticker: str = Field(..., description="Underlying symbol")
    timestamp: datetime = Field(..., description="Data collection timestamp")
    spot_price: float = Field(..., gt=0, description="Current spot price")
    expiries: list[str] = Field(..., description="Available expiration dates")
    chain_data: list[dict[str, Any]] = Field(..., description="Full chain data")

    model_config = {"from_attributes": True}


class SVIParamsSchema(BaseModel):
    """Schema for SVI model parameters."""

    a: float = Field(..., description="Level of variance")
    b: float = Field(..., ge=0, description="Slope magnitude")
    rho: float = Field(..., ge=-1, le=1, description="Rotation parameter")
    m: float = Field(..., description="Translation")
    sigma: float = Field(..., gt=0, description="Smoothness")
    expiry_years: float = Field(..., gt=0, description="Time to expiry in years")
    fit_error: float | None = Field(None, description="RMSE of the fit")


class FittedSurfaceSchema(BaseModel):
    """Schema for fitted volatility surface."""

    ticker: str = Field(..., description="Underlying symbol")
    timestamp: datetime = Field(..., description="Fit timestamp")
    spot_price: float = Field(..., gt=0, description="Spot price at time of fit")
    params_by_expiry: dict[int, SVIParamsSchema] = Field(
        ..., description="SVI params by expiry days"
    )

    model_config = {"from_attributes": True}


class SurfaceFeaturesSchema(BaseModel):
    """Schema for extracted surface features."""

    timestamp: datetime = Field(..., description="Extraction timestamp")
    ticker: str = Field(..., description="Underlying symbol")
    spot: float = Field(..., gt=0, description="Spot price")

    # ATM volatilities
    atm_vol_30d: float | None = Field(None, ge=0, description="30d ATM vol")
    atm_vol_60d: float | None = Field(None, ge=0, description="60d ATM vol")
    atm_vol_90d: float | None = Field(None, ge=0, description="90d ATM vol")

    # Skew
    skew_25d_30d: float | None = Field(None, description="30d 25-delta skew")
    skew_25d_60d: float | None = Field(None, description="60d 25-delta skew")
    skew_25d_90d: float | None = Field(None, description="90d 25-delta skew")

    # Butterfly
    butterfly_30d: float | None = Field(None, description="30d butterfly")
    butterfly_60d: float | None = Field(None, description="60d butterfly")
    butterfly_90d: float | None = Field(None, description="90d butterfly")

    # Term structure
    term_slope: float | None = Field(None, description="Term structure slope")
    forward_vol_30_60: float | None = Field(None, description="30-60d forward vol")
    forward_vol_60_90: float | None = Field(None, description="60-90d forward vol")

    # SVI params
    svi_params: dict[int, dict[str, float]] = Field(
        default_factory=dict, description="SVI params by tenor"
    )

    model_config = {"from_attributes": True}


class AnomalyResultSchema(BaseModel):
    """Schema for a single anomaly detection result."""

    feature_name: str = Field(..., description="Name of the analyzed feature")
    current_value: float = Field(..., description="Current feature value")
    historical_mean: float = Field(..., description="Historical mean")
    historical_std: float = Field(..., description="Historical standard deviation")
    z_score: float = Field(..., description="Z-score from mean")
    percentile: float = Field(..., ge=0, le=100, description="Percentile rank")
    is_anomaly: bool = Field(..., description="Whether flagged as anomaly")
    direction: str = Field(..., description="high, low, or normal")


class AnomalyReportSchema(BaseModel):
    """Schema for complete anomaly report."""

    timestamp: datetime = Field(..., description="Analysis timestamp")
    ticker: str = Field(..., description="Underlying symbol")
    anomalies: list[AnomalyResultSchema] = Field(..., description="Anomaly results")
    summary: dict[str, Any] = Field(..., description="Summary statistics")


class AnomalyLogSchema(BaseModel):
    """Schema for anomaly log record."""

    id: int = Field(..., description="Record ID")
    timestamp: datetime = Field(..., description="Detection timestamp")
    ticker: str = Field(..., description="Underlying symbol")
    feature_name: str = Field(..., description="Feature name")
    current_value: float = Field(..., description="Feature value")
    z_score: float = Field(..., description="Z-score")
    percentile: float = Field(..., description="Percentile")
    is_anomaly: bool = Field(..., description="Anomaly flag")
    direction: str = Field(..., description="Direction")
    trade_suggestion: str | None = Field(None, description="Trade suggestion")

    model_config = {"from_attributes": True}


class TradeIdeaSchema(BaseModel):
    """Schema for trade idea."""

    ticker: str = Field(..., description="Underlying symbol")
    timestamp: datetime = Field(..., description="Generation timestamp")
    direction: str = Field(..., description="Trade direction")
    strategy: str = Field(..., description="Suggested strategy")
    description: str = Field(..., description="Brief description")
    rationale: str = Field(..., description="Detailed rationale")
    confidence: str = Field(..., description="Confidence level")
    z_score: float = Field(..., description="Triggering z-score")
    percentile: float = Field(..., description="Percentile rank")
    entry_criteria: list[str] = Field(..., description="Entry conditions")
    exit_criteria: list[str] = Field(..., description="Exit conditions")
    risks: list[str] = Field(..., description="Key risks")
    related_anomalies: list[str] = Field(..., description="Related anomaly features")


class GreeksSchema(BaseModel):
    """Schema for option Greeks."""

    delta: float = Field(..., description="Delta")
    gamma: float = Field(..., description="Gamma")
    vega: float = Field(..., description="Vega (per 1% vol)")
    theta: float = Field(..., description="Theta (per day)")
    rho: float = Field(..., description="Rho (per 1% rate)")


# API Request schemas
class SurfaceRequest(BaseModel):
    """Request to fetch/fit a volatility surface."""

    ticker: str = Field(..., description="Underlying symbol")
    rate: float = Field(default=0.05, description="Risk-free rate")
    dividend_yield: float = Field(default=0.0, description="Dividend yield")


class AnomalyRequest(BaseModel):
    """Request for anomaly detection."""

    ticker: str = Field(..., description="Underlying symbol")
    zscore_threshold: float = Field(default=2.0, description="Z-score threshold")


class PriceRequest(BaseModel):
    """Request for option pricing."""

    spot: float = Field(..., gt=0, description="Spot price")
    strike: float = Field(..., gt=0, description="Strike price")
    expiry_years: float = Field(..., gt=0, description="Time to expiry in years")
    vol: float = Field(..., gt=0, description="Implied volatility")
    rate: float = Field(default=0.0, description="Risk-free rate")
    dividend_yield: float = Field(default=0.0, description="Dividend yield")
    option_type: str = Field(default="call", description="call or put")


# API Response schemas
class SurfaceResponse(BaseModel):
    """Response containing fitted surface data."""

    ticker: str
    timestamp: datetime
    spot_price: float
    features: SurfaceFeaturesSchema
    params_by_expiry: dict[int, SVIParamsSchema]
    smile_data: dict[int, dict[str, float]]  # IV at standard deltas by tenor


class AnomalyResponse(BaseModel):
    """Response containing anomaly analysis."""

    report: AnomalyReportSchema
    trade_ideas: list[TradeIdeaSchema]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
