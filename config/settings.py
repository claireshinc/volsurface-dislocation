"""Application configuration using Pydantic settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = Field(
        default="sqlite:///./volsurface.db",
        description="Database connection URL",
    )

    # API Settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # OpenBB Settings
    openbb_token: str | None = Field(default=None, description="OpenBB API token")
    default_provider: Literal["cboe", "yfinance", "intrinio"] = Field(
        default="cboe",
        description="Default options data provider",
    )

    # Anomaly Detection
    anomaly_zscore_threshold: float = Field(
        default=2.0,
        description="Z-score threshold for anomaly detection",
    )
    history_lookback_days: int = Field(
        default=252,
        description="Number of trading days for historical lookback",
    )

    # Feature Extraction
    standard_tenors: list[int] = Field(
        default=[30, 60, 90],
        description="Standard tenor days for feature extraction",
    )

    # Data Collection
    default_tickers: list[str] = Field(
        default=["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"],
        description="Default tickers for daily data collection",
    )
    collection_time: str = Field(
        default="16:30",
        description="Daily collection time in ET (HH:MM format)",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings

    Example:
        >>> settings = get_settings()
        >>> settings.database_url
        'sqlite:///./volsurface.db'
    """
    return Settings()
