"""SQLAlchemy database models and session management.

Stores:
- Raw options chain data
- Fitted SVI surface parameters
- Extracted surface features
- Anomaly detection logs
"""

import json
import logging
from datetime import datetime
from typing import Generator

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Text,
    Index,
    Boolean,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from config.settings import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class OptionsChainRecord(Base):
    """Raw options chain data storage."""

    __tablename__ = "options_chains"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    ticker = Column(String(20), nullable=False, index=True)
    spot_price = Column(Float, nullable=False)
    chain_data = Column(Text, nullable=False)  # JSON serialized

    __table_args__ = (
        Index("ix_chain_ticker_timestamp", "ticker", "timestamp"),
    )

    def get_chain_data(self) -> dict:
        """Deserialize chain data."""
        return json.loads(self.chain_data)

    def set_chain_data(self, data: dict) -> None:
        """Serialize chain data."""
        self.chain_data = json.dumps(data)


class FittedSurfaceRecord(Base):
    """Fitted SVI surface parameters storage."""

    __tablename__ = "fitted_surfaces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    ticker = Column(String(20), nullable=False, index=True)
    expiry_days = Column(Integer, nullable=False)  # Days to expiry
    expiry_years = Column(Float, nullable=False)

    # SVI parameters
    svi_a = Column(Float, nullable=False)
    svi_b = Column(Float, nullable=False)
    svi_rho = Column(Float, nullable=False)
    svi_m = Column(Float, nullable=False)
    svi_sigma = Column(Float, nullable=False)
    fit_error = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_surface_ticker_timestamp", "ticker", "timestamp"),
    )

    def to_svi_params(self):
        """Convert to SVIParams object."""
        from analytics.svi_fitter import SVIParams

        return SVIParams(
            a=self.svi_a,
            b=self.svi_b,
            rho=self.svi_rho,
            m=self.svi_m,
            sigma=self.svi_sigma,
            expiry_years=self.expiry_years,
            fit_error=self.fit_error,
        )


class SurfaceFeaturesRecord(Base):
    """Extracted surface features storage."""

    __tablename__ = "surface_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    ticker = Column(String(20), nullable=False, index=True)
    spot_price = Column(Float, nullable=False)

    # ATM volatilities
    atm_vol_30d = Column(Float, nullable=True)
    atm_vol_60d = Column(Float, nullable=True)
    atm_vol_90d = Column(Float, nullable=True)

    # Skew
    skew_25d_30d = Column(Float, nullable=True)
    skew_25d_60d = Column(Float, nullable=True)
    skew_25d_90d = Column(Float, nullable=True)

    # Butterfly
    butterfly_30d = Column(Float, nullable=True)
    butterfly_60d = Column(Float, nullable=True)
    butterfly_90d = Column(Float, nullable=True)

    # Term structure
    term_slope = Column(Float, nullable=True)
    forward_vol_30_60 = Column(Float, nullable=True)
    forward_vol_60_90 = Column(Float, nullable=True)

    # Raw SVI params as JSON
    svi_params_json = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_features_ticker_timestamp", "ticker", "timestamp"),
    )

    def to_surface_features(self):
        """Convert to SurfaceFeatures object."""
        from analytics.feature_extractor import SurfaceFeatures

        svi_params = json.loads(self.svi_params_json) if self.svi_params_json else {}

        return SurfaceFeatures(
            timestamp=self.timestamp,
            ticker=self.ticker,
            spot=self.spot_price,
            atm_vol_30d=self.atm_vol_30d,
            atm_vol_60d=self.atm_vol_60d,
            atm_vol_90d=self.atm_vol_90d,
            skew_25d_30d=self.skew_25d_30d,
            skew_25d_60d=self.skew_25d_60d,
            skew_25d_90d=self.skew_25d_90d,
            butterfly_30d=self.butterfly_30d,
            butterfly_60d=self.butterfly_60d,
            butterfly_90d=self.butterfly_90d,
            term_slope=self.term_slope,
            forward_vol_30_60=self.forward_vol_30_60,
            forward_vol_60_90=self.forward_vol_60_90,
            svi_params=svi_params,
        )

    @classmethod
    def from_surface_features(cls, features) -> "SurfaceFeaturesRecord":
        """Create from SurfaceFeatures object."""
        return cls(
            timestamp=features.timestamp,
            ticker=features.ticker,
            spot_price=features.spot,
            atm_vol_30d=features.atm_vol_30d,
            atm_vol_60d=features.atm_vol_60d,
            atm_vol_90d=features.atm_vol_90d,
            skew_25d_30d=features.skew_25d_30d,
            skew_25d_60d=features.skew_25d_60d,
            skew_25d_90d=features.skew_25d_90d,
            butterfly_30d=features.butterfly_30d,
            butterfly_60d=features.butterfly_60d,
            butterfly_90d=features.butterfly_90d,
            term_slope=features.term_slope,
            forward_vol_30_60=features.forward_vol_30_60,
            forward_vol_60_90=features.forward_vol_60_90,
            svi_params_json=json.dumps(features.svi_params),
        )


class AnomalyLogRecord(Base):
    """Anomaly detection log storage."""

    __tablename__ = "anomaly_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    ticker = Column(String(20), nullable=False, index=True)
    feature_name = Column(String(50), nullable=False)
    current_value = Column(Float, nullable=False)
    historical_mean = Column(Float, nullable=True)
    historical_std = Column(Float, nullable=True)
    z_score = Column(Float, nullable=False)
    percentile = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, nullable=False)
    direction = Column(String(20), nullable=False)
    trade_suggestion = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_anomaly_ticker_timestamp", "ticker", "timestamp"),
        Index("ix_anomaly_ticker_feature", "ticker", "feature_name"),
    )


# Engine and session management
_engine = None
_SessionLocal = None


def get_engine(database_url: str | None = None):
    """Get or create database engine.

    Args:
        database_url: Database connection URL. Uses settings if not provided.

    Returns:
        SQLAlchemy engine
    """
    global _engine

    if _engine is None:
        if database_url is None:
            settings = get_settings()
            database_url = settings.database_url

        _engine = create_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
        )
        logger.info(f"Created database engine: {database_url}")

    return _engine


def get_session() -> Generator[Session, None, None]:
    """Get database session.

    Yields:
        SQLAlchemy session

    Example:
        >>> with get_session() as session:
        ...     records = session.query(SurfaceFeaturesRecord).all()
    """
    global _SessionLocal

    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(bind=engine)

    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database(database_url: str | None = None) -> None:
    """Initialize database tables.

    Args:
        database_url: Database connection URL
    """
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    logger.info("Database tables created")


# Repository functions for common operations
class SurfaceRepository:
    """Repository for surface-related database operations."""

    def __init__(self, session: Session):
        self.session = session

    def save_features(self, features) -> int:
        """Save surface features to database."""
        record = SurfaceFeaturesRecord.from_surface_features(features)
        self.session.add(record)
        self.session.flush()
        return record.id

    def get_features_history(
        self,
        ticker: str,
        days: int = 252,
    ) -> list[SurfaceFeaturesRecord]:
        """Get historical features for a ticker."""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)

        return (
            self.session.query(SurfaceFeaturesRecord)
            .filter(
                SurfaceFeaturesRecord.ticker == ticker,
                SurfaceFeaturesRecord.timestamp >= cutoff,
            )
            .order_by(SurfaceFeaturesRecord.timestamp.desc())
            .all()
        )

    def get_latest_features(self, ticker: str) -> SurfaceFeaturesRecord | None:
        """Get most recent features for a ticker."""
        return (
            self.session.query(SurfaceFeaturesRecord)
            .filter(SurfaceFeaturesRecord.ticker == ticker)
            .order_by(SurfaceFeaturesRecord.timestamp.desc())
            .first()
        )

    def save_anomaly(
        self,
        ticker: str,
        anomaly_result,
        trade_suggestion: str | None = None,
    ) -> int:
        """Save anomaly detection result."""
        record = AnomalyLogRecord(
            timestamp=datetime.now(),
            ticker=ticker,
            feature_name=anomaly_result.feature_name,
            current_value=anomaly_result.current_value,
            historical_mean=anomaly_result.historical_mean,
            historical_std=anomaly_result.historical_std,
            z_score=anomaly_result.z_score,
            percentile=anomaly_result.percentile,
            is_anomaly=anomaly_result.is_anomaly,
            direction=anomaly_result.direction,
            trade_suggestion=trade_suggestion,
        )
        self.session.add(record)
        self.session.flush()
        return record.id

    def get_recent_anomalies(
        self,
        ticker: str,
        hours: int = 24,
    ) -> list[AnomalyLogRecord]:
        """Get recent anomalies for a ticker."""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=hours)

        return (
            self.session.query(AnomalyLogRecord)
            .filter(
                AnomalyLogRecord.ticker == ticker,
                AnomalyLogRecord.timestamp >= cutoff,
                AnomalyLogRecord.is_anomaly == True,
            )
            .order_by(AnomalyLogRecord.timestamp.desc())
            .all()
        )
