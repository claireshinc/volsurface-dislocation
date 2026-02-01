"""FastAPI application entry point.

Provides REST API for volatility surface analysis:
- Surface fitting and retrieval
- Feature extraction
- Anomaly detection
- Trade idea generation
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import surface_router, anomalies_router
from config.settings import get_settings
from data.database import init_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Volatility Archaeologist API")
    settings = get_settings()
    init_database(settings.database_url)
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("Shutting down API")


# Create FastAPI application
app = FastAPI(
    title="Volatility Archaeologist",
    description=(
        "Quantitative finance tool for volatility surface analysis. "
        "Fits SVI models, extracts features, detects anomalies, and generates trade ideas."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "validation_error"},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"},
    )


# Include routers
app.include_router(surface_router, prefix="/api/v1/surface", tags=["Surface"])
app.include_router(anomalies_router, prefix="/api/v1/anomalies", tags=["Anomalies"])


# Root endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Volatility Archaeologist",
        "version": "1.0.0",
        "description": "Volatility surface analysis and anomaly detection API",
        "docs": "/docs",
        "endpoints": {
            "surface": "/api/v1/surface",
            "anomalies": "/api/v1/anomalies",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


@app.get("/api/v1/features/{ticker}")
async def get_features(ticker: str):
    """Get extracted features for a ticker.

    This is a convenience endpoint that combines surface fitting
    and feature extraction.
    """
    from date import date

    from analytics.svi_fitter import SVIFitter
    from analytics.feature_extractor import FeatureExtractor
    from data.collector import OptionsDataCollector, create_synthetic_chain

    try:
        collector = OptionsDataCollector()
        chain = collector.get_chain(ticker.upper())
    except Exception as e:
        logger.warning(f"Could not fetch real data for {ticker}: {e}")
        # Use synthetic data for demo
        chain = create_synthetic_chain(ticker.upper())

    # Fit SVI to available expiries
    fitter = SVIFitter()
    params_list = []

    for expiry in chain.expiries[:5]:  # Limit to first 5 expiries
        try:
            k, w, T = collector.prepare_for_svi_fitting(chain, expiry)
            params = fitter.fit(k, w, T)
            params_list.append(params)
        except Exception as e:
            logger.warning(f"Could not fit expiry {expiry}: {e}")

    if not params_list:
        return {"error": "Could not fit any expiries"}

    # Extract features
    extractor = FeatureExtractor(chain.spot_price)
    features = extractor.extract(params_list, ticker.upper())

    return features.to_dict()


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
