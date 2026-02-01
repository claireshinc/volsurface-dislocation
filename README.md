# Volatility Archaeologist

A quantitative finance tool for volatility surface analysis, anomaly detection, and trade idea generation.

## Features

- **SVI Surface Fitting**: Fits the Stochastic Volatility Inspired (SVI) model to options market data
- **Feature Extraction**: Extracts key surface features (ATM vol, skew, butterfly, term structure)
- **Anomaly Detection**: Detects statistical anomalies by comparing current values to historical distributions
- **Trade Ideas**: Generates actionable trade suggestions based on detected mispricings
- **Interactive Dashboard**: Streamlit-based visualization with 3D surfaces, heatmaps, and more
- **REST API**: FastAPI backend for programmatic access

## Installation

```bash
# Clone repository
cd volsurface-dislocation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

## Quick Start

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

### Run the API

```bash
uvicorn api.main:app --reload
```

API documentation available at http://localhost:8000/docs

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
volsurface-dislocation/
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── config/
│   └── settings.py           # Pydantic settings management
├── data/
│   ├── collector.py          # Options data fetching (OpenBB)
│   ├── database.py           # SQLAlchemy models
│   └── schemas.py            # Pydantic API schemas
├── analytics/
│   ├── svi_fitter.py         # SVI surface fitting
│   ├── black_scholes.py      # BS pricing & Greeks
│   ├── feature_extractor.py  # Surface feature extraction
│   ├── anomaly_detector.py   # Statistical anomaly detection
│   └── trade_ideas.py        # Trade suggestion generation
├── api/
│   ├── main.py               # FastAPI application
│   └── routes/
│       ├── surface.py        # Surface endpoints
│       └── anomalies.py      # Anomaly endpoints
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── scripts/
│   └── init_db.py            # Database initialization
└── tests/
    ├── test_svi_fitter.py
    ├── test_black_scholes.py
    └── test_anomaly_detector.py
```

## SVI Model

The SVI (Stochastic Volatility Inspired) parameterization models total variance as:

```
w(k) = a + b * [ρ*(k-m) + sqrt((k-m)² + σ²)]
```

Where:
- `k = log(K/F)` is log-moneyness
- `a` = level of variance (vertical shift)
- `b` = slope magnitude (wing steepness)
- `ρ` = rotation parameter (-1 to 1, controls skew)
- `m` = translation (horizontal shift)
- `σ` = smoothness (ATM curvature)

## Features Extracted

| Feature | Description |
|---------|-------------|
| ATM Vol (30/60/90d) | At-the-money implied volatility at standard tenors |
| 25Δ Skew | σ(25Δ put) - σ(25Δ call), measures smile asymmetry |
| Butterfly | 0.5*(σ(25Δ put) + σ(25Δ call)) - σ(ATM), measures curvature |
| Term Slope | (σ_90d - σ_30d) / σ_30d, term structure steepness |
| Forward Vol | Implied forward volatility between tenors |

## Anomaly Detection

The detector compares current feature values against a 252-day rolling historical distribution:

- **Z-Score**: `z = (current - mean) / std`
- **Anomaly Flag**: `|z| > 2.0` (configurable)
- **Percentile Rank**: Position in historical distribution

## Trade Ideas

Generated based on detected anomalies:

| Anomaly | Suggested Trade |
|---------|-----------------|
| High ATM Vol | Sell straddles - vol is rich |
| Low ATM Vol | Buy straddles - vol is cheap |
| High Skew | Sell risk reversal - skew is steep |
| Low Skew | Buy risk reversal - skew is flat |
| High Butterfly | Sell wings - convexity is expensive |
| Low Butterfly | Buy wings - tails are cheap |
| Steep Term Structure | Sell calendar spread |
| Flat Term Structure | Buy calendar spread |

## API Endpoints

### Surface

- `GET /api/v1/surface/current/{ticker}` - Get current volatility surface
- `GET /api/v1/surface/smile/{ticker}` - Get volatility smile
- `GET /api/v1/surface/term-structure/{ticker}` - Get term structure
- `POST /api/v1/surface/price` - Calculate option price and Greeks

### Anomalies

- `GET /api/v1/anomalies/current/{ticker}` - Run anomaly detection
- `GET /api/v1/anomalies/history/{ticker}` - Get historical distribution
- `GET /api/v1/anomalies/trade-ideas/{ticker}` - Get trade ideas
- `POST /api/v1/anomalies/load-history/{ticker}` - Load synthetic history

## Configuration

Environment variables (`.env`):

```bash
DATABASE_URL=sqlite:///./volsurface.db
API_HOST=0.0.0.0
API_PORT=8000
ANOMALY_ZSCORE_THRESHOLD=2.0
HISTORY_LOOKBACK_DAYS=252
LOG_LEVEL=INFO
```

## Data Sources

Uses OpenBB for options data with free providers:
- **CBOE** (primary) - Chicago Board Options Exchange
- **yfinance** (fallback) - Yahoo Finance

## Dashboard Visualizations

1. **3D Surface**: Interactive volatility surface plot
2. **Heatmap**: Current vs historical percentiles (green=cheap, red=rich)
3. **Smiles**: Volatility smiles at multiple tenors
4. **Term Structure**: ATM vol by expiry
5. **Distributions**: Historical feature distributions
6. **Math Panel**: SVI parameters and formulas
7. **Anomaly Cards**: Visual alerts for detected anomalies
8. **Trade Ideas**: Detailed trade suggestions with entry/exit criteria

## License

MIT
