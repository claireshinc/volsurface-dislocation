"""Streamlit dashboard for Volatility Archaeologist.

Interactive visualizations:
1. 3D volatility surface
2. Heatmap (IV vs historical percentile)
3. Volatility smiles by tenor
4. ATM vol time series with bands
5. Feature distributions
6. Math panel with SVI params and Greeks
7. Anomaly alerts
8. Trade ideas
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, date

# Page configuration
st.set_page_config(
    page_title="Volatility Archaeologist",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
    }
    .anomaly-high {
        background-color: #ff4444;
        padding: 10px;
        border-radius: 5px;
        color: white;
    }
    .anomaly-low {
        background-color: #4444ff;
        padding: 10px;
        border-radius: 5px;
        color: white;
    }
    .trade-idea {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #00ff00;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def get_surface_data(ticker: str, use_synthetic: bool = False):
    """Fetch and cache surface data."""
    from analytics.svi_fitter import SVIFitter
    from analytics.feature_extractor import FeatureExtractor
    from analytics.anomaly_detector import AnomalyDetector
    from analytics.trade_ideas import TradeIdeaGenerator
    from data.collector import OptionsDataCollector, create_synthetic_chain

    if use_synthetic:
        chain = create_synthetic_chain(ticker)
    else:
        try:
            collector = OptionsDataCollector()
            chain = collector.get_chain(ticker)
        except Exception as e:
            st.warning(f"Could not fetch live data: {e}. Using synthetic data.")
            chain = create_synthetic_chain(ticker)

    # Fit SVI
    fitter = SVIFitter()
    collector = OptionsDataCollector()
    params_list = []

    for expiry in chain.expiries[:6]:
        try:
            k, w, T = collector.prepare_for_svi_fitting(chain, expiry)
            params = fitter.fit(k, w, T)
            params_list.append(params)
        except Exception:
            continue

    # Extract features
    extractor = FeatureExtractor(chain.spot_price)
    features = extractor.extract(params_list, ticker)

    # Anomaly detection with synthetic history
    detector = AnomalyDetector()

    # Load synthetic history for demo
    np.random.seed(42)
    for feature_name in detector.FEATURE_NAMES:
        base = getattr(features, feature_name, None)
        if base is None:
            continue
        timestamps = [datetime.now() for _ in range(252)]
        values = [base + np.random.normal(0, abs(base) * 0.15) for _ in range(252)]
        detector.load_history(ticker, feature_name, timestamps, values)

    report = detector.analyze(features)

    # Trade ideas
    generator = TradeIdeaGenerator()
    ideas = generator.generate(report)

    return {
        "chain": chain,
        "params_list": params_list,
        "features": features,
        "report": report,
        "ideas": ideas,
        "detector": detector,
    }


def plot_3d_surface(params_list, spot: float):
    """Create 3D volatility surface plot."""
    from analytics.svi_fitter import SVIFitter

    # Create grid
    k_range = np.linspace(-0.3, 0.3, 50)
    expiries = sorted([p.expiry_years for p in params_list])

    Z = np.zeros((len(expiries), len(k_range)))

    for i, params in enumerate(sorted(params_list, key=lambda p: p.expiry_years)):
        ivs = SVIFitter.svi_implied_vol(
            k_range, params.a, params.b, params.rho, params.m, params.sigma, params.expiry_years
        )
        Z[i, :] = ivs * 100  # Convert to percentage

    # Convert k to strikes for labeling
    forward = spot  # Simplified
    strikes = forward * np.exp(k_range)
    expiry_days = [int(e * 365) for e in expiries]

    fig = go.Figure(data=[go.Surface(
        x=strikes,
        y=expiry_days,
        z=Z,
        colorscale='Viridis',
        colorbar=dict(title='IV (%)'),
        hovertemplate=(
            'Strike: $%{x:.0f}<br>'
            'Expiry: %{y}d<br>'
            'IV: %{z:.1f}%<extra></extra>'
        ),
    )])

    fig.update_layout(
        title='Implied Volatility Surface',
        scene=dict(
            xaxis_title='Strike ($)',
            yaxis_title='Days to Expiry',
            zaxis_title='Implied Vol (%)',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def plot_smile_comparison(params_list, spot: float):
    """Plot volatility smiles for multiple tenors."""
    from analytics.svi_fitter import SVIFitter

    fig = go.Figure()

    colors = px.colors.qualitative.Plotly
    k_range = np.linspace(-0.2, 0.2, 41)
    forward = spot

    for i, params in enumerate(sorted(params_list, key=lambda p: p.expiry_years)[:4]):
        ivs = SVIFitter.svi_implied_vol(
            k_range, params.a, params.b, params.rho, params.m, params.sigma, params.expiry_years
        )
        strikes = forward * np.exp(k_range)
        days = int(params.expiry_years * 365)

        fig.add_trace(go.Scatter(
            x=strikes,
            y=ivs * 100,
            mode='lines',
            name=f'{days}d',
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.add_vline(x=spot, line_dash="dash", line_color="gray",
                  annotation_text="Spot")

    fig.update_layout(
        title='Volatility Smiles by Tenor',
        xaxis_title='Strike ($)',
        yaxis_title='Implied Volatility (%)',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified',
    )

    return fig


def plot_heatmap(features, detector):
    """Plot heatmap of current vs historical percentiles."""
    feature_names = [
        'atm_vol_30d', 'atm_vol_60d', 'atm_vol_90d',
        'skew_25d_30d', 'skew_25d_60d', 'skew_25d_90d',
        'butterfly_30d', 'butterfly_60d', 'butterfly_90d',
    ]

    labels = [
        'ATM 30d', 'ATM 60d', 'ATM 90d',
        'Skew 30d', 'Skew 60d', 'Skew 90d',
        'Bfly 30d', 'Bfly 60d', 'Bfly 90d',
    ]

    percentiles = []
    for name in feature_names:
        current = getattr(features, name, None)
        if current is None:
            percentiles.append(50)
            continue

        historical = detector.get_historical_values(features.ticker, name)
        if len(historical) == 0:
            percentiles.append(50)
        else:
            pct = np.sum(historical < current) / len(historical) * 100
            percentiles.append(pct)

    # Reshape for heatmap (3x3)
    z = np.array(percentiles).reshape(3, 3)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=['30d', '60d', '90d'],
        y=['ATM Vol', 'Skew', 'Butterfly'],
        colorscale=[
            [0, 'green'],      # Cheap (low percentile)
            [0.5, 'yellow'],   # Fair
            [1, 'red'],        # Rich (high percentile)
        ],
        zmin=0,
        zmax=100,
        text=np.round(z, 0).astype(int),
        texttemplate='%{text}%',
        textfont=dict(size=14, color='black'),
        colorbar=dict(title='Percentile'),
        hovertemplate='%{y} %{x}: %{z:.0f}th percentile<extra></extra>',
    ))

    fig.update_layout(
        title='Current vs Historical (Green=Cheap, Red=Rich)',
        height=300,
    )

    return fig


def plot_distribution(detector, ticker: str, feature: str, current_value: float):
    """Plot historical distribution with current value marked."""
    historical = detector.get_historical_values(ticker, feature)

    if len(historical) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No historical data", showarrow=False)
        return fig

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=historical,
        nbinsx=30,
        name='Historical',
        marker_color='lightblue',
        opacity=0.7,
    ))

    # Current value line
    fig.add_vline(
        x=current_value,
        line_color='red',
        line_width=3,
        annotation_text=f'Current: {current_value:.4f}',
        annotation_position='top',
    )

    # Z-score bands
    bands = detector.get_zscore_bands(ticker, feature)
    if bands:
        fig.add_vline(x=bands['mean'], line_dash='dash', line_color='green',
                      annotation_text='Mean')
        fig.add_vrect(
            x0=bands['minus_2_sigma'], x1=bands['plus_2_sigma'],
            fillcolor='green', opacity=0.1, line_width=0,
        )

    fig.update_layout(
        title=f'{feature} Distribution',
        xaxis_title=feature,
        yaxis_title='Count',
        height=300,
        showlegend=False,
    )

    return fig


def plot_term_structure(features):
    """Plot ATM vol term structure."""
    tenors = [30, 60, 90]
    vols = [
        features.atm_vol_30d,
        features.atm_vol_60d,
        features.atm_vol_90d,
    ]

    # Filter None values
    valid = [(t, v) for t, v in zip(tenors, vols) if v is not None]
    if not valid:
        fig = go.Figure()
        fig.add_annotation(text="No term structure data", showarrow=False)
        return fig

    tenors, vols = zip(*valid)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=tenors,
        y=[v * 100 for v in vols],
        mode='lines+markers',
        name='ATM Vol',
        line=dict(color='blue', width=2),
        marker=dict(size=10),
    ))

    fig.update_layout(
        title='ATM Volatility Term Structure',
        xaxis_title='Days to Expiry',
        yaxis_title='ATM Vol (%)',
        height=300,
    )

    return fig


def render_math_panel(params_list, features):
    """Render math formulas and parameters."""
    st.markdown("### 📐 Mathematical Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**SVI Parameterization**")
        st.latex(r"w(k) = a + b \left[ \rho(k-m) + \sqrt{(k-m)^2 + \sigma^2} \right]")

        st.markdown("Where:")
        st.markdown(r"""
        - $k = \ln(K/F)$ is log-moneyness
        - $a$ = level of variance
        - $b$ = slope magnitude
        - $\rho$ = rotation (skew direction)
        - $m$ = translation (horizontal shift)
        - $\sigma$ = smoothness (ATM curvature)
        """)

    with col2:
        st.markdown("**Black-Scholes Greeks**")
        st.latex(r"d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}")
        st.latex(r"\Delta = N(d_1)")
        st.latex(r"\Gamma = \frac{N'(d_1)}{S\sigma\sqrt{T}}")
        st.latex(r"\mathcal{V} = S \cdot N'(d_1) \cdot \sqrt{T}")

    # SVI Parameters table
    st.markdown("**Fitted SVI Parameters**")

    params_data = []
    for params in sorted(params_list, key=lambda p: p.expiry_years):
        params_data.append({
            'Expiry (days)': int(params.expiry_years * 365),
            'a': f"{params.a:.4f}",
            'b': f"{params.b:.4f}",
            'ρ': f"{params.rho:.4f}",
            'm': f"{params.m:.4f}",
            'σ': f"{params.sigma:.4f}",
            'RMSE': f"{params.fit_error:.6f}" if params.fit_error else "N/A",
        })

    st.dataframe(pd.DataFrame(params_data), hide_index=True)


def render_anomaly_cards(report):
    """Render anomaly alert cards."""
    st.markdown("### 🚨 Anomaly Alerts")

    anomalies = [a for a in report.anomalies if a.is_anomaly]

    if not anomalies:
        st.success("No significant anomalies detected")
        return

    cols = st.columns(min(len(anomalies), 3))

    for i, anomaly in enumerate(anomalies[:6]):
        with cols[i % 3]:
            color = "#ff4444" if anomaly.direction == "high" else "#4444ff"
            direction_emoji = "📈" if anomaly.direction == "high" else "📉"

            st.markdown(f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 10px; margin: 5px 0;">
                <h4 style="margin: 0; color: white;">{direction_emoji} {anomaly.feature_name}</h4>
                <p style="margin: 5px 0; color: white;">
                    <b>Z-Score:</b> {anomaly.z_score:+.2f}<br>
                    <b>Percentile:</b> {anomaly.percentile:.0f}%<br>
                    <b>Current:</b> {anomaly.current_value:.4f}<br>
                    <b>Mean:</b> {anomaly.historical_mean:.4f}
                </p>
            </div>
            """, unsafe_allow_html=True)


def render_trade_ideas(ideas):
    """Render trade idea cards."""
    st.markdown("### 💡 Trade Ideas")

    if not ideas:
        st.info("No trade ideas generated")
        return

    for idea in ideas[:5]:
        confidence_colors = {
            "high": "#00ff00",
            "medium": "#ffff00",
            "low": "#ff8800",
        }
        border_color = confidence_colors.get(idea.confidence.value, "#ffffff")

        with st.expander(f"**{idea.strategy}** ({idea.confidence.value.upper()})"):
            st.markdown(f"**Description:** {idea.description}")
            st.markdown(f"**Rationale:** {idea.rationale}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Entry Criteria:**")
                for criterion in idea.entry_criteria:
                    st.markdown(f"- {criterion}")

            with col2:
                st.markdown("**Exit Criteria:**")
                for criterion in idea.exit_criteria:
                    st.markdown(f"- {criterion}")

            st.markdown("**Risks:**")
            for risk in idea.risks:
                st.warning(risk)

            st.markdown(f"*Z-Score: {idea.z_score:.2f} | Percentile: {idea.percentile:.0f}%*")


def main():
    """Main dashboard application."""
    st.title("📊 Volatility Archaeologist")
    st.markdown("*Quantitative volatility surface analysis and anomaly detection*")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        ticker = st.text_input("Ticker", value="SPY").upper()

        use_synthetic = st.checkbox(
            "Use Synthetic Data",
            value=True,
            help="Use synthetic data for demo (uncheck to fetch live data)"
        )

        st.markdown("---")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()

        st.markdown("---")

        st.markdown("### About")
        st.markdown("""
        This tool analyzes options volatility surfaces using the SVI model
        and detects statistical anomalies in surface features.

        **Features:**
        - SVI surface fitting
        - Feature extraction
        - Anomaly detection
        - Trade idea generation
        """)

    # Load data
    with st.spinner("Loading volatility surface data..."):
        try:
            data = get_surface_data(ticker, use_synthetic)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    chain = data["chain"]
    params_list = data["params_list"]
    features = data["features"]
    report = data["report"]
    ideas = data["ideas"]
    detector = data["detector"]

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Spot Price", f"${chain.spot_price:.2f}")

    with col2:
        if features.atm_vol_30d:
            st.metric("30d ATM Vol", f"{features.atm_vol_30d:.1%}")

    with col3:
        if features.skew_25d_30d:
            st.metric("30d Skew", f"{features.skew_25d_30d:.2%}")

    with col4:
        st.metric("Anomalies", report.anomaly_count,
                  delta="Detected" if report.anomaly_count > 0 else None,
                  delta_color="inverse")

    st.markdown("---")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌊 Surface",
        "📊 Analysis",
        "🚨 Anomalies",
        "💡 Trade Ideas",
        "📐 Math",
    ])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.plotly_chart(plot_3d_surface(params_list, chain.spot_price),
                           use_container_width=True)

        with col2:
            st.plotly_chart(plot_heatmap(features, detector),
                           use_container_width=True)

        st.plotly_chart(plot_smile_comparison(params_list, chain.spot_price),
                       use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(plot_term_structure(features),
                           use_container_width=True)

        with col2:
            feature_select = st.selectbox(
                "Select feature for distribution",
                options=detector.FEATURE_NAMES,
                index=0,
            )
            current = getattr(features, feature_select, 0)
            st.plotly_chart(
                plot_distribution(detector, ticker, feature_select, current),
                use_container_width=True
            )

        # Features table
        st.markdown("### Surface Features")
        features_dict = features.to_dict()
        features_df = pd.DataFrame([
            {"Feature": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
            for k, v in features_dict.items()
            if k not in ['timestamp', 'ticker', 'svi_params'] and v is not None
        ])
        st.dataframe(features_df, hide_index=True)

    with tab3:
        render_anomaly_cards(report)

        st.markdown("### Anomaly Details")

        anomaly_data = []
        for a in report.anomalies:
            anomaly_data.append({
                "Feature": a.feature_name,
                "Current": f"{a.current_value:.4f}",
                "Mean": f"{a.historical_mean:.4f}" if not np.isnan(a.historical_mean) else "N/A",
                "Z-Score": f"{a.z_score:+.2f}",
                "Percentile": f"{a.percentile:.0f}%",
                "Status": "🚨 ANOMALY" if a.is_anomaly else "✅ Normal",
                "Direction": a.direction.upper(),
            })

        st.dataframe(pd.DataFrame(anomaly_data), hide_index=True)

    with tab4:
        render_trade_ideas(ideas)

    with tab5:
        render_math_panel(params_list, features)

    # Footer
    st.markdown("---")
    st.markdown(
        f"*Data as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"{'Synthetic' if use_synthetic else 'Live'} data*"
    )


if __name__ == "__main__":
    main()
