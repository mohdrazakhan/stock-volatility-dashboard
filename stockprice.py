"""Interactive Global Stock Index Volatility & Dynamic Crisis Analytics Dashboard.

Runs:
- Daily Return Calculation & Distribution
- Stationarity Testing (Augmented Dickey-Fuller / ADF)
- ARCH Effect Testing (Engle's ARCH-LM)
- 30-Day Rolling Window Volatility
- GARCH(1,1) Volatility Modeling & Forecasting
- Regime Change Point Detection (PELT & Binary Segmentation)
- Dynamic Multi-Crisis Management (Add, Remove, and Compare any Crisis Windows)
- Statistical Crisis Comparison (Welch's T-Test & Mann-Whitney U Test between any two crises)

Run with:
    streamlit run stockprice.py
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ruptures as rpt
import streamlit as st
from arch import arch_model
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller

DATA_PATH = Path(__file__).with_name("indexData.csv")
DEFAULT_INDEXES = ["N225", "IXIC", "GDAXI", "NSEI", "000001.SS"]

# Default Canonical Crisis Periods
DEFAULT_CRISIS_PERIODS = {
    "2008 GFC": (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31")),
    "COVID-19": (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-30")),
}

# Historical crisis presets for 1-click addition
CRISIS_PRESETS = {
    "Dot-Com Bubble Burst": (pd.Timestamp("2000-03-10"), pd.Timestamp("2002-10-09")),
    "European Debt Crisis": (pd.Timestamp("2011-05-01"), pd.Timestamp("2012-07-31")),
    "1997 Asian Financial Crisis": (pd.Timestamp("1997-07-01"), pd.Timestamp("1998-12-31")),
    "2015-16 Market Selloff": (pd.Timestamp("2015-08-01"), pd.Timestamp("2016-02-28")),
}

st.set_page_config(
    page_title="Global Index Volatility & Crisis Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load, validate, clean, and enrich the index dataset."""
    data = pd.read_csv(path)
    required = {"Date", "Index", "Close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data["Index"] = data["Index"].astype(str)
    data = (
        data.dropna(subset=["Date", "Index", "Close"])
        .sort_values(["Index", "Date"])
        .drop_duplicates(subset=["Index", "Date"], keep="last")
    )
    # Daily Returns (Ratio and Percentage)
    data["Daily Returns"] = data.groupby("Index")["Close"].pct_change()
    data["Daily Return (%)"] = data["Daily Returns"] * 100

    # 30-Day Rolling Volatility (Daily %)
    data["Rolling Volatility"] = (
        data.groupby("Index")["Daily Return (%)"]
        .transform(lambda values: values.rolling(30, min_periods=10).std())
    )
    return data.reset_index(drop=True)


@st.cache_data(show_spinner="Fitting GARCH(1,1) econometric models...")
def fit_garch_models(data: pd.DataFrame, indexes: tuple[str, ...], forecast_horizon: int):
    """Fit GARCH(1,1)-t models and return volatility, diagnostics, and forecasts."""
    volatility_frames = []
    forecast_frames = []
    diagnostics = []
    model_params = []

    for index_name in indexes:
        subset = data[data["Index"] == index_name].dropna(subset=["Daily Return (%)"]).copy()
        if len(subset) < 100:
            continue

        returns = subset["Daily Return (%)"].to_numpy()
        try:
            model = arch_model(returns, mean="Constant", vol="Garch", p=1, q=1, dist="t")
            result = model.fit(disp="off", show_warning=False)
        except Exception:
            continue

        conditional_volatility = np.asarray(result.conditional_volatility)
        volatility_frames.append(
            pd.DataFrame(
                {
                    "Date": subset["Date"].to_numpy(),
                    "Index": index_name,
                    "GARCH Volatility (%)": conditional_volatility,
                }
            )
        )

        # Forecast
        forecast = result.forecast(horizon=forecast_horizon, reindex=False)
        forecast_values = np.sqrt(np.asarray(forecast.variance.iloc[-1]))
        forecast_frames.append(
            pd.DataFrame(
                {
                    "Index": index_name,
                    "Day Ahead": np.arange(1, forecast_horizon + 1),
                    "Forecast Volatility (%)": forecast_values,
                }
            )
        )

        # Residual diagnostics
        standardized = pd.Series(result.std_resid).dropna()
        arch_test = het_arch(standardized, result_object=False)
        ljung_box = acorr_ljungbox(standardized, lags=[10], return_df=True)
        jarque_bera = stats.jarque_bera(standardized)
        adf_test = adfuller(returns, result_object=False)

        alpha = result.params.get("alpha[1]", np.nan)
        beta = result.params.get("beta[1]", np.nan)
        persistence = alpha + beta if not (np.isnan(alpha) or np.isnan(beta)) else np.nan

        diagnostics.append(
            {
                "Index": index_name,
                "Observations": len(returns),
                "ADF Returns p-val": adf_test[1],
                "Persistence (α+β)": persistence,
                "AIC": result.aic,
                "BIC": result.bic,
                "Log-Likelihood": result.loglikelihood,
                "ARCH-LM p-val": arch_test[1],
                "Ljung-Box p-val (Lag 10)": ljung_box["lb_pvalue"].iloc[0],
                "Jarque-Bera p-val": jarque_bera.pvalue,
            }
        )

        model_params.append(
            {
                "Index": index_name,
                "μ (Mean)": result.params.get("mu", np.nan),
                "ω (Omega)": result.params.get("omega", np.nan),
                "α[1] (ARCH)": alpha,
                "β[1] (GARCH)": beta,
                "ν (DoF)": result.params.get("nu", np.nan),
                "Persistence": persistence,
            }
        )

    if not volatility_frames:
        raise ValueError("No selected index has enough observations for GARCH modelling.")

    return (
        pd.concat(volatility_frames, ignore_index=True),
        pd.concat(forecast_frames, ignore_index=True),
        pd.DataFrame(diagnostics),
        pd.DataFrame(model_params),
    )


@st.cache_data(show_spinner="Running ADF & ARCH effect tests...")
def run_stationarity_and_arch_tests(data: pd.DataFrame, indexes: tuple[str, ...]) -> pd.DataFrame:
    """Run ADF unit-root tests on Close vs Returns, and Engle's ARCH-LM test on return residuals."""
    rows = []
    for idx in indexes:
        subset = data[data["Index"] == idx].dropna(subset=["Close", "Daily Return (%)"]).copy()
        if len(subset) < 30:
            continue

        close = subset["Close"].to_numpy()
        returns = subset["Daily Return (%)"].to_numpy()

        # ADF on Close Price
        adf_close = adfuller(close, result_object=False)
        # ADF on Daily Returns
        adf_returns = adfuller(returns, result_object=False)

        # ARCH-LM Test on Mean-Demeaned Returns
        demeaned = returns - np.mean(returns)
        arch_lm = het_arch(demeaned, result_object=False)

        rows.append(
            {
                "Index": idx,
                "Sample Size": len(returns),
                "ADF Close Stat": adf_close[0],
                "ADF Close p-val": adf_close[1],
                "Close State": "Unit Root (Non-Stationary)" if adf_close[1] > 0.05 else "Stationary",
                "ADF Returns Stat": adf_returns[0],
                "ADF Returns p-val": adf_returns[1],
                "Returns State": "Stationary (p < 0.05)" if adf_returns[1] < 0.05 else "Non-Stationary",
                "ARCH-LM Stat": arch_lm[0],
                "ARCH-LM p-val": arch_lm[1],
                "ARCH Effect": "Significant (Volatility Clustering)" if arch_lm[1] < 0.05 else "None",
            }
        )
    return pd.DataFrame(rows)


def detect_change_points(
    series_values: np.ndarray,
    dates: pd.Series,
    method: str = "binseg",
    n_bkps: int = 4,
    pen_multiplier: float = 3.0,
    jump: int = 5,
) -> list[dict]:
    """Detect volatility regime change points via PELT or Binary Segmentation."""
    if len(series_values) < 25:
        return []

    var_s = np.var(series_values)
    if var_s <= 0 or np.isnan(var_s):
        return []

    # Downsample large series to maintain sub-second interactive response
    step = max(1, len(series_values) // 1200)
    sub_vals = series_values[::step]
    sub_var = np.var(sub_vals)

    try:
        if method == "binseg":
            algo = rpt.Binseg(model="l2", jump=max(2, jump // step if step > 1 else jump)).fit(sub_vals)
            raw_cps = algo.predict(n_bkps=n_bkps)
        elif method == "pelt":
            algo = rpt.Pelt(model="l2", jump=max(2, jump // step if step > 1 else jump)).fit(sub_vals)
            penalty = max(0.5, pen_multiplier * np.log(len(sub_vals)) * sub_var)
            raw_cps = algo.predict(pen=penalty)
        else:
            return []

        change_points = [min(cp * step, len(series_values)) for cp in raw_cps]
    except Exception:
        return []

    results = []
    prev_idx = 0
    # change_points includes the end-of-series index as last element
    for cp in change_points:
        cp_safe = min(cp, len(series_values))
        regime_vals = series_values[prev_idx:cp_safe]
        if len(regime_vals) > 0 and prev_idx < len(dates):
            results.append(
                {
                    "Start Date": dates.iloc[prev_idx],
                    "End Date": dates.iloc[min(cp_safe - 1, len(dates) - 1)],
                    "Change Point Index": cp_safe,
                    "Regime Days": len(regime_vals),
                    "Mean Volatility (%)": np.mean(regime_vals),
                    "Min Volatility (%)": np.min(regime_vals),
                    "Max Volatility (%)": np.max(regime_vals),
                }
            )
        prev_idx = cp_safe
    return results


def run_dynamic_crisis_comparison(
    data: pd.DataFrame,
    volatility: pd.DataFrame,
    indexes: tuple[str, ...],
    crisis_a_name: str,
    crisis_a_dates: Tuple[pd.Timestamp, pd.Timestamp],
    crisis_b_name: str,
    crisis_b_dates: Tuple[pd.Timestamp, pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Welch's T-Test and Mann-Whitney U Test comparing any two crisis windows."""
    c_a_start, c_a_end = crisis_a_dates
    c_b_start, c_b_end = crisis_b_dates

    merged = pd.merge(data, volatility, on=["Date", "Index"], how="inner")

    metrics = [
        ("GARCH Volatility (%)", "GARCH Volatility (%)"),
        ("30D Rolling Volatility (%)", "Rolling Volatility"),
        ("Daily Return (%)", "Daily Return (%)"),
    ]

    stat_results = []
    distribution_samples = []

    for idx in indexes:
        sub = merged[merged["Index"] == idx]
        s_a = sub[sub["Date"].between(c_a_start, c_a_end)]
        s_b = sub[sub["Date"].between(c_b_start, c_b_end)]

        if len(s_a) < 5 or len(s_b) < 5:
            continue

        for label, col in metrics:
            v_a = s_a[col].dropna().values
            v_b = s_b[col].dropna().values

            if len(v_a) < 5 or len(v_b) < 5:
                continue

            t_stat, t_pval = ttest_ind(v_a, v_b, equal_var=False)
            u_stat, u_pval = mannwhitneyu(v_a, v_b, alternative="two-sided")

            mean_a, mean_b = np.mean(v_a), np.mean(v_b)
            med_a, med_b = np.median(v_a), np.median(v_b)

            if "Volatility" in label:
                higher_crisis = crisis_a_name if mean_a > mean_b else crisis_b_name
            else:
                higher_crisis = crisis_b_name if mean_b > mean_a else crisis_a_name

            stat_results.append(
                {
                    "Index": idx,
                    "Metric": label,
                    f"{crisis_a_name} N": len(v_a),
                    f"{crisis_a_name} Mean": mean_a,
                    f"{crisis_a_name} Median": med_a,
                    f"{crisis_b_name} N": len(v_b),
                    f"{crisis_b_name} Mean": mean_b,
                    f"{crisis_b_name} Median": med_b,
                    f"Diff ({crisis_a_name} - {crisis_b_name})": mean_a - mean_b,
                    "T-Statistic": t_stat,
                    "T-Test p-val": t_pval,
                    "T-Test Sig (p<0.05)": "Yes" if t_pval < 0.05 else "No",
                    "Mann-Whitney U": u_stat,
                    "U-Test p-val": u_pval,
                    "U-Test Sig (p<0.05)": "Yes" if u_pval < 0.05 else "No",
                    "Higher Period": higher_crisis,
                }
            )

            for val in v_a:
                distribution_samples.append(
                    {"Index": idx, "Crisis": crisis_a_name, "Metric": label, "Value": val}
                )
            for val in v_b:
                distribution_samples.append(
                    {"Index": idx, "Crisis": crisis_b_name, "Metric": label, "Value": val}
                )

    return pd.DataFrame(stat_results), pd.DataFrame(distribution_samples)


def render_mobile_chart(
    fig: go.Figure,
    height: int = 370,
    legend_top: bool = True,
) -> None:
    """Render Plotly figure with mobile-optimized layout, margins, and touch config."""
    updates = dict(
        height=height,
        autosize=True,
        margin=dict(l=8, r=8, t=38, b=25),
        hoverlabel=dict(font_size=11),
    )
    if legend_top:
        updates["legend"] = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.4)",
        )
    fig.update_layout(**updates)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
            ],
            "scrollZoom": False,
        },
    )


def generate_multi_crisis_summary(
    data: pd.DataFrame,
    volatility: pd.DataFrame,
    indexes: tuple[str, ...],
    crisis_periods: dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    """Generate overall summary statistics across all active crisis periods."""
    merged = pd.merge(data, volatility, on=["Date", "Index"], how="inner")
    rows = []
    for c_name, (c_start, c_end) in crisis_periods.items():
        sub_c = merged[merged["Index"].isin(indexes) & merged["Date"].between(c_start, c_end)]
        if sub_c.empty:
            continue
        for idx in indexes:
            sub_idx = sub_c[sub_c["Index"] == idx]
            if len(sub_idx) < 3:
                continue
            rows.append(
                {
                    "Crisis Window": c_name,
                    "Index": idx,
                    "Start Date": str(c_start.date()),
                    "End Date": str(c_end.date()),
                    "Observations": len(sub_idx),
                    "Mean GARCH Vol (%)": sub_idx["GARCH Volatility (%)"].mean(),
                    "Peak GARCH Vol (%)": sub_idx["GARCH Volatility (%)"].max(),
                    "Mean Rolling Vol (%)": sub_idx["Rolling Volatility"].mean(),
                    "Mean Daily Return (%)": sub_idx["Daily Return (%)"].mean(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    st.markdown(
        """
        <style>
        /* Mobile-First Responsive Adjustments */
        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem !important;
                padding-bottom: 2rem !important;
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
            /* Smooth Horizontal Swipeable Tabs on Mobile */
            .stTabs [data-baseweb="tab-list"] {
                gap: 4px;
                overflow-x: auto !important;
                overflow-y: hidden !important;
                flex-wrap: nowrap !important;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: thin;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 6px 12px !important;
                font-size: 0.82rem !important;
                white-space: nowrap !important;
            }
            /* Scale Metrics & Headers for Mobile Screens */
            [data-testid="stMetricValue"] {
                font-size: 1.25rem !important;
            }
            [data-testid="stMetricLabel"] {
                font-size: 0.75rem !important;
            }
            h1 { font-size: 1.45rem !important; }
            h2 { font-size: 1.25rem !important; }
            h3 { font-size: 1.1rem !important; }
            h4 { font-size: 0.98rem !important; }
            /* Expand Plotly containers to edge on phones */
            .js-plotly-plot, .plot-container {
                width: 100% !important;
                max-width: 100vw !important;
            }
            /* Scrollable tables on mobile */
            [data-testid="stDataFrame"] {
                max-width: 100vw !important;
                overflow-x: auto !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📈 Global Stock Index Volatility & Crisis Analytics")
    st.caption(
        "Interactive Econometric Suite: Stationarity (ADF), ARCH Effects, 30D Rolling & GARCH(1,1), "
        "Regime Breaks (PELT & BinSeg), and Dynamic Multi-Crisis Hypothesis Testing (Welch's T-Test & Mann-Whitney U)."
    )

    try:
        data = load_data(str(DATA_PATH))
    except (OSError, ValueError) as error:
        st.error(f"Failed to load dataset: {error}")
        st.stop()

    available = sorted(data["Index"].unique())
    defaults = [name for name in DEFAULT_INDEXES if name in available] or available[:5]

    # Initialize Crisis Periods in Session State
    if "crisis_periods" not in st.session_state:
        st.session_state.crisis_periods = dict(DEFAULT_CRISIS_PERIODS)

    min_date = data["Date"].min().date()
    max_date = data["Date"].max().date()

    with st.sidebar:
        st.header("⚙️ Controls & Parameters")
        selected = st.multiselect("Stock Indexes", available, default=defaults)
        if not selected:
            st.warning("Please select at least one index.")
            st.stop()

        forecast_horizon = st.slider("GARCH Forecast Horizon (Days)", 5, 90, 30)

        date_range = st.date_input(
            "Analysis Period",
            value=(pd.Timestamp("2000-01-01").date(), max_date),
            min_value=min_date,
            max_value=max_date,
        )

        st.markdown("---")
        st.subheader("🛡️ Crisis Windows Manager")
        st.caption("Add, remove, or customize crisis windows dynamically:")

        # List existing crisis windows with delete buttons
        crisis_items = list(st.session_state.crisis_periods.items())
        for name, (start_dt, end_dt) in crisis_items:
            c_col1, c_col2 = st.columns([5, 1])
            with c_col1:
                st.markdown(f"**{name}**")
                st.caption(f"`{start_dt.date()}` to `{end_dt.date()}`")
            with c_col2:
                if st.button("🗑️", key=f"del_{name}", help=f"Remove '{name}'"):
                    del st.session_state.crisis_periods[name]
                    st.rerun()

        # Add / Preset Crisis Expander
        with st.expander("➕ Add Crisis Window", expanded=False):
            # Preset 1-click adder
            preset_options = [p for p in CRISIS_PRESETS.keys() if p not in st.session_state.crisis_periods]
            if preset_options:
                selected_preset = st.selectbox("Select Popular Preset", preset_options)
                if st.button("Add Preset Window", key="btn_add_preset"):
                    st.session_state.crisis_periods[selected_preset] = CRISIS_PRESETS[selected_preset]
                    st.rerun()
                st.markdown("---")

            # Custom Crisis Form
            with st.form("custom_crisis_form", clear_on_submit=True):
                st.markdown("**Add Custom Window:**")
                custom_name = st.text_input("Crisis Name", placeholder="e.g. 2011 Debt Crisis")
                custom_dates = st.date_input(
                    "Crisis Date Range",
                    value=(pd.Timestamp("2011-05-01").date(), pd.Timestamp("2012-07-31").date()),
                    min_value=min_date,
                    max_value=max_date,
                )
                submitted = st.form_submit_button("Add Custom Window")
                if submitted:
                    if not custom_name.strip():
                        st.error("Please enter a crisis name.")
                    elif custom_name.strip() in st.session_state.crisis_periods:
                        st.error(f"Crisis '{custom_name.strip()}' already exists.")
                    elif isinstance(custom_dates, (tuple, list)) and len(custom_dates) == 2:
                        st.session_state.crisis_periods[custom_name.strip()] = (
                            pd.Timestamp(custom_dates[0]),
                            pd.Timestamp(custom_dates[1]),
                        )
                        st.success(f"Added {custom_name.strip()}!")
                        st.rerun()
                    else:
                        st.error("Please select both start and end dates.")

        if st.button("🔄 Reset to Default Crises"):
            st.session_state.crisis_periods = dict(DEFAULT_CRISIS_PERIODS)
            st.rerun()

        st.markdown("---")
        st.caption(f"📁 Dataset: `{DATA_PATH.name}` ({len(data):,} total rows)")

    # Safe Date Range Unpacking
    if isinstance(date_range, (tuple, list)):
        if len(date_range) == 2:
            start_date, end_date = date_range
        elif len(date_range) == 1:
            st.info("Select an end date to complete the analysis period.")
            st.stop()
        else:
            st.stop()
    else:
        start_date = end_date = date_range

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    filtered = data[data["Index"].isin(selected) & data["Date"].between(start_ts, end_ts)]

    if filtered.empty:
        st.warning("No observations found for the chosen date range and indexes.")
        st.stop()

    # Fit GARCH Models (Full series per index for econometric stability)
    try:
        volatility, forecasts, diagnostics, model_params = fit_garch_models(
            data, tuple(selected), forecast_horizon
        )
    except ValueError as err:
        st.error(str(err))
        st.stop()

    period_volatility = volatility[volatility["Date"].between(start_ts, end_ts)]
    merged_period = pd.merge(filtered, period_volatility, on=["Date", "Index"], how="left")

    # Navigation Tabs
    tab_overview, tab_stationarity, tab_garch, tab_regimes, tab_crises, tab_export = st.tabs(
        [
            "📊 Overview & Returns",
            "🔬 Stationarity & ARCH Tests (ADF / ARCH-LM)",
            "📈 30D Rolling & GARCH(1,1)",
            "⚡ Change Point Detection (PELT & BinSeg)",
            "⚔️ Dynamic Crisis Comparison (T-Test & U-Test)",
            "📥 Data & Export",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: OVERVIEW & RETURNS
    # -------------------------------------------------------------
    with tab_overview:
        st.subheader("Market Summary & Return Dynamics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Indexes", len(selected))
        c2.metric("Observations in Period", f"{len(filtered):,}")
        c3.metric("Earliest Date", str(filtered["Date"].min().date()))
        c4.metric("Latest Date", str(filtered["Date"].max().date()))

        idx_summary = (
            filtered.groupby("Index")
            .agg(
                Observations=("Close", "count"),
                Latest_Close=("Close", "last"),
                Mean_Daily_Return=("Daily Return (%)", "mean"),
                Return_Std_Dev=("Daily Return (%)", "std"),
                Avg_Rolling_Vol=("Rolling Volatility", "mean"),
            )
            .reset_index()
        )
        st.dataframe(
            idx_summary.style.format(
                {
                    "Latest_Close": "{:,.2f}",
                    "Mean_Daily_Return": "{:.3f}%",
                    "Return_Std_Dev": "{:.3f}%",
                    "Avg_Rolling_Vol": "{:.3f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Closing Prices")
        fig_close = px.line(
            filtered,
            x="Date",
            y="Close",
            color="Index",
            title="Index Closing Prices",
            labels={"Close": "Closing Price"},
        )
        fig_close.update_layout(hovermode="x unified")
        render_mobile_chart(fig_close, height=370)

        st.markdown("#### Daily Return Series (%) with Volatility Bursts")
        fig_returns = px.line(
            filtered,
            x="Date",
            y="Daily Return (%)",
            color="Index",
            title="Daily Returns (%)",
            labels={"Daily Return (%)": "Return (%)"},
        )
        fig_returns.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig_returns.update_layout(hovermode="x unified")
        render_mobile_chart(fig_returns, height=370)

        st.markdown("#### Daily Return Distributions (Histograms & Tails)")
        fig_hist = px.histogram(
            filtered,
            x="Daily Return (%)",
            color="Index",
            marginal="box",
            nbins=100,
            barmode="overlay",
            opacity=0.6,
            title="Return Distributions with Box Marginals",
        )
        render_mobile_chart(fig_hist, height=370)

    # -------------------------------------------------------------
    # TAB 2: STATIONARITY & ARCH TESTS
    # -------------------------------------------------------------
    with tab_stationarity:
        st.subheader("Econometric Specification Tests")
        st.info(
            "**Methodological Significance:**\n"
            "- **Augmented Dickey-Fuller (ADF) Test**: Evaluates the null hypothesis of a unit root ($H_0$: Non-stationary). "
            "Asset prices are typically integrated of order 1, $I(1)$, while daily returns are stationary, $I(0)$ ($p < 0.05$).\n"
            "- **Engle's ARCH-LM Test**: Evaluates the null hypothesis of no autoregressive conditional heteroskedasticity. "
            "A rejection ($p < 0.05$) formally proves volatility clustering, justifying GARCH modeling."
        )

        stat_tests_df = run_stationarity_and_arch_tests(data, tuple(selected))
        st.markdown("#### Stationarity (ADF) & ARCH Effect (ARCH-LM) Test Results")
        st.dataframe(
            stat_tests_df.style.format(
                {
                    "ADF Close Stat": "{:.3f}",
                    "ADF Close p-val": "{:.4e}",
                    "ADF Returns Stat": "{:.3f}",
                    "ADF Returns p-val": "{:.4e}",
                    "ARCH-LM Stat": "{:.2f}",
                    "ARCH-LM p-val": "{:.4e}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Squared Returns: Empirical Proof of Volatility Clustering")
        st.caption("Clusters of high squared returns signal time-varying conditional variance.")
        sq_df = filtered.copy()
        sq_df["Squared Return (%)"] = sq_df["Daily Return (%)"] ** 2
        fig_sq = px.line(
            sq_df,
            x="Date",
            y="Squared Return (%)",
            color="Index",
            title="Squared Daily Returns (Empirical Volatility Bursts)",
        )
        fig_sq.update_layout(hovermode="x unified")
        render_mobile_chart(fig_sq, height=360)

    # -------------------------------------------------------------
    # TAB 3: ROLLING VOLATILITY & GARCH(1,1)
    # -------------------------------------------------------------
    with tab_garch:
        st.subheader("30-Day Rolling Window vs GARCH(1,1) Conditional Volatility")

        vol_combined = pd.concat(
            [
                filtered[["Date", "Index", "Rolling Volatility"]].rename(
                    columns={"Rolling Volatility": "Volatility (%)"}
                ).assign(Model="30-Day Rolling Window"),
                period_volatility.rename(
                    columns={"GARCH Volatility (%)": "Volatility (%)"}
                ).assign(Model="GARCH(1,1) Conditional"),
            ]
        )

        fig_vol = px.line(
            vol_combined,
            x="Date",
            y="Volatility (%)",
            color="Index",
            line_dash="Model",
            title="Rolling vs GARCH(1,1) Conditional Volatility Comparison",
        )
        fig_vol.update_layout(hovermode="x unified")
        render_mobile_chart(fig_vol, height=420)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### GARCH(1,1) Estimated Parameters")
            st.caption(
                "Model: $r_t = \\mu + \\epsilon_t,\\; \\sigma_t^2 = \\omega + \\alpha \\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$. "
                "Persistence $\\alpha + \\beta \\to 1$ implies long volatility memory."
            )
            st.dataframe(
                model_params[model_params["Index"].isin(selected)].style.format(
                    {col: "{:.4f}" for col in model_params.columns if col != "Index"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        with col_right:
            st.markdown("#### GARCH Model Fit & Residual Diagnostics")
            st.caption(
                "Ljung-Box ($p > 0.05$) and ARCH-LM ($p > 0.05$) on standardized residuals confirm that the GARCH model successfully eliminated autocorrelation and ARCH effects."
            )
            st.dataframe(
                diagnostics[diagnostics["Index"].isin(selected)].style.format(
                    {
                        "Persistence (α+β)": "{:.4f}",
                        "AIC": "{:,.1f}",
                        "BIC": "{:,.1f}",
                        "Log-Likelihood": "{:,.1f}",
                        "ADF Returns p-val": "{:.4e}",
                        "ARCH-LM p-val": "{:.4f}",
                        "Ljung-Box p-val (Lag 10)": "{:.4f}",
                        "Jarque-Bera p-val": "{:.4e}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown(f"#### GARCH Volatility Forecast ({forecast_horizon} Days Ahead)")
        fig_fc = px.line(
            forecasts[forecasts["Index"].isin(selected)],
            x="Day Ahead",
            y="Forecast Volatility (%)",
            color="Index",
            markers=True,
            title=f"{forecast_horizon}-Day Ahead Term Structure of Volatility",
        )
        render_mobile_chart(fig_fc, height=360)

    # -------------------------------------------------------------
    # TAB 4: REGIME CHANGE POINT DETECTION (PELT & BINSEG)
    # -------------------------------------------------------------
    with tab_regimes:
        st.subheader("Volatility Regime Change Point Detection")
        st.caption(
            "Detect structural breaks in volatility regimes using **Binary Segmentation (BinSeg)** "
            "and **Pruned Exact Linear Time (PELT)**."
        )

        active_crises = st.session_state.crisis_periods
        scope_options = ["Full Selected Period"] + [f"Crisis: {name}" for name in active_crises.keys()]

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            target_index = st.selectbox("Select Index for Regime Breakdown", selected)
        with col_opt2:
            scope = st.selectbox("Analysis Scope", scope_options)
        with col_opt3:
            algo_choice = st.radio(
                "Detection Algorithm", ["Both (Side-by-Side)", "Binary Segmentation", "PELT"], horizontal=True
            )

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            bkps_input = st.slider("BinSeg Breakpoint Count", 1, 8, 4)
        with c_p2:
            pelt_pen = st.slider("PELT Penalty Multiplier", 1.0, 15.0, 3.5, step=0.5)

        # Filter series for change-point detection
        sub_vol = period_volatility[period_volatility["Index"] == target_index].sort_values("Date")
        if scope.startswith("Crisis: "):
            c_name = scope.replace("Crisis: ", "")
            if c_name in active_crises:
                c_s, c_e = active_crises[c_name]
                sub_vol = sub_vol[sub_vol["Date"].between(c_s, c_e)]

        if len(sub_vol) < 25:
            st.warning(f"Not enough observations ({len(sub_vol)}) in this scope for change point detection.")
        else:
            dates = sub_vol["Date"]
            vol_vals = sub_vol["GARCH Volatility (%)"].to_numpy()

            binseg_regimes = detect_change_points(vol_vals, dates, method="binseg", n_bkps=bkps_input)
            pelt_regimes = detect_change_points(vol_vals, dates, method="pelt", pen_multiplier=pelt_pen)

            fig_cp = go.Figure()
            fig_cp.add_trace(
                go.Scatter(
                    x=dates,
                    y=vol_vals,
                    mode="lines",
                    name=f"{target_index} GARCH Volatility",
                    line=dict(color="#2980b9", width=2),
                )
            )

            # Draw Regime lines
            if algo_choice in ["Binary Segmentation", "Both (Side-by-Side)"]:
                for r in binseg_regimes[:-1]:
                    fig_cp.add_vline(
                        x=r["End Date"],
                        line=dict(color="red", width=1.5, dash="dash"),
                        annotation_text="BinSeg Break",
                        annotation_position="top left",
                    )

            if algo_choice in ["PELT", "Both (Side-by-Side)"]:
                for r in pelt_regimes[:-1]:
                    fig_cp.add_vline(
                        x=r["End Date"],
                        line=dict(color="purple", width=1.5, dash="dot"),
                        annotation_text="PELT Break",
                        annotation_position="bottom right",
                    )

            fig_cp.update_layout(
                title=f"Volatility Regime Breaks ({target_index} - {scope})",
                xaxis_title="Date",
                yaxis_title="GARCH Volatility (%)",
                hovermode="x unified",
            )
            render_mobile_chart(fig_cp, height=390)

            rcol1, rcol2 = st.columns(2)
            with rcol1:
                st.markdown("#### Binary Segmentation Regimes")
                if binseg_regimes:
                    df_b = pd.DataFrame(binseg_regimes)
                    st.dataframe(
                        df_b.style.format(
                            {
                                "Start Date": lambda d: str(d.date()) if hasattr(d, "date") else str(d),
                                "End Date": lambda d: str(d.date()) if hasattr(d, "date") else str(d),
                                "Mean Volatility (%)": "{:.2f}%",
                                "Min Volatility (%)": "{:.2f}%",
                                "Max Volatility (%)": "{:.2f}%",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No breaks detected.")

            with rcol2:
                st.markdown("#### PELT Regimes")
                if pelt_regimes:
                    df_p = pd.DataFrame(pelt_regimes)
                    st.dataframe(
                        df_p.style.format(
                            {
                                "Start Date": lambda d: str(d.date()) if hasattr(d, "date") else str(d),
                                "End Date": lambda d: str(d.date()) if hasattr(d, "date") else str(d),
                                "Mean Volatility (%)": "{:.2f}%",
                                "Min Volatility (%)": "{:.2f}%",
                                "Max Volatility (%)": "{:.2f}%",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No breaks detected.")

    # -------------------------------------------------------------
    # TAB 5: DYNAMIC CRISIS COMPARISON (T-TEST & MANN-WHITNEY U)
    # -------------------------------------------------------------
    with tab_crises:
        st.subheader("Dynamic Multi-Crisis Analysis & Statistical Testing")
        active_crises = st.session_state.crisis_periods

        if not active_crises:
            st.warning("No crisis windows currently defined. Please add crisis windows in the sidebar.")
        else:
            # Multi-Crisis Overview Table
            st.markdown("#### 🌐 Multi-Crisis Macro Overview")
            st.caption("Summary statistics across all active crisis windows for selected indexes:")
            multi_summary = generate_multi_crisis_summary(data, volatility, tuple(selected), active_crises)
            if not multi_summary.empty:
                st.dataframe(
                    multi_summary.style.format(
                        {
                            "Mean GARCH Vol (%)": "{:.2f}%",
                            "Peak GARCH Vol (%)": "{:.2f}%",
                            "Mean Rolling Vol (%)": "{:.2f}%",
                            "Mean Daily Return (%)": "{:.3f}%",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                # Bar chart comparing mean volatility across crises
                fig_mbar = px.bar(
                    multi_summary,
                    x="Crisis Window",
                    y="Mean GARCH Vol (%)",
                    color="Index",
                    barmode="group",
                    title="Average GARCH Volatility Across Defined Crisis Windows",
                )
                render_mobile_chart(fig_mbar, height=360)

            st.markdown("---")
            st.markdown("#### ⚔️ Head-to-Head Hypothesis Testing (Welch's T-Test & Mann-Whitney U)")
            st.caption(
                "Select any two crisis windows to statistically test differences in volatility distributions."
            )

            crisis_keys = list(active_crises.keys())
            if len(crisis_keys) < 2:
                st.info(
                    "You have only 1 crisis window defined. Please add at least one more crisis window in the sidebar to run two-sample statistical tests."
                )
            else:
                h_col1, h_col2 = st.columns(2)
                with h_col1:
                    crisis_a_name = st.selectbox(
                        "Primary Crisis (Window A)",
                        crisis_keys,
                        index=0,
                        key="sel_crisis_a",
                    )
                with h_col2:
                    default_b_idx = 1 if len(crisis_keys) > 1 else 0
                    crisis_b_name = st.selectbox(
                        "Comparison Crisis (Window B)",
                        crisis_keys,
                        index=default_b_idx,
                        key="sel_crisis_b",
                    )

                if crisis_a_name == crisis_b_name:
                    st.warning("Please select two different crisis windows to run hypothesis testing.")
                else:
                    crisis_a_dates = active_crises[crisis_a_name]
                    crisis_b_dates = active_crises[crisis_b_name]

                    st.markdown(
                        f"> Comparing **{crisis_a_name}** (`{crisis_a_dates[0].date()}` to `{crisis_a_dates[1].date()}`) "
                        f"vs. **{crisis_b_name}** (`{crisis_b_dates[0].date()}` to `{crisis_b_dates[1].date()}`)"
                    )

                    stat_comparison_df, dist_samples_df = run_dynamic_crisis_comparison(
                        data,
                        volatility,
                        tuple(selected),
                        crisis_a_name,
                        crisis_a_dates,
                        crisis_b_name,
                        crisis_b_dates,
                    )

                    if stat_comparison_df.empty:
                        st.warning(
                            "Insufficient overlapping observations across the selected crises for these indexes."
                        )
                    else:
                        st.dataframe(
                            stat_comparison_df.style.format(
                                {
                                    f"{crisis_a_name} Mean": "{:.3f}",
                                    f"{crisis_a_name} Median": "{:.3f}",
                                    f"{crisis_b_name} Mean": "{:.3f}",
                                    f"{crisis_b_name} Median": "{:.3f}",
                                    f"Diff ({crisis_a_name} - {crisis_b_name})": "{:.3f}",
                                    "T-Statistic": "{:.3f}",
                                    "T-Test p-val": "{:.4e}",
                                    "Mann-Whitney U": "{:,.0f}",
                                    "U-Test p-val": "{:.4e}",
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.markdown("#### Distribution Comparison (Boxplots & Violins)")
                        metric_choice = st.selectbox(
                            "Select Metric to Visualize",
                            ["GARCH Volatility (%)", "30D Rolling Volatility (%)", "Daily Return (%)"],
                            key="sel_box_metric",
                        )

                        metric_samples = dist_samples_df[dist_samples_df["Metric"] == metric_choice]
                        if not metric_samples.empty:
                            fig_box = px.box(
                                metric_samples,
                                x="Index",
                                y="Value",
                                color="Crisis",
                                points="outliers",
                                title=f"{metric_choice} Distribution: {crisis_a_name} vs {crisis_b_name}",
                                color_discrete_sequence=["#e74c3c", "#3498db"],
                            )
                            render_mobile_chart(fig_box, height=380)

                        # Volatility Correlation Heatmaps during Crises
                        st.markdown("#### Cross-Index Volatility Correlations")
                        ch_col1, ch_col2 = st.columns(2)

                        with ch_col1:
                            st.markdown(f"**{crisis_a_name} Volatility Correlation**")
                            mat_a = volatility[
                                volatility["Index"].isin(selected)
                                & volatility["Date"].between(crisis_a_dates[0], crisis_a_dates[1])
                            ].pivot(index="Date", columns="Index", values="GARCH Volatility (%)").corr()
                            if mat_a.shape[0] >= 2:
                                fig_mat_a = px.imshow(
                                    mat_a,
                                    text_auto=".2f",
                                    zmin=-1,
                                    zmax=1,
                                    color_continuous_scale="RdBu_r",
                                    title=f"{crisis_a_name} Correlation",
                                )
                                render_mobile_chart(fig_mat_a, height=320, legend_top=False)
                            else:
                                st.info(f"Select at least 2 indexes with {crisis_a_name} data.")

                        with ch_col2:
                            st.markdown(f"**{crisis_b_name} Volatility Correlation**")
                            mat_b = volatility[
                                volatility["Index"].isin(selected)
                                & volatility["Date"].between(crisis_b_dates[0], crisis_b_dates[1])
                            ].pivot(index="Date", columns="Index", values="GARCH Volatility (%)").corr()
                            if mat_b.shape[0] >= 2:
                                fig_mat_b = px.imshow(
                                    mat_b,
                                    text_auto=".2f",
                                    zmin=-1,
                                    zmax=1,
                                    color_continuous_scale="RdBu_r",
                                    title=f"{crisis_b_name} Correlation",
                                )
                                render_mobile_chart(fig_mat_b, height=320, legend_top=False)
                            else:
                                st.info(f"Select at least 2 indexes with {crisis_b_name} data.")

    # -------------------------------------------------------------
    # TAB 6: DATA & EXPORT
    # -------------------------------------------------------------
    with tab_export:
        st.subheader("Data Inspection & Export")
        st.caption("Download cleaned, filtered, or statistical analysis datasets.")

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                "📥 Download Filtered Time-Series (CSV)",
                merged_period.to_csv(index=False).encode("utf-8"),
                "filtered-stock-volatility.csv",
                "text/csv",
            )
        with d_col2:
            if "multi_summary" in locals() and not multi_summary.empty:
                st.download_button(
                    "📥 Download Multi-Crisis Summary (CSV)",
                    multi_summary.to_csv(index=False).encode("utf-8"),
                    "multi-crisis-summary.csv",
                    "text/csv",
                )

        st.dataframe(merged_period, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
