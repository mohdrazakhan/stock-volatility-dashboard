"""Interactive Global Stock Index Volatility & Crisis Analysis Dashboard.

Runs:
- Daily Return Calculation & Distribution
- Stationarity Testing (Augmented Dickey-Fuller / ADF)
- ARCH Effect Testing (Engle's ARCH-LM)
- 30-Day Rolling Window Volatility
- GARCH(1,1) Volatility Modeling & Forecasting
- Regime Change Point Detection (PELT & Binary Segmentation)
- Statistical Crisis Comparison (2008 GFC vs COVID-19 via Welch's T-Test & Mann-Whitney U Test)

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

# Canonical Crisis Periods
CRISIS_PERIODS = {
    "2008 GFC": (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31")),
    "COVID-19": (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-30")),
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


def run_crisis_statistical_comparison(
    data: pd.DataFrame,
    volatility: pd.DataFrame,
    indexes: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Welch's T-Test and Mann-Whitney U Test comparing 2008 GFC vs COVID-19."""
    c2008_start, c2008_end = CRISIS_PERIODS["2008 GFC"]
    covid_start, covid_end = CRISIS_PERIODS["COVID-19"]

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
        s08 = sub[sub["Date"].between(c2008_start, c2008_end)]
        scov = sub[sub["Date"].between(covid_start, covid_end)]

        if len(s08) < 5 or len(scov) < 5:
            continue

        for label, col in metrics:
            v08 = s08[col].dropna().values
            vcov = scov[col].dropna().values

            if len(v08) < 5 or len(vcov) < 5:
                continue

            t_stat, t_pval = ttest_ind(v08, vcov, equal_var=False)
            u_stat, u_pval = mannwhitneyu(v08, vcov, alternative="two-sided")

            mean08, mean_cov = np.mean(v08), np.mean(vcov)
            med08, med_cov = np.median(v08), np.median(vcov)

            if "Volatility" in label:
                higher_crisis = "2008 GFC" if mean08 > mean_cov else "COVID-19"
            else:
                higher_crisis = "COVID-19" if mean_cov > mean08 else "2008 GFC"

            stat_results.append(
                {
                    "Index": idx,
                    "Metric": label,
                    "2008 N": len(v08),
                    "2008 Mean": mean08,
                    "2008 Median": med08,
                    "COVID N": len(vcov),
                    "COVID Mean": mean_cov,
                    "COVID Median": med_cov,
                    "Mean Difference (08 - Cov)": mean08 - mean_cov,
                    "T-Statistic": t_stat,
                    "T-Test p-val": t_pval,
                    "T-Test Sig (p<0.05)": "Yes" if t_pval < 0.05 else "No",
                    "Mann-Whitney U": u_stat,
                    "U-Test p-val": u_pval,
                    "U-Test Sig (p<0.05)": "Yes" if u_pval < 0.05 else "No",
                    "Higher Period": higher_crisis,
                }
            )

            for val in v08:
                distribution_samples.append(
                    {"Index": idx, "Crisis": "2008 GFC", "Metric": label, "Value": val}
                )
            for val in vcov:
                distribution_samples.append(
                    {"Index": idx, "Crisis": "COVID-19", "Metric": label, "Value": val}
                )

    return pd.DataFrame(stat_results), pd.DataFrame(distribution_samples)


def main() -> None:
    st.title("📈 Global Stock Index Volatility & Crisis Analytics")
    st.caption(
        "Interactive Econometric Suite: Stationarity (ADF), ARCH Effects, 30D Rolling & GARCH(1,1), "
        "Regime Breaks (PELT & BinSeg), and Crisis Hypothesis Testing (Welch's T-Test & Mann-Whitney U)."
    )

    try:
        data = load_data(str(DATA_PATH))
    except (OSError, ValueError) as error:
        st.error(f"Failed to load dataset: {error}")
        st.stop()

    available = sorted(data["Index"].unique())
    defaults = [name for name in DEFAULT_INDEXES if name in available] or available[:5]

    with st.sidebar:
        st.header("⚙️ Controls & Parameters")
        selected = st.multiselect("Stock Indexes", available, default=defaults)
        if not selected:
            st.warning("Please select at least one index.")
            st.stop()

        forecast_horizon = st.slider("GARCH Forecast Horizon (Days)", 5, 90, 30)

        min_date = data["Date"].min().date()
        max_date = data["Date"].max().date()
        date_range = st.date_input(
            "Analysis Period",
            value=(pd.Timestamp("2000-01-01").date(), max_date),
            min_value=min_date,
            max_value=max_date,
        )

        st.markdown("---")
        st.caption(f"📁 Dataset: `{DATA_PATH.name}` ({len(data):,} total rows)")
        st.caption("Crisis Windows:")
        st.caption("• 2008 GFC: 01-Sep-2008 to 31-Mar-2009")
        st.caption("• COVID-19: 01-Feb-2020 to 30-Jun-2020")

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
            "⚔️ Crisis Comparison (2008 vs COVID: T-Test & U-Test)",
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
        fig_close.update_layout(hovermode="x unified", height=450)
        st.plotly_chart(fig_close, use_container_width=True)

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
        fig_returns.update_layout(hovermode="x unified", height=450)
        st.plotly_chart(fig_returns, use_container_width=True)

        st.markdown("#### Daily Return Distributions (Histograms & Tails)")
        fig_hist = px.histogram(
            filtered,
            x="Daily Return (%)",
            color="Index",
            marginal="box",
            nbins=120,
            barmode="overlay",
            opacity=0.6,
            title="Return Distributions with Box Marginals (Fat Tail Inspection)",
        )
        fig_hist.update_layout(height=450)
        st.plotly_chart(fig_hist, use_container_width=True)

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
        fig_sq.update_layout(height=420, hovermode="x unified")
        st.plotly_chart(fig_sq, use_container_width=True)

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
        fig_vol.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig_vol, use_container_width=True)

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
        fig_fc.update_layout(height=420)
        st.plotly_chart(fig_fc, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: REGIME CHANGE POINT DETECTION (PELT & BINSEG)
    # -------------------------------------------------------------
    with tab_regimes:
        st.subheader("Volatility Regime Change Point Detection")
        st.caption(
            "Detect structural breaks in volatility regimes using **Binary Segmentation (BinSeg)** "
            "and **Pruned Exact Linear Time (PELT)**."
        )

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            target_index = st.selectbox("Select Index for Regime Breakdown", selected)
        with col_opt2:
            scope = st.selectbox(
                "Analysis Scope",
                ["Full Selected Period", "2008 GFC Period", "COVID-19 Period"],
            )
        with col_opt3:
            algo_choice = st.radio("Detection Algorithm", ["Both (Side-by-Side)", "Binary Segmentation", "PELT"], horizontal=True)

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            bkps_input = st.slider("BinSeg Breakpoint Count", 1, 8, 4)
        with c_p2:
            pelt_pen = st.slider("PELT Penalty Multiplier", 1.0, 15.0, 3.5, step=0.5)

        # Filter series for change-point detection
        sub_vol = period_volatility[period_volatility["Index"] == target_index].sort_values("Date")
        if scope == "2008 GFC Period":
            sub_vol = sub_vol[sub_vol["Date"].between(CRISIS_PERIODS["2008 GFC"][0], CRISIS_PERIODS["2008 GFC"][1])]
        elif scope == "COVID-19 Period":
            sub_vol = sub_vol[sub_vol["Date"].between(CRISIS_PERIODS["COVID-19"][0], CRISIS_PERIODS["COVID-19"][1])]

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
                height=480,
                hovermode="x unified",
            )
            st.plotly_chart(fig_cp, use_container_width=True)

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
    # TAB 5: CRISIS STATISTICAL COMPARISON (2008 GFC vs COVID-19)
    # -------------------------------------------------------------
    with tab_crises:
        st.subheader("Crisis Comparison: 2008 GFC vs COVID-19")
        st.markdown(
            "> **2008 GFC Period:** `2008-09-01` to `2009-03-31` | "
            "**COVID-19 Period:** `2020-02-01` to `2020-06-30`"
        )
        st.caption(
            "Hypothesis Testing: **Welch's Independent Two-Sample T-Test** (parametric test for difference in means) "
            "and **Mann-Whitney U Test** (non-parametric rank-sum test for difference in distributions)."
        )

        stat_comparison_df, dist_samples_df = run_crisis_statistical_comparison(
            data, volatility, tuple(selected)
        )

        if stat_comparison_df.empty:
            st.warning("Insufficient overlapping observations across the crisis windows for selected indexes.")
        else:
            st.markdown("#### Statistical Hypothesis Test Results (2008 GFC vs COVID-19)")
            st.dataframe(
                stat_comparison_df.style.format(
                    {
                        "2008 Mean": "{:.3f}",
                        "2008 Median": "{:.3f}",
                        "COVID Mean": "{:.3f}",
                        "COVID Median": "{:.3f}",
                        "Mean Difference (08 - Cov)": "{:.3f}",
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
            )

            metric_samples = dist_samples_df[dist_samples_df["Metric"] == metric_choice]
            if not metric_samples.empty:
                fig_box = px.box(
                    metric_samples,
                    x="Index",
                    y="Value",
                    color="Crisis",
                    points="outliers",
                    title=f"{metric_choice} Distribution: 2008 GFC vs COVID-19",
                    color_discrete_map={"2008 GFC": "#e74c3c", "COVID-19": "#3498db"},
                )
                fig_box.update_layout(height=460)
                st.plotly_chart(fig_box, use_container_width=True)

            # Volatility Correlation Heatmaps during Crises
            st.markdown("#### Cross-Index Volatility Correlations by Crisis")
            ch_col1, ch_col2 = st.columns(2)
            c2008_start, c2008_end = CRISIS_PERIODS["2008 GFC"]
            covid_start, covid_end = CRISIS_PERIODS["COVID-19"]

            with ch_col1:
                st.markdown("**2008 GFC Volatility Correlation**")
                mat08 = volatility[
                    volatility["Index"].isin(selected) & volatility["Date"].between(c2008_start, c2008_end)
                ].pivot(index="Date", columns="Index", values="GARCH Volatility (%)").corr()
                if mat08.shape[0] >= 2:
                    st.plotly_chart(
                        px.imshow(mat08, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r"),
                        use_container_width=True,
                    )
                else:
                    st.info("Select at least 2 indexes with 2008 data.")

            with ch_col2:
                st.markdown("**COVID-19 Volatility Correlation**")
                mat_cov = volatility[
                    volatility["Index"].isin(selected) & volatility["Date"].between(covid_start, covid_end)
                ].pivot(index="Date", columns="Index", values="GARCH Volatility (%)").corr()
                if mat_cov.shape[0] >= 2:
                    st.plotly_chart(
                        px.imshow(mat_cov, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r"),
                        use_container_width=True,
                    )
                else:
                    st.info("Select at least 2 indexes with COVID data.")

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
            if not stat_comparison_df.empty:
                st.download_button(
                    "📥 Download Crisis Statistical Comparison (CSV)",
                    stat_comparison_df.to_csv(index=False).encode("utf-8"),
                    "crisis-statistical-comparison.csv",
                    "text/csv",
                )

        st.dataframe(merged_period, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
