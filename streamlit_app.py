"""
Time Series Diagnostics Dashboard — Streamlit App
ECON 5200 Lab 20 AI Expansion

Author: Yun
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL, MSTL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import ruptures as rpt
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

# --- FRED API ---
FRED_API_KEY = '55a54e5815bd15d8e2e9ff63c4bc827a'
fred = Fred(api_key=FRED_API_KEY)

# --- Page config ---
st.set_page_config(page_title="Time Series Diagnostics", layout="wide")
st.title("Time Series Diagnostics & Decomposition Dashboard")
st.markdown("ECON 5200 | Lab 20 AI Expansion | Author: Yun")

# --- Sidebar ---
st.sidebar.header("Data Settings")
series_id = st.sidebar.text_input("FRED Series ID", value="RSXFSN")
start_date = st.sidebar.text_input("Start Date", value="2000-01-01")

st.sidebar.markdown("---")
st.sidebar.header("Decomposition")
method = st.sidebar.selectbox("Method", ["STL", "Classical", "MSTL"])
log_transform = st.sidebar.checkbox("Log Transform (for multiplicative data)", value=True)
robust = st.sidebar.checkbox("Robust STL", value=True)
period = st.sidebar.slider("Seasonal Period", min_value=2, max_value=52, value=12)

st.sidebar.markdown("---")
st.sidebar.header("Structural Breaks")
penalty = st.sidebar.slider("PELT Penalty", min_value=1, max_value=50, value=10)

st.sidebar.markdown("---")
st.sidebar.header("Block Bootstrap")
n_bootstrap = st.sidebar.slider("Bootstrap Replications", 50, 500, 200, step=50)
block_size = st.sidebar.slider("Block Size", 2, 20, 8)
run_bootstrap = st.sidebar.button("Generate Bootstrap CI")


# --- Helper functions ---
def run_stl_fn(series, period=12, log_transform=True, robust=True):
    if log_transform:
        if (series <= 0).any():
            raise ValueError("Non-positive values; cannot log-transform.")
        data = np.log(series)
    else:
        data = series
    return STL(data, period=period, robust=robust).fit()


def test_stationarity(series, alpha=0.05):
    adf_stat, adf_p, _, _, _, _ = adfuller(series, autolag='AIC', regression='ct')
    kpss_stat, kpss_p, _, _ = kpss(series, regression='ct', nlags='auto')
    adf_reject = adf_p < alpha
    kpss_reject = kpss_p < alpha
    if adf_reject and not kpss_reject:
        verdict = 'Stationary'
    elif not adf_reject and kpss_reject:
        verdict = 'Non-stationary'
    elif adf_reject and kpss_reject:
        verdict = 'Contradictory'
    else:
        verdict = 'Inconclusive'
    return {'adf_stat': adf_stat, 'adf_p': adf_p,
            'kpss_stat': kpss_stat, 'kpss_p': kpss_p, 'verdict': verdict}


def detect_breaks(series, pen=10):
    signal = series.values
    algo = rpt.Pelt(model='rbf').fit(signal)
    bkps = algo.predict(pen=pen)
    return [series.index[bp] for bp in bkps if bp < len(signal)]


# --- Fetch data ---
try:
    with st.spinner(f"Fetching {series_id} from FRED..."):
        raw = fred.get_series(series_id, observation_start=start_date)
        raw = raw.dropna()
        raw.index = pd.DatetimeIndex(raw.index)
        freq = pd.infer_freq(raw.index)
        if freq:
            raw.index.freq = freq
        else:
            raw = raw.asfreq(raw.index[1] - raw.index[0])
            raw = raw.ffill()
    st.success(f"Loaded {series_id}: {len(raw)} observations ({raw.index[0].date()} to {raw.index[-1].date()})")
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# --- 1. Raw series ---
st.header("1. Raw Series")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(raw, color='#2c3e50', linewidth=0.7)
ax1.set_title(f'{series_id} — Raw Series', fontsize=13)
ax1.set_ylabel('Value')
plt.tight_layout()
st.pyplot(fig1)

# --- 2. Decomposition ---
st.header("2. Decomposition")
try:
    if method == "STL":
        data = np.log(raw) if log_transform else raw
        result = STL(data, period=period, robust=robust).fit()
        fig2, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        data.plot(ax=axes[0], color='#2c3e50', linewidth=0.7)
        axes[0].set_ylabel('Observed')
        axes[0].set_title(f'STL Decomposition (period={period}, log={log_transform})', fontsize=13)
        result.trend.plot(ax=axes[1], color='#e67e22', linewidth=1.0)
        axes[1].set_ylabel('Trend')
        result.seasonal.plot(ax=axes[2], color='#27ae60', linewidth=0.7)
        axes[2].set_ylabel('Seasonal')
        result.resid.plot(ax=axes[3], color='#c0392b', linewidth=0.7)
        axes[3].set_ylabel('Residual')
        axes[3].axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig2)

    elif method == "Classical":
        data = np.log(raw) if log_transform else raw
        result = seasonal_decompose(data, period=period, model='additive')
        fig2 = result.plot()
        fig2.set_size_inches(12, 10)
        fig2.suptitle(f'Classical Decomposition (period={period})', fontsize=13, y=1.01)
        plt.tight_layout()
        st.pyplot(fig2)

    elif method == "MSTL":
        periods_input = st.text_input("MSTL Periods (comma-separated)", value="24,168")
        periods_list = [int(p.strip()) for p in periods_input.split(",")]
        data = np.log(raw) if log_transform else raw
        mstl_result = MSTL(data, periods=periods_list).fit()
        n_panels = 3 + len(periods_list)
        fig2, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)
        data.plot(ax=axes[0], color='#2c3e50', linewidth=0.5)
        axes[0].set_ylabel('Observed')
        axes[0].set_title('MSTL Decomposition', fontsize=13)
        mstl_result.trend.plot(ax=axes[1], color='#e67e22', linewidth=1.0)
        axes[1].set_ylabel('Trend')
        seasonal_df = mstl_result.seasonal
        for i, col in enumerate(seasonal_df.columns):
            seasonal_df[col].plot(ax=axes[2 + i], linewidth=0.5)
            axes[2 + i].set_ylabel(f'Season {periods_list[i]}')
        mstl_result.resid.plot(ax=axes[-1], color='#c0392b', linewidth=0.5)
        axes[-1].set_ylabel('Residual')
        plt.tight_layout()
        st.pyplot(fig2)
except Exception as e:
    st.error(f"Decomposition error: {e}")

# --- 3. Stationarity Tests ---
st.header("3. Stationarity Tests (ADF + KPSS)")
try:
    result_levels = test_stationarity(raw)
    result_diff = test_stationarity(raw.diff().dropna())
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Levels")
        st.metric("ADF p-value", f"{result_levels['adf_p']:.4f}")
        st.metric("KPSS p-value", f"{result_levels['kpss_p']:.4f}")
        st.metric("Verdict", result_levels['verdict'])
    with col2:
        st.subheader("First Difference")
        st.metric("ADF p-value", f"{result_diff['adf_p']:.4f}")
        st.metric("KPSS p-value", f"{result_diff['kpss_p']:.4f}")
        st.metric("Verdict", result_diff['verdict'])
except Exception as e:
    st.error(f"Stationarity test error: {e}")

# --- 4. Structural Breaks ---
st.header("4. Structural Break Detection (PELT)")
try:
    breaks = detect_breaks(raw, pen=penalty)
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.plot(raw, color='#2c3e50', linewidth=0.7, label=series_id)
    for i, bd in enumerate(breaks):
        ax4.axvline(bd, color='red', linestyle='--', alpha=0.7,
                    label=f'Break: {bd.date()}' if i < 5 else None)
    ax4.set_title(f'Structural Breaks (PELT, penalty={penalty})', fontsize=13)
    ax4.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig4)
    st.write(f"**Breaks detected:** {len(breaks)}")
    for bd in breaks:
        st.write(f"  - {bd.date()}")
except Exception as e:
    st.error(f"Break detection error: {e}")

# --- 5. Block Bootstrap ---
st.header("5. Block Bootstrap Confidence Band")
if run_bootstrap:
    with st.spinner(f"Running {n_bootstrap} bootstrap replications..."):
        try:
            stl_orig = run_stl_fn(raw, period=period, log_transform=log_transform, robust=robust)
            orig_trend = stl_orig.trend.values
            orig_seasonal = stl_orig.seasonal.values
            orig_resid = stl_orig.resid.values
            n = len(raw)
            data_for_stl = np.log(raw) if log_transform else raw

            np.random.seed(42)
            boot_trends = np.zeros((n_bootstrap, n))
            for b in range(n_bootstrap):
                boot_resid = np.zeros(n)
                idx = 0
                while idx < n:
                    start = np.random.randint(0, n - block_size + 1)
                    block = orig_resid[start:start + block_size]
                    end = min(idx + block_size, n)
                    boot_resid[idx:end] = block[:end - idx]
                    idx = end
                boot_series = pd.Series(
                    orig_trend + orig_seasonal + boot_resid,
                    index=data_for_stl.index
                )
                boot_series.index.freq = data_for_stl.index.freq
                boot_stl = STL(boot_series, period=period, robust=robust).fit()
                boot_trends[b, :] = boot_stl.trend.values

            lower = np.percentile(boot_trends, 5, axis=0)
            upper = np.percentile(boot_trends, 95, axis=0)

            fig5, ax5 = plt.subplots(figsize=(12, 5))
            ax5.fill_between(data_for_stl.index, lower, upper,
                             alpha=0.25, color='#3498db', label='90% Bootstrap CI')
            ax5.plot(data_for_stl.index, orig_trend, color='#e67e22',
                     linewidth=1.5, label='STL Trend')
            ax5.set_title('Block Bootstrap Confidence Band for Trend', fontsize=13)
            ax5.legend()
            plt.tight_layout()
            st.pyplot(fig5)

            ci_width = upper - lower
            st.write(f"**Mean CI width:** {ci_width.mean():.4f}")
            st.write(f"**CI width range:** [{ci_width.min():.4f}, {ci_width.max():.4f}]")
        except Exception as e:
            st.error(f"Bootstrap error: {e}")
else:
    st.info("Click 'Generate Bootstrap CI' in the sidebar to run.")

# --- Sensitivity note ---
st.markdown("---")
st.markdown("""
### Parameter Sensitivity Notes
- **Log transform**: Essential for multiplicative data (e.g., retail sales). Without it, STL seasonal amplitude grows with the level.
- **Period**: Must match the true seasonal cycle. Wrong period produces meaningless decomposition.
- **PELT penalty**: Lower values detect more breaks (risk of false positives); higher values detect fewer (risk of missing real shifts).
- **Block size**: Must be large enough to preserve autocorrelation. block_size=1 is equivalent to i.i.d. bootstrap, which destroys temporal dependence.
""")
