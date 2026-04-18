"""
decompose.py — Time Series Decomposition & Diagnostics Module (Extended)

Production-grade functions for STL/MSTL decomposition, stationarity testing,
structural break detection, and block bootstrap uncertainty quantification.

Author: Yun
Course: ECON 5200, Lab 20
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL, MSTL
from statsmodels.tsa.stattools import adfuller, kpss
import ruptures as rpt
from typing import List, Dict


def run_stl(series: pd.Series, period: int = 12, log_transform: bool = True,
            robust: bool = True):
    """Apply STL decomposition with optional log-transform.

    For multiplicative seasonality, log-transform converts
    Y = T * S * R into log(Y) = log(T) + log(S) + log(R).
    """
    if log_transform:
        if (series <= 0).any():
            raise ValueError("Non-positive values; cannot log-transform.")
        data = np.log(series)
    else:
        data = series
    return STL(data, period=period, robust=robust).fit()


def test_stationarity(series: pd.Series, alpha: float = 0.05) -> Dict:
    """Run ADF + KPSS and return the 2x2 decision table verdict.

    ADF null: unit root (non-stationary).
    KPSS null: trend-stationary.
    """
    adf_stat, adf_p, _, _, _, _ = adfuller(series, autolag='AIC', regression='ct')
    kpss_stat, kpss_p, _, _ = kpss(series, regression='ct', nlags='auto')
    adf_reject = adf_p < alpha
    kpss_reject = kpss_p < alpha

    if adf_reject and not kpss_reject:
        verdict = 'stationary'
    elif not adf_reject and kpss_reject:
        verdict = 'non-stationary'
    elif adf_reject and kpss_reject:
        verdict = 'contradictory'
    else:
        verdict = 'inconclusive'

    return {'adf_stat': adf_stat, 'adf_p': adf_p,
            'kpss_stat': kpss_stat, 'kpss_p': kpss_p, 'verdict': verdict}


def detect_breaks(series: pd.Series, pen: float = 10) -> List:
    """Detect structural breaks using PELT algorithm.

    Penalty controls bias-variance tradeoff:
    low penalty = more breaks (overfitting risk),
    high penalty = fewer breaks (miss real shifts).
    """
    signal = series.values
    algo = rpt.Pelt(model='rbf').fit(signal)
    bkps = algo.predict(pen=pen)
    return [series.index[bp] for bp in bkps if bp < len(signal)]


def run_mstl(series: pd.Series, periods: List[int], log_transform: bool = False):
    """Apply MSTL for multiple seasonal periods.

    MSTL iteratively removes seasonal components:
    1. Extract seasonal component for period[0] via STL
    2. Subtract from series
    3. Repeat for each subsequent period
    """
    if len(periods) < 1:
        raise ValueError("Provide at least one seasonal period.")
    if len(series) < 2 * max(periods):
        raise ValueError(f"Series too short for period {max(periods)}.")
    if log_transform:
        if (series <= 0).any():
            raise ValueError("Non-positive values; cannot log-transform.")
        data = np.log(series)
    else:
        data = series
    return MSTL(data, periods=periods).fit()


def block_bootstrap_trend(series: pd.Series, n_bootstrap: int = 200,
                          block_size: int = 8, period: int = 4,
                          confidence: float = 0.90, log_transform: bool = True,
                          robust: bool = True) -> Dict:
    """Generate block bootstrap confidence bands for STL trend.

    Block bootstrap preserves autocorrelation by resampling
    contiguous blocks. I.i.d. bootstrap (block_size=1) destroys
    temporal dependence, producing artificially narrow CIs.
    """
    stl_result = run_stl(series, period=period, log_transform=log_transform, robust=robust)
    orig_trend = stl_result.trend.values
    orig_seasonal = stl_result.seasonal.values
    orig_resid = stl_result.resid.values
    n = len(series)
    data = np.log(series) if log_transform else series

    np.random.seed(42)
    boot_trends = np.zeros((n_bootstrap, n))
    alpha = (1 - confidence) / 2

    for b in range(n_bootstrap):
        boot_resid = np.zeros(n)
        idx = 0
        while idx < n:
            start = np.random.randint(0, n - block_size + 1)
            block = orig_resid[start:start + block_size]
            end = min(idx + block_size, n)
            boot_resid[idx:end] = block[:end - idx]
            idx = end
        boot_series = pd.Series(orig_trend + orig_seasonal + boot_resid, index=data.index)
        boot_series.index.freq = data.index.freq
        boot_stl = STL(boot_series, period=period, robust=robust).fit()
        boot_trends[b, :] = boot_stl.trend.values

    return {
        'trend': orig_trend,
        'lower': np.percentile(boot_trends, alpha * 100, axis=0),
        'upper': np.percentile(boot_trends, (1 - alpha) * 100, axis=0),
        'boot_trends': boot_trends,
        'index': data.index
    }


if __name__ == '__main__':
    print('decompose.py loaded successfully.')
    print('Functions: run_stl, test_stationarity, detect_breaks, run_mstl, block_bootstrap_trend')
