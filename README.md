# Time Series Diagnostics & Advanced Decomposition

## Objective

Diagnose, correct, and extend time series decomposition workflows on FRED economic data, demonstrating production-grade proficiency in STL/MSTL decomposition, stationarity testing, block bootstrap uncertainty quantification, and structural break detection.

## Methodology

- **STL Diagnosis & Fix**: Identified that additive STL was applied to retail sales data with multiplicative seasonality. Applied log-transform to convert multiplicative structure to additive before decomposition. Verified fix by confirming seasonal amplitude ratio fell within 0.7–1.3.
- **ADF Test Correction**: Diagnosed a misspecified ADF test using `regression='n'` on trending GDP data. Corrected to `regression='ct'` (constant + trend) and confirmed GDP is non-stationary (p = 0.96). Supplemented with KPSS test to form a 2×2 decision table.
- **MSTL Multi-Seasonal Decomposition**: Applied MSTL to simulated hourly electricity demand data with daily (24h) and weekly (168h) seasonal cycles, successfully separating both periodicities with residual std ≈ 12.2 (true noise = 15).
- **Moving Block Bootstrap**: Implemented block bootstrap (block_size=8, n=200) on log GDP to quantify trend uncertainty. Confirmed wider confidence bands during recessions (2008Q4 width = 0.0106 vs. 2019Q4 width = 0.0056).
- **PELT Structural Break Detection**: Applied PELT changepoint detection to GDP growth with per-regime ADF/KPSS stationarity testing, confirming GDP growth is stationary within detected regimes.
- **Production Module (`decompose.py`)**: Built a reusable Python module with five functions: `run_stl()`, `test_stationarity()`, `detect_breaks()`, `run_mstl()`, and `block_bootstrap_trend()`, all with full docstrings, type hints, and error handling.
- **Interactive Streamlit Dashboard**: Developed a Streamlit app allowing users to fetch any FRED series, select decomposition method (STL/Classical/MSTL), adjust parameters via sliders, view stationarity test results, overlay structural breaks, and generate block bootstrap confidence bands.

## Key Findings

- Retail sales (RSXFSN) exhibit multiplicative seasonality; applying additive STL without log-transform produces seasonal amplitude that grows over time, violating the additive assumption.
- Real GDP (GDPC1) is I(1): the ADF test fails to reject the unit root (p = 0.96) and KPSS rejects stationarity (p < 0.01). First-differenced GDP is stationary (ADF p ≈ 0, KPSS p > 0.10).
- Block bootstrap confidence bands widen during economic recessions, reflecting increased residual volatility and trend uncertainty during downturns.
- PELT structural break detection on GDP growth identifies regime shifts, with per-segment stationarity tests confirming that growth rates are stationary within each regime.

## How to Reproduce

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `notebooks/lab_20_time_series.ipynb` in VS Code or Colab
4. Run all cells top-to-bottom
5. For the Streamlit dashboard: `streamlit run streamlit_app.py`

## Repository Structure
