# financial-volatility-lppl-garch

# GARCH-LPPL Integration for Financial Volatility Forecasting

## Research Overview

This project investigates whether incorporating log-periodic power law (LPPL) signals into GARCH models can improve volatility forecasting, particularly before market crashes or critical periods. The research addresses a key question in financial econometrics: **Can LPPL critical time predictions enhance the accuracy of GARCH volatility forecasts?**

## Background

Financial markets often exhibit patterns of accelerated growth followed by sudden crashes, which standard GARCH models may fail to anticipate. The Log-Periodic Power Law (LPPL) model has been proposed for identifying critical times that may precede market crashes by detecting log-periodic oscillations in price trajectories. This project combines these two approaches to explore if the integration can outperform traditional methods.

## Methodology

The project consists of three main components:

1. **Baseline GARCH Analysis**: Implementation of standard GARCH models (baseline GARCH, asymmetric GJR-GARCH, and EGARCH) to establish benchmark volatility forecasts.

2. **LPPL Critical Time Identification**: Application of the LPPL model to identify potential critical times in market data, with parameters:
   - m (power law exponent): Typically between 0.1-0.9
   - ω (log-periodic angular frequency): Typically between 6-13
   - A, B, C1, C2: Additional model parameters
   - tc: Critical time prediction

3. **LPPL-GARCH Integration**: Two approaches were explored:
   - Direct integration: Combining LPPL components with GARCH volatility models
   - Regime-based approach: Using LPPL signals to identify market regimes and applying appropriate GARCH models to each regime

## Data

The analysis uses S&P 500 index data from 2015 to 2023, encompassing several market events including:
- COVID-19 crash (March 2020)
- February 2021 critical time (as identified by LPPL model)
- Other significant market corrections

## Implementation

The project is implemented in Python with a modular structure:
- `config.py`: Configuration parameters
- `data_utils.py`: Data handling and preprocessing
- `garch_models.py`: Implementation of various GARCH specifications
- `lppl_models.py`: Implementation of the LPPL methodology
- `lppl_garch.py`: Integration of LPPL signals with GARCH models
- `evaluation.py`: Performance metrics and evaluation functions
- `visualization.py`: Visualization utilities
- `main.py`: Main execution script orchestrating the analysis

## Key Findings

1. **Model Performance**: 
   - The asymmetric GJR-GARCH model outperformed other GARCH variants across most evaluation metrics
   - High MAPE values (>150%) indicate challenges in accurately forecasting volatility magnitudes
   - Directional accuracy was generally poor across all models (<35%)

2. **LPPL Analysis**:
   - Successfully identified critical times, including February 13, 2021
   - LPPL parameters varied across different time periods (e.g., m=0.141, ω=8.862 for Feb 2021)
   
3. **Regime Analysis**:
   - The performance of GARCH models differs between normal market periods and periods approaching LPPL-identified critical times
   - This confirms that LPPL can identify regime changes relevant to volatility forecasting

## Challenges and Limitations

1. Direct integration of LPPL components into GARCH models presented implementation challenges
2. Small sample sizes in critical periods limited statistical significance
3. High forecast errors across all models reflect the inherent difficulty of volatility prediction

## Conclusions

This research demonstrates that LPPL models can identify market regimes where volatility dynamics differ, suggesting that a regime-switching approach informed by LPPL signals has potential to enhance volatility forecasting. While the direct integration of LPPL components into GARCH models faced technical challenges, the regime-based approach provides a practical framework for leveraging LPPL signals in risk management.

## Next Steps

1. Extend analysis to additional market indices and time periods
2. Explore more sophisticated integration methods
3. Develop a formal regime-switching model guided by LPPL signals
4. Investigate alternative volatility forecasting methods that may better capture pre-crash dynamics

## Usage

To run the analysis:
1. Install dependencies: `pip install numpy pandas matplotlib seaborn yfinance arch scipy statsmodels`
2. Configure parameters in `config.py`
3. Run the main analysis: `python main.py`
4. For the simplified regime analysis: `python run_lppl_garch_analysis.py`

## Results

Results are saved in the `experiments/experiment_results/` directory, with visual outputs in `experiments/plots/`. The primary findings are summarized in `lppl_garch_summary.md`. 