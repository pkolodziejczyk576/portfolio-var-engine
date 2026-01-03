# Portfolio VaR Engine

A risk management tool focused on Market Risk quantification. It calculates **Value at Risk (VaR)** for investment portfolios using three statistical methodologies.

## Key Features
- **Historical Simulation:** Reconstructs exact portfolio returns based on past market data.
- **Parametric VaR:** Uses Variance-Covariance method assuming normal distribution.
- **Monte Carlo Simulation:** Runs 10,000 simulations to project potential future losses.
- **Visualization:** Generates distribution plots to identify tail risks using `matplotlib`.

## Technologies
- **Python 3.x**
- **Libraries:** `pandas`, `numpy`, `scipy`, `yfinance`, `matplotlib`

## How it works
The script fetches historical data for selected tickers (e.g., SPY, QQQ) via Yahoo Finance API and computes the potential loss for a given confidence level (e.g., 95%) and time horizon.
