# Value at Risk (VaR) Engine – Python

Python implementation of a **Value at Risk (VaR) engine** comparing three approaches:

- **Historical Simulation**
- **Parametric (Variance–Covariance)**
- **Monte Carlo (Historical Bootstrapping)**

The project focuses on **practical differences between models**
---

## Overview

The engine downloads market data, constructs a multi-asset portfolio, and estimates VaR over an arbitrary time horizon.  
Beyond computing VaR, the project highlights **model assumptions, tail behavior, and empirical limitations**.

---

## Methods

### Historical VaR
- Uses actual historical returns
- Reconstructs exact **N-day portfolio returns** via geometric compounding
- No distributional assumptions

**Pros:** Model-free, captures fat tails  
**Cons:** Backward-looking

---

### Parametric VaR
- Assumes **normally distributed log-returns**
- Portfolio volatility computed from the covariance matrix
- Time scaling via √T, mean return set to zero

**Pros:** Fast and tractable  
**Cons:** Underestimates tail risk under high kurtosis

---

### Monte Carlo VaR (Bootstrapping)
- Resamples historical portfolio **log-returns with replacement**
- Simulates future return paths without assuming normality

**Pros:** Preserves empirical skewness and fat tails  
**Cons:** Assumes i.i.d. returns

---

## Key Assumptions

- Constant portfolio weights  
- No transaction costs or liquidity effects  
- Fixed confidence level and time horizon  
- Educational, non-production risk model  

---
You can adjust assets, weights, confidence level, time horizon, and number of Monte Carlo simulations directly in the script.
