import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as stats
import matplotlib.pyplot as plt
import datetime

class VaREngine:
    def __init__(self, tickers, weights, investment, start_date, end_date, days):
        self.days = days
        self.tickers = tickers
        self.weights = np.array(weights)
        self.investment = investment
        
        # Download historical data
        df = yf.download(self.tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            try:
                self.data = df['Close']
            except KeyError:
                self.data = df.xs('Close', axis=1, level=0)
        else:
            self.data = df['Close'] if 'Close' in df.columns else df

        if isinstance(self.data, pd.Series):
            self.data = self.data.to_frame()
            self.data.columns = self.tickers

        self.data = self.data.dropna()


        if len(self.tickers) > 1:
            self.data = self.data[self.tickers]

        # 1. Log Returns for Parametric/Monte Carlo 

        self.log_returns = np.log(self.data / self.data.shift(1)).dropna()
        
        # 2. Simple Returns for Historical Simulation 

        self.simple_returns = self.data.pct_change().dropna()
        
        # Mean and Covariance of log returns (for parametric modeling)
        self.mean_log_returns = self.log_returns.mean()
        self.cov_matrix = self.log_returns.cov()

    def calculate_portfolio_performance(self):
        # This calculates expected Log-Return performance based on linear weights
        port_return = np.sum(self.mean_log_returns * self.weights)
        port_volatility = np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))
        return port_return, port_volatility

    def historical_var(self, confidence_level=0.95):
        """
        Calculates VaR using the Historical Method with exact simple returns.
        We reconstruct exactly what the portfolio N-day return would have been.
        """

        portfolio_simple_returns = self.simple_returns.dot(self.weights)
        
        # Calculate actual N-day cumulative returns using a rolling window
        rolling_cumulative_returns = (
            (1 + portfolio_simple_returns)
            .rolling(window=self.days)
            .apply(np.prod, raw=True) 
            - 1
        ).dropna()
        
        var_horizon = rolling_cumulative_returns.quantile(1 - confidence_level)
        
        
        var_pct_loss = -var_horizon
        
        return self.investment * var_pct_loss, var_pct_loss

    def parametric_var(self, confidence_level=0.95):
        """
        Parametric (Variance-Covariance) VaR.
        Assumes Log-Normal distribution of prices (Normal distribution of returns).
        """
        mu, sigma = self.calculate_portfolio_performance()
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Scale to time horizon
        # Conservative approach: mu=0
        mu_scaled = 0 
        sigma_scaled = sigma * np.sqrt(self.days)
        
        # VaR in Log Space
        var_log = mu_scaled + z_score * sigma_scaled
        
        # Convert to Simple Percentage Loss
        var_pct_loss = 1 - np.exp(var_log)
        
        return self.investment * var_pct_loss, var_pct_loss

    def monte_carlo_var(self, simulations, confidence_level):
        #Get Portfolio stats
        mu, sigma = self.calculate_portfolio_performance()
        
        #Generate random returns for the portfolio

        z_scores = np.random.normal(0, 1, simulations)
        
        # Calculate return for the target_days horizon
        simulated_returns = sigma * np.sqrt(self.days) * z_scores
        
        # Determine VaR
        sorted_sims = np.sort(simulated_returns)
        
        cutoff_index = int((1 - confidence_level) * simulations)
        var_pct_loss = -sorted_sims[cutoff_index]
        
        return self.investment * var_pct_loss, var_pct_loss, simulated_returns

# --- EXECUTION  ---

if __name__ == "__main__":
    tickers = ["SPY", "QQQ", "GLD"]
    weights = [0.5, 0.3, 0.2] 
    investment_value = 6000
    confidence = 0.95
    target_days = 5 
    simulations = 10000  

   #get data
    years = 2
    start_date = datetime.datetime.now() - datetime.timedelta(days=365*years)
    end_date = datetime.datetime.now()

    try:
        engine = VaREngine(tickers, weights, investment_value, start_date, end_date, days=target_days)

        hist_var, hist_pct = engine.historical_var(confidence)
        param_var, param_pct = engine.parametric_var(confidence)
        
        mc_var, mc_pct, mc_sims = engine.monte_carlo_var(simulations=simulations, confidence_level=confidence)

        # Output
        print(f"\n--- Value at Risk (VaR) Report ---")
        print(f"Time Horizon: {target_days} Days")
        print(f"Confidence:   {confidence*100}%")
        print(f"Portfolio:    {tickers}")
        
        print(f"\n1. Historical VaR:   ${hist_var:,.2f} ({hist_pct*100:.2f}%)")
        print(f"   (Calculated using actual {target_days}-day rolling returns)")
        
        print(f"2. Parametric VaR:   ${param_var:,.2f} ({param_pct*100:.2f}%)")
        print(f"   (Calculated using Normal Distribution assumption)")
        
        print(f"3. Monte Carlo VaR:  ${mc_var:,.2f} ({mc_pct*100:.2f}%)")
        print(f"   (Calculated using {simulations} simulations)") 

        # Plot
        plt.figure(figsize=(10, 6))
        plt.hist(mc_sims * 100, bins=50, alpha=0.6, color='skyblue', edgecolor='black', label='Simulated Returns')
        plt.axvline(-mc_pct * 100, color='red', linestyle='--', linewidth=2, label=f'VaR Limit (-{mc_pct*100:.2f}%)')
        plt.title(f"Monte Carlo: {target_days}-Day Potential Return Distribution")
        plt.xlabel("Return (%)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except Exception as e:

        print(f"An error occurred during execution: {e}")

