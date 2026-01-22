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
        
        #Make sure the data is in the right format
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

        #make sure the data is ordered correctly
        self.data = self.data.sort_index()
        if len(self.tickers) > 1:
            self.data = self.data[self.tickers]

        # Calculate Log Returns

        self.log_returns = np.log(self.data / self.data.shift(1)).dropna()
                
        # Calculate Simple Returns

        self.simple_returns = self.data.pct_change().dropna()
        
        # Mean and Covariance of LOG returns (for parametric modeling)
        self.mean_log_returns = self.log_returns.mean()
        self.cov_matrix = self.log_returns.cov()

    def calculate_portfolio_performance(self):
        # This calculates expected Log-Return performance based on linear weights
        # Note: This is an approximation for portfolio log-returns
        port_return = np.sum(self.mean_log_returns * self.weights)
        port_volatility = np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))
        return port_return, port_volatility

    def historical_var(self, confidence_level):
        """
        Calculates VaR using the Historical Method with exact simple returns.
        We reconstruct exactly what the portfolio N-day return would have been.
        """

        portfolio_simple_returns = self.simple_returns.dot(self.weights)
        
        # Calculate actual N-day cumulative returns using a rolling window
        # Geometric compounding: (1+r1)*(1+r2)*... - 1
        # We use apply with np.prod for exact compounding
        rolling_cumulative_returns = (
            (1 + portfolio_simple_returns)
            .rolling(window=self.days)
            .apply(np.prod, raw=True) 
            - 1
        ).dropna()
        
        var_horizon = rolling_cumulative_returns.quantile(1 - confidence_level)
        
        
        var_pct_loss = -var_horizon
        
        return self.investment * var_pct_loss, var_pct_loss, rolling_cumulative_returns

    def parametric_var(self, confidence_level):
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
        
        return self.investment * var_pct_loss, var_pct_loss, sigma  

    def monte_carlo_var(self, simulations, confidence_level):
        """
        Monte Carlo VaR using Historical Bootstrapping.
        We resample actual historical portfolio returns to simulate future paths.
        This avoids Normal distribution assumptions (capture fat tails, skewness),
        """
        # Calculate historical daily log returns for the portfolio
        # We assume constant weights for the history
        portfolio_log_returns = self.log_returns.dot(self.weights)

        # Resample returns with replacement
        # Generate a matrix of random indices: (simulations, days)
        # We sample from the actual observed returns
        simulated_returns_matrix = np.random.choice(portfolio_log_returns, size=(simulations, self.days), replace=True)

        # Calculate cumulative return for each simulation path (horizon)
        # Sum of log returns = Cumulative log return
        simulated_horizon_log_returns = np.sum(simulated_returns_matrix, axis=1)

        # Convert Log Returns -> Simple Returns
        simulated_horizon_simple_returns = np.exp(simulated_horizon_log_returns) - 1

        # Determine VaR
        var_pct_loss = -np.percentile(simulated_horizon_simple_returns, (1 - confidence_level) * 100)
        
        # Calculate dollar value
        var_dollar_loss = self.investment * var_pct_loss

        return var_dollar_loss, var_pct_loss, simulated_horizon_simple_returns

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

        hist_var, hist_pct, hist_data = engine.historical_var(confidence_level=confidence)
        param_var, param_pct, param_sigma = engine.parametric_var(confidence_level=confidence)
        mc_var, mc_pct, mc_sims = engine.monte_carlo_var(simulations=simulations, confidence_level=confidence)
        kurtosis = hist_data.kurtosis()
        # Output
        print(f"\n--- Value at Risk (VaR) Report ---")
        print(f"Time Horizon: {target_days} Days")
        print(f"Confidence:   {confidence*100}%")
        print(f"Portfolio:    {tickers}")
        
        print(f"\n1. Historical VaR:   ${hist_var:,.2f} ({hist_pct*100:.2f}%)")
        print(f"   (Calculated using actual {target_days}-day rolling returns)")
        
        print(f"2. Parametric VaR:   ${param_var:,.2f} ({param_pct*100:.2f}%)")
        print(f"   Portfolio Volatility: {param_sigma * np.sqrt(target_days)}")
        print(f"   (Calculated using Normal Distribution assumption)")
        print(f"   Kurtosis ({target_days}-day rolling): {kurtosis}")

        print(f"3. Monte Carlo VaR:  ${mc_var:,.2f} ({mc_pct*100:.2f}%)")
        print(f"   (Historical Bootstrapping with {simulations} simulations)") 

        # 3 Separate Subplots
        fig, axes = plt.subplots(3, 1, figsize=(8, 8))
        
        # Historical Plots
        axes[0].hist(hist_data * 100, bins=50, alpha=0.6, color='purple', edgecolor='black', label='Historical Returns')
        axes[0].axvline(-hist_pct * 100, color='red', linestyle='--', linewidth=2, label=f'VaR (-{hist_pct*100:.2f}%)')
        axes[0].set_title(f"Historical Simulation: {target_days}-Day Actual Returns")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Parametric Plots
        # Generate Normal Distribution curve
        mu_p, sigma_p = engine.calculate_portfolio_performance()
        sigma_horizon = sigma_p * np.sqrt(target_days)
        # We plot a range of +/- 4 sigmas
        x_axis = np.linspace(-4*sigma_horizon, 4*sigma_horizon, 1000)
        # Use a secondary y-axis for the PDF to make it visible
        axes[1].plot(x_axis * 100, stats.norm.pdf(x_axis, 0, sigma_horizon), color='orange', linewidth=3, label='Normal Distribution')
        axes[1].axvline(-param_pct * 100, color='red', linestyle='--', linewidth=2, label=f'VaR (-{param_pct*100:.2f}%)')
        axes[1].set_title(f"Parametric Method: Normal Distribution Assumption")
        axes[1].set_ylabel("Probability Density")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Monte Carlo Plots
        axes[2].hist(mc_sims * 100, bins=50, alpha=0.6, color='skyblue', edgecolor='black', label='Simulated Returns')
        axes[2].axvline(-mc_pct * 100, color='red', linestyle='--', linewidth=2, label=f'VaR (-{mc_pct*100:.2f}%)')
        axes[2].set_title(f"Monte Carlo (Bootstrap): {simulations} Simulated Paths")
        axes[2].set_xlabel("Return (%)")
        axes[2].set_ylabel("Frequency")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")
