# %%
# Cell 8: Imports (if not already in the environment)
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

# Display settings
pd.set_option('display.max_columns', None)

# %%
# Cell 9: Load Saved Data
# Load previously saved adjusted close prices and daily returns
adj_close = pd.read_csv("adjusted_close_prices.csv", index_col=0, parse_dates=True)
daily_returns = pd.read_csv("daily_returns.csv", index_col=0, parse_dates=True)

# Define lists of long and short tickers (same as provided initially)
long_tickers = [
    "CEG", "LH", "BA", "CARR", "DOW", "PH", "EMR", "JBL", "SWK", "URI", "BSX", "DLTR", "ORCL", "HUBB", "LYB",
    "XYL", "HON", "DD", "ROP", "UNH", "IBM", "GRMN", "CMI", "BKR", "GLW", "SYK", "FTV", "ETN", "CHD", "OTIS",
    "PCAR", "DGX", "AME", "DRI", "APH", "AOS", "HUM", "CLX", "ORLY", "CTAS", "ECL", "TER", "TMUS", "MAS", 
    "TDG", "JNPR", "NSC", "FAST", "PAYX", "ROK", "ITW", "CSCO", "CPRT", "TMO", "OKE", "EXC", "EMN", "PWR", 
    "NEM", "DOV", "VTR", "TXT", "TXN", "PG", "AVY", "DTE", "MGM", "BR", "GD", "ADP", "PPL", "NI", "MLM", 
    "IDXX", "HCA", "SHW", "HWM", "ZTS", "RCL", "GWW", "CDW", "CAH", "HPE", "HD", "HSY", "RTX", "UNP", "MCK", 
    "AES", "FICO", "INTC", "JCI", "ATO", "HAS", "LOW", "ALLE", "WELL", "ISRG", "VRSN", "TRGP", "LMT"
]
short_tickers = [
    "ETSY", "DXCM", "ILMN", "PAYC", "VFC", "ABNB", "APA", "UPS", "EPAM", "CHTR", "MOS", "EXPE", "MPC", 
    "PANW", "VLO", "COR", "BXP", "MRO", "HAL", "MRNA"
]
# Add SPY or another benchmark as separate ticker for benchmarking
market_ticker = "SPY"

all_tickers = long_tickers + short_tickers  # Exclude SPY from all_tickers in optimization

# %%
# Cell 10: Calculate Expected Returns and Covariance Matrix
# Calculate annualized expected returns and covariance matrix from daily returns
expected_returns = daily_returns[all_tickers].mean() * 252  # Annualized
cov_matrix = daily_returns[all_tickers].cov() * 252  # Annualized

# %%
# Cell 11: Calculate Beta for Each Stock Relative to a Market Index
# Assume 'SPY' is included in the data as the market index (or replace with a different index)
market_returns = daily_returns[market_ticker].values.reshape(-1, 1)  # Use SPY as the market benchmark
betas = {}

for ticker in all_tickers:
    model = LinearRegression().fit(market_returns, daily_returns[ticker].values)
    betas[ticker] = model.coef_[0]

# Convert betas to a Series for easy access
betas = pd.Series(betas, index=all_tickers)
print("Calculated Betas:\n", betas.head())

# %%
# Cell 12: Define Optimization Model
# Objective: Maximize the Sharpe ratio
def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.01):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -1 * (portfolio_return - risk_free_rate) / portfolio_volatility

# Constraints for beta neutrality and weight allocations
constraints = [
    {"type": "eq", "fun": lambda w: np.sum(w[:len(long_tickers)]) - 0.5},  # Long positions sum to 0.5
    {"type": "eq", "fun": lambda w: np.sum(w[len(long_tickers):]) + 0.5},  # Short positions sum to -0.5
    {"type": "eq", "fun": lambda w: np.dot(w, betas) - 0}  # Beta neutrality constraint
]

# Bounds for weights: Long weights between 0 and 1; Short weights between -1 and 0
bounds = [(0, 1) if i < len(long_tickers) else (-1, 0) for i in range(len(all_tickers))]

# %%
# Cell 13: Solve for Optimal Weights
initial_weights = np.array([0.5 / len(long_tickers)] * len(long_tickers) +
                           [-0.5 / len(short_tickers)] * len(short_tickers))

# Perform optimization
result = minimize(
    negative_sharpe_ratio,
    initial_weights,
    args=(expected_returns, cov_matrix, 0.01),  # Risk-free rate of 1%
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

# Extract optimized weights
optimal_weights = result.x
print("Optimized Weights:\n", optimal_weights)



# %%
# Cell 14: Evaluate Portfolio Performance
optimized_return = np.dot(optimal_weights, expected_returns)
optimized_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
sharpe_ratio = (optimized_return - 0.01) / optimized_volatility  # Assuming a 1% risk-free rate

print(f"Optimized Portfolio Expected Return: {optimized_return:.2%}")
print(f"Optimized Portfolio Volatility: {optimized_volatility:.2%}")
print(f"Optimized Portfolio Sharpe Ratio: {sharpe_ratio:.2f}")

# %%
# Cell 15: Save Results (Optional)
# Save the optimized weights to a CSV for further analysis
weights_df = pd.DataFrame(optimal_weights, index=all_tickers, columns=["Weight"])
weights_df.to_csv("optimized_weights.csv")
print("Optimized weights saved to CSV.")

# %% Cell 16: 
# Let's inspect the distribution of values in `daily_returns` and `optimized_weights` to identify any potential outliers or unrealistic values.

# Summary statistics for daily_returns to see the range and identify potential outliers
daily_returns_summary = daily_returns.describe()

# Display optimized_weights to check the range and scale of values
optimized_weights_summary = weights_df.describe()

daily_returns_summary, optimized_weights_summary


# %%
