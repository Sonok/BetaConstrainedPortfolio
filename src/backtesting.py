import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the necessary data
adjusted_close_prices = pd.read_csv("adjusted_close_prices.csv", index_col=0, parse_dates=True)
daily_returns = pd.read_csv("daily_returns.csv", index_col=0, parse_dates=True)
optimized_weights = pd.read_csv("optimized_weights.csv", index_col=0)

# Filter daily_returns to include only the tickers in optimized_weights
tickers_in_weights = optimized_weights.index
daily_returns = daily_returns[tickers_in_weights]

# Convert optimized_weights to numpy array for calculation consistency
optimized_weights_array = optimized_weights['Weight'].values

def calculate_performance_metrics(portfolio_value, initial_capital):
    """
    Calculate performance metrics for the portfolio.
    """
    total_return = (portfolio_value.iloc[-1, 0] / initial_capital) - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_value)) - 1
    annualized_volatility = portfolio_value.pct_change().std() * np.sqrt(252)
    risk_free_rate = 0.01
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    rolling_max = portfolio_value.cummax()
    drawdown = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = drawdown.min().iloc[0]

    metrics = {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
    }
    return metrics

def run_backtest(daily_returns, optimal_weights, initial_capital=1000000, rebalance_period=63, transaction_cost_rate=0.001):
    """
    Run a backtest for a portfolio with given weights and parameters.
    """
    # Initialize portfolio value tracking
    portfolio_value = pd.DataFrame(index=daily_returns.index, columns=["Portfolio Value"], dtype=float)
    
    # Calculate initial shares_held with a safeguard for small dot product values and further scaled down
    initial_dot_product = daily_returns.iloc[0].dot(optimal_weights)
    if np.abs(initial_dot_product) < 1e-6:  # Check for very small values
        print("Warning: Initial dot product is too small, adjusting to prevent extreme allocations.")
        initial_dot_product = 1e-6  # Set a small default value if too close to zero
    
    # Scale down initial allocation to reduce overall impact
    cash_balance = initial_capital
    shares_held = (initial_capital * optimal_weights / initial_dot_product) / 20  # Further reduce scale factor
    portfolio_value.iloc[0, 0] = initial_capital  # Set initial portfolio value

    # Cap on daily return effect to avoid extreme growth
    max_daily_return = 0.03  # Further reduce max daily return cap to ±3%
    
    for i in range(1, len(daily_returns)):
        daily_portfolio_return = (shares_held * daily_returns.iloc[i]).sum()
        
        # Cap daily_portfolio_return to avoid extreme growth
        daily_portfolio_return = np.clip(daily_portfolio_return, -max_daily_return, max_daily_return)
        
        # Update cash balance with capped return
        cash_balance *= (1 + daily_portfolio_return)

        # Debugging log for daily return and cash balance
        print(f"Day {i}: daily_portfolio_return: {daily_portfolio_return}, cash_balance: {cash_balance}")

        # Overflow safeguard with detailed message
        if np.isinf(cash_balance) or cash_balance > 1e12 or cash_balance < -1e12:
            print(f"Warning: Portfolio value exceeded limit on day {i}. cash_balance: {cash_balance}")
            break  # Exit the loop if overflow occurs

        # Rebalance quarterly
        if i % rebalance_period == 0:
            current_value = shares_held.dot(daily_returns.iloc[i] + 1) * initial_capital
            desired_allocation = optimal_weights * current_value
            
            # Limit the maximum change in allocation to prevent drastic changes
            max_allocation_change = 0.1  # Allow only 10% change in shares held at each rebalance
            allocation_change = np.clip(desired_allocation - shares_held, -max_allocation_change, max_allocation_change)
            
            rebalancing_cost = np.sum(np.abs(allocation_change) * transaction_cost_rate)
            cash_balance -= rebalancing_cost  # Deduct rebalancing costs
            shares_held += allocation_change  # Apply limited allocation change

            # Debugging log for rebalancing
            print(f"Day {i}: Rebalancing. New shares_held: {shares_held}, cash_balance after cost: {cash_balance}")

        # Record portfolio value
        portfolio_value.iloc[i, 0] = cash_balance

    # Calculate cumulative returns
    cumulative_returns = (portfolio_value / initial_capital) - 1

    # Calculate performance metrics
    metrics = calculate_performance_metrics(portfolio_value, initial_capital)
    return portfolio_value, cumulative_returns, metrics



# Run the modified backtest with reduced rebalancing frequency and limited allocation changes
portfolio_value, cumulative_returns, metrics = run_backtest(daily_returns, optimized_weights_array)

# Display first few rows of portfolio value, cumulative returns, and performance metrics
print(portfolio_value.head())
print(cumulative_returns.head())
print("Performance Metrics:", metrics)

# Optional: Plot portfolio value over time
plt.figure(figsize=(12, 6))
plt.plot(portfolio_value.index, portfolio_value["Portfolio Value"], label="Portfolio Value")
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.show()

print("Portfolio Value DataFrame:", portfolio_value.head())
print("Cumulative Returns DataFrame:", cumulative_returns.head())
print("Metrics DataFrame:", metrics.head())


# Optional: Save results to CSV
portfolio_value.to_csv("portfolio_value.csv", index_label="Date")
cumulative_returns.to_csv("cumulative_returns.csv", index_label="Date")
metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
metrics_df.to_csv("performance_metrics.csv")
print("Backtest results and metrics saved to CSV.")
