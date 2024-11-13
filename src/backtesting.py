# src/backtesting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Data
adj_close = pd.read_csv("adjusted_close_prices.csv", index_col=0, parse_dates=True)
daily_returns = pd.read_csv("daily_returns.csv", index_col=0, parse_dates=True)

# Load optimized weights and align tickers
optimal_weights_df = pd.read_csv("optimized_weights.csv", index_col=0)  # Load weights as DataFrame
optimal_weights = optimal_weights_df.iloc[:, 0].values  # Convert weights to numpy array

# Filter `daily_returns` to include only tickers in `optimal_weights`
tickers_in_weights = optimal_weights_df.index
daily_returns = daily_returns[tickers_in_weights]  # Align columns with weights

def run_backtest(daily_returns, optimal_weights, initial_capital=1000000, rebalance_period=21, transaction_cost_rate=0.001):
    """
    Run a backtest for a portfolio with given weights and parameters.
    """
    # Initialize portfolio value tracking
    portfolio_value = pd.DataFrame(index=daily_returns.index, columns=["Portfolio Value"], dtype=float)
    cash_balance = initial_capital
    
    # Calculate initial shares_held with an alternative approach for stability
    try:
        initial_divisor = daily_returns.iloc[0].dot(optimal_weights)
        if initial_divisor == 0:
            print("Warning: Initial divisor in shares calculation is zero. Adjusting to 1.")
            initial_divisor = 1
        shares_held = (initial_capital * optimal_weights) / initial_divisor
    except Exception as e:
        print(f"Error in shares_held calculation: {e}")
        shares_held = np.zeros_like(optimal_weights)  # Set to zero to avoid crashes

    portfolio_value.iloc[0, 0] = initial_capital  # Set initial portfolio value

    # Track daily portfolio value and apply rebalancing strategy
    for i in range(1, len(daily_returns)):
        # Calculate daily portfolio return
        daily_portfolio_return = (shares_held * daily_returns.iloc[i]).sum()
        
        # Hard cap on daily portfolio return growth (e.g., 2% maximum)
        daily_portfolio_return = min(daily_portfolio_return, 0.02)  # Cap at 2%
        cash_balance += cash_balance * daily_portfolio_return  # Incremental addition

        # Debugging logs for daily changes
        print(f"Day {i}: daily_portfolio_return: {daily_portfolio_return}, cash_balance: {cash_balance}")

        # Overflow safeguard
        if np.isinf(cash_balance) or cash_balance > 1e9:  # Check for overflow
            print(f"Warning: Portfolio value exceeded limit on day {i}. cash_balance: {cash_balance}")
            break  # Exit the loop if overflow occurs

        # Rebalance monthly
        if i % rebalance_period == 0:
            current_value = shares_held.dot(daily_returns.iloc[i] + 1) * initial_capital
            desired_allocation = optimal_weights * current_value
            rebalancing_cost = np.sum(np.abs(desired_allocation - shares_held) * transaction_cost_rate)
            cash_balance -= rebalancing_cost  # Deduct rebalancing costs
            shares_held = desired_allocation  # Rebalance holdings

            # Debugging log for rebalancing
            print(f"Day {i}: Rebalancing. New shares_held: {shares_held}, cash_balance after cost: {cash_balance}")

        # Record portfolio value
        portfolio_value.iloc[i, 0] = cash_balance

    # Calculate cumulative returns
    cumulative_returns = (portfolio_value / initial_capital) - 1

    # Calculate performance metrics
    metrics = calculate_performance_metrics(portfolio_value, initial_capital)
    return portfolio_value, cumulative_returns, metrics

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

def plot_performance(portfolio_value, cumulative_returns):
    """Plot portfolio value and cumulative returns over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value.index, portfolio_value["Portfolio Value"], label="Portfolio Value")
    plt.plot(cumulative_returns.index, cumulative_returns["Portfolio Value"], label="Cumulative Return", linestyle="--")
    plt.title("Portfolio Value and Cumulative Returns Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value / Cumulative Return")
    plt.legend()
    plt.show()

def save_results(portfolio_value, cumulative_returns, metrics):
    """Save backtest results and performance metrics to CSV."""
    portfolio_value.to_csv("portfolio_value.csv", index_label="Date")
    cumulative_returns.to_csv("cumulative_returns.csv", index_label="Date")
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    metrics_df.to_csv("performance_metrics.csv")
    print("Backtest results and metrics saved to CSV.")

# Main section
if __name__ == "__main__":
    # Run the backtest
    portfolio_value, cumulative_returns, metrics = run_backtest(daily_returns, optimal_weights)

    # Plot the performance
    plot_performance(portfolio_value, cumulative_returns)

    # Save the results
    save_results(portfolio_value, cumulative_returns, metrics)

    # Print metrics for quick reference
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2%}")
