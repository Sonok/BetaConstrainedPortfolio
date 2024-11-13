# %%
# Cell 1: Imports
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Display settings
pd.set_option('display.max_columns', None)
print("Libraries imported successfully.")


# %%
# Cell 2: Define Tickers
# Define lists of long and short tickers
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

all_tickers = long_tickers + short_tickers
print(f"Defined {len(long_tickers)} long tickers and {len(short_tickers)} short tickers.")

# %%
# Cell 3
# Fetch historical data for all tickers from January 1, 2022, through yesterday
try:
    data = yf.download(all_tickers, start="2022-01-01", end="2024-11-08")
    adj_close = data["Adj Close"]
    print("Data downloaded successfully.")
except Exception as e:
    print("Error downloading data:", e)
# %%
# Cell 4: Initial Data Exploration
# Display the first few rows of the adjusted close prices
print("Adjusted Close Prices:\n", adj_close.head())

# Check for any missing values per ticker
missing_data = adj_close.isnull().sum()
print("\nMissing data per ticker:\n", missing_data[missing_data > 0])

# Identify and display the specific dates with missing values for CEG
missing_dates_ceg = adj_close[adj_close["CEG"].isna()]
print("\nDates with missing data for CEG:\n", missing_dates_ceg.index)


# The missing data for `CEG` on the specified dates (e.g., '2022-01-11', '2022-01-12', etc.) may be due to several factors:
#
# 1. **New Listing or Reorganization**:
#    - If `CEG` was newly listed or went through a corporate reorganization (e.g., a spin-off, merger, or acquisition), 
#      there may be a gap in the data for the dates immediately following the listing or corporate event.
#
# 2. **Trading Halt or Suspension**:
#    - The stock exchange might have temporarily halted or suspended trading for `CEG` due to unusual activity,
#      regulatory issues, or major announcements, leading to missing data on specific days.
#
# 3. **Data Source Inconsistencies**:
#    - Sometimes, data providers like Yahoo Finance may have incomplete data for certain stocks or dates, which can 
#      lead to gaps in the data, especially for less liquid stocks or stocks with unusual trading patterns.
#
# 4. **Market Holidays**:
#    - Although rare, certain market-specific holidays or partial trading days might cause data gaps. However, this
#      typically affects all tickers rather than just one.
#
# 5. **Technical Errors**:
#    - There may have been an error in the data retrieval process for `CEG` specifically on these dates, either 
#      from the data provider's side or during the data download.
#
# To handle these gaps, you can use forward/backward filling methods to maintain a continuous dataset, or exclude
# the affected dates or ticker if the missing data impacts your analysis.

# %%
# Cell 5: Handle Missing Data
# Option 1: Use ffill to forward fill and then backward fill any remaining missing values
# Here's documenation of the method: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ffill.html
adj_close_filled = adj_close.fillna(method="ffill").fillna(method="bfill")

# Display the cleaned data
print("Data after handling missing values:\n", adj_close_filled.head())
# %%
# Cell 6: Calculate Daily Returns
# Calculate daily returns for each ticker
daily_returns = adj_close.pct_change().dropna()
print("Daily Returns Calculated:\n", daily_returns.head())

# %%
# Cell 6: Visualize Returns
# Plot cumulative returns for a subset of long and short tickers for visualization
sample_tickers = long_tickers[:3] + short_tickers[:3]  # Select a few tickers for visualization
cumulative_returns = (1 + daily_returns[sample_tickers]).cumprod()

plt.figure(figsize=(10, 6))
for ticker in sample_tickers:
    plt.plot(cumulative_returns.index, cumulative_returns[ticker], label=ticker)
plt.title("Cumulative Returns of Selected Tickers")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.show()

# %%
# Cell 7: Save Data (Optional)
# Save adjusted close prices and daily returns to CSV files for later analysis
adj_close.to_csv("adjusted_close_prices.csv")
daily_returns.to_csv("daily_returns.csv")
print("Data saved to CSV")
