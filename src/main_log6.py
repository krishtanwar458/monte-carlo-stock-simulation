import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from gbm import simulate_gbm_path

ticker = "IDEA.NS" #NVDA
start_date = "2018-01-01"
split_date = "2024-12-31"
current_date = "2026-03-10"       
forecast_end_date = "2026-12-31"  

data = yf.download(ticker, start=start_date, end=forecast_end_date, auto_adjust=True)

prices = data["Close"].squeeze()
prices.index = pd.to_datetime(prices.index)

train = prices.loc[:split_date]
actual = prices.loc["2025-01-01":current_date]

log_returns = np.log(train / train.shift(1)).dropna()

r_bar = log_returns.mean()
s_r = log_returns.std()

mu = float(252 * r_bar)
sigma = float(np.sqrt(252) * s_r)

S0 = float(train.iloc[-1])

print("Estimated drift (mu):", mu)
print("Estimated volatility (sigma):", sigma)
print("Initial simulation price:", S0)

forecast_dates = pd.bdate_range(start="2025-01-01", end=forecast_end_date)
N = len(forecast_dates)
T = N / 252

num_paths = 10000
all_paths = np.zeros((num_paths, N + 1))

for i in range(num_paths):
    t, prices_sim = simulate_gbm_path(S0, mu, sigma, T, N)
    all_paths[i] = prices_sim

lower = np.percentile(all_paths, 5, axis=0)
median = np.percentile(all_paths, 50, axis=0)
upper = np.percentile(all_paths, 95, axis=0)

terminal_prices = all_paths[:, -1]

prob_above_initial = np.mean(terminal_prices > S0)
prob_above_120pct = np.mean(terminal_prices > 1.2 * S0)
prob_below_80pct = np.mean(terminal_prices < 0.8 * S0)

p5 = np.percentile(terminal_prices, 5)
p50 = np.percentile(terminal_prices, 50)
p95 = np.percentile(terminal_prices, 95)

theoretical_expectation = S0 * np.exp(mu * T)

print("Expected terminal price (Monte Carlo):", terminal_prices.mean())
print("Expected terminal price (Theory):", theoretical_expectation)
print("5th percentile terminal price:", p5)
print("Median terminal price:", p50)
print("95th percentile terminal price:", p95)
print("Probability terminal price > initial price:", prob_above_initial)
print("Probability terminal price > 1.2*S0:", prob_above_120pct)
print("Probability terminal price < 0.8*S0:", prob_below_80pct)

sim_time = np.arange(N + 1) / 252

actual_dates = actual.index
actual_time = (actual_dates - pd.Timestamp("2025-01-01")).days / 365.25
current_time = (pd.Timestamp(current_date) - pd.Timestamp("2025-01-01")).days / 365.25

plt.figure(figsize=(12, 7))

plt.fill_between(sim_time, lower, upper, alpha=0.3, label="Monte Carlo 5–95% band")
plt.plot(sim_time, median, linewidth=2, label="Median simulation")
plt.plot(actual_time, actual.values, color="black", linewidth=2, label="Actual price")
plt.axvline(x=current_time, color="red", linestyle="--", linewidth=2, label=f"Current date ({current_date})")

plt.title(f"Monte Carlo Forecast vs Real Price ({ticker})")
plt.xlabel(f"Time (years from {split_date})")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

os.makedirs("./outputs/figures", exist_ok=True)
plt.savefig(f"./outputs/figures/log_06_{ticker}_forecast_extended.png", dpi=300, bbox_inches="tight")
plt.show()