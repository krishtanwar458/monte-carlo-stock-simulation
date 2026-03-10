import numpy as np
import matplotlib.pyplot as plt
from gbm import simulate_gbm_path

S0 = 100
mu = 0.08
sigma = 0.20
T = 1
N = 252

num_paths = 10000

all_paths = np.zeros((num_paths, N + 1))

for i in range(num_paths):
    time_grid, prices = simulate_gbm_path(S0, mu, sigma, T, N)
    all_paths[i, :] = prices

terminal_prices = all_paths[:, -1]

prob_above_initial = np.mean(terminal_prices > S0)
prob_above_120 = np.mean(terminal_prices > 120)
prob_below_80 = np.mean(terminal_prices < 80)

expected_terminal_price = np.mean(terminal_prices)

p5 = np.percentile(terminal_prices, 5)
p50 = np.percentile(terminal_prices, 50)
p95 = np.percentile(terminal_prices, 95)

theoretical_expectation = S0 * np.exp(mu * T)

lower_band = np.percentile(all_paths, 5, axis=0)
median_band = np.percentile(all_paths, 50, axis=0)
upper_band = np.percentile(all_paths, 95, axis=0)

plt.figure(figsize=(10, 6))
plt.fill_between(time_grid, lower_band, upper_band, alpha=0.3, label="5th-95th percentile band")
plt.plot(time_grid, median_band, linewidth=2, label="Median path")

plt.xlabel("Time (years)")
plt.ylabel("Asset Price")
plt.title("Monte Carlo Confidence Band for GBM Price Paths")
plt.legend()
plt.grid(True)

plt.savefig("./outputs/figures/log_05_confidence_band.png")
plt.show()

print("Number of simulations:", num_paths)
print("Expected terminal price (Monte Carlo):", expected_terminal_price)
print("Expected terminal price (Theory):", theoretical_expectation)
print("Probability terminal price > initial price:", prob_above_initial)
print("Probability terminal price > 120:", prob_above_120)
print("Probability terminal price < 80:", prob_below_80)
print("5th percentile terminal price:", p5)
print("Median terminal price:", p50)
print("95th percentile terminal price:", p95)