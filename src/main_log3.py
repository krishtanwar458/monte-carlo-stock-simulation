import matplotlib.pyplot as plt
import numpy as np
from gbm import simulate_gbm_path


S0 = 100
mu = 0.08
sigma = 0.20
T = 1
N = 252


num_paths = 10000

terminal_prices = []

for i in range(num_paths):

    time, prices = simulate_gbm_path(S0, mu, sigma, T, N)

    terminal_prices.append(prices[-1])

terminal_prices = np.array(terminal_prices)


plt.figure(figsize=(10,6))

plt.hist(terminal_prices, bins=80)

mu_fit = terminal_prices.mean()
sigma_fit = terminal_prices.std()

x = np.linspace(min(terminal_prices), max(terminal_prices), 500)

pdf = (1/(sigma_fit * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu_fit)/sigma_fit)**2)

plt.plot(x, pdf * len(terminal_prices) * (max(terminal_prices)-min(terminal_prices))/80,
         color="red", linewidth=2, label="Normal Approximation")

plt.legend()

plt.xlabel("Terminal Asset Price")
plt.ylabel("Frequency")
plt.title("Distribution of Terminal Prices (Monte Carlo GBM)")
plt.grid(True)

plt.savefig("./outputs/figures/log_03_terminal_distribution.png")
# plt.show()

print("Number of simulations:", num_paths)
print("Mean terminal price:", terminal_prices.mean())
print("Min terminal price:", terminal_prices.min())
print("Max terminal price:", terminal_prices.max())