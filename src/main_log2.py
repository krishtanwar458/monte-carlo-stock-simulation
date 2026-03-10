import matplotlib.pyplot as plt
import numpy as np
from gbm import simulate_gbm_path

S0 = 100
mu = 0.08
sigma = 0.20
T = 1
N = 252

num_paths = 1000  # number of simulated paths

terminal_prices = []


plt.figure(figsize=(10,6))


for i in range(num_paths):

    time, prices = simulate_gbm_path(S0, mu, sigma, T, N)

    plt.plot(time, prices, alpha=0.5) 

    terminal_prices.append(prices[-1])


plt.xlabel("Time (years)")
plt.ylabel("Asset Price")
plt.title("Monte Carlo Simulation of GBM Price Paths")
plt.grid(True)


plt.savefig("./outputs/figures/log_02_fan_chart.png")
#plt.show()


terminal_prices = np.array(terminal_prices)


print("Number of simulations:", num_paths)
print("Mean terminal price:", terminal_prices.mean())
print("Min terminal price:", terminal_prices.min())
print("Max terminal price:", terminal_prices.max())