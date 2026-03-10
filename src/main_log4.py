import time
import numpy as np
import matplotlib.pyplot as plt
from gbm import simulate_gbm_path
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

S0 = 100
mu = 0.08
sigma = 0.20
T = 1
N = 1000

num_paths = 5000         
plot_paths = 10        

all_paths = np.zeros((num_paths, N + 1))

start_time = time.time()

for i in range(num_paths):
    time_grid, prices = simulate_gbm_path(S0, mu, sigma, T, N)
    all_paths[i, :] = prices

end_time = time.time()
runtime = end_time - start_time

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

for i in range(plot_paths):
    y = np.full_like(time_grid, i)
    ax.plot(time_grid, y, all_paths[i, :], alpha=0.7)

ax.set_xlabel("Time (years)")
ax.set_ylabel("Simulation Index")
ax.set_zlabel("Asset Price")
ax.set_title("3D Monte Carlo Surface of GBM Price Paths")

plt.savefig("./outputs/figures/log_04_3d_surface.png")
plt.show()


terminal_prices = all_paths[:, -1]

print("Number of simulations:", num_paths)
print("Number of plotted paths:", plot_paths)
print("Time steps per path:", N)
print("Total simulated points:", num_paths * (N + 1))
print("Runtime (seconds):", round(runtime, 2))
print("Mean terminal price:", terminal_prices.mean())
print("Min terminal price:", terminal_prices.min())
print("Max terminal price:", terminal_prices.max())