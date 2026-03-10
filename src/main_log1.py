import matplotlib.pyplot as plt
from gbm import simulate_gbm_path

S0 = 100      # initial price
mu = 0.08     # drift
sigma = 0.20  # volatility
T = 1         # 1 year
N = 252       # trading days


time, prices = simulate_gbm_path(S0, mu, sigma, T, N)


plt.figure(figsize=(10,6))
plt.plot(time, prices)
plt.xlabel("Time (years)")
plt.ylabel("Asset Price")
plt.title("Single GBM Price Path")
plt.grid(True)

plt.savefig("./outputs/figures/log_01_single_path.png")
#plt.show()


print("Initial Price:", prices[0])
print("Final Price:", prices[-1])