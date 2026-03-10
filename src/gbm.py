import numpy as np


def simulate_gbm_path(S0, mu, sigma, T, N):
    """
    Simulates a single price path using Geometric Brownian Motion.

    Parameters
    ----------
    S0 : float
        Initial asset price
    mu : float
        Drift (expected return)
    sigma : float
        Volatility
    T : float
        Time horizon (years)
    N : int
        Number of time steps

    Returns
    -------
    time : array
        Time grid
    prices : array
        Simulated price path
    """

    dt = T / N

    # time grid
    time = np.linspace(0, T, N + 1)

    # generate random shocks
    Z = np.random.normal(0, 1, N)

    prices = np.zeros(N + 1)
    prices[0] = S0

    for t in range(N):
        prices[t + 1] = prices[t] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t]
        )

    return time, prices