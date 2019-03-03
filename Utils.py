import numpy as np
from DataGeneration import stock_prices

def EuroOptionPrice_MonteCarlo(r=0.05, sigma=0.3, S0=5, T=1, K=5, Type='call', num_samples=50, num_steps=100):
    S, _=stock_prices(mu=r, sigma=sigma, S0=S0, T=T, T_past=0, num_samples=num_samples, num_steps=num_steps)
    if Type=='call':
        return np.maximum(S[:,-1]-K,0).mean()
    elif Type=='put':
        return np.maximum(K-S[:,-1],0).mean()
    else:
        raise ValueError('Invalid option Type: vslid options are "call" or "put".')