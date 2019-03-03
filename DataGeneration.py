import numpy as np

def stock_prices(mu=0.05, sigma=0.3, S0=5, T=1, T_past=0, num_samples=50, num_steps=100):

    # Initialization:
    S = np.zeros([num_samples, num_steps + 1])
    S[:, 0] = S0
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    num_past_steps = int(T_past // dt)
    Spast = np.zeros([num_samples, num_past_steps + 1])
    Spast[:, 0] = S0

    # Generating normal RVs:
    normal_sample = np.random.normal(0, 1, [num_samples, num_steps])
    normal_sample_past = np.random.normal(0, 1, [num_samples, num_past_steps])

    # Initializing drift and volatility:
    if (type(mu) is list) or (type(mu) is tuple):
        # case of uncertain drift:
        Mu = np.random.uniform(mu[0], mu[1], num_samples)
    else:
        # case of fixed drift:
        Mu = mu
    if (type(sigma) is list) or (type(sigma) is tuple):
        # case of uncertain volatility:
        Sigma = np.random.uniform(sigma[0], sigma[1], num_samples)
    else:
        # case of fixed volatility:
        Sigma = sigma

    # Euler-Maruyama method for generating future stock prices
    for k in range(1, num_steps + 1):
        S[:, k] = S[:, k - 1] + Mu * S[:, k - 1] * dt + Sigma * S[:, k - 1] * normal_sample[:, k - 1] * sqrt_dt

    # Euler-Maruyama method for generating past stock prices
    for k in range(1, num_past_steps + 1):
        Spast[:, k] = Spast[:, k - 1] + Mu * Spast[:, k - 1] * dt + Sigma * Spast[:, k - 1] \
                      * normal_sample_past[:,k - 1] * sqrt_dt
    Spast = Spast[:, 1:]
    Spast = np.fliplr(Spast)

    # Conacetenating past and future stock prices:
    S = np.hstack([Spast, S])

    # Generating time axis corresponding to the stock prices
    timepoints = np.hstack([np.arange(-num_past_steps, 0), np.array([0]), np.arange(1, num_steps + 1)]) * dt

    return S, timepoints


def stock_prices_generator(mu=0.05, sigma=0.3, S0=5, T=1, T_past=0, num_samples=50, num_steps=100, maxgen=np.inf,
                           randomseed=None, modelinput=False):
    # Init randomseed:
    if randomseed is not None:
        np.random.seed(randomseed)

    # Generate (fixed) timepoint input for the NN:
    _, timepts = stock_prices(mu, sigma, S0, T, T_past, num_samples, num_steps)
    timepoints = np.ones((num_samples, 1)).dot(timepts.reshape(1, -1))

    # Generating batch:
    k = 0
    while k < maxgen:
        k = k + 1

        # Generating stock prices:
        S, _ = stock_prices(mu, sigma, S0, T, T_past, num_samples, num_steps)
        if modelinput:
            # If input should be used in NN; we expand timepoints and stockprices and return (yield) both
            input_to_train = [np.expand_dims(S, axis=2), np.expand_dims(timepoints, axis=2)]
            # Target is dummy (zeros)
            dummy_target = np.zeros([S.shape[0], S.shape[1], 3])
            # Yielding input for training and dummy targets
            yield input_to_train, dummy_target
        else:
            # Yielding stockprices only:
            yield S