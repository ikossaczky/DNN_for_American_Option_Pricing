import numpy as np


def stock_prices(mu=0.05, sigma=0.3, S0=5, T=1, T_past=0, num_samples=50, num_steps=100):
    """
    Generate stock price paths using Euler-Maruyama method (Geometric Brownian Motion).
    
    Parameters:
    -----------
    mu : float, list, or tuple
        Drift parameter (interest rate). If list/tuple, random uniform sampling.
    sigma : float, list, or tuple
        Volatility parameter. If list/tuple, random uniform sampling.
    S0 : float
        Initial stock price.
    T : float
        Time to expiration.
    T_past : float
        Time period before t=0 to generate past prices (default: 0).
    num_samples : int
        Number of sample paths to generate.
    num_steps : int
        Number of time steps.
    
    Returns:
    --------
    S : np.ndarray
        Stock price paths with shape (num_samples, num_steps + 1 + num_past_steps).
    timepoints : np.ndarray
        Corresponding time points.
    """
    # Initialize arrays
    S = np.zeros([num_samples, num_steps + 1])
    S[:, 0] = S0
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    
    num_past_steps = int(T_past // dt)
    Spast = np.zeros([num_samples, num_past_steps + 1])
    Spast[:, 0] = S0

    # Generate random normal samples
    normal_sample = np.random.normal(0, 1, [num_samples, num_steps])
    normal_sample_past = np.random.normal(0, 1, [num_samples, num_past_steps])

    # Initialize drift and volatility (support for uncertain parameters)
    if isinstance(mu, (list, tuple)):
        # Case of uncertain drift: sample uniformly from range
        Mu = np.random.uniform(mu[0], mu[1], num_samples)
    else:
        # Case of fixed drift
        Mu = mu
        
    if isinstance(sigma, (list, tuple)):
        # Case of uncertain volatility: sample uniformly from range
        Sigma = np.random.uniform(sigma[0], sigma[1], num_samples)
    else:
        # Case of fixed volatility
        Sigma = sigma

    # Euler-Maruyama method for generating future stock prices
    for k in range(1, num_steps + 1):
        S[:, k] = (S[:, k - 1] + 
                   Mu * S[:, k - 1] * dt + 
                   Sigma * S[:, k - 1] * normal_sample[:, k - 1] * sqrt_dt)

    # Euler-Maruyama method for generating past stock prices
    for k in range(1, num_past_steps + 1):
        Spast[:, k] = (Spast[:, k - 1] + 
                       Mu * Spast[:, k - 1] * dt + 
                       Sigma * Spast[:, k - 1] * normal_sample_past[:, k - 1] * sqrt_dt)
    
    # Remove initial value and reverse past prices (so they're in chronological order)
    Spast = Spast[:, 1:]
    Spast = np.fliplr(Spast)

    # Concatenate past and future stock prices
    S = np.hstack([Spast, S])

    # Generate time axis corresponding to the stock prices
    timepoints = np.hstack([
        np.arange(-num_past_steps, 0), 
        np.array([0]), 
        np.arange(1, num_steps + 1)
    ]) * dt

    return S, timepoints


def stock_prices_generator(mu=0.05, sigma=0.3, S0=5, T=1, T_past=0, num_samples=50, 
                           num_steps=100, maxgen=np.inf, randomseed=None, modelinput=False):
    """
    Generator function for producing batches of stock price paths for neural network training.
    
    Parameters:
    -----------
    mu : float, list, or tuple
        Drift parameter (interest rate).
    sigma : float, list, or tuple
        Volatility parameter.
    S0 : float
        Initial stock price.
    T : float
        Time to expiration.
    T_past : float
        Time period before t=0 to generate past prices.
    num_samples : int
        Number of sample paths per batch.
    num_steps : int
        Number of time steps.
    maxgen : float
        Maximum number of batches to generate (default: infinite).
    randomseed : int, optional
        Random seed for reproducibility.
    modelinput : bool
        If True, returns formatted input for neural network training.
        If False, returns only stock prices.
    
    Yields:
    -------
    If modelinput=True:
        input_to_train : list
            [stock_prices, timepoints] formatted for neural network input.
        dummy_target : np.ndarray
            Dummy target array (zeros) with shape (num_samples, num_steps + 1, 3).
    If modelinput=False:
        S : np.ndarray
            Stock price paths.
    """
    # Initialize random seed if provided
    if randomseed is not None:
        np.random.seed(randomseed)

    # Generate fixed timepoint input for the neural network
    _, timepts = stock_prices(mu, sigma, S0, T, T_past, num_samples, num_steps)
    timepoints = np.ones((num_samples, 1)).dot(timepts.reshape(1, -1))

    # Generate batches
    k = 0
    while k < maxgen:
        k += 1

        # Generate stock prices
        S, _ = stock_prices(mu, sigma, S0, T, T_past, num_samples, num_steps)
        
        if modelinput:
            # Format input for neural network training
            # Expand dimensions to match expected input shape
            input_to_train = [
                np.expand_dims(S, axis=2), 
                np.expand_dims(timepoints, axis=2)
            ]
            # Dummy target (zeros) - not used in custom loss function
            dummy_target = np.zeros([S.shape[0], S.shape[1], 3])
            yield input_to_train, dummy_target
        else:
            # Return stock prices only
            yield S
