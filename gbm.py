# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 18:11:19 2023

@author: Pablo Parguiña Silva
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_GBM_ABM_paths(n_paths, n_timesteps, T, r, sigma, S_0, include_start=True, random_seed=None):
    """
    Generates paths for Arithmetic Brownian Motion and Geometric Brownian
    Motion.

    Parameters
    ----------
    n_paths : int
        Number of paths of the stock to simualte (or whatever other underlying).
    n_timesteps : int
        Controls the granularity of the simulation. The more timesteps, the
        higher the accuracy of this discrete approximation of ABM/GBM.
    T : float
        Length of the time period to simulate, such that when T=1, the parameter
        sigma is the volatility over that period. T can be lower or higher than
        one.
    r : float
        Constant interest rate.
    sigma : float
        Volatility over a period of Δt=1 (typically annual volatility).
    S_0 : float
        Starting value of the series (normally a stock or some other underlying).
    include_start : Bool, optional
        Whether to include the S_0 value in these simulations. The default is 
        True.
    random_seed : TYPE, optional
        Random seed. If None, no random seed is specified. The default is None.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    if random_seed is not None:
        np.random.seed(random_seed)
    delta_t = T/float(n_timesteps)
    n_steps = n_timesteps + 1 if include_start else n_timesteps
    X = np.random.normal((r - sigma**2/2)*delta_t, sigma*np.sqrt(delta_t), size=(n_steps, n_paths))
    
    # In case we want to ensure that the brownian returns have the correct mean and std
    # I comment this section because it doesn't seem to be an improvement
    # mus = np.zeros((X.shape[0], 1))
    # mus[:,:] = (r - sigma**2/2)*delta_t
    # sigmas = np.zeros((X.shape[0], 1))
    # sigmas[:, :] = sigma*np.sqrt(delta_t)
    # X = (X-np.mean(X, axis=1).reshape(-1, 1) + mus)/np.std(X, axis=1).reshape(-1, 1)*sigmas
    if include_start:
        X[0, :] = 0
    
    X = np.cumsum(X, axis=0) + np.log(S_0)
    S = np.exp(X)
    first_timestep = 0 if include_start else delta_t
    time = np.linspace(first_timestep, T, n_steps)
    return {'time':time, 'X':X, 'S':S}

if __name__ == '__main__':
    n_paths = 10000
    n_timesteps = 200
    T = 1
    r = 0.8
    sigma = 0.2
    S_0 = 100
    include_start = True
    paths = simulate_GBM_ABM_paths(n_paths, n_timesteps, T, r, sigma, S_0, include_start=include_start)
    time, S = paths['time'], paths['S']
    fig, ax = plt.subplots()
    ax.plot(time, S, color='black', alpha=0.05)