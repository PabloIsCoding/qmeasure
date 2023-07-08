# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:40:30 2023

@author: Pablo Parguiña Silva
"""

import numpy as np
from scipy.stats import norm, binom
from math import e

def _build_log_matrix(value1, value2, N):
    result = np.ones((N, N))
    V1 = np.zeros(N)
    V1[1:] = value1
    result[0, :] = V1.cumsum()
    for i in range(1, N):
        result[i] = result[0] - value1*i + value2*i
    return result

def _build_matrix(value1, value2, N):
    result = np.ones((N, N))
    V1 = np.ones(N)
    V1[1:] = value1
    result[0, :] = V1.cumprod()
    for i in range(1, N):
        result[i] = result[0] / value1**i * value2**i
    return np.triu(result)

def _build_underlying_matrix(S_0, u, N):
    d = 1/u
    S = np.triu(np.exp(_build_log_matrix(np.log(u), np.log(d), N)))
    return S*S_0

def _build_probabilities(p, N, timestep_number):
    q = 1-p
    P = np.zeros(N)
    rv = binom(N, p)
    return rv.pmf(np.arange(N-1, -1, -1))

def _european_payoffs(underlying_matrix, K, kind='C'):
    if kind=='C':
        payoffs = np.clip(underlying_matrix[:, -1] - K, a_min=0, a_max=np.inf)
    else:
        payoffs = np.clip(K-underlying_matrix[:, -1], a_min=0, a_max=np.inf)
    return payoffs

def value_european_option_binomial(S_0, T, sigma, K, r, kind='C', N=1000):
    dt = T/float(N)
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d)/(u - d)
    S = _build_underlying_matrix(S_0, u, N)
    P = _build_probabilities(p, N, N)
    payoffs = _european_payoffs(S, K, kind)
    return sum(payoffs*P)*np.exp(-r*T)

def calculate_u(sigma, dt, r):
    u1 = 1/2*e**(-r*dt)*(-np.sqrt((sigma**2*(-dt) - e**(2*r*dt) - 1)**2 - 4*e**(2*r*dt)) + sigma**2*dt + e**(2*r*dt) + 1)
    u2 = 1/2*e**(-r*dt)*(np.sqrt((sigma**2*(-dt) - e**(2*r*dt) - 1)**2 - 4*e**(2*r*dt)) + sigma**2*dt + e**(2*r*dt) + 1)
    return u1, u2

def value_european_option_BS(S_0, T, sigma, K, r, kind='C'):
    d1 = (np.log(S_0/K) + (r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    N = norm.cdf
    if kind=='C':
        value = S_0*N(d1) - K*np.exp(-r*T)*N(d2)
    else:
        value = K*np.exp(-r*T)*N(-d2) - S_0*N(-d1)
    return value

if __name__=='__main__':
    import matplotlib.pyplot as plt
    Ns = list(range(3, 30, 2))+[40, 50, 60, 70, 90, 120, 150, 300, 600, 900, 1200, 1600, 2200, 3000, 6000,9000, 15000, 30000]
    values = [value_european_option_binomial(S_0=42, T=0.5, sigma=0.2, K=40, r=0.1, kind='P', N=n) for n in Ns]
    bs_value = value_european_option_BS(S_0=42, T=0.5, sigma=0.2, K=40, r=0.1, kind='P')
    fig, ax = plt.subplots()
    ax.plot(Ns, np.repeat(bs_value, len(Ns)), label='Black Scholes')
    ax.plot(Ns, values, label='Binomial')
    ax.set_xlabel('Número de timesteps')
    ax.set_ylabel('Valor')
    ax.legend()
    