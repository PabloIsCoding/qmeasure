# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:40:30 2023

@author: Pablo Parguiña Silva
"""

import numpy as np
from scipy.stats import norm, binom
from math import e
from numba import njit, float64, int64

CALL, PUT = 0, 1
EUROPEAN, AMERICAN = 0, 1
    

def _build_log_matrix(value1, value2, N):
    """
    Generates all the possible paths of the underlying, doing everything in
    logs, on the hopes that this is numerically more stable. 

    Parameters
    ----------
    value1 : float
        This will typically be u, the raw return in the case the underlying goes
        up.
    value2 : float
        This will typically be d, the raw return in the case the underlying goes
        down.
    N : int
        Number of timesteps. The resulting matrix is (N+1)x(N+1)

    Returns
    -------
    result : float
        DESCRIPTION.

    """
    result = np.ones((N+1, N+1))
    V1 = np.zeros(N+1)
    V1[1:] = value1
    result[0, :] = V1.cumsum()
    for i in range(1, N+1):
        result[i] = result[0] - value1*i + value2*i
    return result

def _build_matrix(value1, value2, N):
    result = np.ones((N+1, N+1))
    V1 = np.ones(N+1)
    V1[1:] = value1
    result[0, :] = V1.cumprod()
    for i in range(1, N+1):
        result[i] = result[0] / value1**i * value2**i
    return np.triu(result)

def _build_underlying_matrix(S_0, u, N):
    """
    Creates the full matrix of the underlying, starting at a value of S_0, 
    where u is the raw return between steps when the underlying goes up and N 
    is the size of the matrix (the matrix includes S_0, so the number of steps 
    is N-1).

    Parameters
    ----------
    S_0 : float
        Underlying price at t=0.
    u : float
        Raw return in the case .
    N : int
        Number of timesteps. The resulting matrix is (N+1)x(N+1)

    Returns
    -------
    (N+1)x(N+1) float-64 numpy array
        The array constains all the possible values the underlying can take.
    """
    
    d = 1/u
    S = np.triu(np.exp(_build_log_matrix(np.log(u), np.log(d), N)))
    return S*S_0


@njit(float64[:](float64, float64, float64, float64, float64, float64, int64, int64), fastmath=True, cache=True)
def _calculate_american_option_crr(S_0, T, sigma, K, r, q=0., option_type=CALL, N=1000):
    dt = T/float(N)
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    a = np.exp((r-q)*dt)
    p = (a - d)/(u - d)
    number_of_upward_moves = np.arange(N, -1, -1) # [0, ..., N]
    possible_prices = S_0 * u**number_of_upward_moves * d**(N-number_of_upward_moves)
    discount_dt = np.exp(-r*dt)
    values = np.clip(possible_prices - K, a_min=0, a_max=np.inf) if option_type == CALL else np.clip(K - possible_prices, a_min=0, a_max=np.inf)
    option_coef = 1 if option_type == CALL else -1
    for i in range(1, N+1):
        possible_prices = possible_prices[:-1] / u
        expected_values = (values[:-1]*p + np.roll(values, -1)[:-1]*(1-p))*discount_dt # Values at N-i without exercising early
        payoff = np.clip((possible_prices - K)*option_coef, a_min=0, a_max=np.inf) # early exercise
        values = np.maximum(payoff, expected_values) # Values at N-i, possibly exercising early

        if i == N-1:
            delta = (values[1] - values[0])/(possible_prices[0] - possible_prices[1])*option_coef
    return np.array((values[0], delta))

def _calculate_european_option_crr(S_0, T, sigma, K, r, q=0., option_type=CALL, N=1000):
    dt = T/float(N)
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    a = np.exp((r-q)*dt)
    p = (a - d)/(u - d)
    number_of_upward_moves = np.arange(N, -1, -1) # [0, ..., N]
    possible_prices = S_0 * u**number_of_upward_moves * d**(N-number_of_upward_moves)
    values = np.clip(possible_prices - K, a_min=0, a_max=np.inf) if option_type == CALL else np.clip(K - possible_prices, a_min=0, a_max=np.inf)
    return np.array((np.sum(binom(N, p).pmf(number_of_upward_moves)*values)*np.exp(-r*T), 0.0)) # TODO: add greeks

def calculate_option_crr(S_0, T, sigma, K, r, q=0., option_type=CALL, payoff_type=EUROPEAN, N=1000):
    """
    Values an european option with the binomial method. The underlying could be
    a stock, a exchange rate or a future. If it is a future, both r and q should
    be zero.

    Parameters
    ----------
    S_0 : float
        Underlying price at t=0.
    T : float
        Time to expiration, in fraction of the year. For example if t=1 is 1 
        year and the option expires in 6 months, T=0.5
    sigma : float
        volatility for Δt=1 (typically annual volatility).
    K : float
        Strike price of the option.
    r : float
        Constant interest rate over the period of the option.
    q : float
        Constant continuous dividend yield, or constant foreign risk free rate 
        in the case of FX options.
    option_type : int, optional
        Either CALL (=0) or PUT (=1)
    payoff_type : int, optional
        Either EUROPEAN (=0) or AMERICAN (=1)
    N : int, optional
        Number of steps. Increase it to get a better approximation of the value
        of the option, although it can require a lot of memory. 
        The default is 1000.

    Returns
    -------
    value : float
        Value of the option in monetary units.

    """
    
    if not option_type in [CALL, PUT]:
        raise Exception(f"Only valid option types are {CALL} or {PUT}, but got {option_type}")
    if not payoff_type in [EUROPEAN, AMERICAN]:
        raise Exception(f"Only valid option types are {EUROPEAN} or {AMERICAN}, but got {payoff_type}")
    
    if payoff_type == EUROPEAN:
        value, delta = _calculate_european_option_crr(S_0, T, sigma, K, r, q=q, option_type=option_type, N=N)
    else:
        value, delta = _calculate_american_option_crr(S_0, T, sigma, K, r, q=q, option_type=option_type, N=N)
    return value, delta



def calculate_u(sigma, dt, r):
    """
    Returns the u parameter for the binomial model for a non-dividend paying stock. 
    The typical formula you read in books is an approximation. One of the two 
    returned values is the actual u. In practice this doesn't matter much and 
    that's why I didn't bother to use it in the end, but I wanted to keep it 
    just in case.

    Parameters
    ----------
    sigma : float
        volatility for Δt=1 (typically annual volatility).
    dt : float
        timestep between steps in the binomial model.
    r : float
        Constant interest rate.

    Returns
    -------
    u1 : float
        Possible value of u, the raw return in the case the underlying goes up.
    u2 : float
        The other possible value of u.

    """
    u1 = 1/2*e**(-r*dt)*(-np.sqrt((sigma**2*(-dt) - e**(2*r*dt) - 1)**2 - 4*e**(2*r*dt)) + sigma**2*dt + e**(2*r*dt) + 1)
    u2 = 1/2*e**(-r*dt)*(np.sqrt((sigma**2*(-dt) - e**(2*r*dt) - 1)**2 - 4*e**(2*r*dt)) + sigma**2*dt + e**(2*r*dt) + 1)
    return u1, u2

def value_european_option_BS(S_0, T, sigma, K, r, q=0, kind='C'):
    """
    Value of a european option according to the Black-Scholes model.

    Parameters
    ----------
    S_0 : float
        Underlying price at t=0.
    T : float
        Time to expiration, in fraction of the year. For example if t=1 is 1 
        year and the option expires in 6 months, T=0.5
    sigma : float
        volatility for Δt=1 (typically annual volatility). 
    K : float
        Strike price of the option.
    r : float
        Constant interest rate over the period of the option.
    q : float
        Constant continuous dividend yield.
    kind : String, optional
        'C' for call option, anything else for put option. The default is 'C'.

    Returns
    -------
    value : float
        Value of the option in monetary units.

    """
    d1 = (np.log(S_0/K) + (r-q+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    N = norm.cdf
    if kind=='C':
        value = S_0*np.exp(-q*T)*N(d1) - K*np.exp(-r*T)*N(d2)
    else:
        value = K*np.exp(-r*T)*N(-d2) - S_0*np.exp(-q*T)*N(-d1)
    return value

@njit(float64[:](float64, float64, float64, float64, int64, float64, int64,
                 float64), fastmath=True, cache=True)
def crr_tree_val(stock_price,
                 ccInterestRate,  # continuously compounded
                 ccDividendRate,  # continuously compounded
                 volatility,  # Black scholes volatility
                 num_steps_per_year,
                 time_to_expiry,
                 option_type,
                 strike_price):
    """ Value an American option using a Binomial Treee """

    # num_steps = int(num_steps_per_year * time_to_expiry)

    # if num_steps < 30:
    #     num_steps = 30

    # OVERRIDE JUST TO SEE
    num_steps = num_steps_per_year

#    print(num_steps)
    # this is the size of the step
    dt = time_to_expiry / num_steps
    r = ccInterestRate
    q = ccDividendRate

    # the number of nodes on the tree
    num_nodes = int(0.5 * (num_steps + 1) * (num_steps + 2))
    stock_values = np.zeros(num_nodes)
    stock_values[0] = stock_price

    option_values = np.zeros(num_nodes)
    u = np.exp(volatility * np.sqrt(dt))
    d = 1.0 / u
    sLow = stock_price

    probs = np.zeros(num_steps)
    periodDiscountFactors = np.zeros(num_steps)

    # store time independent information for later use in tree
    for iTime in range(0, num_steps):
        a = np.exp((r - q) * dt)
        probs[iTime] = (a - d) / (u - d)
        periodDiscountFactors[iTime] = np.exp(-r * dt)

    for iTime in range(1, num_steps + 1):
        sLow *= d
        s = sLow
        for iNode in range(0, iTime + 1):
            index = 0.5 * iTime * (iTime + 1)
            stock_values[int(index + iNode)] = s
            s = s * (u * u)

    # work backwards by first setting values at expiry date
    index = int(0.5 * num_steps * (num_steps + 1))

    for iNode in range(0, iTime + 1):

        s = stock_values[index + iNode]

        if option_type == 0: # EUROPEAN CALL
            option_values[index + iNode] = np.maximum(s - strike_price, 0.0)
        elif option_type == 1:# EUROPEAN PUT
            option_values[index + iNode] = np.maximum(strike_price - s, 0.0)
        elif option_type == 2: # AMERICAN CALL
            option_values[index + iNode] = np.maximum(s - strike_price, 0.0)
        elif option_type == 3:
            option_values[index + iNode] = np.maximum(strike_price - s, 0.0)

    # begin backward steps from expiry to value date
    for iTime in range(num_steps - 1, -1, -1):

        index = int(0.5 * iTime * (iTime + 1))

        for iNode in range(0, iTime + 1):

            s = stock_values[index + iNode]

            exerciseValue = 0.0

            if option_type == 0:
                exerciseValue = 0.0
            elif option_type == 1:
                exerciseValue = 0.0
            elif option_type == 2:
                exerciseValue = np.maximum(s - strike_price, 0.0)
            elif option_type == 3:
                exerciseValue = np.maximum(strike_price - s, 0.0)

            nextIndex = int(0.5 * (iTime + 1) * (iTime + 2))

            nextNodeDn = nextIndex + iNode
            nextNodeUp = nextIndex + iNode + 1

            vUp = option_values[nextNodeUp]
            vDn = option_values[nextNodeDn]
            futureExpectedValue = probs[iTime] * vUp
            futureExpectedValue += (1.0 - probs[iTime]) * vDn
            holdValue = periodDiscountFactors[iTime] * futureExpectedValue

            if option_type == 0:
                option_values[index + iNode] = holdValue
            elif option_type == 1:
                option_values[index + iNode] = holdValue
            elif option_type == 2:
                option_values[index +
                              iNode] = np.maximum(exerciseValue, holdValue)
            elif option_type == 3:
                option_values[index +
                              iNode] = np.maximum(exerciseValue, holdValue)

    # We calculate all of the important Greeks in one go
    price = option_values[0]
    delta = (option_values[2] - option_values[1]) / \
        (stock_values[2] - stock_values[1])
    deltaUp = (option_values[5] - option_values[4]) / \
        (stock_values[5] - stock_values[4])
    deltaDn = (option_values[4] - option_values[3]) / \
        (stock_values[4] - stock_values[3])
    gamma = (deltaUp - deltaDn) / (stock_values[2] - stock_values[1])
    theta = (option_values[4] - option_values[0]) / (2.0 * dt)
    results = np.array([price, delta, gamma, theta])
    return results

if __name__=='__main__':
    import matplotlib.pyplot as plt
    Ns = np.arange(5,60)
    S_0 = 50.
    T = 5./12.
    r = 0.1
    q = 0.0
    sigma = 0.4
    K = 50.
    kind='P'
    option_type = PUT
    payoff_type = AMERICAN
    
    bs_value = value_european_option_BS(S_0=S_0, T=T, sigma=sigma, K=K, r=r, q=q, kind=kind)
    crr_values = [calculate_option_crr(S_0=S_0, T=T, sigma=sigma, K=K, r=r, q=q, option_type=option_type, payoff_type=payoff_type , N=n)[0] for n in Ns]
    finpy_values = [crr_tree_val(S_0, r, q, sigma, i, T, 3, K)[0] for i in Ns]
    
    fig, ax = plt.subplots()
    ax.plot(Ns, np.repeat(bs_value, len(Ns)), label='Black Scholes')
    ax.plot(Ns, crr_values, label='CRR', lw=1)
    ax.plot(Ns, finpy_values, label='Finpy', lw=0.5)
    ax.set_xlabel('Number of timesteps')
    ax.set_ylabel('Value')
    ax.legend()
    