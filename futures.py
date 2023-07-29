import numpy as np

def simulate_futures_kiesel_2f(Ts, sigma1, sigma2, kappa, n_scenarios, n_timesteps, end, t=0.):
    dt = end/float(n_timesteps)
    Ts = np.array(Ts) # Just in case a list is passed as argument instead of a numpy array
    sigmas1 = np.exp(-kappa*(Ts-t))*sigma1*np.sqrt(dt)
    sigmas2 = np.repeat(sigma2, len(Ts))*np.sqrt(dt)
    dW1, dW2 = np.random.normal(size=(n_scenarios, n_timesteps, 2)).T # dWi = 2D array with shape (n_timesteps, n_scenarios)
    dW1 = np.tile(dW1, (len(Ts), 1)).reshape(len(Ts), dW1.shape[0], dW1.shape[1]) # dWi = 3D array with shape (len(Ts), n_timesteps, n_scenarios)
    dW2 = np.tile(dW2, (len(Ts), 1)).reshape(len(Ts), dW2.shape[0], dW2.shape[1])
    dW1 = (dW1.T * sigmas1).T
    dW2 = (dW2.T * sigmas2).T
    dF = dW1 + dW2
    return np.arange(1, n_timesteps+1)*dt, np.exp(np.cumsum(dF, axis=1))




if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    
    Ts=np.arange(0, 7)
    sigma1=0.37
    sigma2=0.15
    kappa=1.4
    n_scenarios=3
    n_timesteps=5
    end=20
    t=0.
    
    start_time = time.perf_counter()
    xtime, F = simulate_futures_kiesel_2f(Ts, sigma1, sigma2, kappa, n_scenarios, n_timesteps, end, t=t)
    end_time = time.perf_counter()
    print(f'Function simulate_futures_kiesel_2f took {end_time - start_time:.4f} seconds')
    
    fig, ax = plt.subplots()
    ax.plot(xtime, F[0])
    plt.show()
    
    
    fig, ax = plt.subplots()
    ax.plot(F[:, 0, :], color='black', alpha=0.05)
    plt.show()