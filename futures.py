import numpy as np

def simulate_futures_kiesel_2f(Ts, sigma1, sigma2, kappa, n_scenarios, n_timesteps, end, t=0.):
    dt = end/float(n_timesteps)
    Ts = np.array(Ts) # Just in case a list is passed as argument instead of a numpy array
    mus = np.zeros((len(Ts), 2))
    sigmas1 = np.exp(-kappa*(Ts-t))*sigma1*np.sqrt(dt)
    sigmas2 = np.repeat(sigma2, len(Ts))*np.sqrt(dt)
    sigmas = np.array((sigmas1, sigmas2)).T
    dW1, dW2 = np.random.normal(mus, sigmas, size=(n_scenarios, n_timesteps, len(Ts), 2)).T # dWi = 3D array with shape (len(Ts), n_timesteps, n_scenarios)
    dF = dW1 + dW2
    return np.exp(np.cumsum(dF, axis=1))
    






if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    
    F = simulate_futures_kiesel_2f(np.arange(0, 11), 0.37, 0.15, 1.4, 5, 12*10, 20)
    fig, ax = plt.subplots()
    ax.plot(F[0])
    plt.show()
    
    
    fig, ax = plt.subplots()
    ax.plot(F[:, 0, :])
    plt.show()