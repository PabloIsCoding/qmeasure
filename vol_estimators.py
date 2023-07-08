# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.special import gamma

def close_to_close(log_returns_close:pd.Series, N=50) -> float:
    log_returns_close = log_returns_close.iloc[-N:]
    variance = (log_returns_close-log_returns_close.mean()).std() # -mean to avoid fitting noise
    # Calculate bias to correct Jensen's inequality
    bias = np.sqrt(2/N)*gamma(N/2)/gamma((N-1)/2)
    return np.sqrt(variance)/bias

print('hi')