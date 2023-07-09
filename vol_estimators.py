import numpy as np
import pandas as pd
from scipy.special import gamma


def get_dts(date_range, freq):
    # This function is used when we assume that only the days 
    # in date_range matter (in the sense that volatility is
    # spread evenly between each datetime in date_range)
    full_year_datetimes = pd.date_range(f'{date_range[0].year}-01-01', 
                                        f'{date_range[-1].year}-12-31', 
                                        freq=freq)
    year_lengths = 1/pd.Series(full_year_datetimes, name='year_lengths').groupby(full_year_datetimes.year).apply(len)
    dts = year_lengths.loc[date_range.year].values
    return dts

def simul(annual_mu, annual_sigma, start_date, end_date, freq='D', n_sims=10_000):
    datetimes = pd.date_range(start_date, end_date, freq=freq)
    dts = get_dts(datetimes, freq)
    mu = (annual_mu*dts).reshape(-1, 1)
    sigma = (annual_sigma*np.sqrt(dts)).reshape(-1, 1)
    X = np.random.normal(mu, sigma, size=(len(datetimes), n_sims))
    X[0] = 0
    X = np.exp(X.cumsum(axis=0))
    return pd.DataFrame(X, datetimes)


def close_to_close_vol(log_returns_close, N=None):
    if N is not None: # N=None allows use as series.rolling(N).apply(close_to_close_vol)
        log_returns_close = log_returns_close.iloc[-N:]
    N = len(log_returns_close)
    variance = (log_returns_close-log_returns_close.mean()).var() # -mean to avoid fitting noise
    # Calculate bias to correct Jensen's inequality
    bias = np.sqrt(2/N)*gamma(N/2)/gamma((N-1)/2)
    return np.sqrt(variance)/bias



def parkinson_vol(high_price, low_price, N=50):
    high_price = high_price.iloc[-N:]
    low_price = low_price.iloc[-N:]
    sslogdif = ((np.log(high_price) - np.log(low_price))**2).sum()
    coef = 1/(4*N*np.log(2))
    return np.sqrt(coef*sslogdif)



def garman_klass_vol_simple(log_returns_close, log_returns_high, log_returns_low, N=50):
    # CAREFUL: The original paper doesn't seem to have this estimator 
    assert(len(log_returns_high) >= N and len(log_returns_low) >= N, 
           f"Lengths of log_returns_high and log_returns_low must be greater than N. Passed: len(high) = {len(log_returns_high)}, len(low) = {len(log_returns_low)}, N = {N}")
    log_returns_close = log_returns_close.iloc[-N:]
    log_returns_high = log_returns_high.iloc[-N:]
    log_returns_low = log_returns_low.iloc[-N:]
    first = (np.log(log_returns_high/log_returns_low.values)**2).iloc[1:].sum()
    second = ((2*np.log(2)-1) * np.log(log_returns_close.pct_ch)**2).iloc[1:].sum() # Generates 1 nan, hence iloc[1:]
    return np.sqrt(1/N) * np.sqrt(first/2 - second)

def get_high_low_close(prices_series):
    # prices_series must have intraday data
    return pd.DataFrame({
        'high':prices_series.groupby(prices_series.index.date).max(),
        'low':prices_series.groupby(prices_series.index.date).min(),
        'close':prices_series[pd.Series(prices_series.index).groupby(prices_series.index.date).apply(lambda x:x==x.max()).values].values
        })
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    annual_sigma = 0.3
    X = simul(0.01,annual_sigma, '2007-01-01', '2009-01-01', freq='H')
    X.iloc[:,:20].plot(legend=False, color='black', alpha=0.2, lw=1)
    s = X[0].apply(np.log).diff().iloc[1:]
    ctc = close_to_close_vol(s, 50)
    rolling_ctc = s.rolling(50).apply(close_to_close_vol)
    normal = s.iloc[-50:].std()
    
    # Tests on parkinson
    s = X[1]#[X.index.hour.isin(np.arange(9, 18))]
    hlc = get_high_low_close(s)
    tienes_parkinson = parkinson_vol(hlc['high'], hlc['low'])
    annual_parkinson = tienes_parkinson*np.sqrt(365)
    annual_ctc = close_to_close_vol(hlc['close'].apply(np.log).diff().iloc[1:], 50)*np.sqrt(365)
    
    
    
    print('ok')
