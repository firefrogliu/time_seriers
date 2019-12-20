import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import vmdpy

DIR_DATA = 'data/obs_update_nwp_allyear.csv'


def VMD(ts, window_size, fre_observed=24):
    """using the latest window_size history values"""
    ## parameters
    alpha = 2000  # moderate bandwidth constraint
    tau = 0.  # noise-tolerance (no strict fidelity enforcement)
    K = 5  # no. modes
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-7
    ## start to run
    length = len(ts)
    res = np.ones((length, K), dtype=np.float) * np.NaN
    for t in range(window_size-1, length, fre_observed):
        sub_ts = ts[(t - window_size + 1):(t + 1)]
        u, u_hat, omega = vmdpy.VMD(sub_ts, alpha, tau, K, DC, init, tol)
        u = np.transpose(u)
        res[(t-fre_observed+1):(t+1)] = u[-fre_observed::]

    return np.transpose(res)

def VMD_inc(ts, window_size, fre_observed=24):
    """using all history values"""
    ## parameters
    alpha = 2000  # moderate bandwidth constraint
    tau = 0.  # noise-tolerance (no strict fidelity enforcement)
    K = 5  # no. modes
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-7
    ## start to run
    length = len(ts)
    res = np.ones((length, K), dtype=np.float) * np.NaN
    for t in range(window_size-1, length, fre_observed):
        sub_ts = ts[:(t + 1)]
        u, u_hat, omega = vmdpy.VMD(sub_ts, alpha, tau, K, DC, init, tol)
        u = np.transpose(u)
        res[(t-fre_observed+1):(t+1)] = u[-fre_observed::]

    return np.transpose(res)

if __name__ == '__main__':
    df = pd.read_csv(DIR_DATA)
    wind = df['wind'].values
    window_size = 24*50

    # ## sm.tsa.seasonal_decompose
    # res = sm.tsa.seasonal_decompose(wind, freq=window_size)
    # ts_components = [wind, res.seasonal, res.trend, res.resid]
    # n_components = len(ts_components)
    # titles = ['wind', 'seasonal', 'trend', 'resid']
    # fig, axs = plt.subplots(n_components, 1, sharex=True)
    # for ts, ax, title in zip(ts_components, axs, titles):
    #     ax.plot(ts[window_size:(window_size + 1000)])
    #     ax.set_ylabel(title)
    # plt.savefig('cache/sm.tsa.seasonal_decompose_{}.png'.format(window_size))
    # plt.close()

    ## VMD
    ts_components = VMD(wind, window_size)
    n_components = len(ts_components)
    titles = ['comp_{}'.format(i) for i in range(n_components)]
    fig, axs = plt.subplots(n_components, 1, sharex=True)
    for ts, ax, title in zip(ts_components, axs, titles):
        ax.plot(ts[window_size:(window_size+1000)])
        ax.set_ylabel(title)
    plt.savefig('cache/vmd_{}.png'.format(window_size))
    plt.close()

    ## VMD_inc
    ts_components = VMD_inc(wind, window_size)
    n_components = len(ts_components)
    titles = ['comp_{}'.format(i) for i in range(n_components)]
    fig, axs = plt.subplots(n_components, 1, sharex=True)
    for ts, ax, title in zip(ts_components, axs, titles):
        ax.plot(ts[window_size:(window_size + 1000)])
        ax.set_ylabel(title)
    plt.savefig('cache/vmd_inc_{}.png'.format(window_size))
    plt.close()


