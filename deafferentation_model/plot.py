import matplotlib.pyplot as plt
import numpy as np

from . import util


def imshow(z, ax=None):
    if ax is None:
        figure, ax = plt.subplots(1, 1)
    y_dim, x_dim = z.dims
    y = z.coords[y_dim].values
    x = z.coords[x_dim].values * 1e3
    ax.pcolormesh(x, y, z, shading='gouraud')
    ax.set_yscale('log')
    ticks = 2**np.arange(7, 14, 1)
    ax.set_yticks(ticks)
    ax.set_yticks([], minor=True)
    ax.set_yticklabels(f'{t*1e-3:.2f}' for t in ticks)
    ax.set_ylabel('Cochlear Frequency (kHz)')
    ax.set_xlabel('Time (msec)')


def plot_cochleagrams(trace, ax=None, cf_lb=150, cf_ub=20e3, color='k'):
    if ax is None:
        figure, ax = plt.subplots(1, 1)

    t0 = trace.posterior.theta_i.values[..., 0]
    t1 = trace.posterior.theta_i.values[..., 1]
    t2 = trace.posterior.theta_i.values[..., 2]

    lb, ub = util.inv_greenwood([cf_lb, cf_ub])
    cf_i = np.linspace(lb, ub, 100)
    cf = util.greenwood(cf_i)

    y_ln = -np.exp(t0[..., np.newaxis]) \
        * (cf_i - t1[..., np.newaxis]) ** 2 \
        + t2[..., np.newaxis]

    y = np.exp(y_ln)
    y_mean = y.mean(axis=(0, 1))
    ax.semilogx(cf, y_mean.T, '-', color=color, alpha=0.25)

    ticks = 2**np.arange(np.log2(200), 14)
    ticks = ticks[::2]
    ax.xaxis.set_ticks(ticks);
    ax.xaxis.set_ticks([], minor=True);
    ax.xaxis.set_ticklabels(f'{t*1e-3:.1f}' for t in ticks);
    ax.set_xlabel('Cochlear Frequency (kHz)')
    ax.set_ylabel('Predicted functional synapses')
