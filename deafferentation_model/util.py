import numpy as np
from scipy import signal
import pandas as pd


def tone_conv(s, fs, frequency, detrend=None, window=None):
    frequency_shape = tuple([Ellipsis] + [np.newaxis]*s.ndim)
    frequency = np.asarray(frequency)[frequency_shape]
    if detrend is not None:
        s = signal.detrend(s, type=detrend, axis=-1)
    n = s.shape[-1]
    if window is not None:
        w = signal.get_window(window, n)
        s = w/w.mean()*s
    t = np.arange(n)/fs
    r = 2.0*s*np.exp(-1.0j*(2.0*np.pi*t*frequency))
    return np.mean(r, axis=-1)


def tone_power_conv(s, fs, frequency, detrend=None, window=None):
    r = tone_conv(s, fs, frequency, detrend, window)
    return np.abs(r)/np.sqrt(2.0)


def db(x, ref=1):
    return 20 * np.log10(x/ref)


def spl_to_pa(spl):
    return np.sqrt(2) * 20e-6 * 10**(spl/20)


def greenwood(x):
    x = np.asanyarray(x)
    return 165.4 * (10 ** (2.1 * x) - 0.88)


def inv_greenwood(f):
    f = np.asanyarray(f)
    return np.log10(f / 165.4 + 0.88) / 2.1


def csd(s, fs, window=None, waveform_averages=None):
    if waveform_averages is not None:
        new_shape = (waveform_averages, -1) + s.shape[1:]
        s = s.reshape(new_shape).mean(axis=0)
    s = signal.detrend(s, type='linear', axis=-1)
    n = s.shape[-1]
    if window is not None:
        w = signal.get_window(window, n)
        s = w/w.mean()*s
    return np.fft.rfft(s, axis=-1)/n


def phase(s, fs, window=None, waveform_averages=None, unwrap=True):
    c = csd(s, fs, window, waveform_averages)
    p = np.angle(c)
    if unwrap:
        p = np.unwrap(p)
    return p


def psd(s, fs, window=None, waveform_averages=None):
    c = csd(s, fs, window, waveform_averages)
    return 2*np.abs(c)/np.sqrt(2.0)


def psd_freq(s, fs):
    return np.fft.rfftfreq(s.shape[-1], 1.0/fs)


def psd_df(s, fs, *args, **kw):
    p = psd(s, fs, *args, **kw)
    freqs = pd.Index(psd_freq(s, fs), name='frequency')
    if p.ndim == 1:
        name = s.name if isinstance(s, pd.Series) else 'psd'
        return pd.Series(p, index=freqs, name=name)
    else:
        index = s.index if isinstance(s, pd.DataFrame) else None
        return pd.DataFrame(p, columns=freqs, index=index)
