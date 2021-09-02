import numpy as np
import scipy as sp
from scipy import signal

from deafferentation_model import util


ABR_STIM_DESCRIPTIONS = {
    'StElizabeth': {
        4000: {
            'levels': [90 + 23],
            'duration': 8 / 4000,
            'rise_time': 4 / 4000,
        }
    },
    'NCRAR-ABR': {
        3000: {
            'levels': [110],
            'duration': 7.5 / 3000,
            'rise_time': 0.5e-3,
        },
        4000: {
            'levels': [80, 90, 100, 110],
            'duration': 8 / 4000,
            'rise_time': 0.5e-3,
        },
        6000: {
            'levels': [110],
            'duration': 9 / 6000,
            'rise_time': 0.5e-3,
        }
    },
    'NCRAR-EFR': {
        4000: {
            'levels': [90, 100, 110],
            'duration': 8 / 4000,
            'rise_time': 0.5e-3,
        },
    }
}


def blackman_envelope(fs, offset, samples, start_time, rise_time, duration):
    '''
    Generates cosine-squared envelope. Can handle generating fragments (i.e.,
    incomplete sections of the waveform).

    Parameters
    ----------
    fs : float
        Sampling rate
    offset : int
        Offset to begin generating waveform at (in samples relative to start)
    samples : int
        Number of samples to generate
    start_time : float
        Start time of envelope
    '''
    t = (np.arange(samples, dtype=np.double) + offset)/fs
    n_window = int(round(rise_time * fs))
    ramp = signal.blackman(n_window*2)

    m_null_pre = (t < start_time)
    m_onset = (t >= start_time) & (t < (start_time + rise_time))

    # If duration is set to infinite, than we only apply an *onset* ramp.
    # This is used, in particular, for the DPOAE stimulus in which we want
    # to ramp on a continuous tone and then play it continuously until we
    # acquire a sufficient number of epochs.
    if duration != np.inf:
        m_offset = (t >= (start_time+duration-rise_time)) & \
            (t < (start_time+duration))
        m_null_post = t >= (duration+start_time)
    else:
        m_offset = np.zeros_like(t, dtype=np.bool)
        m_null_post = np.zeros_like(t, dtype=np.bool)

    t_null_pre = t[m_null_pre]
    t_onset = t[m_onset] - start_time
    t_offset = t[m_offset] - start_time
    t_ss = t[~(m_null_pre | m_onset | m_offset | m_null_post)]
    t_null_post = t[m_null_post]

    f_null_pre = np.zeros(len(t_null_pre))

    f_lower = ramp[:n_window][np.flatnonzero(m_onset)]
    i_upper = np.flatnonzero(m_offset)
    i_upper -= i_upper.min()
    f_upper = ramp[n_window:][i_upper]
    f_middle = np.ones(len(t_ss))
    f_null_post = np.zeros(len(t_null_post))

    concat = [f_null_pre, f_lower, f_middle, f_upper, f_null_post]
    return np.concatenate(concat, axis=-1)


def generate_tone(fs, frequency, level, duration, rise_time, total_duration=4e-3):
    total_samples = round(total_duration * fs)
    t = np.arange(total_samples)/fs
    tone = np.cos(2*np.pi*frequency*t)
    env = blackman_envelope(fs, 0, total_samples, 0, rise_time, duration)
    return tone * env * util.spl_to_pa(level)


def generate_tones_old(study, fs=100e3, total_duration=4e-3):
    total_samples = round(total_duration * fs)
    t = np.arange(total_samples)/fs
    tone_stim = {}
    for frequency, ti in ABR_STIM_DESCRIPTIONS[study].items():
        duration = ti['duration']
        levels = ti['levels']
        rise_time = ti['rise_time']
        tone = np.cos(2*np.pi*frequency*t)
        env = blackman_envelope(fs, 0, total_samples, 0, rise_time, duration)
        for level in levels:
            tone_stim[frequency, level] = tone * env * util.spl_to_pa(level)
    return t, tone_stim


def generate_efr(fs, fc, fm, depth, level, stim_duration, rise_time, actual_duration):
    total_samples = round(stim_duration * fs)
    t = np.arange(total_samples)/fs
    tone = np.cos(2*np.pi*fc*t)
    mod = depth/2.0*np.sin(2.0*np.pi*fm*t - np.pi/2)+1.0-depth/2.0
    efr = tone * mod * util.spl_to_pa(level)
    env = cos2envelope(fs, 0, len(efr), 0, rise_time, stim_duration)
    efr *= env
    actual_samples = int(round(actual_duration * fs))
    return t[:actual_samples], efr[:actual_samples]


def fir_filter(fs, x, fl, fh, n_taps=1024, transition_width=0.1, mode='full'):
    fl_norm, fh_norm = np.array([fl, fh]) * 2 / fs
    frequency_breakpoints = [
        0,
        fl_norm * (1 - transition_width),
        fl_norm,
        np.mean([fl_norm, fh_norm]),
        fh_norm,
        fh_norm * (1 + transition_width),
        1,
    ]
    gains = [0, 3e-5, 1, 1, 1, 3e-5, 0]
    b = signal.firwin2(n_taps, frequency_breakpoints, gains)
    x_filt = np.convolve(x, b, mode=mode)
    return x_filt


def filtered_click(fs, level, duration=100e-6, full_duration=5e-3, trim=1e-3,
                   polarity='rarefaction', center=5e-3):
    # Use peak to peak equivalent
    sf = util.spl_to_pa(level)

    samples = int(round(fs * duration))
    trim_samples = int(round(fs * trim))
    window_samples = int(round(fs * full_duration))
    c = int(round(fs * center))
    click = np.zeros(window_samples)
    click[c:c+samples] = 1

    click_filt = fir_filter(fs, click, 350, 3000)
    click_filt /= np.ptp(click_filt)
    click_filt = click_filt * sf
    i = np.argmax(click_filt)
    click_filt = click_filt[i-trim_samples:i+trim_samples]
    segment = np.zeros(window_samples)
    segment[c-trim_samples:c+trim_samples] = click_filt

    if polarity == 'condensation':
        pass
    elif polarity == 'rarefaction':
        segment = -segment
    return segment



def broadband_noise(fs, level, duration):
    samples = int(round(fs * duration))
    #sf = np.sqrt(fs / 2) * util.spl_to_pa(level)
    # TODO: HACK ALERT!
    sf = util.spl_to_pa(level + 25)
    return np.random.uniform(-sf*np.sqrt(3), sf*np.sqrt(3), size=samples)


def cos2ramp(t, rise_time, phi=0):
    return np.sin(2*np.pi*t*1.0/rise_time*0.25+phi)**2


def cos2envelope(fs, offset, samples, start_time, rise_time, duration):
    '''
    Generates cosine-squared envelope. Can handle generating fragments (i.e.,
    incomplete sections of the waveform).

    Parameters
    ----------
    fs : float
        Sampling rate
    offset : int
        Offset to begin generating waveform at (in samples relative to start)
    samples : int
        Number of samples to generate
    start_time : float
        Start time of envelope
    '''
    t = (np.arange(samples, dtype=np.double) + offset)/fs

    m_null_pre = (t < start_time)
    m_onset = (t >= start_time) & (t < (start_time + rise_time))

    # If duration is set to infinite, than we only apply an *onset* ramp.
    # This is used, in particular, for the DPOAE stimulus in which we want
    # to ramp on a continuous tone and then play it continuously until we
    # acquire a sufficient number of epochs.
    if duration != np.inf:
        m_offset = (t >= (start_time+duration-rise_time)) & \
            (t < (start_time+duration))
        m_null_post = t >= (duration+start_time)
    else:
        m_offset = np.zeros_like(t, dtype=np.bool)
        m_null_post = np.zeros_like(t, dtype=np.bool)

    t_null_pre = t[m_null_pre]
    t_onset = t[m_onset] - start_time
    t_offset = t[m_offset] - start_time
    t_ss = t[~(m_null_pre | m_onset | m_offset | m_null_post)]
    t_null_post = t[m_null_post]

    f_null_pre = np.zeros(len(t_null_pre))
    f_lower = cos2ramp(t_onset, rise_time, 0)
    f_upper = cos2ramp(t_offset-(duration-rise_time), rise_time, np.pi/2)
    f_middle = np.ones(len(t_ss))
    f_null_post = np.zeros(len(t_null_post))

    concat = [f_null_pre, f_lower, f_middle, f_upper, f_null_post]
    return np.concatenate(concat, axis=-1)


def make_pink(fs, noise):
    n = len(noise)
    csd = np.fft.rfft(noise)
    i_reference = int(round(1000 * n / fs))
    i = np.arange(len(csd))
    phase = np.angle(csd)
    magnitude = np.abs(csd) * np.sqrt(i_reference / i)
    magnitude[0] = np.abs(csd[0])
    pink_csd = magnitude * (np.cos(phase) + 1j * np.sin(phase))
    return np.fft.irfft(pink_csd)
