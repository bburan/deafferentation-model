# Yeah, totally a hack.
from pathlib import Path
import pickle

from joblib import Parallel, delayed
import numpy as np
from scipy import signal
from xarray import DataArray

#!!!! HACK ALERT!!!!
path = str(Path(__file__).parent.parent / 'Verhulstetal2018Model')
import sys
sys.path.append(path)
import inner_hair_cell2018 as ihc
import auditory_nerve2018 as anf
import ic_cn2018 as nuclei
from cochlear_model2018 import cochlea_model

from . import util


def get_efr(an, reference=1):
    cn = nuclei.cochlearNuclei(an.values, an.attrs['fs'])
    ic = nuclei.inferiorColliculus(cn, an.attrs['fs'])
    w1 = nuclei.M1 * an
    w3 = nuclei.M3 * cn
    w5 = nuclei.M5 * ic
    efr = w1 + w3 + w5
    return DataArray(efr / reference, coords=an.coords, dims=an.dims, attrs=an.attrs, name='efr')


def get_efr_power(efr, fm=110, n_trim=1, detrend=None, window='hanning'):
    i = int(round(n_trim/fm * efr.fs))
    power = util.tone_power_conv(efr[..., i:], efr.fs, fm, detrend=detrend, window=window)
    coords = dict(efr.coords.items())
    coords.pop('time')
    return DataArray(power, coords=coords, dims=efr.dims[:-1], name='efr_power')


def get_abr(fs, model, resample=15):
    magic_constant = 0.118
    vm = ihc.inner_hair_cell_potential(model.Vsolution * magic_constant, fs)
    if resample is not None:
        vm_resampled = signal.decimate(vm, resample, axis=0)
        vm_resampled[0:5,:] = vm[0,0] #resting value to eliminate noise from decimate
        vm = vm_resampled
        fs = fs / resample

    t = np.arange(vm.shape[0])/fs

    result = {
        't': t,
        'cf': model.cf,
        'fs': fs,
        'HSR': anf.auditory_nerve_fiber(vm, fs, 2) * fs,
        'MSR': anf.auditory_nerve_fiber(vm, fs, 1) * fs,
        'LSR': anf.auditory_nerve_fiber(vm, fs, 0) * fs,
    }
    return result


def _model_abr(cache_file=None, **model_kwargs):
    model = cochlea_model()
    model.init_model(**model_kwargs)
    model.solve()
    result = get_abr(100e3, model)
    if cache_file is not None:
        with cache_file.open('wb') as fh:
            pickle.dump(result, fh)
    else:
        return result


def _concat_results(results, stim, subjects):
    rates = []
    time = results[0]['t']
    cf = results[0]['cf']
    n_subjects = len(subjects)
    n_stim = len(stim)
    n_time = len(time)
    n_cf = len(cf)
    for rate in ['LSR', 'MSR', 'HSR']:
        x = np.concatenate([r[rate].T[np.newaxis] for r in results])
        x.shape = (1, n_stim, n_subjects, n_cf, n_time)
        rates.append(x)
    stim = [f'{s}' for s in stim]
    return DataArray(
        np.concatenate(rates),
        dims=['SR', 'stimulus', 'subject', 'CF', 'time'],
        coords={
            'SR': ['LSR', 'MSR', 'HSR'],
            'stimulus': stim,
            'subject': subjects,
            'CF': cf,
            'time': time,
        },
        name='anf',
    )


def model_an(stim_waveforms, subject_poles, cache_path):
    '''
    Returns
    -------
    results : dict
    '''
    model_kwargs = dict(
        samplerate=100e3,
        sections=1000,
        probe_freq='abr',
        Zweig_irregularities=0,
        IrrPct=0,
        non_linearity_type='vel',
        subject=1,
    )

    jobs = []
    stim_keys = []
    subject_keys = []
    for stim_key, stim in stim_waveforms.items():
        for subject_key, poles in subject_poles.iterrows():
            cache_file = cache_path / f'{subject_key}_{stim_key}.pkl'
            if cache_file.exists():
                continue
            job = delayed(_model_abr)(stim=stim, sheraPo=poles,
                                      cache_file=cache_file, **model_kwargs)
            jobs.append(job)

    print(f'Queued {len(jobs)} jobs')
    Parallel(n_jobs=12, verbose=50)(jobs)
    print('Done!')


def load_modeled_an(stim_waveforms, subject_poles, cache_path):
    results = []
    keys = []
    subjects = []
    for key, _ in stim_waveforms.items():
        keys.append(key)
        subjects = []
        for subject, _ in subject_poles.iterrows():
            subjects.append(subject)
            cache_file = cache_path / f'{subject}_{key}.pkl'
            with cache_file.open('rb') as fh:
                results.append(pickle.load(fh))
    result = _concat_results(results, keys, subjects)
    result.attrs['fs'] = np.mean(np.diff(result.time)**-1)
    return result
