from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray

from .dataset import Dataset
from .. import model
from .. import poles
from .. import stim


BASE_PATH = Path('/media/nutshell/work/OHSU/data/CMAP_data/Bramhall_FOIA_data')


CACHE_PATH = Path('/media/nutshell/work/OHSU/cache/bramhall_abr')
ABR_CACHE_PATH = CACHE_PATH / 'abr'
OAE_CACHE_PATH = CACHE_PATH / 'oae'
TRACE_CACHE_PATH = CACHE_PATH / 'trace'
ABR_CACHE_PATH.mkdir(parents=True, exist_ok=True)
OAE_CACHE_PATH.mkdir(parents=True, exist_ok=True)
TRACE_CACHE_PATH.mkdir(parents=True, exist_ok=True)


ABR_STIM = {
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
}


class BramhallABRDataset(Dataset):

    base_path = BASE_PATH

    def __init__(self):
        self.info = self.load_subject_info()
        tone_stim = self.generate_abr_stim()
        self.abr_stim_map = {f'{k}': i for i, k in enumerate(tone_stim)}

    def load_dpgram(self):
        dpgram_file = self.base_path / 'dpgram_data_65_55.csv'
        dpgram = pd.read_csv(dpgram_file, index_col=0)
        dpgram.columns = np.round(dpgram.columns.astype('f'), 2)
        dpgram.columns.name = 'f2'
        dpgram.index.name = 'subject'
        return dpgram

    def load_poles(self):
        return poles.compute_poles_dpgram_v2(self.load_dpgram())

    def generate_abr_stim(self, fs=100e3, total_duration=4e-3):
        tone_stim = {}
        for frequency, ti in ABR_STIM.items():
            for level in ti['levels']:
                tone_stim[frequency, level] = \
                    stim.generate_tone(fs=fs, frequency=frequency,
                                       level=level, duration=ti['duration'],
                                       rise_time=ti['rise_time'],
                                       total_duration=total_duration)

        return tone_stim

    def load_subject_info(self, synapse_prediction=False, abr=False):
        subject_file = self.base_path / 'subject_data.csv'
        subject_data = pd.read_csv(subject_file, index_col=0).set_index('SubjectID', verify_integrity=True)
        subject_data['i_sex'] = (subject_data['Gender'] == 'Female').astype('i')
        self.subject_map = {s: i for i, s in enumerate(sorted(subject_data.index.values))}
        self.subject_map['NH'] = 0
        subject_data['i_subj'] = self.map_subject(subject_data.index)
        subject_data['has_tinnitus'] = (subject_data['Tinnitus'] == 'Yes').astype('i')
        if synapse_prediction:
            pred = self.load_synapse_prediction()
            subject_data = subject_data.join(pred, on='i_subj')
        if abr:
            abr = self.load_abr()
            abr = abr.groupby(['frequency', 'level', 'i_subj'])['w1_amplitude'].mean()
            abr4k = abr[4, 110].rename('abr')
            abr_mean_110 = abr.unstack('frequency').loc[110].mean(axis=1).rename('abr_mean_110')
            subject_data = subject_data.join(abr4k, on='i_subj')
            subject_data = subject_data.join(abr_mean_110, on='i_subj')
        result = subject_data.set_index('i_subj', verify_integrity=True).sort_index()
        return result
        #return subject_data.reset_index()

    def load_synapse_prediction(self):
        trace = az.from_netcdf(TRACE_CACHE_PATH / 'trace.nc')

        tm_mean = trace.posterior.theta_i[..., 2].mean(dim=('chain', 'draw')).values
        tm_std = trace.posterior.theta_i[..., 2].std(dim=('chain', 'draw')).values
        s = np.exp(trace.posterior.theta_i[..., 2])
        sm_mean = s.mean(dim=('chain', 'draw')).values
        sm_std = s.std(dim=('chain', 'draw')).values
        s_hdi = az.hdi(s, hdi_prob=0.9).theta_i.values
        s_lb = s_hdi[:, 0]
        s_ub = s_hdi[:, 1]
        df = pd.DataFrame({
            'tm_mean': tm_mean,
            'tm_std': tm_std,
            'sm_mean': sm_mean,
            'sm_std': sm_std,
            's_lb': s_lb,
            's_ub': s_ub,
            's': sm_mean,
        })
        df.index.name = 'i_subj'
        return df

    def load_abr(self):
        abr_file = self.base_path / 'abr_data_replicates.csv'
        abr = pd.read_csv(abr_file)
        abr = abr.query('frequency != 1').copy()
        abr['i_subj'] = self.map_subject(abr['SubjectID'])
        freq = abr['frequency'].astype('i') * 1000
        level = abr['level'].astype('i')
        abr['i_stim'] = self.map_abr_stimulus(freq, level)
        return abr

    def map_subject(self, subject):
        subject = np.asarray(subject)
        return np.array([self.subject_map[s] for s in subject])

    def map_abr_stimulus(self, *keys):
        if len(keys) != 1:
            keys = [f'{k}' for k in zip(*keys)]
        else:
            keys = np.asarray(keys[0])
        return np.array([self.abr_stim_map[k] for k in keys])

    def load_simulated_abr(self, which='subject', tlb=0.7e-3, tub=1.4e-3):
        result = xarray.load_dataarray(ABR_CACHE_PATH / f'abr_power_{which}.nc')
        i_stim = self.map_abr_stimulus(result.stimulus)
        if np.any(np.diff(i_stim) != 1):
            raise ValueError('We have a problem with stim ordering')
        i_subj = self.map_subject(result.subject)
        if np.any(np.diff(i_subj) != 1):
            raise ValueError('We have a problem with subj ordering')
        return result.loc[..., tlb:tub]


def regenerate_abr_simulations(which='subject'):
    ds = BramhallABRDataset()
    abr_stim = ds.generate_abr_stim()
    if which == 'subject':
        subject_poles = ds.load_poles()
    elif which == 'NH':
        subject_poles = ds.load_nh_poles()
    else:
        raise ValueError(f'Unknown pole type requested "{which}"')

    model.model_an(abr_stim, subject_poles, ABR_CACHE_PATH)
    simulated_abr = model.load_modeled_an(abr_stim, subject_poles, ABR_CACHE_PATH)
    simulated_abr.to_netcdf(ABR_CACHE_PATH / f'abr_power_{which}.nc')
