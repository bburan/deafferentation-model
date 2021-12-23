from pathlib import Path

import numpy as np
import pandas as pd
import xarray

from .dataset import Dataset
from .. import model
from .. import poles
from .. import stim


BASE_PATH = Path('/media/nutshell/work/OHSU/data/CMAP_data/Bramhall_FOIA_data_EFR')
CACHE_PATH = Path('/media/nutshell/work/OHSU/cache/bramhall_efr')
EFR_CACHE_PATH = CACHE_PATH / 'efr'
ABR_CACHE_PATH = CACHE_PATH / 'abr'
TRACE_CACHE_PATH = CACHE_PATH / 'trace'
ABR_CACHE_PATH.mkdir(parents=True, exist_ok=True)
EFR_CACHE_PATH.mkdir(parents=True, exist_ok=True)
TRACE_CACHE_PATH.mkdir(parents=True, exist_ok=True)


class BramhallEFRDataset(Dataset):

    base_path = BASE_PATH

    def __init__(self):
        self.info = self.load_subject_info()
        self.efr_depths = (0.4, 0.63, 1.0)
        self.abr_levels = (90, 100, 110)
        self.subjects_with_no_efr = {237, 257, 261, 268, 276, 293, 304, 332, 347, 356, 357}

    def load_dpgram(self):
        other = pd.read_csv(self.base_path / 'DP_audio_and_demographic_data.csv', index_col=0)
        dp = other.set_index(['SubjectID', 'F2'])[['2*F1-F2 Ldp', '2*F1-F2 Ndp']]
        return dp['2*F1-F2 Ldp'].unstack()

    def map_subject(self, subject):
        subject = np.asarray(subject)
        return np.array([self.subject_map[s] for s in subject])

    def map_efr_stimulus(self, depth):
        depth = np.asarray(depth).astype('double')
        return np.array([self.efr_depths.index(d) for d in depth])

    def map_abr_stimulus(self, level):
        level = np.asarray(level).astype('i')
        return np.array([self.abr_levels.index(l) for l in level])

    def load_subject_info(self):
        df = pd.read_csv(self.base_path / 'DP_audio_and_demographic_data.csv', index_col=0)
        cols = ['SubjectID', 'Gender', 'AgeAtScrn', 'Tinnitus', 'Noise Group', 'LENSQscore', 'Overall MOS weight']
        subject_info = df[cols].groupby('SubjectID').first()
        subject_info['i_sex'] = subject_info['Gender'].map({1: 0, 2: 1})
        self.subject_map = {s: i for i, s in enumerate(sorted(subject_info.index.values))}
        subject_info['i_subj'] = self.map_subject(subject_info.index)
        subject_info['has_tinnitus'] = (subject_info['Tinnitus'] - 1).astype('i')
        return subject_info.reset_index()

    def load_poles(self):
        return poles.compute_poles_dpgram_v2(self.load_dpgram())

    def load_efr(self):
        efr = pd.read_csv(self.base_path / 'EFR_data.csv', index_col=0)
        efr['ModDepth'] = efr['ModDepth'].map({0: 1, 4: 0.63, 8: 0.4})
        efr = efr.set_index(['SubjectID', 'Fmod', 'Fcar', 'ModDepth'])
        efr = 20*np.log10(efr)
        efr_nf = 0.5 * efr.iloc[:, :5].mean(axis=1) + 0.5 * efr.iloc[:, 6:].mean(axis=1)
        efr = efr.iloc[:, 5] - efr_nf
        efr = efr.rename('efr_power').reset_index()
        efr['i_subj'] = self.map_subject(efr['SubjectID'])
        efr['i_stim'] = self.map_efr_stimulus(efr['ModDepth'])
        return efr

    def load_abr(self):
        abr = pd.read_csv(self.base_path / 'ABR_data.csv', index_col=0)
        m = abr['SubjectID'].apply(lambda s: s not in self.subjects_with_no_efr)
        abr = abr.loc[m]
        abr['i_subj'] = self.map_subject(abr['SubjectID'])
        abr['i_stim'] = self.map_abr_stimulus(abr['level'])
        return abr

    def generate_efr_stim(self, fs=100e3, actual_duration=200e-3, rise_time=25e-3):
        level = 80
        fm = 110
        fc = 4e3
        stim_duration = 200e-3
        return {d: stim.generate_efr(fs, fc, fm, d, level, stim_duration, rise_time, actual_duration)[-1] for d in self.efr_depths}

    def generate_abr_stim(self, fs=100e3, actual_duration=200e-3, rise_time=25e-3):
        frequency = 4000
        duration = 8 / frequency
        rise_time = 0.5e-3
        return {l: stim.generate_tone(fs, frequency, l, duration, rise_time) for l in self.abr_levels}

    def load_simulated_abr(self, tlb=0.7e-3, tub=1.4e-3):
        result = xarray.load_dataarray(ABR_CACHE_PATH / 'abr_power.nc')
        i_stim = self.map_abr_stimulus(result.stimulus)
        if np.any(np.diff(i_stim) != 1):
            raise ValueError('We have a problem with stim ordering')
        i_subj = self.map_subject(result.subject)
        if np.any(np.diff(i_subj) != 1):
            raise ValueError('We have a problem with subj ordering')
        return result.loc[..., tlb:tub]

    def load_simulated_efr(self):
        result = xarray.load_dataarray(EFR_CACHE_PATH / 'efr_power.nc')
        i_stim = self.map_efr_stimulus(result.stimulus)
        if np.any(np.diff(i_stim) != 1):
            raise ValueError('We have a problem with stim ordering')
        i_subj = self.map_subject(result.subject)
        if np.any(np.diff(i_subj) != 1):
            raise ValueError('We have a problem with subj ordering')
        return result


def regenerate_efr_simulations():
    ds = BramhallEFRDataset()
    subject_poles = ds.load_poles()
    efr_stim = ds.generate_efr_stim(actual_duration=3/110, rise_time=0)

    model.model_an(efr_stim, subject_poles, EFR_CACHE_PATH)
    an_efr = model.load_modeled_an(efr_stim, subject_poles, EFR_CACHE_PATH)
    efr = model.get_efr(an_efr, reference=1e-9)
    simulated_efr = model.get_efr_power(efr)
    simulated_efr.to_netcdf(EFR_CACHE_PATH / 'efr_power.nc')


def regenerate_abr_simulations():
    ds = BramhallEFRDataset()
    subject_poles = ds.load_poles()
    abr_stim = ds.generate_abr_stim()

    model.model_an(abr_stim, subject_poles, ABR_CACHE_PATH)
    simulated_abr = model.load_modeled_an(abr_stim, subject_poles, ABR_CACHE_PATH)
    simulated_abr.to_netcdf(ABR_CACHE_PATH / 'abr_power.nc')
