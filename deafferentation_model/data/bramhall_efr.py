from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import Dataset
from .. import poles
from .. import stim


BASE_PATH = Path('/media/nutshell/work/OHSU/data/CMAP_data/Bramhall_FOIA_data_EFR')


class BramhallEFRDataset(Dataset):

    base_path = BASE_PATH

    def __init__(self):
        self.info = self.load_subject_info()
        self.subject_map = {s: i for i, s in enumerate(sorted(self.info.index.values))}
        self.efr_depths = (0.4, 0.63, 1.0)
        self.abr_levels = (90, 100, 110)
        self.subjects_with_no_efr = {237, 257, 261, 268, 276, 293, 304, 332, 347, 356, 357}

    def load_dpgram(self):
        other = pd.read_csv(self.base_path / 'DP_audio_and_demographic_data.csv', index_col=0)
        dp = other.set_index(['SubjectID', 'F2'])[['2*F1-F2 Ldp', '2*F1-F2 Ndp']]
        return dp['2*F1-F2 Ldp'].unstack()

    def map_subject(self, subject):
        return np.array([self.subject_map[s] for s in subject])

    def map_efr_stimulus(self, depth):
        return np.array([self.efr_depths.index(d) for d in depth])

    def map_abr_stimulus(self, level):
        return np.array([self.abr_levels.index(l) for l in level])

    def load_subject_info(self):
        df = pd.read_csv(self.base_path / 'DP_audio_and_demographic_data.csv', index_col=0)
        cols = ['SubjectID', 'Gender', 'AgeAtScrn', 'Tinnitus', 'Noise Group', 'LENSQscore', 'Overall MOS weight']
        subject_info = df[cols].groupby('SubjectID').first()
        subject_info['i_sex'] = subject_info['Gender'].map({1: 0, 2: 1})
        return subject_info

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
        efr['i_subject'] = self.map_subject(efr['SubjectID'])
        efr['i_stimulus'] = self.map_efr_stimulus(efr['ModDepth'])
        return efr

    def load_abr(self):
        abr = pd.read_csv(self.base_path / 'ABR_data.csv', index_col=0)
        m = abr['SubjectID'].apply(lambda s: s not in self.subjects_with_no_efr)
        abr = abr.loc[m]
        abr['i_subject'] = self.map_subject(abr['SubjectID'])
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
