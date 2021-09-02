from pathlib import Path

import numpy as np
import pandas as pd


from .dataset import Dataset
from .. import poles
from .. import stim


BASE_PATH = Path('/media/nutshell/work/OHSU/data/CMAP_data/Carcagno/2020-08-08_16-06_aging_synap_part_I/samuele_c-ageing_synap_part_i-17ca78e2b0ca/datasets')


stim_info = {
    (80, 'HP Noise'): {'click_level': 80, 'spectrum_level': 40},
    (105, 'HP Noise'): {'click_level': 105, 'spectrum_level': 65},
    (80, 'Quiet'): {'click_level': 80, 'spectrum_level': -400},
    (105, 'Quiet'): {'click_level': 105, 'spectrum_level': -400},
}


efr_info = {
    4000: {
        'levels': [90, 100, 110],
        'duration': 8 / 4000,
        'rise_time': 0.5e-3,
    },
}

def make_abr_stim(fs, stim_info, click_latency, which='all'):
    click = stim.filtered_click(100e3, stim_info['click_level'],
                                full_duration=30e-3, center=click_latency)
    noise = stim.broadband_noise(fs, stim_info['spectrum_level'], 30e-3)
    env = stim.cos2envelope(fs, 0, len(noise), 0, 5e-3, 30e-3)
    ramped_noise = env * noise
    pink_noise = stim.make_pink(fs, ramped_noise)
    pink_noise_hp = stim.fir_filter(fs, pink_noise, 3500, 8000,
                                    transition_width=0.2, n_taps=512,
                                    mode='same')
    if which == 'all':
        return pink_noise_hp + click
    elif which == 'noise':
        return pink_noise_hp
    elif which == 'click':
        return click


class CarcagnoDataset(Dataset):

    base_path = BASE_PATH

    def __init__(self):
        pass

    def load_abr(self):
        cols = ['cndLabel', 'stimLev', 'id', 'peakTroughAmp']
        abr = pd.read_csv(self.base_path / 'ABR.csv')
        abr = abr.query('wave == "I"') \
            .query('montage == "FzIpsiMast"')[cols] \
            .set_index(cols[:-1])['peakTroughAmp']
        return abr.unstack('stimLev').unstack('cndLabel')

    def _generate_abr_stim(self, fs, click_latency, which='all'):
        return {k: make_abr_stim(fs, v, click_latency, which) for k, v in stim_info.items()}

    def generate_abr_stim(self, fs=100e3):
        stim_sets = {}
        latencies = np.linspace(5e-3, 13e-3, 50, endpoint=False)
        for i, latency in enumerate(latencies):
            stim = self._generate_abr_stim(fs, latency)
            for k, v in stim.items():
                if (i == 0) or (k[1] != 'Quiet'):
                    stim_sets[latency, k] = v
        return stim_sets

    def load_audiogram(self):
        df = pd.read_csv(self.base_path / 'audio_by_freq_no_ear.csv', index_col='id')
        cols = [c for c in df if c.startswith('audio')]
        df = df.loc[:, cols]
        df.columns = [float(c[5:-3]) for c in cols]
        return df

    def load_poles(self):
        return poles.compute_poles_audiogram(self.load_audiogram())
