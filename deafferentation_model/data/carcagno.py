from pathlib import Path
import pickle

import arviz as az
import numpy as np
import pandas as pd
import xarray

from .dataset import Dataset
from .. import model
from .. import poles
from .. import stim


BASE_PATH = Path('/media/nutshell/work/OHSU/data/CMAP_data/Carcagno/2020-08-08_aging_synap_part_I/datasets')
BASE_PATH_PSYCHO = Path('/media/nutshell/work/OHSU/data/CMAP_data/Carcagno/2020-09-14_aging_synap_part_II/datasets')

CACHE_PATH = Path('/media/nutshell/work/OHSU/cache/carcagno')
EFR_CACHE_PATH = CACHE_PATH / 'efr'
ABR_CACHE_PATH = CACHE_PATH / 'abr'
MEAN_ABR_CACHE_PATH = CACHE_PATH / 'mean_abr'
TRACE_CACHE_PATH = CACHE_PATH / 'trace'

ABR_CACHE_PATH.mkdir(parents=True, exist_ok=True)
MEAN_ABR_CACHE_PATH.mkdir(parents=True, exist_ok=True)
EFR_CACHE_PATH.mkdir(parents=True, exist_ok=True)
TRACE_CACHE_PATH.mkdir(parents=True, exist_ok=True)


STIM_INFO = {
    (80, 'HP Noise'): {'click_level': 80, 'spectrum_level': 40},
    (105, 'HP Noise'): {'click_level': 105, 'spectrum_level': 65},
    (80, 'Quiet'): {'click_level': 80, 'spectrum_level': -400},
    (105, 'Quiet'): {'click_level': 105, 'spectrum_level': -400},
}


EFR_INFO = {
    4000: {
        'levels': [90, 100, 110],
        'duration': 8 / 4000,
        'rise_time': 0.5e-3,
    },
}

def make_abr_stim(fs, stim_info, click_latency, which='all'):
    duration = 20e-3
    trim = 1e-3

    # This code is designed to parallel the Carcagno approach (based on code
    # provided by Carcagno) and has been verified to match the output of the
    # original Carcagno code, point-by-point, for clicks and the overall RMS
    # and frequency spectrum for noise.
    noise = stim.broadband_noise(fs, stim_info['spectrum_level'],
                                 duration + 60e-3)
    pink = stim.make_pink(fs, noise)
    filtered = stim.fir_filter(fs, pink, 3500, 8000, transition_width=0.2,
                               n_taps=512)
    ilb = int(round(30e-3 * fs))
    iub = int(round((30e-3 + 20e-3) * fs))

    # Now, trim down the noise to the desired window. I assume the extra
    # samples are mainly to allow for filtering and pink noise conversion.
    filtered = filtered[ilb:iub]
    env = stim.cos2envelope(fs, 0, len(filtered), 0, 5e-3, 20e-3)
    ramped_noise = filtered * env

    # Now generate the click and then filter again (?). This is consistent with
    # the Carcagno code.
    click = stim.filtered_click(fs,
                                stim_info['click_level'],
                                full_duration=duration,
                                fl=350,
                                fh=8000,
                                polarity='rarefaction')
    click = stim.fir_filter(fs, click, 350, 3000, 1024, 0.1, 'same')

    # Now, window the click (again, consistent with the Carcagno code.
    n_trim = int(np.round(trim * fs))
    i = np.abs(click).argmax()
    click_trimmed = click[i-n_trim:i+n_trim]

    # Now, figure out where to put it.
    click_location = int(np.round(click_latency * fs))
    click = np.zeros_like(ramped_noise)
    click[click_location-n_trim:click_location+n_trim] = click_trimmed

    if which == 'all':
        return ramped_noise + click
    elif which == 'noise':
        return ramped_noise
    elif which == 'click':
        return click


class CarcagnoDataset(Dataset):

    base_path = BASE_PATH
    base_path_psycho = BASE_PATH_PSYCHO

    def __init__(self):
        self.abr_stim_map = {f'{k}': i for i, k in enumerate(STIM_INFO.keys())}
        audiogram = self.load_audiogram()
        self.subj_map = {k: i for i, (k, _) in enumerate(audiogram.iterrows())}
        self.subj_map['NH'] = 0

    def load_abr(self):
        cols = ['cndLabel', 'stimLev', 'id', 'peakTroughAmp']
        abr = pd.read_csv(self.base_path / 'ABR.csv')
        abr = abr.query('wave == "I"') \
            .query('montage == "FzIpsiMast"')[cols] \
            .set_index(cols[:-1])['peakTroughAmp'].rename('w1_amplitude') \
            .reset_index().copy()
        abr['i_stim'] = self.map_abr_stimulus(abr['stimLev'], abr['cndLabel'])
        abr['i_subj'] = self.map_subject(abr['id'])
        return abr.dropna()

    def _generate_abr_stim(self, fs, click_latency, which='all'):
        return {k: make_abr_stim(fs, v, click_latency, which) for k, v in STIM_INFO.items()}

    def generate_abr_stim(self, fs=100e3, n_latencies=17, latency_cutoff=np.inf):
        stim_sets = {}
        latencies = np.linspace(5e-3, 13e-3, n_latencies, endpoint=True)
        latencies = latencies[latencies < latency_cutoff]
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
        audiogram = self.load_audiogram()
        audiogram.columns *= 1e3
        return poles.compute_poles_audiogram(audiogram)

    def map_subject(self, subject):
        subject = np.asarray(subject)
        return np.array([self.subj_map[s] for s in subject])

    def map_abr_stimulus(self, *keys):
        if len(keys) != 1:
            keys = [f'{k}' for k in zip(*keys)]
        else:
            keys = np.asarray(keys[0])
        return np.array([self.abr_stim_map[k] for k in keys])

    def check_response(self, x):
        i_stim = self.map_abr_stimulus(x.stimulus)
        if np.any(np.diff(i_stim) != 1):
            raise ValueError('We have a problem with stim ordering')
        i_subj = self.map_subject(x.subject)
        if np.any(np.diff(i_subj) != 1):
            raise ValueError('We have a problem with subj ordering')

    def load_subject_info(self, cog=False, synapse_prediction=False, abr=False, abr_scale='uv'):
        subject_file = self.base_path / 'demographic.csv'
        subject_data = pd.read_csv(subject_file)
        subject_data['i_sex'] = (subject_data['gender'] == 'F').astype('i')
        subject_data['i_subj'] = self.map_subject(subject_data['id'])
        if cog:
            cog_info = self.load_cog()
            subject_data = subject_data.join(cog_info, on='i_subj')
        if synapse_prediction:
            pred = self.load_synapse_prediction()
            subject_data = subject_data.join(pred, on='i_subj')
        if abr:
            abr = self.load_abr().query('stimLev == 105').query('cndLabel == "Quiet"')
            abr = abr.set_index('i_subj', verify_integrity=True)['w1_amplitude'].rename('abr')
            if abr_scale == 'uv':
                abr *= 1e-3
            else:
                raise ValueError(f'Unrecognized ABR scale "{abr_scale}"')
            subject_data = subject_data.join(abr, on='i_subj')
        result = subject_data.set_index('i_subj', verify_integrity=True).sort_index()
        return result

    def load_simulated_abr(self, which='subject', tlb=3.15e-3, tub=4.35e-3):
        result = xarray.load_dataarray(MEAN_ABR_CACHE_PATH / f'mean_abr_{which}.nc')
        i_stim = self.map_abr_stimulus(result.stimulus)
        if np.any(np.diff(i_stim) != 1):
            raise ValueError('We have a problem with stim ordering')
        i_subj = self.map_subject(result.subject)
        if np.any(np.diff(i_subj) != 1):
            raise ValueError('We have a problem with subj ordering')
        return result.loc[..., tlb:tub]

    def load_sam_thresholds(self):
        am_file = self.base_path_psycho / 'AM.csv'
        am_data = pd.read_csv(am_file)
        am_data['i_subj'] = self.map_subject(am_data['id'])
        return am_data

    def load_ssq(self):
        ssq_file = self.base_path_psycho / 'SSQ.csv'
        ssq_data = pd.read_csv(ssq_file)
        ssq_data['i_subj'] = self.map_subject(ssq_data['id'])
        return ssq_data

    def load_dtt(self):
        dtt_file = self.base_path_psycho / 'DTT.csv'
        dtt_data = pd.read_csv(dtt_file)
        dtt_data['i_subj'] = self.map_subject(dtt_data['id'])
        dtt_data['i_level'] = (dtt_data['level'] == 'high').astype('i')
        return dtt_data

    def load_crm(self):
        crm_file = self.base_path_psycho / 'CRM.csv'
        crm_data = pd.read_csv(crm_file)
        crm_data['i_subj'] = self.map_subject(crm_data['id'])
        return crm_data

    def load_cog(self):
        cols = ['id', 'zLongestSpanForw', 'zLongestSpanBackw', 'zRavenScore',
                'zReadSpanRecalledScore']
        cog = pd.read_csv(self.base_path_psycho / 'd_cog.csv')[cols]
        cog['i_subj'] = self.map_subject(cog['id'])
        del cog['id']
        return cog.set_index('i_subj', verify_integrity=True)

    def load_synapse_prediction(self):
        trace = az.from_netcdf(TRACE_CACHE_PATH / 'trace_subject_parabola.nc')
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


def average_stim_response(stim_response, stim_key):
    # Average single stimulus response (e.g., 80 dB HP noise) for a single
    # subject.
    fs_resample = 100e3 / 15
    lb, ub = -20, 25
    trimmed_response = []
    for i, latency in enumerate(stim_response.stimulus):
        latency = float(latency)
        index = int(round(latency * fs_resample))
        stim_trimmed = stim_response[:, i, :, :, index+lb:index+ub]
        new_time = np.arange(ub-lb) / fs_resample
        stim_trimmed = stim_trimmed.assign_coords(time=new_time)
        trimmed_response.append(stim_trimmed)

    stim_response_trimmed = xarray.concat(trimmed_response, dim='stimulus')
    stim_response_mean = stim_response_trimmed.mean('stimulus')
    stim_response_mean = stim_response_mean.assign_coords(stimulus=f'{stim_key}')
    return stim_response_mean


def average_subject_response(subject, abr_stim_sets):
    # Average all stimulus responses (but create one average per stim) for a
    # single subject.
    averaged_responses = []
    for stim_key, latencies in abr_stim_sets.items():
        results = []
        for l in latencies:
            key = (l, stim_key)
            cache_file  = ABR_CACHE_PATH / f'{subject}_{key}.pkl'
            with cache_file.open('rb') as fh:
                results.append(pickle.load(fh))

        stim_response = model._concat_results(results, latencies, [subject])
        stim_response_mean = average_stim_response(stim_response, stim_key)
        cache_file = ABR_CACHE_PATH / F'{subject}_{stim_key}.nc'
        stim_response_mean.to_netcdf(cache_file)
        averaged_responses.append(stim_response_mean)

    result = xarray.concat(averaged_responses, dim='stimulus')
    result = result.transpose('SR', 'stimulus', ...)
    result.to_netcdf(MEAN_ABR_CACHE_PATH / f'{subject}.nc')


# CODE TO REGEN ABR SIM
def regenerate_abr_simulations(which='subject', n_jobs=12):
    ds = CarcagnoDataset()
    abr_stim = ds.generate_abr_stim()
    if which == 'subject':
        subject_poles = ds.load_poles()
    elif which == 'NH':
        subject_poles = ds.load_nh_poles()
    else:
        raise ValueError(f'Unknown pole type requested "{which}"')
    model.model_an(abr_stim, subject_poles, ABR_CACHE_PATH, n_jobs=n_jobs)


def average_abr_simulations(which='subject'):
    ds = CarcagnoDataset()
    abr_stim = ds.generate_abr_stim()
    if which == 'subject':
        subject_poles = ds.load_poles()
    elif which == 'NH':
        subject_poles = ds.load_nh_poles()
    else:
        raise ValueError(f'Unknown pole type requested "{which}"')

    abr_stim_sets = {}
    for latency, stim in abr_stim.keys():
        abr_stim_sets.setdefault(stim, []).append(latency)
    for subject, _ in subject_poles.iterrows():
        average_subject_response(subject, abr_stim_sets)

    abrs = []
    for subject, _ in subject_poles.iterrows():
        a = xarray.load_dataarray(MEAN_ABR_CACHE_PATH / f'{subject}.nc')
        abrs.append(a)
    simulated_abr = xarray.concat(abrs, dim='subject')
    simulated_abr.to_netcdf(MEAN_ABR_CACHE_PATH / f'mean_abr_{which}.nc')


def verify_stim():
    import matplotlib.pyplot as plt

    fs = 100e3

    click_level = 80
    spectrum_level = 40
    c = make_abr_stim(fs, {'click_level': click_level, 'spectrum_level': spectrum_level}, 1e-3, 'click')
    n = make_abr_stim(fs, {'click_level': click_level, 'spectrum_level': spectrum_level}, 5e-3, 'noise')
    s = make_abr_stim(fs, {'click_level': click_level, 'spectrum_level': spectrum_level}, 5e-3)

    c = c[:int(fs*2e-3)]
    plt.plot(n)
    plt.plot(c)

    plt.figure()
    psd = util.db(util.psd_df(c, fs), 20e-6)
    plt.plot(psd)
    psd = util.db(util.psd_df(n, fs), 20e-6)
    plt.plot(psd)
    plt.axis(xmin=0, xmax=20e3, ymin=-50)

    c_actual = np.load(example_files / 'click_filtered_lo_40_rare_.npy')[:, 1]
    psd = util.db(util.psd_df(c_actual, 48e3), 20e-6)
    plt.plot(psd)
    n_actual = np.load(example_files / 'noise_filtered_lo_40_rare_.npy')[:, 1]
    psd = util.db(util.psd_df(n_actual, 48e3), 20e-6)
    plt.plot(psd)

    #plt.axhline(65)
    #plt.axvline(8000)
    #plt.axvline(3500)
    #plt.axvline(350)
