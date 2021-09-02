from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io.matlab import loadmat

from keras.models import load_model
from keras import backend as K


CONSTANTS_PATH = Path(__file__).parent / 'consts'


def load_nh_poles():
    pole_path = CONSTANTS_PATH / 'StartingPoles_NH.dat'
    return np.loadtxt(pole_path)


def _load_pole_constants():
    constants_path = CONSTANTS_PATH / 'VerhulstModel_poles'
    consts = loadmat(constants_path / 'cf.mat')
    cf = consts['cf'].ravel()
    consts = loadmat(constants_path / 'PoleTrajs.mat')
    SMax = consts['SMax']
    nh_starting_poles = np.loadtxt(constants_path / 'StartingPoles_NH.dat')
    nh_poles = nh_starting_poles[1:]
    return cf, nh_poles, SMax


def _compute_poles_audiogram(audiogram):
    cf, nh_poles, SMax = _load_pole_constants()
    poles = np.arange(0.036, 0.3025, 0.001)
    audiogram_interp = np.interp(cf, audiogram.index.values, audiogram.values)

    i = np.interp(nh_poles, poles, np.arange(len(poles)))
    i = np.round(i).astype('i')
    GainDiff = SMax[i, range(1000)] - SMax

    m = GainDiff > audiogram_interp
    i = np.argmax(m, axis=0)
    i[~m.any(axis=0)] = -1
    hi_poles = poles[i]
    hi_poles = np.pad(hi_poles, (1, 0), 'edge')
    return hi_poles


def compute_poles_audiogram(audiograms):
    return audiograms.apply(_compute_poles_audiogram, axis=1, result_type='expand')


def compute_poles_dpgram_v1(dpgrams):
    # load the mean of training data
    val1 = scipy.io.loadmat(CONSTANTS_PATH / 'DPGram_Poles' / 'train_mean_val.mat')
    mean_val = val1['train_mean_val']

    # load the SD of training data
    val2 = scipy.io.loadmat(CONSTANTS_PATH / 'DPGram_Poles' / 'train_sd_val.mat')
    sd_val = val2['train_sd_val']

    trained_model = load_model(CONSTANTS_PATH / 'DPGram_Poles' / 'dpgram_model') # load the trained NN

    dpgram_shift = dpgrams.max(axis=0) - dpgrams
    dpgram_shift_norm = (dpgram_shift - mean_val) / sd_val

    predicted_poles = trained_model.predict(dpgram_shift_norm.values)
    poles = np.log(predicted_poles/(1-predicted_poles))
    poles = 1 / (1 + np.exp(-poles))
    poles = 0.036 + poles * (0.2910-0.036)
    poles = pd.DataFrame(poles, index=dpgrams.index)
    return poles


def compute_poles_dpgram_v2(dpgrams):
    mean_val = loadmat(CONSTANTS_PATH / 'DPGram_Poles_ver2_modified' / 'train_mean_val_ver2.mat')['train_mean_val_ver2']

    # load the SD of training data
    sd_val = loadmat(CONSTANTS_PATH / 'DPGram_Poles_ver2_modified' / 'train_sd_val_ver2.mat')['train_sd_val_ver2']

    # load flat1 dpgram amplitudes (simulated)
    val3 = loadmat(CONSTANTS_PATH / 'DPGram_Poles_ver2_modified' / 'dpgram_Flat1.mat')
    best_flat = val3['dpgram'].T.ravel()

    # load NH dpgram amplitudes (simulated)
    val4 = loadmat(CONSTANTS_PATH / 'DPGram_Poles_ver2_modified' / 'dpgram_Normal.mat')
    NH = val4['dpgram'].T.ravel()

    #plt.plot(f2_frequency, best_flat, label='Best')
    #plt.plot(f2_frequency, NH, label='NH')
    #plt.legend()

    NH_shift = best_flat - NH
    exp_NH = dpgrams.max().values - NH_shift
    exp_sim_shift = NH - exp_NH

    dpgrams_shift = dpgrams + exp_sim_shift
    dpgrams_norm = (dpgrams_shift - mean_val) / sd_val

    def custom_activation(x):
        return (0.036+K.sigmoid(x) * (0.302-0.036))

    trained_model = load_model(CONSTANTS_PATH / 'DPGram_Poles_ver2_modified' / 'dpgram_model_ver2',
                               custom_objects={'custom_activation': custom_activation})

    nn_poles = trained_model.predict(dpgrams_norm)  # predicts the individual pole function
    nn_poles = pd.DataFrame(nn_poles, index=dpgrams.index)
    return nn_poles
