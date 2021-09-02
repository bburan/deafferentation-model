# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 20:24:37 2020

@author: Sarineh
"""

import numpy as np
import scipy.io
from keras.models import load_model
from keras import backend as K



# load the mean of training data
val1 = scipy.io.loadmat('train_mean_val_ver2.mat')
mean_val = val1['train_mean_val_ver2']

# load the SD of training data
val2 = scipy.io.loadmat('train_sd_val_ver2.mat')
sd_val = val2['train_sd_val_ver2']

# load flat1 dpgram amplitudes (simulated)
val3 = scipy.io.loadmat('dpgram_Flat1.mat')
best_flat = np.transpose(val3['dpgram'])

# load NH dpgram amplitudes (simulated)
val4 = scipy.io.loadmat('dpgram_Normal.mat')
NH = np.transpose(val4['dpgram'])


exp_dpgram = np.random.rand(5,11) # the experimentally measured dpgrams should come here,
#  To test, I generated random values as 5 subjects with dp amplitudes measured at 11 frequencies
# exp_dpgram is the experimental dpgram:
# It is a matrix (the number of columns equals to 11), each row represents the subject
# each column represents the measured dp amplitude at specific frequency,
# the order of frequencies is: 996, 1265, 1582, 2003, 2519, 3175, 3996, 5039, 6351, 8003, 10078


# generate NH subject: in the previous version we matched Flat1 profile
# to the best NH subject in the experiment by defining relative measures. Here we generated a NH subject as follows:
# 1) calculate the dpgram amplitude shift of NH profile from the flat1 (NH_shift)
# 2) determine the best dpgram amplitudes across the measured f2 primaries (np.max(exp_dpgram,axis=0))
# 3) subtract the "NH_shift" from the best experimental dpgram amplitudes to create a NH subject
# 4) match generated NH subject to the simulated NH dpgram (exp_dpgram_new)

# In this way, we can either directly compare the simulated and experimental dpgrams or subtract
# the NH dpgram afterwards and compare the dpgram shifts (similar to the previous approach, but with a better match)


# calculate the dpgram shift of NH from the best flat
NH_shift = best_flat - NH

exp_NH = np.max(exp_dpgram,axis=0) - NH_shift

exp_sim_shift = NH - exp_NH

exp_dpgram_new = exp_dpgram + exp_sim_shift

exp_dpgram_new_norm = (exp_dpgram_new - mean_val)/sd_val # normalize with the mean and sd of the training data

# load the trained NN

def custom_activation(x):
    return (0.036+K.sigmoid(x) * (0.302-0.036))

trained_model = load_model('dpgram_model_ver2', custom_objects={'custom_activation': custom_activation})
Poles = trained_model.predict (exp_dpgram_new_norm)  # predicts the individual pole function
