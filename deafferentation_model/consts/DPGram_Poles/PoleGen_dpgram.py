# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:54:46 2020

@author: Sarineh Keshishzadeh
"""
import numpy as np
import scipy.io
import math
from keras.models import load_model

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# load the mean of training data
val1 = scipy.io.loadmat('train_mean_val.mat')
mean_val = val1['train_mean_val']

# load the SD of training data
val2 = scipy.io.loadmat('train_sd_val.mat')
sd_val = val2['train_sd_val']

trained_model = load_model('dpgram_model') # load the trained NN

exp_dpgram = np.random.rand(5,11) # the experimentally measured dpgrams should come here,
#  for testing I generated random values as 5 subjects with dp amplitudes measured at 11 frequencies
# exp_dpgram is the experimental dpgram: 
# It is a matrix (the number of columns equals to 11), each row represents the subject 
# each column represents the measured dp amplitude at specific frequency, 
# the order of frequencies is: 996, 1265, 1582, 2003, 2519, 3175, 3996, 5039, 6351, 8003, 10078

exp_dpgram_shift = np.abs(exp_dpgram -np.max(exp_dpgram,axis=0))  # calculate the shift from the best dp amplitude at each frequency

exp_dpgram_shift_norm = (exp_dpgram_shift - mean_val)/sd_val # normalize with the mean and sd of the training data


predicted_poles = trained_model.predict (exp_dpgram_shift_norm)  # predicts the individual pole function
Poles = np.log(predicted_poles/(1-predicted_poles))
sigmoid = np.vectorize(sigmoid)
Poles= 0.036 + (sigmoid(Poles)*(0.2910-0.036))
#for i in range(np.size(exp_dpgram_shift_norm,axis=0)):
 #   Poles[i,:] = 0.036 + (sigmoid_poles(Poles[i,:])*(0.2910-0.036))
 