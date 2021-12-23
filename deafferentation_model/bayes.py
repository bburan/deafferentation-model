import pickle

import numpy as np
import pymc3 as mc

from .util import inv_greenwood


sr_dist = np.array([0.12, 0.15, 0.73])[:, np.newaxis, np.newaxis, np.newaxis]
cum_sr_dist = np.array([1.0, 0.88, 0.73])[:, np.newaxis, np.newaxis, np.newaxis]
with open('results/wu_priors_greenwood_19_subjects.pkl', 'rb') as fh:
    priors = pickle.load(fh)
with open('results/wu_priors_greenwood_age_19_subjects.pkl', 'rb') as fh:
    age_priors = pickle.load(fh)


def create_model(cf, n_subjects, i_sex, abr_sim=None, i_abr_stim=None,
                 i_abr_subj=None, abr_obs=None, efr_sim=None, i_efr_stim=None,
                 i_efr_subj=None, efr_obs=None, a_0_mean=-10, a_0_std=1,
                 a_1_mean=0, a_1_std=1, synaptogram_shape='parabola'):
    '''
    Parameters
    ----------
    cf : array
        Frequencies of the modeled auditory nerve fibers.
    n_subjects : int
        Number of subjects included in the data.
    i_sex : array (int)
        Indicator variable for sex of subject i. Male = 0, female = 1.
    abr_sim : array
        5-dimensional array representing the predicted firing rate (over time)
        of a low, medium or high spontaneous rate auditory nerve fiber of a
        particular characteristic frequency to ABR stimulus j for subject i.
        Dimensions are spontaneous rate x stimulus x subject x characteristic
        frequency x time.
    i_abr_stim : array (int)
        Index of the ABR stimulus for observation n.
    i_abr_subj : array (int)
        Index of the subject for observation n.
    abr_obs : array
        Array containing the ABR observations in long format.
    '''
    x = inv_greenwood(cf)
    include_efr = efr_sim is not None
    include_abr = abr_sim is not None

    with mc.Model() as model:

        if synaptogram_shape == 'parabola':
            theta_nh_i = age_priors['theta_mu']
            ANFperIHC_nh = np.exp(-np.exp(theta_nh_i[0]) * (x - theta_nh_i[1])**2 + theta_nh_i[2])

            theta = mc.Normal('theta', priors['theta_mu'], priors['theta_sd'], shape=3)
            theta_i = mc.MvNormal('theta_i', theta, cov=priors['cov'], shape=(n_subjects, 3))[:, :, np.newaxis]
            ANFperIHC = np.exp(-np.exp(theta_i[:, 0]) * (x - theta_i[:, 1])**2 + theta_i[:, 2])

        elif synaptogram_shape == 'flat':
            theta_nh_i = age_priors['theta_mu']
            ANFperIHC_nh = np.exp(theta_nh_i[2])

            theta = mc.Normal('theta', priors['theta_mu'][2], priors['theta_sd'][2])
            theta_i_sd = mc.HalfCauchy('theta_i_sigma', 1)
            theta_i = mc.Normal('theta_i', theta, sd=theta_i_sd, shape=n_subjects)[:, np.newaxis]
            ANFperIHC = np.exp(theta_i)

        nh_count = ANFperIHC_nh * sr_dist
        cum_nh_count = ANFperIHC_nh * cum_sr_dist
        i_count = (nh_count - (cum_nh_count - ANFperIHC).clip(0, np.inf)).clip(0, np.inf)

        ########################################################################
        # EFR
        ########################################################################
        # Sex-specific scaling factors for EFR
        if include_efr:
            e_0 = mc.Normal('e_0', 0, 10)
            e_1 = mc.Normal('e_1', 0, 1)
            e_i = np.exp(e_0 + e_1 * i_sex)[:, np.newaxis]

            efr_scaled = e_i * i_count * efr_sim
            efr_power = efr_scaled.sum(axis=(0, -1))
            efr_pred = mc.Deterministic('efr_pred', 20*np.log10(efr_power))
            efr_pred_long = efr_pred[i_efr_stim, i_efr_subj]

            efr_sigma = mc.HalfCauchy('efr_sigma', 1)
            efr = mc.Normal('efr', efr_pred_long, sd=efr_sigma, observed=efr_obs)

        ########################################################################
        # ABR
        ########################################################################
        if include_abr:
            a_0 = a_0_mean if a_0_std is None else mc.Normal('a_0', a_0_mean, a_0_std)
            a_1 = a_1_mean if a_1_std is None else mc.Normal('a_1', a_1_mean, a_1_std)
            a_i = np.exp(a_0 + a_1 * i_sex)[:, np.newaxis, np.newaxis]

            # Properly finding wave 1 requires including some time dimension as well
            abr_scaled = a_i * i_count[..., np.newaxis] * abr_sim
            abr_pred = abr_scaled.mean(axis=(0, -2))
            w1_pred = mc.Deterministic('w1_pred', abr_pred.max(axis=-1))
            w1_pred_long = w1_pred[i_abr_stim, i_abr_subj]

            w1_sigma = mc.HalfCauchy('w1_sigma', 1)
            w1 = mc.Normal('w1', w1_pred_long, sd=w1_sigma, observed=abr_obs)

    return model


def fit_data(target_accept=0.9, **kwargs):
    with create_model(**kwargs):
        return mc.sample(return_inferencedata=True,
                         target_accept=target_accept)
