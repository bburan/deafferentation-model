import numpy as np
import pymc3 as mc

from .util import inv_greenwood


sr_dist = np.array([0.12, 0.15, 0.73])[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
cum_sr_dist = np.array([1.0, 0.88, 0.73])[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]


def fit(result, priors, age_priors):
    i_sex = result['i_sex']
    i_stimuli = result['i_stimuli'].astype('i')
    i_subject = result['i_subject'].astype('i')
    cf = result['cf']
    r = result['rates']
    w1_obs = result['w1']

    n_stim = i_stimuli.max() + 1
    n_subjects = i_subject.max() + 1
    x = inv_greenwood(cf)[:, np.newaxis]

    with mc.Model() as abr_model:
        #theta_nh = mc.Normal('theta_nh', age_priors['theta_mu'], age_priors['theta_sd'], shape=3)
        #theta_nh_i = mc.MvNormal('theta_nh_i', theta_nh, cov=age_priors['cov'], shape=3)
        theta_nh_i = age_priors['theta_mu']

        theta = mc.Normal('theta', priors['theta_mu'], priors['theta_sd'], shape=3)
        theta_i = mc.MvNormal('theta_i', theta, cov=priors['cov'], shape=(n_subjects, 3))[:, :, np.newaxis, np.newaxis]
        c_0 = mc.Normal('c_0', -10, 1)
        c_1 = mc.Normal('c_1', 0, 1)
        c_i = np.exp(c_0 + c_1 * i_sex)[:, np.newaxis, np.newaxis]

        ANFperIHC_nh = np.exp(-np.exp(theta_nh_i[0]) * (x - theta_nh_i[1])**2 + theta_nh_i[2])
        ANFperIHC = np.exp(-np.exp(theta_i[:, 0]) * (x - theta_i[:, 1])**2 + theta_i[:, 2])

        nh_count = ANFperIHC_nh * sr_dist
        cum_nh_count = ANFperIHC_nh * cum_sr_dist
        i_count = (nh_count - (cum_nh_count - ANFperIHC).clip(0, np.inf)).clip(0, np.inf)

        scaled_rates = c_i * i_count * r

        # Average across CF and SR
        abr = scaled_rates.mean(axis=(-2, 0))
        w1 = abr.max(axis=-1)

        w1_pred = w1[i_stimuli, i_subject]

        sigma = mc.HalfCauchy('sigma', 1)
        w1_amplitude = mc.Normal('w1', w1_pred, sd=sigma, observed=w1_obs)
        return mc.sample(return_inferencedata=True)


def parabola(x, theta):
    return np.exp(-np.exp(theta[..., 0, :, :]) * (x - theta[..., 1, :, :])**2 + theta[..., 2, :, :])


def fit_with_efr(result, priors, age_priors):
    i_sex = result['i_sex']
    i_stimuli = result['i_stimuli'].astype('i')
    i_subject = result['i_subject'].astype('i')
    cf = result['cf']
    w1_obs = result['w1']

    r_abr = result['rates_abr']
    r_efr = result['rates_efr']

    n_stim = i_stimuli.max() + 1
    n_subjects = i_subject.max() + 1
    x = inv_greenwood(cf)[:, np.newaxis]

    with mc.Model() as abr_model:
        #theta_nh = mc.Normal('theta_nh', age_priors['theta_mu'], age_priors['theta_sd'], shape=3)
        #theta_nh_i = mc.MvNormal('theta_nh_i', theta_nh, cov=age_priors['cov'], shape=3)
        theta_nh_i = age_priors['theta_mu']

        theta = mc.Normal('theta', priors['theta_mu'], priors['theta_sd'], shape=3)
        theta_i = mc.MvNormal('theta_i', theta, cov=priors['cov'], shape=(n_subjects, 3))[:, :, np.newaxis, np.newaxis]

        ANFperIHC_nh = parabola(x, theta_nh_i)
        ANFperIHC = parabola(x, theta_i)
        nh_count = ANFperIHC_nh * sr_dist
        cum_nh_count = ANFperIHC_nh * cum_sr_dist
        i_count = (nh_count - (cum_nh_count - ANFperIHC).clip(0, np.inf)).clip(0, np.inf)

        # Now, calculate ABR. Average across CF and SR
        a_0 = mc.Normal('a_0', -10, 1)
        a_1 = mc.Normal('a_1', 0, 1)
        a_i = np.exp(a_0 + a_1 * i_sex)[:, np.newaxis, np.newaxis]

        scaled_rates_abr = a_i * i_count * r_abr
        abr = scaled_rates.mean(axis=(-2, 0))
        w1 = abr.max(axis=-1)
        w1_pred = w1[i_stimuli, i_subject]
        sigma = mc.HalfCauchy('sigma', 1)
        w1_amplitude = mc.Normal('w1', w1_pred, sd=sigma, observed=w1_obs)

        if include_efr:
            e_0 = mc.Normal('a_0', -10, 1)
            e_1 = mc.Normal('a_1', 0, 1)
            e_i = np.exp(e_0 + e_1 * i_sex)[:, np.newaxis, np.newaxis]
            scaled_rates_efr = e_i * i_count * r_efr
            efr = get_efr(scaled_rates_efr)

        return mc.sample(return_inferencedata=True)
