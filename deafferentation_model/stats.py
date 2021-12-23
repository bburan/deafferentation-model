import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

from .util import calc_synapses


###############################################################################
# utility
###############################################################################
def add_predictions(df, trace, frequencies=[None, 4e3]):
    theta_i = trace.posterior.theta_i
    to_add = {}
    for frequency in frequencies:
        label = 'm' if frequency is None else f'{frequency*1e-3:.0f}kHz'
        s = calc_synapses(frequency, theta_i)
        t = np.log(s)
        to_add[f't{label}_mean'] = t.mean(dim=('chain', 'draw')).values
        to_add[f't{label}_std'] = t.std(dim=('chain', 'draw')).values
        to_add[f's{label}_mean'] = s.mean(dim=('chain', 'draw')).values
        to_add[f's{label}_std'] = s.std(dim=('chain', 'draw')).values

    to_add = pd.DataFrame(to_add)
    to_add.index.name = 'i_subj'
    return df.join(to_add, on='i_subj')


###############################################################################
# linear
###############################################################################
def linear_fit_age(age, t_mean, t_std, target_accept=0.8, use_skeptical=False):

    age = np.asarray(age)
    t_mean = np.asarray(t_mean)
    t_std = np.asarray(t_std)

    np.random.seed(987654321)
    with pm.Model() as win_model:
        # Set up the conditional means prior
        if use_skeptical:
            beta0 = pm.Normal('beta0', 10, 2.5)
            beta1 = pm.Normal('beta1', mu=0, sd=1)
        else:
            beta0 = pm.Normal('beta0', mu=17, sd=3.8)
            beta1 = pm.Normal('beta1', mu=-1.1, sd=0.4)
        mu = beta0 + beta1 * age / 10
        ll2 = pm.Normal('age', np.log(mu), sigma=t_std, observed=t_mean)
        return pm.sample(return_inferencedata=True,
                         target_accept=target_accept)


def linear_plot_age(trace, color='k', ax=None, pred_bounds=None):
    if ax is None:
        figure, ax = plt.subplots(1, 1, figsize=(4, 4))

    si = np.linspace(0, 100, 100)

    b0 = trace.posterior.beta0.values[..., np.newaxis]
    b1 = trace.posterior.beta1.values[..., np.newaxis]

    y = b0 + b1 * si / 10
    y_mean = y.mean(axis=(0, 1))
    y_lb, y_ub = az.hdi(y).T

    ax.plot(si, y_mean, '-', color=color)
    ax.fill_between(si, y_lb, y_ub, color=color, alpha=0.25)
    ax.set_ylabel('Synapses')
    ax.set_xlabel('Age')


def linear_fit_dtt(score, i_level, i, t_mean, t_std, cog, target_accept=0.8):
    x1, m1, s1 = 16, -10, 1
    x2, m2, s2 = 6, -9, 1

    score = np.asarray(score)
    i = np.asarray(i)
    t_mean = np.asarray(t_mean)
    t_std = np.asarray(t_std)
    cog = np.asarray(cog)

    n_subjects = len(t_mean)
    assert (i.max() + 1) == n_subjects
    assert len(set(i)) == n_subjects

    np.random.seed(987654321)
    with pm.Model() as win_model:
        # Set up the conditional means prior
        beta0 = pm.Flat('intercept')
        beta1 = pm.Flat('synapse')

        beta2 = pm.Normal('level', 0, 1)
        beta3 = pm.Normal('synapse:level', 0, 1)
        beta4 = pm.Normal('cognition', 0, 1)

        i_low = pm.Deterministic('i_low', beta0)
        i_high = pm.Deterministic('i_high', beta0 + beta2)
        s_low = pm.Deterministic('s_low', beta1)
        s_high = pm.Deterministic('s_high', beta1 + beta3)

        mu1 = beta0 + beta1 * x1
        mu2 = beta0 + beta1 * x2
        d1_win = pm.Normal.dist(m1, s1)
        d2_win = pm.Normal.dist(m2, s2)
        log_prior = pm.Potential('joint_beta', d1_win.logp(mu1) + d2_win.logp(mu2))

        subject_sd = pm.HalfCauchy('subject_sd', 1)
        subject = pm.Normal('subject', 0, subject_sd, shape=n_subjects)
        t_mu = pm.Normal('t_mu', 0, 10, shape=n_subjects)
        pm.Normal('t', t_mu, t_std, observed=t_mean)

        s_mu = np.exp(t_mu)
        sin_mu = beta0 + \
            subject[i] + \
            beta1 * s_mu[i] + \
            beta2 * i_level + \
            beta3 * i_level * s_mu[i] + \
            beta4 * cog[i]

        tau = pm.HalfCauchy('tau', 1)
        ll2 = pm.Normal('dtt', mu=sin_mu, sigma=tau, observed=score)
        #nu = pm.Exponential('nu_minus_1', 1/29.0) + 1
        #ll2 = pm.StudentT('dtt', mu=sin_mu, nu=nu, sigma=tau, observed=score)
        return pm.sample(return_inferencedata=True, target_accept=target_accept)


def linear_plot_dtt(trace, color='k', ax=None, pred_bounds=None):
    if ax is None:
        figure, ax = plt.subplots(1, 1, figsize=(4, 4))

    si = np.linspace(0, 100, 100)

    b0 = trace.posterior.beta0.values[..., np.newaxis]
    b1 = trace.posterior.beta1.values[..., np.newaxis]

    y = b0 + b1 * si
    y_mean = y.mean(axis=(0, 1))
    y_lb, y_ub = az.hdi(y).T

    ax.plot(si, y_mean, '-', color=color)
    ax.fill_between(si, y_lb, y_ub, color=color, alpha=0.25)
    ax.set_ylabel('Synapses')
    ax.set_xlabel('Age')


###############################################################################
# logodds
###############################################################################
p_to_logodds = lambda x: np.log(x / (1 - x))
logistic = lambda x: np.exp(x) / (1 + np.exp(x))
logodds_to_p = logistic


def logodds_fit(i, t_mean, t_std, target_accept=0.8, joint_posterior=True):
    # Set joint_posterior to False to verify no strong bias in fitting
    i = np.asarray(i).astype('i')
    t_mean = np.asarray(t_mean)
    t_std = np.asarray(t_std)

    # probability of non-veteran having tinnitus
    x1, a1, b1 = np.max(np.exp(t_mean)), 1, 25
    # probability of veteran having tinitus
    x2, a2, b2 = np.mean(np.exp(t_mean)), 2, 5
    tinnitus_d1 = pm.Beta.dist(a1, b1)
    tinnitus_d2 = pm.Beta.dist(a2, b2)

    np.random.seed(987654321)
    with pm.Model() as model:
        beta0 = pm.Flat('beta0')
        beta1 = pm.Flat('beta1')

        if joint_posterior:
            f1 = logistic(beta0 + beta1 * x1)
            f2 = logistic(beta0 + beta1 * x2)
            pi_beta1 = (f1**(a1-1)) * ((1-f1)**(b1-1)) * f1 * (1-f1)
            pi_beta2 = (f2**(a2-1)) * ((1-f2)**(b2-1)) * f2 * (1-f2)
            ll = np.log(pi_beta1) + np.log(pi_beta2)
            log_prior = pm.Potential('joint_beta', ll)

        t_mu = pm.Normal('t_mu', 0, 10, shape=len(t_mean))

        logit_p = beta0 + beta1 * np.exp(t_mu)
        ll1 = pm.Bernoulli('tinnitus', logit_p=logit_p, observed=i)
        ll2 = pm.Normal('synapses', t_mu, t_std, observed=t_mean)
        return pm.sample(return_inferencedata=True,
                         target_accept=target_accept)


def logodds_prob(trace, value_dict=None):
    if value_dict is None:
        s = np.exp(trace.posterior.t_mu.mean(dim=('chain', 'draw'))).values
        value_dict = {
            'min': s.min(),
            'max': s.max(),
            'mean': s.mean(),
        }
    result = {}
    for k, v in value_dict.items():
        result[f'logodds_{k}'] = trace.posterior.beta0 + trace.posterior.beta1 * v
        result[f'prob_{k}'] = logodds_to_p(result[f'logodds_{k}']) * 100
    return az.convert_to_inference_data(result)


def logodds_plot(trace, color='k', ax=None):
    if ax is None:
        figure, ax = plt.subplots(1, 1, figsize=(4, 4))

    lb, ub = 0, 16
    si = np.arange(lb, ub, 0.1)

    b0 = trace.posterior.beta0.values[..., np.newaxis]
    b1 = trace.posterior.beta1.values[..., np.newaxis]
    p = logodds_to_p(b0 + b1 * si) * 100
    p_mean = p.mean(axis=(0, 1))
    p_lb, p_ub = az.hdi(p).T

    ax.plot(si, p_mean, '-', color=color)
    ax.fill_between(si, p_lb, p_ub, color=color, alpha=0.25)
    ax.set_xlabel('Synapses')
    ax.set_ylabel('Probability (%)')

    y = np.clip(trace.observed_data.T.values * 100, 10, 90)
    y = y + np.random.uniform(-5, 5, len(y))
    x = np.exp(trace.observed_data.t.values)
    ax.plot(x, y, 'ko')
