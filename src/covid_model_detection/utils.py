import numpy as np

SERO_DAYS = 15
PCR_DAYS = 11


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def logit_to_linear(mean, sd):
    linear_mean = expit(mean)
    linear_sd = (np.exp(mean) / (1.0 + np.exp(mean)) ** 2) * sd

    return linear_mean, linear_sd


def linear_to_logit(mean, sd):
    logit_mean = logit(mean)
    logit_sd = sd / (mean * (1.0 - mean))

    return logit_mean, logit_sd


def se_from_ss(p, n):
    return np.sqrt((p * (1 - p)) / n)


def ss_from_se(p, se):
    return (p * (1 - p)) / (se ** 2) 


def ss_from_ci(m, l, u, transformation='logit_95'):
    if transformation == 'logit_95':
        se = (logit(u) - logit(l)) / 3.92
        se *= (m / (1.0 + m) ** 2)
        ss = ss_from_se(m, se)
    else:
        raise ValueError('Assumes `logit_95`.')
    
    return ss
