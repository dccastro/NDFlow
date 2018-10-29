from typing import Union

import numpy as np
from numpy import linalg
from scipy import special, stats

_TWO_PI = 2. * np.pi
_LOG_TWO_PI = np.log(_TWO_PI)


def _quad(x, A):
    return np.einsum('nd,de,ne->n', x, A, x)


def _mahalanobis(x, A):
    return np.einsum('nd,de,ne->n', x, linalg.inv(A), x)


def norm_rvs(mu, tau, N=None):
    return np.random.normal(mu, np.sqrt(1. / tau), size=N)


def norm_pdf(x, mu, tau):
    return np.sqrt(tau / _TWO_PI) * np.exp(-.5 * tau * (x - mu) ** 2)


def norm_logpdf(x, mu, tau):
    return .5 * (np.log(tau) - _LOG_TWO_PI - tau * (x - mu) ** 2)


def norm_cdf(x, mu, tau):
    return special.ndtr(np.sqrt(tau) * (x - mu))


def mvnorm_rvs(mu, precision, N=None):
    return np.random.multivariate_normal(mu, linalg.inv(precision), size=N)


def mvnorm_rvs2(mu, cov, N=None):
    return np.random.multivariate_normal(mu, cov, size=N)


def mvnorm_pdf(x, mu, precision):
    D = x.shape[-1]
    xx = _quad(x - mu, precision)
    return np.sqrt(linalg.det(precision) / _TWO_PI ** D) * np.exp(-.5 * xx)


def mvnorm_logpdf(x, mu, precision):
    D = x.shape[-1]
    xx = _quad(x - mu, precision)
    return .5 * (np.log(linalg.det(precision)) - _LOG_TWO_PI * D - xx)


def mvnorm_pdf2(x, mu, cov):
    D = x.shape[-1]
    xx = _mahalanobis(x - mu, cov)
    return np.sqrt(1. / (linalg.det(cov) * _TWO_PI ** D)) * np.exp(-.5 * xx)


def mvnorm_logpdf2(x, mu, cov):
    D = x.shape[-1]
    xx = _mahalanobis(x - mu, cov)
    return .5 * (-np.log(linalg.det(cov)) - _LOG_TWO_PI * D - xx)


def gamma_rvs(alpha, beta, N=None):
    return np.random.gamma(alpha, 1. / beta, size=N)


def gamma_pdf(x, alpha, beta):
    return stats.gamma.pdf(x, alpha, 0, 1. / beta)


def gamma_logpdf(x, alpha, beta):
    return stats.gamma.logpdf(x, alpha, 0, 1. / beta)


def t_rvs(df, N=None):
    return np.random.standard_t(df, size=N)


def t_pdf(x, df):
    return np.exp(t_logpdf(x, df))


def t_logpdf(x, df):
    return np.log(1. + x ** 2 / df) * (-.5 * df - .5) \
           - .5 * np.log(df) - special.betaln(.5, .5 * df)


def mvt_pdf(x, df, loc, scale):
    return np.exp(mvt_logpdf(x, df, loc, scale))


def mvt_logpdf(x, df, loc, scale):
    D = x.shape[1] if x.ndim > 1 else 1
    xx = _mahalanobis(x - loc, scale)
    # noinspection PyTypeChecker
    return special.gammaln(.5 * (df + D)) - special.gammaln(.5 * df) \
           - .5 * (D * np.log(np.pi * df) + linalg.slogdet(scale)[1]
                   + (df + D) * np.log(1. + xx / df))


def mvt_rvs(df, loc, scale, N=None):
    u = np.random.chisquare(df, size=N)
    y = np.random.multivariate_normal(np.zeros(scale.shape[0]), scale, size=N)
    return loc + y * np.sqrt(df / u)[:, None]


def beta_rvs(alpha, beta, N=None):
    return np.random.beta(alpha, beta, size=N)


def wishart_rvs(a, B, N=1):
    return stats.wishart.rvs(2. * a, .5 * linalg.inv(B), size=N)


def invwishart_rvs(a, B, N=1):
    return stats.invwishart.rvs(2. * a, 2. * linalg.inv(B), size=N)


def multipsi(a: Union[float, np.ndarray], d: int) -> Union[float, np.ndarray]:
    return special.psi([a + .5 * (1. - i) for i in range(d)]).sum(0)
