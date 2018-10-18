import numpy as np

from . import stats
from .base import LikelihoodFactor, PriorFactor, L2DensityMixin
from .student import StudentsT
from .suff_stats import SuffStats


class Normal(LikelihoodFactor, L2DensityMixin):
    __slots__ = ('mu', 'tau')

    def __init__(self, mu, tau):
        self.mu = np.atleast_1d(mu)
        self.tau = np.atleast_1d(tau)

    def sample(self, N=None):
        size = (N, 1) if N is not None else None
        return stats.norm_rvs(self.mu, self.tau, size)

    def likelihood(self, x: np.ndarray) -> float:
        return stats.norm_pdf(x, self.mu, self.tau).flatten()

    def loglikelihood(self, x: np.ndarray) -> float:
        return stats.norm_logpdf(x, self.mu, self.tau).flatten()

    def sqnorm(self):
        return np.sqrt(np.prod(.25 * self.tau / np.pi))

    def product(self, other: 'Normal'):
        return stats.norm_pdf(self.mu, other.mu, 1. / (1. / self.tau + 1. / other.tau)).prod()

    def cumulative(self, x):
        return stats.norm_cdf(x, self.mu, self.tau).flatten()

    def __str__(self):
        return f"Normal(mu={self.mu}, tau={self.tau})"


class NormalSuffStats(SuffStats):
    def __init__(self):
        self.N = 0
        self.m1 = 0.
        self.m2 = 0.

    def _suff_stats(self, X):
        X = np.atleast_2d(X)
        return dict(N=X.shape[0], m1=X.sum(axis=0), m2=(X ** 2).sum(axis=0))


class NormalGamma(PriorFactor):
    __slots__ = ('mu', 'nu', 'alpha', 'tau')

    def __init__(self, mu, nu, alpha, tau):
        self.mu = np.atleast_1d(mu)
        self.nu = nu
        self.alpha = alpha
        self.tau = tau

    def sample(self):
        tau = stats.gamma_rvs(self.alpha, self.alpha / self.tau)
        mu = stats.norm_rvs(self.mu, self.nu * tau)
        return Normal(mu, tau)

    def mean(self):
        return Normal(self.mu, self.tau)

    def mode(self):
        return Normal(self.mu, self.tau * (self.alpha - 1.) / self.alpha)

    @staticmethod
    def fit(X, alpha, nu):
        loc = X.mean(axis=0)
        scale = X.std(axis=0)
        mu = loc
        tau = alpha / (alpha - 1) * (1.0 + nu) / (nu * scale ** 2)
        return NormalGamma(mu, nu, alpha, tau)

    def predictive(self):
        df = 2.0 * self.alpha
        loc = self.mu
        scale = np.sqrt((1.0 + self.nu) / (self.nu * self.tau))
        return StudentsT(df, loc, scale)

    def _posterior(self, suff_stats: NormalSuffStats):
        if suff_stats.N == 0:
            return self

        N = suff_stats.N
        x_mean = suff_stats.m1 / N
        x_var = suff_stats.m2 / N - x_mean ** 2.

        mu = self.mu
        nu = self.nu
        alpha = self.alpha
        beta = alpha / self.tau

        nu_ = nu + N
        mu_ = (nu * mu + N * x_mean) / nu_

        alpha_ = alpha + .5 * N
        beta_ = beta + .5 * N * (x_var + nu * ((x_mean - mu) ** 2.) / nu_)

        return NormalGamma(mu_, nu_, alpha_, alpha_ / beta_)

    def _suff_stats(self):
        return NormalSuffStats()

    @property
    def ndim(self):
        return self.mu.shape[0]

    def __str__(self):
        return f"NormalGamma(mu={self.mu}, nu={self.nu}, alpha={self.alpha}, tau={self.tau})"
