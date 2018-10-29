import numpy as np

from .base import BaseMixture
from ..distributions import normal


class WeightedVariationalDPGMM(BaseMixture):
    def __init__(self, weights, alpha, prior, truncation, num_samples):
        """
        Parameters
        ----------
        weights : np.ndarray
            The weights to be used for each datum
        alpha : float
            The Dirichlet Process's concentration/strength/mass parameter.
        prior : dist.normal.NormalGamma
            Parameters of the Normal-Gamma prior.
        truncation : int
            The truncation level (i.e. maximum number of components).
        num_samples : int
            The number of samples/data points.
        """
        self.weights = weights
        self.alpha = alpha
        self.prior = prior
        self.truncation = truncation
        self.num_samples = num_samples

        self.gamma1 = np.ones(truncation - 1)
        self.gamma2 = alpha * np.ones(truncation - 1)

        phi_ = (alpha / (1 + alpha)) ** (np.arange(truncation) - 1)
        phi_ /= phi_.sum()
        self.phi = np.tile(phi_, (num_samples, 1))

        self.mu = prior.mu[None]
        self.zeta = (prior.nu * prior.tau)[None]
        self.a = np.atleast_1d(prior.alpha)[None]
        self.b = (prior.alpha / prior.tau)[None]

    def component_params(self):
        mu_ = self.mu
        tau_ = self.a / self.b
        components_ = [normal.Normal(mu_[k], tau_[k]) for k in range(self.truncation)]
        return components_

    def assignment_probs(self):
        weights_ = self.weights / self.weights.sum()
        return weights_ @ self.phi

    def component_likelihoods(self, X):
        components_ = self.component_params()
        return np.asarray([component_k.likelihood(X) for component_k in components_]).T

    def predict(self, X=None):
        return self.phi.argmax(axis=-1)

    @property
    def num_components(self):
        return self.truncation

    @property
    def concentration(self):
        return self.alpha

    def assignment_posteriors(self, X=None):
        return self.phi
