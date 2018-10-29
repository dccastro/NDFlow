import numpy as np

from .base import BaseMixture
from ..distributions import normal


class MixtureModel(BaseMixture):
    def __init__(self, components, weights):
        """
        Parameters
        ----------
        components : list[normal.Normal]
        weights : array_like
        """
        self.components = components
        self.weights = np.array(weights)

    def sample_component(self, N=None):
        return np.random.choice(self.num_components, size=N, p=self.assignment_probs())

    def sample_observation(self, k, N=None):
        return self.components[k].sample(N)

    def component_likelihoods(self, X):
        return np.asarray([component.likelihood(X) for component in self.components]).T

    def cumulative_distribution(self, X):
        cdfs = np.asarray([component.cumulative(X) for component in self.components]).T
        return cdfs @ self.assignment_probs()

    def component_params(self):
        return self.components

    def predict(self, X):
        return self.assignment_posteriors(X).argmax(axis=-1)

    def assignment_probs(self):
        return self.weights

    @property
    def num_components(self):
        return len(self.weights)

    def mean_variance(self):
        mean = sum(p * t.mu for p, t in zip(self.weights, self.components))
        var = sum(p / t.tau for p, t in zip(self.weights, self.components)) \
            + sum(p * t.mu ** 2 for p, t in zip(self.weights, self.components)) \
            - mean ** 2
        return mean.squeeze(), var.squeeze()
