from typing import Union

import numpy as np

from .suff_stats import SuffStats


class LikelihoodFactor(object):
    def sample(self, num_samples: int = None) -> np.ndarray:
        """
        Draws samples from this distribution.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples. If not given, a single sample will be drawn, and the result will be a
            flat array (or a scalar, if the distribution is univariate).

        Returns
        -------
        (num_samples, ndim) np.ndarray
            The samples.
        """
        raise NotImplementedError

    def likelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the likelihood of observations under this distribution.

        Parameters
        ----------
        x : (N, ndim) np.ndarray
            Observations.

        Returns
        -------
        (N,) np.ndarray
            The likelihoods of each observation.
        """
        raise NotImplementedError

    def loglikelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the log-likelihood of observations under this distribution. Unless overridden by a
        subclass, it is equivalent to ``np.log(self.likelihood(x))``.

        Parameters
        ----------
        x : (N, ndim) np.ndarray
            Observations.

        Returns
        -------
        (N,) np.ndarray
            The log-likelihoods of each observation.
        """
        return np.log(self.likelihood(x))


class PriorFactor(object):
    def sample(self) -> LikelihoodFactor:
        """
        Draws a sample from this prior distribution.

        Returns
        -------
        LikelihoodFactor
            The parameters of the likelihood model sampled from this prior.
        """
        raise NotImplementedError

    def mean(self) -> LikelihoodFactor:
        """
        Computes the expected values of the likelihood parameters modelled by this prior.

        Returns
        -------
        LikelihoodFactor
            The mean likelihood parameters.
        """
        raise NotImplementedError

    def mode(self) -> LikelihoodFactor:
        """
        Computes the most likely values of the likelihood parameters modelled by this prior.

        Returns
        -------
        LikelihoodFactor
            The modal likelihood parameters.
        """
        raise NotImplementedError

    def predictive(self) -> LikelihoodFactor:
        """
        Gets the predictive distribution for this prior, i.e. the distribution of observations with
        likelihood parameters modelled by this prior.

        Returns
        -------
        LikelihoodFactor
            The parameters of the predictive distribution.
        """
        raise NotImplementedError

    def _posterior(self, suff_stats: SuffStats) -> 'PriorFactor':
        """
        Gets the posterior distribution based on this prior and the given sufficient statistics.

        Parameters
        ----------
        suff_stats : SuffStats
            Sufficient statistics collected from observations.

        Returns
        -------
        PriorFactor
            The posterior distribution over the likelihood parameters. If this prior is conjugate to
            the likelihood, the returned posterior will be of the same type.
        """
        raise NotImplementedError

    def posterior(self, suff_stats: Union[SuffStats, np.ndarray]) -> 'PriorFactor':
        """
        Gets the posterior distribution based on this prior and the given sufficient statistics.

        Parameters
        ----------
        suff_stats : SuffStats | np.ndarray
            Sufficient statistics collected from observations.

        Returns
        -------
        PriorFactor
            The posterior distribution over the likelihood parameters. If this prior is conjugate to
            the likelihood, the returned posterior will be of the same type.
        """
        if not isinstance(suff_stats, SuffStats):
            suff_stats = self.get_suff_stats(suff_stats)
        return self._posterior(suff_stats)

    def _suff_stats(self) -> SuffStats:
        """
        Get an empty sufficient statistics object for this prior.

        Returns
        -------
        SuffStats
            The sufficient statistics
        """
        raise NotImplementedError

    def get_suff_stats(self, x: np.ndarray = None) -> SuffStats:
        """
        Get sufficient statistics from the given observations.

        Parameters
        ----------
        x : (N, ndim) np.ndarray

        Returns
        -------
        SuffStats
            The sufficient statistics
        """
        suff_stats = self._suff_stats()
        if x is not None:
            suff_stats.add(x)
        return suff_stats

    @property
    def ndim(self) -> int:
        """
        Dimensionality of this distribution.

        Returns
        -------
        int
            Number of dimensions.
        """
        raise NotImplementedError


class L2DensityMixin(object):
    def sqnorm(self) -> float:
        """
        Computes the squared L2 norm of this distribution. Unless overridden by a subclass, this is
        equivalent to ``self.product(self)``.

        Returns
        -------
        float
            The squared L2 norm.
        """
        return self.product(self)

    def product(self, lik_factor: 'L2DensityMixin') -> float:
        """
        Computes the L2 inner product between this distribution and the one given.

        Parameters
        ----------
        lik_factor : LikelihoodFactor
            The distribution (of the same type) with which to compute the product.

        Returns
        -------
        float
            The L2 inner product.
        """
        raise NotImplementedError
