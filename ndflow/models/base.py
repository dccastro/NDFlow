from typing import Iterable, Sized

import numpy as np


class BaseMixture:
    """A generic mixture model base class."""

    def sample_component(self, N=None):
        """Sample component index.

        Parameters
        ----------
        N : int
            Size of the sample to generate. If `None`, generates a single value, otherwise a
            `ndarray` of corresponding size.

        Returns
        -------
        int | np.ndarray[int]
        """
        raise NotImplementedError

    def sample_observation(self, k, N=None):
        """Sample observations from the given component.

        Parameters
        ----------
        k : int
            Component index.
        N : int
            Size of the sample to generate. If `None`, generates a single value, otherwise a
            `ndarray` of corresponding size.

        Returns
        -------
        float | np.ndarray
        """
        raise NotImplementedError

    def sample(self, N=None):
        """Generates a sample under this model.

        Parameters
        ----------
        N : int
            Size of the sample to generate. If `None`, generates a single sample, otherwise
            `ndarrays` of corresponding size.

        Returns
        -------
        (Any, int) | (np.ndarray[Any], np.ndarray[int])
            The observation(s) and the corresponding component assignment(s).
        """
        if N is None:
            z = self.sample_component()
            x = self.sample_observation(z)
            return x, z
        X = None
        Z = self.sample_component(N)
        counts = np.bincount(Z, minlength=self.num_components)
        for k in range(self.num_components):
            X_k = self.sample_observation(k, counts[k])
            if X is None:
                if X_k.ndim == 1:
                    X = np.zeros(N, dtype=X_k.dtype)
                else:
                    X = np.zeros((N, X_k.shape[-1]), dtype=X_k.dtype)
            X[Z == k] = X_k
        return X, Z

    def component_likelihoods(self, X):
        """Computes the likelihoods for each different component.

        Parameters
        ----------
        X : float | np.ndarray
            Observation(s).

        Returns
        -------
        np.ndarray
            The likelihoods.
        """
        raise NotImplementedError

    def assignment_probs(self):
        """Computes the prior probabilities of an observation belonging to each component.

        Returns
        -------
        np.ndarray
            The probabilities.
        """
        raise NotImplementedError

    def assignment_posteriors(self, X):
        """Computes the posterior component probabilities for the given observations based on the
        model's prior probabilities.

        Parameters
        ----------
        X : np.ndarray
            Observations.

        Returns
        -------
        np.ndarray
            Posterior component probabilities.
        """
        p_z = self.assignment_probs()[:self.num_components]
        p_x_z = self.component_likelihoods(X)
        p_z_x = p_x_z * p_z[None, :] + np.finfo(np.float64).eps
        p_z_x /= p_z_x.sum(axis=-1, keepdims=True)
        return p_z_x

    def predict(self, X):
        """Predicts component assignments.

        Parameters
        ----------
        X : np.ndarray
            Observations.

        Returns
        -------
        np.ndarray[int]
            Predicted component indices.
        """
        raise NotImplementedError

    def marginal_likelihood(self, X):
        """Calculates the marginal likelihood of each observation under this model.

        Parameters
        ----------
        X : np.ndarray
            Observations.

        Returns
        -------
        np.ndarray
            Likelihoods.
        """
        liks = self.component_likelihoods(X)
        p_z = self.assignment_probs()[:self.num_components]
        return liks @ p_z

    def loglikelihood(self, X):
        """Calculates the joint log-likelihood of the given set of observations under this model.

        Parameters
        ----------
        X : np.ndarray
            Observations.

        Returns
        -------
        float
            Log-likelihood.
        """
        marg = self.marginal_likelihood(X)
        return np.log(marg).sum()

    @property
    def num_components(self):
        """Number of components in the mixture model.

        Returns
        -------
        int
        """
        raise NotImplementedError

    def component_params(self):
        """Parameters of each component in the mixture.

        Returns
        -------
        List
        """
        raise NotImplementedError

    def prune(self, counts_tol=1e-3):
        from .mixture import MixtureModel

        weights = self.assignment_probs()
        components = self.component_params()

        if isinstance(counts_tol, (Iterable, Sized)):
            counts = counts_tol
            if len(counts) != self.num_components:
                raise ValueError(f"Incompatible number of components: "
                                 f"expected {self.num_components}, got {len(counts)}")

            keep = np.nonzero(counts)
            keep = keep[0]
        elif isinstance(counts_tol, float):
            tol = counts_tol
            keep = np.where(weights > tol)[0]
        else:
            raise ValueError("counts_tol should be iterable or float")

        weights_ = weights[keep] / weights[keep].sum()
        components_ = [components[k] for k in keep]

        return MixtureModel(components_, weights_)
