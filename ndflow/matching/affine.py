from copy import deepcopy
from typing import Tuple

import numpy as np

from ..models.mixture import MixtureModel


class AffineTransform:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def __call__(self, x):
        return self.intercept + self.slope * x


def align(source_gmm: MixtureModel, target_gmm: MixtureModel) -> Tuple[MixtureModel, AffineTransform]:
    """Affinely align two Gaussian mixture models (GMMs).

    Parameters
    ----------
    source_gmm : ndflow.models.mixture.MixtureModel
        Source GMM.
    target_gmm : ndflow.models.mixture.MixtureModel
        GMM to whose mean and variance `source_gmm` will be aligned.

    Returns
    -------
    aligned_gmm : ndflow.models.mixture.MixtureModel
        Result of translating and rescaling `source_gmm`.
    alignment : AffineTransform
        Callable object representing the same affine transformation applied to `source_gmm`.
    """
    target_mean, target_var = target_gmm.mean_variance()
    return transform(source_gmm, target_mean, target_var)


def transform(source_gmm: MixtureModel, target_mean: float, target_var: float):
    """Affinely transform a Gaussian mixture model (GMM) to match given mean and variance.

    Parameters
    ----------
    source_gmm : ndflow.models.mixture.MixtureModel
        Source GMM.
    target_mean : float
        New mean to which `source_gmm` will be translated.
    target_var : float
        New variance to which `source_gmm` will be rescaled.

    Returns
    -------
    aligned_gmm : ndflow.models.mixture.MixtureModel
        Result of translating and rescaling `source_gmm`.
    alignment : AffineTransform
        Callable object representing the same affine transformation applied to `source_gmm`.
    """
    aligned_gmm = deepcopy(source_gmm)
    source_mean, source_var = source_gmm.mean_variance()

    slope = np.sqrt(target_var / source_var)
    intercept = target_mean - slope * source_mean

    alignment = AffineTransform(slope, intercept)

    for tk in aligned_gmm.components:
        tk.mu = alignment(tk.mu)
        tk.tau /= slope ** 2

    return aligned_gmm, alignment
