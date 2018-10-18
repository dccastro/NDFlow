from copy import deepcopy

import numpy as np

from ..models.mixture import MixtureModel


class AffineTransform:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def __call__(self, x):
        return self.intercept + self.slope * x


def align(source_gmm: MixtureModel, target_gmm: MixtureModel):
    target_mean, target_var = target_gmm.mean_variance()
    return transform(source_gmm, target_mean, target_var)


def transform(source_gmm: MixtureModel, target_mean: float, target_var: float):
    aligned_gmm = deepcopy(source_gmm)
    source_mean, source_var = source_gmm.mean_variance()

    slope = np.sqrt(target_var / source_var)
    intercept = target_mean - slope * source_mean

    alignment = AffineTransform(slope, intercept)

    for tk in aligned_gmm.components:
        tk.mu = alignment(tk.mu)
        tk.tau /= slope ** 2

    return aligned_gmm, alignment
