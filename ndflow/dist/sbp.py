import numpy as np
from scipy import stats


def _get_probs(betas: np.ndarray) -> np.ndarray:
    p = np.append(betas, [1.])
    p[1:] *= (1. - betas).cumprod()
    return p


class SBP:
    """Stick-Breaking Process"""

    def __init__(self, alpha: float = 1.):
        self.alpha = alpha
        self.betas = None

    def sample(self, truncation: int) -> np.ndarray:
        self.betas = stats.beta.rvs(1., self.alpha, size=truncation - 1)
        return _get_probs(self.betas)

    def mean(self, truncation: int) -> np.ndarray:
        self.betas = np.ones(truncation - 1) * (1. / (1. + self.alpha))
        return _get_probs(self.betas)

    def posterior(self, Z: np.ndarray, truncation: int):
        counts = np.bincount(Z, minlength=truncation)
        cum_sum = np.roll(counts[::-1].cumsum()[::-1], -1)
        cum_sum[-1] = 0
        a_ = 1. + counts[:-1]
        b_ = self.alpha + cum_sum[:-1]
        return SBPPosterior(a_, b_, counts)


class SBPPosterior:
    def __init__(self, a, b, counts):
        self.a = a
        self.b = b
        self.counts = counts
        self.betas = None

    def sample(self) -> np.ndarray:
        self.betas = stats.beta.rvs(self.a, self.b)
        return _get_probs(self.betas)

    def mean(self) -> np.ndarray:
        self.betas = self.a / (self.a + self.b)
        return _get_probs(self.betas)

    @property
    def num_categories(self):
        return len(self.counts)

    @property
    def num_samples(self):
        return sum(self.counts)
