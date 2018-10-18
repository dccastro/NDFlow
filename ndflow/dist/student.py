import numpy as np

from . import stats
from .base import LikelihoodFactor


class StudentsT(LikelihoodFactor):
    def __init__(self, df, loc, scale):
        self.df = df
        self.loc = loc
        self.scale = scale

    def sample(self, N=None):
        size = (N, 1) if N is not None else None
        x = self.loc + self.scale * stats.t_rvs(self.df, size)
        return x

    def likelihood(self, x):
        p = stats.t_pdf((x - self.loc) / self.scale, self.df) / self.scale
        return p.flatten()

    def loglikelihood(self, x):
        log_p = stats.t_logpdf((x - self.loc) / self.scale, self.df) - np.log(self.scale)
        return log_p.flatten()

    def __str__(self):
        return f"StudentsT(df={self.df}, loc={self.loc}, scale={self.scale})"
