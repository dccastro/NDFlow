import numpy as np

from ..models.mixture import MixtureModel


def grads(gmm1: MixtureModel, gmm2: MixtureModel, W: np.ndarray, V: np.ndarray):
    dmu = []
    for k, tk in enumerate(gmm1.components):
        dmuk = 0
        dmuk += sum(W[k, l] * _grad(tk, tl) for l, tl in enumerate(gmm1.components))
        dmuk -= sum(V[k, m] * _grad(tk, tm) for m, tm in enumerate(gmm2.components))
        dmu.append(dmuk)
    return dmu


def _grad(tk, tl):
    return (tl.mu - tk.mu) / (1. / tl.tau + 1. / tk.tau)


def update(gmm1: MixtureModel, dmu, lrate):
    for k, tk in enumerate(gmm1.components):
        tk.mu -= lrate * dmu[k]
