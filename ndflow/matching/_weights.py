import numpy as np

from ..models.mixture import MixtureModel


def grads(gmm1: MixtureModel, gmm2: MixtureModel, W: np.ndarray, V: np.ndarray):
    pi1 = gmm1.assignment_probs()
    dpi = []
    for k in range(gmm1.num_components):
        dpik = (sum(W[k, :]) - sum(V[k, :])) / pi1[k]
        dpi.append(dpik)
    return dpi


def update(gmm1, dpi, lrate):
    xi = np.log(gmm1.pi)
    dxi = gmm1.pi * (1 - gmm1.pi) * np.array(dpi)  # Softmax chain rule
    xi -= lrate * dxi
    gmm1.pi = np.exp(xi) / np.exp(xi).sum()  # Softmax
