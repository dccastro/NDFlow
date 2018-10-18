import numpy as np

from ..models.mixture import MixtureModel


def grads(gmm1: MixtureModel, gmm2: MixtureModel, W: np.ndarray, V: np.ndarray):
    dl = []
    for k, tk in enumerate(gmm1.components):
        dlk = 0

        lk_muk = tk.tau * tk.mu
        dlk += sum(W[k, l] * _grad(tk, tl, lk_muk) for l, tl in enumerate(gmm1.components))
        dlk -= sum(V[k, m] * _grad(tk, tm, lk_muk) for m, tm in enumerate(gmm2.components))
        dlk *= 0.5

        dl.append(dlk)
    return dl


def _grad(tk, tl, lk_muk):
    illk = 1 / (tl.tau + tk.tau)
    mulk = illk * (tl.tau * tl.mu + lk_muk)
    return 1 / tk.tau - illk - (tk.mu - mulk) ** 2


def update(gmm1: MixtureModel, dl, lrate, var_reg: float = None):
    reg = 0
    for k, tk in enumerate(gmm1.components):
        lk = tk.tau
        if var_reg is not None:
            reg += 1 / lk
            dl[k] += var_reg * (-1 / lk ** 2)
        ck = np.sqrt(lk)
        dck = 2 * dl[k] * ck  # Square-root chain rule
        ck -= lrate * dck
        tk.tau = ck ** 2
    return var_reg * reg
