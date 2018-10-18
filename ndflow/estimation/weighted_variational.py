import numpy as np
from scipy import special

from ..dist import normal, sbp, stats
from ..models.weighted_variational import WeightedVariationalDPGMM


def init(vdpgmm: WeightedVariationalDPGMM, X: np.ndarray, w: np.ndarray):
    prior = sbp.SBP(vdpgmm.alpha).sample(vdpgmm.truncation)
    components = [vdpgmm.prior.sample() for _ in range(vdpgmm.num_components)]
    likelihoods = np.array([tk.likelihood(X) for tk in components]).T  # (N, K)
    phi = likelihoods * w[:, None] * prior[None, :] + np.finfo(float).eps
    phi /= phi.sum(-1, keepdims=True)
    vdpgmm.phi = phi


def step(vdpgmm: WeightedVariationalDPGMM, X: np.ndarray, w: np.ndarray):
    prior = vdpgmm.prior
    phi = vdpgmm.phi
    vdpgmm.mu, vdpgmm.zeta, mean_bound = _update_mean_params(X, w, prior, phi, vdpgmm.a, vdpgmm.b)
    vdpgmm.a, vdpgmm.b, precision_bound = _update_precision_params(X, w, prior, phi, vdpgmm.mu,
                                                                   vdpgmm.zeta)

    vdpgmm.gamma1, vdpgmm.gamma2, sbp_bound = _update_sbp_params(w, vdpgmm.alpha, phi)

    vdpgmm.phi, assig_bound = _update_assig_params(X, w, vdpgmm.mu, vdpgmm.zeta, vdpgmm.a,
                                                   vdpgmm.b, vdpgmm.gamma1, vdpgmm.gamma2)

    bound = assig_bound + mean_bound + precision_bound + sbp_bound

    return phi, bound


def _update_mean_params(X: np.ndarray,  # (N, D)
                        w: np.ndarray,  # (N,)
                        prior: normal.NormalGamma,
                        phi_: np.ndarray,  # (N, K)
                        a_: np.ndarray,  # (K, D)
                        b_: np.ndarray):  # (K, D)
    mu0, nu = prior.mu[None], prior.nu
    n_ = (nu + w @ phi_)[:, None]

    mu_ = (nu * mu0 + np.einsum('nk,n,nd->kd', phi_, w, X)) / n_
    zeta_ = a_ / b_ * n_

    bound = (.5 * (np.log(nu) + special.psi(a_) - np.log(b_) -
                   nu * a_ / b_ * ((mu_ - mu0) ** 2 + 1. / zeta_) - np.log(zeta_) + 1.)).sum()

    return mu_, zeta_, bound


def _update_precision_params(X: np.ndarray,  # (N, D)
                             w: np.ndarray,  # (N,)
                             prior: normal.NormalGamma,
                             phi_: np.ndarray,  # (N, K)
                             mu_: np.ndarray,  # (K, D)
                             zeta_: np.ndarray):  # (K, D)
    mu0, nu = prior.mu[None], prior.nu
    a, b = prior.alpha, (prior.alpha / prior.tau)[None]
    phi_sum = w @ phi_
    n_ = (nu + phi_sum)[:, None]

    a_ = a + .5 * (phi_sum[:, None] + 1.)  # type: np.ndarray
    b_ = b + .5 * (np.einsum('nk,n,nkd->kd', phi_, w, (X[:, None] - mu_[None]) ** 2) +
                   nu * (mu_ - mu0) ** 2 + n_ / zeta_)

    bound = (special.gammaln(a_) - special.gammaln(a) + a * np.log(b / b_) +
             (a - a_) * special.psi(a_) - a_ * b / b_ + a_).sum()

    return a_, b_, bound


def _update_sbp_params(w: np.ndarray,  # (N,)
                       alpha: float,
                       phi_: np.ndarray):
    phi_sum = w @ phi_
    phi_cumsum = np.roll(phi_sum[::-1].cumsum()[::-1], -1)  # phi_cumsum[i] = phi_sum[i+1:].sum()
    phi_cumsum[-1] = 0.

    gamma1_ = 1. + phi_sum
    gamma2_ = alpha + phi_cumsum

    psi_1 = special.psi(gamma1_)
    psi_2 = special.psi(gamma2_)
    psi_12 = special.psi(gamma1_ + gamma2_)

    bound = (np.log(alpha) + (1. - gamma1_) * (psi_1 - psi_12) +
             (alpha - gamma2_) * (psi_2 - psi_12)).sum()

    return gamma1_, gamma2_, bound


def _update_assig_params(X: np.ndarray,  # (N, D)
                         w: np.ndarray,  # (N,)
                         mu_: np.ndarray,  # (K, D)
                         zeta_: np.ndarray,  # (K, D)
                         a_: np.ndarray,  # (K, D)
                         b_: np.ndarray,  # (K, D)
                         gamma1_: np.ndarray,  # (K,)
                         gamma2_: np.ndarray):  # (K,)
    psi_a = special.psi(a_)[None]
    psi_1 = special.psi(gamma1_)[None]
    psi_2 = special.psi(gamma2_)[None]
    psi_12 = special.psi(gamma1_ + gamma2_)[None]
    psi_1_12 = psi_1 - psi_12
    psi_2_12_cum = np.roll((psi_2 - psi_12).cumsum(), 1)
    psi_2_12_cum[0] = 0.

    S = .5 * (psi_a - np.log(b_)[None]
              - (a_ / b_)[None] * ((X[:, None] - mu_[None]) ** 2 + 1. / zeta_[None])).sum(-1) \
        + psi_1_12 + psi_2_12_cum

    phi_ = np.exp(S)
    phi_ /= phi_.sum(axis=1, keepdims=True)

    K = phi_.shape[1]
    N = w.sum()

    nz = (phi_ > np.finfo(phi_.dtype).eps)
    entropy = -(phi_[nz] * np.log(phi_[nz])).sum()  # 0*log(0) = 0
    bound = (phi_ * S).sum() - .5 * N * K * np.log(2. * np.pi) + entropy

    return phi_, bound


def _update_concentration(s1: float,
                          s2: float,
                          gamma1_: np.ndarray,
                          gamma2_: np.ndarray):
    psi_2 = special.psi(gamma2_)
    psi_12 = special.psi(gamma1_ + gamma2_)
    K = gamma1_.shape[0]

    w1 = s1 + K - 1
    w2 = s2 - (psi_2 - psi_12).sum()

    return w1 / w2


def update(vdpgmm: WeightedVariationalDPGMM, fcn):
    if fcn == 'mean':
        mu_ = vdpgmm.mu
        tau_ = vdpgmm.a / vdpgmm.b
    elif fcn == 'sample':
        mu_ = stats.norm_rvs(vdpgmm.mu, vdpgmm.zeta)
        tau_ = stats.gamma_rvs(vdpgmm.a, vdpgmm.b)
    else:
        raise ValueError("fcn must be either 'mean' or 'sample'")

    vdpgmm.components = [normal.Normal(mu_[k], tau_[k]) for k in range(vdpgmm.truncation)]


def update_concentration(vdpgmm: WeightedVariationalDPGMM, s1: float, s2: float):
    vdpgmm.alpha = _update_concentration(s1, s2, vdpgmm.gamma1, vdpgmm.gamma2)
