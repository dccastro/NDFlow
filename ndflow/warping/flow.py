from ._flow_base import GMMFlowBase
from ..dist import normal
from ..models.mixture import MixtureModel


def _q_k(x, theta):
    q_k = theta.likelihood(x)
    q_k_prime = -((theta.tau * (x - theta.mu)).T * q_k).T
    return q_k, q_k_prime


def _u_k(x, theta, theta_dot, t):
    mu_dot, tau_dot = theta_dot
    u_k_prime = -tau_dot / (2. * theta.tau)
    u_k = mu_dot + u_k_prime * (x - theta.mu)
    return u_k, u_k_prime


def velocity(gmm: MixtureModel, gmm_dot, x, t):
    u = 0
    u_prime = 0
    q_q_prime_ = [_q_k(x, tk) for tk in gmm.components]
    pi = gmm.weights
    q = sum(pi_k * qqp_k[0] for pi_k, qqp_k in zip(pi, q_q_prime_))
    q_prime = sum(pi_k * qqp_k[1] for pi_k, qqp_k in zip(pi, q_q_prime_))
    for k, theta_k in enumerate(gmm.components):
        q_k, q_prime_k = q_q_prime_[k]
        post_k = pi[k] * q_k / q
        u_k, u_k_prime = _u_k(x, theta_k, gmm_dot[k], t)

        u += post_k * u_k
        u_prime += (pi[k] * q_prime_k - post_k * q_prime) / q * u_k
        u_prime += post_k * u_k_prime
    return u, u_prime


def _update_gmm(gmm: MixtureModel, gmm_dot, dt):
    theta1_new = [normal.Normal(tk.mu + tk_dot[0] * dt,
                                tk.tau + tk_dot[1] * dt)
                  for tk, tk_dot in zip(gmm.components, gmm_dot)]
    return MixtureModel(theta1_new, gmm.weights)


class GMMFlow(GMMFlowBase):
    def get_gmm_dot(self, gmm, t):
        return [(tk_tgt.mu - tk_src.mu, tk_tgt.tau - tk_src.tau)
                for tk_src, tk_tgt in zip(self.source_gmm.components, self.target_gmm.components)]

    def get_gmm_velocity(self, gmm, gmm_dot, x, t):
        return velocity(gmm, gmm_dot, x, t)
