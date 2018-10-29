import numpy as np

from . import ode
from ..models.mixture import MixtureModel


class FlowBase(object):
    def simulate(self, x0, time_seq, pool=None):
        if pool is None:
            X, log_jac = self._do_simulate(x0, time_seq)
            X = np.array(X)
        else:
            def chunks(N):
                num_chunks = pool._processes
                chunk_size = (N - 1) // num_chunks + 1
                return [slice(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]

            result = pool.starmap(
                    self._do_simulate, [(x0[chunk], time_seq) for chunk in chunks(len(x0))])

            X, log_jac = list(zip(*result))
            X = np.concatenate(X, axis=1)
            log_jac = np.concatenate(log_jac, axis=1).T

        X = np.swapaxes(X, 0, 1)
        jac = np.exp(log_jac)

        return X, jac

    def velocity(self, x, t):
        raise NotImplementedError

    def _do_simulate(self, x0, time_seq):
        def update(state, state_dot, dt):
            x, log_jac = state
            u, div_u = state_dot
            x_ = x + u * dt
            log_jac_ = log_jac + div_u * dt
            return x_, log_jac_

        def state_dot(state, t):
            x, log_jac = state
            u, div_u = self.velocity(x, t)
            return u, div_u

        state0 = x0, np.zeros(len(x0))
        states, time_seq = ode.integrate(state_dot, state0, update, t=time_seq, method='rk4')
        return list(zip(*states))


class GMMFlowBase(FlowBase):
    def __init__(self, source_gmm: MixtureModel, target_gmm: MixtureModel):
        self.source_gmm = source_gmm
        self.target_gmm = target_gmm

    def get_gmm(self, t):
        return interp_gmm(self.source_gmm, self.target_gmm, t)

    def get_gmm_dot(self, gmm, t):
        raise NotImplementedError

    def get_gmm_velocity(self, gmm, gmm_dot, x, t):
        raise NotImplementedError

    def velocity(self, x, t):
        gmm = self.get_gmm(t)
        gmm_dot = self.get_gmm_dot(gmm, t)
        return self.get_gmm_velocity(gmm, gmm_dot, x, t)

    def simulate_flow(self, X0, dt, backwards=False, pool=None):
        t = np.arange(0., 1. + dt, dt)
        if backwards:
            t = t[::-1]
        return self.simulate(X0, t, pool)


def interp_gmm(gmm1: MixtureModel, gmm2: MixtureModel, s: float):
    from ..distributions import normal
    weights_ = (1 - s) * gmm1.assignment_probs() + s * gmm2.assignment_probs()
    components_ = [normal.Normal((1 - s) * t1.mu + s * t2.mu, (1 - s) * t1.tau + s * t2.tau)
                   for t1, t2 in zip(gmm1.components, gmm2.components)]
    return MixtureModel(components_, weights_)
