import numpy as np

from ..models.mixture import MixtureModel


class GMML2Norm(object):
    def __init__(self, gmm: MixtureModel):
        self.gmm = gmm

        self.weights = gmm.assignment_probs()
        self.weight_prod = np.outer(self.weights, self.weights)

        prod = np.ones((gmm.num_components, gmm.num_components))
        for m in range(gmm.num_components):
            prod[m, m] *= gmm.components[m].sqnorm()
            for m_ in range(m):
                prod[m, m_] = prod[m_, m] = gmm.components[m].product(gmm.components[m_])
        self.component_prod = prod

    def sqnorm(self) -> float:
        return (self.weight_prod * self.component_prod).sum()

    def component_prod_gmm(self, k: int) -> float:
        return (self.weight_prod[k] * self.component_prod[k]).sum() / self.weights[k]


class GMML2Product(object):
    def __init__(self, gmm1: MixtureModel, gmm2: MixtureModel):
        self.gmm1 = gmm1
        self.gmm2 = gmm2

        self.weights1 = gmm1.assignment_probs()
        self.weights2 = gmm2.assignment_probs()
        self.weight_prod = np.outer(self.weights1, self.weights2)

        prod = np.ones((gmm1.num_components, gmm2.num_components))
        for m in range(gmm1.num_components):
            for k in range(gmm2.num_components):
                prod[m, k] = gmm1.components[m].product(gmm2.components[k])
        self.component_prod = prod

    def product(self) -> float:
        return (self.weight_prod * self.component_prod).sum()

    def component1_prod_gmm2(self, k: int) -> float:
        return (self.weight_prod[k, :] * self.component_prod[k, :]).sum() / self.weights1[k]

    def component2_prod_gmm1(self, k: int) -> float:
        return (self.weight_prod[:, k] * self.component_prod[:, k]).sum() / self.weights2[k]


def l2(norm1: GMML2Norm, norm2: GMML2Norm, prod12: GMML2Product):
    dmk = prod12.product()
    dmm = norm1.sqnorm()
    dkk = norm2.sqnorm()

    div = -dmk + .5 * dmm + .5 * dkk

    return np.max([0, div])


def cauchy_schwartz(norm1: GMML2Norm, norm2: GMML2Norm, prod12: GMML2Product):
    dmk = prod12.product()
    dmm = norm1.sqnorm()
    dkk = norm2.sqnorm()

    div = -np.log(dmk) + .5 * np.log(dmm) + .5 * np.log(dkk)

    return np.max([0, div])
