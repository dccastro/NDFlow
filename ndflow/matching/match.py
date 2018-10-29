from copy import deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np

from . import _means, _precs, _weights
from .divergences import GMML2Norm, GMML2Product, cauchy_schwartz, l2
from ..distributions.normal import Normal
from ..models.mixture import MixtureModel


def standardise_gmm(gmm: MixtureModel):
    mean, var = gmm.mean_variance()
    components_ = [Normal((t.mu - mean) / np.sqrt(var), var * t.tau) for t in gmm.components]

    def transform(x):
        return (x - mean[None]) / np.sqrt(var)[None]

    return MixtureModel(components_, gmm.assignment_probs()), transform


def match_gmms(gmm1: MixtureModel, gmm2: MixtureModel, opt: str,
               lrate: Union[float, Dict[str, float]], n_iter: int, var_reg: float = None,
               use_dcs: bool = False) -> Tuple[List[MixtureModel], np.ndarray]:
    divs = np.zeros(n_iter)

    norm1 = GMML2Norm(gmm1)
    norm2 = GMML2Norm(gmm2)
    prod12 = GMML2Product(gmm1, gmm2)

    # TODO Fix gradient scale
    gmm1s = [None] * n_iter  # type: List[MixtureModel]
    for i in range(n_iter):
        result = update(gmm1, gmm2, opt, lrate, var_reg, norm1, norm2, prod12, use_dcs)
        gmm1, divs[i] = result['gmm1'], result['div']
        norm1, prod12 = result['norm1'], result['prod12']
        gmm1s[i] = gmm1

    return gmm1s, divs


def update(gmm1: MixtureModel, gmm2: MixtureModel, opt: str,
           lrate: Union[float, Dict[str, float]], var_reg: float,
           gmm_norm1: GMML2Norm = None, gmm_norm2: GMML2Norm = None,
           gmm_prod12: GMML2Product = None, use_dcs: bool = False):
    if gmm_norm1 is None:
        gmm_norm1 = GMML2Norm(gmm1)
    if gmm_norm2 is None:
        gmm_norm2 = GMML2Norm(gmm2)
    if gmm_prod12 is None:
        gmm_prod12 = GMML2Product(gmm1, gmm2)

    if not isinstance(lrate, Dict):
        lrate = {p: lrate for p in ['weights', 'means', 'precs']}

    grad = gradient(gmm1, gmm2, opt, gmm_norm1, gmm_prod12, use_dcs)
    result = _update_gmm(gmm1, gmm2, gmm_norm2, grad, lrate, opt, var_reg, use_dcs)
    return result


def gradient(gmm1: MixtureModel, gmm2: MixtureModel, opt: str,
             gmm_norm1: GMML2Norm = None, gmm_prod12: GMML2Product = None,
             use_dcs: bool = False):
    if gmm_norm1 is None:
        gmm_norm1 = GMML2Norm(gmm1)
    if gmm_prod12 is None:
        gmm_prod12 = GMML2Product(gmm1, gmm2)

    W = gmm_norm1.weight_prod * gmm_norm1.component_prod
    V = gmm_prod12.weight_prod * gmm_prod12.component_prod

    if use_dcs:
        W /= gmm_norm1.sqnorm()
        V /= gmm_prod12.product()

    grad = {
        'weights': _weights.grads(gmm1, gmm2, W, V) if 'w' in opt else [],
        'means': _means.grads(gmm1, gmm2, W, V) if 'm' in opt else [],
        'precs': _precs.grads(gmm1, gmm2, W, V) if 'p' in opt else []
    }
    return grad


def _update_gmm(gmm1, gmm2, norm2, grad, param_lrates, opt, var_reg, use_dcs):
    gmm1_ = deepcopy(gmm1)
    reg_cost = 0.

    if 'w' in opt:
        _weights.update(gmm1_, grad['weights'], param_lrates['weights'])
    if 'm' in opt:
        _means.update(gmm1_, grad['means'], param_lrates['means'])
    if 'p' in opt:
        reg_cost = _precs.update(gmm1_, grad['precs'], param_lrates['precs'], var_reg)

    norm1_ = GMML2Norm(gmm1_)
    prod12_ = GMML2Product(gmm1_, gmm2)
    if use_dcs:
        div = cauchy_schwartz(norm1_, norm2, prod12_)
    else:
        div = l2(norm1_, norm2, prod12_)

    result = {
        'div'   : div,
        'reg'   : reg_cost,
        'obj'   : div if var_reg is None else (div + var_reg * reg_cost),
        'lrate' : param_lrates,
        'gmm1'  : gmm1_,
        'norm1' : norm1_,
        'prod12': prod12_
    }
    return result
