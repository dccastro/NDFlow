from typing import Dict, Union

import numpy as np
from scipy.interpolate import interp1d

from . import util
from .estimation import params
from .estimation.fit import fit_dpgmm
from .matching.affine import align
from .matching.match import match_gmms
from .models.mixture import MixtureModel
from .warping import flow

DEFAULT_MODEL_PARAMS = params.ModelParams(
    concentration=.5,
    truncation=20,
    conc_prior=None,
    prior={'a': 2., 'nu': 0.1},
    rand_init=True
)

DEFAULT_FIT_PARAMS = params.FitParams(
    tol=1e-5,
    min_iter=100,
    max_iter=500,
    max_wait=10
)

GMM_FILENAME_SUFFIX = "_gmm.pickle"
MATCH_FILENAME_SUFFIX = "_match.pickle"


def estimate(data: np.ndarray,
             model_params: params.ModelParams = DEFAULT_MODEL_PARAMS,
             fit_params: params.FitParams = DEFAULT_FIT_PARAMS,
             interactive: bool = False) -> MixtureModel:
    levels = None
    values, weights = util.quantise(data, levels)
    dpgmm, best_loglik, best_iter, logliks = fit_dpgmm(values, weights, model_params, fit_params)
    gmm = dpgmm.prune()

    if interactive:
        import matplotlib.pyplot as plt
        util.plot_gmm(gmm, values, values, weights)
        plt.show()

    return gmm


def match(source_gmm: MixtureModel, target_gmm: MixtureModel, n_iter: int = 200,
          lrate: Union[float, Dict[str, float]] = None, var_reg: float = None,
          use_dcs: bool = False) -> MixtureModel:
    if lrate is None:
        lrate = {'means': 1e-1, 'precs': 1e1}
    gmm_path = match_gmms(source_gmm, target_gmm, 'ml', lrate, n_iter, var_reg, use_dcs)[0]
    matched_gmm = gmm_path[-1]
    return matched_gmm


def warp(data: np.ndarray, source_gmm: MixtureModel, matched_gmm: MixtureModel, dt: float = .01,
         n_mesh: int = 200):
    gmm_flow = flow.GMMFlow(source_gmm, matched_gmm)

    mesh = np.linspace(data.min(), data.max(), n_mesh)
    mesh_trajectories, jac = gmm_flow.simulate_flow(mesh, dt)
    transformed_mesh = mesh_trajectories[:, -1]

    transformation = interp1d(mesh, transformed_mesh, fill_value='extrapolate')
    transformed_data = transformation(data)

    return transformed_data
