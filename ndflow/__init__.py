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


def estimate(data: np.ndarray, levels: int = None,
             model_params: params.ModelParams = DEFAULT_MODEL_PARAMS,
             fit_params: params.FitParams = DEFAULT_FIT_PARAMS,
             interactive: bool = False) -> MixtureModel:
    """Estimates the density of the data histogram.

    Fits a Dirichlet process Gaussian mixture model (DPGMM) to a 1D histogram of the data. Default
    arguments work well in practice.

    Parameters
    ----------
    data : array_like
        Input data array. Can be of any shape, but will be flattened.
    levels : int or None, optional
        Number of levels at which to quantise the data. If `None` (default), data will be cast to
        `int` and integer values in the data range will be used.
    model_params : ndflow.estimation.params.ModelParams, optional
        Hyperparameters of the DPGMM. See `ndflow.fit.fit_dpgmm` for details.
    fit_params : ndflow.estimation.params.FitParams, optional
        Optimisation settings. See `ndflow.fit.fit_dpgmm` for details.
    interactive : bool
        Whether to plot the histogram and estimated GMM density (requires `matplotlib`).

    Returns
    -------
    ndflow.models.mixture.MixtureModel
        The estimated GMM.

    See Also
    --------
    ndflow.fit.fit_dpgmm
    """
    values, weights = util.quantise(data, levels)
    dpgmm, best_loglik, best_iter, logliks = fit_dpgmm(values, weights, model_params, fit_params)
    gmm = dpgmm.prune()

    if interactive:
        import matplotlib.pyplot as plt
        util.plot_gmm(gmm, values, values, weights)
        plt.show()

    return gmm


def match(source_gmm: MixtureModel, target_gmm: MixtureModel, n_iter: int = 200,
          lrate: Union[float, Dict[str, float]] = None, var_reg: float = None) -> MixtureModel:
    """Matches two Gaussian mixture models (GMMs) by minimising their L2 divergence.

    Default arguments work well in practice.

    Parameters
    ----------
    source_gmm : ndflow.models.mixture.MixtureModel
        Source GMM.
    target_gmm : ndflow.models.mixture.MixtureModel
        Target GMM.
    n_iter : int, optional
        Number of iterations.
    lrate : float or dict, optional
        Learning rate/step size. Single value or dictionary with keys `'means'` and `'precs'` for
        mixture components' means and precisions, respectively.
    var_reg : float, optional
        Variance regularisation parameter. If `None` (default), no regularisation is applied.

    Returns
    -------
    ndflow.models.mixture.MixtureModel
        The resulting GMM after matching `source_gmm` to `target_gmm`.
    """
    if lrate is None:
        lrate = {'means': 1e-1, 'precs': 1e1}
    gmm_path = match_gmms(source_gmm, target_gmm, 'ml', lrate, n_iter, var_reg, use_dcs=False)[0]
    matched_gmm = gmm_path[-1]
    return matched_gmm


def warp(data: np.ndarray, source_gmm: MixtureModel, matched_gmm: MixtureModel, dt: float = .01,
         n_mesh: int = 200) -> np.ndarray:
    """Warps the data through a diffeomorphic density flow between two GMM densities.

    Transports a mesh in the range of the data through a density flow with numerical integration,
    then transforms the data with piecewise linear interpolation. Default arguments work well in
    practice.

    Parameters
    ----------
    data : array_like
        Input data array, of any shape.
    source_gmm : ndflow.models.mixture.MixtureModel
        Source GMM.
    matched_gmm : ndflow.models.mixture.MixtureModel
        Matched GMM corresponding to `source_gmm`, as obtained via `ndflow.match()`.
    dt : float, optional
        Integration step size.
    n_mesh : int, optional
        Number of points in the mesh used to interpolate the results.

    Returns
    -------
    np.ndarray
        The transformed data, with the same shape as `data`.
    """
    gmm_flow = flow.GMMFlow(source_gmm, matched_gmm)

    mesh = np.linspace(data.min(), data.max(), n_mesh)
    mesh_trajectories, jac = gmm_flow.simulate_flow(mesh, dt)
    transformed_mesh = mesh_trajectories[:, -1]

    transformation = interp1d(mesh, transformed_mesh, fill_value='extrapolate')
    transformed_data = transformation(data)

    return transformed_data
