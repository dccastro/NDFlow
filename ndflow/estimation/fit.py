import time
from copy import deepcopy
from typing import List

import numpy as np

from . import weighted_variational
from ..distributions import normal
from ..models.weighted_variational import WeightedVariationalDPGMM


def _get_prior(values, weights, model_params):
    alpha, nu = model_params.prior['a'], model_params.prior['nu']
    loc = np.average(values, axis=0, weights=weights)
    scale = np.sqrt(np.average(values ** 2, axis=0, weights=weights) - loc ** 2)
    mu = loc
    tau = (1. + nu) / (nu * scale ** 2)
    prior = normal.NormalGamma(mu, nu, alpha, tau)
    return prior


def _init_dpgmm(values, weights, model_params):
    prior = _get_prior(values, weights, model_params)
    dpgmm = WeightedVariationalDPGMM(weights, model_params.concentration, prior,
                                     model_params.truncation, values.shape[0])
    if model_params.rand_init:
        weighted_variational.init(dpgmm, values, weights)
    return dpgmm


def fit_dpgmm(values, weights, model_params, fit_params, verbose=False):
    """Fits a Dirichlet process Gaussian mixture model (DPGMM) to the given data.

    Estimates the parameters of a DPGMM using weighted variational inference, based on a
    histogram of the data. This is considerably faster than using the data as-is, especially if
    the range of values is bounded and the data is very large (e.g. 2D/3D images).

    Parameters
    ----------
    values : array_like
    weights : array_like

    model_params : ndflow.estimation.params.ModelParams
        Hyperparameters of the model:

        - `concentration`: (float) Dirichlet process concentration parameter (>0).
        - `truncation`: (int) Maximum number of mixture components.
        - `conc_prior`: (Tuple[float, float] or None) Shape and rate parameters (>0) of a gamma
          prior on the concentration hyperparameter. If `None`, it is assumed fixed, otherwise
          posterior inference will also be performed for the concentration.
        - `prior`: (dict with 'a' and 'nu' keys and float values) Parameters of the normal-gamma
          prior on the mixture components. 'a' is the shape parameter (>0) and 'nu' is the precision
          multiplier (>0).
        - `rand_init`: (bool) Whether to initialise the mixture weights with a random sample from
          the stick-breaking process or with its expectation.

    fit_params : ndflow.estimation.params.FitParams
        Optimisation settings:

        - `tol`: (float) Relative tolerance for improvements in the objective function.
        - `min_iter`, `max_iter`: (int) Minimum and maximum numbers of iterations.
        - `max_wait`: (int) Maximum number of additional iterations to execute after stopping
          criterion is met, in expectation of further improvement.

    verbose : bool
        Whether to print detailed information at each iteration.

    Returns
    -------
    best_dpgmm : WeightedVariationalDPGMM
        Fitted model.
    best_elbo : float
        ELBO of the fitted model.
    best_iter : int
        Number of iterations until the best model was reached.
    elbos : List[float]
        Training curve of ELBOs.
    """
    values = np.asarray(values).reshape(-1, 1)
    weights = np.asarray(weights)

    dpgmm = _init_dpgmm(values, weights, model_params)

    elbos = []
    best_elbo = float('-inf')
    best_dpgmm = dpgmm
    best_iter = -1
    wait = 0
    for it in range(1, fit_params.max_iter + 1):
        t0 = time.perf_counter()

        _, elbo = weighted_variational.step(dpgmm, values, weights)
        if model_params.conc_prior is not None:
            weighted_variational.update_concentration(dpgmm, *model_params.conc_prior)

        dt = time.perf_counter() - t0
        if verbose:
            print(f"> Iteration {it:3d} ({dt:.3f} s): ELBO = {elbo}")

        elbos.append(elbo)

        if it >= fit_params.min_iter:
            if np.isfinite(best_elbo):
                improvement = (elbos[-1] - best_elbo) / abs(best_elbo)
            else:
                improvement = float('inf')
            if improvement > fit_params.tol:
                best_elbo = elbos[-1]
                best_dpgmm = deepcopy(dpgmm)
                best_iter = it
                wait = 0
            else:
                wait += 1

            if wait >= fit_params.max_wait:
                break

    return best_dpgmm, best_elbo, best_iter, elbos
