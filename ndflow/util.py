import os

import numpy as np

import ndflow
from ndflow.models.mixture import MixtureModel


def list_images(imgs_dir):
    import SimpleITK as sitk

    for filename in os.listdir(imgs_dir):
        path = os.path.join(imgs_dir, filename)
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        try:
            reader.ReadImageInformation()
            yield filename
        except RuntimeError:
            continue  # Probably not an image file, skip


def list_gmms(gmms_dir):
    return (filename for filename in os.listdir(gmms_dir)
            if filename.endswith(ndflow.GMM_FILENAME_SUFFIX))


def list_matches(matches_dir):
    return (filename for filename in os.listdir(matches_dir)
            if filename.endswith(ndflow.MATCH_FILENAME_SUFFIX))


def quantise(data, levels: int = None):
    """Quantise data into discrete values, similarly to a histogram.

    Parameters
    ----------
    data : array_like
        Input data array.
    levels : int or None, optional
        Number of levels at which to quantise the data. If `None`, data will be cast to `int` and
        integer values in the data range will be used.

    Returns
    -------
    values : np.ndarray
        Values to which `data` was quantised.
    weights : np.ndarray
        Array of counts of items collapsed into each of the `values`.
    """
    data = np.asarray(data).flatten()
    if levels is None:
        data = data.astype(int)
        data_min = data.min()
        weights = np.bincount(data - data_min)
        values = np.arange(len(weights), dtype=int) + data_min
    else:
        weights, bins = np.histogram(data, bins=levels, density=False)
        values = .5 * (bins[:-1] + bins[1:])  # Bin centres
    return values, weights


def plot_gmm(gmm: MixtureModel, x, values=None, weights=None, ax=None, **kwargs):
    """Plot a Gaussian mixture model (GMM) density.

    Parameters
    ----------
    gmm : ndflow.models.mixture.MixtureModel
    x : array_like
        Values at which to evaluate the GMM likelihood.
    values, weights : np.ndarray, optional
        Quantised data distribution as computed by `quantise()`. If given, will plot a histogram
        alongside the GMM density.
    ax : matplotlib.axes.Axes, optional
        Axes onto which to draw. Defaults to the current axes.
    kwargs
        Keyword arguments passed through to the `plot()` call.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if values is not None and weights is not None:
        # Compute histogram bars' parameters in case values are not evenly spaced
        widths = np.empty(values.shape[0])
        widths[1:] = values[1:] - values[:-1]
        widths[0] = widths[1]
        edges = values - .5 * widths
        heights = weights / (weights.sum() * widths)

        ax.bar(edges, heights, widths, align='edge', linewidth=0, alpha=.5)

    ax.plot(x, gmm.marginal_likelihood(x), **kwargs)
