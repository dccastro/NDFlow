import argparse
import os
import pickle

import numpy as np

import ndflow
from ndflow.estimation.fit import fit_dpgmm
from ndflow.matching import affine
from ndflow.util import plot_gmm


def _get_range(gmm: ndflow.MixtureModel):
    low = min(comp.mu - 3. / np.sqrt(comp.tau) for comp in gmm.components)
    high = max(comp.mu + 3. / np.sqrt(comp.tau) for comp in gmm.components)
    return low, high


def average_gmms(input_gmms_dir: str, output_gmm_path: str, interactive: bool = True):
    os.makedirs(os.path.dirname(output_gmm_path), exist_ok=True)

    def gmm_loader():
        for gmm_filename in os.listdir(input_gmms_dir):
            if gmm_filename.endswith(ndflow.GMM_FILENAME_SUFFIX):
                gmm_path = os.path.join(input_gmms_dir, gmm_filename)
                with open(gmm_path, 'rb') as f:
                    yield pickle.load(f)

    gmms, nums_samples = zip(*((r['gmm'], r['num_samples']) for r in gmm_loader()))
    gmms = list(gmms)

    # Affinely align all GMMs
    means_vars = np.asarray([gmm.mean_variance() for gmm in gmms]).squeeze()
    avg_mean, avg_var = means_vars.mean(axis=0)
    aligned_gmms = [affine.transform(gmm, avg_mean, avg_var)[0] for gmm in gmms]

    # Compute an average pseudo-histogram
    lows, highs = zip(*(_get_range(gmm) for gmm in aligned_gmms))
    low, high = min(lows), max(highs)
    values = np.linspace(low, high, 500)
    liks = np.asarray([gmm.marginal_likelihood(values) for gmm in aligned_gmms])
    liks /= liks.sum(axis=1, keepdims=True)
    pseudo_hist = np.asarray(nums_samples) @ liks

    # Fit 'average' GMM to average pseudo-histogram
    avg_gmm = fit_dpgmm(values, pseudo_hist, ndflow.DEFAULT_MODEL_PARAMS,
                        ndflow.DEFAULT_FIT_PARAMS)[0].prune()

    with open(output_gmm_path, 'wb') as f:
        pickle.dump({'gmm': avg_gmm, 'num_samples': pseudo_hist.sum()}, f, pickle.HIGHEST_PROTOCOL)

    if interactive:
        import matplotlib.pyplot as plt
        for gmm in aligned_gmms:
            plot_gmm(gmm, values, lw=.5, c='C1')
        plot_gmm(avg_gmm, values, values, pseudo_hist)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NDFlow - GMM averaging")
    parser.add_argument('input',
                        help="input GMMs directory")
    parser.add_argument('output',
                        help="output GMM file")
    parser.add_argument('-i', '--interactive', action='store_true',
                        help="plot the results")
    args = parser.parse_args()

    input_gmms_dir = args.input
    output_gmm_path = args.output

    average_gmms(input_gmms_dir, output_gmm_path, args.interactive)
