import argparse
import multiprocessing
import os
import pickle

import SimpleITK as sitk

import ndflow
from ndflow.estimation import params


def estimate_single(img_path: str, gmm_path: str, background: float = None,
                    model_params: params.ModelParams = ndflow.DEFAULT_MODEL_PARAMS,
                    fit_params: params.FitParams = ndflow.DEFAULT_FIT_PARAMS):
    img = sitk.ReadImage(img_path)
    data = sitk.GetArrayFromImage(img)
    if background is not None:
        data = data[data > background]  # Exclude background

    print(f"Estimating density for image {img_path}...")
    gmm = ndflow.estimate(data, model_params, fit_params)

    with open(gmm_path, 'wb') as f:
        pickle.dump({'gmm': gmm, 'num_samples': data.size}, f, pickle.HIGHEST_PROTOCOL)
    print(f"GMM saved to {gmm_path}")

    return gmm


def _img_and_gmm_paths(img_filename, imgs_dir, gmms_dir):
    img_path = os.path.join(imgs_dir, img_filename)
    gmm_path = os.path.join(gmms_dir, img_filename + ndflow.GMM_FILENAME_SUFFIX)
    return img_path, gmm_path


def _estimate_single(args):
    try:
        img_filename, imgs_dir, gmms_dir, background, model_params, fit_params = args
        img_path, gmm_path = _img_and_gmm_paths(img_filename, imgs_dir, gmms_dir)
        return estimate_single(img_path, gmm_path, background, model_params, fit_params)
    except RuntimeError as e:  # SimpleITK throws generic runtime exceptions
        # Probably not an image file, skip
        print(e)
        return None


def estimate_group(imgs_dir: str, gmms_dir: str, background: float,
                   model_params: params.ModelParams = ndflow.DEFAULT_MODEL_PARAMS,
                   fit_params: params.FitParams = ndflow.DEFAULT_FIT_PARAMS):
    os.makedirs(gmms_dir, exist_ok=True)

    def args_generator():
        for img_filename in os.listdir(imgs_dir):
            if os.path.isfile(os.path.join(imgs_dir, img_filename)):
                yield img_filename, imgs_dir, gmms_dir, background, model_params, fit_params

    with multiprocessing.Pool() as pool:
        list(pool.imap_unordered(_estimate_single, args_generator()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NDFlow - density estimation")
    parser.add_argument('input',
                        help="input image file or directory. If a directory is given, "
                             "will process all image files inside it.")
    parser.add_argument('output',
                        help="output directory to store density estimation results (*.pickle).")
    parser.add_argument('-b', '--background', type=float,
                        help="threshold for background intensities. If given, voxels with "
                             "intensity <= background will be excluded, otherwise the entire "
                             "image will be used.")
    args = parser.parse_args()

    single = os.path.isfile(args.input)
    imgs_dir, img_filename = os.path.split(args.input)
    gmms_dir = imgs_dir if args.output is None else args.output

    if single:
        img_path, gmm_path = _img_and_gmm_paths(img_filename, imgs_dir, gmms_dir)
        estimate_single(img_path, gmm_path, args.background)
    else:
        imgs_dir = args.input
        estimate_group(imgs_dir, gmms_dir, args.background)
