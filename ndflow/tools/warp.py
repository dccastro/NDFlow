import argparse
import multiprocessing
import os
import pickle

import SimpleITK as sitk

from ndflow import api


def warp_single(input_img_path: str, match_path: str, ouput_img_path: str):
    img = sitk.ReadImage(input_img_path)
    data = sitk.GetArrayFromImage(img)

    with open(match_path, 'rb') as f:
        match_results = pickle.load(f)

    alignment = match_results['alignment']
    aligned_gmm = match_results['aligned_gmm']
    matched_gmm = match_results['matched_gmm']

    print(f"Warping image {input_img_path}...")
    aligned_data = alignment(data)
    warped_data = api.warp(aligned_data, aligned_gmm, matched_gmm)

    warped_img = sitk.GetImageFromArray(warped_data)
    warped_img.CopyInformation(img)
    sitk.WriteImage(warped_img, ouput_img_path)
    print(f"Warped image saved to {ouput_img_path}...")


def _input_match_and_output_paths(img_filename, input_imgs_dir, matches_dir, output_imgs_dir):
    input_img_path = os.path.join(input_imgs_dir, img_filename)
    match_path = os.path.join(matches_dir, img_filename + api.MATCH_FILENAME_SUFFIX)
    output_img_path = os.path.join(output_imgs_dir, img_filename)
    return input_img_path, match_path, output_img_path


def _warp_single(args):
    try:
        warp_single(*args)
    except RuntimeError as e:  # SimpleITK throws generic runtime exceptions
        # Probably not an image file, skip
        print(e)


def warp_group(input_imgs_dir: str, matches_dir: str, output_imgs_dir: str):
    os.makedirs(output_imgs_dir, exist_ok=True)

    def args_generator():
        for img_filename in os.listdir(input_imgs_dir):
            yield _input_match_and_output_paths(img_filename, input_imgs_dir,
                                                matches_dir, output_imgs_dir)

    with multiprocessing.Pool() as pool:
        list(pool.imap_unordered(_warp_single, args_generator()))


def main():
    parser = argparse.ArgumentParser(description="NDFlow - image intensity warping")
    parser.add_argument('input',
                        help="input image file or directory. If a directory is given, "
                             "will process all image files inside it.")
    parser.add_argument('output',
                        help="output image directory. Should be different from input directory, "
                             "as normalised image(s) will be saved with the same filename.")
    parser.add_argument('-m', '--match',
                        help="matches directory. Defaults to input image directory.")
    args = parser.parse_args()

    single = os.path.isfile(args.input)
    output_imgs_dir = args.output

    if single:
        input_imgs_dir, img_filename = os.path.split(args.input)
        matches_dir = input_imgs_dir if args.match is None else args.match
        input_img_path, match_path, output_img_path = _input_match_and_output_paths(
            img_filename, input_imgs_dir, matches_dir, output_imgs_dir)
        warp_single(input_img_path, match_path, output_img_path)
    else:
        input_imgs_dir = args.input
        matches_dir = input_imgs_dir if args.match is None else args.match
        warp_group(input_imgs_dir, matches_dir, output_imgs_dir)


if __name__ == '__main__':
    main()
