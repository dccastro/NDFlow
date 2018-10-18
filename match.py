import argparse
import multiprocessing
import os
import pickle

import ndflow


def match_single(source_gmm_path: str, target_gmm_path: str, output_path: str):
    with open(source_gmm_path, 'rb') as f:
        source_gmm = pickle.load(f)['gmm']
    with open(target_gmm_path, 'rb') as f:
        target_gmm = pickle.load(f)['gmm']

    print(f"Aligning GMM {source_gmm_path} to {target_gmm_path}...")
    aligned_gmm, alignment = ndflow.align(source_gmm, target_gmm)
    matched_gmm = ndflow.match(aligned_gmm, target_gmm)

    with open(output_path, 'wb') as f:
        pickle.dump({'alignment': alignment,
                     'aligned_gmm': aligned_gmm,
                     'matched_gmm': matched_gmm},
                    f, pickle.HIGHEST_PROTOCOL)
    print(f"Matched GMM saved to {output_path}")


def _gmm_and_match_paths(gmm_filename, gmms_dir, matches_dir):
    gmm_path = os.path.join(gmms_dir, gmm_filename)
    match_filename = gmm_filename.replace(ndflow.GMM_FILENAME_SUFFIX, ndflow.MATCH_FILENAME_SUFFIX)
    match_path = os.path.join(matches_dir, match_filename)
    return gmm_path, match_path


def _match_single(args):
    return match_single(*args)


def match_group(source_gmms_dir: str, target_gmm_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    def args_generator():
        for gmm_filename in os.listdir(source_gmms_dir):
            if gmm_filename.endswith(ndflow.GMM_FILENAME_SUFFIX):
                source_gmm_path, output_path = \
                    _gmm_and_match_paths(gmm_filename, source_gmms_dir, output_dir)
                yield source_gmm_path, target_gmm_path, output_path

    with multiprocessing.Pool() as pool:
        list(pool.imap_unordered(_match_single, args_generator()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NDFlow - density matching")
    parser.add_argument('source',
                        help="source GMM file or directory. If a directory is given, "
                             "will process all GMM files inside it.")
    parser.add_argument('target',
                        help="target GMM file")
    parser.add_argument('-o', '--output',
                        help="output directory. Defaults to source directory.")
    args = parser.parse_args()

    single = os.path.isfile(args.source)
    source_gmms_dir, source_gmm_filename = os.path.split(args.source)
    target_gmm_path = args.target

    if single:
        output_dir = source_gmms_dir if args.output is None else args.output
        source_gmm_path, output_path = \
            _gmm_and_match_paths(source_gmm_filename, source_gmms_dir, output_dir)
        match_single(source_gmm_path, target_gmm_path, output_path)
    else:
        source_gmms_dir = args.source
        output_dir = source_gmms_dir if args.output is None else args.output
        match_group(source_gmms_dir, target_gmm_path, output_dir)
