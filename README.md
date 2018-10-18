# NDFlow

This is the code repository for [_Nonparametric Density Flows for MRI Intensity Normalisation_](http://link.springer.com/10.1007/978-3-030-00928-1_24), 
presented at the MICCAI 2018 conference. We have also made available an [extended manuscript on 
arXiv](https://arxiv.org/abs/1806.02613), including some mathematical proofs and additional 
experimental results.

If you would like to use this tool, please consider citing our work according to the following 
BibTeX entry:
```
@inproceedings{castro2018ndflow,
    title = {Nonparametric Density Flows for {MRI} Intensity Normalisation},
    author = {Castro, Daniel C. and Glocker, Ben},
    year = {2018}
    booktitle = {Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2018},
    pages = {206--214},
    series = {LNCS},
    volume = {11070},
    publisher = {Springer, Cham},
    address = {Granada, Spain},
    doi = {10.1007/978-3-030-00928-1_24},
    eprint = {arXiv:1806.02613},
}
```

## Dependencies
- Python 3.6 (not tested with earlier versions)
- `numpy`
- `scipy`
- `SimpleITK` (only for the scripts)
- `matplotlib` (only for the scripts and if you use the `--interactive` option)

## API Usage
The most important functionality is made available in the `ndflow` namespace, which includes the 
following functions (see corresponding docstrings in `ndflow.__init__.py` for details):
- `ndflow.estimate`: GMM fitting
- `ndflow.align`: affine alignment between GMMs
- `ndflow.match`: minimise L2 divergence between GMMs
- `ndflow.warp`: simulate diffeomorphic flow between two GMM densities

These functions should be used if dealing directly with `numpy` arrays and
`ndflow.models.mixture.MixtureModel` instances.

## CLI Usage
We mirror the API with scripts that also handle I/O (with SimpleITK) and workflow-specific tasks 
like GMM averaging (run them with the `-h` for usage instructions):
- `estimate.py`: fit GMMs to one or a collection of image files
- `average.py`: compute an 'average' GMM out of a collection of GMMs
- `match.py`: match one or a collection of GMMs to a target GMM
- `warp.py`: transform intensities in one or a collection of images according to the respective 
matched GMMs

These scripts wrap functions which may also be used programmatically.  

### Group Normalisation
- `imgs_dir`: Input images directory
- `gmms_dir`: Directory for the estimated Gaussian mixture models (GMMs)
- `average_gmm_file`: File path for the average GMM (i.e. 'intensity atlas')
- `matches_dir`: Directory for density matching results
- `out_imgs_dir`: Normalised images directory (different from `imgs_dir`, as images will be saved
 with the same filename)
```
> python estimate.py <imgs_dir> <gmms_dir>
> python average.py <gmms_dir> <average_gmm_file>
> python match.py <gmms_dir> <average_gmm_file> -m <matches_dir>
> python warp.py <imgs_dir> <matches_dir> -o <out_imgs_dir>
```

### Image-to-Image Normalisation
- `src_img_file`: Source image file
- `tgt_img_file`: Target image file
- `src_gmm_file`: Source GMM file (in source image directory with `src_img_file` prefix)
- `tgt_gmm_file`: Target GMM file (in target image directory with `tgt_img_file` prefix)
- `out_img_dir`: Normalised image directory (different from `imgs_dir`, as image will be saved
 with the same filename)
```
> python estimate.py <src_img_file>
> python estimate.py <tgt_img_file>
> python match.py <src_gmm_file> <tgt_gmm_file>
> python warp.py <src_img_file> <out_img_dir>
```
