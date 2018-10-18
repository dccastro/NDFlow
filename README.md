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

## Typical Group Normalisation Workflow
- `imgs_dir`: Input images directory
- `gmms_dir`: Directory for the estimated Gaussian mixture models (GMMs)
- `average_gmm_file`: File path for the average GMM (i.e. 'intensity atlas')
- `matches_dir`: Directory for density matching results
- `out_imgs_dir`: Normalised images directory (different from `imgs_dir`, as images will be saved
 with the same filename)
```
> python estimate.py <imgs_dir> <gmms_dir>
> python average.py <gmms_dir> <average_gmm_file>
> python match.py <gmms_dir> <average_gmm_file> <matches_dir>
> python warp.py <imgs_dir> <matches_dir> <out_imgs_dir>
```

## Typical Image-to-Image Normalisation Workflow
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
