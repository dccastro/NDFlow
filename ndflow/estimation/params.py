from collections import namedtuple

ModelParams = namedtuple('ModelParams', [
    'concentration',
    'truncation',
    'conc_prior',
    'prior',
    'rand_init'
])

FitParams = namedtuple('FitParams', [
    'tol',
    'min_iter',
    'max_iter',
    'max_wait'
])
