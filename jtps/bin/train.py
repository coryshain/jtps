import os
import shutil
import itertools
import argparse

from jtps.config import Config
from jtps.kwargs import MODEL_KWARGS
from jtps.model import Classifier
from jtps.util import stderr


SEARCH_FIELDS = ['optim_name', 'learning_rate', 'use_jtps']
SEARCH_ABBRV = ['o', 'l', 'j']
USE_JTPS = [True]


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Trains a DNN model from a config file.
    ''')
    argparser.add_argument('config', help='Path to configuration file.')
    argparser.add_argument('-r', '--restart', action='store_true', help='Restart training even if model checkpoint exists (this will overwrite existing checkpoint)')
    argparser.add_argument('-c', '--force_cpu', action='store_true', help='Do not use GPU. If not specified, GPU usage defaults to the value of the **use_gpu_if_available** configuration parameter.')
    args = argparser.parse_args()

    p = Config(args.config)

    if args.force_cpu or not p['use_gpu_if_available']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    stderr('Initializing model...\n\n')

    if args.restart and os.path.exists(p.outdir + '/tensorboard'):
        shutil.rmtree(p.outdir + '/tensorboard')

    kwargs = {}
    for kwarg in MODEL_KWARGS:
        kwargs[kwarg.key] = p[kwarg.key]

    grid_search = []
    for f, a in zip(SEARCH_FIELDS, SEARCH_ABBRV):
        dtype = type(kwargs[f])
        val = [(f, a, dtype(x)) for x in str(kwargs[f]).split()]
        grid_search.append(val)

    grid_search = itertools.product(*grid_search)

    for settings in grid_search:
        print(settings)
        kwargs_cur = kwargs.copy()

        outdir = []

        for triple in settings:
            f, a, v = triple
            outdir += ['%s%s' % (a, str(v))]
            kwargs_cur[f] = v

        kwargs_cur['outdir'] = kwargs_cur['outdir'] + '/%s' % '_'.join(outdir)

        model = Classifier(
            **kwargs_cur
        )

        model.build(seed=12345)

        model.fit(p['n_iter'])

        del model



