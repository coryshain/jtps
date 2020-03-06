import sys
import os
import shutil
if sys.version_info[0] == 2:
    import ConfigParser as configparser
else:
    import configparser

from .kwargs import MODEL_KWARGS

class Config(object):
    def __init__(self, path):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(path)

        # SETTINGS
        # Output directory
        settings = config['settings']
        self.outdir = settings.get('outdir', None)
        if self.outdir is None:
            self.outdir = settings.get('logdir', None)
        if self.outdir is None:
            self.outdir = './jtps_test_model/'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        if not os.path.normpath(os.path.realpath(path)) == os.path.normpath(os.path.realpath(self.outdir + '/config.ini')):
            shutil.copy2(path, self.outdir + '/config.ini')

        # Process config settings
        self.model_settings = self.build_model_settings(settings)
        self.model_settings['n_iter'] = settings.getint('n_iter', 1000)
        gpu_frac = settings.get('gpu_frac', None)
        if gpu_frac in [None, 'None']:
            gpu_frac = None
        else:
            try:
                gpu_frac = float(gpu_frac)
            except:
                raise ValueError('gpu_frac parameter invalid: %s' % gpu_frac)
        self.model_settings['gpu_frac'] = gpu_frac
        self.model_settings['use_gpu_if_available'] = settings.getboolean('use_gpu_if_available', True)

    def __getitem__(self, item):
        return self.model_settings[item]

    def build_model_settings(self, settings):
        out = {}

        # Initialization keyword arguments
        out['outdir'] = self.outdir
        for kwarg in MODEL_KWARGS:
            out[kwarg.key] = kwarg.kwarg_from_config(settings)

        return out


