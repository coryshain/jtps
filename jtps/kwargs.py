from functools import cmp_to_key

class Kwarg(object):
    """
    Data structure for storing keyword arguments and their docstrings.

    :param key: ``str``; Key
    :param default_value: Any; Default value
    :param dtypes: ``list`` or ``class``; List of classes or single class. Members can also be specific required values, either ``None`` or values of type ``str``.
    :param descr: ``str``; Description of kwarg
    """

    def __init__(self, key, default_value, dtypes, descr, aliases=None):
        if aliases is None:
            aliases = []
        self.key = key
        self.default_value = default_value
        if not isinstance(dtypes, list):
            self.dtypes = [dtypes]
        else:
            self.dtypes = dtypes
        self.dtypes = sorted(self.dtypes, key=cmp_to_key(Kwarg.type_comparator))
        self.descr = descr
        self.aliases = aliases

    def dtypes_str(self):
        if len(self.dtypes) == 1:
            out = '``%s``' %self.get_type_name(self.dtypes[0])
        elif len(self.dtypes) == 2:
            out = '``%s`` or ``%s``' %(self.get_type_name(self.dtypes[0]), self.get_type_name(self.dtypes[1]))
        else:
            out = ', '.join(['``%s``' %self.get_type_name(x) for x in self.dtypes[:-1]]) + ' or ``%s``' %self.get_type_name(self.dtypes[-1])

        return out

    def get_type_name(self, x):
        if isinstance(x, type):
            return x.__name__
        if isinstance(x, str):
            return '"%s"' %x
        return str(x)

    def in_settings(self, settings):
        out = False
        if self.key in settings:
            out = True

        if not out:
            for alias in self.aliases:
                if alias in settings:
                    out = True
                    break

        return out

    def kwarg_from_config(self, settings):
        if len(self.dtypes) == 1:
            val = {
                str: settings.get,
                int: settings.getint,
                float: settings.getfloat,
                bool: settings.getboolean
            }[self.dtypes[0]](self.key, None)

            if val is None:
                for alias in self.aliases:
                    val = {
                        str: settings.get,
                        int: settings.getint,
                        float: settings.getfloat,
                        bool: settings.getboolean
                    }[self.dtypes[0]](alias, self.default_value)
                    if val is not None:
                        break

            if val is None:
                val = self.default_value

        else:
            from_settings = settings.get(self.key, None)
            if from_settings is None:
                for alias in self.aliases:
                    from_settings = settings.get(alias, None)
                    if from_settings is not None:
                        break

            if from_settings is None:
                val = self.default_value
            else:
                parsed = False
                for x in reversed(self.dtypes):
                    if x == None:
                        if from_settings == 'None':
                            val = None
                            parsed = True
                            break
                    elif isinstance(x, str):
                        if from_settings == x:
                            val = from_settings
                            parsed = True
                            break
                    else:
                        try:
                            val = x(from_settings)
                            parsed = True
                            break
                        except ValueError:
                            pass

                assert parsed, 'Invalid value "%s" received for %s' %(from_settings, self.key)

        return val

    @staticmethod
    def type_comparator(a, b):
        '''
        Types precede strings, which precede ``None``
        :param a: First element
        :param b: Second element
        :return: ``-1``, ``0``, or ``1``, depending on outcome of comparison
        '''
        if isinstance(a, type) and not isinstance(b, type):
            return -1
        elif not isinstance(a, type) and isinstance(b, type):
            return 1
        elif isinstance(a, str) and not isinstance(b, str):
            return -1
        elif isinstance(b, str) and not isinstance(a, str):
            return 1
        else:
            return 0





MODEL_KWARGS = [

    # Global
    Kwarg(
        'outdir',
        './jtps_test_model/',
        str,
        "Path to output directory, where logs and model parameters are saved."
    ),
    Kwarg(
        'task',
        'mnist',
        str,
        "Task to perform. One of ``['mnist', 'cifar']``."
    ),
    Kwarg(
        'base_optimizer',
        'AdamOptimizer',
        str,
        "Tensorflow class name of base optimizer to use. Supports any optimizer class in ``tf.train``, as well as ``'NadamOptimizer'``."
    ),
    Kwarg(
        'use_jtps',
        False,
        [bool, str],
        "Whether to modify the base optimizer using JTPS. If ``False``, runs a baseline model. If ``True``, runs a test model."
    ),
    Kwarg(
        'float_type',
        'float32',
        str,
        "``float`` type to use throughout the network."
    ),
    Kwarg(
        'int_type',
        'int32',
        str,
        "``int`` type to use throughout the network (used for tensor slicing)."
    ),

    # Encoder
    Kwarg(
        'encoder_type',
        'cnn',
        str,
        "Encoder network to use. One of ``dense``, ``cnn``, or ``rnn``."
    ),
    Kwarg(
        'encoder_conv_kernel_size',
        5,
        int,
        "Size of kernel to use in convolutional encoder layers. Ignored if no convolutional encoder layers in the model.",
        aliases=['conv_kernel_size']
    ),
    Kwarg(
        'n_layers_encoder',
        None,
        [int, None],
        "Number of layers to use for encoder. If ``None``, inferred from length of **n_units_encoder**."
    ),
    Kwarg(
        'n_units_encoder',
        None,
        [int, str, None],
        "Number of units to use in layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_encoder** space-delimited integers, one for each layer in order from bottom to top. ``None`` is not permitted and will raise an error -- it exists here simply to force users to specify a value."
    ),
    Kwarg(
        'n_max_pool_encoder',
        None,
        [int, str, None],
        "Number of units to max pool over in CNN encoder. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_encoder** space-delimited integers, one for each layer in order from bottom to top. If ``None``, no max pooling."
    ),
    Kwarg(
        'encoder_activation',
        None,
        [str, None],
        "Name of activation to use at the output of the encoder",
    ),
    Kwarg(
        'encoder_inner_activation',
        'tanh',
        [str, None],
        "Name of activation to use for any internal layers of the encoder",
        aliases=['inner_activation']
    ),
    Kwarg(
        'encoder_recurrent_activation',
        'sigmoid',
        [str, None],
        "Name of activation to use for recurrent activation in recurrent layers of the encoder. Ignored if encoder is not recurrent.",
        aliases=['recurrent_activation']
    ),
    Kwarg(
        'encoder_resnet_n_layers_inner',
        None,
        [int, None],
        "Implement internal encoder layers as residual layers with **resnet_n_layers_inner** internal layers each. If ``None``, do not use residual layers.",
        aliases=['resnet_n_layers_inner']
    ),

    # Encoder normalization
    Kwarg(
        'encoder_weight_normalization',
        False,
        bool,
        "Apply weight normalization to encoder. Ignored unless encoder is recurrent."
    ),
    Kwarg(
        'encoder_layer_normalization',
        False,
        bool,
        "Apply layer normalization to encoder. Ignored unless encoder is recurrent."
    ),
    Kwarg(
        'encoder_batch_normalization_decay',
        None,
        [float, None],
        "Decay rate to use for batch normalization in internal encoder layers. If ``None``, no batch normalization.",
        aliases=['batch_normalization_decay']
    ),

    # Encoder regularization
    Kwarg(
        'encoder_weight_regularization',
        None,
        [str, float, None],
        "If ``str``, underscore-delimited name and scale of encoder weight regularization. If ``float``, scale of encoder L2 weight regularization. If ``None``, no encoder weight regularization."
    ),
    Kwarg(
        'encoder_state_regularization',
        None,
        [str, float, None],
        "If ``str``, underscore-delimited name and scale of encoder state regularization. If ``float``, scale of encoder L2 state regularization. If ``None``, no encoder state regularization."
    ),
    Kwarg(
        'encoder_dropout',
        None,
        [float, None],
        "Dropout rate to use in the encoder",
    ),
    
    # Classifier
    Kwarg(
        'n_layers_classifier',
        None,
        [int, None],
        "Number of layers to use for classifier. If ``None``, inferred from length of **n_units_classifier**."
    ),
    Kwarg(
        'n_units_classifier',
        None,
        [int, str, None],
        "Number of units to use in layers. Can be an ``int``, which will be used for all layers, a ``str`` with **n_layers_classifier** space-delimited integers, one for each layer in order from bottom to top. ``None`` is not permitted and will raise an error -- it exists here simply to force users to specify a value."
    ),
    Kwarg(
        'classifier_activation',
        None,
        [str, None],
        "Name of activation to use at the output of the classifier",
    ),
    Kwarg(
        'classifier_inner_activation',
        'tanh',
        [str, None],
        "Name of activation to use for any internal layers of the classifier",
        aliases=['inner_activation']
    ),
    Kwarg(
        'classifier_resnet_n_layers_inner',
        None,
        [int, None],
        "Implement internal classifier layers as residual layers with **resnet_n_layers_inner** internal layers each. If ``None``, do not use residual layers.",
        aliases=['resnet_n_layers_inner']
    ),

    # Classifier normalization
    Kwarg(
        'classifier_weight_normalization',
        False,
        bool,
        "Apply weight normalization to classifier. Ignored unless classifier is recurrent."
    ),
    Kwarg(
        'classifier_layer_normalization',
        False,
        bool,
        "Apply layer normalization to classifier. Ignored unless classifier is recurrent."
    ),
    Kwarg(
        'classifier_batch_normalization_decay',
        None,
        [float, None],
        "Decay rate to use for batch normalization in internal classifier layers. If ``None``, no batch normalization.",
        aliases=['batch_normalization_decay']
    ),

    # Classifier regularization
    Kwarg(
        'classifier_weight_regularization',
        None,
        [str, float, None],
        "If ``str``, underscore-delimited name and scale of classifier weight regularization. If ``float``, scale of classifier L2 weight regularization. If ``None``, no classifier weight regularization."
    ),
    Kwarg(
        'classifier_state_regularization',
        None,
        [str, float, None],
        "If ``str``, underscore-delimited name and scale of classifier state regularization. If ``float``, scale of classifier L2 state regularization. If ``None``, no classifier state regularization."
    ),
    Kwarg(
        'classifier_dropout',
        None,
        [float, None],
        "Dropout rate to use in the classifier",
    ),

    # Optimization
    Kwarg(
        'optim_name',
        'Nadam',
        [str, None],
        """Name of the optimizer to use. Must be one of:
    
            - ``'SGD'``
            - ``'Momentum'``
            - ``'AdaGrad'``
            - ``'AdaDelta'``
            - ``'Adam'``
            - ``'FTRL'``
            - ``'RMSProp'``
            - ``'Nadam'``"""
    ),
    Kwarg(
        'max_global_gradient_norm',
        None,
        [float, None],
        'Maximum allowable value for the global norm of the gradient, which will be clipped as needed. If ``None``, no gradient clipping.'
    ),
    Kwarg(
        'epsilon',
        1e-8,
        float,
        "Epsilon for numerical stability."
    ),
    Kwarg(
        'optim_epsilon',
        1e-8,
        float,
        "Epsilon parameter to use if **optim_name** in ``['Adam', 'Nadam']``, ignored otherwise."
    ),
    Kwarg(
        'learning_rate',
        0.001,
        [float, str],
        "Initial value for the learning rate."
    ),
    Kwarg(
        'learning_rate_min',
        0.,
        float,
        "Minimum value for the learning rate."
    ),
    Kwarg(
        'lr_decay_family',
        None,
        [str, None],
        "Functional family for the learning rate decay schedule (no decay if ``None``)."
    ),
    Kwarg(
        'lr_decay_rate',
        1.,
        float,
        "coefficient by which to decay the learning rate every ``lr_decay_steps`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'lr_decay_iteration_power',
        1,
        float,
        "Power to which the iteration number ``t`` should be raised when computing the learning rate decay."
    ),
    Kwarg(
        'lr_decay_steps',
        1,
        int,
        "Span of iterations over which to decay the learning rate by ``lr_decay_rate`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'lr_decay_staircase',
        False,
        bool,
        "Keep learning rate flat between ``lr_decay_steps`` (ignored if ``lr_decay_family==None``)."
    ),
    Kwarg(
        'ema_decay',
        None,
        [float, None],
        "Decay factor to use for exponential moving average for parameters (used in prediction)."
    ),
    Kwarg(
        'minibatch_size',
        128,
        [int, None],
        "Size of minibatches to use for fitting (full-batch if ``None``)."
    ),
    Kwarg(
        'eval_minibatch_size',
        100000,
        [int, None],
        "Size of minibatches to use for prediction/evaluation (full-batch if ``None``)."
    ),

    # Checkpoint
    Kwarg(
        'save_freq',
        1,
        int,
        "Frequency with which to save model checkpoints."
    ),
    Kwarg(
        'eval_freq',
        1,
        int,
        "Frequency with which to evaluate model."
    ),
    Kwarg(
        'log_graph',
        False,
        bool,
        "Log the network graph to Tensorboard"
    )
]

def model_docstring():
    out = "**Model arguments**\n\n"

    for kwarg in MODEL_KWARGS:
        out += '- **%s**: %s; %s\n' % (kwarg.key, kwarg.dtypes_str(), kwarg.descr)

    return out
