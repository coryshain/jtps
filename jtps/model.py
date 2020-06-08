import sys
import os
import math
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from jtps.kwargs import MODEL_KWARGS
from jtps.backend import *
from jtps.opt import get_clipped_optimizer_class, get_JTPS_optimizer_class
from jtps.util import *


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


class Classifier(object):
    ############################################################
    # Initialization methods
    ############################################################

    _INITIALIZATION_KWARGS = MODEL_KWARGS

    _doc_header = """
        DNN model for classification tasks.
    """
    _doc_kwargs = '\n'.join([' ' * 8 + ':param %s' % x.key + ': ' + '; '.join(
        [x.dtypes_str(), x.descr]) + ' **Default**: ``%s``.' % (
                                 x.default_value if not isinstance(x.default_value, str) else "'%s'" % x.default_value)
                             for
                             x in _INITIALIZATION_KWARGS])
    __doc__ = _doc_header + _doc_kwargs

    def __init__(self, **kwargs):
        for kwarg in Classifier._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, kwargs.pop(kwarg.key, kwarg.default_value))

        data = getattr(tf.keras.datasets, self.task.lower())
        (x_train, y_train), (x_test, y_test) = data.load_data()

        self.input_shape = list(x_train.shape[1:])
        self.num_classes = int(np.max(y_train)) + 1

        self._initialize_session()
        self._initialize_metadata()

    def _initialize_session(self):
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf_config)

    def _initialize_metadata(self):
        self.FLOAT_TF = getattr(tf, self.float_type)
        self.FLOAT_NP = getattr(np, self.float_type)
        self.INT_TF = getattr(tf, self.int_type)
        self.INT_NP = getattr(np, self.int_type)
        self.UINT_TF = getattr(np, 'u' + self.int_type)
        self.UINT_NP = getattr(tf, 'u' + self.int_type)

        self.regularizer_map = {}

        assert not self.n_units_encoder is None, 'You must provide a value for **n_units_encoder** when initializing the model.'
        if isinstance(self.n_units_encoder, str):
            self.units_encoder = [int(x) for x in self.n_units_encoder.split()]
        elif isinstance(self.n_units_encoder, int):
            if self.n_layers_encoder is None:
                self.units_encoder = [self.n_units_encoder]
            else:
                self.units_encoder = [self.n_units_encoder] * self.n_layers_encoder
        else:
            self.units_encoder = self.n_units_encoder

        if self.n_layers_encoder is None:
            self.layers_encoder = len(self.units_encoder)
        else:
            self.layers_encoder = self.n_layers_encoder
        if len(self.units_encoder) == 1:
            self.units_encoder = [self.units_encoder[0]] * self.layers_encoder

        assert len(self.units_encoder) == self.layers_encoder, 'Misalignment in number of layers between n_layers_encoder and n_units_encoder.'

        if isinstance(self.n_max_pool_encoder, str):
            self.max_pool_encoder = [int(x) for x in self.n_max_pool_encoder.split()]
            if len(self.max_pool_encoder) == 1:
                self.max_pool_encoder *= self.layers_encoder
        elif isinstance(self.n_max_pool_encoder, int):
            if self.n_layers_encoder is None:
                self.max_pool_encoder = [self.n_max_pool_encoder]
            else:
                self.max_pool_encoder = [self.n_max_pool_encoder] * self.n_layers_encoder
        else:
            self.max_pool_encoder = [self.n_max_pool_encoder] * self.n_layers_encoder

        assert not self.n_units_classifier is None, 'You must provide a value for **n_units_classifier** when initializing the model.'
        if isinstance(self.n_units_classifier, str):
            self.units_classifier = [int(x) for x in self.n_units_classifier.split()]
        elif isinstance(self.n_units_classifier, int):
            if self.n_layers_classifier is None:
                self.units_classifier = [self.n_units_classifier]
            else:
                self.units_classifier = [self.n_units_classifier] * self.n_layers_classifier
        else:
            self.units_classifier = self.n_units_classifier

        if self.n_layers_classifier is None:
            self.layers_classifier = len(self.units_classifier)
        else:
            self.layers_classifier = self.n_layers_classifier
        if len(self.units_classifier) == 1:
            self.units_classifier = [self.units_classifier[0]] * self.layers_classifier

        assert len(self.units_classifier) == self.layers_classifier, 'Misalignment in number of layers between n_layers_classifier and n_units_classifier.'

        if isinstance(self.use_jtps, str):
            self.use_jtps = self.use_jtps.lower() == 'true'

        if isinstance(self.learning_rate, str):
            self.learning_rate = float(self.learning_rate)

        if isinstance(self.meta_learning_rate, str):
            if self.meta_learning_rate.lower() == 'none':
                self.meta_learning_rate = None
            else:
                self.meta_learning_rate = float(self.meta_learning_rate)

        self.predict_mode = False

    def _pack_metadata(self):
        md = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }
        for kwarg in Classifier._INITIALIZATION_KWARGS:
            md[kwarg.key] = getattr(self, kwarg.key)
        return md

    def _unpack_metadata(self, md):
        self.input_shape = md.pop('input_shape')
        self.num_classes = md.pop('num_classes')
        for kwarg in Classifier._INITIALIZATION_KWARGS:
            setattr(self, kwarg.key, md.pop(kwarg.key, kwarg.default_value))

    def __getstate__(self):
        return self._pack_metadata()

    def __setstate__(self, state):
        self._unpack_metadata(state)
        self._initialize_session()
        self._initialize_metadata()

    ############################################################
    # Private model construction methods
    ############################################################

    def _initialize_inputs(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.training = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=[], name='training')

                self.X = tf.placeholder(self.FLOAT_TF, [None] + self.input_shape)
                self.y = tf.placeholder(self.INT_TF, [None])

                inputs = self.X
                if self.task.lower() == 'mnist':
                    inputs = inputs[..., None]
                self.inputs = inputs

                self.global_step = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.INT_TF,
                    name='global_step'
                )
                self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)
                self.global_batch_step = tf.Variable(
                    0,
                    trainable=False,
                    dtype=self.INT_TF,
                    name='global_batch_step'
                )
                self.incr_global_batch_step = tf.assign(self.global_batch_step, self.global_batch_step + 1)

                self.loss_summary = tf.placeholder(tf.float32, name='loss_summary_placeholder')
                self.reg_summary = tf.placeholder(tf.float32, name='regularizer_summary_placeholder')
                self.acc_summary = tf.placeholder(tf.float32, name='acc_summary_placeholder')
                self.p_summary = tf.placeholder(tf.float32, name='p_summary_placeholder')
                self.r_summary = tf.placeholder(tf.float32, name='r_summary_placeholder')
                self.f1_summary = tf.placeholder(tf.float32, name='f1_summary_placeholder')

    def _initialize_encoder(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                encoder = self.inputs
                conv_dim = len(encoder.shape) - 2

                if self.encoder_type.lower() in ['rnn', 'cnn_rnn']:

                    if self.encoder_type == 'cnn_rnn':
                        encoder = ConvLayer(
                            self.encoder_conv_kernel_size,
                            training=self.training,
                            n_filters=self.frame_dim,
                            dim=conv_dim,
                            padding='same',
                            activation=tf.nn.elu,
                            batch_normalization_decay=self.encoder_batch_normalization_decay,
                            session=self.sess,
                            name='RNN_preCNN'
                        )(encoder)

                        encoder = tf.reshape(
                            encoder,
                            [
                                tf.shape(encoder)[0],
                                tf.reduce_prod(tf.shape(encoder)[1:-1]),
                                tf.shape(encoder[-1])
                            ]
                        )

                    encoder = MultiRNNLayer(
                        training=self.training,
                        units=self.units_encoder,
                        layers=self.layers_encoder,
                        activation=self.encoder_inner_activation,
                        inner_activation=self.encoder_inner_activation,
                        recurrent_activation=self.encoder_recurrent_activation,
                        return_sequences=False,
                        name='RNNEncoder',
                        session=self.sess
                    )(encoder)

                elif self.encoder_type.lower() == 'cnn':
                    for i in range(self.layers_encoder):
                        if i > 0 and self.encoder_resnet_n_layers_inner and self.encoder_resnet_n_layers_inner > 1:
                            encoder = ConvResidualLayer(
                                self.encoder_conv_kernel_size,
                                training=self.training,
                                n_filters=self.units_encoder[i],
                                dim=conv_dim,
                                padding='same',
                                layers_inner=self.encoder_resnet_n_layers_inner,
                                activation=self.encoder_inner_activation,
                                activation_inner=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess,
                                name='CNNEncoder_l%d' % i
                            )(encoder)
                        else:
                            encoder = ConvLayer(
                                self.encoder_conv_kernel_size,
                                training=self.training,
                                n_filters=self.units_encoder[i],
                                dim=conv_dim,
                                padding='same',
                                activation=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess,
                                name='CNNEncoder_l%d' % i
                            )(encoder)

                        if self.max_pool_encoder[i]:
                            MaxPool = {
                                1: tf.keras.layers.MaxPool1D,
                                2: tf.keras.layers.MaxPool2D,
                                3: tf.keras.layers.MaxPool3D
                            }[conv_dim]
                            encoder = MaxPool(self.max_pool_encoder[i], padding='same')(encoder)

                    encoder = tf.layers.Flatten()(encoder)

                elif self.encoder_type.lower() == 'dense':
                    encoder = tf.layers.Flatten()(encoder)

                    for i in range(self.layers_encoder):
                        if i > 0 and self.encoder_resnet_n_layers_inner and self.encoder_resnet_n_layers_inner > 1:
                            encoder = DenseResidualLayer(
                                training=self.training,
                                units=self.units_encoder[i],
                                layers_inner=self.encoder_resnet_n_layers_inner,
                                activation=self.encoder_inner_activation,
                                activation_inner=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess,
                                name='DenseEncoder_l%d' % i
                            )(encoder)
                        else:
                            encoder = DenseLayer(
                                training=self.training,
                                units=self.units_encoder[i],
                                activation=self.encoder_inner_activation,
                                batch_normalization_decay=self.encoder_batch_normalization_decay,
                                session=self.sess,
                                name='DenseEncoder_l%d' % i
                            )(encoder)

                else:
                    raise ValueError('Encoder type "%s" is not currently supported' % self.encoder_type)

                self.encoder = encoder

    def _initialize_classifier(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                classifier = self.encoder

                for i in range(self.layers_classifier):
                    if i > 0 and self.classifier_resnet_n_layers_inner and self.classifier_resnet_n_layers_inner > 1:
                        classifier = DenseResidualLayer(
                            training=self.training,
                            units=self.units_classifier[i],
                            layers_inner=self.classifier_resnet_n_layers_inner,
                            activation=self.classifier_inner_activation,
                            activation_inner=self.classifier_inner_activation,
                            batch_normalization_decay=self.classifier_batch_normalization_decay,
                            session=self.sess,
                            name='DenseClassifier_l%d' % i
                        )(classifier)
                    else:
                        classifier = DenseLayer(
                            training=self.training,
                            units=self.units_classifier[i],
                            activation=self.classifier_inner_activation,
                            batch_normalization_decay=self.classifier_batch_normalization_decay,
                            session=self.sess,
                            name='DenseClassifier_l%d' % i
                        )(classifier)

                logits = DenseLayer(
                    training=self.training,
                    units=self.num_classes,
                    activation=None,
                    batch_normalization_decay=None,
                    session=self.sess,
                    name='DenseClassifier_final'
                )(classifier)

                self.logits = logits
                self.preds_soft = tf.nn.softmax(self.logits, axis=-1)
                self.preds = tf.argmax(self.logits, axis=-1)

    def _initialize_optimizer(self):
        name = self.optim_name.lower()
        use_jtps = self.use_jtps

        with self.sess.as_default():
            with self.sess.graph.as_default():
                lr = tf.constant(self.learning_rate, dtype=self.FLOAT_TF)
                if name is None:
                    self.lr = lr
                    return None
                if self.lr_decay_family is not None:
                    lr_decay_steps = tf.constant(self.lr_decay_steps, dtype=self.INT_TF)
                    lr_decay_rate = tf.constant(self.lr_decay_rate, dtype=self.FLOAT_TF)
                    lr_decay_staircase = self.lr_decay_staircase

                    if self.lr_decay_iteration_power != 1:
                        t = tf.cast(self.step, dtype=self.FLOAT_TF) ** self.lr_decay_iteration_power
                    else:
                        t = self.step

                    if self.lr_decay_family.lower() == 'linear_decay':
                        if lr_decay_staircase:
                            decay = tf.floor(t / lr_decay_steps)
                        else:
                            decay = t / lr_decay_steps
                        decay *= lr_decay_rate
                        self.lr = lr - decay
                    else:
                        self.lr = getattr(tf.train, self.lr_decay_family)(
                            lr,
                            t,
                            lr_decay_steps,
                            lr_decay_rate,
                            staircase=lr_decay_staircase,
                            name='learning_rate'
                        )
                    if np.isfinite(self.learning_rate_min):
                        lr_min = tf.constant(self.learning_rate_min, dtype=self.FLOAT_TF)
                        INF_TF = tf.constant(np.inf, dtype=self.FLOAT_TF)
                        self.lr = tf.clip_by_value(self.lr, lr_min, INF_TF)
                else:
                    self.lr = lr

                clip = self.max_global_gradient_norm

                optimizer_args = [self.lr]
                optimizer_kwargs = {}
                if name == 'momentum':
                    optimizer_args += [0.9]

                optimizer_class = {
                    'sgd': tf.train.GradientDescentOptimizer,
                    'momentum': tf.train.MomentumOptimizer,
                    'adagrad': tf.train.AdagradOptimizer,
                    'adadelta': tf.train.AdadeltaOptimizer,
                    'ftrl': tf.train.FtrlOptimizer,
                    'rmsprop': tf.train.RMSPropOptimizer,
                    'adam': tf.train.AdamOptimizer,
                    'nadam': tf.contrib.opt.NadamOptimizer
                }[name]

                if clip:
                    optimizer_class = get_clipped_optimizer_class(optimizer_class, session=self.sess)
                    optimizer_kwargs['max_global_norm'] = clip

                if use_jtps:
                    optimizer_class = get_JTPS_optimizer_class(optimizer_class, session=self.sess)
                    optimizer_kwargs['meta_learning_rate'] = self.meta_learning_rate

                optim = optimizer_class(*optimizer_args, **optimizer_kwargs)

                self.optim = optim

    def _initialize_objective(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.loss = tf.losses.sparse_softmax_cross_entropy(self.y, self.logits)
                total_loss = self.loss
                if len(self.regularizer_map) > 0:
                    self.regularizer_loss = tf.add_n(self._apply_regularization(reduce_each=True, reduce_all=False))
                    total_loss += self.regularizer_loss
                else:
                    self.regularizer_loss = tf.constant(0.)
                self.total_loss = total_loss

                self.train_op = self.optim.minimize(self.total_loss, global_step=self.global_batch_step)

                if self.use_jtps:
                    self.jtps_lambda = self.optim.get_flattened_lambdas()
                    self.jtps_lambda_min = tf.reduce_min(self.jtps_lambda)
                    self.jtps_lambda_max = tf.reduce_max(self.jtps_lambda)
                    self.jtps_lambda_mean = tf.reduce_mean(self.jtps_lambda)

    def _initialize_logging(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                tf.summary.scalar('objective/loss', self.loss_summary, collections=['objective'])
                tf.summary.scalar('objective/regularizer_loss', self.reg_summary, collections=['objective'])
                tf.summary.scalar('evaluation/acc', self.acc_summary, collections=['evaluation'])
                tf.summary.scalar('evaluation/p', self.p_summary, collections=['evaluation'])
                tf.summary.scalar('evaluation/r', self.r_summary, collections=['evaluation'])
                tf.summary.scalar('evaluation/f1', self.f1_summary, collections=['evaluation'])

                self.summary_objective = tf.summary.merge_all(key='objective')
                self.summary_evaluation = tf.summary.merge_all(key='evaluation')

                if self.log_graph:
                    self.writer_train = tf.summary.FileWriter(self.outdir + '/tensorboard/train', self.sess.graph)
                    self.writer_test = tf.summary.FileWriter(self.outdir + '/tensorboard/test', self.sess.graph)
                else:
                    self.writer_train = tf.summary.FileWriter(self.outdir + '/tensorboard/train')
                    self.writer_test = tf.summary.FileWriter(self.outdir + '/tensorboard/test')

    def _initialize_saver(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.saver = tf.train.Saver()

                self.check_numerics_ops = [tf.check_numerics(v, 'Numerics check failed') for v in tf.trainable_variables()]

    def _initialize_ema(self):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.ema_decay:
                    vars = [var for var in tf.get_collection('trainable_variables') if 'BatchNorm' not in var.name]

                    self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)
                    self.ema_op = self.ema.apply(vars)
                    self.ema_map = {}
                    for v in vars:
                        self.ema_map[self.ema.average_name(v)] = v
                    self.ema_saver = tf.train.Saver(self.ema_map)




    ############################################################
    # Private utility methods
    ############################################################

    def _add_regularization(self, var, regularizer):
        if regularizer is not None:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    self.regularizer_map[var] = regularizer

    def _apply_regularization(self, reduce_all=False, reduce_each=True):
        regularizer_losses = []
        for var in self.regularizer_map:
            reg_loss = tf.contrib.layers.apply_regularization(self.regularizer_map[var], [var])
            if reduce_each:
                n = tf.maximum(tf.cast(tf.reduce_prod(tf.shape(reg_loss)), self.FLOAT_TF), self.epsilon)
                reg_loss = tf.reduce_sum(reg_loss) / n
            regularizer_losses.append(reg_loss)

        if reduce_all:
            n = len(regularizer_losses)
            regularizer_loss = tf.add_n(regularizer_losses) / tf.maximum(n, self.epsilon)

        return regularizer_losses

    # Thanks to Ralph Mao (https://github.com/RalphMao) for this workaround
    def _restore_inner(self, path, predict=False, allow_missing=False):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                try:
                    if predict:
                        self.ema_saver.restore(self.sess, path)
                    else:
                        self.saver.restore(self.sess, path)
                except tf.errors.DataLossError:
                    stderr('Read failure during load. Trying from backup...\n')
                    if predict:
                        self.ema_saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                    else:
                        self.saver.restore(self.sess, path[:-5] + '_backup.ckpt')
                except tf.errors.NotFoundError as err:  # Model contains variables that are missing in checkpoint, special handling needed
                    if allow_missing:
                        reader = tf.train.NewCheckpointReader(path)
                        saved_shapes = reader.get_variable_to_shape_map()
                        model_var_names = sorted(
                            [(var.name, var.name.split(':')[0]) for var in tf.global_variables()])
                        ckpt_var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                                                 if var.name.split(':')[0] in saved_shapes])

                        model_var_names_set = set([x[1] for x in model_var_names])
                        ckpt_var_names_set = set([x[1] for x in ckpt_var_names])

                        missing_in_ckpt = model_var_names_set - ckpt_var_names_set
                        if len(missing_in_ckpt) > 0:
                            stderr(
                                'Checkpoint file lacked the variables below. They will be left at their initializations.\n%s.\n\n' % (
                                    sorted(list(missing_in_ckpt))))
                        missing_in_model = ckpt_var_names_set - model_var_names_set
                        if len(missing_in_model) > 0:
                            stderr(
                                'Checkpoint file contained the variables below which do not exist in the current model. They will be ignored.\n%s.\n\n' % (
                                    sorted(list(missing_in_ckpt))))

                        restore_vars = []
                        name2var = dict(
                            zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

                        with tf.variable_scope('', reuse=True):
                            for var_name, saved_var_name in ckpt_var_names:
                                curr_var = name2var[saved_var_name]
                                var_shape = curr_var.get_shape().as_list()
                                if var_shape == saved_shapes[saved_var_name]:
                                    restore_vars.append(curr_var)

                        if predict:
                            self.ema_map = {}
                            for v in restore_vars:
                                self.ema_map[self.ema.average_name(v)] = v
                            saver_tmp = tf.train.Saver(self.ema_map)
                        else:
                            saver_tmp = tf.train.Saver(restore_vars)

                        saver_tmp.restore(self.sess, path)
                    else:
                        raise err





    ############################################################
    # Public methods
    ############################################################

    def build(self, seed=None, outdir=None, restore=True):
        if seed is not None:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    tf.set_random_seed(seed)
                    np.random.seed(seed)

        if outdir is None:
            if not hasattr(self, 'outdir'):
                self.outdir = './jtps_test_model/'
        else:
            self.outdir = outdir

        self._initialize_inputs()
        self._initialize_encoder()
        self._initialize_classifier()
        self._initialize_optimizer()
        self._initialize_objective()
        self._initialize_ema()
        self._initialize_saver()
        self._initialize_logging()

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.report_uninitialized = tf.report_uninitialized_variables(
                    var_list=None
                )

        self.load(restore=restore)

        self.sess.graph.finalize()

    def initialized(self):
        """
        Check whether model has been initialized.

        :return: ``bool``; whether the model has been initialized.
        """
        with self.sess.as_default():
            with self.sess.graph.as_default():
                uninitialized = self.sess.run(self.report_uninitialized)
                if len(uninitialized) == 0:
                    return True
                else:
                    return False

    def save(self, dir=None):

        assert not self.predict_mode, 'Cannot save while in predict mode, since this would overwrite the parameters with their moving averages.'

        if dir is None:
            dir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                failed = True
                i = 0

                # Try/except to handle race conditions in Windows
                while failed and i < 10:
                    try:
                        self.saver.save(self.sess, dir + '/model.ckpt')
                        with open(dir + '/m.obj', 'wb') as f:
                            pickle.dump(self, f)
                        failed = False
                    except Exception:
                        sys.stderr.write('Write failure during save. Retrying...\n')
                        time.sleep(1)
                        i += 1
                if i >= 10:
                    sys.stderr.write('Could not save model to checkpoint file. Saving to backup...\n')
                    self.saver.save(self.sess, dir + '/model_backup.ckpt')
                    with open(dir + '/m.obj', 'wb') as f:
                        pickle.dump(self, f)

    def load(self, outdir=None, predict=False, restore=True, allow_missing=True):
        """
        Load weights from a DNN-Seg checkpoint and/or initialize the DNN-Seg model.
        Missing weights in the checkpoint will be kept at their initializations, and unneeded weights in the checkpoint will be ignored.

        :param outdir: ``str``; directory in which to search for weights. If ``None``, use model defaults.
        :param predict: ``bool``; load EMA weights because the model is being used for prediction. If ``False`` load training weights.
        :param restore: ``bool``; restore weights from a checkpoint file if available, otherwise initialize the model. If ``False``, no weights will be loaded even if a checkpoint is found.
        :param allow_missing: ``bool``; load all weights found in the checkpoint file, allowing those that are missing to remain at their initializations. If ``False``, weights in checkpoint must exactly match those in the model graph, or else an error will be raised. Leaving set to ``True`` is helpful for backward compatibility, setting to ``False`` can be helpful for debugging.
        :return:
        """
        if outdir is None:
            outdir = self.outdir
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if not self.initialized():
                    self.sess.run(tf.global_variables_initializer())
                    tf.tables_initializer().run()
                if restore and os.path.exists(outdir + '/checkpoint'):
                    self._restore_inner(outdir + '/model.ckpt', predict=predict, allow_missing=allow_missing)
                else:
                    if predict:
                        sys.stderr.write('No EMA checkpoint available. Leaving internal variables unchanged.\n')

    def set_predict_mode(self, mode):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                if self.ema_decay:
                    reload = mode != self.predict_mode
                    if reload:
                        self.load(predict=mode)

                self.predict_mode = mode

    def report_settings(self, indent=0):
        out = ' ' * indent + 'MODEL SETTINGS:\n'
        for kwarg in Classifier._INITIALIZATION_KWARGS:
            val = getattr(self, kwarg.key)
            out += ' ' * (indent + 2) + '%s: %s\n' %(kwarg.key, "\"%s\"" %val if isinstance(val, str) else val)

        return out

    def report_n_params(self, indent=0):
        with self.sess.as_default():
            with self.sess.graph.as_default():
                n_params = 0
                var_names = [v.name for v in tf.trainable_variables()]
                var_vals = self.sess.run(tf.trainable_variables())
                out = ' ' * indent + 'TRAINABLE PARAMETERS:\n'
                for i in range(len(var_names)):
                    v_name = var_names[i]
                    v_val = var_vals[i]
                    cur_params = np.prod(np.array(v_val).shape)
                    n_params += cur_params
                    out += ' ' * indent + '  ' + v_name.split(':')[0] + ': %s\n' % str(cur_params)
                out += ' ' * indent + '  TOTAL: %d\n\n' % n_params

                return out

    def fit(self, n_iter, minibatch_size=None, verbose=True):
        if self.global_step.eval(session=self.sess) == 0:
            if verbose:
                stderr('Saving initial weights...\n')
            self.save()

        data = getattr(tf.keras.datasets, self.task.lower())
        (x_train, y_train), (x_test, y_test) = data.load_data()
        
        if len(y_train.shape) > 1:
            y_train = np.squeeze(y_train)
        if len(y_test.shape) > 1:
            y_test = np.squeeze(y_test)

        n = len(x_train)

        if minibatch_size is None:
            minibatch_size = self.minibatch_size
        n_minibatch = int(math.ceil(n / minibatch_size))

        if verbose:
            stderr('*' * 100 + '\n')
            stderr(self.report_settings())
            stderr('\n')
            stderr(self.report_n_params())
            stderr('\n')
            stderr('*' * 100 + '\n\n')

        to_run = [
            self.train_op,
            self.total_loss,
            self.regularizer_loss,
            self.preds
        ]

        if self.use_jtps:
            to_run += [
                self.jtps_lambda_mean,
                self.jtps_lambda_min,
                self.jtps_lambda_max
            ]

        with self.sess.as_default():
            with self.sess.graph.as_default():
                while self.global_step.eval(session=self.sess) < n_iter:
                    if verbose:
                        t0_iter = time.time()
                        stderr('-' * 50 + '\n')
                        stderr('Iteration %d\n' % int(self.global_step.eval(session=self.sess) + 1))
                        stderr('\n')

                        pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                    ix, ix_inv = get_random_permutation(len(x_train))

                    loss = 0
                    reg = 0
                    preds_train = []

                    for i in range(0, n, minibatch_size):
                        X_cur = x_train[ix[i:i+minibatch_size]]
                        y_cur = y_train[ix[i:i+minibatch_size]]

                        fd_minibatch = {
                            self.X: X_cur,
                            self.y: y_cur,
                            self.training: True
                        }

                        out = self.sess.run(to_run, feed_dict=fd_minibatch)
                        if self.use_jtps:
                            _, loss_cur, reg_cur, preds_train_cur, lambda_mean_cur, lambda_min_cur, lambda_max_cur = out
                        else:
                            _, loss_cur, reg_cur, preds_train_cur = out

                        preds_train.append(preds_train_cur)

                        eval_dict = self.evaluate(y_cur, preds_train_cur)

                        loss += loss_cur
                        reg += reg_cur

                        summary_objective = self.sess.run(
                            self.summary_objective,
                            {
                                self.loss_summary: loss_cur,
                                self.reg_summary: reg_cur,
                            }
                        )
                        self.writer_train.add_summary(summary_objective, self.global_batch_step.eval(session=self.sess))

                        if verbose:
                            pb_vector = [
                                ('loss', loss_cur),
                                ('acc', eval_dict['acc']),
                                ('f1', eval_dict['f1']),
                            ]
                            if self.use_jtps:
                                pb_vector.append(('l', lambda_mean_cur))
                                pb_vector.append(('l min', lambda_min_cur))
                                pb_vector.append(('l max', lambda_max_cur))
                            pb.update(
                                i // minibatch_size + 1,
                                pb_vector
                            )
                            
                    loss /= n_minibatch
                    reg /= n_minibatch

                    self.sess.run(self.incr_global_step)

                    if self.global_step.eval(session=self.sess) % self.save_freq == 0:
                        self.save()

                    # summary_objective = self.sess.run(
                    #     self.summary_objective,
                    #     {
                    #         self.loss_summary: loss_cur,
                    #         self.reg_summary: reg_cur,
                    #     }
                    # )
                    # self.writer_train.add_summary(summary_objective, self.global_batch_step.eval(session=self.sess))

                    if self.global_step.eval(session=self.sess) % self.eval_freq == 0:
                        preds_train = np.concatenate(preds_train)[ix_inv]
                        eval_train = self.evaluate(y_train, preds_train)

                        preds_test = self.predict(x_test, verbose=verbose)
                        eval_test = self.evaluate(y_test, preds_test)

                        stderr('\n')
                        stderr('TEST SET EVAL:\n')
                        stderr('  ACC: %.2f\n' % (eval_test['acc'] * 100))
                        stderr('  P:   %.2f\n' % (eval_test['p'] * 100))
                        stderr('  R:   %.2f\n' % (eval_test['r'] * 100))
                        stderr('  F1:  %.2f\n\n' % (eval_test['f1'] * 100))

                        summary_eval_train = self.sess.run(
                            self.summary_evaluation,
                            {
                                self.acc_summary: eval_train['acc'],
                                self.p_summary: eval_train['p'],
                                self.r_summary: eval_train['r'],
                                self.f1_summary: eval_train['f1'],
                            }
                        )
                        self.writer_train.add_summary(summary_eval_train, self.global_step.eval(session=self.sess))
                        
                        summary_eval_test = self.sess.run(
                            self.summary_evaluation,
                            {
                                self.acc_summary: eval_test['acc'],
                                self.p_summary: eval_test['p'],
                                self.r_summary: eval_test['r'],
                                self.f1_summary: eval_test['f1'],
                            }
                        )
                        self.writer_test.add_summary(summary_eval_test, self.global_step.eval(session=self.sess))

                    if verbose:
                        t1_iter = time.time()
                        time_str = pretty_print_seconds(t1_iter - t0_iter)
                        sys.stderr.write('Iteration time: %s\n' % time_str)

    def predict(self, X, minibatch_size=None, verbose=True):
        n = len(X)

        if minibatch_size is None:
            minibatch_size = self.eval_minibatch_size
        n_minibatch = int(math.ceil(n / minibatch_size))

        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.set_predict_mode(True)

                if verbose:
                    t0_iter = time.time()
                    stderr('Extracting predictions...\n')

                    pb = tf.contrib.keras.utils.Progbar(n_minibatch)

                preds = []

                for i in range(0, n, minibatch_size):
                    X_cur = X[i:i + minibatch_size]

                    fd_minibatch = {
                        self.X: X_cur,
                        self.training: False
                    }

                    preds_cur = self.sess.run(self.preds, feed_dict=fd_minibatch)

                    preds.append(preds_cur)

                    if verbose:
                        pb.update(i // minibatch_size + 1)

                preds = np.concatenate(preds, axis=0)

                if verbose:
                    t1_iter = time.time()
                    time_str = pretty_print_seconds(t1_iter - t0_iter)
                    sys.stderr.write('Prediction time: %s\n' % time_str)

                self.set_predict_mode(False)
                    
                return preds

    def evaluate(self, X, y):
        if len(y.shape) > 1:
            y = np.squeeze(y)

        eval_dict = evaluate_classifier(X, y)

        return eval_dict
