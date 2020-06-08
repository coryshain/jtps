import re
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.layers.utils import conv_output_length


if hasattr(rnn_cell_impl, 'LayerRNNCell'):
    LayerRNNCell = rnn_cell_impl.LayerRNNCell
else:
    LayerRNNCell = rnn_cell_impl._LayerRNNCell


parse_initializer = re.compile('(.*_initializer)(_(.*))?')


def get_session(session):
    if session is None:
        sess = tf.get_default_session()
    else:
        sess = session

    return sess


def make_clipped_linear_activation(lb=None, ub=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if lb is None:
                lb = -np.inf
            if ub is None:
                ub = np.inf
            return lambda x: tf.clip_by_value(x, lb, ub)


def get_activation(activation, session=None, training=True, from_logits=True, sample_at_train=True, sample_at_eval=False):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            hard_sigmoid = tf.keras.backend.hard_sigmoid

            if activation:
                if isinstance(activation, str):
                    if activation.lower().startswith('cla'):
                        _, lb, ub = activation.split('_')
                        if lb in ['None', '-inf']:
                            lb = None
                        else:
                            lb = float(lb)
                        if ub in ['None', 'inf']:
                            ub = None
                        else:
                            ub = float(ub)
                        out = make_clipped_linear_activation(lb=lb, ub=ub, session=session)
                    elif activation.lower() == 'hard_sigmoid':
                        out = hard_sigmoid
                    elif activation.lower() == 'round':
                        def out(x):
                            return tf.round(x)
                    elif activation.lower() == 'stop_gradient':
                        def out(x):
                            return tf.stop_gradient(x)
                    elif activation.lower() == 'argmax':
                        def out(x):
                            dim = x.shape[-1]
                            one_hot = tf.one_hot(tf.argmax(x, axis=-1), dim)
                            return one_hot
                    elif activation.lower() in ['bsn', 'csn']:
                        if activation.lower() == 'bsn':
                            sample_fn_inner = bernoulli_straight_through
                            round_fn_inner = round_straight_through
                            if from_logits:
                                logits2probs = tf.sigmoid
                            else:
                                logits2probs = lambda x: x
                        else: # activation.lower() == 'csn'
                            sample_fn_inner = argmax_straight_through
                            round_fn_inner = categorical_sample_straight_through
                            if from_logits:
                                logits2probs = tf.nn.softmax
                            else:
                                logits2probs = lambda x: x

                        def make_sample_fn(s, logit_fn=logits2probs, fn=sample_fn_inner):
                            def sample_fn(x):
                                return fn(logit_fn(x), session=s)

                            return sample_fn

                        def make_round_fn(s, logit_fn=logits2probs, fn=round_fn_inner):
                            def round_fn(x):
                                return fn(logit_fn(x), session=s)

                            return round_fn

                        sample_fn = make_sample_fn(session)
                        round_fn = make_round_fn(session)

                        if sample_at_train:
                            train_fn = sample_fn
                        else:
                            train_fn = round_fn

                        if sample_at_eval:
                            eval_fn = sample_fn
                        else:
                            eval_fn = round_fn

                        out = lambda x: tf.cond(training, lambda: train_fn(x), lambda: eval_fn(x))

                    elif activation.lower().startswith('slow_sigmoid'):
                        split = activation.split('_')
                        if len(split) == 2:
                            # Default to a slowness parameter of 1/2
                            scale = 0.5
                        else:
                            try:
                                scale = float(split[2])
                            except ValueError:
                                raise ValueError('Parameter to slow_sigmoid must be a valid float.')

                        out = lambda x: tf.sigmoid(0.5 * x)

                    else:
                        out = getattr(tf.nn, activation)
                else:
                    out = activation
            else:
                out = lambda x: x

    return out


def get_initializer(initializer, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if isinstance(initializer, str):
                initializer_name, _, initializer_params = parse_initializer.match(initializer).groups()

                kwargs = {}
                if initializer_params:
                    kwarg_list = initializer_params.split('-')
                    for kwarg in kwarg_list:
                        key, val = kwarg.split('=')
                        try:
                            val = float(val)
                        except Exception:
                            pass
                        kwargs[key] = val

                tf.keras.initializers.he_normal()

                if 'identity' in initializer_name:
                    return tf.keras.initializers.Identity
                elif 'he_' in initializer_name:
                    return tf.keras.initializers.VarianceScaling(scale=2., mode='fan_in', distribution='normal')
                else:
                    out = getattr(tf, initializer_name)
                    if 'glorot' in initializer:
                        out = out()
                    else:
                        out = out(**kwargs)
            else:
                out = initializer

            return out


def get_regularizer(init, scale=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if scale is None and isinstance(init, str) and '_' in init:
                try:
                    init_split = init.split('_')
                    scale = float(init_split[-1])
                    init = '_'.join(init_split[:-1])
                except ValueError:
                    pass

            if scale is None:
                scale = 0.001

            if init is None:
                out = None
            elif isinstance(init, str):
                if init.lower() == 'l1_regularizer':
                    out = lambda x, scale=scale: tf.abs(x) * scale
                if init.lower() == 'l2_regularizer':
                    out = lambda x, scale=scale: x**2 * scale
                else:
                    out = getattr(tf.contrib.layers, init)(scale=scale)
            elif isinstance(init, float):
                out = tf.contrib.layers.l2_regularizer(scale=init)
            else:
                out = init

            return out


def get_dropout(rate, training=True, noise_shape=None, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            if rate:
                def make_dropout(rate):
                    return lambda x: tf.layers.dropout(x, rate=rate, noise_shape=noise_shape, training=training)
                out = make_dropout(rate)
            else:
                out = lambda x: x

            return out


def round_straight_through(x, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            fw_op = tf.round
            bw_op = tf.identity
            return replace_gradient(fw_op, bw_op, session=session)(x)


def bernoulli_straight_through(x, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            fw_op = lambda x: tf.ceil(x - tf.random_uniform(tf.shape(x)))
            bw_op = tf.identity
            return replace_gradient(fw_op, bw_op, session=session)(x)

def argmax_straight_through(x, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            fw_op = lambda x: tf.one_hot(tf.argmax(x, axis=-1), x.shape[-1])
            bw_op = tf.identity
            return replace_gradient(fw_op, bw_op, session=session)(x)

def categorical_sample_straight_through(x, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            fw_op = lambda x: tf.one_hot(tf.contrib.distributions.Categorical(probs=x).sample(), x.shape[-1])
            bw_op = tf.identity
            return replace_gradient(fw_op, bw_op, session=session)(x)


def replace_gradient(fw_op, bw_op, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            def new_op(x):
                fw = fw_op(x)
                bw = bw_op(x)
                out = bw + tf.stop_gradient(fw-bw)
                return out
            return new_op


def compose_lambdas(lambdas):
    def composed_lambdas(x, **kwargs):
        out = x
        for l in lambdas:
            out = l(out, **kwargs)
        return out

    return composed_lambdas


def make_lambda(layer, session=None, use_kwargs=False):
    session = get_session(session)

    with session.as_default():
        with session.graph.as_default():
            if use_kwargs:
                def apply_layer(x, **kwargs):
                    return layer(x, **kwargs)
            else:
                def apply_layer(x, **kwargs):
                    return layer(x)
            return apply_layer


############################################################
# Cells
############################################################

class MultiLSTMCell(LayerRNNCell):
    def __init__(
            self,
            num_units,
            num_layers,
            training=True,
            forget_bias=1.0,
            activation=None,
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            kernel_initializer='glorot_uniform_initializer',
            bias_initializer='zeros_initializer',
            refeed_outputs=False,
            reuse=None,
            name=None,
            dtype=None,
            session=None
    ):
        self.session = get_session(session)

        with self.session.as_default():
            with self.session.graph.as_default():
                super(MultiLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

                if not isinstance(num_units, list):
                    self._num_units = [num_units] * num_layers
                else:
                    self._num_units = num_units

                assert len(self._num_units) == num_layers, 'num_units must either be an integer or a list of integers of length num_layers'

                self._num_layers = num_layers
                self._forget_bias = forget_bias

                self._training=training

                self._activation = get_activation(activation, session=self.session, training=self._training)
                self._inner_activation = get_activation(inner_activation, session=self.session, training=self._training)
                self._recurrent_activation = get_activation(recurrent_activation, session=self.session, training=self._training)

                self._kernel_initializer = get_initializer(kernel_initializer, session=self.session)
                self._bias_initializer = get_initializer(bias_initializer, session=self.session)

                self._refeed_outputs = refeed_outputs

    def _regularize(self, var, regularizer):
        if regularizer is not None:
            with self.session.as_default():
                with self.session.graph.as_default():
                    reg = tf.contrib.layers.apply_regularization(regularizer, [var])
                    self.regularizer_losses.append(reg)

    @property
    def state_size(self):
        out = []
        for l in range(self._num_layers):
            size = (self._num_units[l], self._num_units[l])
            out.append(size)

        out = tuple(out)

        return out

    @property
    def output_size(self):
        out = self._num_units[-1]

        return out

    def build(self, inputs_shape):
        with self.session.as_default():
            with self.session.graph.as_default():
                if inputs_shape[1].value is None:
                    raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

                self._kernel = []
                self._bias = []

                for l in range(self._num_layers):
                    if l == 0:
                        bottom_up_dim = inputs_shape[1].value
                    else:
                        bottom_up_dim = self._num_units[l-1]

                    recurrent_dim = self._num_units[l]
                    output_dim = 4 * self._num_units[l]
                    if self._refeed_outputs and l == 0:
                        refeed_dim = self._num_units[-1]
                    else:
                        refeed_dim = 0

                    kernel = self.add_variable(
                        'kernel_%d' %l,
                        shape=[bottom_up_dim + recurrent_dim + refeed_dim, output_dim],
                        initializer=self._kernel_initializer
                    )
                    self._kernel.append(kernel)

                    bias = self.add_variable(
                        'bias_%d' %l,
                        shape=[1, output_dim],
                        initializer=self._bias_initializer
                    )
                    self._bias.append(bias)

        self.built = True

    def call(self, inputs, state):
        with self.session.as_default():
            new_state = []

            h_below = inputs
            for l, layer in enumerate(state):
                c_behind, h_behind = layer

                # Gather inputs
                layer_inputs = [h_below, h_behind]

                if self._refeed_outputs and l == 0:
                    prev_outputs = state[-1][1]
                    layer_inputs.append(prev_outputs)

                layer_inputs = tf.concat(layer_inputs, axis=1)

                # Compute gate pre-activations
                s = tf.matmul(
                    layer_inputs,
                    self._kernel[l]
                )

                # Add bias
                s = s + self._bias[l]

                # Alias useful variables
                if l < self._num_layers - 1:
                    # Use inner activation if non-final layer
                    activation = self._inner_activation
                else:
                    # Use outer activation if final layer
                    activation = self._activation
                units = self._num_units[l]

                # Forget gate
                f = self._recurrent_activation(s[:, :units] + self._forget_bias)
                # Input gate
                i = self._recurrent_activation(s[:, units:units * 2])
                # Output gate
                o = self._recurrent_activation(s[:, units * 2:units * 3])
                # Cell proposal
                g = activation(s[:, units * 3:units * 4])

                # Compute new cell state
                c = f * c_behind + i * g

                # Compute the gated output
                h = o * activation(c)

                new_state.append((c, h))

                h_below = h

            new_state = tuple(new_state)
            new_output = new_state[-1][1]

            return new_output, new_state




############################################################
# Layers
############################################################


class DenseLayer(object):
    def __init__(
            self,
            training=True,
            units=None,
            use_bias=True,
            kernel_initializer='glorot_uniform_initializer',
            bias_initializer='zeros_initializer',
            kernel_regularizer=None,
            bias_regularizer=None,
            activation=None,
            sample_at_train=False,
            sample_at_eval=False,
            batch_normalization_decay=None,
            normalize_weights=False,
            reuse=None,
            session=None,
            name=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = get_initializer(kernel_initializer, session=self.session)
        if bias_initializer is None:
            bias_initializer = 'zeros_initializer'
        self.bias_initializer = get_initializer(bias_initializer, session=self.session)
        self.kernel_regularizer = get_regularizer(kernel_regularizer, session=self.session)
        self.bias_regularizer = get_regularizer(bias_regularizer, session=self.session)
        self.activation = get_activation(
            activation,
            session=self.session,
            training=self.training,
            from_logits=True,
            sample_at_train=sample_at_train,
            sample_at_eval=sample_at_eval
        )
        self.batch_normalization_decay = batch_normalization_decay
        self.normalize_weights = normalize_weights
        self.reuse = reuse
        self.name = name

        self.dense_layer = None
        self.projection = None

        self.initializer = get_initializer(kernel_initializer, self.session)

        self.built = False

    def build(self, inputs):
        if not self.built:
            if self.units is None:
                out_dim = inputs.shape[-1]
            else:
                out_dim = self.units

            with self.session.as_default():
                with self.session.graph.as_default():
                    self.dense_layer = tf.layers.Dense(
                        out_dim,
                        use_bias=self.use_bias,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        _reuse=self.reuse,
                        name=self.name
                    )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():

                H = self.dense_layer(inputs)

                if self.normalize_weights:
                    self.w = self.dense_layer.kernel
                    self.g = tf.Variable(tf.ones(self.w.shape[1]), dtype=tf.float32)
                    self.v = tf.norm(self.w, axis=0)
                    self.dense_layer.kernel = self.v

                if self.batch_normalization_decay:
                    H = tf.contrib.layers.batch_norm(
                        H,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None,
                        reuse=self.reuse,
                        scope=self.name
                    )
                if self.activation is not None:
                    H = self.activation(H)

                return H


class DenseResidualLayer(object):

    def __init__(
            self,
            training=True,
            units=None,
            use_bias=True,
            kernel_initializer='glorot_uniform_initializer',
            bias_initializer='zeros_initializer',
            kernel_regularizer=None,
            bias_regularizer=None,
            layers_inner=3,
            activation_inner=None,
            activation=None,
            sample_at_train=False,
            sample_at_eval=False,
            batch_normalization_decay=0.9,
            project_inputs=False,
            normalize_weights=False,
            reuse=None,
            session=None,
            name=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.use_bias = use_bias

        self.layers_inner = layers_inner
        self.kernel_initializer = get_initializer(kernel_initializer, session=self.session)
        if bias_initializer is None:
            bias_initializer = 'zeros_initializer'
        self.bias_initializer = get_initializer(bias_initializer, session=self.session)
        self.kernel_regularizer = get_regularizer(kernel_regularizer, session=self.session)
        self.bias_regularizer = get_regularizer(bias_regularizer, session=self.session)
        self.activation_inner = get_activation(
            activation_inner,
            session=self.session,
            training=self.training,
            from_logits=True,
            sample_at_train=sample_at_train,
            sample_at_eval=sample_at_eval
        )
        self.activation = get_activation(
            activation,
            session=self.session,
            training=self.training,
            from_logits=True,
            sample_at_train=sample_at_train,
            sample_at_eval=sample_at_eval
        )
        self.batch_normalization_decay = batch_normalization_decay
        self.project_inputs = project_inputs
        self.normalize_weights = normalize_weights
        self.reuse = reuse
        self.name = name

        self.dense_layers = None
        self.projection = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    if self.units is None:
                        out_dim = inputs.shape[-1]
                    else:
                        out_dim = self.units

                    self.dense_layers = []

                    for i in range(self.layers_inner):
                        if self.name:
                            name = self.name + '_i%d' % i
                        else:
                            name = None

                        l = tf.layers.Dense(
                            out_dim,
                            use_bias=self.use_bias,
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            _reuse=self.reuse,
                            name=name
                        )
                        self.dense_layers.append(l)

                    if self.project_inputs:
                        if self.name:
                            name = self.name + '_projection'
                        else:
                            name = None

                        self.projection = tf.layers.Dense(
                            out_dim,
                            use_bias=self.use_bias,
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            kernel_regularizer=self.kernel_regularizer,
                            bias_regularizer=self.bias_regularizer,
                            _reuse=self.reuse,
                            name=name
                        )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():

                F = inputs
                for i in range(self.layers_inner - 1):
                    F = self.dense_layers[i](F)
                    if self.batch_normalization_decay:
                        if self.name:
                            name = self.name + '_i%d' % i
                        else:
                            name = None
                        F = tf.contrib.layers.batch_norm(
                            F,
                            decay=self.batch_normalization_decay,
                            center=True,
                            scale=True,
                            zero_debias_moving_mean=True,
                            is_training=self.training,
                            updates_collections=None,
                            reuse=self.reuse,
                            scope=name
                        )
                    if self.activation_inner is not None:
                        F = self.activation_inner(F)

                F = self.dense_layers[-1](F)
                if self.batch_normalization_decay:
                    if self.name:
                        name = self.name + '_i%d' % (self.layers_inner - 1)
                    else:
                        name = None
                    F = tf.contrib.layers.batch_norm(
                        F,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None,
                        reuse=self.reuse,
                        scope=name
                    )

                if self.project_inputs:
                    x = self.projection(inputs)
                else:
                    x = inputs

                H = F + x

                if self.activation is not None:
                    H = self.activation(H)

                return H

class ConvLayer(object):
    def __init__(
            self,
            kernel_size,
            training=True,
            n_filters=None,
            dim=1,
            stride=1,
            padding='valid',
            use_bias=True,
            activation=None,
            dropout=None,
            batch_normalization_decay=0.9,
            reuse=None,
            session=None,
            name=None
    ):
        self.session = get_session(session)
        with session.as_default():
            with session.graph.as_default():
                self.training = training
                self.n_filters = n_filters
                self.dim = dim
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.use_bias = use_bias
                self.activation = get_activation(activation, session=self.session, training=self.training)
                self.dropout = get_dropout(dropout, session=self.session, noise_shape=None, training=self.training)
                self.batch_normalization_decay = batch_normalization_decay
                self.reuse = reuse
                self.name = name

                self.conv_1d_layer = None

                self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    if self.n_filters is None:
                        out_dim = inputs.shape[-1]
                    else:
                        out_dim = self.n_filters

                    if self.dim == 1:
                        CNN = tf.keras.layers.Conv1D
                    elif self.dim == 2:
                        CNN = tf.keras.layers.Conv2D
                    elif self.dim == 3:
                        CNN = tf.keras.layers.Conv3D
                    else:
                        raise ValueError('dim must be in [1, 2, 3]')

                    self.conv_1d_layer = CNN(
                        out_dim,
                        self.kernel_size,
                        padding=self.padding,
                        strides=self.stride,
                        use_bias=self.use_bias,
                        name=self.name
                    )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():
                H = inputs

                H = self.conv_1d_layer(H)

                if self.batch_normalization_decay:
                    H = tf.contrib.layers.batch_norm(
                        H,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None,
                        reuse=self.reuse,
                        scope=self.name
                    )

                if self.activation is not None:
                    H = self.activation(H)

                return H


class ConvResidualLayer(object):
    def __init__(
            self,
            kernel_size,
            training=True,
            n_filters=None,
            dim=1,
            stride=1,
            padding='valid',
            use_bias=True,
            layers_inner=3,
            activation=None,
            activation_inner=None,
            batch_normalization_decay=0.9,
            project_inputs=False,
            n_timesteps=None,
            n_input_features=None,
            reuse=None,
            session=None,
            name=None
    ):
        self.session = get_session(session)

        self.training = training
        self.n_filters = n_filters
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.layers_inner = layers_inner
        self.activation = get_activation(activation, session=self.session, training=self.training)
        self.activation_inner = get_activation(activation_inner, session=self.session, training=self.training)
        self.batch_normalization_decay = batch_normalization_decay
        self.project_inputs = project_inputs
        self.n_timesteps = n_timesteps
        self.n_input_features = n_input_features
        self.reuse = reuse
        self.name = name

        self.conv_layers = None
        self.projection = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            if self.n_filters is None:
                out_dim = inputs.shape[-1]
            else:
                out_dim = self.n_filters

            self.built = True

            self.conv_layers = []

            with self.session.as_default():
                with self.session.graph.as_default():

                    if self.dim == 1:
                        CNN = tf.keras.layers.Conv1D
                        conv_output_shapes = [[int(inputs.shape[1]), int(inputs.shape[2])]]
                    elif self.dim == 2:
                        CNN = tf.keras.layers.Conv2D
                        conv_output_shapes = [[int(inputs.shape[1]), int(inputs.shape[2]), int(inputs.shape[3])]]
                    elif self.dim == 3:
                        CNN = tf.keras.layers.Conv3D
                        conv_output_shapes = [[int(inputs.shape[1]), int(inputs.shape[2]), int(inputs.shape[3]), int(inputs.shape[4])]]
                    else:
                        raise ValueError('dim must be in [1, 2, 3]')

                    for i in range(self.layers_inner):
                        if isinstance(self.stride, list):
                            cur_strides = self.stride[i]
                        else:
                            cur_strides = self.stride

                        if self.name:
                            name = self.name + '_i%d' % i
                        else:
                            name = None

                        l = CNN(
                            out_dim,
                            self.kernel_size,
                            padding=self.padding,
                            strides=cur_strides,
                            use_bias=self.use_bias,
                            name=name
                        )

                        if self.padding in ['causal', 'same'] and self.stride == 1:
                            output_shape = conv_output_shapes[-1]
                        else:
                            output_shape = [
                                conv_output_length(
                                    x,
                                    self.kernel_size,
                                    self.padding,
                                    self.stride
                                ) for x in conv_output_shapes[-1]
                            ]

                        conv_output_shapes.append(output_shape)

                        self.conv_layers.append(l)

                    self.conv_output_shapes = conv_output_shapes

                    if self.project_inputs:
                        self.projection = tf.keras.layers.Dense(
                            self.conv_output_shapes[-1][0] * out_dim,
                            input_shape=[self.conv_output_shapes[0][0] * self.conv_output_shapes[0][1]]
                        )

            self.built = True

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():
                F = inputs

                for i in range(self.layers_inner - 1):
                    F = self.conv_layers[i](F)

                    if self.batch_normalization_decay:
                        if self.name:
                            name = self.name + '_i%d' % i
                        else:
                            name = None
                        F = tf.contrib.layers.batch_norm(
                            F,
                            decay=self.batch_normalization_decay,
                            center=True,
                            scale=True,
                            zero_debias_moving_mean=True,
                            is_training=self.training,
                            updates_collections=None,
                            reuse=self.reuse,
                            scope=name
                        )
                    if self.activation_inner is not None:
                        F = self.activation_inner(F)

                F = self.conv_layers[-1](F)

                if self.batch_normalization_decay:
                    if self.name:
                        name = self.name + '_i%d' % (self.layers_inner - 1)
                    else:
                        name = None
                    F = tf.contrib.layers.batch_norm(
                        F,
                        decay=self.batch_normalization_decay,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True,
                        is_training=self.training,
                        updates_collections=None,
                        reuse=self.reuse,
                        scope=name
                    )

                if self.project_inputs:
                    x = tf.layers.Flatten()(inputs)
                    x = self.projection(x)
                    x = tf.reshape(x, tf.shape(F))
                else:
                    x = inputs

                H = F + x

                if self.activation is not None:
                    H = self.activation(H)

                return H


class MultiRNNLayer(object):
    def __init__(
            self,
            training=True,
            units=None,
            layers=1,
            activation=None,
            inner_activation='tanh',
            recurrent_activation='sigmoid',
            kernel_initializer='glorot_uniform_initializer',
            bias_initializer='zeros_initializer',
            refeed_outputs=False,
            return_sequences=True,
            name=None,
            session=None
    ):
        self.session = get_session(session)

        self.training = training
        self.units = units
        self.layers = layers
        self.activation = activation
        self.inner_activation = inner_activation
        self.recurrent_activation = recurrent_activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.refeed_outputs = refeed_outputs
        self.return_sequences = return_sequences
        self.name = name

        self.rnn_layer = None

        self.built = False

    def build(self, inputs):
        if not self.built:
            with self.session.as_default():
                with self.session.graph.as_default():
                    # RNN = getattr(tf.keras.layers, self.rnn_type)

                    if self.units is None:
                        units = [inputs.shape[-1]] * self.layers
                    else:
                        units = self.units

                    # self.rnn_layer = RNN(
                    #     out_dim,
                    #     return_sequences=self.return_sequences,
                    #     activation=self.activation,
                    #     recurrent_activation=self.recurrent_activation
                    # )
                    # self.rnn_layer = tf.contrib.rnn.BasicLSTMCell(
                    #     out_dim,
                    #     activation=self.activation,
                    #     name=self.name
                    # )

                    self.rnn_layer = MultiLSTMCell(
                        units,
                        self.layers,
                        training=self.training,
                        activation=self.activation,
                        inner_activation=self.inner_activation,
                        recurrent_activation=self.recurrent_activation,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        refeed_outputs=self.refeed_outputs,
                        name=self.name,
                        session=self.session
                    )

            self.built = True

    def __call__(self, inputs, mask=None):
        if not self.built:
            self.build(inputs)

        with self.session.as_default():
            with self.session.graph.as_default():
                # H = self.rnn_layer(inputs, mask=mask)
                if mask is None:
                    sequence_length = None
                else:
                    sequence_length = tf.reduce_sum(mask, axis=1)

                H, _ = tf.nn.dynamic_rnn(
                    self.rnn_layer,
                    inputs,
                    sequence_length=sequence_length,
                    dtype=tf.float32
                )

                if not self.return_sequences:
                    H = H[:,-1]

                return H