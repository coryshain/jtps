import tensorflow as tf
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes, ops
from tensorflow.python.ops import control_flow_ops, gradients, state_ops, resource_variable_ops, variables
from tensorflow.python.training.optimizer import _get_variable_for, _get_processor, Optimizer
from tensorflow.python.util import nest

from jtps.backend import get_session

def get_JTPS_optimizer_class(base_optimizer_class, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            class JTPSOptimizer(base_optimizer_class):
                def __init__(self, *args, meta_learning_rate=None, granularity='variable', linking_fn='softplus', **kwargs):
                    super(JTPSOptimizer, self).__init__(*args, **kwargs)
                    if meta_learning_rate is None:
                        if len(args) > 1:
                            learning_rate = args[0]
                        else:
                            learning_rate = kwargs['learning_rate']
                        self._meta_learning_rate = learning_rate
                    else:
                        self._meta_learning_rate = meta_learning_rate
                    self.granularity = granularity.lower()
                    self.scalar_lambda = None
                    self.linking_fn = linking_fn
                    self._delta_prev = None
                    self._delta_prev_t = None
                    self._lambda = None
                    self._lambda_t = None
                    self._lambdas = None
                    self._previous_initialized = False
                    self._lambda_optimizer_class = self.__class__.__bases__[0]
                    if len(args) > 0:
                        args = list(args)
                        args[0] = self._meta_learning_rate
                    else:
                        kwargs['learning_rate'] = self._meta_learning_rate
                    self.lambda_optimizer = self._lambda_optimizer_class(*args, **kwargs)

                def get_linking_fn(self):
                    fn = self.linking_fn
                    if isinstance(fn, str):
                        if fn.lower() == 'identity':
                            out = lambda x: x
                            inv = lambda x: x
                        elif fn.lower() == 'softplus':
                            out = tf.nn.softplus
                            inv = tf.contrib.distributions.softplus_inverse
                        else:
                            raise ValueError('Unrecognized linking function "%s"' % fn)
                    else:
                        out, inv = fn

                    return out, inv

                def get_lambda(self, var):
                    if self.granularity == 'scalar':
                        return self.scalar_lambda
                    if self.granularity in ['cell', 'variable']:
                        return self.get_slot(var, 'lambda')

                    raise ValueError('Unrecognized value for parameter ``granularity``: "%s"' % self.granularity)

                def get_flattened_lambdas(self, var_list=None):
                    fn, _ = self.get_linking_fn()

                    if var_list is None:
                        var_list = tf.trainable_variables()

                    if self.granularity == 'scalar':
                        lambdas = self.scalar_lambda
                    elif self.granularity == 'variable':
                        lambdas = tf.stack(
                            [self.get_lambda(var) for var in var_list],
                            axis=0
                        )
                    else:
                        lambdas = tf.concat(
                            [tf.reshape(self.get_lambda(var), [-1]) for var in var_list],
                            axis=0
                        )

                    lambdas = fn(lambdas)

                    return lambdas

                def _create_slots(self, var_list):
                    _, fn_inv = self.get_linking_fn()

                    if self.granularity == 'scalar':
                        self.scalar_lambda = tf.Variable(fn_inv(1.).eval(session=session), name='JTPS_lambda')

                    for v in var_list:
                        self._zeros_slot(v, 'delta', self._name)
                        self._zeros_slot(v, 'theta', self._name)
                        if self.granularity == 'cell':
                            self._get_or_make_slot_with_initializer(
                                v,
                                tf.constant_initializer(fn_inv(1.).eval(session=session)),
                                v.shape,
                                v.dtype,
                                "lambda",
                                self._name
                            )
                        elif self.granularity == 'variable':
                            self._get_or_make_slot_with_initializer(
                                v,
                                tf.constant_initializer(fn_inv(1.).eval(session=session)),
                                tf.TensorShape([]),
                                v.dtype,
                                "lambda",
                                self._name
                            )
                        elif self.granularity == 'scalar':
                            pass
                        else:
                            raise ValueError('Unrecognized value for parameter ``granularity``: "%s"' % self.granularity)

                    super(JTPSOptimizer, self)._create_slots(var_list)

                    if self.granularity == 'scalar':
                        lambdas = [self.scalar_lambda]
                    else:
                        lambdas = [self.get_slot(var, 'lambda') for var in var_list]
                    self.lambda_optimizer._create_slots(lambdas)
                    self.lambda_optimizer._prepare()

                def _apply_dense(self, grad, var):
                    theta_setter_op = self.get_slot(var, 'theta').assign(var)
                    with tf.control_dependencies([theta_setter_op]):
                        fn, _ = self.get_linking_fn()

                        base_update_op = super(JTPSOptimizer, self)._apply_dense(grad, var)

                        with tf.control_dependencies([base_update_op]):
                            delta_prev = self.get_slot(var, 'delta')
                            var_t = self.get_slot(var, 'theta')
                            l_var = self.get_lambda(var)

                            delta = var - var_t
                            var_fn = var_t + fn(l_var) * delta_prev
                            l_grad = grad * tf.gradients(var_fn, l_var)[0]
                            if self.granularity != 'cell':
                                l_grad = tf.reduce_sum(l_grad)
                            l_update = self.lambda_optimizer._apply_dense(l_grad, l_var)
                            # l_update = super(JTPSOptimizer, self)._apply_dense(l_grad, l_var)
                            # l_update = l_var.assign_sub(l_grad)
                            # l_update = tf.no_op()

                            with tf.control_dependencies([l_update]):
                                new_var = var_t + l_var * delta
                                # new_var = tf.Print(new_var, [
                                #     'theta grad',
                                #     tf.reduce_mean(grad),
                                #     tf.reduce_min(grad),
                                #     tf.reduce_max(grad),
                                #     'delta prev',
                                #     tf.reduce_mean(delta_prev),
                                #     tf.reduce_min(delta_prev),
                                #     tf.reduce_max(delta_prev),
                                #     'lambda_grad',
                                #     tf.reduce_mean(l_grad),
                                #     tf.reduce_max(l_grad),
                                #     tf.reduce_mean(l_grad)
                                # ])
                                var_update = state_ops.assign(var, new_var)

                                delta_prev_update = delta_prev.assign(delta)

                                return control_flow_ops.group(*[var_update, delta_prev_update])

                def _apply_sparse(self, grad, var):
                    raise NotImplementedError("Sparse gradient updates are not supported.")

            return JTPSOptimizer