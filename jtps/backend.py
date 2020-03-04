import tensorflow as tf
from tensorflow.python.ops import control_flow_ops, state_ops
from tensorflow.python.framework import ops
from tensorflow.python.eager import context


def get_session(session):
    if session is None:
        sess = tf.get_default_session()
    else:
        sess = session

    return sess

## Thanks to Keisuke Fujii (https://github.com/blei-lab/edward/issues/708) for this idea
def get_clipped_optimizer_class(base_optimizer_class, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            class ClippedOptimizer(base_optimizer_class):
                def __init__(self, *args, max_global_norm=None, **kwargs):
                    super(ClippedOptimizer, self).__init__(*args, **kwargs)
                    self.max_global_norm = max_global_norm

                def compute_gradients(self, *args, **kwargs):
                    grads_and_vars = super(ClippedOptimizer, self).compute_gradients(*args, **kwargs)
                    if self.max_global_norm is None:
                        return grads_and_vars
                    grads = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)[0]
                    vars = [v for _, v in grads_and_vars]
                    grads_and_vars = []
                    for grad, var in zip(grads, vars):
                        grads_and_vars.append((grad, var))
                    return grads_and_vars

                def apply_gradients(self, grads_and_vars, **kwargs):
                    if self.max_global_norm is None:
                        return grads_and_vars
                    grads, _ = tf.clip_by_global_norm([g for g, _ in grads_and_vars], self.max_global_norm)
                    vars = [v for _, v in grads_and_vars]
                    grads_and_vars = []
                    for grad, var in zip(grads, vars):
                        grads_and_vars.append((grad, var))

                    return super(ClippedOptimizer, self).apply_gradients(grads_and_vars, **kwargs)

            return ClippedOptimizer


def get_JTPS_optimizer_class(base_optimizer_class, session=None):
    session = get_session(session)
    with session.as_default():
        with session.graph.as_default():
            class JTPSOptimizer(base_optimizer_class):
                def __init__(self, *args, **kwargs):
                    super(JTPSOptimizer, self).__init__(*args, **kwargs)
                    self._delta_prev = None
                    self._delta_prev_t = None
                    self._lambda = None
                    self._lambda_t = None
                    self._lambdas = None
                    self._previous_initialized = False

                def _create_slots(self, var_list):
                    # Create the beta1 and beta2 accumulators on the same device as the first
                    # variable. Sort the var_list to make sure this device is consistent across
                    # workers (these need to go on the same PS, otherwise some updates are
                    # silently ignored).

                    # Create slots for the first and second moments.
                    for v in var_list:
                        self._zeros_slot(v, 'delta', self._name)
                        self._zeros_slot(v, 'theta', self._name)
                        self._get_or_make_slot_with_initializer(v, tf.ones_initializer, v.shape, v.dtype, "lambda", self._name)

                    lambdas = [self.get_slot(var, 'lambda') for var in var_list]
                    super(JTPSOptimizer, self)._create_slots(var_list + lambdas)

                def _apply_dense(self, grad, var):
                    theta_setter_op = self.get_slot(var, 'theta').assign(var)
                    with tf.control_dependencies([theta_setter_op]):
                        base_update_op = super(JTPSOptimizer, self)._apply_dense(grad, var)

                        delta_prev = self.get_slot(var, 'delta')
                        l = self.get_slot(var, 'lambda')
                        var_t = self.get_slot(var, 'theta')

                        with tf.control_dependencies([base_update_op]):
                            delta = var - var_t
                            # delta = tf.Print(delta, [delta])
                            l_grad = grad * delta_prev
                            l_var = self.get_slot(var, 'lambda')
                            l_update = super(JTPSOptimizer, self)._apply_dense(l_grad, l_var)

                            with tf.control_dependencies([l_update]):
                                new_var = var + l * delta
                                var_update = state_ops.assign(var, new_var)

                                delta_prev_update = delta_prev.assign(delta)

                                return control_flow_ops.group(*[var_update, delta_prev_update])

                def _apply_sparse(self, grad, var):
                    raise NotImplementedError("Sparse gradient updates are not supported.")

            return JTPSOptimizer