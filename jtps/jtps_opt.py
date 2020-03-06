import tensorflow as tf
from tensorflow.python.ops import control_flow_ops, state_ops

from jtps.backend import get_session

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

                        with tf.control_dependencies([base_update_op]):
                            delta_prev = self.get_slot(var, 'delta')
                            var_t = self.get_slot(var, 'theta')
                            delta = var - var_t
                            l_grad = grad * delta_prev
                            l_var = self.get_slot(var, 'lambda')
                            l_update = super(JTPSOptimizer, self)._apply_dense(l_grad, l_var)
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