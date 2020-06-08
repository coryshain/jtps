import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from jtps.opt import get_JTPS_optimizer_class

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

optimizer_names = ['GradientDescentOptimizer', 'RMSPropOptimizer', 'AdamOptimizer']
N = 500
P = 500
M = N // P
D = 100
L = 0.001

with tf.Session() as sess:
    # Optimizing Rosenbrock function
    np.random.seed(0)
    x1_init = np.random.uniform(low=-3, high=3, size=(1,))
    x2_init = np.random.uniform(low=-3, high=3, size=(1,))

    x1 = tf.Variable(x1_init)
    x2 = tf.Variable(x2_init)
    loss = (1 - x1)**2 + 100 * (x2 - x1**2)**2
    reset_init = control_flow_ops.group(*[tf.assign(x1, x1_init), tf.assign(x2, x2_init)])

    optimizers = []
    optimizers_jtps = []
    for name in optimizer_names:
        optimizers.append(getattr(tf.train, name)(L))
        optimizers_jtps.append(get_JTPS_optimizer_class(getattr(tf.train, name), session=sess)(L))
    train_ops = [opt.minimize(loss) for opt in optimizers]
    train_ops_jtps = [opt.minimize(loss) for opt in optimizers_jtps]

    sess.run(tf.global_variables_initializer())

    table = {}

    for o in range(len(optimizer_names)):
        for USE_JTPS in [False, True]:
            if USE_JTPS:
                sys.stderr.write('Fitting using %s-JTPS...\n' % optimizer_names[o])
                opt = optimizers_jtps[o]
                train_op = train_ops_jtps[o]
                l = opt.get_flattened_lambdas()
                l_mean = tf.reduce_mean(l)
                loss_all = np.zeros(P)
                lambda_all = np.zeros(P)
            else:
                sys.stderr.write('Fitting using %s...\n' % optimizer_names[o])
                opt = optimizers[o]
                train_op = train_ops[o]
                loss_all = np.zeros(P)

            sess.run(reset_init)
            _x1, _x2, _y = sess.run([x1, x2, loss])
            print(_x1)
            print(_x2)
            print(_y)

            for i in range(N):
                pb = tf.contrib.keras.utils.Progbar(N)
                if USE_JTPS:
                    _, loss_cur, l_mean_cur = sess.run([train_op, loss, l_mean])
                    if i % M == 0:
                        loss_all[i // M] = loss_cur
                        lambda_all[i // M] = l_mean_cur
                    pb.update(i + 1, values=[('loss', loss_cur), ('mean_lambda', l_mean_cur)])
                else:
                    _, loss_cur = sess.run([train_op, loss])
                    if i % M == 0:
                        loss_all[i // M] = loss_cur
                    pb.update(i + 1, values=[('loss', loss_cur)])

            if USE_JTPS:
                table[optimizer_names[o] + 'JTPSLoss'] = loss_all
                table[optimizer_names[o] + 'JTPSLambda'] = lambda_all
            else:
                table[optimizer_names[o] + 'Loss'] = loss_all

    table = pd.DataFrame(table)
    table.to_csv('learning_curves.csv', index=False)