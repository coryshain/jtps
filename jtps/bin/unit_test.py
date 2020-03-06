import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from jtps.jtps_opt import get_JTPS_optimizer_class

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

optimizer_names = ['GradientDescentOptimizer', 'RMSPropOptimizer', 'AdamOptimizer', 'AdagradOptimizer', 'AdadeltaOptimizer', 'FtrlOptimizer']
N = 10000
P = 1000
M = N // P
D = 100
L = 0.01

with tf.Session() as sess:
    x1 = tf.Variable(tf.zeros([D//2]))
    x2 = tf.Variable(tf.zeros([D//2]))
    x = tf.concat([x1, x2], axis=0)
    y = tf.Variable(tf.linspace(-100., 100., D), trainable=False)
    reset_init = control_flow_ops.group(*[tf.assign(x1, tf.zeros([D//2])), tf.assign(x2, tf.zeros([D//2]))])
    loss = tf.reduce_mean((y - x)**2)

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
                l = tf.concat([tf.reshape(opt.get_slot(var, 'lambda'), [-1]) for var in tf.trainable_variables()], axis=0)
                l_mean = tf.reduce_mean(l)
                loss_all = np.zeros(P)
                lambda_all = np.zeros(P)
            else:
                sys.stderr.write('Fitting using %s...\n' % optimizer_names[o])
                opt = optimizers[o]
                train_op = train_ops[o]
                loss_all = np.zeros(P)

            sess.run(reset_init)

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