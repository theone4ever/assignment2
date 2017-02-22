# As usual, a bit of setup


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
# import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver


import argparse
import sys

import tensorflow as tf

# %matplotlib inline
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2

# def rel_error(x, y):
#     """ returns relative error """
#     return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
#
#
# data = get_CIFAR10_data()
# for k, v in data.iteritems():
#     print('%s: ' % k, v.shape)



FLAGS = None

total_epoch = 5
batch_size = 50
display_step = 1


def conv_fwd(x, w, b, conv_param):
    with tf.Session() as sess:
        X = tf.constant(x, dtype="float")
        W = tf.constant(w, dtype="float")
        B = tf.constant(b, dtype="float")

        print([conv_param['stride'],
               conv_param['stride'], conv_param['stride'], conv_param['stride']])
        conv_ = tf.nn.conv2d(X, W,
                             strides=[1,
                                      conv_param['stride'], conv_param['stride'], 1],
                             padding='SAME')+B

        return sess.run(conv_)

"""Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, out_channels]` """
x_shape = (2, 4, 4, 3)  # batch, height, width, channel
w_shape = (4, 4, 3, 3) # height, width, channel, output
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)



b = np.linspace(-0.1, 0.2, num=3)

conv_param = {'stride': 2, 'pad': 1}
# out, _ = conv_fwd(x, w, b, conv_param)
# print(out)




x1_shape = (2, 3, 4, 4)  # batch, height, width, channel
w1_shape = (3, 3, 4, 4) # output, height, width, channel
x1 = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x1_shape)
w1 = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w1_shape)

sess = tf.Session()
X1 = tf.constant(x1)
W1 = tf.constant(w1)
v1, v2 = sess.run([X1, W1])


X2 = tf.transpose(X1, perm=[0, 2, 3, 1])
W2 = tf.transpose(W1, perm=[3, 2, 0, 1])

v3, v4 = sess.run([X2, W2])
print(v1.shape)
print(v3.shape)
print(v2.shape)
print(v4.shape)




