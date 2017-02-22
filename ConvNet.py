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

    """
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
        - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
        - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    with tf.Session() as sess:
        X = tf.constant(x, dtype="float")
        W = tf.constant(w, dtype="float")
        B = tf.constant(b, dtype="float")

        X_trans = tf.transpose(X, perm=[0, 2, 3, 1]) # [N, C, H, W] =>[N, H, W, C]
        W_trans = tf.transpose(W, perm=[2, 3, 1, 0]) # [F, C, HH, WW] => [HH, WW, C, F]

        conv_ = tf.nn.conv2d(X, W,
                             strides=[1, conv_param['stride'], conv_param['stride'], 1],
                             padding='SAME')
        res = sess.run(conv_)
        print(res.shape)
        conv_trans = tf.transpose(res, perm=[0,3,1,2])
        print(sess.run(conv_trans).shape)

        return sess.run(conv_trans)

"""Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, out_channels]` """
x_shape = (2, 3, 4, 4)  # batch, channel, width, channel
w_shape = (3, 3, 4, 4) # height, width, channel, output
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)


b = np.linspace(-0.1, 0.2, num=3)

conv_param = {'stride': 2, 'pad': 1}
out= conv_fwd(x, w, b, conv_param)
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
# print(v1.shape)
# print(v3.shape)
# print(v2.shape)
# print(v4.shape)


matrix = np.array([ [[1,2,3], [4, 5, 6]], [[7,8,9],[10, 11,12]]])
matrix.shape
matrix.transpose()





