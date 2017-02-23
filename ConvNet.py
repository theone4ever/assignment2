# As usual, a bit of setup


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt
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
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

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
        W_trans = tf.transpose(W, perm=[2, 3, 1, 0]) # [F, C, HH, WW] => [HH, WW, C, O]


        conv_ = tf.nn.conv2d(X_trans, W_trans,
                             strides=[1, conv_param['stride'], conv_param['stride'], 1],
                             padding='SAME')+b  # [b, H, W, O]


        print(sess.run(conv_).shape)
        conv_trans = tf.transpose(conv_, perm=[0,3,1,2])

        return sess.run(conv_trans)



from scipy.misc import imread, imresize

kitten, puppy = imread('kitten.jpg'), imread('puppy.jpg')
# kitten is wide, and puppy is already square
d = kitten.shape[1] - kitten.shape[0]
kitten_cropped = kitten[:, d/2:-d/2, :]

img_size = 200   # Make this smaller if it runs too slow
x = np.zeros((2, 3, img_size, img_size))
x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1))
x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))

# Set up a convolutional weights holding 2 filters, each 3x3
w = np.zeros((2, 3, 3, 3)) # output, width, heigh, channel

# The first filter converts the image to grayscale.
# Set up the red, green, and blue channels of the filter.
w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]

# Second filter detects horizontal edges in the blue channel.
w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

# Vector of biases. We don't need any bias for the grayscale
# filter, but for the edge detection filter we want to add 128
# to each output so that nothing is negative.
b = np.array([0, 128])

def conv_fwd2(x, w, b, conv_param):

    """
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (O, WW, HH, C)
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
        X1 = tf.constant(x, dtype="float")
        W1 = tf.constant(w, dtype="float")
        B1 = tf.constant(b, dtype="float")

        X_trans1 = tf.transpose(X1, perm=[0, 2, 3, 1]) # [N, C, H, W] =>[N, H, W, C]
        W_trans1 = tf.transpose(W1, perm=[2, 1, 3, 0]) # [O, WW, HH, C] => [HH, WW, C, O]


        conv1_ = tf.nn.conv2d(X_trans1, W_trans1,
                              strides=[1, conv_param['stride'], conv_param['stride'], 1],
                              padding='SAME')+B1  # [b, H, W, O]


        print(sess.run(conv1_).shape)
        conv_trans1 = tf.transpose(conv1_, perm=[0,3,1,2])

        return sess.run(conv_trans1)


def conv_max_pool(x, pool_param):
    """
   - x: Input data of shape (N, C, H, W)
   - pool_param: A dictionary with the following keys:
       - 'pool_width':
       - 'pool_height':
       - 'stride':

   Returns a tuple of:
   - out: Output data, of shape (N, C, H, W) where H' and W' are given by
   """
    with tf.Session() as sess:
        X = tf.constant(x, dtype="float")
        X_trans = tf.transpose(X, perm=[0, 2, 3, 1])
        pooling = tf.nn.max_pool(X_trans, ksize=[1, pool_param['pool_height'], pool_param['pool_width'], 1], strides=[1,pool_param['stride'],pool_param['stride'], 1], padding='SAME')
        pooling_trans = tf.transpose(pooling, perm=[0, 3, 1,2])  # [b, H, W, C]  =>  (b, C, H, W)
        return sess.run(pooling_trans)

# Compute the result of convolving each input in x with each filter in w,
# offsetting by b, and storing the results in out.
out, _ = conv_fwd2(x, w, b, {'stride': 1, 'pad': 1})

def imshow_noax(img, normalize=True):
    print("image shape:")
    print(img.shape)
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')


plt.interactive(False)
# Show the original images and the results of the conv operation
plt.subplot(2, 3, 1)
imshow_noax(puppy, normalize=False)
plt.title('Original image')
plt.subplot(2, 3, 2)
imshow_noax(out[0, 0])
plt.title('Grayscale')
plt.subplot(2, 3, 3)
imshow_noax(out[0, 1])
plt.title('Edges')
plt.subplot(2, 3, 4)
imshow_noax(kitten_cropped, normalize=False)
plt.subplot(2, 3, 5)
imshow_noax(out[1, 0])
plt.subplot(2, 3, 6)
imshow_noax(out[1, 1])
plt.show()








