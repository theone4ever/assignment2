import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
import tensorflow as tf


class TFConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype


        with tf.Session() as sess:
            w1_init = tf.truncated_normal([num_filters, input_dim[0], filter_size, filter_size], stddev=0.1)
            self.params['W1'] = sess.run(w1_init) # [O, C, HH, WW]
            b1_init = tf.constant(0.1, shape=[num_filters], name="B1")
            self.params['B1'] = sess.run(b1_init) # [O]

            w2_init = tf.truncated_normal([input_dim[1]/2 * input_dim[2]/2*num_filters, hidden_dim], stddev=0.1)
            self.params['W2'] = sess.run(w2_init) # [H/2*W/2*O/2, Hidden_dim]
            b2_init = tf.constant(0.1, shape=[hidden_dim], name="B2")
            self.params['B2'] = sess.run(b2_init) #[hidden_dim]

            w3_init = tf.truncated_normal([hidden_dim, num_classes], stddev=0.1)
            self.params['W3'] = sess.run(w3_init) # [Hidden_dim, num_class]
            b3_init = tf.constant(0.1, shape=[num_classes], name="B3")
            self.params['B3'] = sess.run(b3_init) #[hidden_dim]



        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

         Inputs:
            - X: Array of input data of shape (N, d_1, ..., d_k)
            - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
            If y is None, then run a test-time forward pass of the model and return:
            - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        """
        # W1, b1 = self.params['W1'], self.params['b1']
        # W2, b2 = self.params['W2'], self.params['b2']
        # W3, b3 = self.params['W3'], self.params['b3']


        W1 = tf.Variable(self.params['W1'], name="W1")
        B1 = tf.Variable(self.params['B1'], name="B1")

        W2 = tf.Variable(self.params['W2'], name="W2")
        B2 = tf.Variable(self.params['B2'], name="B2")

        W3 = tf.Variable(self.params['W3'], name="W3")
        B3 = tf.Variable(self.params['B3'], name="B3")

        # X:  (N, C, H, W)
        # W1: (O, C, HH, WW)





        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.params['W1'].shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            x = tf.placeholder(shape=[None, None, None,None], dtype='float')
            x_trans = tf.transpose(x, perm=[0, 2, 3, 1]) # [N, C, H, W] => [n, h, w, c]
            W1_trans = tf.transpose(W1, perm=[2,3,1,0]) #  (O, C, HH, WW)=> [HH, WW, C,  O]
            h_conv = tf.nn.conv2d(x_trans, W1_trans, strides=[1, conv_param['stride'], conv_param['stride'], 1], padding="SAME")+B1 # [b, H, W, O]
            print("conv shape:{0}".format(sess.run(h_conv, feed_dict={x: X}).shape))
            h_conv_relu = tf.nn.relu(h_conv)
            h_pool = tf.nn.max_pool(h_conv_relu,
                                    ksize=[1, pool_param['pool_height'], pool_param['pool_width'], 1],
                                    strides=[1, pool_param['stride'], pool_param['stride'], 1], padding='SAME')
            print(sess.run(h_pool, feed_dict={x: X}).shape)

            h_pool_reshape = tf.reshape(h_pool, [-1, self.params['W1'].shape[0]*16*16])
            h_fc = tf.matmul(h_pool_reshape, W2)+B2
            h_fc_relu = tf.nn.relu(h_fc)

            h_fc2 = tf.matmul(h_fc_relu, W3)+B3
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h_fc2))
            loss = sess.run(cross_entropy, feed_dict={x:X})
            return loss





