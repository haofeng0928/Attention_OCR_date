__author__ = 'moonkey'

# from keras import models, layers
import logging
import numpy as np
# from src.data_util.synth_prepare import SynthGen
# import keras.backend as K
import tensorflow as tf


def var_random(name, shape, regularizable=False):
    '''
    Initialize a random variable using xavier initialization.
    Add regularization if regularizable=True
    :param name:
    :param shape:
    :param regularizable:
    :return:
    '''
    v = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    if regularizable:
        with tf.name_scope(name + '/Regularizer/'):
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(v))
    return v


def max_2x2pool(incoming, name):
    """
    max pooling on 2 dims.
    :param incoming:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        return tf.nn.max_pool(incoming, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')


def max_2x1pool(incoming, name):
    """
    max pooling only on image width
    :param incoming:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        return tf.nn.max_pool(incoming, ksize=(1, 2, 1, 1), strides=(1, 2, 1, 1), padding='VALID')


def ConvRelu(incoming, num_filters, filter_size, name):
    """
    Add a convolution layer followed by a Relu layer.
    :param incoming:
    :param num_filters:
    :param filter_size:
    :param name:
    :return:
    """
    num_filters_from = incoming.get_shape().as_list()[3]
    with tf.variable_scope(name):
        conv_W = var_random('W', tuple(filter_size) + (num_filters_from, num_filters), regularizable=True)
        after_conv = tf.nn.conv2d(incoming, conv_W, strides=(1, 1, 1, 1), padding='SAME')
        return tf.nn.relu(after_conv)


def batch_norm(incoming, is_training):
    """
    batch normalization
    :param incoming:
    :param is_training:
    :return:
    """
    return tf.contrib.layers.batch_norm(incoming, is_training=is_training, scale=True, decay=0.99)


def ConvReluBN(incoming, num_filters, filter_size, name, is_training, padding_type='SAME'):
    """
    Convolution -> Batch normalization -> Relu
    :param incoming:
    :param num_filters:
    :param filter_size:
    :param name:
    :param is_training:
    :param padding_type:
    :return:
    """
    num_filters_from = incoming.get_shape().as_list()[3]
    with tf.variable_scope(name):
        conv_W = var_random('W', tuple(filter_size) + (num_filters_from, num_filters), regularizable=True)
        after_conv = tf.nn.conv2d(incoming, conv_W, strides=(1, 1, 1, 1), padding=padding_type)
        after_bn = batch_norm(after_conv, is_training)
        return tf.nn.relu(after_bn)


def dropout(incoming, is_training, keep_prob=0.5):
    return tf.contrib.layers.dropout(incoming, keep_prob=keep_prob, is_training=is_training)


def tf_create_attention_map(incoming):
    '''
    flatten hight and width into one dimention of size attn_length
    :param incoming: 3D Tensor [batch_size x cur_h x cur_w x num_channels]
    :return: attention_map: 3D Tensor [batch_size x attn_length x attn_size].
    '''
    shape = incoming.get_shape().as_list()
    print("shape of incoming is: {}".format(incoming.get_shape()))
    print(shape)
    return tf.reshape(incoming, (-1, np.prod(shape[1:3]), shape[3]))


class CNN(object):
    """
    Usage for tf tensor output:
    o = CNN(x).tf_output()
    """

    def __init__(self, input_tensor, is_training):
        self._build_network(input_tensor, is_training)

    def _build_network(self, input_tensor, is_training):
        """
        https://github.com/bgshih/crnn/blob/master/model/crnn_demo/config.lua
        :return:
        """
        print('CNN input_tensor dim: {}'.format(input_tensor.get_shape()))
        net = tf.transpose(input_tensor, perm=[0, 2, 3, 1])  # (b,c,h,w) => (b,h,w,c) # (?, 32, ?, 1)

        # net = tf.add(net, (-128.0))
        # net = tf.multiply(net, (1/128.0))
        
        net = ConvRelu(net, 64, (3, 3), 'conv_conv1')  # (?, 32, ?, 64)
        net = max_2x2pool(net, 'conv_pool1')  # (?, 16, ?/2, 64)

        net = ConvRelu(net, 128, (3, 3), 'conv_conv2')
        net = max_2x2pool(net, 'conv_pool2')  # (?, 8, ?/2, 128)

        net = ConvReluBN(net, 256, (3, 3), 'conv_conv3', is_training)
        net = ConvRelu(net, 256, (3, 3), 'conv_conv4')
        net = max_2x1pool(net, 'conv_pool3')  # (?, 4, ?, 256)

        net = ConvReluBN(net, 512, (3, 3), 'conv_conv5', is_training)
        net = ConvRelu(net, 512, (3, 3), 'conv_conv6')
        net = max_2x1pool(net, 'conv_pool4')  # (?, 2, ?/2, 512)

        net = ConvReluBN(net, 512, (2, 2), 'conv_conv7', is_training, "VALID")  # (?, 1, ?, 512)
        # (2,2,25,512) => (2,1,24,512)

        net = dropout(net, is_training)
        print('CNN dim before squeeze: {}'.format(net.get_shape()))  # 1x32x100 -> 24x512

        net = tf.squeeze(net, axis=1)  # (?, ?, 512)
        print('CNN output_tensor dim: {}'.format(net.get_shape()))

        self.model = net

    def tf_output(self):
        return self.model
    '''
    def __call__(self, input_tensor):
        return self.model(input_tensor)
    '''
    def save(self):
        pass


def test_feed_dict():
    from numpy import random
    in_put = tf.placeholder(tf.float32, shape=(None, 1, 32, None), name='in_put')

    img_data = random.rand(2,1,32,100)

    cnn_model = CNN(in_put, True)

    out = cnn_model.tf_output()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        res = sess.run(out, feed_dict={in_put: img_data})

        print(res.shape)


def test_cnn():

    img_data = tf.random_normal(shape=(2,1,32,100),mean=0.0,stddev=1.0,dtype=tf.float32)

    cnn_model = CNN(img_data,True)

    out = cnn_model.tf_output()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        res = sess.run(out)

        print(res.shape)


if __name__ == '__main__':

    test_feed_dict()

    #test_cnn()



