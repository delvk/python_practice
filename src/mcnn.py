"""
Network defination for multi-column Convolutional Neural Network.

3 columns with different receptive fields in order to model crowd at different perspectives.

Fuse layer which concatenates different column outputs and fuses the features with a learning 1x1 filters.

"""
# import library
import tensorflow as tf
import layer as L
import numpy as np


def first_net_9x9(x):
    net = L.conv(x, name='conv_9x9_1', kernel_size=9, n_output=16)
    net = L.pool(net, name='pool_9x9_1', kernel_size=2, stride=2)

    net = L.conv(net, name='conv_9x9_2', kernel_size=7, n_output=32)
    net = L.pool(net, name='pool_9x9_2', kernel_size=2, stride=2)

    net = L.conv(net, name='conv_9x9_3', kernel_size=7, n_output=16)
    net = L.conv(net, name='conv_9x9_4', kernel_size=7, n_output=8)

    return net


def second_net_7x7(x):
    net = L.conv(x, name='conv_7x7_1', kernel_size=7, n_output=20)
    net = L.pool(net, name='pool_7x7_1', kernel_size=2, stride=2)

    net = L.conv(net, name='conv_7x7_2', kernel_size=5, n_output=40)
    net = L.pool(net, name='pool_7x7_1', kernel_size=2, stride=2)

    net = L.conv(net, name='conv_7x7_3', kernel_size=5, n_output=20)
    net = L.conv(net, name='conv_7x7_4', kernel_size=5, n_output=10)

    return net


def third_net_5x5(x):
    net = L.conv(x, name='conv_5x5_1', kernel_size=5, n_output=24)
    net = L.pool(net, name='pool_5x5_1', kernel_size=2, stride=2)

    net = L.conv(net, name='conv_5x5_2', kernel_size=3, n_output=48)
    net = L.pool(net, name='pool_5x5_1', kernel_size=2, stride=2)

    net = L.conv(net, name='conv_5x5_3', kernel_size=3, n_output=24)
    net = L.conv(net, name='conv_5x5_4', kernel_size=3, n_output=12)

    return net


def fuse_layer(x1, x2, x3):
    x_concat = tf.concat([x1, x2, x3], axis=3)
    return L.conv(x_concat, name="fuse_1x1_conv", kernel_size=1, n_output=1)


def build_network(input_tensor, norm=False):
    """
    Build the model with 3 column fuse
    - Input: tensor image, resize to 28x28, grayscale
    - Output: estimate density map tensor 

    :param: norm - normalize the image to 0-1 value instead of 0-255
    """
    tf.summary.image('input', input_tensor, 1)
    if norm:
        input_tensor = tf.cast(input_tensor, tf.float32)*(1./255)-0.5
    # mapping network
    net_1_output = first_net_9x9(input_tensor)
    net_2_output = second_net_7x7(input_tensor)
    net_3_output = third_net_5x5(input_tensor)
    # fuse network
    fuse_net = fuse_layer(net_1_output, net_2_output, net_3_output)

    return fuse_net