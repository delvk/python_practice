def add(a, b):
    return a + b


"""
MCNN layer defination using tensorflow

1. conv: convolutional layer
2. pool: pooling layer
3. loss: loss layer to compute the mean square
"""

# import tensorflow and numpy
import tensorflow as tf
import numpy as np


def conv(input_tensor, name, kernel_size, n_output, stride=1, activation=tf.nn.relu):
    """
    Convolutional layer:
    :param input_tensor: Input tensor (feature map/image)
    :param name: name of this convolutional layer
    :param kernel size: size of a square filter matrix
    :param n_out: number of output feature maps
    :param stride: stride value, default = 1
    :param activation_fn: nonlinear activation fucntion, default is relu
    :return: output feature map after activation
    """

    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.Variable(
            tf.truncated_normal(
                shape=(kernel_size, kernel_size, n_in, n_output), stddev=0.01
            ),
            dtype=tf.float32,
            name="weights",
        )
        biases = tf.Variable(
            tf.constant(0.0, shape=[n_output]), dtype=tf.float32, name="biases"
        )
        conv = tf.nn.conv2d(
            input_tensor, weights, (1, stride, stride, 1), padding="SAME"
        )
        activation = activation(tf.nn.bias_add(conv, biases))
        tf.summary.histogram("weights", weights)
        return activation


def pool(input_tensor, name, kernel_size, stride):
    """
    Max Pooling layer
    :param input_tensor: input tensor (feature map) to the pooling layer
    :param name: name of the layer
    :param kernel_size: scale down size 
    :param stride: stride across size,
    :return: output tensor (feature map) with reduced feature size (Scaled down by 2).
    """
    return tf.nn.max_pool(
        input_tensor,
        ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, stride, stride, 1],
        padding="SAME",
        name=name,
    )


def loss(estimate, grouth_truth):
    """
    Computes mean square error between the network estimated density map and the ground truth density map.
    :param est: Estimated density map
    :param gt: Ground truth density map
    :return: scalar loss after doing pixel wise mean square error.
    """
    return tf.losses.mean_squared_error(estimate, grouth_truth)


def test_loss_layer():
    if __name__ == "__main__":
        x = tf.placeholder(tf.float32, [1, 20, 20, 1])
        y = tf.placeholder(tf.float32, [1, 20, 20, 1])
        mse = loss(x, y)
        sess = tf.Session()
        dict = {
            x: 5 * np.ones(shape=(1, 20, 20, 1)),
            y: 1 * np.ones(shape=(1, 20, 20, 1)),
        }
        loss_value = sess.run(mse, feed_dict=dict)
        print('x size: {}'.format(x.shape))
        x = 5 * np.ones(shape=(20, 20))
        y = 1 * np.ones(shape=(20, 20))
        mae = abs(x-y)
        print("MSE: {:.2f}".format(mae))
        print("sum x: {}, sum y: {}".format(np.sum(x), np.sum(y)))
        sess.close()

test_loss_layer()
