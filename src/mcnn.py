"""
Network defination for multi-column Convolutional Neural Network.

3 columns with different receptive fields in order to model crowd at different perspectives.

Fuse layer which concatenates different column outputs and fuses the features with a learning 1x1 filters.

"""
# import library
import tensorflow as tf

# import layer as L
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
        # tf.summary.histogram("weights", weights)
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
    :return: scalar loss after doing pixel wise mse and mae.
    """
    mse = tf.losses.mean_squared_error(estimate, grouth_truth)
    mae = tf.losses.absolute_difference(estimate, grouth_truth)

    return mse, mae


def test_loss_layer():
    if __name__ == "__main__":
        x = tf.placeholder(tf.float32, [1, 20, 20, 1])
        y = tf.placeholder(tf.float32, [1, 20, 20, 1])
        mse = loss(x, y)
        sess = tf.Session()
        dict = {
            x: 5 * np.ones(shape=(1, 20, 20, 1)),
            y: 4 * np.ones(shape=(1, 20, 20, 1)),
        }
        loss_value = sess.run(mse, feed_dict=dict)
        print("MSE: {:.2f}".format(loss_value))
        sess.close()


def first_net_9x9(x):
    net = conv(x, name="conv_9x9_1", kernel_size=9, n_output=16)
    net = pool(net, name="pool_9x9_1", kernel_size=2, stride=2)

    net = conv(net, name="conv_9x9_2", kernel_size=7, n_output=32)
    net = pool(net, name="pool_9x9_2", kernel_size=2, stride=2)

    net = conv(net, name="conv_9x9_3", kernel_size=7, n_output=16)
    net = conv(net, name="conv_9x9_4", kernel_size=7, n_output=8)

    return net


def second_net_7x7(x):
    net = conv(x, name="conv_7x7_1", kernel_size=7, n_output=20)
    net = pool(net, name="pool_7x7_1", kernel_size=2, stride=2)

    net = conv(net, name="conv_7x7_2", kernel_size=5, n_output=40)
    net = pool(net, name="pool_7x7_1", kernel_size=2, stride=2)

    net = conv(net, name="conv_7x7_3", kernel_size=5, n_output=20)
    net = conv(net, name="conv_7x7_4", kernel_size=5, n_output=10)

    return net


def third_net_5x5(x):
    net = conv(x, name="conv_5x5_1", kernel_size=5, n_output=24)
    net = pool(net, name="pool_5x5_1", kernel_size=2, stride=2)

    net = conv(net, name="conv_5x5_2", kernel_size=3, n_output=48)
    net = pool(net, name="pool_5x5_1", kernel_size=2, stride=2)

    net = conv(net, name="conv_5x5_3", kernel_size=3, n_output=24)
    net = conv(net, name="conv_5x5_4", kernel_size=3, n_output=12)

    return net


def fuse_layer(x1, x2, x3):
    x_concat = tf.concat([x1, x2, x3], axis=3)
    return conv(x_concat, name="fuse_1x1_conv", kernel_size=1, n_output=1)


def build_network(input_tensor, norm=False):
    """
    Build the model with 3 column fuse
    - Input: tensor image, resize to 28x28, grayscale
    - Output: estimate density map tensor 

    :param: norm - normalize the image to 0-1 value instead of 0-255
    """
    tf.summary.image("input", input_tensor, 1)
    if norm:
        input_tensor = tf.cast(input_tensor, tf.float32) * (1.0 / 255) - 0.5
    # mapping network
    net_1_output = first_net_9x9(input_tensor)
    net_2_output = second_net_7x7(input_tensor)
    net_3_output = third_net_5x5(input_tensor)
    # fuse network
    fuse_net = fuse_layer(net_1_output, net_2_output, net_3_output)

    return fuse_net


if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [1, 192, 256, 1])
    net = build_network(x)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    d_map = sess.run(
        net, feed_dict={x: 255 * np.ones(shape=(1, 192, 256, 1), dtype=np.float32)}
    )
    prediction = np.asarray(d_map)
    prediction = np.squeeze(prediction, axis=0)
    prediction = np.squeeze(prediction, axis=2)
    print("===========================================")
    print("d_map shape: ", d_map.shape)

