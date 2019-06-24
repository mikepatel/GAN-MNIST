# Contains functions to build the following models:
#   - Generator
#   - Discriminator


################################################################################
# IMPORTs
import tensorflow as tf
from tensorflow.layers import conv2d, conv2d_transpose, dense, \
    batch_normalization, flatten, dropout


################################################################################
DROP_RATE = 0.5


################################################################################
# Leaky ReLU
def my_leaky_relu(tensor):
    return tf.nn.leaky_relu(tensor, alpha=0.3)


################################################################################
# GENERATOR
def build_generator(noise, reuse=False):
    with tf.variable_scope("generator") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        #
        t = noise

        t = dense(
            inputs=t,
            units=7*7*256,
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = tf.reshape(t, shape=[-1, 7, 7, 256])

        #
        t = conv2d_transpose(
            inputs=t,
            filters=128,
            kernel_size=[5, 5],
            strides=1,
            padding="same",
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = conv2d_transpose(
            inputs=t,
            filters=64,
            kernel_size=[5, 5],
            strides=2,
            padding="same",
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = conv2d_transpose(
            inputs=t,
            filters=1,
            kernel_size=[5, 5],
            strides=2,
            padding="same",
            activation=tf.tanh
        )

        image = t
        #print("\nGen output image shape: {}".format(image.shape))
        return image


################################################################################
# DISCRIMINATOR
def build_discriminator(image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        #
        t = image

        t = conv2d(
            inputs=t,
            filters=64,
            kernel_size=[5, 5],
            strides=2,
            padding="same",
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = conv2d(
            inputs=t,
            filters=128,
            kernel_size=[5, 5],
            strides=2,
            padding="same",
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = dropout(
            inputs=t,
            rate=DROP_RATE
        )

        t = flatten(
            inputs=t
        )

        t = dense(
            inputs=t,
            units=1,
            activation=tf.sigmoid
        )

        decision = t
        #print("\nD output shape: {}".format(decision.shape))
        return decision
