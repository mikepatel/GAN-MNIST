"""
Michael Patel
June 2019

Python 3.6.5
TF 1.12.0

File description:
    For model definitions for Generator and Discriminator
"""
################################################################################
# Imports
import tensorflow as tf

from parameters import DROPOUT_RATE, LEAKY_ALPHA


################################################################################
# Leaky ReLU
def my_leaky_relu(tensor):
    return tf.nn.leaky_relu(tensor, alpha=LEAKY_ALPHA)


################################################################################
# Generator
def build_generator(noise, reuse=False):
    with tf.variable_scope("generator") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # Input
        t_input = noise

        # Layer 1
        t = tf.layers.conv2d_transpose(
            inputs=t_input,
            filters=1024,
            kernel_size=(4, 4),
            kernel_initializer=tf.zeros_initializer(),
            strides=(1, 1),
            padding="valid"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 2
        t = tf.layers.conv2d_transpose(
            inputs=t,
            filters=512,
            kernel_size=(4, 4),
            kernel_initializer=tf.zeros_initializer(),
            strides=(2, 2),
            padding="same"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 3
        t = tf.layers.conv2d_transpose(
            inputs=t,
            filters=256,
            kernel_size=(4, 4),
            kernel_initializer=tf.zeros_initializer(),
            strides=(2, 2),
            padding="same"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 4
        t = tf.layers.conv2d_transpose(
            inputs=t,
            filters=128,
            kernel_size=(4, 4),
            kernel_initializer=tf.zeros_initializer(),
            strides=(2, 2),
            padding="same"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 5
        t = tf.layers.conv2d_transpose(
            inputs=t,
            filters=1,
            kernel_size=(4, 4),
            kernel_initializer=tf.zeros_initializer(),
            strides=(2, 2),
            padding="same"
        )

        t = tf.nn.tanh(t)

        # Output
        t_output = t

        print("Generator output shape: {}".format(t_output.shape))
        return t_output


################################################################################
# Discriminator
def build_discriminator(image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # Input
        t_input = image

        # Layer 1
        t = tf.layers.conv2d(
            inputs=t_input,
            filters=128,
            kernel_size=(4, 4),
            kernel_initializer=tf.zeros_initializer(),
            strides=(2, 2),
            padding="same"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 2
        t = tf.layers.conv2d(
            inputs=t,
            filters=256,
            kernel_size=(4, 4),
            kernel_initializer=tf.zeros_initializer(),
            strides=(2, 2),
            padding="same"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 3
        t = tf.layers.conv2d(
            inputs=t,
            filters=512,
            kernel_size=(4, 4),
            kernel_initializer=tf.zeros_initializer(),
            strides=(2, 2),
            padding="same"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 4
        t = tf.layers.conv2d(
            inputs=t,
            filters=1024,
            kernel_size=(4, 4),
            kernel_initializer=tf.zeros_initializer(),
            strides=(2, 2),
            padding="same"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 5
        t = tf.layers.conv2d(
            inputs=t,
            filters=1,
            kernel_size=(4, 4),
            kernel_initializer=tf.zeros_initializer(),
            strides=(1, 1),
            padding="valid"
        )

        t = tf.nn.sigmoid(t)

        t_output = t

        print("Discriminator output shape: {}".format(t_output.shape))
        return t_output
