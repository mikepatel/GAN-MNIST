"""
Michael Patel
April 2020

Project description:
    Build a GAN using the MNIST data set

File description:
    For model definitions
"""
################################################################################
import tensorflow as tf

from parameters import *


################################################################################
# Discriminator
def build_discriminator():
    m = tf.keras.Sequential()

    # Layer 1: Conv: 14x14x64
    m.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        strides=2,
        padding="same",
        input_shape=(28, 28, 1)
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 2: Dropout
    m.add(tf.keras.layers.Dropout(rate=DROPOUT_RATE))

    # Layer 3: Conv: 7x7x128
    m.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(5, 5),
        strides=2,
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 4: Dropout
    m.add(tf.keras.layers.Dropout(rate=DROPOUT_RATE))

    # Layer 5: Flatten
    m.add(tf.keras.layers.Flatten())

    # Layer 6: Output
    m.add(tf.keras.layers.Dense(
        units=1
    ))

    return m


# Generator
def build_generator():
    m = tf.keras.Sequential()

    # Layer 1: Fully connected
    m.add(tf.keras.layers.Dense(
        units=7*7*256,
        use_bias=False,
        input_shape=(100, )
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 2: Reshape
    m.add(tf.keras.layers.Reshape(
        target_shape=(7, 7, 256)
    ))

    # Layer 3: Conv: 7x7x128
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=(5, 5),
        strides=1,
        padding="same",
        use_bias=False
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 4: Conv: 14x14x64
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=64,
        kernel_size=(5, 5),
        strides=2,
        padding="same",
        use_bias=False
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer 5: Conv: 28x28x1
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=(5, 5),
        strides=2,
        padding="same",
        use_bias=False,
        activation=tf.keras.activations.tanh
    ))

    return m
