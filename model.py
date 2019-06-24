"""
Michael Patel
June 2019

Python 3.6.5
TF 1.12.0

Project description:

Datasets:
    - MNIST

Notes:
    - DCGAN: https://arxiv.org/pdf/1511.06434.pdf
    -

Things to examine:

"""
################################################################################
# Imports
import tensorflow as tf
from parameters import *


################################################################################
# Generator
def build_generator():
    m = tf.keras.Sequential()

    #
    m.add(tf.keras.layers.Dense(
        units=7*7*256,
        input_shape=(Z_DIM, )
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU())

    #
    m.add(tf.keras.layers.Reshape((7, 7, 256)))

    #
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU())

    #
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=64,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU())

    #
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        activation=tf.keras.activations.tanh
    ))

    m.summary()

    return m


################################################################################
# Discriminator
def build_discriminator():
    m = tf.keras.Sequential()

    #
    m.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        input_shape=(28, 28, 1)
    ))
    m.add(tf.keras.layers.LeakyReLU())

    #
    m.add(tf.keras.layers.Dropout(rate=0.3))  # fraction of input units to drop

    #
    m.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same"
    ))
    m.add(tf.keras.layers.LeakyReLU())

    #
    m.add(tf.keras.layers.Dropout(rate=0.3))

    #
    m.add(tf.keras.layers.Flatten())

    #
    m.add(tf.keras.layers.Dense(
        units=1
    ))
