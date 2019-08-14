"""
Michael Patel
August 2019

Python 3.6.5
TensorFlow 1.12.0

Project description:
   To conduct GAN experimentation for learning purposes.
   Based on the DCGAN paper: https://arxiv.org/pdf/1511.06434.pdf

File description:
    To build model definitions for Generator and Discriminator

Dataset: MNIST handwritten digits

"""
################################################################################
# Imports
import tensorflow as tf

from parameters import DROPOUT_RATE


################################################################################
# Generator
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        # Fully connected layer
        self.fc = tf.keras.layers.Dense(
            units=7*7*64,
            use_bias=False
        )

        # Batchnormalization layer
        self.batchnorm = tf.keras.layers.BatchNormalization()

        # Convolutional layer #1
        self.conv1 = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            use_bias=False
        )

        # Convolutional layer #2
        self.conv2 = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False
        )

        # Convolutional layer #3
        self.conv3 = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False
        )

    # forward call
    def call(self, x, training=True):
        x = self.fc(x)
        x = self.batchnorm(x, training=training)
        x = tf.nn.relu(x)

        x = tf.reshape(x, shape=(-1, 7, 7, 64))

        x = self.conv1(x)
        x = self.batchnorm(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.batchnorm(x, training=training)
        x = tf.nn.relu(x)

        x = tf.nn.tanh(self.conv3(x))

        return x


################################################################################
# Discriminator
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Convolutional layer #1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same"
        )

        # Convolutional layer #2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding="same"
        )

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(rate=DROPOUT_RATE)

        # Flattening layer
        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layer
        self.fc = tf.keras.layers.Dense(
            units=1
        )

    # forward pass
    def call(self, x, training=True):
        x = tf.nn.leaky_relu(self.conv1(x))

        x = self.dropout(x, training=training)

        x = tf.nn.leaky_relu(self.conv2(x))

        x = self.dropout(x, training=training)

        x = self.flatten(x)

        x = self.fc(x)

        return x
