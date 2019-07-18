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
            kernel_size=[4, 4],
            strides=(1, 1),
            padding="valid"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 2
        t = tf.layers.conv2d_transpose(
            inputs=t,
            filters=512,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 3
        t = tf.layers.conv2d_transpose(
            inputs=t,
            filters=256,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 4
        t = tf.layers.conv2d_transpose(
            inputs=t,
            filters=128,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 5
        t = tf.layers.conv2d_transpose(
            inputs=t,
            filters=1,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same"
        )

        t = tf.nn.tanh(t)

        # Output
        t_output = t

        print("Generator output shape: {}".format(t_output.shape))
        return t_output

        """
        # Input layer
        t = tf.layers.dense(
            inputs=t_input,
            units=7*7*512,
            kernel_initializer=tf.initializers.random_normal(stddev=0.02),
            #kernel_initializer=tf.initializers.glorot_normal,
            activation=tf.nn.relu
        )

        t = tf.layers.batch_normalization(inputs=t)

        # Reshape layer
        t = tf.reshape(
            tensor=t,
            shape=[-1, 7, 7, 512]
        )

        # Conv layer 1
        t = tf.layers.conv2d_transpose(
            inputs=t,
            filters=256,
            kernel_size=[3, 3],
            kernel_initializer=tf.initializers.random_normal(stddev=0.02),
            #kernel_initializer=tf.initializers.glorot_normal,
            strides=[2, 2],
            padding="same",
            activation=tf.nn.relu
        )

        t = tf.layers.batch_normalization(inputs=t)

        # Conv layer 2
        t = tf.layers.conv2d_transpose(
            inputs=t,
            filters=128,
            kernel_size=[3, 3],
            kernel_initializer=tf.initializers.random_normal(stddev=0.02),
            #kernel_initializer=tf.initializers.glorot_normal,
            strides=[2, 2],
            padding="same",
            activation=tf.nn.relu
        )

        t = tf.layers.batch_normalization(inputs=t)

        # Conv layer 3
        t = tf.layers.conv2d_transpose(
            inputs=t,
            filters=1,
            kernel_size=[3, 3],
            kernel_initializer=tf.initializers.random_normal(stddev=0.02),
            #kernel_initializer=tf.initializers.glorot_normal,
            strides=[1, 1],
            padding="same",
            activation=tf.tanh
        )

        # Output
        t_output = t

        print("Generator output shape: {}".format(t_output.shape))
        return t_output
        """

    """
    m = tf.keras.Sequential()

    # Input layer
    m.add(tf.keras.layers.Dense(
        units=7*7*512,
        input_shape=(Z_DIM, )
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU())
    #m.add(tf.keras.layers.ReLU())

    # Reshape layer
    m.add(tf.keras.layers.Reshape((7, 7, 512)))

    # Convolutional Layer 1
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=512,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU())

    # Convolutional Layer 2
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=256,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU())

    # Convolutional Layer 3
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same"
    ))
    m.add(tf.keras.layers.BatchNormalization())
    m.add(tf.keras.layers.LeakyReLU())

    # Convolutional Layer 4
    m.add(tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        activation=tf.keras.activations.tanh
    ))

    # configure model training
    m.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
        metrics=["accuracy"]
    )

    m.summary()

    return m
    """


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
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 2
        t = tf.layers.conv2d(
            inputs=t,
            filters=256,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 3
        t = tf.layers.conv2d(
            inputs=t,
            filters=512,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 4
        t = tf.layers.conv2d(
            inputs=t,
            filters=1024,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same"
        )

        t = tf.layers.batch_normalization(inputs=t)

        t = tf.nn.leaky_relu(features=t, alpha=LEAKY_ALPHA)

        # Layer 5
        t = tf.layers.conv2d(
            inputs=t,
            filters=1,
            kernel_size=[4, 4],
            strides=(1, 1),
            padding="valid"
        )

        t = tf.nn.sigmoid(t)

        t_output = t

        print("Discriminator output shape: {}".format(t_output.shape))
        return t_output

        """
        # Conv layer 1
        t = tf.layers.conv2d(
            inputs=t_input,
            filters=64,
            kernel_size=[3, 3],
            kernel_initializer=tf.initializers.random_normal(stddev=0.02),
            #kernel_initializer=tf.initializers.glorot_normal,
            strides=[2, 2],
            padding="same",
            activation=my_leaky_relu
        )

        t = tf.layers.batch_normalization(inputs=t)

        # Dropout 1
        t = tf.layers.dropout(
            inputs=t,
            rate=DROPOUT_RATE
        )

        # Conv layer 2
        t = tf.layers.conv2d(
            inputs=t,
            filters=128,
            kernel_size=[3, 3],
            kernel_initializer=tf.initializers.random_normal(stddev=0.02),
            #kernel_initializer=tf.initializers.glorot_normal,
            strides=[2, 2],
            padding="same",
            activation=my_leaky_relu
        )

        t = tf.layers.batch_normalization(inputs=t)

        # Dropout 2
        t = tf.layers.dropout(
            inputs=t,
            rate=DROPOUT_RATE
        )

        # Flatten
        t = tf.layers.flatten(inputs=t)

        # Output
        t = tf.layers.dense(
            inputs=t,
            units=1,
            activation=tf.sigmoid
        )
        """



    """
    m = tf.keras.Sequential()

    # Convolutional Layer 1
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

    # Convolutional Layer 2
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

    # configure model training
    m.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
        metrics=["accuracy"]
    )

    m.summary()

    return m
    """
