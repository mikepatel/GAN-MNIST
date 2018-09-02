# GAN using matmul operations

# dataset: MNIST
# training set size: 60k
# test set size: 10k
# 28x28x1
# 10 class labels (0-9)
# relatively small dataset

# NOTES:
# reading MNIST data function is deprecated

################################################################################
# IMPORTs
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import datetime

################################################################################
print(tf.__version__)

# load dataset
mnist = input_data.read_data_sets("MNIST_data/")  # deprecated

#
batch_size = 64
noise_size = 32  # defines dimension input to generator
num_epochs = 20000
drop_prob = 0.5


################################################################################
# Discriminator
def discriminator(image, reuse=False):
    with tf.variable_scope("d_") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        input_image = image

        # Layer 1
        d_w1 = tf.get_variable(
            "d_w1",
            shape=[5, 5, 1, 32],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        d_b1 = tf.get_variable(
            "d_b1",
            shape=[32],
            initializer=tf.constant_initializer(0)
        )

        d1 = tf.nn.conv2d(
            input=input_image,
            filter=d_w1,
            strides=[1, 1, 1, 1],
            padding="SAME"
        )

        d1 += d_b1

        d1 = tf.nn.dropout(
            d1,
            keep_prob=drop_prob
        )

        d1 = tf.nn.leaky_relu(
            features=d1,
            alpha=0.2
        )

        # prepare for Layer 2
        d2 = tf.nn.max_pool(
            d1,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME"
        )

        # Layer 2
        d_w2 = tf.get_variable(
            "d_w2",
            shape=[5, 5, 32, 64],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        d_b2 = tf.get_variable(
            "d_b2",
            shape=[64],
            initializer=tf.constant_initializer(0)
        )

        d2 = tf.nn.conv2d(
            input=d2,
            filter=d_w2,
            strides=[1, 1, 1, 1],
            padding="SAME"
        )

        d2 += d_b2

        d2 = tf.nn.dropout(
            d2,
            keep_prob=drop_prob
        )

        d2 = tf.nn.leaky_relu(
            features=d2,
            alpha=0.2
        )

        # prepare for Layer 3
        d3 = tf.nn.max_pool(
            d2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME"
        )

        # Layer 3
        d_w3 = tf.get_variable(
            "d_w3",
            shape=[7*7*64, 1024],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        d_b3 = tf.get_variable(
            "d_b3",
            shape=[1024],
            initializer=tf.constant_initializer(0)
        )

        d3 = tf.reshape(
            d3,
            shape=[-1, 7*7*64]
        )
        d3 = tf.matmul(d3, d_w3)
        d3 += d_b3

        d3 = tf.nn.leaky_relu(
            features=d3,
            alpha=0.2
        )

        d3 = tf.nn.dropout(
            d3,
            keep_prob=drop_prob
        )

        # Layer 4
        d_w4 = tf.get_variable(
            "d_w4",
            shape=[1024, 1],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        d_b4 = tf.get_variable(
            "d_b4",
            shape=[1],
            initializer=tf.constant_initializer(0)
        )

        d4 = tf.matmul(d3, d_w4)
        d4 += d_b4

        d4 = tf.nn.sigmoid(d4)

        return d4


################################################################################
# Generator
def generator(noise, reuse=False):
    with tf.variable_scope("g_") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        input_noise = noise

        # Layer 1
        g_w1 = tf.get_variable(
            "g_w1",
            shape=[32, 1024],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        g_b1 = tf.get_variable(
            "g_b1",
            shape=[1024],
            initializer=tf.constant_initializer(0)
        )

        g1 = tf.matmul(input_noise, g_w1)
        g1 += g_b1

        g1 = tf.nn.leaky_relu(
            features=g1,
            alpha=0.2
        )

        # Layer 2
        g_w2 = tf.get_variable(
            "g_w2",
            shape=[1024, 7*7*64],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        g_b2 = tf.get_variable(
            "g_b2",
            shape=[7*7*64],
            initializer=tf.constant_initializer(0)
        )

        g2 = tf.matmul(g1, g_w2)
        g2 += g_b2

        g2 = tf.nn.leaky_relu(
            features=g2,
            alpha=0.2
        )

        # prepare for Layer 3
        g2 = tf.reshape(g2, shape=[batch_size, 7, 7, 64])

        # Layer 3
        g_w3 = tf.get_variable(
            "g_w3",
            shape=[5, 5, 32, 64],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        g_b3 = tf.get_variable(
            "g_b3",
            shape=[32],
            initializer=tf.constant_initializer(0)
        )

        g3 = tf.nn.conv2d_transpose(
            value=g2,
            filter=g_w3,
            strides=[1, 2, 2, 1],
            output_shape=[batch_size, 14, 14, 32],
            padding="SAME"
        )

        g3 += g_b3

        g3 = tf.nn.leaky_relu(
            features=g3,
            alpha=0.2
        )

        # Layer 4
        g_w4 = tf.get_variable(
            "g_w4",
            shape=[5, 5, 1, 32],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        g_b4 = tf.get_variable(
            "g_b4",
            shape=[1],
            initializer=tf.constant_initializer(0)
        )

        g4 = tf.nn.conv2d_transpose(
            value=g3,
            filter=g_w4,
            strides=[1, 2, 2, 1],
            output_shape=[batch_size, 28, 28, 1],
            padding="SAME"
        )

        g4 += g_b4

        g4 = tf.nn.sigmoid(g4)

        return g4


################################################################################
# Training
d_in = tf.placeholder(
    name="d_in",
    dtype=tf.float32,
    shape=[None, 28, 28, 1]
)

g_in = tf.placeholder(
    name="g_in",
    dtype=tf.float32,
    shape=[None, noise_size]
)

g = generator(g_in, reuse=False)
d_real = discriminator(d_in, reuse=False)
d_fake = discriminator(g, reuse=True)

# Loss functions
d_real_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_real,
        labels=tf.ones_like(d_real)  # cross real w/ 1
    )
)

d_fake_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake,
        labels=tf.zeros_like(d_fake)  # cross fake w/ 0
    )
)

g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake,
        labels=tf.ones_like(d_fake)  # cross generated images from g w/ 1
    )
)

# variable lists
train_vars = tf.trainable_variables()
d_vars = [var for var in train_vars if "d_" in var.name]
g_vars = [var for var in train_vars if "g_" in var.name]

# Optimizer
d_real_opt = tf.train.AdamOptimizer(
    learning_rate=0.0002,
    beta1=0.9
).minimize(d_real_loss, var_list=d_vars)

d_fake_opt = tf.train.AdamOptimizer(
    learning_rate=0.0002,
    beta1=0.9
).minimize(d_fake_loss, var_list=d_vars)

g_opt = tf.train.AdamOptimizer(
    learning_rate=0.0002,
    beta1=0.9
).minimize(g_loss, var_list=g_vars)

# run training
sess = tf.Session()

# tensorboard to view training
tf.summary.scalar(
    name="Discriminator Real Loss",
    tensor=d_real_loss
)

tf.summary.scalar(
    name="Discriminator Fake Loss",
    tensor=d_fake_loss
)

tf.summary.scalar(
    name="Generator Loss",
    tensor=g_loss
)

tensorboard_generated_images = generator(g_in, reuse=True)
tf.summary.image(
    name="Generated Images",
    tensor=tensorboard_generated_images,
    max_outputs=25
)

merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())

# Training iterations
for i in range(num_epochs):
    # discriminator input
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])

    # generator input
    noise_batch = np.random.normal(0, 1, size=[batch_size, noise_size])

    # train discriminator on both real and fake images
    _, _, real_loss_for_d, fake_loss_for_d = sess.run(
        [d_real_opt, d_fake_opt, d_real_loss, d_fake_loss],
        feed_dict={
            d_in: real_image_batch,
            g_in: noise_batch
        }
    )

    # update tensorboard
    if i % 100 == 0:
        noise_batch = np.random.normal(0, 1, size=[batch_size, noise_size])
        summary = sess.run(
            merged,
            feed_dict={
                d_in: real_image_batch,
                g_in: noise_batch
            }
        )
        writer.add_summary(
            summary=summary,
            global_step=i
        )
