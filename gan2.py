# GAN

# dcgan
# using tf.keras

# dataset: MNIST

# Notes:
#   - relu -> LeakyReLU?
#   - Add more layer depth?

##################################################################################################
# IMPORTs
import os
import numpy as np
from datetime import datetime

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Reshape, Dropout, \
    BatchNormalization, Flatten, LeakyReLU
from tensorflow.keras.activations import tanh, sigmoid, relu
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing import image

##################################################################################################
# HYPERPARAMETERS and DESIGN CHOICES

# based on MNIST images: 28x28 greyscale
num_rows = 28
num_cols = 28
num_channels = 1

latent_dim = 100
NUM_EPOCHS = 10000
BATCH_SIZE = 64
DROPOUT_RATE = 0.2
LEAKY_RELU_ALPHA = 0.2

##################################################################################################
# load dataset
(train_images, train_labels), (_, _) = mnist.load_data()

# rescale -1 to 1 and normalize
# use in coordination with tanh activation
train_images = train_images.reshape(train_images.shape[0], num_rows, num_cols, num_channels)
train_images = train_images.astype(np.float32)
train_images = (train_images - 127.5) / 127.5

##################################################################################################
# GENERATOR
g = Sequential()

g.add(Dense(
    units=7*7*128,
    input_dim=latent_dim,
    activation=relu
))

g.add(BatchNormalization())

g.add(Reshape(
    target_shape=(7, 7, 128)
))

g.add(Conv2DTranspose(
    filters=64,
    kernel_size=[5, 5],
    strides=2,
    padding="same",
    activation=relu
))

g.add(BatchNormalization())

g.add(Conv2DTranspose(
    filters=32,
    kernel_size=[5, 5],
    strides=2,
    padding="same",
    activation=relu
))

g.add(BatchNormalization())

g.add(Conv2DTranspose(
    filters=1,
    kernel_size=[5, 5],
    strides=1,
    padding="same",
    activation=tanh
))

g.summary()

##################################################################################################
# DISCRIMINATOR
d = Sequential()

d.add(Conv2D(
    filters=32,
    kernel_size=[5, 5],
    strides=2,
    input_shape=(num_rows, num_cols, num_channels),
    padding="same",
    activation=relu
))

d.add(Dropout(rate=DROPOUT_RATE))

d.add(Conv2D(
    filters=64,
    kernel_size=[5, 5],
    strides=2,
    padding="same",
    activation=relu
))

d.add(Dropout(rate=DROPOUT_RATE))

d.add(Conv2D(
    filters=128,
    kernel_size=[5, 5],
    strides=1,
    padding="same",
    activation=relu
))

d.add(Dropout(rate=DROPOUT_RATE))

d.add(Flatten())

d.add(Dense(
    units=1,
    activation=sigmoid
))

d.summary()

##################################################################################################
# TRAINING
g.compile(
    loss=binary_crossentropy,
    optimizer=Adam(),
    metrics=["accuracy"]
)

d.compile(
    loss=binary_crossentropy,
    optimizer=Adam(),
    metrics=["accuracy"]
)

d.trainable = False  # freeze weights

gan = Sequential()
gan.add(g)
gan.add(d)
gan.compile(
    loss=binary_crossentropy,
    optimizer=Adam(),
    metrics=["accuracy"]
)

# callbacks
dir = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
if not os.path.exists(dir):
    os.makedirs(dir)

history_file = dir + "\gan_mnist_keras.h5"
save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
tb_callback = TensorBoard(log_dir=dir)



for e in range(NUM_EPOCHS+1):
    # generator
    noise_vector = np.random.rand(BATCH_SIZE, latent_dim)
    gen_images = g.predict(noise_vector)

    # discriminator
    indices = np.random.randint(0, train_images.shape[0], size=BATCH_SIZE)
    real_images = train_images[indices]

    # concatentate real and gen images for discriminator
    d_images = np.concatenate((real_images, gen_images))

    # labels "real", "fake"
    real_labels = np.ones((BATCH_SIZE, 1))
    fake_labels = np.zeros((BATCH_SIZE, 1))

    # add random noise to labels
    real_labels += 0.05 * np.random.random(real_labels.shape)
    fake_labels += 0.05 * np.random.random(fake_labels.shape)

    # concatenate labels
    d_labels = np.concatenate((real_labels, fake_labels))

    # train discriminator
    d.trainable = True
    d_loss = d.train_on_batch(d_images, d_labels)

    # misleading labels: "all these images are real" - obviously a lie
    misleading_labels = np.ones((BATCH_SIZE, 1))

    # train generator
    d.trainable = False
    noise_vector = np.random.rand(BATCH_SIZE, latent_dim)
    gan_loss = gan.train_on_batch(noise_vector, misleading_labels)

    if e % 200 == 0:
        # save model weights
        gan.save_weights(history_file)

        # print metrics
        print("\nEpoch: {}".format(e))
        print("Discriminator Real: ", d_loss[0])
        print("Discriminator Fake: ", d_loss[1])
        print("GAN: ", gan_loss)

    if e % 500 == 0:
        # save a generated image
        image_file = dir + "\\" + str(e) + "_gen.png"
        img = image.array_to_img(gen_images[0] * 255., scale=False)
        img.save(image_file)

        # save a real image for comparison
        image_file = dir + "\\" + str(e) + "_real.png"
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(image_file)
