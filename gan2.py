# GAN

# dataset: MNIST

# using tf.keras

# Notes:

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
# HYPERPARAMETERS and CONSTANTS

# based on MNIST images: 28x28 greyscale
num_rows = 28
num_cols = 28
num_channels = 1

latent_dim = 100
NUM_EPOCHS = 10000
BATCH_SIZE = 256


##################################################################################################
class GAN():
    def __init__(self):
        self.d = None
        self.g = None
        self.gan = None

    # DISCRIMINATOR
    def build_discriminator(self):
        d = Sequential()

        d.add(Conv2D(
            filters=64,
            kernel_size=[5, 5],
            strides=2,
            input_shape=(num_rows, num_cols, num_channels),
            padding="same",
            activation=LeakyReLU(0.2)
        ))

        d.add(Dropout(rate=0.4))

        d.add(Conv2D(
            filters=128,
            kernel_size=[5, 5],
            strides=2,
            padding="same",
            activation=LeakyReLU(0.2)
        ))

        d.add(Dropout(rate=0.4))

        d.add(Conv2D(
            filters=256,
            kernel_size=[5, 5],
            strides=2,
            padding="same",
            activation=LeakyReLU(0.2)
        ))

        d.add(Dropout(rate=0.4))

        d.add(Conv2D(
            filters=512,
            kernel_size=[5, 5],
            strides=1,
            padding="same",
            activation=LeakyReLU(0.2)
        ))

        d.add(Flatten())

        d.add(Dropout(rate=0.4))

        d.add(Dense(
            units=1,
            activation=sigmoid
        ))

        #d.summary()
        return d

    # GENERATOR
    def build_generator(self):
        g = Sequential()

        g.add(Dense(
            units=7 * 7 * 256,  # 7x7 64-channel fmap
            input_dim=latent_dim,
            activation=relu
        ))

        g.add(BatchNormalization())

        g.add(Reshape(
            target_shape=(7, 7, 256)
        ))

        g.add(Conv2DTranspose(
            filters=128,
            kernel_size=[5, 5],
            strides=2,
            padding="same",
            activation=relu
        ))

        g.add(BatchNormalization())

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
            strides=1,
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

        #g.summary()
        return g

    # GAN
    def build_gan(self):
        gan = Sequential()

        gan.add(self.build_generator())
        gan.add(self.build_discriminator())
        gan.compile(
            loss=binary_crossentropy,
            optimizer=Adam(),
            metrics=["accuracy"]
        )

        #gan.summary()
        return gan

    # TRAIN
    def train(self, train_images):
        self.d = self.build_discriminator()
        self.d.compile(
            loss=binary_crossentropy,
            optimizer=Adam(),
            metrics=["accuracy"]
        )

        self.gan = self.build_gan()

        self.g = self.build_generator()

        # callbacks
        dir = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
        if not os.path.exists(dir):
            os.makedirs(dir)

        history_file = dir + "\gan_mnist_keras.h5"
        save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
        tb_callback = TensorBoard(log_dir=dir)

        for e in range(NUM_EPOCHS + 1):
            # generator
            # noise_vector = np.random.normal(size=(BATCH_SIZE, latent_dim))  # sample random points in latent space
            noise_vector = np.random.uniform(-1.0, 1.0, size=(BATCH_SIZE, latent_dim))
            gen_images = self.g.predict(noise_vector)

            # discriminator
            indices = np.random.randint(0, train_images.shape[0], size=BATCH_SIZE)
            real_images = train_images[indices]

            # concatenate real and gen images for discriminator
            d_images = np.concatenate((real_images, gen_images))

            # labels "real", "fake"
            real_labels = np.zeros((BATCH_SIZE, 1))
            fake_labels = np.ones((BATCH_SIZE, 1))

            # add random noise to labels
            real_labels += 0.05 * np.random.random(real_labels.shape)
            fake_labels += 0.05 * np.random.random(fake_labels.shape)

            # concatenate labels
            d_labels = np.concatenate((real_labels, fake_labels))

            # train discriminator
            # can concatenate images to simulate mixing
            #d.trainable = True
            # d_real_loss = d.train_on_batch(real_images, real_labels)
            # d_fake_loss = d.train_on_batch(gen_images, fake_labels)
            d_loss = self.d.train_on_batch(d_images, d_labels)

            # train generator, draw new random points in latent space
            # noise_vector = np.random.normal(size=(BATCH_SIZE, latent_dim))  # sample random points in latent space
            noise_vector = np.random.uniform(-1.0, 1.0, size=(BATCH_SIZE, latent_dim))

            # misleading label: "all these images are real" - obviously a lie
            misleading_labels = np.ones((BATCH_SIZE, 1))

            # train generator
            #d.trainable = False
            gan_loss = self.gan.train_on_batch(noise_vector, misleading_labels)  # discriminator weights are frozen

            if e % 200 == 0:
                # save model weights
                self.gan.save_weights(history_file)

                # Tensorboard

                # print metrics
                print("Epoch: {}".format(e))

            if e % 500 == 0:
                # save images
                # save a generated image
                image_file = dir + "\\" + str(e) + "_Gen" + ".png"
                img = image.array_to_img(gen_images[0] * 255., scale=False)
                img.save(image_file)

                # save a real image for comparison
                image_file = dir + "\\" + str(e) + "_Real" + ".png"
                img = image.array_to_img(real_images[0] * 255., scale=False)
                img.save(image_file)


##################################################################################################
if __name__ == "__main__":
    # load dataset
    (train_images, train_labels), (_, _) = mnist.load_data()

    # rescale -1 to 1 and normalize
    train_images = train_images.reshape(train_images.shape[0], num_rows, num_cols, num_channels)
    train_images = train_images.astype(np.float32)
    train_images = (train_images - 127.5) / 127.5

    gan = GAN()
    gan.train(train_images)
