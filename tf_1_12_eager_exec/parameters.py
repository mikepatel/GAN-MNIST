"""
Michael Patel
August 2019

Python 3.6.5
TensorFlow 1.12.0

Project description:
   To conduct GAN experimentation for learning purposes.
   Based on the DCGAN paper: https://arxiv.org/pdf/1511.06434.pdf

File description:
    To hold constants and model hyperparameters

Dataset: MNIST handwritten digits

"""
################################################################################
# based on MNIST images: 28x28 greyscale
IMAGE_ROWS = 28
IMAGE_COLS = 28
IMAGE_CHANNELS = 1

# Training
NUM_EPOCHS = 250
BATCH_SIZE = 64
BUFFER_SIZE = 60000  # size of training set
NUM_GEN_IMAGES = 16
G_LEARNING_RATE = 0.0001
D_LEARNING_RATE = 0.0001
BETA_1 = 0.5

# Model
NOISE_DIM = 100
DROPOUT_RATE = 0.3
LEAKY_ALPHA = 0.3

