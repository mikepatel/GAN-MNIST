"""
Michael Patel
June 2019

Python 3.6.5
TF 1.12.0

File description:
    For constants and model hyperparameters

"""
################################################################################
# based on MNIST images: 28x28 greyscale
IMAGE_ROWS = 28
IMAGE_COLS = 28
IMAGE_CHANNELS = 1

# Training
NUM_EPOCHS = 3000  # 100000
BATCH_SIZE = 128
BUFFER_SIZE = 60000  # size of training set
NUM_GEN = 10
G_LEARNING_RATE = 0.0001
D_LEARNING_RATE = 0.0001
BETA_1 = 0.5

# Model
Z_DIM = 100
DROPOUT_RATE = 0.5
LEAKY_ALPHA = 0.2
