import tensorflow as tf
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from generator import generator
from discriminator import discriminator
import datetime

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

tf.reset_default_graph() 

noise_size = 32
batch_size = 65
training_epoch = 75
#pretrain_epoch = 2
keep_prob = 0.5
input_generator = tf.placeholder(dtype = tf.float32, shape = [None, noise_size], name = 'input_generator')
input_discriminator = tf.placeholder(dtype = tf.float32, shape = [None, 28,28,1], name = 'input_discriminator')
discriminatorReal = discriminator(input_discriminator, keep_prob = keep_prob, reuse = False)
generatorFake = generator(input_generator, batch_size, keep_prob = keep_prob, reuse = False)
discriminatorFake = discriminator(generatorFake, keep_prob = keep_prob, reuse = True)

discriminator_Real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminatorReal, labels = tf.ones_like(discriminatorReal)))
discriminator_Fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminatorFake, labels = tf.zeros_like(discriminatorFake)))
discriminator_Total_loss = discriminator_Real_loss + discriminator_Fake_loss

generator_Fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminatorFake, labels = tf.ones_like(discriminatorFake)))

trainable_variables = tf.trainable_variables()

discriminator_variables = []
generator_variables = []

for var in trainable_variables:
	if('generator_' in var.name):
		generator_variables.append(var)
	if('discriminator_' in var.name):
		discriminator_variables.append(var)


discriminator_Train = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss = discriminator_Total_loss, var_list = discriminator_variables)
generator_Train = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss = generator_Fake_loss, var_list = generator_variables)

sess = tf.Session()


tf.summary.scalar(name = 'Discriminator real loss', tensor = discriminator_Real_loss)
tf.summary.scalar(name = 'Discriminator fake loss', tensor = discriminator_Fake_loss)
tf.summary.scalar(name = 'Discriminator loss', tensor = discriminator_Total_loss)
tf.summary.scalar(name = 'Generator loss', tensor = generator_Fake_loss)
tensorboard_generated_images = generator(input_generator, batch_size, keep_prob = keep_prob, reuse = True)
tf.summary.image(name = 'Generated images', tensor = tensorboard_generated_images, max_outputs = 5)
merged_data = tf.summary.merge_all()
logdir = 'tensorboard_mnist/'+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+'/'
writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())


'''
for i in range(pretrain_epoch*(mnist.train.num_examples//batch_size)):
	input_fake = np.random.normal(0,1,size = [batch_size,noise_size])
	input_real = mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
	discriminatorTrain, discriminatorRealLoss, discriminatorFakeLoss = sess.run([discriminator_Train,discriminator_Real_loss, discriminator_Fake_loss], feed_dict = {input_discriminator:input_real, input_generator:input_fake})
'''	

for i in range(training_epoch*(mnist.train.num_examples//batch_size)):
	input_fake = np.random.normal(0,1,size = [batch_size,noise_size])
	input_real = mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
	discriminatorTrain, discriminatorRealLoss, discriminatorFakeLoss = sess.run([discriminator_Train,discriminator_Real_loss, discriminator_Fake_loss], feed_dict = {input_discriminator:input_real, input_generator:input_fake})
		
	input_fake = np.random.normal(0,1,size = [batch_size,noise_size])
	generatorTrain = sess.run(generator_Train, feed_dict = {input_generator:input_fake})
		
	if(i%100 == 0):
		input_fake = np.random.normal(0,1,size = [batch_size,noise_size])
		summary = sess.run(merged_data, feed_dict = {input_discriminator:input_real, input_generator:input_fake})
		writer.add_summary(summary = summary,global_step = i)

	