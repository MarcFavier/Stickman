from blocks import residual_block, inverted_residual_block, conv_block, up_sample

import tensorflow as tf
import sys
import numpy as np

def create_vgg(x):
	print("x en entrée", x.shape)
	x = tf.keras.layers.Conv2D(16,
		kernel_size=7,
		strides=4,
		use_bias=False,
		padding='same',
		kernel_initializer='he_normal',
		kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Conv2D(32,
		kernel_size=3,
		strides=2,
		use_bias=False,
		padding='same',
		kernel_initializer='he_normal',
		kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	x = tf.keras.layers.Conv2D(32,
		kernel_size=3,
		strides=1,
		use_bias=False,
		padding='same',
		kernel_initializer='he_normal',
		kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	print("x en sortie", x.shape)
	return x

def create_mobile(x):
	#input size (None,196,196,3)
	x = conv_block(x, num_filters = 16, kernel_size=3, strides=2, activation='relu')
	#x = inverted_residual_block(x, stride = 1,  num_filters = 17, expansion = 1)
	#print("x apres residual",x.shape)
	x = inverted_residual_block(x, stride = 2,  num_filters = 16, expansion = 6)
	x = inverted_residual_block(x, stride = 2,  num_filters = 24, expansion = 6)
	#output size (None, 25, 25,32)
	return x


possible_backbones = {
	'VGG':create_vgg,
	'MOBILE':create_mobile
}
