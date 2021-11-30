import tensorflow as tf
from blocks import residual_block, inverted_residual_block, conv_block, up_sample


def create_scalar_heads(x, num_key_points, scale):
	y = tf.keras.layers.Flatten()(x)
	outputs = scale * tf.keras.layers.Dense(
		2*num_key_points,
		activation='sigmoid',
		use_bias=False,
		kernel_initializer='he_normal')(y)
	return outputs

def create_mobile_heads(x, num_key_points, scale):
	print("On entre dans la tete")
	x = inverted_residual_block(x, stride = 2,  num_filters = 32, expansion = 6)
	x = inverted_residual_block(x, stride = 2,  num_filters = 64, expansion = 6)
	x = conv_block(x, num_filters = 256, kernel_size=1, strides=1, activation='relu')
	x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
	x = scale*conv_block(x, num_filters = 2*num_key_points, kernel_size=1, strides=1, activation='sigmoid')
	x = tf.keras.layers.Flatten()(x)

	return x


possible_heads = {
	'scalar' : create_scalar_heads,
	'mob_heads' : create_mobile_heads,
}