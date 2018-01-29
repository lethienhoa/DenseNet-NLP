import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops
from tensorflow.contrib import layers as layers_lib

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
sequence_max_length = 260

class VDCNN():
    """
    A Very Deep CNN
    based on the Very Deep Convolutional Networks for Natural Language Processing paper.
    """

    def __init__(self, num_classes=2, cnn_filter_size=3, pooling_filter_size=2, num_filters_per_size=(64,128,256,512), 
			num_rep_block=(16,16,16,6), num_quantized_chars=len(alphabet), l2_reg_weight_decay=0.0001):

	def highwayUnit(input_layer, num_filters_per_size_i, cnn_filter_size, i, j):
		with tf.variable_scope("highway_unit_" + str(i) + "_" + str(j)):
			H = slim.conv2d(input_layer, num_filters_per_size_i, [1, cnn_filter_size])
			T = slim.conv2d(input_layer, num_filters_per_size_i, [1, cnn_filter_size], 
					biases_initializer=tf.constant_initializer(-1.0), activation_fn=tf.nn.sigmoid)
					#We initialize with a negative bias to push the network to use the skip connection
			output = H*T + input_layer*(1.0-T)
			return output

	def resUnit(input_layer, num_filters_per_size_i, cnn_filter_size, i, j):
		print input_layer.get_shape()
		with tf.variable_scope("res_unit_" + str(i) + "_" + str(j)):
			part1 = slim.batch_norm(input_layer, activation_fn=None)
			part2 = tf.nn.relu(part1)
			part3 = slim.conv2d(part2, num_filters_per_size_i, [1, cnn_filter_size], activation_fn=None)
			print part3.get_shape()
			part4 = slim.batch_norm(part3, activation_fn=None)
			part5 = tf.nn.relu(part4)
			part6 = slim.conv2d(part5, num_filters_per_size_i, [1,cnn_filter_size], activation_fn=None)
			print part6.get_shape()
			print ""
			output = input_layer + part6
			return output

        self.input_x = tf.placeholder(tf.float32, [None, num_quantized_chars, sequence_max_length, 1], name="input_x")		
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
	self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

	# Input Dim : 70 x 176 x 1

        # ================ First Conv Layer ================
	h = slim.conv2d(self.input_x, num_filters_per_size[0], [num_quantized_chars, cnn_filter_size], normalizer_fn=slim.batch_norm, scope = 'conv0', padding='VALID')

	# Output Dim : 1 x 176 x 64

        # ================ Conv Block 64, 128, 256, 512 ================
	for i in range(0,len(num_filters_per_size)):
		for j in range(0,num_rep_block[i]):
			h = resUnit(h, num_filters_per_size[i], cnn_filter_size, i, j)
			#h = highwayUnit(h, num_filters_per_size[i], cnn_filter_size, i, j)
		h = slim.max_pool2d(h, [1,pooling_filter_size], scope='pool_%s' % i)

	print h.get_shape()

        # ================ Layer FC ================
	# Global avg max pooling
	h = math_ops.reduce_mean(h, [1, 2], name='pool5', keep_dims=True)
	print h.get_shape()

	# Conv
	h = slim.conv2d(h, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='output')

	# FC & Dropout
	scores = slim.flatten(h)
	print scores.get_shape()

	self.scores = scores
        pred1D = tf.argmax(scores, 1, name="predictions")
	self.predictions = pred1D
	y1D = tf.argmax(self.input_y, 1)       
	    
        # ================ Loss and Accuracy ================
        # CalculateMean cross-entropy loss
        with tf.name_scope("evaluate"):
            losses = tf.nn.softmax_cross_entropy_with_logits(scores, self.input_y)
            self.loss = tf.reduce_mean(losses)

            correct_predictions = tf.equal(pred1D, y1D)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

	    zeros_like = tf.zeros_like(y1D)
	    ones_like = tf.ones_like(y1D)

	    PP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(pred1D, zeros_like), 
						tf.equal(y1D, zeros_like)), 'float'))
	    NN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(pred1D, ones_like), 
						tf.equal(y1D, ones_like)), 'float'))
	    PN = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(pred1D, zeros_like), 
						tf.equal(y1D, ones_like)), 'float'))
	    NP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(pred1D, ones_like), 
						tf.equal(y1D, zeros_like)), 'float'))

	    self.PP = PP
	    self.PN = PN
	    self.NP = NP
	    self.NN = NN
