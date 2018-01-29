import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
sequence_max_length = 1014

class model():
    """
    DenseNet Model for Text Classification.
    """

    def __init__(self, num_classes, cnn_filter_size=(3,3,3,3), pooling_filter_size=(2,2,2,2), num_filters_per_size=(64,128,256,512), num_rep_block=(4,4,4,4), num_quantized_chars=len(alphabet)):

	def denseBlock(input_layer, num_filters_per_size_i, cnn_filter_size_i, i, num_rep_block_i):
		with tf.variable_scope("dense_unit_%s" % i):
			nodes = []
			a = slim.conv2d(input_layer, num_filters_per_size_i, [1, cnn_filter_size_i], weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True), normalizer_fn=slim.batch_norm)
			nodes.append(a)
			print a.get_shape()
			for z in range(num_rep_block_i-1):
				#b = slim.conv2d(tf.concat(3,nodes), num_filters_per_size_i, [1, cnn_filter_size_i], weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True), normalizer_fn=slim.batch_norm)
				b = slim.conv2d(tf.concat(nodes,3), num_filters_per_size_i, [1, cnn_filter_size_i], weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True), normalizer_fn=slim.batch_norm)
				nodes.append(b)
				print b.get_shape()
			print ""
			return b

        self.input_x = tf.placeholder(tf.float32, [None, num_quantized_chars, sequence_max_length, 1], name="input_x")		
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
	self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

	# Input Dim : 70 x 176 x 1

        # ================ First Conv Layer ================
	h = slim.conv2d(self.input_x, num_filters_per_size[0], [num_quantized_chars, cnn_filter_size[0]], weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True), normalizer_fn=slim.batch_norm, scope = 'conv0', padding='VALID')

	# Output Dim : 1 x 176 x 64

        # ================ Conv Block 64, 128, 256, 512 =================
	for i in range(0,len(num_filters_per_size)):
		h = denseBlock(h, num_filters_per_size[i], cnn_filter_size[i], i, num_rep_block[i])

		# Transition Layer
		if i<>len(num_filters_per_size)-1:
			h = slim.conv2d(h, num_filters_per_size[i+1], [1, cnn_filter_size[i]], weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True), normalizer_fn=slim.batch_norm, scope='conv-last-%s' % i) 
		
		# Max pooling 1/2
		h = slim.max_pool2d(h, [1,pooling_filter_size[i]], stride=pooling_filter_size[i], scope='pool_%s' % i)
		print h.get_shape()

	# Output Dim : 1 x 8 x 512

        # ================ Layer FC ================
	# Output Dim : 1 x 4096

	# Max pooling filtersize = 8, stride = 8
	h = slim.max_pool2d(h, [1, 8], stride=8, scope='pool_final')
	h = slim.flatten(h)
	print h.get_shape()

	# Global avg max pooling
	#h = math_ops.reduce_mean(h, [1, 2], name='pool5', keep_dims=True)
	#h = slim.flatten(h)
	#print ""
	#print h.get_shape()

	h = slim.fully_connected(h, 2048, activation_fn=None, scope='FC1')
	print h.get_shape()
	h = slim.fully_connected(h, 2048, activation_fn=None, scope='FC2')
	print h.get_shape()
	scores = slim.fully_connected(h, num_classes, activation_fn=None, scope='output')
	print scores.get_shape()
        
	self.scores = scores
        pred1D = tf.argmax(scores, 1, name="predictions")
	self.predictions = pred1D
	y1D = tf.argmax(self.input_y, 1)       
	    
        # ================ Loss and Accuracy ================
        # CalculateMean cross-entropy loss
        with tf.name_scope("evaluate"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=scores)
            self.loss = tf.reduce_mean(losses)

            correct_predictions = tf.equal(pred1D, y1D)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
