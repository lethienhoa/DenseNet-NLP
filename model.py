import tensorflow as tf

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
sequence_max_length = 144 # Twitter has only 140 characters. We pad 4 blanks characters more to the right of tweets to be conformed with the architecture of A. Conneau et al (2016)
top_k_max_pooling_size = 8
epsilon = 1e-3 # Batch Norm

class VDCNN():
    """
    A Very Deep CNN
    based on the Very Deep Convolutional Networks for Natural Language Processing paper.
    """
    def __init__(self, num_classes=2, cnn_filter_size=3, pooling_filter_size=2, num_filters_per_size=(64,128,256,512), 
			num_rep_block=(10,10,4,4), num_quantized_chars=70, sequence_max_length=144, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.float32, [None, num_quantized_chars, sequence_max_length, 1], name="input_x")		
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.is_training = tf.placeholder(tf.bool, name="phase")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

	# Input Dim : 70 x 144 x 1

        # ================ First Conv Layer ================
        with tf.name_scope("first-conv-layer"):
            filter_shape = [num_quantized_chars, cnn_filter_size, 1, num_filters_per_size[0]]	
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 1, 1, 1], padding="SAME", name="first-conv")	
            h = tf.nn.bias_add(conv, b)

	# Output Dim : 1 x 144 x 64

        # ================ Conv Block 64, 128, 256, 512 ================
	# Output Dim of Conv Block 64 : 1 x 72 x 64
	# Output Dim of Conv Block 128: 1 x 36 x 128
	# Output Dim of Conv Block 256: 1 x 18 x 256       
	# Output Dim of Conv Block 512: 1 x 18 x 512
	for i in range(0,4):
		with tf.name_scope("conv-block"):
		    filter_shape = [1, cnn_filter_size, num_filters_per_size[i], num_filters_per_size[i]]	

		    # ================ Internal Loop of each Conv Block 64, 128, 256, 512 ================
		    for j in range(0,num_rep_block[i]):
			with tf.name_scope("sub1"):
			    W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W1")
			    conv1 = tf.nn.conv2d(h, W1, strides=[1, 1, 1, 1], padding="SAME", name="conv1")	
			    # bias is replaced by batch_norm (same effect as beta in BN)
			    batch_norm1 = tf.contrib.layers.batch_norm(conv1                                          
									  center=True, scale=True, decay=0.9, 
									  is_training=self.is_training,	
						# When we want to evaluate the model, set to False to use moving_mean, moving_average
									  name='bn1')	
			    h1 = tf.nn.relu(batch_norm1, name="relu1")
			with tf.name_scope("sub2"):
			    W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W2")
			    conv2 = tf.nn.conv2d(h1, W2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")	
			    batch_norm2 = tf.contrib.layers.batch_norm(conv2                                          
									  center=True, scale=True, decay=0.9, 
									  is_training=self.is_training, name='bn2')
			    h = tf.nn.relu(batch_norm2, name="relu2")
		if (i<>3):	# don't do max pooling at the last conv block
		    with tf.name_scope("max-pooling"):
			h = tf.nn.max_pool(h, ksize=[1, 1, pooling_filter_size, 1],
		        		      strides=[1, 1, 2, 1], padding='VALID', name="pool")

	# ================ Top k-max pooling ================

	top_k_max_pooling = tf.nn.top_k(tf.transpose(h), k=top_k_max_pooling_size)
	top_k_max_pooling = tf.transpose(top_k_max_pooling)
	# Output Dim : 1 x 8 x 512

        # ================ Layer FC 1 ================
        num_features_total = top_k_max_pooling_size * num_filters_per_size[3]				
        h_pool_flat = tf.reshape(top_k_max_pooling, [-1, num_features_total])			# num_features_total = 8 * 512 = 4096
	# Output Dim : 1 x 4096

        # Fully connected layer 1
        with tf.name_scope("fc-1"):
            W = tf.Variable(tf.truncated_normal([num_features_total, 4096], stddev=0.05), name="W")
            # W = tf.get_variable("W", shape=[num_features_total, 1024],
            #                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[4096]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

            fc_1_output = tf.nn.relu(tf.nn.xw_plus_b(h_pool_flat, W, b), name="fc-1-out")

	# Output Dim : 1 x 4096

        # ================ Layer FC 2 ================

        # Fully connected layer 2
        with tf.name_scope("fc-2"):
            W = tf.Variable(tf.truncated_normal([4096, 2048], stddev=0.05), name="W")
            # W = tf.get_variable("W", shape=[1024, 1024],
            #                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2048]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

            fc_2_output = tf.nn.relu(tf.nn.xw_plus_b(fc_1_output, W, b), name="fc-2-out")

	# Output Dim : 1 x 2048

        # ================ Layer FC 3 ================
        # Fully connected layer 3
        with tf.name_scope("fc-3"):
            W = tf.Variable(tf.truncated_normal([2048, num_classes], stddev=0.05), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)

	    # Output Dim : 1 x num_classes

            scores = tf.nn.xw_plus_b(fc_2_output, W, b, name="output")
            predictions = tf.argmax(scores, 1, name="predictions")
        # ================ Loss and Accuracy ================
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
