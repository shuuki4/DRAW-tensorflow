"""
Author : Beomsu Kim
Date : Aug 14, 2016
"""

import tensorflow as tf
import numpy as np
import cPickle, gzip
import os

class DRAW(object) :
	# class that creates DRAW tensorflow computation graph, and train / perform basic experiments
	# Model : DRAW : A Recurrent Neural Network For Image Generation, arXiv : 1502.04623 [cs.CV], May 2015
	# Experiment Data : binarized MNIST

	def single_linear(self, shape, input_tensor, time, scope_name="linear") :
		# generates single linear transformation layer
		# parameter : 
		#	shape : shape of W. [input_dim, output_dim]
		# 	input_tensor : input tensor, should be [None, input_dim]
		#	time : current timestamp. to determine to reuse variable or not.
		#	scope_name : optional. name of the variable scope.
		# returns :
		#	transformed result tensor

		thres = np.sqrt(6.0 / (shape[0] + shape[1]))
		with tf.variable_scope(scope_name) :
			if time > 1 : tf.get_variable_scope().reuse_variables()
			weights = tf.get_variable("weights", shape, initializer = tf.random_uniform_initializer(minval=-thres, maxval=thres))
			biases = tf.get_variable("biases", [shape[1]], initializer = tf.constant_initializer(0.0))
		return tf.matmul(input_tensor, weights)+biases

	def attention_extract(self, time, layer_type) :
		# extracts gaussian filter parameters and filterbanks from the h_dec layer 
		# parameters :
		# 	layer_type : string, "read" or "write".
		# returns :
		#	tuple, (gamma_value tensor, filterbank_x, filterbank_y)

		assert (layer_type in ["read", "write"]), "Layer type should be 'read' or 'write'!"

		if layer_type == "read" : h_dec = self.h_dec_list[time-1]
		elif layer_type == "write" : h_dec = self.h_dec_list[time]	

		g_x_tilde = self.single_linear([self.dec_size, 1], h_dec, time, scope_name=layer_type+"_g_x")
		g_x = 0.5 * (self.image_shape[1] + 1.0) * (g_x_tilde + 1.0)
		g_y_tilde = self.single_linear([self.dec_size, 1], h_dec, time, scope_name=layer_type+"_g_y")
		g_y = 0.5 * (self.image_shape[0] + 1.0) * (g_y_tilde + 1.0)

		sigma_square = tf.reshape(tf.exp(self.single_linear([self.dec_size, 1], h_dec, time, scope_name=layer_type+"_sigma")), [self.mini_batch_size, 1, 1])
		delta_tilde = tf.exp(self.single_linear([self.dec_size, 1], h_dec, time, scope_name=layer_type+"_delta"))
		delta = ((max(self.image_shape[0], self.image_shape[1])-1.0) / (self.N - 1.0)) * delta_tilde
		gamma = tf.exp(self.single_linear([self.dec_size, 1], h_dec, time, scope_name=layer_type+"_gamma"))

		delta_weights = tf.constant([(float(x)-self.N/2.0-0.5) for x in range(1, self.N+1)], dtype=tf.float32)
		mu_x = tf.reshape(g_x + tf.mul(delta, delta_weights), [self.mini_batch_size, self.N, 1]) # (mini_batch_size * N * 1)
		mu_y = tf.reshape(g_y + tf.mul(delta, delta_weights), [self.mini_batch_size, self.N, 1]) 

		if layer_type == "write" :
			self.gx_list[time] = tf.squeeze(g_x)
			self.gy_list[time] = tf.squeeze(g_y)
			self.delta_list[time] = tf.squeeze(delta)
			self.sigma_sq_list[time] = tf.squeeze(sigma_square)

		a = tf.reshape(tf.constant(range(1, self.image_shape[1]+1), dtype=tf.float32), [1, 1, self.image_shape[1]])
		b = tf.reshape(tf.constant(range(1, self.image_shape[0]+1), dtype=tf.float32), [1, 1, self.image_shape[1]])
		eps = 1e-10
		filter_x = tf.exp(-tf.square(a - mu_x) / (2 * sigma_square)) + eps # (mini_batch_size * N * image_shape[1])
		filter_y = tf.exp(-tf.square(b - mu_y) / (2 * sigma_square)) + eps

		filter_x = filter_x / tf.reduce_sum(filter_x, 2, keep_dims=True)
		filter_y = filter_y / tf.reduce_sum(filter_y, 2, keep_dims=True)

		return (gamma, filter_x, filter_y)

	def read_layer(self, x, time, attention = True) :
		# generates 'read' layer, shape depending on the 'attention' parameter  
		# parameter :
		#	x : image input, [None, height, width, channel]
		#	time : current timestep
		# 	attention : attention ? look full image : look only attentioned part of image
		# returns :
		#	result tensor of read layer

		x_reshape = tf.reshape(x, [self.mini_batch_size, self.input_dim])
		prev_c = self.canvas_list[time-1]
		prev_c_reshape = tf.reshape(prev_c, [self.mini_batch_size, self.input_dim])
		hat_x = x - tf.nn.sigmoid(prev_c)
		hat_x_reshape = tf.reshape(hat_x, [self.mini_batch_size, self.input_dim])

		if attention :
			gamma, filter_x, filter_y = self.attention_extract(time, "read")
			x_div_filter = [tf.squeeze(tensor) for tensor in tf.split(3, self.image_shape[2], x)]
			hat_x_div_filter = [tf.squeeze(tensor) for tensor in tf.split(3, self.image_shape[2], hat_x)]
			filtered_tensors = [tf.batch_matmul(tf.batch_matmul(filter_y, tensor), tf.transpose(filter_x, perm=[0, 2, 1])) \
								for tensor in (x_div_filter + hat_x_div_filter)]
			gamma_mul_tensors = [(tf.reshape(tensor, [self.mini_batch_size, -1]) * gamma) for tensor in filtered_tensors]
			return tf.concat(1, gamma_mul_tensors) 

		else : return tf.concat(1, [x_reshape, hat_x_reshape])

	def encoder(self, input_tensor, time) :
		# generates 'encoder' LSTM layer's computation nodes 
		# return : lstm's output, h_enc^time 
		# parameter :
		#	input_tensor : [None, encoder_input_dim]
		#	time : timestamp, 1 to self.max_time
		# returns :
		#	result tensor of encoder lstm, only output (h_enc)

		with tf.variable_scope("encoder_lstm") :
			if time>1 : tf.get_variable_scope().reuse_variables()
			output, self.enc_lstm_state = self.enc_lstm(input_tensor, self.enc_lstm_state)	
			return output

	def decoder(self, input_tensor, time) :
		# generates 'decoder' LSTM layer's computation nodes
		# return : lstm's output, h_dec^time
		# parameter :
		# 	input_tensor : [None, decoder_input_dim(=latent_dim)]
		#	time : timestamp
		# returns :
		#	result tensor of decoder lstm, only output (h_dec)

		with tf.variable_scope("decoder_lstm") :
			if time>1 : tf.get_variable_scope().reuse_variables()
			output, self.dec_lstm_state = self.dec_lstm(input_tensor, self.dec_lstm_state)
			return output

	def write_layer(self, input_tensor, time, attention = True) :
		# generates 'write' layer, shape depending on the 'attention' parameter
		# parameter :
		#	input_tensor : input tensor
		#	time : current timestamp
		#	attention : boolean for attention
		# returns :
		#	result tensor of write layer

		if attention :
			out_shape = [self.mini_batch_size, self.N, self.N, self.image_shape[2]]
			out_dim = out_shape[1] * out_shape[2] * out_shape[3]
		else :
			out_shape = self.batch_image_shape
			out_dim = self.input_dim

		w_t = tf.reshape(self.single_linear([self.dec_size, out_dim], input_tensor, time, scope_name="write"), out_shape)

		if attention : 
			gamma, filter_x, filter_y = self.attention_extract(time, "write")
			gamma = tf.reshape(tf.inv(gamma), [self.mini_batch_size, 1, 1])

			filtered_tensors = [tf.batch_matmul(tf.batch_matmul(tf.transpose(filter_y, perm=[0, 2, 1]), tf.squeeze(tensor)), filter_x) \
								for tensor in tf.split(3, self.image_shape[2], w_t)]
			gamma_mul_tensors = [tf.reshape(tensor * gamma, [self.mini_batch_size, self.image_shape[0], self.image_shape[1], 1]) for tensor in filtered_tensors]
			return tf.concat(3, gamma_mul_tensors) 
		else : 
			return w_t

	def step_path(self, path) :
		return 'saved_step_'+str(abs(hash(path)%172062407))

	def __init__(self, image_shape, is_training, model_path = None, attention = True, max_time = 10, filter_size = 5, \
					batch_size = 256, hidden_dim = 300, latent_dim = 10) :
		# parameter :
		#	image_shape : shape of image, list of [height, width, channel]
		#	is_training : true if training, false if test or experiment
		#	model_path : path to save/load model. default path : '/model/model.ckpt'
		#	attention : boolean parameter to make model with attention or not
		#	max_time : variable T, maximum time
		#	filter_size : length of the gaussian filter grid. (filter_size * filter_size) shape of gaussian filters are generated.
		#	batch_size : size of the batch while training & generating image
		#	hidden_dim : dimension of hidden layers (output layer of encoding & decoding lstms)
		#	latent_dim : dimension of the latent space

		assert (len(image_shape)==3 and (isinstance(image_shape, list) or isinstance(image_shape, tuple))), \
			"Parameter 'image_shape' should be a list/tuple with three elements! (height, width, channel)"
		tf.reset_default_graph()

		# class parameters
		self.model_path = model_path
		self.is_training = is_training

		# model parameters
		self.mini_batch_size = batch_size
		self.max_time = max_time # max time sequence, T
		self.enc_size = hidden_dim
		self.dec_size = hidden_dim
		self.image_shape = image_shape # (height, width, channel)
		self.batch_image_shape = [self.mini_batch_size] + list(image_shape)
		self.input_dim = image_shape[0] * image_shape[1] * image_shape[2]
		self.latent_dim = latent_dim # dimension for latent vector (z)
		self.attention = attention
		self.N = filter_size # grid length for gaussian filter
		self.learning_rate = 0.003
		self.model_path = model_path

		# state variables & lstm cells
		self.canvas_0 = tf.Variable(tf.random_normal(self.image_shape), name="canvas_0")
		self.h_enc_0 = tf.Variable(tf.random_normal([self.enc_size]), name="h_enc_0")
		self.h_dec_0 = tf.Variable(tf.random_normal([self.dec_size]), name="h_dec_0")
		self.canvas_list = [tf.tile(tf.reshape(self.canvas_0, [1]+self.image_shape), [self.mini_batch_size, 1, 1, 1])] + ([None]*(self.max_time))
		self.h_enc_list = [tf.tile(tf.reshape(self.h_enc_0, [1, self.enc_size]), [self.mini_batch_size, 1])] + ([None]*(self.max_time))
		self.h_dec_list = [tf.tile(tf.reshape(self.h_dec_0, [1, self.dec_size]), [self.mini_batch_size, 1])] + ([None]*(self.max_time))
		self.enc_lstm = tf.nn.rnn_cell.LSTMCell(self.enc_size)
		self.dec_lstm = tf.nn.rnn_cell.LSTMCell(self.dec_size)
		self.enc_lstm_state = self.enc_lstm.zero_state(self.mini_batch_size, tf.float32)
		self.dec_lstm_state = self.dec_lstm.zero_state(self.mini_batch_size, tf.float32)
		self.gx_list = [None]*(self.max_time+1)
		self.gy_list = [None]*(self.max_time+1)
		self.delta_list = [None]*(self.max_time+1)
		self.sigma_sq_list = [None]*(self.max_time+1)

		# build network

		self.x_input = tf.placeholder(tf.float32, shape=self.batch_image_shape)	
		mu_square_accum = tf.zeros([])
		sigma_square_accum = tf.zeros([])
		log_sigma_square_accum = tf.zeros([])

		for time in xrange(1, self.max_time+1) :
			
			with tf.name_scope('encoding') :
				read_out = self.read_layer(self.x_input, time, self.attention)
				enc_out = self.h_enc_list[time] = self.encoder(read_out, time)
			
			with tf.name_scope('sample') :
				# sample : just sample one example for DRAW Network
				
				noise = tf.random_normal([self.mini_batch_size, self.latent_dim])
				if is_training : # while training, generate z_mu and z_sigma from the output of encoder, and sample one z
					z_mu = self.single_linear([self.enc_size, self.latent_dim], enc_out, time, scope_name="z_mu")
					z_sigma = tf.exp(self.single_linear([self.enc_size, self.latent_dim], enc_out, time, scope_name="z_sigma"))
				else : # while test, just generate z by normal distribution ~ N(0, 1).	
					z_mu = tf.zeros([self.mini_batch_size, self.latent_dim])
					z_sigma = tf.ones([self.mini_batch_size, self.latent_dim])
				sample_out = z_mu + tf.mul(z_sigma, noise)

				# accumulate squares for future loss calculation
				mu_square_accum += tf.reduce_mean(tf.reduce_sum(tf.square(z_mu), 1))
				sigma_square = tf.square(z_sigma)
				sigma_square_accum += tf.reduce_mean(tf.reduce_sum(sigma_square, 1))
				log_sigma_square_accum += tf.reduce_mean(tf.reduce_sum(tf.log(sigma_square), 1))

			with tf.name_scope('decoding') :
				dec_out = self.h_dec_list[time] = self.decoder(sample_out, time)
				write_out = self.write_layer(dec_out, time, self.attention)
				self.canvas_list[time] = self.canvas_list[time-1] + write_out

		# results, losses and optimizer
		with tf.name_scope('results') :
			self.all_images = tf.concat(0, [tf.nn.sigmoid(tf.reshape(canv, [1]+self.batch_image_shape)) for canv in self.canvas_list])
			self.final_image = tf.nn.sigmoid(self.canvas_list[self.max_time])
			cross_entropy = tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(self.canvas_list[self.max_time], self.x_input, name="loss_x"), [self.mini_batch_size, self.input_dim])
			self.loss_x = tf.reduce_mean(tf.reduce_sum(cross_entropy, 1))
			self.loss_z = 0.5 * (mu_square_accum + sigma_square_accum - log_sigma_square_accum - self.max_time)
			self.total_loss = self.loss_x + self.loss_z
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

			if self.attention : 
				# gradient clip, for attention training
				gvs = self.optimizer.compute_gradients(self.total_loss)
				clip_gvs = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gvs if grad is not None]
				self.optimizer = self.optimizer.apply_gradients(clip_gvs)
			else :
				self.optimizer = self.optimizer.minimize(self.total_loss)

		# summary writer, write just if training
		if is_training :
			with tf.name_scope('summary') :
				tf.scalar_summary('loss_x', self.loss_x)
				tf.scalar_summary('loss_z', self.loss_z)
				tf.scalar_summary('total_loss', self.total_loss)
				tf.histogram_summary('final_image', self.final_image)
				self.merged_summary = tf.merge_all_summaries()

		# if class is instantiated for image generation, try to load model here
		if not is_training :
			self.sess = tf.Session()
			saver = tf.train.Saver()
			self.load_model(saver, self.sess)

	def load_model(self, saver, sess) :
		# helper function to load model

		model_restore_success = False
		if (self.model_path is not None or self.is_training is False) :
			if self.model_path is None : model_path = "/model/model.ckpt" # default path
			else : model_path = self.model_path
			try :
				saver.restore(sess, model_path)
				model_restore_success = True
				print "Model Successfully Restored."
			except ValueError as e:
				if self.is_training : print "Invalid model path : Train from scratch"
				else :
					print "Model load failed : " + e.message
					raise

			if model_restore_success and self.is_training :
				with open(self.step_path(self.model_path), 'rb') as f :
					step = cPickle.load(f)

		if not model_restore_success :
			sess.run(tf.initialize_all_variables())

	def save_model(self, step, saver, sess) :
		# helper function to save model

		if self.model_path is None :
			my_path = "/model/model.ckpt" # default path
			# try to make directory
			if not os.path.exists("/model") :
				try : os.makedirs("/model")
				except OSError as e :
					if e.errno != errno.EEXIST : raise
		else : my_path = self.model_path

		saver.save(sess, my_path)
		with open(self.step_path(my_path), 'wb') as f :
			cPickle.dump(step, f)

	def train(self, train_data, valid_data, max_epoch = 60) :
		# train the model by given train data & valid data.
		# model first tries to load data from the given model path. if fails, just initiates new training from the scratch
		# parameter :
		#	train_data : numpy array, with (#_data, height, width, #_channel)
		#	valid_data : numpy array, same format with train_data
		#	max_epoch : optional. changes the max epoch of training session

		assert (self.is_training), "To train, class should be created with 'is_training = True'!"
		assert (train_data.shape[1:] == valid_data.shape[1:] == tuple(self.image_shape)), "The shape of train data & validation data should be equal with 'image_shape'!"

		step = 0

		with tf.Session() as sess :
			saver = tf.train.Saver()
			summary_writer = tf.train.SummaryWriter("/att_summary", sess.graph)
			self.load_model(saver, sess)

			print "Train Start!"
			for epoch in xrange(1, max_epoch+1) :
				# train
				np.random.shuffle(train_data)
				for i in xrange(0, train_data.shape[0], self.mini_batch_size) :
					#if (i==0 or int(i/10000) > int((i-self.mini_batch_size)/10000)) : print "Epoch %d, %d" % (epoch, i) # for short-term logging
					if i+self.mini_batch_size > train_data.shape[0] : break 
					current_input = train_data[i:i+self.mini_batch_size]
					_, summary = sess.run([self.optimizer, self.merged_summary], feed_dict = {self.x_input : current_input})
					summary_writer.add_summary(summary, step)
					step += 1

				# validation, per 5 epoch
				if epoch % 5 == 0 :
					lx_sum = lz_sum = l_sum = count = 0.0
					for i in xrange(0, valid_data.shape[0], self.mini_batch_size) :
						if i + self.mini_batch_size > valid_data.shape[0] : break
						current_input = valid_data[i:i+self.mini_batch_size]
						lx_val, lz_val, l_val = sess.run([self.loss_x, self.loss_z, self.total_loss], feed_dict = {self.x_input : current_input})
						lx_sum += lx_val; lz_sum += lz_val; l_sum += l_val
						count += 1.0

					lx_sum = lx_sum / count ; lz_sum = lz_sum / count; l_sum = l_sum / count
					summary = tf.Summary(value=[
						tf.Summary.Value(tag="val_loss_x", simple_value = lx_sum),
						tf.Summary.Value(tag="val_loss_z", simple_value = lz_sum),
						tf.Summary.Value(tag="val_total_loss", simple_value = l_sum)
					])
					summary_writer.add_summary(summary, step)
					print "Validation, Epoch %d / Loss_x : %f, Loss_z : %f, Loss : %f" % (epoch, lx_sum, lz_sum, l_sum)

				# save model, per 20 epoch and when train stops
				if epoch % 20 == 0 or epoch == max_epoch:
					self.save_model(step, saver, sess)

	def generate(self) :
		# returns :
		#	batch of randomly generated images (numpy array), through time
		#	[time, batch, image_height, image_width, channel] 

		assert (self.is_training is False), "To generate images, 'is_training' should be false!"
		return self.sess.run([self.all_images])[0]

	def generate_attend(self) :
		# returns :
		#	batch of randomly generated images through time, [time, batch, image_height, image_width, channel]
		#	and the list of (write) gaussian filters through time. Careful : filter starts from timestamp 1.
		#	filter : [()]

		assert (self.is_training is False), "To generate images, 'is_training' should be false!"
		assert (self.attention), "To generate filters, model should be trained with 'attention=True'!"

		results = self.sess.run([self.all_images] + self.gx_list[1:] + self.gy_list[1:] + self.delta_list[1:] + self.sigma_sq_list[1:])
		images = results[0]
		filters = []
		for i in range(self.max_time) :
			filters.append((results[1+i], results[1+self.max_time+i], results[1+self.max_time*2+i], results[1+self.max_time*3+i]))			

		return images, filters

def binarize(load_results) :
	return tuple((np.where(d_x>0.5, 1.0, 0.0), d_y) for (d_x, d_y) in load_results)

if __name__ == "__main__" :
	
	# training process for MNIST
	mnist_data_path = "mnist.pkl.gz"
	with gzip.open(mnist_data_path, "rb") as f :
		train_set, valid_set, _temp = binarize(cPickle.load(f))	# binarized MNIST
		del _temp												# in this network, just use train set & validation set!
		train_set = train_set[0]; valid_set = valid_set[0]
		train_set = np.reshape(train_set, [train_set.shape[0], 28, 28, 1])
		valid_set = np.reshape(valid_set, [valid_set.shape[0], 28, 28, 1])

	my_DRAW = DRAW(image_shape=[28, 28, 1], is_training=True, model_path="/model/attention_model.ckpt")
	my_DRAW.train(train_set, valid_set, max_epoch = 100)
	
	# image generation process
	my_DRAW = DRAW(image_shape=[28, 28, 1], is_training=False, model_path="/model/attention_model.ckpt")
	generated_images, generated_filters = my_DRAW.generate_attend()
	with open('generated_images_filters.npy', 'wb') as f:
		cPickle.dump((generated_images, generated_filters), f)
