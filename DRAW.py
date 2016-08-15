"""
Author : Beomsu Kim
Date : Aug 14, 2016
"""

import tensorflow as tf
import numpy as np
import cPickle, gzip
import os

from tf.nn.rnn_cell import LSTMCell, DropoutWrapper

class DRAW(object) :
	# class that creates DRAW tensorflow computation graph, and train / perform basic experiments
	# Model : DRAW : A Recurrent Neural Network For Image Generation, arXiv : 1502.04623 [cs.CV], May 2015
	# Experiment Data : binarized MNIST

	## helper functions

	def single_linear(shape, input_tensor, scope_name="linear") :
		# generates single linear transformation layer
		# parameter : 
		#	shape : shape of W. [input_dim, output_dim]
		# 	input_tensor : input tensor, should be [None, input_dim]
		#	scope_name : optional. name of the variable scope.

		thres = np.sqrt(6.0 / (shape[0] + shape[1]))
		with tf.variable_scope(scope_name) :
			weights = tf.Variable(tf.random_uniform(shape, minval=-thres, maxval=thres), name="weights")
			biases = tf.Variable(tf.zeros([shape[1]]), name="biases")
		return tf.matmul(input_tensor, weights)+biases

	def read_layer(self, x, time, attention = False) :
		# generates 'read' layer, shape depending on the 'attention' parameter  
		# parameter :
		#	x : image input, [None, flattened(28*28)]
		#	time : current timestep
		# 	attention : attention ? full image : attentioned part of image

		x_reshape = tf.reshape(x, [self.mini_batch_size, self.input_dim])
		prev_c_reshape = tf.reshape(self.canvas_list[time-1], [self.mini_batch_size, self.input_dim])
		
		prev_h_dec = self.h_dec_list[time-1]
		hat_x_reshape = x_reshape - tf.nn.sigmoid(prev_c_reshape)

		if attention :
			# TODO : implement
			return tf.concat(1, [x_reshape, hat_x_reshape])
		else : return tf.concat(1, [x_reshape, hat_x_reshape])

	def encoder(self, input_tensor, time) :
		# generates 'encoder' LSTM layer's computation nodes 
		# return : lstm's output, h_enc^time 
		# parameter :
		#	input_tensor : [None, encoder_input_dim]
		#	time : timestamp, 1 to self.max_time
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
		with tf.variable_scope("decoder_lstm") :
			if time>1 : tf.get_variable_scope().reuse_variables()
			output, self.dec_lstm_state = self.dec_lstm(input_tensor, self.dec_lstm_state)
			return output

	def write_layer(self, input_tensor, time, attention = False) :
		# generates 'write' layer, shape depending on the 'attention' parameter
		# parameter :
		#	input_tensor : input tensor
		#	time : current timestamp
		#	attention : boolean for attention

		if attention : 
			# TODO : implement
			return single_linear([self.dec_lstm, self.input_dim], input_tensor, scope_name="write")
		else : 
			return single_linear([self.dec_lstm, self.input_dim], input_tensor, scope_name="write")

	def step_path(path) :
		return 'saved_step'+str(hash(model_path)%172062407)

	def __init__(self, image_shape, is_training = True, model_path=None) :
		# parameter :
		#	image_shape : shape of image, list of [height, width, channel]
		#	is_training : true if training, false if test / experiment
		#	model_path : path to save/load model

		assert (len(image_shape)==3 and (isinstance(image_shape, list) or isinstance(image_shape, tuple))), \
			"Parameter 'image_shape' should be a list/tuple with three elements! (height, width, channel)"

		# class parameters
		self.model_path = model_path
		self.is_training = is_training

		# model parameters
		self.mini_batch_size = 64
		self.max_time = 10 # max time sequence, T
		self.keep_prob = 0.5 # keep probability for dropout
		self.enc_size = 300
		self.dec_size = 300
		self.image_shape = image_shape # (height, width, channel)
		self.batch_image_shape = [self.mini_batch_size] + list(image_shape)
		self.input_dim = image_shape[0] * image_shape[1] * image_shape[2]
		self.latent_dim = 30 # dimension for latent vector (z)
		self.attention = False
		self.learning_rate = 0.003
		self.model_path = model_path

		# state variables & lstm cells
		self.canvas_0 = tf.Variable(tf.random_normal(self.image_shape), name="canvas_0")
		self.h_enc_0 = tf.Variable(tf.random_normal([self.enc_size]), name="h_enc_0")
		self.h_dec_0 = tf.Variable(tf.random_normal([self.dec_size]), name="h_dec_0")
		self.canvas_list = [tf.tile(self.canvas_0, [self.mini_batch_size, 1, 1, 1])] + ([None]*(self.max_time))
		self.h_enc_list = [tf.tile(self.h_enc_0, [self.mini_batch_size, 1])] + ([None]*(self.max_time))
		self.h_dec_list = [tf.tile(self.h_dec_0, [self.mini_batch_size, 1])] + ([None]*(self.max_time))
		self.enc_lstm = LSTMCell(self.enc_size, name="encoder_lstm")
		self.dec_lstm = LSTMCell(self.dec_size, name="decoder_lstm")
		self.enc_lstm_state = self.enc_lstm.zero_state(self.mini_batch_size)
		self.dec_lstm_state = self.dec_lstm.zero_state(self.mini_batch_size)
		if is_training : (self.enc_lstm, self.dec_lstm) = tuple([DropoutWrapper(x, output_keep_prob = self.keep_prob) for x in [self.enc_lstm, self.dec_lstm]])

		# build network
		self.x_input = tf.placeholder(tf.float32, shape=self.batch_image_shape)
		noise = tf.truncated_normal([self.mini_batch_size, self.latent_dim])
		
		mu_square_accum = tf.zeros([])
		sigma_square_accum = tf.zeros([])
		log_sigma_square_accum = tf.zeros([])

		for time in xrange(1, self.max_time+1) :
			
			with tf.name_scope('encoding') :
				read_out = self.read_layer(x_input, time)
				enc_out = self.h_enc_list[time] = self.encoder(read_out, time)
			
			with tf.name_scope('sample') :
				# sample : just sample one example for DRAW Network
				if is_training : # while training, geenerate z_mu and z_sigma from the output of encoder, and sample one z
					z_mu = single_linear([self.enc_size, self.latent_dim], enc_out, scope_name="z_mu")
					z_sigma = tf.exp(single_linear([self.enc_size, self.latent_dim], enc_out, scope_name="z_sigma"))
				else : # while test, just generate z by normal distribution ~ N(0, 1).	
					z_mu = tf.zeros([self.mini_batch_size, self.latent_dim])
					z_sigma = tf.ones([self.mini_batch_size, self.latent_dim])
				sample_out = z_mu + tf.mul(z_sigma, z_noise)

				# accumulate squares for future loss calculation
				mu_square_accum += tf.reduce_sum(tf.reduce_mean(tf.square(z_mu), 1))
				sigma_square = tf.square(z_sigma)
				sigma_square_accum += tf.reduce_sum(tf.reduce_mean(sigma_square, 1))
				log_sigma_square_accum += tf.reduce_sum(tf.reduce_mean(tf.log(sigma_square), 1))

			with tf.name_scope('decoding') :
				dec_out = self.h_dec_list[time] = self.decoder(sample_out, time)
				write_out = self.write(dec_out, time)
				self.canvas_list[time] = self.canvas_list[time-1] + tf.reshape(write_out, self.batch_image_shape)

		# results, losses and optimizer
		with tf.name_scope('results') :
			self.final_image = tf.nn.sigmoid(self.canvas_list[self.max_time])
			self.loss_x = tf.nn.sigmoid_cross_entropy_with_logits(self.canvas_list[self.max_time], x_input, name="loss_x")
			self.loss_z = 0.5 * (mu_square_accum + sigma_square_accum - log_sigma_suqare_accum - self.max_time)
			self.total_loss = loss_x + loss_z
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

		# summary writer, write just if training
		if is_training :
			with tf.name_scope('summary') :
				tf.scalar_summary('loss_x', loss_x)
				tf.scalar_summary('loss_z', loss_z)
				tf.scalar_summary('total_loss', total_loss)
				tf.histogram_summary('final_image', final_image)
				self.merged_summary = tf.merge_all_summaries()

		# if class is instantiated for image generation, try to load model here
		if not is_training :
			self.sess = tf.Session()
			saver = tf.train.Saver()
			self.load_model(saver, self.sess)

	def load_model(self, saver, sess) :
		# helper function to load model

		model_restore_success = False
		if self.model_path is not None :
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
				with open(step_path(self.model_path), 'rb') as f :
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
		with open(step_path(my_path), 'wb') as f :
			cPickle.dump(step)

	def train(self, train_data, valid_data, max_epoch = 60) :
		# train the model by given train data & valid data.
		# model first tries to load data from the given model path. if fails, just initiates new training from the scratch
		# parameter :
		#	train_data : numpy array, with (#_data, height, width, #_channel)
		#	valid_data : numpy array, same format with train_data
		#	max_epoch : optional. changes the max epoch of training session

		assert (self.is_training), "To train, class should be created with 'is_training = True'!"
		assert (train_data.shape[1:] == valid_data.shape[1:] == self.image_shape), "The shape of train data & validation data should be equal with 'image_shape'!"

		step = 0

		with tf.Session() as sess :
			saver = tf.train.Saver()
			summary_writer = tf.train.SummaryWriter("/summary", sess.graph)
			self.load_model(saver, sess)

			for epoch in xrange(1, max_epoch+1) :
				# train
				np.random.shuffle(train_data)
				for i in xrange(0, train_data.shape[0], self.mini_batch_size) :
					current_input = train_data[i:i+self.mini_batch_size]
					_, summary = sess.run([self.optimizer, self.merged_summary], feed_dict = {self.x_input : current_input})
					summary_writer.add_summary(summary, step)
					step += 1

				# validation, per 5 epoch
				if epoch % 5 == 0 :
					lx_sum = lz_sum = l_sum = count = 0.0
					for i in xrange(0, val_data.shape[0], self.mini_batch_size) :
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

				# save model, per 20 epoch
				if epoch % 20 == 0 :
					self.save_model(step, saver, sess)

	def generate(self) :
		# returns mini batch of randomly generated images
		return self.sess.run([self.final_image])

def binarize(load_results) :
	return tuple((np.where(d_x>0.5, 1.0, 0.0), d_y) for (d_x, d_y) in load_results)

if __name__ == "__main__" :
	mnist_data_path = "mnist.pkl.gz"
	with gzip.open(mnist_data_path, "rb") as f :
		train_set, valid_set, _temp = binarize(cPickle.load(f))	# binarized MNIST
		del _temp												# in this network, just use train set & validation set!
		train_set = train_set[0]; valid_set = valid_set[0]
		train_set = np.reshape(train_set, [train_set.shape[0], 28, 28, 1])
		valid_set = np.reshape(valid_set, [valid_set.shape[0], 28, 28, 1])

	my_DRAW = DRAW(image_shape=[28, 28, 1], is_training=True)
	my_DRAW.train()
