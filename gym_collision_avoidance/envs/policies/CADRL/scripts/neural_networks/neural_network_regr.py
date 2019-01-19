#!/usr/bin/env python

import numpy as np
import numpy.matlib
import pickle
import matplotlib.pyplot as plt
from test_data import generate_sinusoids
from nn_training_param import NN_training_param
import os
import time


# fully connected nerual network 

class Neural_network_regr:
	def __init__(self, nn_training_param, plotting_func=None, X_vis=None):
		self.set_training_param(nn_training_param)
		self.plotting_func = plotting_func
		self.X_vis = X_vis

	def initialize_network_param(self, num_hidden_layers, hidden_layers_size, input_dim, output_dim):
		self.id_num = -1
		self.num_hidden_layers = num_hidden_layers
		self.hidden_layers_size = hidden_layers_size
		self.input_dim = input_dim
		self.output_dim = output_dim
		#print 'input_dim, output_dim', input_dim, output_dim
		#print 'hidden_layers_size' , hidden_layers_size
		self.layers_dim = np.append(input_dim, \
				np.append(hidden_layers_size, output_dim))
		#print self.layers_dim
		self.num_layers = self.num_hidden_layers + 2

		assert(num_hidden_layers == len(hidden_layers_size))
		self.initialize_nn_weights()
		self.avg_vec = np.zeros((input_dim,))
		self.std_vec = np.ones((input_dim,))

		self.output_dim_weights = np.ones((output_dim,))
		self.output_avg_vec = np.zeros((output_dim,))
		self.output_std_vec = np.zeros((output_dim,))
		self.print_nn()

	def save_neural_network(self, filename):
		# save weights
		nn_list = []
		nn_list.append(self.W)
		nn_list.append(self.b)
		# save avg_vec and std_vec
		nn_list.append(self.avg_vec)
		nn_list.append(self.std_vec)
		nn_list.append(self.output_avg_vec)
		nn_list.append(self.output_std_vec)
		nn_list.append(self.id_num)
		pickle.dump(nn_list, open(filename, "wb"))
		return

	def load_neural_network(self, filename):
		nn_list = pickle.load(open(filename, "rb"))
		self.W = nn_list[0]
		self.b = nn_list[1]
		self.avg_vec = nn_list[2]
		self.std_vec = nn_list[3]
		self.output_avg_vec = nn_list[4]
		self.output_std_vec = nn_list[5]
		try:
			self.id_num = nn_list[6]
		except:
			self.id_num = 0

		# retrieve network params
		self.num_hidden_layers = len(self.W) - 1
		self.hidden_layers_size = np.zeros((self.num_hidden_layers,))
		for i in xrange(1, len(self.W)):
			self.hidden_layers_size[i-1] = self.W[i].shape[0]
		self.input_dim = self.W[0].shape[0]
		self.output_dim = self.W[-1].shape[1]
		#print 'input_dim, output_dim', input_dim, output_dim
		#print 'hidden_layers_size' , hidden_layers_size
		self.layers_dim = np.append(self.input_dim, \
				np.append(self.hidden_layers_size, self.output_dim))
		#print self.layers_dim
		self.num_layers = self.num_hidden_layers + 2
		# self.print_nn()
		self.output_dim_weights = np.ones((self.output_dim,))
		return

	def set_plotting_func(self, func, X_vis):
		self.plotting_func = func
		self.X_vis = X_vis

	def set_training_stepsize(self, sgd_stepsize_mode='fixed_decay', sgd_step_c=0.1, sgd_step_epsilon=0.1):
		self.nn_training_param.sgd_stepsize_mode = sgd_stepsize_mode
		self.nn_training_param.sgd_step_c = sgd_step_c
		self.nn_training_param.sdg_step_epsilon = sgd_step_epsilon
		if sgd_stepsize_mode == 'momentum' or sgd_stepsize_mode == 'sum_of_grad' \
			or sgd_stepsize_mode == 'rmsprop':
			self.initialize_sum_of_grad()
		

	def print_nn(self):
		print('---------------------------------------------------------')
		print('~~ neural_network_regr structure ~~')
		print('id', self.id_num)
		print('num_hidden_layers: %d' % self.num_hidden_layers)
		print('layers_dim', self.layers_dim)

		print('~~ neural_network_regr training param ~~')
		print('sgd_step_size: %f' % self.nn_training_param.sgd_step_size)
		print('reg_lambda: %f' % self.nn_training_param.reg_lambda)
		print('nb_iter: %d' % self.nn_training_param.nb_iter)
		print('sgd_batch_size: %d' % self.nn_training_param.sgd_batch_size)
		print('w_scale: %f' % self.nn_training_param.w_scale)
		print('avg_vec', self.avg_vec)
		print('std_vec', self.std_vec)
		print('out_avg_vec', self.output_avg_vec)
		print('output_std_vec', self.output_std_vec)
		print('---------------------------------------------------------')


	def initialize_nn_weights(self):
		# neural network parameters 
		self.W = list()
		self.dW = list()
		self.b = list()
		self.db = list()
		for i in xrange(self.num_hidden_layers+1):
			layer_input_dim = self.layers_dim[i]
			layer_output_dim = self.layers_dim[i+1]
			fan_in_weight = np.sqrt(2.0/layer_input_dim) 
			# print fan_in_weight
			self.W.append(self.nn_training_param.w_scale *
				(np.random.rand(layer_input_dim, layer_output_dim)-0.5))
			# self.W.append(fan_in_weight *
				# (np.random.randn(layer_input_dim, layer_output_dim)))
			self.dW.append(np.zeros((layer_input_dim, layer_output_dim)))
			self.b.append(np.zeros((1, layer_output_dim)))
			self.db.append(np.zeros((1, layer_output_dim)))

	def initialize_sum_of_grad(self):
		# for sum of grad
		self.sum_dW = list()
		self.sum_db = list()
		for i in xrange(self.num_hidden_layers+1):
			layer_input_dim = self.layers_dim[i]
			layer_output_dim = self.layers_dim[i+1]
			self.sum_dW.append(np.zeros((layer_input_dim, layer_output_dim)))
			self.sum_db.append(np.zeros((1, layer_output_dim)))

	def initialize_derivatives(self):
		self.dW = list()
		self.db = list()
		for i in xrange(self.num_hidden_layers+1):
			layer_input_dim = self.layers_dim[i]
			layer_output_dim = self.layers_dim[i+1]
			self.dW.append(np.zeros((layer_input_dim, layer_output_dim)))
			self.db.append(np.zeros((1, layer_output_dim)))

	def set_training_param(self, nn_training_param):
		self.nn_training_param = nn_training_param


	# compute shifts to the x-y variables
	def compute_offset(self, X, Y, input_output_ranges):
		if input_output_ranges == None:
			self.avg_vec = np.mean(X, axis = 0)
			self.std_vec = np.max(np.std(X, axis = 0), 0.01)
			self.output_avg_vec = np.mean(Y, axis = 0)
			self.output_std_vec = np.max(np.std(Y, axis = 0), 0.01)
		else:
			self.avg_vec = input_output_ranges[0]
			self.std_vec = input_output_ranges[1]
			self.output_avg_vec = input_output_ranges[2]
			self.output_std_vec = input_output_ranges[3]

	# scale X (xRaw_2_x)
	def xRaw_2_x(self, X_raw):
		if len(X_raw.shape) > 1:
			nb_examples = X_raw.shape[0]
		else:
			nb_examples = 1
		X = (X_raw - np.matlib.repmat(self.avg_vec, nb_examples, 1)) \
			/ np.matlib.repmat(self.std_vec, nb_examples, 1)
		return X

	# scale Y (yRaw_2_y)
	def yRaw_2_y(self, Y_raw):
		if len(Y_raw.shape) > 1:
			nb_examples = Y_raw.shape[0]
		else:
			nb_examples = 1
		Y = (Y_raw - np.matlib.repmat(self.output_avg_vec, nb_examples, 1)) \
			/ np.matlib.repmat(self.output_std_vec, nb_examples, 1)
		return Y

	# scale Y (y_2_yraw)
	def y_2_yRaw(self, Y):
		if len(Y.shape) > 1:
			nb_examples = Y.shape[0]
		else:
			nb_examples = 1
		Y_raw = Y * np.matlib.repmat(self.output_std_vec, nb_examples, 1) \
			+ np.matlib.repmat(self.output_avg_vec, nb_examples, 1)
		return Y_raw

	# back propagation
	def backprop(self, X, Y, step_size, iter):
		# training param
		param = self.nn_training_param
		# for forward/backward propogation
		nb_layers = self.num_hidden_layers + 1
		forward_prop_o = []
		backward_prop_xi = []
		for i in xrange(nb_layers):
			forward_prop_o.append(np.empty([1,1]))
			backward_prop_xi.append(np.empty([1,1]))

		batch_size = X.shape[0]

		out = X.copy()
		y_out = Y.copy()
		# one step back prop
		for layer in xrange(nb_layers-1):
			# RelU
			# print 'layer', layer
			# print 'self.W[layer].shape', self.W[layer].shape
			# print 'out', out.shape
			tmp = np.dot(out, self.W[layer]) \
				+ np.matlib.repmat(self.b[layer], batch_size, 1)
			forward_prop_o[layer] = tmp * (tmp>0)
			# dropout
			# p = 0.80
			# dropout_mask = (np.random.rand(*tmp.shape) < p) / p
			# forward_prop_o[layer] *= dropout_mask

			out = forward_prop_o[layer].copy()

		# last layer, derivative of quadratic cost
		scores = y_out - \
				 (np.dot(forward_prop_o[-2], self.W[nb_layers-1]) + \
				 np.matlib.repmat(self.b[nb_layers-1], batch_size, 1))
		scores = - np.matlib.repmat(self.output_dim_weights, batch_size, 1) * scores
		# print scores.shape
		# print expscores.shape
		# print expscores.sum(axis=1).shape
		# print np.matlib.repmat(expscores.sum(axis=1), k,1).transpose().shape

			
		#### backward pass starting from the output, i.e. the class probabilities
		ds = np.clip(scores, -1, 1) 		# partial derivative of loss wrt scores
		ds = ds / batch_size
		backward_prop_xi[nb_layers-1] = ds.copy()

		for j in xrange(nb_layers-1, 0, -1):
			# print 'j', j
			# print 'backward_prop_xi[j].shape', backward_prop_xi[j].shape
			# print 'forward_prop_o[j-1].transpose().shape', forward_prop_o[j-1].transpose().shape
			# print '(param.reg_lambda * self.W[j]).shape', (param.reg_lambda * self.W[j]).shape
			self.dW[j] = np.dot(forward_prop_o[j-1].transpose(), backward_prop_xi[j]) \
						+ param.reg_lambda * self.W[j]
			self.db[j] = backward_prop_xi[j].sum(axis=0)
			# compute xi for previous layer and threshold at 0 if O is <0 (ReLU gradient update)
			backward_prop_xi[j-1] = np.dot(backward_prop_xi[j], self.W[j].transpose()) 
			backward_prop_xi[j-1] = backward_prop_xi[j-1] * (forward_prop_o[j-1]>0)

		self.dW[0] = np.dot(X.transpose(), backward_prop_xi[0]) \
						+ param.reg_lambda * self.W[0]
		self.db[0] = backward_prop_xi[0].sum(axis=0)

		#### subgradient updates
		# print 'param.sgd_stepsize_mode', param.sgd_stepsize_mode
		# method 1: fixed_decay
		if param.sgd_stepsize_mode == 'fixed_decay':
			# update step size (e.g. decrease every ... iterations)
			if (iter % 200) == 0:
				step_size = step_size / 1.5
			# print 'fixed decay, step size', step_size
			# gradient udpate
			for j in xrange(nb_layers):
				self.W[j] = self.W[j] - step_size * self.dW[j]
				self.b[j] = self.b[j] - step_size * self.db[j]
		# method 2: sqrt_decay
		elif param.sgd_stepsize_mode == 'sqrt_decay':
			c = param.sgd_step_c
			epsilon = param.sgd_step_epsilon
			step_size = c / (np.sqrt(iter) + epsilon)
			# gradient udpate
			for j in xrange(nb_layers):
				self.W[j] = self.W[j] - step_size * self.dW[j]
				self.b[j] = self.b[j] - step_size * self.db[j]
		# method 3: sum of gradients
		elif param.sgd_stepsize_mode == 'sum_of_grad':
			c = param.sgd_step_c
			epsilon = param.sgd_step_epsilon
			for j in xrange(nb_layers):
				self.sum_dW[j] += np.square(self.dW[j])
				self.sum_db[j] += np.square(self.db[j])
				self.W[j] -= c / (np.sqrt(self.sum_dW[j]) + epsilon) * self.dW[j]
				self.b[j] -= c / (np.sqrt(self.sum_db[j]) + epsilon) * self.db[j]
			step_size = np.amax(c / (np.sqrt(self.sum_dW[0]) + epsilon)) 
		# method 4: momentum
		elif param.sgd_stepsize_mode == 'momentum':
			if step_size > 0.01:
				alpha = 0.5
			else:
				alpha = 0.99
			if (iter % 200) == 0:
				step_size = step_size / 1.5
			for j in xrange(nb_layers):
				self.sum_dW[j] = alpha * self.sum_dW[j] - step_size * (self.dW[j])
				self.sum_db[j] = alpha * self.sum_db[j] - step_size * (self.db[j])
				self.W[j] = self.W[j] + self.sum_dW[j]
				self.b[j] = self.b[j] + self.sum_db[j]
		# method 5: rmsprop
		elif param.sgd_stepsize_mode == 'rmsprop':
			alpha = 0.9
			if (iter % 200) == 0:
				step_size = step_size / 1.5
			for j in xrange(nb_layers):
				self.sum_dW[j] = alpha * self.sum_dW[j] + (1-alpha) * np.square(self.dW[j])
				self.sum_db[j] = alpha * self.sum_db[j] + (1-alpha) * np.square(self.db[j])
				self.W[j] -= 0.01 * step_size * self.dW[j] / (np.sqrt(self.sum_dW[j]+1e-6))
				self.b[j] -= 0.01 * step_size * self.db[j] / (np.sqrt(self.sum_db[j]+1e-6))
		else:
			assert('unknown nerual network training type')

		return step_size


	# training from scratch
	def train_nn(self, dataset, ERM=0, dataset_test=None, ifPrint=True, input_output_ranges=None):
		# unique training id
		self.id_num = np.random.randint(1000)
		''' process data '''
		X = dataset[0]
		Y = dataset[1]
		nb_examples = X.shape[0]
		# normalize dataset
		self.compute_offset(X, Y, input_output_ranges)
		X = self.xRaw_2_x(X)
		Y = self.yRaw_2_y(Y)
		# error checking
		try:
			assert(np.any(np.isnan(X)) == False)
			assert(np.any(np.isnan(Y)) == False)
		except:
			print('X', X)
			print('Y', Y)
			assert(0)

		param = self.nn_training_param
		''' if using sum_of_gradient step_size '''
		if param.sgd_stepsize_mode == 'sum_of_grad':
			self.initialize_sum_of_grad()
		if param.sgd_stepsize_mode == 'momentum':
			self.initialize_sum_of_grad()
		if param.sgd_stepsize_mode == 'rmsprop':
			self.initialize_sum_of_grad()
		
		''' training '''
		# start training
		step_size = param.sgd_step_size
		t_start = time.time()

		if ERM == 1:
			num_samples = nb_examples
		else:
			num_samples = param.sgd_batch_size

		# main loop
		for i in xrange(param.nb_iter):
			if ERM == 1 or param.sgd_batch_size > nb_examples: #if full gradient
				batch_examples = np.arange(nb_examples)
				batch_size = nb_examples
			else: 	# else SGD with minibatch size
				batch_size = param.sgd_batch_size
				batch_examples = np.random.permutation(np.arange(nb_examples))[:batch_size]

			#### forward pass starting from input
			step_size = self.backprop(X[batch_examples,:], Y[batch_examples], step_size, i)

			#### print to screen for debugging
			if (i % np.ceil(param.nb_iter/100.0)) == 0 and ifPrint:
				z_train, z_sq_loss = self.evaluate_network_loss(X, Y)
				print('Iter %d, time elapsed: %f, Training disrete error: %f, square loss: %f, step_size_mode: %s,  step size=%f' % \
							 (i, time.time()-t_start, z_train, z_sq_loss, param.sgd_stepsize_mode, step_size))
				if self.plotting_func != None and self.X_vis != None:
					title_string = 'iter %d' % i
					figure_name = 'training'
					Y_vis = self.make_prediction_raw(self.X_vis)
					self.plotting_func(self.X_vis, Y_vis, title_string, figure_name=figure_name)
				if dataset_test != None:
					X_test = dataset_test[0]
					Y_test = dataset_test[1]
					nb_test_ex = X_test.shape[0]
					X_test = self.xRaw_2_x(X_test)
					Y_test = self.yRaw_2_y(Y_test)
					z_test, z_sq_test = self.evaluate_network_loss(X_test, Y_test)
					print('Test discrete error: %f, test square loss: %f ' % (z_test, z_sq_test))
				print(' ')

				# spiral_X_vis = pickle.load(open(file_dir+"/test_data/spiral_dataset_vis.p", "rb"))
				# scores = neural_network.make_prediction_raw(spiral_X_vis)
				# spiral_Y_vis = np.argmax(scores, axis = 1)
				
				# generate_spirals.plot_spiral_dataset(spiral_X_vis, spiral_Y_vis, title_string, if_new_figure)



	# evaluating network loss
	def evaluate_network_loss(self, X, Y):
		Y_hat = self.make_prediction(X)
		sqloss = self.compute_sqloss(Y_hat, Y)

		scores = Y - Y_hat
		batch_size = Y.shape[0]
		scores = np.matlib.repmat(self.output_dim_weights, batch_size, 1) * np.square(scores)
		Y_hat = np.sum(scores, axis = 1)
		threshold = 0.25
		discrete_loss = (Y.squeeze() > 0.25).sum() / float(Y.shape[0])
		return discrete_loss, sqloss

	def make_prediction(self, X):
		if len(X.shape) > 1:
			nb_examples = X.shape[0]
		else:
			nb_examples = 1
		nb_layers = self.num_hidden_layers + 1
		out = X
		for layer in xrange(nb_layers-1):
			tmp = np.dot(out, self.W[layer]) \
					+ np.matlib.repmat(self.b[layer], nb_examples, 1)
			out = tmp * (tmp>0)
		y_hat = np.dot(out, self.W[nb_layers-1]) + \
					 np.matlib.repmat(self.b[nb_layers-1], nb_examples, 1)
		return y_hat

	def compute_sqloss(self, Y_hat, Y):	
		# print Y_hat
		# print 'Y', Y	
		# assert(0)
		batch_size = Y.shape[0]
		scores = Y - Y_hat
		scores = np.matlib.repmat(self.output_dim_weights, batch_size, 1) * scores
		sq_loss = 0.5 * np.sum(np.square(scores)) / batch_size
		return sq_loss

	# requires scaling the input dimension
	def make_prediction_raw(self, X_raw):
		X = self.xRaw_2_x(X_raw)
		Y_scale = self.make_prediction(X)
		Y_hat = self.y_2_yRaw(Y_scale)
		return Y_hat


if __name__ == '__main__':
	print('hello world from neural_network.py')
	file_dir = os.path.dirname(os.path.realpath(__file__))
	plt.rcParams.update({'font.size': 18})

	''' test on the spiral dataset '''
	# dataset_name = "/sinusoid1D"; 
	dataset_name = "/sinusoid2D";
	sinusoid_dataset = pickle.load(open(file_dir+"/test_data" + dataset_name + "_dataset_train.p","rb"))
	sinusoid_X = sinusoid_dataset[0]
	sinusoid_Y = sinusoid_dataset[1]
	sinusoid_X_vis = pickle.load(open(file_dir+"/test_data" + dataset_name + "_dataset_vis.p", "rb"))

	print('sinusoid_X.shape', sinusoid_X.shape)
	print('sinusoid_Y.shape', sinusoid_Y.shape)

	# plot training data
	# generate_spirals.plot_spiral_dataset(spiral_dataset.X, spiral_dataset.Y, 'training data')
	
	# plot visualization data
	# spiral_Y_vis = np.zeros((spiral_X_vis.shape[0],), dtype='int')
	# generate_spirals.plot_spiral_dataset(spiral_X_vis, spiral_Y_vis, 'visulaization data')

	# plt.show()

	''' initializing neural network ''' 
	sgd_step_size = 10.0/20.0
	reg_lambda = 1.0/1000.0
	nb_iter = 1000
	sgd_batch_size = 50 #0
	w_scale = 0.1
	sgd_stepsize_mode = 'training data' 
	sgd_step_c = 0.1
	sgd_step_epsilon = 0.1

	nn_training_param = NN_training_param(sgd_step_size, reg_lambda, nb_iter, sgd_batch_size, w_scale)

	num_hidden_layers = 3
	hidden_layers_size = np.array([300, 200, 200],dtype='int32')
	input_dim = sinusoid_X.shape[1]
	output_dim = sinusoid_Y.shape[1]
	neural_network_regr = Neural_network_regr(nn_training_param)
	neural_network_regr.initialize_network_param(num_hidden_layers, \
					hidden_layers_size, input_dim, output_dim)
	neural_network_regr.set_plotting_func(generate_sinusoids.plot_sinusoid_dataset, sinusoid_X_vis)

	''' training the neural network '''
	ERM = 0
	generate_sinusoids.plot_sinusoid_dataset(sinusoid_X, sinusoid_Y, 'training data')
	# method 1
	# neural_network_regr.set_training_stepsize('fixed_decay')
	# method 2
	# neural_network_regr.set_training_stepsize('sqrt_decay', 0.5, 0.1)
	# method 3
	# neural_network_regr.set_training_stepsize('sum_of_grad', 0.1, 0.1)
	# method 4
	# neural_network_regr.set_training_stepsize('momentum', 0.1, 0.1)
	# method 5
	neural_network_regr.set_training_stepsize('rmsprop', 0.1, 0.1)
	# training
	neural_network_regr.set_plotting_func(generate_sinusoids.plot_sinusoid_dataset, sinusoid_X_vis)
	neural_network_regr.train_nn(sinusoid_dataset, ERM=ERM)

	''' make prediction '''
	sinusoid_Y_vis = neural_network_regr.make_prediction_raw(sinusoid_X_vis)
	generate_sinusoids.plot_sinusoid_dataset(sinusoid_X_vis, sinusoid_Y_vis, 'regression')
	generate_sinusoids.plot_sinusoid_dataset_compare(\
		sinusoid_X, sinusoid_Y, sinusoid_X_vis, sinusoid_Y_vis)

	plt.show()

