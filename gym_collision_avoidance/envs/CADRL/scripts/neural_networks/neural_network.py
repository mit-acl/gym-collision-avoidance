#!/usr/bin/env python

import numpy as np
import numpy.matlib
import pickle
import matplotlib.pyplot as plt
from test_data import generate_spirals
from nn_training_param import NN_training_param
import os
import time

# fully connected nerual network 

class Neural_network:
	def __init__(self, nn_training_param, num_hidden_layers, \
					hidden_layers_size, input_dim, output_dim, plotting_func=None,X_vis=None):
		self.set_training_param(nn_training_param)
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
		self.print_nn()
		self.initialize_nn_weights()
		self.avg_vec = np.zeros((input_dim,))
		self.std_vec = np.ones((input_dim,))
		self.plotting_func = plotting_func
		self.X_vis = X_vis

	def save_neural_network(self, filename):
		# save weights
		nn_list = []
		nn_list.append(self.W)
		nn_list.append(self.b)
		# save avg_vec and std_vec
		nn_list.append(self.avg_vec)
		nn_list.append(self.std_vec)
		pickle.dump(nn_list, open(filename, "wb"))
		return

	def load_neural_network(self, filename):
		nn_list = pickle.load(open(filename, "rb"))
		self.W = nn_list[0]
		self.b = nn_list[1]
		self.avg_vec = nn_list[2]
		self.std_vec = nn_list[3]
		return

	def set_plotting_func(self, func, X_vis):
		self.plotting_func = func
		self.X_vis = X_vis

	def set_training_stepsize(self, sgd_stepsize_mode='fixed_decay', sgd_step_c=0.1, sgd_step_epsilon=0.1):
		self.nn_training_param.sgd_stepsize_mode = sgd_stepsize_mode
		self.nn_training_param.sgd_step_c = sgd_step_c
		self.nn_training_param.sdg_step_epsilon = sgd_step_epsilon
		

	def print_nn(self):
		print('---------------------------------------------------------')
		print('~~ neural_network structure ~~')
		print('num_hidden_layers: %d' % self.num_hidden_layers)
		print('layers_dim', self.layers_dim)

		print('~~ neural_network training param ~~')
		print('sgd_step_size: %f' % self.nn_training_param.sgd_step_size)
		print('reg_lambda: %f' % self.nn_training_param.reg_lambda)
		print('nb_iter: %d' % self.nn_training_param.nb_iter)
		print('sgd_batch_size: %d' % self.nn_training_param.sgd_batch_size)
		print('w_scale: %f' % self.nn_training_param.w_scale)
		print('---------------------------------------------------------')


	def initialize_nn_weights(self):
		# neural parameters 
		self.W = list()
		self.dW = list()
		self.b = list()
		self.db = list()
		# for sum of grad
		self.sum_dW = list()
		self.sum_db = list()

		for i in xrange(self.num_hidden_layers+1):
			layer_input_dim = self.layers_dim[i]
			layer_output_dim = self.layers_dim[i+1]
			self.W.append(self.nn_training_param.w_scale * \
				np.random.randn(layer_input_dim, layer_output_dim))
			self.dW.append(np.zeros((layer_input_dim, layer_output_dim)))
			self.b.append(np.zeros((1, layer_output_dim)))
			self.db.append(np.zeros((1, layer_output_dim)))
			self.sum_dW.append(np.zeros((layer_input_dim, layer_output_dim)))
			self.sum_db.append(np.zeros((1, layer_output_dim)))

	def set_training_param(self, nn_training_param):
		self.nn_training_param = nn_training_param


	def train_nn(self, dataset, ERM=0, dataset_test=None):
		X = dataset[0]
		Y = dataset[1].astype(int)
		nb_examples = X.shape[0]
		k = self.layers_dim[-1]
		# normalize dataset
		self.avg_vec = np.mean(X, axis = 0)
		self.std_vec = np.std(X, axis = 0)
		X = (X - np.matlib.repmat(self.avg_vec, nb_examples, 1)) \
			/ np.matlib.repmat(self.std_vec, nb_examples, 1)
		
		# for SGD
		gamma = 0.95
		param = self.nn_training_param
		step_size = param.sgd_step_size
		sgd_stepsize_mode = param.sgd_stepsize_mode
		sgd_step_c = param.sgd_step_c
		sgd_step_epsilon = param.sgd_step_epsilon

		# start training
		t_start = time.time()

		# for forward/backward propogation
		nb_layers = self.num_hidden_layers+1
		forward_prop_o = []
		backward_prop_xi = []
		if ERM == 1:
			num_samples = nb_examples
		else:
			num_samples = param.sgd_batch_size
		for i in xrange(nb_layers):
			forward_prop_o.append(np.empty([1,1]))
			backward_prop_xi.append(np.empty([1,1]))
			#forward_prop_o.append(np.zeros(num_samples, self.layers_dim[i+1]))
			#backward_prop_xi.append()

		# main loop
		for i in xrange(param.nb_iter):
			if ERM == 1 or param.sgd_batch_size > nb_examples: #if full gradient
				batch_examples = np.arange(nb_examples)
				batch_size = nb_examples
			else: 	# else SGD with minibatch size
				batch_size = param.sgd_batch_size
				batch_examples = np.random.permutation(np.arange(nb_examples))[:batch_size]

			#### forward pass starting from input
			out = X[batch_examples,:].copy()
			y_out = Y[batch_examples].copy()
			for layer in xrange(nb_layers-1):
				# RelU
				tmp = np.dot(out, self.W[layer]) \
					+ np.matlib.repmat(self.b[layer], batch_size, 1)
				forward_prop_o[layer] = tmp * (tmp>0)
				out = forward_prop_o[layer].copy()
			# last layer, softmax
			scores = np.dot(forward_prop_o[-2], self.W[nb_layers-1]) + \
					 np.matlib.repmat(self.b[nb_layers-1], batch_size, 1)
			scores = scores - np.outer(np.amax(scores, axis=1), np.ones(k,)) 
			expscores = np.exp(scores)
			# print scores.shape
			# print expscores.shape
			# print expscores.sum(axis=1).shape
			# print np.matlib.repmat(expscores.sum(axis=1), k,1).transpose().shape
			p = np.divide(expscores, np.matlib.repmat(expscores.sum(axis=1), k,1).transpose())

			#### print to screen for debugging
			if (i % np.ceil(param.nb_iter/100.0)) == 0:
				z_train, z_logloss = self.evaluate_network_loss(X, Y)
				print('Iter %d, time elapsed: %f, Training 0-1 error: %f, log loss: %f, step_size_mode: %s,  step size=%f' % \
							 (i, time.time()-t_start, z_train, z_logloss, sgd_stepsize_mode, step_size))
				if self.plotting_func != None and self.X_vis != None:
					title_string = 'iter %d' % i
					if_new_figure = 1 - (i > 0)
					Y_vis = self.make_prediction_raw(self.X_vis)
					self.plotting_func(self.X_vis, Y_vis, title_string, if_new_figure)
				if dataset_test != None:
					X_test = dataset_test[0]
					Y_test = dataset_test[1].astype(int)
					nb_test_ex = X_test.shape[0]
					X_test = (X_test - np.matlib.repmat(self.avg_vec, nb_test_ex, 1)) \
								/ np.matlib.repmat(self.std_vec, nb_test_ex, 1)
					z_test, z_logloss_test = self.evaluate_network_loss(X_test, Y_test)
					print('Test 0-1 error: %f, test log loss: %f ' % (z_test, z_logloss_test))
				print(' ')

				# spiral_X_vis = pickle.load(open(file_dir+"/test_data/spiral_dataset_vis.p", "rb"))
				# scores = neural_network.make_prediction_raw(spiral_X_vis)
				# spiral_Y_vis = np.argmax(scores, axis = 1)
				
				# generate_spirals.plot_spiral_dataset(spiral_X_vis, spiral_Y_vis, title_string, if_new_figure)
				
			#### backward pass starting from the output, i.e. the class probabilities
			ds = p.copy() 		# partial derivative of loss wrt scores
			ds[np.arange(ds.shape[0]),y_out.squeeze()] -= 1
			ds = ds / batch_size
			backward_prop_xi[nb_layers-1] = ds.copy()

			for j in xrange(nb_layers-1, 0, -1):
				# print 'j', j
				# print 'backward_prop_xi[j].shape', backward_prop_xi[j].shape
				# print 'forward_prop_o[j-1].transpose().shape', forward_prop_o[j-1].transpose().shape
				# print '(param.reg_lambda * self.W[j]).shape', (param.reg_lambda * self.W[j]).shape
				self.dW[j] = np.dot(forward_prop_o[j-1].transpose(), backward_prop_xi[j]) + \
							param.reg_lambda * self.W[j]
				self.db[j] = backward_prop_xi[j].sum(axis=0)
				# compute xi for previous layer and threshold at 0 if O is <0 (ReLU gradient update)
				backward_prop_xi[j-1] = numpy.dot(backward_prop_xi[j], self.W[j].transpose()) 
				backward_prop_xi[j-1] = backward_prop_xi[j-1] * (forward_prop_o[j-1]>0)

			self.dW[0] = np.dot(X[batch_examples,:].transpose(), backward_prop_xi[0]) + \
							param.reg_lambda * self.W[0]
			self.db[0] = backward_prop_xi[0].sum(axis=0)

			#### subgradient updates
			# method 1: fixed_decay
			if sgd_stepsize_mode == 'fixed_decay':
				# update step size (e.g. decrease every ... iterations)
				if (i % 2000) == 0:
					step_size = step_size/1.5
				# gradient udpate
				for j in xrange(nb_layers):
					self.W[j] = self.W[j] - step_size * self.dW[j]
					self.b[j] = self.b[j] - step_size * self.db[j]
			# method 2: sqrt_decay
			elif sgd_stepsize_mode == 'sqrt_decay':
				c = sgd_step_c
				epsilon = sgd_step_epsilon
				step_size = c / (np.sqrt(i) + epsilon)
				# gradient udpate
				for j in xrange(nb_layers):
					self.W[j] = self.W[j] - step_size * self.dW[j]
					self.b[j] = self.b[j] - step_size * self.db[j]
			# method 3: sum of gradients
			elif sgd_stepsize_mode == 'sum_of_grad':
				c = sgd_step_c
				epsilon = sgd_step_epsilon
				for j in xrange(nb_layers):
					self.sum_dW[j] += np.square(self.dW[j])
					self.sum_db[j] += np.square(self.db[j])
					self.W[j] -= c / (np.sqrt(self.sum_dW[j]) + epsilon) * self.dW[j]
					self.b[j] -= c / (np.sqrt(self.sum_db[j]) + epsilon) * self.db[j]
				step_size = np.amax(c / (np.sqrt(self.sum_dW[0]) + epsilon)) 



	# evaluating network loss
	def evaluate_network_loss(self, X, Y):
		scores = self.make_prediction(X)
		logloss = self.compute_logloss(scores, Y)
		Y_hat = np.argmax(scores, axis = 1)
		zero_one_loss = (Y.squeeze() != Y_hat).sum() / float(Y.shape[0])
		return zero_one_loss, logloss

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
		scores = np.dot(out, self.W[nb_layers-1]) + \
					 np.matlib.repmat(self.b[nb_layers-1], nb_examples, 1)
		return scores

	def compute_logloss(self, scores, Y):		
		k = scores.shape[1]
		scores = scores - np.outer(np.amax(scores, axis=1), np.ones(k,)) 
		expscores = np.exp(scores)
		p = np.divide(expscores, np.matlib.repmat(expscores.sum(axis=1), k,1).transpose())
		logloss = np.log(p[np.arange(p.shape[0]),Y.squeeze()]).sum()
		return logloss

	# requires scaling the input dimension
	def make_prediction_raw(self, X_raw):
		if len(X_raw.shape) > 1:
			n = X_raw.shape[0]
		else:
			n = 1
		X = (X_raw - np.matlib.repmat(self.avg_vec, n, 1)) \
			/ np.matlib.repmat(self.std_vec, n, 1)
		return self.make_prediction(X)

	def predict_y_hat(self, X_raw):
		scores = self.make_prediction_raw(X_raw)
		# print 'scores', scores
		return np.argmax(scores, axis = 1)








if __name__ == '__main__':
	print('hello world from neural_network.py')
	file_dir = os.path.dirname(os.path.realpath(__file__))
	plt.rcParams.update({'font.size': 18})

	''' test on the spiral dataset '''
	spiral_dataset = pickle.load(open(file_dir+"/test_data/spiral_dataset_train.p","rb"))
	spiral_X = spiral_dataset[0]
	spiral_Y = spiral_dataset[1]
	spiral_X_vis = pickle.load(open(file_dir+"/test_data/spiral_dataset_vis.p", "rb"))

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
	input_dim = spiral_X.shape[1]
	output_dim = np.amax(spiral_Y) + 1
	neural_network = Neural_network(nn_training_param, num_hidden_layers, \
					hidden_layers_size, input_dim, output_dim)
	neural_network.set_plotting_func(generate_spirals.plot_spiral_datasetWrapper, spiral_X_vis)

	''' training the neural network '''
	ERM = 0
	generate_spirals.plot_spiral_dataset(spiral_X, spiral_Y, 'training data')
	# method 1
	# neural_network.set_training_stepsize('fixed_decay')
	# method 2
	# neural_network.set_training_stepsize('sqrt_decay', 1.0, 0.1)
	# method 3
	neural_network.set_training_stepsize('sum_of_grad', 0.1, 0.1)
	# training
	neural_network.train_nn(spiral_dataset, ERM=ERM)

	''' make prediction '''
	scores = neural_network.make_prediction_raw(spiral_X_vis)
	spiral_Y_vis = np.argmax(scores, axis = 1)
	generate_spirals.plot_spiral_dataset(spiral_X_vis, spiral_Y_vis, 'classification')

	plt.show()

