#!/usr/bin/env python

import numpy as np
import numpy.matlib
import pickle
import matplotlib.pyplot as plt
from gym_collision_avoidance.envs.policies.CADRL.scripts.neural_networks.test_data import generate_symmetric_sinusoids
from gym_collision_avoidance.envs.policies.CADRL.scripts.neural_networks.nn_training_param import NN_training_param
from gym_collision_avoidance.envs.policies.CADRL.scripts.neural_networks.multiagent_network_param import Multiagent_network_param
import os
import time
import copy


# fully connected nerual network with weight sharing for 
# capturing symmetry in multiagent systems

class Neural_network_regr_multi:
	def __init__(self, nn_training_param, plotting_func=None, X_vis=None):
		self.set_training_param(nn_training_param)
		self.plotting_func = plotting_func
		self.X_vis = X_vis


	# layer_info = [[num_types, nodes_per_type], [num_types, nodes_per_type]]
	def initialize_network_param(self, layers_info, layers_type, multiagent_net_param=None):
		self.id_num = -1

		#print self.layers_dim
		if multiagent_net_param is not None:
			self.multiagent_net_param = multiagent_net_param
		else:
			self.multiagent_net_param = Multiagent_network_param(layers_info, layers_type)

		# populate other fields from layers_info
		self.num_layers = len(layers_info)
		self.num_hidden_layers = self.num_layers - 2
		self.layers_dim = []
		for i in range(len(layers_info)):
			self.layers_dim.append(int(np.sum(layers_info[i][:,0]*layers_info[i][:,1])))
		print(self.layers_dim)
		self.input_dim = self.layers_dim[0]
		self.output_dim = self.layers_dim[-1]

		self.initialize_nn_weights()
		self.avg_vec = np.zeros((self.input_dim,))
		self.std_vec = np.ones((self.input_dim,))

		self.output_dim_weights = np.ones((self.output_dim,))
		self.output_avg_vec = np.zeros((self.output_dim,))
		self.output_std_vec = np.zeros((self.output_dim,))
		# self.print_nn()


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
		nn_list.append(self.multiagent_net_param.layers_info)
		nn_list.append(self.multiagent_net_param.layers_type)
		nn_list.append(self.multiagent_net_param.symmetric_indices)
		nn_list.append(self.multiagent_net_param.symmetric_indices_b)
		nn_list.append(self.id_num)
		pickle.dump(nn_list, open(filename, "wb"))
		return

	def load_neural_network(self, filename):
		with open(filename, 'rb') as fo:
		    try:
		        nn_list = pickle.load(fo)
		    except UnicodeDecodeError:  #python 3.x
		        fo.seek(0)
		        nn_list = pickle.load(fo, encoding='latin1')
		self.W = nn_list[0]
		self.b = nn_list[1]
		self.avg_vec = nn_list[2]
		self.std_vec = nn_list[3]
		self.output_avg_vec = nn_list[4]
		self.output_std_vec = nn_list[5]
		# multiagent_net_param
		layers_info = nn_list[6]
		layers_type = nn_list[7]
		symmetric_indices = nn_list[8]
		symmetric_indices_b = nn_list[9]
		self.multiagent_net_param = Multiagent_network_param(layers_info, layers_type, \
			symmetric_indices=symmetric_indices, symmetric_indices_b=symmetric_indices_b)
		self.id_num = nn_list[10]

		# retrieve network params
		self.num_hidden_layers = len(self.W) - 1
		self.input_dim = self.W[0].shape[0]
		self.output_dim = self.W[-1].shape[1]
		#print 'input_dim, output_dim', input_dim, output_dim
		#print 'hidden_layers_size' , hidden_layers_size
		self.layers_dim = []
		for i in  range(len(layers_info)):
			self.layers_dim.append(int(np.sum(layers_info[i][:,0]*layers_info[i][:,1])))
		#print self.layers_dim
		self.num_layers = self.num_hidden_layers + 2
		# self.print_nn()
		self.output_dim_weights = np.ones((self.output_dim,))
		self.load_symBlocks()
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

	def load_symBlocks(self):
		self.sym_W = list()
		self.sym_dW = list()
		self.sym_b = list()
		self.sym_db = list()
		# ith layer, jth symmetry block
		for i in range(self.num_hidden_layers+1):
			sym_W_layer = list()
			sym_dW_layer = list()
			sym_b_layer = list()
			sym_db_layer = list()
			# W, dW
			for j in range(len(self.multiagent_net_param.symmetric_indices[i])):
				a = self.multiagent_net_param.symmetric_indices[i][j][0, 0]
				b = self.multiagent_net_param.symmetric_indices[i][j][0, 1]
				c = self.multiagent_net_param.symmetric_indices[i][j][0, 2]
				d = self.multiagent_net_param.symmetric_indices[i][j][0, 3]
				sym_W_layer.append(self.W[i][a:b,c:d].copy())
				sym_dW_layer.append(np.zeros((b-a, d-c)))
			# b, db
			for k in range(len(self.multiagent_net_param.symmetric_indices_b[i])):
				a = self.multiagent_net_param.symmetric_indices_b[i][k][0, 0]
				b = self.multiagent_net_param.symmetric_indices_b[i][k][0, 1]
				sym_b_layer.append(self.b[i][0,a:b].copy())
				sym_db_layer.append(np.zeros((b-a, )))
			

			self.sym_W.append(sym_W_layer)
			self.sym_dW.append(sym_dW_layer)
			self.sym_b.append(sym_b_layer)
			self.sym_db.append(sym_db_layer)


	def initialize_nn_weights(self):
		# compute symmetric indices blocks
		self.sym_W = list()
		self.sym_dW = list()
		self.sym_b = list()
		self.sym_db = list()
		# ith layer, jth symmetry block
		for i in range(self.num_hidden_layers+1):
			sym_W_layer = list()
			sym_dW_layer = list()
			sym_b_layer = list()
			sym_db_layer = list()
			# W, dW
			for j in range(len(self.multiagent_net_param.symmetric_indices[i])):
				num_rows = self.multiagent_net_param.symmetric_indices[i][j][0,1] - \
							self.multiagent_net_param.symmetric_indices[i][j][0,0]
				num_cols = self.multiagent_net_param.symmetric_indices[i][j][0,3] - \
							self.multiagent_net_param.symmetric_indices[i][j][0,2]
				sym_W_layer.append(self.nn_training_param.w_scale * \
					(np.random.rand(num_rows, num_cols)-0.5))
				sym_dW_layer.append(np.zeros((num_rows, num_cols)))
			# b, db
			for k in range(len(self.multiagent_net_param.symmetric_indices_b[i])):
				num_cols = self.multiagent_net_param.symmetric_indices_b[i][k][0,1] - \
					self.multiagent_net_param.symmetric_indices_b[i][k][0,0]
				sym_b_layer.append(np.zeros((1, num_cols)))
				sym_db_layer.append(np.zeros((1, num_cols)))
			

			self.sym_W.append(sym_W_layer)
			self.sym_dW.append(sym_dW_layer)
			self.sym_b.append(sym_b_layer)
			self.sym_db.append(sym_db_layer)

		# neural network parameters 
		self.W = list()
		self.dW = list()
		self.b = list()
		self.db = list()
		for i in range(self.num_hidden_layers+1):
			if self.multiagent_net_param.layers_type[i] == 'conn':
				layer_input_dim = self.layers_dim[i]
				layer_output_dim = self.layers_dim[i+1]
				fan_in_weight = np.sqrt(2.0/layer_input_dim) 
				# print fan_in_weight
				self.W.append(np.zeros((layer_input_dim, layer_output_dim)))
				self.dW.append(np.zeros((layer_input_dim, layer_output_dim)))
				self.b.append(np.zeros((1, layer_output_dim)))
				self.db.append(np.zeros((1, layer_output_dim)))
			elif self.multiagent_net_param.layers_type[i] == 'max':
				self.W.append([])
				self.dW.append([])
				self.b.append([])
				self.db.append([])


		self.symIndices_2_mat()

	def symIndices_2_mat(self):
		for i in range(self.num_hidden_layers+1):
			# W
			for j in range(len(self.multiagent_net_param.symmetric_indices[i])):
				for jj in range(self.multiagent_net_param.symmetric_indices[i][j].shape[0]):
					a = self.multiagent_net_param.symmetric_indices[i][j][jj,0]
					b = self.multiagent_net_param.symmetric_indices[i][j][jj,1]
					c = self.multiagent_net_param.symmetric_indices[i][j][jj,2]
					d = self.multiagent_net_param.symmetric_indices[i][j][jj,3]
					# print '~~~', i, j
					# print a,b,c,d
					# # print self.sym_W[i][j].shape
					# print self.W[i].shape
					# print i, self.W[i].shape, a,b,c,d
					# print self.W[i][a:b,c:d].shape, self.sym_W[i][j].shape

					self.W[i][a:b,c:d] = self.sym_W[i][j]
			# b
			for k in range(len(self.multiagent_net_param.symmetric_indices_b[i])):
				for kk in range(self.multiagent_net_param.symmetric_indices_b[i][k].shape[0]):
					a = self.multiagent_net_param.symmetric_indices_b[i][k][kk,0]
					b = self.multiagent_net_param.symmetric_indices_b[i][k][kk,1]
					# print 'i,k,a,b', i, k, a, b
					# print 'self.b[i].shape', self.b[i].shape
					# print 'self.sym_b[i][k].shape', self.sym_b[i][k].shape
					# print self.b[i][a:b].shape
					self.b[i][0,a:b] = self.sym_b[i][k]

	def dW_2_symIndices(self):
		for i in range(self.num_hidden_layers+1):
			# update sym_dW
			for j in range(len(self.multiagent_net_param.symmetric_indices[i])):
				self.sym_dW[i][j][:] = 0
				for jj in range(self.multiagent_net_param.symmetric_indices[i][j].shape[0]):
					a = self.multiagent_net_param.symmetric_indices[i][j][jj,0]
					b = self.multiagent_net_param.symmetric_indices[i][j][jj,1]
					c = self.multiagent_net_param.symmetric_indices[i][j][jj,2]
					d = self.multiagent_net_param.symmetric_indices[i][j][jj,3]
					self.sym_dW[i][j] += self.dW[i][a:b,c:d]
			# update sym_db
			for k in range(len(self.multiagent_net_param.symmetric_indices_b[i])):
				self.sym_db[i][k][:] = 0
				for kk in range(self.multiagent_net_param.symmetric_indices_b[i][k].shape[0]):
					a = self.multiagent_net_param.symmetric_indices_b[i][k][kk,0]
					b = self.multiagent_net_param.symmetric_indices_b[i][k][kk,1]
					self.sym_db[i][k] += self.db[i][a:b]

	def update_symIndices(self, param, step_size, iteration):
		# method 1: fixed_decay
		if param.sgd_stepsize_mode == 'fixed_decay':
			# update step size (e.g. decrease every ... iterations)
			if (iteration % 200) == 0:
				step_size = step_size / 1.5
			# print 'fixed decay, step size', step_size
			# gradient udpate
			for i in range(self.num_hidden_layers+1):
				# update sym_dW
				for j in range(len(self.multiagent_net_param.symmetric_indices[i])):
					self.sym_W[i][j] -= step_size * self.sym_dW[i][j]
				# update sym_db
				for k in range(len(self.multiagent_net_param.symmetric_indices_b[i])):
					self.sym_b[i][k] -= step_size * self.sym_db[i][k]

		# method 2: sqrt_decay
		elif param.sgd_stepsize_mode == 'sqrt_decay':
			c = param.sgd_step_c
			epsilon = param.sgd_step_epsilon
			step_size = c / (np.sqrt(iteration) + epsilon)
			# gradient udpate
			for i in range(self.num_hidden_layers+1):
				# update sym_dW
				for j in range(len(self.multiagent_net_param.symmetric_indices[i])):
					self.sym_W[i][j] -= step_size * self.sym_dW[i][j]
				# update sym_db
				for k in range(len(self.multiagent_net_param.symmetric_indices_b[i])):
					self.sym_b[i][k] -= step_size * self.sym_db[i][k]

		# method 3: sum of gradients
		elif param.sgd_stepsize_mode == 'sum_of_grad':
			c = param.sgd_step_c
			epsilon = param.sgd_step_epsilon
			for i in range(self.num_hidden_layers+1):
				# update sym_dW
				for j in range(len(self.multiagent_net_param.symmetric_indices[i])):
					self.sum_sym_dW[i][j] += np.square(self.sym_dW[i][j])
					self.sym_W[i][j] -= c / (np.sqrt(self.sum_sym_dW[i][j]) + epsilon) \
										* self.sym_dW[i][j]
				# update sym_db
				for k in range(len(self.multiagent_net_param.symmetric_indices_b[i])):
					self.sum_sym_db[i][k] += np.square(self.sym_db[i][k])
					self.sym_b[i][k] -= c / (np.sqrt(self.sum_sym_db[i][k]) + epsilon) \
										 * self.sym_db[i][k]
			# just for debugging
			step_size = np.amax(c / (np.sqrt(self.sum_sym_dW[0][0]) + epsilon)) 
		
		# method 4: momentum
		elif param.sgd_stepsize_mode == 'momentum':
			if step_size > 0.01:
				alpha = 0.5
			else:
				alpha = 0.99
			if (iteration % 200) == 0:
				step_size = step_size / 1.5
			for i in range(self.num_hidden_layers+1):
				# update sym_dW
				for j in range(len(self.multiagent_net_param.symmetric_indices[i])):
					self.sum_sym_dW[i][j] = alpha * self.sum_sym_dW[i][j] \
										- step_size * self.sym_dW[i][j]
					self.sym_W[i][j] += self.sum_sym_dW[i][j]
				# update sym_db
				for k in range(len(self.multiagent_net_param.symmetric_indices_b[i])):
					self.sum_sym_db[i][k] = alpha * self.sum_sym_db[i][k] \
										- step_size * self.sym_db[i][k]
					self.sym_b[i][k] += self.sum_sym_db[i][k]

		# method 5: rmsprop
		elif param.sgd_stepsize_mode == 'rmsprop':
			alpha = 0.9
			for i in range(self.num_hidden_layers+1):
				# update sym_dW
				for j in range(len(self.multiagent_net_param.symmetric_indices[i])):
					self.sum_sym_dW[i][j] = alpha * self.sum_sym_dW[i][j] + \
										(1-alpha) * np.square(self.sym_dW[i][j])
					self.sym_W[i][j] -= 0.01 * step_size * self.sym_dW[i][j] / \
									(np.sqrt(self.sum_sym_dW[i][j])+0.001)
				# update sym_db
				for k in range(len(self.multiagent_net_param.symmetric_indices_b[i])):
					self.sum_sym_db[i][k] = alpha * self.sum_sym_db[i][k] + \
										(1-alpha) * np.square(self.sym_db[i][k])
					self.sym_b[i][k] -= 0.01 * step_size * self.sym_db[i][k] / \
									(np.sqrt(self.sum_sym_db[i][k])+0.001)
		else:
			assert('unknown nerual network training type')
		return step_size




	def initialize_sum_of_grad(self):
		# for sum of grad
		self.sum_sym_dW = copy.deepcopy(self.sym_dW)
		self.sum_sym_db = copy.deepcopy(self.sym_db)
		for i in range(self.num_hidden_layers+1):
			# update sym_dW
			for j in range(len(self.multiagent_net_param.symmetric_indices[i])):
				self.sum_sym_dW[i][j][:] = 0
			# update sym_db
			for k in range(len(self.multiagent_net_param.symmetric_indices_b[i])):
				self.sum_sym_db[i][k][:] = 0


	def initialize_derivatives(self):
		self.dW = list()
		self.db = list()
		for i in range(self.num_hidden_layers+1):
			if self.multiagent_net_param.layers_type[i] == 'conn':
				layer_input_dim = self.layers_dim[i]
				layer_output_dim = self.layers_dim[i+1]
				self.dW.append(np.zeros((layer_input_dim, layer_output_dim)))
				self.db.append(np.zeros((1, layer_output_dim)))
			elif self.multiagent_net_param.layers_type[i] == 'max':
				self.dW.append([])
				self.db.append([])

	def set_training_param(self, nn_training_param):
		self.nn_training_param = nn_training_param


	# compute shifts to the x-y variables
	def compute_offset(self, X, Y, input_output_ranges):
		if input_output_ranges is None:
			self.avg_vec = np.mean(X, axis = 0)
			self.std_vec = np.std(X, axis = 0)
			self.output_avg_vec = np.mean(Y, axis = 0)
			self.output_std_vec = np.std(Y, axis = 0)
		else:
			self.avg_vec = input_output_ranges[0]
			self.std_vec = input_output_ranges[1]
			self.output_avg_vec = input_output_ranges[2]
			self.output_std_vec = input_output_ranges[3]

		# debug
		# print 'computing offset'
		# print 'avg_vec', self.avg_vec
		# print 'std_vec', self.std_vec
		# print 'out_avg_vec', self.output_avg_vec
		# print 'output_std_vec', self.output_std_vec

		# if input_output_ranges is not None:
		# 	avg_vec = input_output_ranges[0]
		# 	std_vec = input_output_ranges[1]
		# 	output_avg_vec = input_output_ranges[2]
		# 	output_std_vec = input_output_ranges[3]
		# 	print 'avg_vec', self.avg_vec - avg_vec
		# 	print 'std_vec', self.std_vec - std_vec
		# 	print 'out_avg_vec', self.output_avg_vec - output_avg_vec
		# 	print 'output_std_vec', self.output_std_vec - output_std_vec
		# raw_input()

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
	def backprop(self, X, Y, step_size, iteration):
		if_nn_nav = False
		# if X.shape[1] >= 7 + 8 and (X.shape[1] - 7 ) % 8 == 0:
		# 	if_nn_nav = True
		# 	num_other_agents = (X.shape[1] - 7 ) / 8
		# 	agent_off_indices = []
		# 	for i in range(1, num_other_agents+1):
		# 		inds = np.where(X[:,7+(8*i)-1] < 1e-5)[0]
		# 		agent_off_indices.append(inds)
		# 		assert(np.all(X[inds, 7+8*(i-1):7+(8*i)]==0))
				# print inds
				# raw_input()

		# training param
		param = self.nn_training_param
		# for forward/backward propogation
		nb_layers = self.num_hidden_layers + 1
		forward_prop_o = []
		backward_prop_xi = []
		for i in range(nb_layers):
			forward_prop_o.append(np.empty([1,1]))
			backward_prop_xi.append(np.empty([1,1]))

		batch_size = X.shape[0]

		out = X.copy()
		y_out = Y.copy()
		# one step back prop
		for layer in range(nb_layers-1):
			# RelU
			# print 'layer', layer
			# print 'self.W[layer].shape', self.W[layer].shape
			# print 'out', out.shape
			if self.multiagent_net_param.layers_type[layer] == 'conn':
				tmp = np.dot(out, self.W[layer]) \
					+ np.matlib.repmat(self.b[layer], batch_size, 1)
				forward_prop_o[layer] = tmp * (tmp>0)
			elif self.multiagent_net_param.layers_type[layer] == 'max':
				num_pts = out.shape[0]
				next_layer_size = np.sum(self.multiagent_net_param.layers_info[layer][:,1])
				forward_prop_o[layer] = np.zeros((num_pts, next_layer_size))
				cur_s_ind = 0
				next_s_ind = 0
				for ii in range(self.multiagent_net_param.layers_info[layer].shape[0]):
					num_agents = self.multiagent_net_param.layers_info[layer][ii,0]
					stride = self.multiagent_net_param.layers_info[layer][ii,1]
					cur_e_ind = cur_s_ind + num_agents * stride
					next_e_ind = next_s_ind + stride
					# print '---'
					# print out[:,cur_s_ind:cur_e_ind].shape
					# # print block_form.shape
					# print 'num_pts,', num_pts, 'stride', stride
					# print forward_prop_o[layer][:,next_s_ind:next_e_ind].shape
					block_form = np.reshape(out[:,cur_s_ind:cur_e_ind], (num_pts,-1,stride))


					forward_prop_o[layer][:,next_s_ind:next_e_ind] = \
						np.max(block_form, axis=1)
					cur_s_ind = cur_e_ind
					next_s_ind = next_e_ind

				# print 'layer', layer
				# print 'out', out
				# print 'forward_prop_o[layer]', forward_prop_o[layer]
				# raw_input()

					

			# for more than one agent
			# if if_nn_nav == True and self.multiagent_net_param.layers_info[layer+1].shape[0] == 2:
			# 	stride = self.multiagent_net_param.layers_info[layer+1][1,1]
			# 	start_ind = self.multiagent_net_param.layers_info[layer+1][0,1]
			# 	for tt in range(num_other_agents):
			# 		forward_prop_o[layer][agent_off_indices[tt],start_ind:start_ind+stride] = 0
			# 		start_ind += stride
					# raw_input()

			# dropout
			# p = 0.80
			# dropout_mask = (np.random.rand(*forward_prop_o[layer].shape) < p) / p
			# forward_prop_o[layer] *= dropout_mask

			out = forward_prop_o[layer].copy()

		# last layer, softmax
		# print 'y_out.shape', y_out.shape
		# print 'self.output_dim_weights', self.output_dim_weights
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

		for j in range(nb_layers-1, 0, -1):
			if self.multiagent_net_param.layers_type[j] == 'conn':
				# print 'j', j
				# print 'backward_prop_xi[j].shape', backward_prop_xi[j].shape
				# print 'forward_prop_o[j-1].transpose().shape', forward_prop_o[j-1].transpose().shape
				# print '(param.reg_lambda * self.W[j]).shape', (param.reg_lambda * self.W[j]).shape
				self.dW[j] = np.dot(forward_prop_o[j-1].transpose(), backward_prop_xi[j]) \
							+ param.reg_lambda * self.W[j] #/ (self.W[j].shape[0] * self.W[j].shape[1])
				# self.dW[j] = np.dot(forward_prop_o[j-1].transpose(), backward_prop_xi[j]) \
							# + param.reg_lambda * 0.1 * (self.W[j]>0) /  (self.W[j].shape[0] * self.W[j].shape[1])
				# self.dW[j] = np.dot(forward_prop_o[j-1].transpose(), backward_prop_xi[j]) \
				# 			+ param.reg_lambda * 0.1 * np.sign(self.W[j])
				self.db[j] = backward_prop_xi[j].sum(axis=0)
				# compute xi for previous layer and threshold at 0 if O is <0 (ReLU gradient update)
				backward_prop_xi[j-1] = numpy.dot(backward_prop_xi[j], self.W[j].transpose()) 
				backward_prop_xi[j-1] = backward_prop_xi[j-1] * (forward_prop_o[j-1]>0)
			
			elif self.multiagent_net_param.layers_type[j] == 'max':
				# compute xi for previous layer for max operator
				num_pts = backward_prop_xi[j].shape[0]
				prev_layer_size = np.sum(self.multiagent_net_param.layers_info[j][:,0] \
					* self.multiagent_net_param.layers_info[j][:,1])
				backward_prop_xi[j-1] = np.zeros((num_pts, np.sum(prev_layer_size)))
				cur_s_ind = 0
				prev_s_ind = 0
				for jj in range(self.multiagent_net_param.layers_info[j].shape[0]):
					num_agents = self.multiagent_net_param.layers_info[j][jj,0]
					stride = self.multiagent_net_param.layers_info[j][jj,1]
					cur_e_ind = cur_s_ind + stride
					for jjj in range(num_agents):
						prev_e_ind = prev_s_ind + stride
						# print backward_prop_xi[j-1][:,prev_s_ind:prev_e_ind].shape
						# print 'what', cur_s_ind, cur_e_ind, backward_prop_xi[j][:,cur_s_ind:cur_e_ind].shape
						# print forward_prop_o[j-1][:,prev_s_ind:prev_e_ind].shape
						# print 'how', cur_s_ind, cur_e_ind, forward_prop_o[j][:,cur_s_ind:cur_e_ind].shape
						backward_prop_xi[j-1][:,prev_s_ind:prev_e_ind] = \
							backward_prop_xi[j][:,cur_s_ind:cur_e_ind] * \
							(forward_prop_o[j-1][:,prev_s_ind:prev_e_ind] >= \
							(forward_prop_o[j][:,cur_s_ind:cur_e_ind]))
						prev_s_ind = prev_e_ind
					cur_s_ind = cur_e_ind

				# print 'forward_prop_o[j-1]', forward_prop_o[j-1]
				# print 'forward_prop_o[j]', forward_prop_o[j]
				# print 'backward_prop_xi[j-1]', backward_prop_xi[j-1]
				# print 'backward_prop_xi[j]', backward_prop_xi[j]
				# raw_input()
					
		self.dW[0] = np.dot(X.transpose(), backward_prop_xi[0]) \
						+ param.reg_lambda * self.W[0] #/ (self.W[0].shape[0] * self.W[0].shape[1])
		# self.dW[0] = np.dot(X.transpose(), backward_prop_xi[0]) \
		# 				+ param.reg_lambda * 0.1 * (self.W[0]>0) / (self.W[0].shape[0] * self.W[0].shape[1])
		self.db[0] = backward_prop_xi[0].sum(axis=0)

		# update symmetirc_db
		self.dW_2_symIndices()

		#### subgradient updates
		step_size = self.update_symIndices(param, step_size, iteration)
		self.symIndices_2_mat()
		
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
		for i in range(param.nb_iter):
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
				if self.plotting_func is not None and self.X_vis is not None:
					title_string = 'iter %d' % i
					figure_name = 'training'
					Y_vis = self.make_prediction_raw(self.X_vis)
					self.plotting_func(self.X_vis, Y_vis, title_string, figure_name=figure_name)
				if dataset_test is not None:
					X_test = dataset_test[0]
					Y_test = dataset_test[1]
					nb_test_ex = X_test.shape[0]
					X_test = self.xRaw_2_x(X_test)
					Y_test = self.yRaw_2_y(Y_test)
					z_test, z_sq_test = self.evaluate_network_loss(X_test, Y_test)
					print('Test discrete error: %f, test square loss: %f ' % (z_test, z_sq_test))
				print(' ')

		print('checking symmetry condition')
		self.debug_symmemtric(X)



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
			X = X[np.newaxis,:]

		if_nn_nav = False
		# if X.shape[1] >= 7 + 8 and (X.shape[1] - 7 ) % 8 == 0:
		# 	if_nn_nav = True
		# 	num_other_agents = (X.shape[1] - 7 ) / 8
		# 	agent_off_indices = []
		# 	for i in range(1, num_other_agents+1):
		# 		inds = np.where(X[:,7+(8*i)-1] < 1e-5)[0]
		# 		agent_off_indices.append(inds)
		# 		try:
		# 			assert(np.all(X[inds, 7+8*(i-1):7+(8*i)]==0))
		# 		except AssertionError:
		# 			print inds
		# 			print X[inds, 7+8*(i-1):7+(8*i)]
		# 			assert(0)

		nb_layers = self.num_hidden_layers + 1
		out = X
		for layer in range(nb_layers-1):
			if self.multiagent_net_param.layers_type[layer] == 'conn':
				tmp = np.dot(out, self.W[layer]) \
					+ np.matlib.repmat(self.b[layer], nb_examples, 1)
				out = tmp * (tmp>0)
			elif self.multiagent_net_param.layers_type[layer] == 'max':
				num_pts = out.shape[0]
				next_layer_size = np.sum(self.multiagent_net_param.layers_info[layer][:,1])
				out_next = np.zeros((num_pts, np.sum(next_layer_size)))
				if num_pts == 0:
					out = out_next
					continue
				cur_s_ind = 0
				next_s_ind = 0
				for ii in range(self.multiagent_net_param.layers_info[layer].shape[0]):
					num_agents = self.multiagent_net_param.layers_info[layer][ii,0]
					stride = self.multiagent_net_param.layers_info[layer][ii,1]
					cur_e_ind = cur_s_ind + num_agents * stride
					next_e_ind = next_s_ind + stride
					# print '---'
					# print out[:,cur_s_ind:cur_e_ind].shape
					# # print block_form.shape
					# print 'num_pts,', num_pts
					# print out_next[:,next_s_ind:next_e_ind].shape
					block_form = np.reshape(out[:,cur_s_ind:cur_e_ind], (num_pts,-1,stride))
					# print out[:,cur_s_ind:cur_e_ind].shape
					# print block_form.shape
					# print 'num_pts,', num_pts
					# print forward_prop_o[layer][:,next_s_ind:next_e_ind].shape
					out_next[:,next_s_ind:next_e_ind] = \
						np.max(block_form, axis=1)
					cur_s_ind = cur_e_ind
					next_s_ind = next_e_ind

					# print 'layer', layer
					# print 'out', out
					# print 'out_next', out_next

				out = out_next
				# raw_input()

			# if if_nn_nav == True and self.multiagent_net_param.layers_info[layer+1].shape[0] == 2:
			# 	stride = self.multiagent_net_param.layers_info[layer+1][1,1]
			# 	start_ind = self.multiagent_net_param.layers_info[layer+1][0,1]
			# 	for tt in range(num_other_agents):
			# 		out[agent_off_indices[tt],start_ind:start_ind+stride] = 0
			# 		start_ind += stride



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

	# debug, test whether network is symmentry
	def debug_symmemtric(self, X_raw):
		Y_nominal = self.make_prediction_raw(X_raw)
		# preturb input
		layer_info = self.multiagent_net_param.layers_info[0]
		
		num_perturbations = 10
		for perturb in range(num_perturbations):
			# generate perturbation
			start_ind = 0
			X_raw_cp = X_raw.copy()
			for i in range(layer_info.shape[0]):
				num_type = layer_info[i,0]
				stride = layer_info[i,1]
				if num_type > 1:
					other_ind = min(1, np.random.randint(num_type-1)+1)
					# print 'layer_info', layer_info
					# print 'i, num_type, stride', i, num_type, stride
					# print 'start_ind', start_ind
					# print 'other_ind', other_ind
					X_raw_cp[:,start_ind:start_ind+stride] = \
						X_raw[:,start_ind+other_ind*stride:start_ind+(other_ind+1)*stride]
					X_raw_cp[:,start_ind+other_ind*stride:start_ind+(other_ind+1)*stride] = \
						X_raw[:,start_ind:start_ind+stride]
					# debug
					# X_diff = X_raw_cp - X_raw
					# print 'X_diff[1,:]', X_diff[1,:]
				Y_hat = self.make_prediction_raw(X_raw_cp)
				try:
					assert(np.linalg.norm(Y_hat - Y_nominal)<1e-6)
				except AssertionError:
					print('symmetric condition not met')
					print('X_raw, Y_nominal', X_raw, Y_nominal)
					print('X_raw_cp, Y_raw', X_raw_cp, Y_hat)
					assert(0)
				# update start_ind
				start_ind += num_type * stride

		print('passed %d random cases' % num_perturbations)


if __name__ == '__main__':
	print('hello world from neural_network.py')
	file_dir = os.path.dirname(os.path.realpath(__file__))
	plt.rcParams.update({'font.size': 18})

	''' test on the spiral dataset '''
	dataset_name = "/sinusoid_sum_1out";
	sinusoid_dataset = pickle.load(open(file_dir+"/test_data" + dataset_name + "_dataset_train.p","rb"))
	sinusoid_sum_X = sinusoid_dataset[0]
	sinusoid_sum_Y = sinusoid_dataset[1]
	sinusoid_sum_X_vis = pickle.load(open(file_dir+"/test_data" + dataset_name + "_dataset_vis.p", "rb"))

	print('sinusoid_sum_X.shape', sinusoid_sum_X.shape)
	print('sinusoid_sum_Y.shape', sinusoid_sum_Y.shape)

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

	# note: layers_info must be compatible with hidden_layers_size
	layers_info = []
	layers_type = []
	layers_info.append(np.array([[2, 1]])); layers_type.append('conn')
	layers_info.append(np.array([[2, 50]])); layers_type.append('conn')
	# layers_info.append(np.array([[2, 50]])); layers_type.append('conn')
	layers_info.append(np.array([[2, 50]])); layers_type.append('max')
	layers_info.append(np.array([[1, 50]])); layers_type.append('conn')

	layers_info.append(np.array([[1, 100]])); layers_type.append('conn')
	layers_info.append(np.array([[1, 1]])); layers_type.append('conn')

	neural_network_regr_multi = Neural_network_regr_multi(nn_training_param)
	neural_network_regr_multi.initialize_network_param(layers_info, layers_type)
	neural_network_regr_multi.set_plotting_func(\
		generate_symmetric_sinusoids.plot_sinusoid_dataset, sinusoid_sum_X_vis)

	''' training the neural network '''
	ERM = 0
	generate_symmetric_sinusoids.plot_sinusoid_dataset(sinusoid_sum_X, sinusoid_sum_Y, 'training data')
	# method 1
	# neural_network_regr_multi.set_training_stepsize('fixed_decay')
	# method 2
	# neural_network_regr_multi.set_training_stepsize('sqrt_decay', 0.5, 0.1)
	# method 3
	# neural_network_regr_multi.set_training_stepsize('sum_of_grad', 0.1, 0.1)
	# method 4
	# neural_network_regr_multi.set_training_stepsize('momentum', 0.1, 0.1)
	# method 5
	neural_network_regr_multi.set_training_stepsize('rmsprop', 0.1, 0.1)
	# training
	neural_network_regr_multi.set_plotting_func(\
		generate_symmetric_sinusoids.plot_sinusoid_dataset, sinusoid_sum_X_vis)
	neural_network_regr_multi.train_nn(sinusoid_dataset, ERM=ERM)

	''' make prediction '''
	sinusoid_sum_Y_vis = neural_network_regr_multi.make_prediction_raw(sinusoid_sum_X_vis)
	generate_symmetric_sinusoids.plot_sinusoid_dataset(sinusoid_sum_X_vis, sinusoid_sum_Y_vis, 'regression')
	generate_symmetric_sinusoids.plot_sinusoid_dataset_compare(\
		sinusoid_sum_X, sinusoid_sum_Y, sinusoid_sum_X_vis, sinusoid_sum_Y_vis)
	''' debugging '''
	neural_network_regr_multi.debug_symmemtric(sinusoid_sum_X_vis)
	plt.show()

