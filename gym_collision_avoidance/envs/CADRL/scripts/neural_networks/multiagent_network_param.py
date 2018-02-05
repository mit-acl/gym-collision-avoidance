import numpy as np

EPS = 1e-6

# layers_info = [layer_info_layer1, layer_info_layer2, ...]
# layer_info_layer1 (type, numpy.array([[num_type_1, num_states], \
#                                       [num_type_2, num_states], ... ])

class Multiagent_network_param:
	# layers_info
	# symmetric_indices
	def __init__(self, layers_info, layers_type, symmetric_indices=None, symmetric_indices_b=None):
		self.layers_info = layers_info
		self.layers_type = layers_type
		if symmetric_indices != None:
			self.symmetric_indices = symmetric_indices
			assert(symmetric_indices_b != None)
			self.symmetric_indices_b = symmetric_indices_b
		else:
			self.symmetric_indices, self.symmetric_indices_b, layers_type_post = \
				self.compute_default_indices(layers_info, layers_type)
			self.layers_type = layers_type_post


		# self.layers_type = layers_type_post
		# for layer_type in self.layers_type:
		# 	print layer_type
		# raw_input()

	def compute_default_indices(self, layers_info, layers_type):
		symmetric_indices = []
		symmetric_indices_b = []
		num_layers = len(layers_info)

		# layers before the fully connected layer
		for i in xrange(num_layers):
			layer_info = layers_info[i]
			next_layer_info = layers_info[i+1]
			# begin fully connected layers
			if next_layer_info.shape[0] == 1 and next_layer_info[0,0] == 1 and layers_type[i]!='max':
				# print next_layer_info.shape
				# print next_layer_info
				fully_conn_layer_number = i
				break
			if layers_type[i] == 'conn':
				symmetric_indices_layer, symmetric_indices_layer_b = \
					self.compute_default_indices_layer(layer_info, next_layer_info)
				symmetric_indices.append(symmetric_indices_layer) 
				symmetric_indices_b.append(symmetric_indices_layer_b)
			elif layers_type[i] == 'self':
				symmetric_indices_layer, symmetric_indices_layer_b = \
					self.compute_default_indices_layer(layer_info, next_layer_info, self_only=True)
				symmetric_indices.append(symmetric_indices_layer) 
				symmetric_indices_b.append(symmetric_indices_layer_b)
				layers_type[i] = 'conn'
			elif layers_type[i] == 'max':
				symmetric_indices.append([]) 
				symmetric_indices_b.append([])

			
		# transition to fully connected layer
		# print fully_conn_layer_number
		symmetric_indices_layer = []
		symmetric_indices_layer_b = []
		symmetric_indices_layer_b.append(np.array([[0, \
			layers_info[fully_conn_layer_number+1][0,1]]], dtype=np.int32))
		layer_info = layers_info[fully_conn_layer_number]
		num_types = layer_info.shape[0]
		layer_start_ind = 0 
		for i in xrange(num_types):
			num_agents_of_type_i = layer_info[i, 0]
			str_cur = layer_info[i, 1]
			sym_inds = np.zeros((num_agents_of_type_i, 4), dtype=np.int32)
			for ii in xrange(num_agents_of_type_i):
				s_ind = layer_start_ind + ii * str_cur
				e_ind = layer_start_ind + (ii+1) * str_cur
				next_s_ind = 0
				next_e_ind = layers_info[fully_conn_layer_number+1][0,1]
				sym_inds[ii,:] = np.array([s_ind, e_ind, next_s_ind, next_e_ind])
			symmetric_indices_layer.append(sym_inds)
			layer_start_ind += layer_info[i,0] * layer_info[i,1]
		symmetric_indices.append(symmetric_indices_layer)
		symmetric_indices_b.append(symmetric_indices_layer_b)
		# print symmetric_indices_b

		# fully connected layers
		for i in xrange(fully_conn_layer_number+1, num_layers-1):
			layer_info = layers_info[i]
			next_layer_info = layers_info[i+1]
			assert(layer_info.shape[0] == 1 and layer_info[0,0] == 1)
			assert(next_layer_info.shape[0] == 1 and next_layer_info[0,0] == 1)
			symmetric_indices_layer, symmetric_indices_layer_b = \
				self.compute_default_indices_layer(layer_info, next_layer_info)
			symmetric_indices.append(symmetric_indices_layer) 
			symmetric_indices_b.append(symmetric_indices_layer_b) 

		return symmetric_indices, symmetric_indices_b, layers_type

	def compute_default_indices_layer(self, layer_info, next_layer_info, self_only=False):
		assert(layer_info.shape[0] == next_layer_info.shape[0])
		# must have equal number of agents per layer
		for i in xrange(len(layer_info)):
			try:
				assert(layer_info[i,0] == next_layer_info[i,0])
			except AssertionError:
				print('num_agents dont match')
				print('layer_info', layer_info)
				print('next_layer_info', next_layer_info)
				assert(0)
		symmetric_indices = []
		symmetric_indices_b = []
		
		num_types = layer_info.shape[0]
		
		# within group
		layer_start_ind = 0
		next_layer_start_ind = 0
		for i in xrange(num_types):
			assert(layer_info[i,0] > 0)
			assert(layer_info[i,1] > 0)
			num_agents_of_type_i = layer_info[i, 0]
			# winthin_group, self_inds
			self_inds = np.zeros((num_agents_of_type_i, 4), dtype=np.int32)
			self_inds_b = np.zeros((num_agents_of_type_i, 2), dtype=np.int32)
			str_cur = layer_info[i,1]
			str_next = next_layer_info[i,1]
			for ii in xrange(num_agents_of_type_i):
				s_ind = layer_start_ind + ii * str_cur
				e_ind = layer_start_ind + (ii+1) * str_cur
				next_s_ind = next_layer_start_ind + ii * str_next
				next_e_ind = next_layer_start_ind + (ii+1) * str_next
				self_inds[ii,:] = np.array([s_ind, e_ind, next_s_ind, next_e_ind])
				self_inds_b[ii,:] = np.array([next_s_ind, next_e_ind])
			symmetric_indices.append(self_inds)
			symmetric_indices_b.append(self_inds_b)
			# print 'self_inds', self_inds
			# raw_input()
			
			# within group, self_other_inds
			if num_agents_of_type_i > 1:
				self_other_inds = np.zeros((num_agents_of_type_i * \
					(num_agents_of_type_i - 1),4), dtype=np.int32)
				counter = 0
				for ii in xrange(num_agents_of_type_i):
					for jj in xrange(num_agents_of_type_i):
						if jj == ii:
							continue
						s_ind = layer_start_ind + ii * str_cur
						e_ind = layer_start_ind + (ii+1) * str_cur
						next_s_ind = next_layer_start_ind + jj * str_next
						next_e_ind = next_layer_start_ind + (jj+1) * str_next
						self_other_inds[counter,:] = \
							np.array([s_ind, e_ind, next_s_ind, next_e_ind])
						counter += 1
				symmetric_indices.append(self_other_inds)
				# print 'self_other_inds', self_other_inds
				# raw_input()

			layer_start_ind += layer_info[i,0] * layer_info[i,1]
			next_layer_start_ind += next_layer_info[i,0] * next_layer_info[i,1]

		# with other group
		if num_types >= 1 and self_only == False:
			layer_start_ind = 0
			for i in xrange(num_types):
				str_cur = layer_info[i,1]
				next_layer_start_ind = 0
				for j in xrange(num_types):
					if j > 0:
						next_layer_start_ind += next_layer_info[j-1,0] * next_layer_info[j-1,1]
					if i == j:
						continue
					str_next = next_layer_info[j,1]
					num_agent_of_type_i = layer_info[i,0]
					num_agent_of_type_j = next_layer_info[j,0]

					# initialize other_sym_inds
					other_sym_inds = np.zeros((num_agent_of_type_i*num_agent_of_type_j,4), dtype=np.int32)
					counter = 0
					for ii in xrange(num_agent_of_type_i):
						for jj in xrange(num_agent_of_type_j):
							s_ind = layer_start_ind + ii * str_cur
							e_ind = layer_start_ind + (ii+1) * str_cur
							next_s_ind = next_layer_start_ind + jj * str_next
							next_e_ind = next_layer_start_ind + (jj+1) * str_next
							other_sym_inds[counter,:] = \
								np.array([s_ind, e_ind, next_s_ind, next_e_ind])
							counter += 1
					symmetric_indices.append(other_sym_inds)
					# print next_layer_info
					# print str_next
					# print i,j,'other_sym_inds', other_sym_inds
					# print next_layer_start_ind
					# raw_input()
				
				layer_start_ind += layer_info[i,0] * layer_info[i,1]
		# print 'hello', len(symmetric_indices)
		# print symmetric_indices
		# print symmetric_indices_b
		# raw_input()

		return symmetric_indices, symmetric_indices_b

	def print_symmetricIndices(self):
		for i, symmetric_indices_layer in enumerate(self.symmetric_indices):
			print(' ===== ===== layer %d ===== type %s =====' % (i, self.layers_type[i]))
			for j, block in enumerate(symmetric_indices_layer):
				print(' ~~~ W: layer %d, type %d ~~~' % (i,j))
				print(block)
			print(' ~~~ b: layer %d ~~~' % i)
			print(self.symmetric_indices_b[i])

	def check_valid_symmetricIndices(self):
		for i, symmetric_indices_layer in enumerate(self.symmetric_indices):
			# require every group to have the same dimension
			for group in symmetric_indices_layer:
				dx = group[:,1] - group[:,0]
				dy = group[:,3] - group[:,2]
				dx[:] -= dx[0]
				dy[:] -= dy[0]
				assert(np.linalg.norm(dx)<EPS and np.linalg.norm(dy)<EPS)

			# (TODO) don't allow overlap
		print('passed check_valid_symmetricIndices')
		return



if __name__ == '__main__':
	print('hello world from multiagent_network_param.py')

	layers_info = []
	layers_info.append(np.array([[1,2],[2,1]]))
	layers_info.append(np.array([[1,3],[2,2]]))
	layers_info.append(np.array([[1,1]]))
	mn_param = Multiagent_network_param(layers_info)
	mn_param.print_symmetricIndices()
	mn_param.check_valid_symmetricIndices()


