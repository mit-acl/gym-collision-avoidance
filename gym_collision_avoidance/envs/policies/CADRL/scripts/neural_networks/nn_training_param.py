class NN_training_param:
	# sgd_step_size: initial eta (should decay as a function of time)
	# reg_lambda: regularization parameter
	# nb_iter: number of training iterations
	# sgd_batch_size: batch size of each stochastic gradient descent step
	# w_scale: parameter for initializing the neural network
	def __init__(self, sgd_step_size, reg_lambda, nb_iter, sgd_batch_size, w_scale, \
				sgd_stepsize_mode='fixed_decay', sgd_step_c=0.1, sgd_step_epsilon=0.1):
		self.sgd_step_size = sgd_step_size 		
		self.reg_lambda = reg_lambda			 
		self.nb_iter = nb_iter				
		self.sgd_batch_size = sgd_batch_size
		self.w_scale = w_scale
		self.sgd_stepsize_mode = sgd_stepsize_mode
		self.sgd_step_c = sgd_step_c
		self.sgd_step_epsilon = sgd_step_epsilon

	def writeToFile(filename):
		np_array = []
		np_array.append(self.sgd_step_size)
		np_array.append(self.reg_lambda)
		np_array.append(self.nb_iter)				
		np_array.append(self.sgd_batch_size)
		np_array.append(self.w_scale)
		np_array.append(self.sgd_stepsize_mode)
		np_array.append(self.sgd_step_c)
		np_array.append(self.sgd_step_epsilon)
		pickle.dump(np_array, open(filename, "wb"))
		return

	def loadFromFile(filename):
		np_array = pickle.load(open(filename, "rb"))
		self.sgd_step_size = np_array[0]	
		self.reg_lambda = np_array[1] 
		self.nb_iter = np_array[2]			
		self.sgd_batch_size = np_array[3]
		self.w_scale = np_array[4]
		self.sgd_stepsize_mode = np_array[5]
		self.sgd_step_c = np_array[6]
		self.sgd_step_epsilon = np_array[7]
		return