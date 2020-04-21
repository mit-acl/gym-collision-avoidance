#!/usr/bin/env python
import sys
import os
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir+'/../neural_networks')

import numpy as np
import numpy.matlib
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import copy
import time

import neural_network_regr_multi as nn
import nn_navigation_value_multi as nn_nav
import pedData_processing_multi as pedData
import global_var as gb
import gen_rand_testcases as gen_tc

# setting up global variables
COLLISION_COST = gb.COLLISION_COST
DIST_2_GOAL_THRES = gb.DIST_2_GOAL_THRES
GETTING_CLOSE_PENALTY = gb.GETTING_CLOSE_PENALTY
GETTING_CLOSE_RANGE = gb.GETTING_CLOSE_RANGE
EPS = gb.EPS
# terminal states
NON_TERMINAL=gb.NON_TERMINAL
COLLIDED=gb.COLLIDED
REACHED_GOAL=gb.REACHED_GOAL
# plotting colors
plt_colors = gb.plt_colors
GAMMA = gb.RL_gamma
DT_NORMAL = gb.RL_dt_normal
TRAINING_DT = gb.TRAINING_DT

def compute_plot_stats(traj_raw_multi):
	time_to_reach_goal, traj_lengths, min_sepDist, if_completed_vec \
		= pedData.computeStats(traj_raw_multi)
	num_agents = len(traj_raw_multi) - 1
	agents_speed = np.zeros((num_agents,))
	for i in xrange(num_agents):
		agents_speed[i] = traj_raw_multi[i+1][0,5]
	agents_time = time_to_reach_goal
	agents_len = traj_lengths
	min_dist = min_sepDist
	return agents_speed, agents_time, agents_len, min_dist


class NN_rl_training_param:
	# sgd_step_size: initial eta (should decay as a function of time)
	# reg_lambda: regularization parameter
	# nb_iter: number of training iterations
	# sgd_batch_size: batch size of each stochastic gradient descent step
	# w_scale: parameter for initializing the neural network
	def __init__(self, num_episodes, numpts_per_eps, expr_size, \
				gamma, sgd_batch_size, greedy_epsilon):
		self.num_episodes = num_episodes
		self.numpts_per_eps = numpts_per_eps
		self.expr_size = expr_size
		self.gamma = gamma
		self.sgd_batch_size = sgd_batch_size
		self.greedy_epsilon = greedy_epsilon 		

	def writeToFile(filename):
		np_array = []
		np_array.append(self.num_episodes)
		np_array.append(self.numpts_per_eps)				
		np_array.append(self.expr_size)
		np_array.append(self.gamma)
		np_array.append(self.sgd_batch_size)
		np_array.append(self.greedy_epsilon)
		pickle.dump(np_array, open(filename, "wb"))
		return

	def loadFromFile(filename):
		np_array = pickle.load(open(filename, "rb"))
		self.num_episodes = np_array[0]
		self.numpts_per_eps = np_array[1]
		self.expr_size = np_array[2]
		self.gamma = np_array[3]
		self.sgd_batch_size = np_array[4]
		self.greedy_epsilon = np_array[5]		
		return

class NN_rl:
	def __init__(self, nn_rl_training_param, nn_training_param, value_net, ifSave):
		self.nn_rl_training_param = nn_rl_training_param
		self.nn_training_param = nn_training_param
		self.training_score = []
		self.bad_testcases = []; self.bad_testcases_tmp = []; self.bad_testcases_update_iter = []
		self.eval_epsd_stride = 5
		self.test_cases = preset_testCases()
		self.value_net = value_net
		self.value_net_copy = copy.deepcopy(value_net)
		self.old_value_net = copy.deepcopy(value_net)
		self.best_value_net = copy.deepcopy(value_net)
		self.ifSave = ifSave
		self.passing_side = 'none'
		self.mode = self.value_net.mode+'_'
		self.epsilon_use_other_net = 0.3
		self.value_net_other = None
		self.num_agents = value_net.num_agents
		pass

	def writeToFile(self, file_dir, iteration):
		v_net_fname = file_dir+"/../../pickle_files/multi/" + self.value_net.mode \
			+ '_' + self.passing_side \
			+ "/RL_selfplay/%d_agents_policy_iter_"%self.num_agents + str(iteration) + ".p"
		score_fname = file_dir+"/../../pickle_files/multi/" + self.value_net.mode \
			+ '_' + self.passing_side  \
			+ "/%d_agents_RL_training_score.p"%self.num_agents
		if self.ifSave:
			self.value_net.nn.save_neural_network(v_net_fname)
			pickle.dump(self.training_score, open(score_fname, "wb"))
		pass

	def loadFromFile(self, file_dir, v_net_filename):
		filename_nn =  file_dir+"/../../pickle_files/multi/" + self.value_net.mode \
			+ '_' + self.passing_side \
			+ "/RL_selfplay/" + v_net_filename
		self.value_net.nn.load_neural_network(filename_nn)
		self.value_net_copy.nn.load_neural_network(filename_nn)
		score_fname = file_dir+"/../../pickle_files/multi/" + self.value_net.mode \
			+ '_' + self.passing_side + "/%d_agents_RL_training_score.p"%self.num_agents
		try:
			self.scores = pickle.load(open(score_fname,"rb"))
		except:
			print('no score file exists')
		pass

	def loadOldNet(self, file_dir, iteration):
		v_net_fname = file_dir+"/../../pickle_files/multi/" + self.value_net.mode \
			+ '_' + self.passing_side \
			+ "/RL_selfplay/%d_agents_policy_iter_"%self.num_agents + str(max(0,iteration-100)) + ".p"
		self.old_value_net.nn.load_neural_network(v_net_fname)
		self.value_net.old_value_net = self.old_value_net

	def deep_RL_train(self, file_dir):
		t_start = time.time()
		self.training_score = []
		param = self.nn_rl_training_param
		self.value_net.nn.initialize_derivatives()
		self.value_net.old_value_net = self.old_value_net

		# initialize experience
		num_states = 7 + 8 * (self.num_agents - 1)
		self.X = np.zeros((param.expr_size,num_states))
		self.Y = np.zeros((param.expr_size,1))
		self.values_diff = np.zeros((param.expr_size,))
		self.current_expr_ind = 0
		self.total_training_pts = 0
		path_times = None
		collisions = None
		best_iter_time = np.inf
		best_iter = 0

		# for each episode
		for kk in xrange(self.nn_rl_training_param.num_episodes):
			numpts_cur_eps = 0
			epsilon = 0.3 - np.amin((0.25, kk / 2500.0))
			self.value_net.dt_forward = 1.0 #- np.amin((0.7, kk / 150.0))
			self.value_net_copy.dt_forward = 1.0 #- np.amin((0.7, kk / 150.0))
			self.value_net.radius_buffer = 0.0
			self.value_net_copy.radius_buffer = 0.0
			# self.value_net.passing_side_training_weight = 0.2 + np.amin((0.5, kk / 500.0))
			# side_length = np.amin((6, 1.0 + kk / 50.0))
			
			# if kk > 300:
			# if kk % 2 == 0:
			side_length = np.random.rand() * 4.0 + 3.0
			# else:
				# side_length = np.random.rand() * 2.0 + 1.0


			# evaluate network
			if kk % self.eval_epsd_stride == 0:
				self.value_net.radius_buffer = 0.0
				self.value_net_copy.radius_buffer = 0.0
				path_times, collisions, values = \
					self.evaluate_current_network(path_times, collisions, iteration=kk, plot_mode='one')
				# score = np.array([np.sum(path_times), np.sum(collisions)])
				score = np.hstack((path_times, collisions, values))
				self.training_score.append(score)
				num_cases = len(self.test_cases)
				print('time: %.2f, epsd: %d, time: %.3f, value: %.3f, num_bad_cases: %.2f, best_iter %d' % (time.time()-t_start, kk, \
					np.sum(score[0:num_cases]), np.sum(score[2*num_cases:3*num_cases]), len(self.bad_testcases), best_iter))

				# plot a test case
				if kk > 0 and self.current_expr_ind > 0:
					ind = np.random.randint(0, np.max((1,self.current_expr_ind)))
					x_plot = self.X[ind,:]
					y_plot = self.Y[ind,:]
					title_string = 'epsd: %d, time: %.1f, value: %.3f' % \
						(kk, np.sum(score[0:num_cases]), np.sum(score[2*num_cases:3*num_cases]))
					self.value_net.plot_ped_testCase(x_plot, y_plot, title_string, \
						'test_case in RL self play')
					# plot a training traj
					agents_speed, agents_time, agents_len, min_dist = compute_plot_stats(traj_raw_multi)
					title_string = 'a%d, t %.2f, sp %.2f, len %.2f \n %s; min_dist %.2f a%d t %.2f, sp %.2f, len %.2f' % \
						(0, agents_time[0], agents_speed[0], agents_len[0], \
							self.passing_side, min_dist, 1, agents_time[1], agents_speed[1], agents_len[1])
					num_agents = len(traj_raw_multi) - 1
					if num_agents > 2:
						for tt in xrange(2, num_agents):
							agent_string = '\n a%d, t %.2f, sp %.2f, len %.2f' % \
								(tt, agents_time[tt], agents_speed[tt], agents_len[tt])
							title_string += agent_string
					pedData.plot_traj_raw_multi(traj_raw_multi, title_string, figure_name=self.mode+'training_traj' )

			# reset value_net_copy to value_net
			if kk % 5 == 0:
				# cur_iter_time = np.sum(score[0:num_cases])
				# # print best_iter_time, best_iter_time
				# if best_iter_time > cur_iter_time:
				# 	self.best_value_net = copy.deepcopy(self.value_net)
				# 	best_iter_time = cur_iter_time
				# 	best_iter = kk
					# print 'recorded change at iteration', kk
				
				self.value_net_copy = copy.deepcopy(self.value_net)

			# if kk % 50 == 0:
			# 	self.value_net = copy.deepcopy(self.best_value_net)

			
			# raw_input()
			# save 
			if kk % 50 == 0:
				self.writeToFile(file_dir, kk)
				# self.loadOldNet(file_dir, kk)
				self.plot_training_score(file_dir)

			# for stats
			strt_line_training_pts = 0
			nn_training_pts = 0

			if kk < 200:
				step_size = 1.0 / max(40+kk, kk)
			else:
				step_size = 1.0 / (2000+int(kk/1000)*1000)

			while (numpts_cur_eps < param.numpts_per_eps):
				is_permit_straight = np.random.binomial(1, 0.0)
				is_overtake = np.random.binomial(1, 0.2)
				# is_permit_straight = False
				num_agents = self.num_agents

				if_static = np.random.rand() < 0.2
				# if kk < 200:
				# 	if_static = True
				if_end_near_bnd = np.random.rand() < 0.2

				# train on bad cases
				if_trained_on_badcases = False
				if np.random.rand() < 0.5 and len(self.bad_testcases) > 0:
					bad_case_ind = np.random.randint(len(self.bad_testcases))
					if self.bad_testcases_update_iter[bad_case_ind] < kk - 1:
					# if True:
						if_trained_on_badcases = True
						self.bad_testcases_update_iter[bad_case_ind] = kk
						agents_state = self.bad_testcases[bad_case_ind]
						num_repeat = 2
						traj_raw_multi, x, y, values_diff, \
							if_resolved = self.trainTestCase(agents_state, num_repeat)
						if if_resolved == True or np.random.rand() > 0.8:
							self.bad_testcases.pop(bad_case_ind)	
							self.bad_testcases_update_iter.pop(bad_case_ind)			
						self.bad_testcases_tmp = []
						# print 'bad test case with %d /%d pts' % (len(x), len(x) + numpts_cur_eps)
						
						if len(x) > 0:
							x_train = self.value_net.nn.xRaw_2_x(x)
							y_train = self.value_net.nn.yRaw_2_y(y)
							# step_size = 1.0 / max(2000+kk, kk)
							self.value_net.nn.set_training_stepsize('rmsprop')
							self.value_net.nn.backprop(x_train, y_train, step_size, kk)
							# print 'after len(self.bad_testcases)', len(self.bad_testcases)
				
				# train on random cases
				if if_trained_on_badcases == False: 
					test_case = gen_tc.generate_rand_test_case_multi(num_agents, side_length, \
						np.array([0.1,1.2]), np.array([0.3, 0.5]), \
						is_end_near_bnd=if_end_near_bnd, is_static = if_static)
					# debugging
					# if np.random.rand() > 0.0: #0.5:
					# 	test_case = self.test_cases[np.random.randint(4)]
					# 	test_case = self.test_cases[1]


					# print 'self.value_net.dt_forward', self.value_net.dt_forward
					x = []; y = [];

					if len(x) == 0:
						ifRandHeading = np.random.binomial(1, 0.3)
						# ifRandHeading = False

						traj_raw_multi, time_to_complete = \
							self.value_net.generate_traj(test_case, rl_epsilon=epsilon, \
								figure_name='no_plot', stopOnCollision=True, ifRandHeading=ifRandHeading,\
								ifNonCoop=True)
						num_pts = len(traj_raw_multi[0])
						if num_pts < 2:
							continue

						# print 'generate traj test case'
						# pedData.plot_traj_raw_multi(traj_raw, 'what is wrong', figure_name='tmp_traj' )
						x, y, values_diff = self.rawTraj_2_trainingData(traj_raw_multi, param.gamma, kk)
						nn_training_pts += len(x)
						if np.random.rand() > 0.9:
							traj_raw = pedData.reflectTraj(traj_raw_multi)

						agents_speed, agents_time, agents_len, min_dist = compute_plot_stats(traj_raw_multi)


						
						if len(self.bad_testcases_tmp) > 0:
							if len(self.bad_testcases) < 50:
								self.bad_testcases += self.bad_testcases_tmp
								self.bad_testcases_update_iter += [kk-1] * len(self.bad_testcases_tmp)
						self.bad_testcases_tmp = []
						# print 'rand test case with %d /%d pts' % (len(x), len(x) + numpts_cur_eps)
				if len(x) > 0:
					self.append_to_experience(x, y, values_diff, param.expr_size)
					numpts_cur_eps += len(x)
				# print 'numpts_cur_eps', numpts_cur_eps

			# train the value network
			for tt in xrange(2):
				# sample a random minibatch
				nb_examples = min(self.total_training_pts, param.expr_size)
				# half good and half bad
				if np.random.rand() > 1.1:
					minibatch = np.random.permutation(np.arange(nb_examples))[0:param.sgd_batch_size*2]

					# bad_inds = np.where(self.values_diff>0.05)[0]
					# half_num = param.sgd_batch_size/2
					# if len(bad_inds) > half_num:
					# 	minibatch_bad = np.argpartition(self.values_diff, -half_num)[-half_num:]
					# 	minibatch_rand = np.random.permutation(np.arange(nb_examples))[0:half_num:]
					# 	minibatch = np.union1d(minibatch_bad, minibatch_rand)
					# else:
					# 	minibatch = bad_inds
					# 	print 'here'
					values_raw = np.squeeze(self.value_net_copy.nn.make_prediction_raw(self.X[:nb_examples,:]))
					values_diff = abs((values_raw - np.squeeze(self.Y[:nb_examples]))\
						/np.squeeze(self.Y[:nb_examples]))
					half_num = param.sgd_batch_size / 2.0
					minibatch_bad = np.argpartition(values_diff, -half_num)[-half_num:]
					# np.set_printoptions(edgeitems=4, precision=4,formatter={'float': '{: 0.4f}'.format})
					# print 'max', values_diff[minibatch_bad]
					# print 'dist', self.X[minibatch_bad,0:7]
					# raw_input()
					# print 'rand', values_diff[0:nb_examples]
					# raw_input()
					minibatch = minibatch_bad
					minibatch_rand = np.random.permutation(np.arange(nb_examples))[0:half_num:]
					# print minibatch_bad.shape
					# print minibatch_rand.shape
					minibatch = np.union1d(minibatch_bad, minibatch_rand)
				else:
					minibatch = np.random.permutation(np.arange(nb_examples))[0:param.sgd_batch_size]
					# max_dist_inds = np.argpartition(self.X[:,0], int(nb_examples/10))[-int(nb_examples/5):]
					# minibatch = np.union1d(minibatch, max_dist_inds)

				# print minibatch
				# scale using nn coordinate
				x_train_raw = self.X[minibatch,:]
				y_train_raw = self.Y[minibatch]
				# if self.total_training_pts > param.expr_size and kk > 0: #30:
				# 	print 'median', np.median(x_train_raw, axis=0)
				# 	print 'mean', np.mean(x_train_raw, axis=0)
				# 	print 'std', np.std(x_train_raw, axis=0)
				# 	print 'rel_median', (np.median(x_train_raw, axis=0) - self.value_net.nn.avg_vec) / self.value_net.nn.std_vec
				# 	print 'rel_std', np.std(x_train_raw, axis=0) / self.value_net.nn.std_vec
				# 	print 'min', np.amin(x_train_raw, axis=0)
				# 	print 'max', np.amax(x_train_raw, axis=0)
				# 	print 'iter', kk
				# 	raw_input()
				x_train = self.value_net.nn.xRaw_2_x(x_train_raw)
				y_train = self.value_net.nn.yRaw_2_y(y_train_raw)
				# check
				# try:
				# 	assert(np.all(np.squeeze(y_train_raw) <= (0.97**(x_train_raw[:,0]/0.2)+0.01)))
				# except AssertionError:
				# 	num_pts = len(y_train_raw)
				# 	print 'num_pts', num_pts
				# 	for i in xrange(num_pts):
				# 		if True: #y_train_raw[i] > 0.97**(x_train_raw[i,0]/0.2) + 0.01:
				# 			# print '---'
				# 			# print 'x_train[i,:]', x_train_raw[i,:]
				# 			print 'y_train[i] - bnd', y_train_raw[i] - 0.97**(x_train_raw[i,0]/0.2)
				# 	assert(0)

				# update value network
				# print step_size
				# step_size = 0.0
				# self.value_net.nn.set_training_stepsize('fixed_decay')
				# self.value_net.nn.set_training_stepsize('momentum')
				self.value_net.nn.set_training_stepsize('rmsprop')
				self.value_net.nn.backprop(x_train, y_train, step_size, kk)

			# print '   added %d strt_line pts, %d nn_pts' % (strt_line_training_pts, nn_training_pts)



		# plot at end of training
		self.plot_training_score(file_dir)
		self.evaluate_current_network()

	def plot_training_score(self, file_dir):
		if len(self.training_score) > 0:
			fig = plt.figure('training score', figsize=(10,8))
			plt.clf()
			ax1 = fig.add_subplot(1,1,1)
			ax2 = ax1.twinx()

			episodes = self.eval_epsd_stride * np.arange(len(self.training_score))
			num_cases = self.training_score[0].shape[0] / 3
			scores_np = np.asarray(self.training_score)

			total_time_vec = np.sum(scores_np[:,0:num_cases], axis=1)
			collision_vec = np.sum(scores_np[:,num_cases:2*num_cases], axis=1)
			value_vec = np.sum(scores_np[:,2*num_cases:3*num_cases], axis=1)

			ax1.plot(episodes, total_time_vec, 'b')
			ax2.plot(episodes, value_vec, 'r')
			ax1.set_xlabel('episode')
			ax1.set_ylabel('time (s)')
			ax2.set_ylabel('value')
			plt.draw()
			plt.pause(0.0001)
			if self.ifSave:
				plt.savefig(file_dir+"/../../pickle_files/multi/"+ self.value_net.mode +\
					'_' + self.passing_side + "/%d_agents_training_score.png"%self.num_agents,bbox_inches='tight')
		else:
			print('no training score')
		

	def append_to_experience(self, x, y, values_diff, expr_size):
		num_pts = len(x)
		assert(num_pts == len(y))
		assert(num_pts < expr_size)
		gamma = GAMMA
		dt_normal = DT_NORMAL
		for i in xrange(num_pts):
			try:
				assert(y[i] <= gamma ** (x[i,0]/dt_normal)+0.0001)
				assert(x[i,1] > 0.1 - EPS)
			except:
				print('x', x[i,:])
				print('y', y[i])
				print('bnd', gamma ** (x[i,0]/dt_normal))
				assert 0, 'not valid training point'
		if self.current_expr_ind + num_pts < expr_size:
			end_ind = self.current_expr_ind + num_pts
			self.X[self.current_expr_ind:end_ind,:] = x
			self.Y[self.current_expr_ind:end_ind,:] = y
			self.values_diff[self.current_expr_ind:end_ind] =  values_diff
			self.current_expr_ind = end_ind
		else:
			y_num_pts = expr_size - self.current_expr_ind
			self.X[self.current_expr_ind:expr_size,:] = x[0:y_num_pts,:]
			self.Y[self.current_expr_ind:expr_size,:] = y[0:y_num_pts,:]
			self.values_diff[self.current_expr_ind:expr_size] = values_diff[0:y_num_pts]
			self.X[0:num_pts-y_num_pts,:] = x[y_num_pts:num_pts,:]
			self.Y[0:num_pts-y_num_pts,:] = y[y_num_pts:num_pts,:]
			self.values_diff[0:num_pts-y_num_pts] = values_diff[y_num_pts:num_pts]
			self.current_expr_ind = num_pts - y_num_pts
		self.total_training_pts += num_pts
		# print 'self.current_expr_ind', self.current_expr_ind 
		# print 'self.total_training_pts', self.total_training_pts
		# try:
		# 	if y[0] < 0:
		# 		print x
		# 		print y
		# 		t = raw_input('press any key to continue: ')
		# except:
		# 	print x
		# 	print y
		# 	assert(0)
		return


	def evaluate_current_network(self, prev_path_times=None, prev_collisions=None,  iteration=0, plot_mode='all'):
		num_test_cases = len(self.test_cases)
		path_times = np.zeros((num_test_cases,), dtype=float) 
		collisions = np.zeros((num_test_cases,), dtype=bool) 
		plot_number = np.random.randint(len(self.test_cases))
		values = np.zeros((num_test_cases,), dtype=float) 
		for i, test_case in enumerate(self.test_cases):
			traj_raw_multi, time_to_complete = \
				self.value_net.generate_traj(test_case, figure_name='no_plot', stopOnCollision=False)

			# plotting (debugging)
			agents_speed, agents_time, agents_len, min_dist = compute_plot_stats(traj_raw_multi)
			title_string = 'case: %d; a%d, t %.2f, sp %.2f, len %.2f \n %s; min_dist %.2f a%d t %.2f, sp %.2f, len %.2f' % \
				(i, 0, agents_time[0], agents_speed[0], agents_len[0], \
					self.passing_side, min_dist, 1, agents_time[1], agents_speed[1], agents_len[1])
			num_agents = len(traj_raw_multi) - 1
			if num_agents > 2:
				for tt in xrange(2, num_agents):
					agent_string = '\n a%d, t %.2f, sp %.2f, len %.2f' % \
						(tt, agents_time[tt], agents_speed[tt], agents_len[tt])
					title_string += agent_string
			
			if_collided = min_dist < 0.0
			collisions[i] = if_collided
			path_times[i] = np.sum(agents_time)
			if plot_mode == 'all': # plot every time case
				pedData.plot_traj_raw_multi(traj_raw_multi, title_string)
				# 	% (i, agent_1_time, agent_2_time, total_time)
			elif plot_mode == 'one' and i == plot_number: # only plot one test case
				pedData.plot_traj_raw_multi(traj_raw_multi, title_string, figure_name=self.mode+'evaluate')
			else:
				pass

			# plot bad trajectories
			if iteration > 200 and prev_path_times!=None and \
				(collisions[i] == True or (path_times[i] - prev_path_times[i]) > 3.0):
				figure_name_str = 'bad_traj_tc_%d' % (i)
				title_string = ('iter %d ;' % iteration) + title_string 
				pedData.plot_traj_raw_multi(traj_raw_multi, title_string, figure_name=self.mode+figure_name_str)

			agent_state = traj_raw_multi[1][0,:]
			other_agents_state = []
			num_agents = len(traj_raw_multi) - 1
			for tt in xrange(1, num_agents):
				other_agents_state.append(traj_raw_multi[tt+1][0,:])
			values[i] = self.value_net.find_states_values(agent_state, other_agents_state)
			
		# np.set_printoptions(precision=4)
		value_str = '    tc(0-%d)' % num_test_cases  
		path_times_str = '    tc(0-%d)' % num_test_cases 
		for tt in xrange(num_test_cases):
			value_str += ', %.3f' % values[tt]
			path_times_str += ', %.3f' % path_times[tt]
		print(value_str)
		print(path_times_str)					
		return path_times, collisions, values

	# for plotting purposes
	def plot_test_cases(self, folder_dir, filename_str, format_str):
		for i, test_case in enumerate(self.test_cases):
			traj_raw_multi, time_to_complete = \
				self.value_net.generate_traj(test_case, figure_name='no_plot')
			# file name (iteration # and test case #)
			filename = folder_dir + '/tc' + str(i) + '_' + filename_str + format_str

			# trajectory stats
			# a1_speed = traj_raw[0,6]
			# a2_speed = traj_raw[0,15]
			# a1_len = np.sum(np.linalg.norm(traj_raw[0:-1, 1:3] - traj_raw[1:, 1:3], axis=1)) + \
			# 			np.linalg.norm(traj_raw[-1, 1:3] - traj_raw[-1, 7:9])
			# a2_len = np.sum(np.linalg.norm(traj_raw[0:-1, 10:12] - traj_raw[1:, 10:12], axis=1)) + \
			# 			np.linalg.norm(traj_raw[-1, 10:12] - traj_raw[-1, 16:18])
			# min_dist = np.amin(np.linalg.norm(traj_raw[:,1:3]-traj_raw[:,10:12], axis=1)) - \
			# 			traj_raw[0,9] - traj_raw[0,18]
			agents_speed, agents_time, agents_len, min_dist = compute_plot_stats(traj_raw_multi)
			title_string = 'case: %d; a%d, t %.2f, sp %.2f, len %.2f \n %s; min_dist %.2f a%d t %.2f, sp %.2f, len %.2f' % \
				(i, 0, agents_time[0], agents_speed[0], agents_len[0], \
					self.passing_side, min_dist, 1, agents_time[1], agents_speed[1], agents_len[1])
			num_agents = len(traj_raw_multi) - 1
			if num_agents > 2:
				for tt in xrange(2, num_agents):
					agent_string = '\n a%d, t %.2f, sp %.2f, len %.2f' % \
						(tt, agents_time[tt], agents_speed[tt], agents_len[tt])
					title_string += agent_string

			pedData.plot_traj_raw_multi(traj_raw_multi, title_string, 'plot_test_cases')
			if self.ifSave:
				plt.savefig(filename, bbox_inches='tight')


	# find intended next states(traj_raw_multi)
	# def find_intended_future_state_value(self, agent_state, agent_action_xy, other_agents_state, dt_forward):
	# 	num_states = 7 + 8 * (self.num_agents - 1)

	# 	agent_action_theta = np.array([np.linalg.norm(agent_action_xy), \
	# 		np.arctan2(agent_action_xy[1], agent_action_xy[0])])
		
	# 	# forward propagate to next states
	# 	dt = dt_forward
	# 	num_other_agents = len(other_agents_state)
	# 	agent_next_state = self.value_net_copy.update_state(agent_state, agent_action_theta, dt)
	# 	others_action_xy = [other_agents_state[tt][2:4] for tt in xrange(num_other_agents)]
	# 	others_next_state = []
	# 	for tt in xrange(num_other_agents):
	# 		# print np.linalg.norm(others_action_xy[tt])
	# 		# print np.arctan2(others_action_xy[tt][1], others_action_xy[tt][0])
	# 		action_theta = np.array([np.linalg.norm(others_action_xy[tt]), \
	# 			np.arctan2(others_action_xy[tt][1], others_action_xy[tt][0]) ])
	# 		others_next_state.append(self.value_net_copy.update_state(other_agents_state[tt], \
	# 			action_theta, dt))

	# 	# value of next state
	# 	# dt_backup = 1.0
	# 	ref_prll_vec, ref_orth_vec, state_nn = \
	# 		pedData.rawState_2_agentCentricState(\
	# 			agent_next_state, others_next_state, self.num_agents)
	# 	value = self.value_net_copy.find_states_values(agent_next_state, others_next_state)
	# 	return state_nn, value


	# find intended next states(traj_raw_multi)
	def find_deviation_cases(self, traj_raw_multi):
		time_to_reach_goal, traj_lengths, min_sepDist, if_completed_vec \
			= pedData.computeStats(traj_raw_multi)
		num_agents = len(traj_raw_multi) - 1
		time_vec = traj_raw_multi[0]
		num_pts = len(time_vec)
		
		max_deviation = 0.0
		max_deviation_ind = 0.0
		max_ind_dt_forward = 0.0

		future_time_ind = 0
		for j in xrange(1,num_pts-1):
			deviation_vec = np.zeros((num_agents,))
			while time_vec[future_time_ind] - time_vec[j] < 1.0 \
					and future_time_ind<num_pts-1:
				future_time_ind += 1
			if future_time_ind >= num_pts:
				break
			dt_forward = time_vec[future_time_ind] - time_vec[j]
			
			for i in xrange(num_agents):
				if time_to_reach_goal[i] > future_time_ind:
					continue
				agent_state_pos = traj_raw_multi[i+1][j,0:2]
				agent_action_xy_chosen = traj_raw_multi[i+1][j+1,2:4]
				agent_intended_pos = agent_state_pos + \
						agent_action_xy_chosen * dt_forward
				agent_future_pos = traj_raw_multi[i+1][future_time_ind ,0:2]
				deviation_vec[i] = np.linalg.norm(agent_intended_pos - \
						agent_future_pos) / traj_raw_multi[i+1][0,5]

			max_deviation_tmp = np.max(deviation_vec)
			if max_deviation_tmp > max_deviation:
				max_deviation = max_deviation_tmp
				max_deviation_ind = j
				max_ind_dt_forward = dt_forward

		# build test case
		test_case = np.zeros((num_agents, 6))
		j = max_deviation_ind
		dt_forward = max_ind_dt_forward
		for i in xrange(num_agents):
			test_case[i,0:2] = traj_raw_multi[i+1][j,0:2] + \
					dt_forward * traj_raw_multi[i+1][j+1,2:4]
			test_case[i,2:4] = traj_raw_multi[i+1][j,6:8]
			test_case[i,4] = traj_raw_multi[i+1][j,5]
			test_case[i,5] = traj_raw_multi[i+1][j,8]
		# print dt_forward
		# print test_case
		# raw_input()
				
		return test_case

	# returns
	# time_2_goal_vec, time_2_goal_bnd, agent_centric_states, values, action_rewards
	def rawTraj_2_trainingStats(self, time_vec, traj_raw_multi, agent_num, iteration=0):
		num_pts = len(time_vec)
		# compute stats
		# print time_vec.shape, agent_states.shape, other_agent_states.shape
		agent_states = traj_raw_multi[agent_num+1]
		other_agents_states = [traj_raw_multi[tt] for tt in \
				xrange(1, len(traj_raw_multi)) if tt!=agent_num+1] 
		# print 'agent_number+1', agent_num+1
		# print 'other', [tt for tt in \
		# 		xrange(1, len(traj_raw_multi)) if tt!=agent_num+1] 
		time_to_reach_goal, traj_lengths, min_sepDist, if_completed_vec \
			= pedData.computeStats(traj_raw_multi)
		agent_speed = agent_states[0,5]

		# initialize return values
		time_2_goal_vec = np.empty((num_pts,)); time_2_goal_vec[:] = np.nan
		time_2_goal_bnd = np.empty((num_pts,)); time_2_goal_bnd[:] = np.nan
		num_states = 7 + 8 * (self.num_agents -1)
		agent_centric_states = np.zeros((num_pts, num_states))
		values = np.zeros((num_pts,))
		action_rewards = np.zeros((num_pts,))

		gamma = GAMMA
		dt_normal = DT_NORMAL
		agent_desired_speed = agent_speed
		counter = 0
		time_bnd = np.linalg.norm(agent_states[0,0:2]-agent_states[0,6:8])/agent_states[0,5]
		ifReachedGoal = False

		# filter speeds
		num_other_agents = len(other_agents_states)
		other_agents_filtered_vel = np.zeros((num_pts, num_other_agents * 2))
		dt_vec = time_vec.copy(); dt_vec[1:] = time_vec[1:] - time_vec[:-1]; dt_vec[0] = dt_vec[1]
		time_past_one_ind = 0
		for i in xrange(num_pts):
			while time_vec[i] - time_vec[time_past_one_ind] > 0.45:
				time_past_one_ind += 1
			agent_pos = agent_states[i,0:2]
			dt_past_vec = dt_vec[time_past_one_ind:i+1]
			for j in xrange(num_other_agents):
				past_vel = other_agents_states[j][time_past_one_ind:i+1,2:5]
				if np.linalg.norm(agent_pos - other_agents_states[j][i,0:2]) < 0.5:
					other_agents_filtered_vel[i,j*2:(j+1)*2] = \
						nn_nav.filter_vel(dt_past_vec, past_vel, ifClose=True)
				else:
					other_agents_filtered_vel[i,j*2:(j+1)*2] = \
						nn_nav.filter_vel(dt_past_vec, past_vel, ifClose=False)

		for i in xrange(num_pts):
			counter += 1
			agent_state = agent_states[i,:]
			other_agents_state = [other_agents_states[tt][i,:].copy() for tt in xrange(len(other_agents_states))]
			# for j in xrange(num_other_agents):
			# 	# print i,j, 'before', other_agents_state[j][2:4] 
			# 	other_speed = other_agents_filtered_vel[i,j*2]
			# 	other_angle = other_agents_filtered_vel[i,j*2+1]
			# 	other_agents_state[j][2] = other_speed * np.cos(other_angle)
			# 	other_agents_state[j][3] = other_speed * np.sin(other_angle)
			# 	# print 'after', other_agents_state[j][2:4] 
				# raw_input()

			# print 'd_2_goal', np.linalg.norm(agent_state[0:2] - agent_state[6:8])
			# print 'time %.3f, time_to_reach_goal %.3f' %(time_vec[i], time_to_reach_goal[agent_num]) 
			# print '---- ifReachedGoal ---', ifReachedGoal
			
			# time 2 goal
			if ifReachedGoal:
				time_2_goal_vec[i] = 0.0
			elif if_completed_vec[agent_num]:
				time_2_goal_vec[i] = time_to_reach_goal[agent_num] - time_vec[i]
				try:
					assert(time_2_goal_vec[i] > -EPS)
				except AssertionError:
					print(time_to_reach_goal[agent_num])
					print(time_vec[i])
					assert(0)

			# # agent_centric_state
			# agent_speed = agent_state[5]
			# assert(agent_speed > 0.1 - EPS)
			# dt_backward_max = max(self.value_net.dt_forward, 0.5/agent_speed)
			# # dt_forward_max = self.dt_forward
			# dist_to_goal = np.linalg.norm(agent_state[6:8]- agent_state[0:2])
			# time_to_goal = dist_to_goal / agent_speed
			# dt_backward= min(dt_backward_max, time_to_goal) #1.0

			# ii = i
			# while ii > 0:
			# 	if time_vec[i] - time_vec[ii] > dt_backward:
			# 		ii = ii - 1
			# other_agents_past_state = [other_agents_states[tt][ii,:].copy() for tt in xrange(len(other_agents_states))]

			# ref_prll, ref_orth, state_nn = \
			# 	pedData.rawState_2_agentCentricState( \
			# 	agent_state, other_agents_past_state, self.num_agents)
			# agent_centric_states[i,:] = state_nn.copy()

			ref_prll, ref_orth, state_nn = \
				pedData.rawState_2_agentCentricState( \
				agent_state, other_agents_state, self.num_agents)
			agent_centric_states[i,:] = state_nn.copy()
			
			# time_2_goal_bnd
			time_2_goal_bnd[i] = state_nn[0] / agent_speed
			# time_2_goal_bnd[i] = time_bnd - time_vec[i]

			# action_rewards and values
			if i == 0:
				values[0] = self.value_net_copy.find_states_values(agent_state, other_agents_state)
			if i < num_pts - 1:
				# note i+1
				agent_next_state = agent_states[i+1,:]
				other_agents_next_state = [other_agents_states[tt][i+1,:] for tt in xrange(len(other_agents_states))]
				
				dt_forward = time_vec[i+1] - time_vec[i]
				state_value, action_reward = \
				self.value_net_copy.find_next_state_pair_value_and_action_reward(agent_state, \
					agent_next_state, other_agents_state, \
					other_agents_next_state, dt_forward)

				# print 'method 1: state_value, ', state_value1
				cur_dist_vec = [np.linalg.norm(agent_state[0:2] - other_agent_state[0:2])-\
					agent_state[8]-other_agent_state[8] for \
					other_agent_state in other_agents_state]
				cur_dist = min(cur_dist_vec)
				# min_dists = [np.linalg.norm(agent_next_state[0:2] - other_agent_next_state[0:2])-\
				# 	agent_next_state[8]-other_agent_next_state[8] for \
				# 	other_agent_next_state in other_agents_next_state]
				# # print 'i, cur_dist, next_dist', i, cur_dist, min(min_dists)
				# # min_dist = np.array([min(min_dists)]) #- np.random.rand() * 0.05
				# min_dist = np.array([cur_dist]) + 1.0
				action_reward = self.value_net_copy.find_action_rewards_train(agent_state, \
					cur_dist, dt_forward)
				# action_reward_min = min(action_reward, action_reward_2)
				# if action_reward_min < -EPS:
				# 	print action_reward, action_reward_2, action_reward < action_reward_2
				# 	raw_input()
				# action_reward = action_reward_min

				if abs(state_value) < EPS:
					state_value = 0.01
				# state_value = self.value_net_copy.find_states_values(agent_next_state, other_agents_next_state)
				# # print 'method 2: state_value, ', state_value

				# if abs(state_value1 - state_value) > 0.01:
				# 	print 'method 1: state_value, ', state_value1
				# 	print 'method 2: state_value, ', state_value
				# 	print 'num_agents', len(other_agents_state)

				# 	print ' --- 111 ---'
				# 	state_value1, action_reward = \
				# 	self.value_net_copy.find_next_state_pair_value_and_action_reward(agent_state, \
				# 		agent_next_state, other_agents_state, \
				# 		other_agents_next_state, dt_forward)
				# 	print ' --- 222 ---'
				# 	state_value = self.value_net_copy.find_states_values(agent_next_state, other_agents_next_state)
				# 	raw_input()
				action_rewards[i] = action_reward			
				values[i+1] = state_value
			if i == num_pts - 1:
				cur_dist_vec = [np.linalg.norm(agent_state[0:2] - other_agent_state[0:2])-\
						agent_state[8]-other_agent_state[8] for \
						other_agent_state in other_agents_state]
				cur_dist = min(cur_dist_vec)
				min_dists = np.array(cur_dist_vec) + 1.0
				dt_forward = 1.0 
				action_rewards[i] = self.value_net_copy.find_action_rewards(agent_state, \
					cur_dist, min_dists, dt_forward)[0]

			# terminal states
			is_terminal_state = self.value_net_copy.if_terminal_state(agent_state, other_agents_state)
			if is_terminal_state == COLLIDED:
				values[i] = COLLISION_COST
				action_rewards[i] = 0.0
				break
			elif is_terminal_state == REACHED_GOAL:
				Dt_bnd = state_nn[0] / state_nn[1]
				values[i] = (gamma ** (Dt_bnd * state_nn[1] / dt_normal))
				action_rewards[i] = 0.0
				ifReachedGoal = True
				break
			# sufficiently close to goal but also close to the other agent
			elif np.linalg.norm(agent_state[0:2]-agent_state[6:8]) < DIST_2_GOAL_THRES:
				Dt_bnd = state_nn[0] / state_nn[1]
				values[i] = (gamma ** (Dt_bnd * state_nn[1] / dt_normal))
				ifReachedGoal = True
				break
			# debug
			# print 'time, dist_to_goal, pref_speed', time_vec[i], \
			# 	np.linalg.norm(agent_state[0:2]-agent_state[6:8]), agent_state[5]
			# if np.linalg.norm(agent_state[0:2]-agent_state[6:8])<DIST_2_GOAL_THRES:
			# 	print 'here'
			# 	print agent_state
			# 	print other_agent_state
			# 	print np.linalg.norm(agent_state[0:2]-other_agent_state[0:2])- \
			# 		agent_state[8]-other_agent_state[8]
		
		eff_pts = min(num_pts, counter)
		# print 'num_pts, counter, eff_pts', num_pts, counter, eff_pts
		try:
			assert(num_pts>0)
		except:
			for i in xrange(1,len(traj_raw_multi)):
				print(traj_raw_multi[i][0,:])
			assert(0)
		return time_2_goal_vec[0:eff_pts], time_2_goal_bnd[0:eff_pts], \
			agent_centric_states[0:eff_pts,:], values[0:eff_pts], action_rewards[0:eff_pts]

	def rawTraj_2_trainingData(self, traj_raw_multi, gamma, iteration, ifOnlyFirstAgent=False):
		time_vec = traj_raw_multi[0]

		num_agents = len(traj_raw_multi) - 1
		agents_time_2_goal_vec_list = []
		agents_time_2_goal_bnd_list = []
		agents_centric_states_list = []
		agents_values_list = []
		agents_action_reward_list = []
		agents_extra_time_list = []

		X = []; Y = []; values_diff = []

		for tt in xrange(num_agents):
			time_2_goal_vec, time_2_goal_bnd, agent_centric_states, \
				values, action_rewards = self.rawTraj_2_trainingStats( \
				time_vec, traj_raw_multi, tt, iteration=iteration)
			extra_time = self.computeExtraTime(time_2_goal_vec,time_2_goal_bnd, \
				time_vec[0:len(time_2_goal_bnd)])

			agents_time_2_goal_vec_list.append(time_2_goal_vec)
			agents_time_2_goal_bnd_list.append(time_2_goal_bnd)
			agents_centric_states_list.append(agent_centric_states)
			agents_values_list.append(values)
			agents_action_reward_list.append(action_rewards)
			agents_extra_time_list.append(extra_time)


		dt = TRAINING_DT

		for tt in xrange(num_agents):
			if ifOnlyFirstAgent and tt > 0:
				break
			# skip straight line trajectories
			# if abs(agents_time_2_goal_vec_list[tt][0] - np.linalg.norm(traj_raw_multi[tt+1][0,0:2]-\
				# traj_raw_multi[tt+1][0,6:8])/traj_raw_multi[tt+1][0,5]) < EPS:
			path_length = np.linalg.norm(traj_raw_multi[tt+1][0,0:2]-\
				traj_raw_multi[tt+1][0,6:8])
			exp_min_time = path_length /traj_raw_multi[tt+1][0,5]
			if_completed = np.isnan(agents_time_2_goal_vec_list[tt][0]) == False
			if path_length < EPS or (if_completed and (agents_time_2_goal_vec_list[tt][0] / exp_min_time < 1.05)):
				continue
			agent_num_pts = len(agents_time_2_goal_bnd_list[tt])
			# don't include stationary agents
			# if agent_num_pts < 2:
			# 	continue
			other_agents_extra_time = [agents_extra_time_list[i] for i in xrange(num_agents) if i!=tt]
			other_agents_states = [traj_raw_multi[i+1] for i in xrange(num_agents) if i!=tt]
			agent_states = traj_raw_multi[tt+1]
			X1, Y1, values_diff1 = self.trainingStats_2_trainingData(time_vec[0:agent_num_pts], dt, \
				agents_time_2_goal_vec_list[tt], agents_time_2_goal_bnd_list[tt], agents_centric_states_list[tt], \
				agents_values_list[tt],	agents_action_reward_list[tt], other_agents_extra_time, \
				agent_states, other_agents_states, iteration, traj_raw_multi=traj_raw_multi)
			# print X1[1,:]
			# print Y1[1,:]
			# raw_input()
			if len(X) == 0:
				X = X1.copy()
				Y = Y1.copy()
				values_diff = values_diff1
			else:
				X = np.vstack((X, X1.copy()))
				Y = np.vstack((Y, Y1.copy()))
				values_diff = np.hstack((values_diff, values_diff1))

			# X_future, Y_future = self.find_intended_future_states(traj_raw_multi)
			# X = np.vstack((X, X_future.copy()))
			# Y = np.vstack((Y, Y_future.copy()))

		# num_pts = len(X)
		# num_pts_thres = 300
		# if num_pts > num_pts_thres:
		# 	minibatch = np.random.permutation(np.arange(num_pts))[0:num_pts_thres]
		# 	X = X[minibatch,:]
		# 	Y = Y[minibatch,:]

		return X, Y, values_diff

	# def debug_rawTraj_2_trajStats(self):
	# 	for i, test_case in enumerate(self.test_cases):
	# 		if i != 2:
	# 			continue
	# 		traj_raw, agent_1_time, agent_2_time, if_collided = \
	# 			self.value_net.generate_traj(test_case, figure_name='no_plot')

	# 		traj_raw_multi = pedData.traj_raw_2_traj_raw_multi(traj_raw)
	# 		time_vec = traj_raw_multi[0]
	# 		agent_states = traj_raw_multi[1]
	# 		other_agent_states = traj_raw_multi[2]

	# 		time_vec = traj_raw[:,0]
	# 		agent_1_states = traj_raw[:,1:10] 
	# 		agent_2_states = traj_raw[:,10:19]

	# 		a1_time_2_goal_vec, a1_time_2_goal_bnd, a1_agent_centric_states, \
	# 			a1_values, a1_action_rewards = self.rawTraj_2_trainingStats( \
	# 			time_vec, agent_states, other_agent_states)
			
	# 		# np.set_printoptions(precision=4,formatter={'float': '{: 0.3f}'.format})
	# 		# zero_inds = np.where(a1_action_rewards<EPS)[0]
	# 		# a1_action_rewards[zero_inds] = 0
	# 		# print a1_action_rewards[zero_inds]

	# 		a2_time_2_goal_vec, a2_time_2_goal_bnd, a2_agent_centric_states, \
	# 			a2_values, a2_action_rewards = self.rawTraj_2_trainingStats( \
	# 			time_vec, other_agent_states, agent_states)
	# 		# zero_inds = np.where(a2_action_rewards<EPS)[0]
	# 		# a2_action_rewards[zero_inds] = 0
	# 		# print a2_action_rewards[zero_inds]

	# 		print '--- test_case %d --- ' % i
	# 		self.rawTraj_2_trajStats(time_vec, agent_1_states, agent_2_states, \
	# 			a1_time_2_goal_vec, a1_agent_centric_states, ifPlot=True)
	# 		self.rawTraj_2_trajStats(time_vec, agent_2_states, agent_1_states, \
	# 			a2_time_2_goal_vec, a2_agent_centric_states, ifPlot=True)

	# 		gamma = 0.97 
	# 		X, Y = self.rawTraj_2_trainingData(traj_raw, gamma, 0)

	# compute trajectory properties, such as passing on the left of the other vehicle
	def rawTraj_2_trajStats(self, time_vec, agent_states, other_agent_states, \
		time_2_goal_vec, agent_centric_states, iteration=0, ifPlot=False):
		num_pts = len(time_vec) - 1

		if np.isnan(time_2_goal_vec[0]):
			return np.ones((num_pts,))

		bad_inds_oppo, bad_inds_same, bad_inds_tangent = \
			self.value_net.find_bad_inds(agent_centric_states)

		#scaling factor
		d = np.linalg.norm(agent_states[:-1,0:2] - agent_states[:-1,6:8], axis=1)
		v = agent_states[0,5]
		getting_close_penalty = GAMMA ** (d/DT_NORMAL) * (1.0 - GAMMA ** (-v/DT_NORMAL))
		penalty = np.zeros((num_pts,))
		penalty[bad_inds_oppo] = 0.7 * getting_close_penalty[bad_inds_oppo]
		penalty[bad_inds_same] = 0.7 * getting_close_penalty[bad_inds_same]
		penalty[bad_inds_tangent] = 0.7 * getting_close_penalty[bad_inds_tangent]
		
		time_2_goal_upper_bnd = np.zeros((num_pts,))
		time_2_goal_upper_bnd[bad_inds_oppo] = time_2_goal_vec[bad_inds_oppo] + 1.0
		time_2_goal_upper_bnd[bad_inds_same] = time_2_goal_vec[bad_inds_same] + 1.0
		time_2_goal_upper_bnd[bad_inds_tangent] = time_2_goal_vec[bad_inds_tangent] + 1.0
		dt_normal = DT_NORMAL
		value_upper_bnd = GAMMA ** (time_2_goal_upper_bnd * agent_states[0,5] / dt_normal) 
		# print dt_normal
		# print value_upper_bnd
		# raw_input()

		# penalty[bad_inds_same] += -0.2

		# penalty = np.clip(penalty, -0.1, 0.0)
		if ifPlot: #len(bad_inds_oppo) > 3 or len(bad_inds_same) or len(bad_inds_tangent) :
			# print 'heading_diff[bad_inds_oppo]', heading_diff[bad_inds_oppo]
			# print 'tangent_inds', tangent_inds
			# print 'stationary_inds', stationary_inds			
			traj_raw = np.hstack((time_vec[:,np.newaxis], agent_states, other_agent_states))
			pedData.plot_traj_raw_multi(traj_raw, 'from rawTraj_2_trajStats', figure_name="raw_traj")
			if len(bad_inds_oppo) > 0:
				print('bad_inds_oppo', bad_inds_oppo)
				traj_raw_bad = np.hstack((time_vec[bad_inds_oppo,np.newaxis], agent_states[bad_inds_oppo,:], \
					other_agent_states[bad_inds_oppo,:]))
				# print('traj_raw_bad', traj_raw_bad)
				pedData.plot_traj_raw_multi(traj_raw_bad, 'from rawTraj_2_trajStats, bad inds oppo', figure_name="bad_inds_oppo")
				# raw_input()
			if len(bad_inds_same) > 0:
				print('bad_inds_same', bad_inds_same)
				traj_raw_bad = np.hstack((time_vec[bad_inds_same,np.newaxis], agent_states[bad_inds_same,:], \
					other_agent_states[bad_inds_same,:]))
				# print('traj_raw_bad', traj_raw_bad)
				pedData.plot_traj_raw_multi(traj_raw_bad, 'from rawTraj_2_trajStats, bad inds same', figure_name="bad_inds_same")
				# raw_input()
			if len(bad_inds_tangent) > 0:
				print('bad_inds_tangent', bad_inds_tangent)
				traj_raw_bad = np.hstack((time_vec[bad_inds_tangent,np.newaxis], agent_states[bad_inds_tangent,:], \
					other_agent_states[bad_inds_tangent,:]))
				# print('traj_raw_bad', traj_raw_bad)
				pedData.plot_traj_raw_multi(traj_raw_bad, 'from rawTraj_2_trajStats, bad inds tangent', figure_name="bad_inds_tangent")
				# raw_input()
			print(penalty)
			raw_input()

		# if iteration < 200:
		# 	penalty[bad_inds_same] = 3.0 * getting_close_penalty[bad_inds_same]

		# return penalty
		return value_upper_bnd

	def computeExtraTime(self, time_2_goal_vec, time_bnd, time_vec):
		# method 1
		# if np.isnan(time_2_goal_vec[0]):
		# 	extra_time = np.zeros((len(time_2_goal_vec),))
		# 	extra_time[:] = np.inf
		# else:
		# 	extra_time = np.clip(time_2_goal_vec - time_bnd, 0, 100) 
		# try:
		# 	assert(np.all(extra_time>-EPS))
		# except AssertionError:
		# 	print 'extra_time', extra_time
		# 	print 'time_2_goal_vec', time_2_goal_vec
		# 	print 'time_bnd', time_bnd
		# 	assert(0)
		# return extra_time

		# print 'time_bnd', time_bnd
		# print 'time_2_goal_vec',time_2_goal_vec
		# print np.clip(time_2_goal_vec - time_bnd, 0, 100) 

		# method 2
		if np.isnan(time_2_goal_vec[0]):
			extra_time_individual = np.zeros((len(time_2_goal_vec),))
			extra_time_individual[:] = np.inf
		elif len(time_vec) < 2:
			extra_time_individual = np.zeros((len(time_2_goal_vec),))
			extra_time_individual[:] = 0
		else:
			dt_time_vec = time_vec.copy()
			dt_time_vec[:-1] = time_vec[1:]-time_vec[:-1]; dt_time_vec[-1] = dt_time_vec[-2]
			dt_2_goal = time_bnd.copy()
			dt_2_goal[:-1] = time_bnd[:-1]-time_bnd[1:]; dt_2_goal[-1] = dt_2_goal[-2]

			extra_time_individual_raw = dt_time_vec - dt_2_goal
		 
			try:
				assert(np.all(extra_time_individual_raw>-EPS))
			except AssertionError:
				print('extra_time_individual_raw', extra_time_individual_raw)
				print('dt_time_vec', dt_time_vec)
				print('dt_2_goal', dt_2_goal)
				assert(0)
			# print 'extra_time_individual', extra_time_individual
			width = 5
			num_pts = len(extra_time_individual_raw)
			extra_time_individual = extra_time_individual_raw.copy()
			for i in xrange(num_pts):
				extra_time_individual[i] = \
					np.sum(extra_time_individual_raw[max(0,i-width):min(i+width, num_pts)])
			
		return extra_time_individual
		

	def minFutureRewards(self, action_rewards):
		num_pts = len(action_rewards)
		future_min_rewards = action_rewards.copy()
		for i in xrange(num_pts):
			future_min_rewards[i] = np.min(action_rewards[i:])
		return future_min_rewards


	def trainingStats_2_trainingData(self, time_vec, dt, time_2_goal_vec, \
		time_2_goal_bnd, agent_centric_states, values, \
		action_rewards, other_agents_extra_time, agent_states, other_agents_states, iteration, traj_raw_multi=None):
		num_pts = len(time_vec)
		num_states = 7 + 8 * (self.num_agents - 1)
		X = np.zeros((num_pts,num_states)); X_future = np.zeros((num_pts,num_states)); X_stuck = np.zeros((0,num_states))
		Y = np.zeros((num_pts,1)); Y_future = np.zeros((num_pts,1)); Y_stuck = np.zeros((0,1))
		future_value_inds = []
		extra_time = self.computeExtraTime(time_2_goal_vec,time_2_goal_bnd, time_vec)
		dist_travelled_vec = np.linalg.norm(agent_states[1:,0:2]-agent_states[0:-1,0:2], axis=1)
		dist_travelled_vec = np.append(dist_travelled_vec,[0])
		# if len(other_extra_time) > num_pts:
		# 	other_extra_time = other_extra_time[0:num_pts]
		# else:
		# 	other_extra_time_tmp = np.zeros((num_pts,))
		# 	other_extra_time_tmp[0:len(other_extra_time)] = other_extra_time
		# 	other_extra_time = other_extra_time_tmp


		# if other agents have collided
		if_other_collided = False
		num_other_agents = len(other_agents_states)
		for i in xrange(num_other_agents):
			for j in xrange(i+1, num_other_agents):
				dist = np.linalg.norm(other_agents_states[i][-1, 0:2] - 
					other_agents_states[j][-1, 0:2]) - \
					other_agents_states[i][-1,8] - other_agents_states[j][-1,8]
				if dist < 0:
					if_other_collided = True
		
		# if agent has collided with others
		if_agent_collided = False
		for i in xrange(num_other_agents):
			dist = np.linalg.norm(agent_states[-1, 0:2] - 
					other_agents_states[i][-1, 0:2]) - \
					agent_states[-1,8] - other_agents_states[i][-1,8]
			if dist < 0.0:
				if_agent_collided = True
				break

		# dist_2_others (see README.txt)
		others_columns_inds = [7 + 6 + 8*(tt) for tt in xrange(num_other_agents)] 
		min_dist_2_others = np.min(agent_centric_states[:,others_columns_inds], axis = 1)

		gamma = GAMMA
		dt_normal = DT_NORMAL
		agent_desired_speed = agent_centric_states[0,1]

		j = 0
		dt_forward_vec = np.zeros((len(time_2_goal_bnd),))
		if_extra = False
		if_stuck = False
		if_stuck_counter = 0
		counter = 0
		for i in xrange(num_pts-1):
			while time_vec[j] - time_vec[i] < dt and j < num_pts-1:
				if min_dist_2_others[j+1] > 0 or j<=i:
					j += 1
				# elif min_dist_2_others[j] < GETTING_CLOSE_RANGE:
				# 	break
				else:
					break
			if i == num_pts - 1:
				j = i

			# skip points
			# if time_2_goal_vec[i] < time_2_goal_bnd[i] * 1.01:
			# 	# print 'time_2_goal_vec[i], time_2_goal_bnd[i]', time_2_goal_vec[i], time_2_goal_bnd[i]
			# 	# raw_input()
			# 	if np.random.rand() > 0.2:
			# 		continue
				# else:
				# 	break

			X[counter,:] = agent_centric_states[i,:]
			
			# compute value using q-learning update
			# print 'j, num_pts', j, num_pts
			# print len(time_2_goal_vec), len(time_2_goal_bnd), \
			# 	len(agent_centric_states), len(agent_centric_states), \
			# 	len(values), len(action_rewards), len(other_extra_time)

			# neural net output is non-sensible (negative value and zero reward)
			value_bnd = (gamma ** (agent_centric_states[i,0] / dt_normal))
			# if values[j] < 0 and agent_centric_states[j,13] > 0.1:
			# 	state_value = max(0, value_bnd - 0.2)

			# action_reward = action_rewards[i] #np.min(action_rewards[i:max(i+1,j)])
			action_reward = np.min(action_rewards[i:max(i+1,j)])


			############################################################################
			# use one point
			# print 'i %d, j %d' %(i, j)
			dt_forward = time_vec[j] - time_vec[i]
			# dist_travelled = np.linalg.norm(agent_states[j,0:2]-agent_states[i,0:2])
			dist_travelled = np.sum(dist_travelled_vec[i:j])
			dt_forward_scaled = dist_travelled / agent_desired_speed
			assert(np.isnan(dt_forward_scaled)==0)
			# dt_forward_adj = 1.0 * dt_forward + 0.0 * dt_forward_scaled
			dt_forward_adj = 0.5 * dt_forward + 0.5 * dt_forward_scaled
			# dt_forward_adj = 1.0 * dt_forward
			# print dt_forward, dt_forward_scaled
			# raw_input()

			# try:
			# 	assert(dt_forward +EPS >= dt_forward_adj)
			# except:
			# 	print 'dt_forward', dt_forward
			# 	print 'dt_forward_scaled',dt_forward_scaled
			# 	print 'dt_forward_adj', dt_forward_adj
			# 	print 'dist_travalled', dist_travelled
			# 	print 'dist_travelled / agent_desired_speed', dist_travelled / agent_desired_speed
			# 	assert(0) 

			state_value = values[j]
			value_q_learning = action_reward + gamma ** (dt_forward_adj * \
				agent_desired_speed / dt_normal) * state_value
			dt_forward_vec[i] = dt_forward

			###########################################################################
			# use all points upto 1 seconds into the future
			# print 'i %d, j %d' %(i, j)
		# 	upper_ind =  j+1 # j+1
		# 	lower_ind =  min(j, i+5) # i+1 
			# dt_forward = time_vec[lower_ind:upper_ind] - time_vec[i] 
			# state_values = values[lower_ind:upper_ind]

			# agent_speeds = agent_centric_states[lower_ind:upper_ind,2]
			# # dt_forward_post = dt_forward.copy()
			# # dt_forward_tmp = time_vec[i+1:j+1] - time_vec[i:j] 
			# # for tt in xrange(1,j-i):
			# # 	dt_forward_post[tt-1] = dt_forward[tt-1] * 0.2 + 0.8 * np.sum(agent_speeds[0:tt] / agent_desired_speed \
			# # 		* dt_forward_tmp[0:tt])
			# dist_travelled = dist_travelled_vec[lower_ind:upper_ind].copy()
			# dist_travelled[0] += np.sum(dist_travelled_vec[i:lower_ind])
			# for tt in xrange(1, len(dist_travelled)):
			# 	dist_travelled[tt] += dist_travelled[tt-1]
			# # dist_travelled = np.linalg.norm(agent_states[lower_ind:upper_ind,0:2]-agent_states[i,0:2], axis=1)
			# dt_forward_post = 0.5 * dt_forward + 0.5 * dist_travelled / agent_desired_speed


			# value_q_learning = action_reward + np.mean(gamma ** (dt_forward_post * \
			# 	agent_desired_speed / dt_normal) * state_values)
			# dt_forward_vec[i] = time_vec[j] - time_vec[i]
			# try:
			# 	assert(np.isnan(value_q_learning) == False)
			# except:
			# 	print value_q_learning
			# 	print action_reward
			# 	print dt_forward_post
			# 	print dt_forward_post * agent_desired_speed / dt_normal
			# 	assert(0)
			############################################################################
			if value_q_learning > value_bnd:
				value_q_learning = value_bnd

			# compute value using actual time to reach goal
			if (not if_other_collided) and min_dist_2_others[-1] > 0 and \
				np.isnan(time_2_goal_vec[i]) and \
				(abs(agent_centric_states[i,0] - agent_centric_states[-1,0]) < 1.0) \
				and (abs(agent_centric_states[i,0] - agent_centric_states[-1,0]) < \
					1.0 * agent_centric_states[0,1]): # stuck
				# print 'min_dist_2_others[-1] > 0', min_dist_2_others[-1] > 0
				# value_q_learning = value_q_learning * 0.8 * (agent_centric_states[i,2] / agent_centric_states[i,1])
				value_q_learning = 0.01
				# value_q_learning = max(0.01, value_q_learning - 0.2)
				if_stuck = True
				if_stuck_counter += 1



			# if trajectory is bad 
			# vehicle thinks it can reach goal faster than it actually did 
			# if not np.isnan(time_2_goal_vec[i]) and value_q_learning > EPS:
			# 	agent_desired_speed = agent_centric_states[0,1]
			# 	time_2_goal_value = np.log(value_q_learning) / np.log(gamma)  * dt_normal / max(EPS, agent_desired_speed)
			# 	if time_2_goal_value < time_2_goal_vec[i] - 1.0 or time_2_goal_value < time_2_goal_vec[i] * 0.8:
			# 		value_q_learning *= 0.9

					# print 'time_2_goal_value', time_2_goal_value
					# print 'i', i
					# print 'time_2_goal_vec[i]', time_2_goal_vec[i]
					# raw_input()

			# if np.min(action_rewards[i:]) > -EPS:
			# 	value = max(value_q_learning, value_reach_goal)
			# else:
			value = value_q_learning
			
			# value = max(value_q_learning, value_reach_goal)

			# penalize if the other agent took a lot more time
			# num_other_agents = len(other_agents_extra_time)
			# for tt, other_agent_states in enumerate(other_agents_states):
			# 	offset = 7 + 2 + tt * 8
			# 	dist_2_other = np.linalg.norm(agent_centric_states[i, offset:offset+2])
			# 	other_dist_2_goal = np.linalg.norm(other_agent_states[-1, 0:2]-other_agent_states[-1, 6:8])
			# 	agent_speed = agent_centric_states[0, 1]
			# 	other_agent_speed = other_agent_states[0, 5]
			# 	other_extra_time = other_agents_extra_time[tt]
			# 	if len(other_extra_time) <= i:
			# 		continue
			# 	if np.isnan(time_2_goal_vec[i]) == False and other_dist_2_goal <= DIST_2_GOAL_THRES and \
			# 		other_extra_time[i] - extra_time[i] > 0.5 \
			# 		and time_2_goal_vec[i] > 1.0 and dist_2_other < 2.5 \
			# 		and agent_speed > other_agent_speed - 0.2:
			# 		# and np.linalg.norm(other_agent_states[i,2:4]) > 0.5*other_agent_speed:
			# 		# print other_extra_time[i], extra_time[i], dist_2_other, \
			# 		# 	est, other_extra_time[i]-est - extra_time[i]
			# 		penalty = gamma ** (min((other_extra_time[i] - extra_time[i]), 2.0) \
			# 			* agent_desired_speed / dt_normal)
			# 		value *= penalty
			# 		break
			
				# if_extra = True
			# 	# print 'here'
			# 	# print 'other_extra_time[i]', other_extra_time[i]
			# 	# print 'extra_time[i]', extra_time[i] 
			# 	# print penalty
			# 	# raw_input()

			# # to speed up convergence
			# # if iteration < 200 and time_2_goal_vec[i] < dt and dist_2_other > 2.5:
			# # 	value = value_bnd
			Y[counter,0] = max(value, -0.25)
			# if value_q_learning == 0.01:
			# 	X_stuck_pt, Y_stuck_pt = self.createStateSample(X[counter-1,:])
			# 	# print X_stuck.shape, Y_stuck.shape
			# 	X_stack = np.vstack((X_stuck, X_stuck_pt))
			# 	Y_stack = np.vstack((Y_stuck, Y_stuck_pt))
			# 	# print X_stuck_pt, Y_stuck_pt
			
			# print counter
			# if if_stuck_counter > 20:
			# 	break

			# future values
			# agent_state = agent_states[i,:].copy()
			# other_agents_state = [other_agents_states[tt][i,:].copy() for tt in xrange(len(other_agents_states))]
			# state_nn_future, value_future = \
			# 	self.find_intended_future_state_value(agent_state, agent_states[i+1,2:4], other_agents_state, dt_forward_vec[i])
			# X_future[counter,:] = state_nn_future.copy()
			# Y_future[counter,:] = value_future
			# future_value_inds.append(j)
			# # print 'value_future, values[j], dt_forward_vec[i]', value_future, values[j], dt_forward_vec[i]
			# Y_future[i,0] = min(value_future, values[j])
			counter += 1

		# if counter < num_pts:
		# 	print counter
		# 	print num_pts
		# 	raw_input()
		# print counter
		# debug
		# min_dist_2_others = np.min(agent_centric_states[:,[13,21,29]], axis = 1)
		# if np.any(Y[:,0]<EPS) and iteration > 0:
		# if iteration > 0:
		# 	np.set_printoptions(precision=4,formatter={'float': '{: 0.3f}'.format})
		# 	print 'time_2_goal_vec, time_2_goal_bnd, dist_2_other, values, action_rewards, dt_forward, value_bnd, value_train'
		# 	value_bnd = GAMMA ** (agent_centric_states[:,0] / DT_NORMAL)
		# 	print np.vstack((time_2_goal_vec, time_2_goal_bnd, min_dist_2_others, values, action_rewards, \
		# 		dt_forward_vec, value_bnd, X[:,0], Y[:,0])).transpose()
		# 	print min_dist_2_others[-1]
		# 	raw_input() 

		# if traj is too long
		if False and counter > 100:
			stride = int(counter / 100) + 1
			X = X[0:counter:stride,]
			Y = Y[0:counter:stride,]
			agent_centric_states = agent_centric_states[0:counter:stride,:]
			time_vec = time_vec[0:counter:stride]
			values = values[0:counter:stride]
			action_rewards = action_rewards[0:counter:stride]
		else:
			X = X[0:counter,:]
			Y = Y[0:counter,:]
		# print 'counter', counter

		# X_future = X_future[0:counter]
		# Y_future = Y_future[0:counter]
		# Y_min_value = Y[np.clip(np.array(future_value_inds), 0, counter-1)]
		# # print Y_min_value.shape, Y_future.shape
		# Y_future = np.minimum(Y_future, Y_min_value)
		# # print Y_future.shape
		# # print np.hstack((Y_future, Y[np.clip(np.array(future_value_inds), 0, counter-1)]))
		# # raw_input()
		# X = np.vstack((X,X_future))
		# Y = np.vstack((Y,Y_future))

		values_raw = np.squeeze(self.value_net_copy.nn.make_prediction_raw(X))
		# if if_stuck:
		# 	print 'X, Y'
		# 	print np.hstack((X, Y, values_raw[:,np.newaxis]))
		# 	raw_input()
		# print values_raw.shape
		# print values.shape
		min_dist_2_others = np.min(agent_centric_states[:,[13,21,29]], axis = 1)
		values_diff = abs((Y[:,0]-values_raw) / Y[:,0])
		# zero_inds = np.where(abs(Y[:,0])<EPS)[0]
		# if len(zero_inds) > 0:
		# 	print 'wrong', zero_inds, counter
		# 	print X[zero_inds,:]
		# 	print Y[zero_inds,0] 
		# 	print values_raw[zero_inds]
		# 	raw_input()
		# values_diff = abs((Y[:,0]-values[:-1]) / Y[:,0])
		# print Y[:,0].shape
		# print values_diff.shape
		
		###################################################################
		# # method 1
		num_selected_inds = int(len(X)/5)
		inds = np.argpartition(values_diff, -num_selected_inds)[-num_selected_inds:]
		bad_inds = np.where(values_diff>0.1)[0]
		inds = np.union1d(bad_inds, inds)
		rand_inds = np.random.permutation(np.arange(len(X)))[0:num_selected_inds]
		inds = np.union1d(inds, rand_inds)

		# good_inds = np.argpartition(values_diff, num_selected_inds)[:num_selected_inds]
		# inds = np.union1d(inds, good_inds)
		inds = np.arange(len(X))
		###################################################################
		# # method 2
		# all_inds = np.arange(len(X))
		# toward_goal_inds = np.where(abs(X[:,3]) < 0.2)[0]
		# # print 'toward_goal_inds %d' \
		# 	# %(len(toward_goal_inds))

		# far_inds = np.where(min_dist_2_others < 0.3)[0]
		# toward_goal_inds = np.setdiff1d(toward_goal_inds,far_inds)
		# # print 'toward_goal_inds %d, not toward_goal_inds %d, total %d' \
		# 	# %(len(toward_goal_inds), len(X) - len(toward_goal_inds), len(X))
		# # raw_input()
		# bad_inds = np.setdiff1d(all_inds, toward_goal_inds)
		# inds = bad_inds
		# if len(bad_inds) == 0:
		# 	bad_inds = [0]
		# toward_goal_inds_sample = \
		# 	np.random.permutation(toward_goal_inds)[0:len(bad_inds)]
		# inds = np.union1d(bad_inds, toward_goal_inds_sample)
		# # bad_inds_2 = np.where(Y[:,0]<0.6)[0]
		# # inds = np.union1d(inds, bad_inds_2)
		###################################################################
		
		X = X[inds,:]
		Y = Y[inds,:]
		values_diff = values_diff[inds]

		# debug
		# if counter > 300 or if_agent_collided:
		# 	values_bnd = GAMMA ** (X[:,0]/DT_NORMAL)
		# 	values = values[inds]
		# 	print agent_desired_speed
		# 	print values.shape
		# 	np.set_printoptions(edgeitems=4, precision=4,formatter={'float': '{: 0.4f}'.format})
		# 	print 'dist_2_goal, min_dist_2_others, dt, value_bnd, training_value, raw_values, action_rewardsvalues_diff'
		# 	print np.vstack((X[:,0], min_dist_2_others[inds], time_vec[inds], values_bnd, Y[:,0], values, action_rewards[inds], values_diff)).transpose()
		# 	raw_input()

		values_diff = values_diff[:]
		# bellman backup
		X1 = X.copy() 
		Y1 = Y.copy() 
		values_diff1 = values_diff.copy() 
		speed_factors = np.random.rand(len(X1))
		angles_factors = (np.random.rand(len(X1)) - 0.5 ) * 0.1
		X1[:,2] *= speed_factors; X1[:,3] = (X1[:,3] + angles_factors + np.pi) % (np.pi * 2) - np.pi
		X1[:,4] = X1[:,2] * np.cos(X1[:,3])
		X1[:,5] = X1[:,2] * np.sin(X1[:,3])
		X = np.vstack((X,X1))
		Y = np.vstack((Y,Y1))
		values_diff = np.hstack((values_diff, values_diff1))

		# stuck points
		# X = np.vstack((X,X_stuck))
		# Y = np.vstack((Y,Y_stuck))
		if if_stuck:
			bad_states = []
			num_states = len(agent_states); start_ind = max(num_states-1-np.random.randint(20),0)
			bad_states.append(agent_states[start_ind,:])
			for tt in xrange(num_other_agents):
				bad_states.append(other_agents_states[tt][start_ind,:])
			self.bad_testcases_tmp.append(bad_states)
		# else:
		# 	start_ind = np.argmax(extra_time)
		# 	if extra_time[start_ind]>0.7 and \
		# 		np.sum(extra_time[start_ind:max(start_ind+10, len(extra_time)-1)])>3.0:
		# 		print 'added new case'
		# 		bad_states = []
		# 		start_ind = max(0, start_ind-10)
		# 		num_states = len(agent_states)
		# 		bad_states.append(agent_states[start_ind,:])
		# 		for tt in xrange(num_other_agents):
		# 			bad_states.append(other_agents_states[tt][start_ind,:])
		# 		self.bad_testcases_tmp.append(bad_states)
		return X, Y, values_diff

	def trainTestCase(self, agents_state, num_repeat):
		param = self.nn_rl_training_param
		init_pos = agents_state[0][0:2].copy()
		X = []; Y = []; values_diff = []
		if_resolved = False
		for i in xrange(num_repeat):
			if i > 0:
				tmp_counter = 0
				while tmp_counter < 100:
					tmp_counter += 1
					# agents_state[0][0:2] += np.random.rand(2) * 1.0 - 0.5
					agents_state[0][0:2] = np.random.rand(2) * 3.0 - 1.5
					min_dist = min([np.linalg.norm(agents_state[0][0:2]-agents_state[tt][0:2])-\
						agents_state[0][8] - agents_state[tt][8] for tt in xrange(1,len(agents_state))])
					if min_dist > 0:
						break
			traj_raw_multi, time_to_complete = self.value_net.generate_traj_from_states(agents_state, \
				rl_epsilon=0.1, figure_name='no_plot', stopOnCollision=True, ifNonCoop=True)
			if len(traj_raw_multi[0]) <=1:
				# print 'here?'
				continue
			X1, Y1, values_diff1 = self.rawTraj_2_trainingData(traj_raw_multi, param.gamma, 0, ifOnlyFirstAgent=True)

			if len(X1) <=1:
				# print 'here?'
				continue
			if len(X1) > 0:
				# static_inds = np.where(Y1<0.05)[0]
				# rand_heading = np.random.rand(len(static_inds)) * np.pi - np.pi/2.0
				# # print np.random.rand(len(static_inds)).shape
				# # print X1[static_inds,1].shape
				# rand_speed = np.random.rand(len(static_inds)) * X1[0,1]
				# X1[static_inds,2] = rand_speed 
				# X1[static_inds,3] = rand_heading
				# X1[static_inds,4] = rand_speed * np.cos(rand_heading)
				# X1[static_inds,5] = rand_speed * np.sin(rand_heading)
				if len(X) == 0:
					X = X1.copy()
					Y = Y1.copy()
					values_diff = values_diff1
				else:
					X = np.vstack((X, X1.copy()))
					Y = np.vstack((Y, Y1.copy()))
					values_diff = np.hstack((values_diff, values_diff1))
			if i==0 and np.linalg.norm(traj_raw_multi[1][-1,0:2] - traj_raw_multi[1][-1,6:8]) < DIST_2_GOAL_THRES:
				if_resolved = True
			


			# print 'i, len(X), len(X1), if_resolved', i, len(X), len(X1), if_resolved
			# pedData.plot_traj_raw_multi(traj_raw_multi, 'bad_cases', \
			# 	figure_name=self.mode+'bad_case' )
			# print X1[:,0].shape, Y1.shape
			# np.set_printoptions(edgeitems=4, precision=4,formatter={'float': '{: 0.4f}'.format})
			# print 'X1, Y1', np.vstack((X1[:,0],X1[:,1],X1[:,2],X1[:,3],X1[:,4],X1[:,5],Y1[:,0])).transpose()
			# raw_input()

		agents_state[0][0:2] = init_pos

		return traj_raw_multi, X, Y, values_diff, if_resolved


		# if time_to_complete
		


	def createStuckSample(self, x):
		agent_state, other_agents_state = pedData.agentCentricState_2_rawState_noRotate(x)
		actions_theta = self.value_net_copy.find_feasible_actions(agent_state)
		# print agent_state
		# print actions_theta
		# print other_agents_state
		state_values, X_intended = self.value_net_copy.find_next_states_values(agent_state, \
			actions_theta, other_agents_state,if_return_future_states=True)
		best_action_ind = np.argmax(state_values)
		X = np.delete(X_intended,best_action_ind, axis=0)
		Y = np.delete(state_values,best_action_ind)
		# print X.shape, Y.shape, 'here'
		return X, Y[:,np.newaxis]

	def createNeighborSamples(self, X, Y, minibatch_size):
		num_states = 7 + 8 * (self.num_agents - 1)
		num_inputs = len(X)
		num_pts = 0
		X_pad = np.zeros((num_pts,num_states));
		while (num_pts < minibatch_size):
			ind = np.random.randint(num_inputs) 

	# def createSetOfTestCases(self, num_cases):
	# 	test_cases = 



def preset_testCases():
	test_cases = []
	# hardcoded to be 4 agents for now
	test_cases.append(np.array([[-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],\
							[3.0, 0.0, -3.0, 0.0, 1.0, 0.3], \
							[-3.0, -3.0, 3.0, -3.0, 1.0, 0.3], \
							[3.0, -3.0, -3.0, -3.0, 1.0, 0.3] ]))
	test_cases.append(np.array([[-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],\
							[0.0, -3.0, 0.0, 3.0, 1.0, 0.5], \
							[3.0, 0.0, -3.0, 0.0, 1.0, 0.5],\
							[0.0, 3.0, 0.0, -3.0, 1.0, 0.5] ]))	
	test_cases.append(np.array([[-2.0, -1.5, 2.0, 1.5, 1.0, 0.5],\
								[-2.0, 1.5, 2.0, -1.5, 1.0, 0.5],\
								[-2.0, -4.0, 2.0, -4.0, 0.9, 0.35], \
								[2.0, -4.0, -2.0, -4.0, 0.85, 0.45] ]))
	test_cases.append(np.array([[-4.0, 0.0, 4.0, 0.0, 1.0, 0.4], \
							[-2.0, 0.0, 2.0, 0.0, 0.5, 0.4], \
							[-4.0, -4.0, 4.0, -4.0, 1.0, 0.4], \
							[-2.0, -4.0, 2.0, -4.0, 0.5, 0.4]]))
	return test_cases

# def preset_testCases():
# 	test_cases = []
# 	# fixed speed and radius
# 	test_cases.append(np.array([[-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],\
# 								[3.0, 0.0, -3.0, 0.0, 1.0, 0.3]]))
# 	test_cases.append(np.array([[-3.0, 0.0, 3.0, 0.0, 1.0, 0.3], \
# 								[3.0/1.4,-3.0/1.4,-3.0/1.4,3.0/1.4, 1.0, 0.3]]))
# 	test_cases.append(np.array([[-2.0, -1.5, 2.0, 1.5, 1.0, 0.5],\
# 								[-2.0, 1.5, 2.0, -1.5, 1.0, 0.5]]))
# 	test_cases.append(np.array([[-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],\
# 								[0.0, -3.0, 0.0, 3.0, 1.0, 0.5]]))	
# 	# variable speed and radius
# 	test_cases.append(np.array([[-2.5, 0.0, 2.5, 0.0, 1.0, 0.3],\
# 								[2.5, 0.0, -2.5, 0.0, 0.8, 0.4]]))
# 	test_cases.append(np.array([[-3.0, 0.0, 3.0, 0.0, 0.6, 0.5], \
# 								[3.0/1.4,-3.0/1.4,-3.0/1.4,3.0/1.4, 1.0, 0.4]]))
# 	test_cases.append(np.array([[-2.0, 0.0, 2.0, 0.0, 0.9, 0.35], \
# 								[2.0,0.0,-2.0,0.0, 0.85, 0.45]]))
# 	test_cases.append(np.array([[-4.0, 0.0, 4.0, 0.0, 1.0, 0.4], \
# 								[-2.0, 0.0, 2.0, 0.0, 0.5, 0.4]]))
# 	return test_cases

def generate_figures_at_iters(nn_rl, file_dir, iters):
	format_str = '.png'
	for i in iters:
		v_net_filename = "%d_agents_policy_iter_"%nn_rl.num_agents + str(i) + ".p"
		nn_rl.loadFromFile(file_dir, v_net_filename)
		folder_dir = file_dir + "/../../pickle_files/multi/" + nn_rl.value_net.mode + \
			'_' + nn_rl.passing_side + "/figures"
		figure_name = "iter"+str(i)
		nn_rl.plot_test_cases(folder_dir, figure_name, format_str)


# load NN_rl
def load_NN_rl(file_dir, num_agents, mode, passing_side, filename=None, ifSave=False):
	''' initializing RL training param ''' 
	try:
		nn_rl_training_param = pickle.load(open(\
			file_dir+"/../../pickle_files/multi/nn_rl_training_param.p", "rb"))
		assert(0)
	except:
		num_episodes = 2001
		numpts_per_eps = 1000
		expr_size = numpts_per_eps * 10 
		gamma = 0.97 
		sgd_batch_size = 500
		greedy_epsilon = 0.1

		nn_rl_training_param = NN_rl_training_param(num_episodes, numpts_per_eps, expr_size, \
				gamma, sgd_batch_size, greedy_epsilon)
		pickle.dump(nn_rl_training_param, open(\
			file_dir+"/../../pickle_files/multi/nn_rl_training_param.p", "wb"))

	nn_training_param = pickle.load(open(file_dir+"/../../pickle_files/multi/nn_training_param.p", "rb"))
	
	# filename_nn = "/%d_agents_policy_iter_"%self.num_agents + str(150) + ".p"
	value_net = nn_nav.load_NN_navigation_value(file_dir, num_agents, mode, passing_side, filename=filename)
	nn_rl = NN_rl(nn_rl_training_param, nn_training_param, value_net, ifSave=ifSave)
	nn_rl.passing_side = passing_side
	# if folder doesn't exist, create it
	directory = file_dir+"/../../pickle_files/multi/"+value_net.mode+'_'+passing_side
	if not os.path.exists(directory):
		os.makedirs(directory)
		os.makedirs(directory+'/figures')
		os.makedirs(directory+'/RL_selfplay')
	return nn_rl


if __name__ == '__main__':
	print('hello world from nn_rl.py')
	file_dir = os.path.dirname(os.path.realpath(__file__))
	plt.rcParams.update({'font.size': 18})

	# load nn_rl
	num_agents = 4
	mode = 'no_constr'; passing_side = 'none'
	# mode = 'no_constr'; passing_side = 'left'
	# mode = 'no_constr'; passing_side = 'right'
	# mode = 'rotate_constr'; passing_side = 'none'
	# mode = 'rotate_constr'; passing_side = 'left'
	# mode = 'rotate_constr'; passing_side = 'right'

	if_save = True
	nn_rl = load_NN_rl(file_dir, num_agents, mode, passing_side, ifSave=if_save)
	# nn_rl = load_NN_rl(file_dir, num_agents, mode, passing_side, filename="/%d_agents_policy_iter_"%num_agents + str(200) + ".p")
	# nn_rl.value_net.passing_side = passing_side
	# nn_rl.value_net.passing_side = passing_side
	# nn_rl.value_net_copy.passing_side = passing_side
	
	# folder_dir = file_dir+"/../../pickle_files/multi/"+mode+"/figures"
	# nn_rl.plot_test_cases(folder_dir, 'initial', '.png')
	
	# load a different value net
	# mode="no_constr"; filename = "%d_agents_policy_iter_500.p"%self.num_agents
	# nn_rl.loadFromFile(file_dir, filename)
	
	# plot current value net
	# nn_rl.evaluate_current_network(plot_mode='all')
	
	# train neural network
	# test_case = nn_nav.generate_random_test_case(6, \
	# 				np.array([0.5,1.2]), np.array([0.3, 0.5]))
	# nn_rl.straightLine_traj(test_case, 0.97, figure_name='straight line training traj')
	# raw_input()
	nn_rl.deep_RL_train(file_dir)
	# nn_rl.debug_rawTraj_2_trajStats()

	# generating figures and save to file
	if if_save == True:
		generate_figures_at_iters(nn_rl, file_dir, \
			[50, 100, 250, 400, 500, 800, 1000])
	plt.show()
