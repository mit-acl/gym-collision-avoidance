#!/usr/bin/env python
import sys
sys.path.append('../neural_networks')

import numpy as np
import numpy.matlib
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import copy
import os
import time

import neural_network_regr_multi as nn
import nn_navigation_value_multi as nn_nav
import pedData_processing_multi as pedData
import nn_rl_multi as nn_rlearning
import global_var as gb

# setting up global variables
COLLISION_COST = gb.COLLISION_COST
DIST_2_GOAL_THRES = gb.DIST_2_GOAL_THRES
GETTING_CLOSE_PENALTY = gb.GETTING_CLOSE_PENALTY
GETTING_CLOSE_RANGE = gb.GETTING_CLOSE_RANGE
EPS = gb.EPS
# terminal states
NON_TERMINAL = gb.NON_TERMINAL
COLLIDED = gb.COLLIDED
REACHED_GOAL = gb.REACHED_GOAL
# plotting colors
plt_colors = gb.plt_colors
gamma = gb.RL_gamma
dt_normal = gb.RL_dt_normal

np.set_printoptions(precision=4)

def evaluate_on_test_case(test_case, nn_rl, if_plot=True):
	traj_raw_multi, time_to_complete = \
		nn_rl.value_net.generate_traj(test_case, figure_name='no_plot',stopOnCollision=True)

	path_time = np.sum(time_to_complete)
	# plotting (debugging)
	# a1_speed = np.linalg.norm(traj_raw[0,5:7])
	# a2_speed = np.linalg.norm(traj_raw[0,14:16])
	# a1_len = np.sum(np.linalg.norm(traj_raw[0:-1, 1:3] - traj_raw[1:, 1:3], axis=1))
	# a2_len = np.sum(np.linalg.norm(traj_raw[0:-1, 10:12] - traj_raw[1:, 10:12], axis=1))
	# min_dist = np.amin(np.linalg.norm(traj_raw[:,1:3]-traj_raw[:,10:12], axis=1))  - \
	# 				traj_raw[0,9] - traj_raw[0,18]
	agents_speed, agents_time, agents_len, min_dist = nn_rlearning.compute_plot_stats(traj_raw_multi)
	title_string = 'a%d, t %.2f, sp %.2f, len %.2f \n %s; min_dist %.2f a%d t %.2f, sp %.2f, len %.2f' % \
		(0, agents_time[0], agents_speed[0], agents_len[0], \
			nn_rl.passing_side, min_dist, 1, agents_time[1], agents_speed[1], agents_len[1])
	num_agents = len(traj_raw_multi) - 1
	if num_agents > 2:
		for tt in xrange(2, num_agents):
			agent_string = '\n a%d, t %.2f, sp %.2f, len %.2f' % \
				(tt, agents_time[tt], agents_speed[tt], agents_len[tt])
			title_string += agent_string

	if if_plot: # plot every time case
		pedData.plot_traj_raw_multi(traj_raw_multi, title_string)
		print('from nn_debug')
		for tt in xrange(num_agents):
			print('a%d, t %.2f, sp %.2f, len %.2f' % \
				(tt, agents_time[tt], agents_speed[tt], agents_len[tt]))

	# debug rawTraj_2_sample
	# print 'agent 1', traj_raw[-1,1:10]
	# print 'agent 2', traj_raw[-1,10:19]
	# nn_rl.rawTraj_2_trainingData(traj_raw, 0.97)
	# x, y, traj_raw, agent_1_time, agent_2_time = \
	# 		nn_rl.straightLine_traj(test_case, 0.97, figure_name='straight line training traj')

	# within GETTING_CLOSE_RANGE, not really collided
	if_collided = min_dist<GETTING_CLOSE_RANGE/2.0
							
	return path_time, if_collided, traj_raw_multi, title_string

def stress_test_case(nn_rl, num_run, test_case=None, figure_name_str=None):
	t_start = time.time()
	path_time_vec = np.zeros((num_run,1))
	if_collided_vec = np.zeros((num_run,1))
	max_failed_trajs = 10
	num_failed_trajs = 0

	# whether generate random test cases
	if test_case == None:
		use_random_test_case = True
	else:
		use_random_test_case = False

	# main loop
	for i in xrange(num_run):
		if use_random_test_case:
			side_length = np.random.rand() * (6.0 - 3.0) + 3.0 
			# is_static=1
			test_case = nn_nav.generate_rand_test_case_multi(nn_rl.num_agents, side_length, \
					np.array([0.3,1.2]), np.array([0.3, 0.5]))


		
		path_time, if_collided, traj_raw_multi, title_string = \
			evaluate_on_test_case(test_case, nn_rl, if_plot=False)
		path_time_vec[i] = path_time
		time_to_reach_goal, traj_lengths, min_sepDist, if_completed_vec \
			= pedData.computeStats(traj_raw_multi)
		# print 'min_dist', min_sepDist
		# raw_input()

		if_collided_vec[i] = if_collided
		# print traj_raw
		# plotting
		# first trajectory 
		if i == 0:
			if figure_name_str == None:
				figure_name_str = 'first_plot'
			pedData.plot_traj_raw_multi(traj_raw_multi, title_string, \
				figure_name=figure_name_str)
		
		# if reached goal
		num_agents = len(traj_raw_multi) - 1
		agents_final_states = [traj_raw_multi[t][-1,:] for t in xrange(1, num_agents+1)]
		agents_reached_goal = []

		if_all_successful = True
		for tt in xrange(num_agents):
			agent_final_state = agents_final_states[tt]
			others_final_state = [agents_final_states[t] for t in xrange(num_agents) if t!= tt]
			tmp = nn_rl.value_net.if_terminal_state(\
				agent_final_state, others_final_state)
			agents_reached_goal.append(tmp)
			if tmp != REACHED_GOAL:
				if_all_successful = False

		# if if_collided == False:
		# 	continue

		if i > 0: #if_all_successful == False:
			figure_name_str = 'bad_traj_' + str(num_failed_trajs)
			pedData.plot_traj_raw_multi(traj_raw_multi, title_string, \
				figure_name = figure_name_str)
			num_failed_trajs += 1
			# print 'details of the failed test case'
			# print if_collided, if_agent1_reached_goal==REACHED_GOAL, \
				# if_agent2_reached_goal==REACHED_GOAL
			# print 'agent1_final_state', agent1_final_state 
			# print 'dist_to_goal 1', np.linalg.norm(agent1_final_state[0:2]-agent1_final_state[6:8])
			# print 'agent2_final_state', agent2_final_state 
			# print 'agent2_final 2', np.linalg.norm(agent2_final_state[0:2]-agent2_final_state[6:8])
			num_plts = 1
			# for plotting/debugging
			time_vec = traj_raw_multi[0]
			num_pts = len(time_vec)

			# filter speeds
			agents_filtered_vel_xy = np.zeros((num_pts, num_agents * 2))
			dt_vec = time_vec.copy(); dt_vec[1:] = time_vec[1:] - time_vec[:-1]; dt_vec[0] = dt_vec[1]
			time_past_one_ind = 0
			agents_states = [traj_raw_multi[tt] for tt in xrange(1, num_agents+1)]
			for ii in xrange(num_pts):
				while time_vec[ii] - time_vec[time_past_one_ind] > 0.45:
					time_past_one_ind += 1
				dt_past_vec = dt_vec[time_past_one_ind:ii+1]
				for jj in xrange(num_agents):
					past_vel = agents_states[jj][time_past_one_ind:ii+1,2:5]
					filter_vel_theta = nn_nav.filter_vel(dt_past_vec, past_vel, ifClose=True)
					agents_filtered_vel_xy[ii,jj*2] = filter_vel_theta[0] * np.cos(filter_vel_theta[1])
					agents_filtered_vel_xy[ii,jj*2+1] = filter_vel_theta[0] * np.sin(filter_vel_theta[1])
							
			
			for j in xrange(num_pts):
				agents_state = [traj_raw_multi[tt][j,:] for tt in xrange(1, num_agents+1)]
				# print 'agents_state', agents_state
				# raw_input()
				agents_dist_2_goal = [np.linalg.norm(a_s[0:2]-a_s[6:8]) for a_s in agents_state]

				# within GETTING_CLOSE_RANGE, not really collided
				if_collided_tmp = False
				min_dist_cur_pt = np.inf
				if_too_close = False
				for k in xrange(num_agents):
					for h in xrange(k+1, num_agents):
						dist_tmp = np.linalg.norm(agents_state[k][0:2] - agents_state[h][0:2]) - \
							agents_state[k][8] - agents_state[h][8]
						if dist_tmp < min_dist_cur_pt:
							min_dist_cur_pt = dist_tmp
				if min_dist_cur_pt < GETTING_CLOSE_RANGE:
					if_collided_tmp = True

				if_no_improvement_vec = np.zeros((num_agents,), dtype=bool)
				agent_desired_speed = np.zeros((num_agents,))
				agent_actual_speed = np.zeros((num_agents,))
				for k in xrange(num_agents):
					agent_desired_speed[k] = np.linalg.norm(agents_state[k][2:4])
					agent_actual_speed[k] = agents_state[k][5]
					if_no_improvement_vec[k] = (agent_actual_speed[k] / agent_desired_speed[k]) < 0.2\
						and (agents_dist_2_goal[k] > DIST_2_GOAL_THRES)
				
				if True: #if_collided_tmp and True: #(if_collided or if_no_improvement_1 or if_no_improvement_2):
					print('------ stress_test_case() in nn_debug.py')
					for k in xrange(num_agents):
						print '%dth agent' % k
						agent_state = agents_state[k]
						filtered_others_state = [agents_state[tt].copy() for tt in xrange(num_agents)]
						for tt in xrange(num_agents):
							filtered_others_state[tt][2:4] = agents_filtered_vel_xy[j,tt*2: (tt+1)*2]
						others_state = [filtered_others_state[tt].copy() for tt in xrange(num_agents) if tt!=k]

						# print '~~~ agent %d state'%k, agent_state
						# print 'in debug'
						print('agent_state', agent_state)
						# print 'others_state[0]', others_state[0]
						# print '~~~~~~~~~~'
						# raw_input()
						print('current value', nn_rl.value_net_copy.find_states_values(agent_state, others_state))
						# debug 
						agent_state_copy = agent_state.copy()
						agent_state_copy[2:4] = 0
						print('current value_zero', nn_rl.value_net_copy.find_states_values(agent_state_copy, others_state))


						if if_no_improvement_vec[k]:
							print('no improvement %d, pref speed: %.3f, actual_speed: %.3f ' % \
							(k, agent_desired_speed[k], agent_actual_speed[k]))

						# raw_input()
						if  True: #agents_dist_2_goal[k] > DIST_2_GOAL_THRES:
							ref_prll, ref_orth, state_nn = \
									pedData.rawState_2_agentCentricState(agent_state, others_state, nn_rl.num_agents)
							# print 'before rotate', state_nn
							# a_s, o_s = \
							# 	pedData.agentCentricState_2_rawState_noRotate(state_nn)
							# print 'a_s_after', a_s
							# ref_prll_after, ref_orth_after, state_nn_after = \
							# 		pedData.rawState_2_agentCentricState(a_s, o_s, nn_rl.num_agents)
							# print 'after rotate', state_nn_after
							# raw_input()

							title_string = 'a%d; time: %.3f, dist_2_goal: %.3f' % (k, traj_raw_multi[0][j], agents_dist_2_goal[k])
							# nn_rl.value_net.plot_ped_testCase(state_nn, None, title_string, 'a1 '+str(j))
							plt_colors_custom = copy.deepcopy(plt_colors)
							plt_colors_custom[0] = plt_colors[k]
							plt_colors_custom[1:k+1] = plt_colors[0:k]
							nn_rl.value_net.plot_ped_testCase(state_nn, None, title_string, 'a%d'%k, plt_colors_custom=plt_colors_custom)
							num_plts += 1
					raw_input()
				if if_collided_tmp:
					raw_input()

			print(test_case)
			plt.show()
			
def checkNetworkStructure(value_net):
	num_layers = len(value_net.nn.W)
	print('num_layers', num_layers)
	# test 1: zero blocks
	for i in xrange(num_layers):
		print('layer %d', i)
		multi_net_param = value_net.nn.multiagent_net_param
		layer_info = multi_net_param.layers_info[i]
		next_layer_info = multi_net_param.layers_info[i+1]
		if layer_info.shape[0] > 1 and next_layer_info.shape[0] > 1:
			s_ind = layer_info[0,1]
			s_stride = layer_info[1,1]
			next_s_ind = next_layer_info[0,1]
			next_s_stride = next_layer_info[1,1]
			num_other_agents = layer_info[1,0]
			for j in xrange(num_other_agents):
				for k in xrange(num_other_agents):
					if j == k:
						continue
					print('checking %d, %d block', j, k)
					a = s_ind + j * s_stride
					b = s_ind + (j+1) * s_stride
					c = next_s_ind + k * next_s_stride
					d = next_s_ind + (k+1) * next_s_stride
					try:
						assert(np.all(value_net.nn.W[i][a:b,c:d] == 0))
					except AssertionError:
						print('a, b, c, d',a,b,c,d)
						print(value_net.nn.W[i][a:b,c:d])
						assert(0)
	print('passed test 1')

	# test 2: symmetry block
	for i in xrange(num_layers):
		print('layer %d', i)
		multi_net_param = value_net.nn.multiagent_net_param
		layer_info = multi_net_param.layers_info[i]
		next_layer_info = multi_net_param.layers_info[i+1]
		if layer_info.shape[0] > 1 and next_layer_info.shape[0] > 1:
			s_ind = layer_info[0,1]
			s_stride = layer_info[1,1]
			next_s_ind = next_layer_info[0,1]
			next_s_stride = next_layer_info[1,1]
			num_other_agents = layer_info[1,0]
			# 2.1
			a = 0
			b = s_ind
			c = next_s_ind 
			d = next_s_ind + next_s_stride
			base_block = value_net.nn.W[i][a:b,c:d]
			for j in xrange(1, num_other_agents):
				print('checking %dth h-block', j)
				c = next_s_ind + j * next_s_stride
				d = next_s_ind + (j+1) * next_s_stride
				try:
					assert(np.all(value_net.nn.W[i][a:b,c:d] == base_block))
				except AssertionError:
					print('a, b, c, d',a,b,c,d)
					print(value_net.nn.W[i][a:b,c:d] - base_block)
					assert(0)

			# 2.2 
			a = s_ind
			b = s_ind + s_stride
			c = 0
			d = next_s_ind
			base_block = value_net.nn.W[i][a:b,c:d]
			for j in xrange(1, num_other_agents):
				print('checking %dth v-block', j)
				a = s_ind + j * s_stride
				b = s_ind + (j+1) * s_stride
				try:
					assert(np.all(value_net.nn.W[i][a:b,c:d] == base_block))
				except AssertionError:
					print('a, b, c, d',a,b,c,d)
					print(value_net.nn.W[i][a:b,c:d] - base_block)
					assert(0)

	print('passed test 2')
	# raw_input()



	# compute stats
	# avg_time = np.mean(path_time_vec)
	# min_time = np.amin(path_time_vec)
	# max_time = np.amax(path_time_vec)
	# print 'stats: average time: %.2f, min_time: %.2f, max_time: %.2f' % \
	# 	(avg_time, min_time, max_time)
	# print ' %d success, %d failed, out of %d total' % \
	# 	(num_run - num_failed_trajs, num_failed_trajs, num_run)

if __name__ == '__main__':
	print('hello world from nn_rl_debug_multi.py')
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
	# nn_rl = nn_rlearning.load_NN_rl(file_dir, num_agents, mode, passing_side, ifSave=if_save)
	nn_rl = nn_rlearning.load_NN_rl(file_dir, num_agents, mode, passing_side, \
		filename="/%d_agents_policy_iter_"%num_agents + str(1000) + ".p")
	# checkNetworkStructure(nn_rl.value_net)


	
	# generate_test_cases
	test_cases = nn_rlearning.preset_testCases()

	# test on a single test case
	# evaluate_on_test_case(test_cases[0], nn_rl)
	# for i in xrange(len(test_cases)):
	# 	print 'stress testing test case ', i
	# 	figure_name_string = 'tc %d' % i
	# 	stress_test_case(nn_rl, 100, test_case=test_cases[i], \
	# 		figure_name_str=figure_name_string)

	# find failure test case
	# print 'find failure test case'
	# stress_test_case(nn_rl, 10000)

	# evaluate on a specific test case
	# print 'evaluate on a specific test case'
	# test_case_1 = np.array([[ 0., 0., 0., 0., 0.9371,  0.3891],\
	# 	[-4.3747,-3.0931,2.2546,1.8193,0.5329,0.4372]])
 	# test_case_1 = np.array([[3.1102,4.1044,0.1511,-2.2209,1.1189,0.4875],\
 	# [-1.3528,-1.0405,4.2813,-1.4878,0.3122,0.3964]])
 	# test_case_1 = np.array([[-0.1631,2.5575,2.0771,-0.5941,0.9334,0.454],\
 	# 	[3.1122,-0.4285,-1.8304,1.4933,0.4737,0.3929]])
 	# test_case_1 = np.array([[-0.4849,-2.2173,0.9728,0.9295,1.1130,0.3152], \
 	# 	[1.5232,1.9212,0.2667,-0.4048,1.1847,0.4931]])
 	# test_case_1 = np.array([[ 0.0396,-2.5722,1.1056,2.7615,0.1382,0.4068],\
 	# 	[-0.3462,1.5669,0.7105,-1.5124,0.2512,0.492]])
 	# test_case_1 = np.array([[-0.3880527,-0.14968902,-0.17325492,1.74011012,1.06562294,0.48081785],\
 	# 	[-2.26733982,-1.2264162,2.39980384,2.3181015,0.525117,0.44677194]])
 	# test_case_1 = np.array([[-3.0, 3.0, 3.0, 3.0, 1.0, 0.5], \
	# 	[3.0,-3.0,-3.0,-3.0, 0.2, 0.5]])
	# test_case_1 = np.array([[-2.0, 0.0, 2.0, 0.0, 0.9, 0.35], \
	# 	[2.0,0.0,-2.0,0.0, 0.85, 0.45]])
	# test_case_1 = np.array([[3.2906,2.3044,-2.8087,-3.1036,0.8679,0.4948], \
 			# [-3.0417,-2.6092,-1.1845,-3.1195,0.1038,0.3029]])
 	test_cases = nn_rlearning.preset_testCases()
 	# test_case_1 = test_cases[3]
 	# test_case_1 = nn_nav.generate_rand_test_case_multi(num_agents, 3, \
		# 			np.array([0.1,1.2]), np.array([0.3, 0.5]), is_static=True)

 	test_case_1 = np.array([[0,  1.3,   0,  1.3,  1.0,   0.5],\
 		[0,  0,   0,  0,  1.0,   0.5],\
 		[0,  -1.3,   0,  -1.3,  1.0,   0.5],\
 		[ -3, 0,  3,  0,  1.0,  0.5]])


 	test_case_1 = test_cases[2]
 	# test_case_1 = np.array([[ 1.02, 4.10, 5.78, 4.06, 1.0, 0.3],\
 	#  	[3.79, 4.56, -0.21, 3.56, 1.07, 0.3]])
 	# test_case_1 = np.array([[ -3.0, 0.0, 3.0, 0.0, 1.0, 0.5],\
 	#  	[3.0, 0.0, -3.0, 0.0, 1.0, 0.5]])

 	test_case_1 = np.array([[5.84,  1.72,   0,  0.0,  1.0,   0.5],\
 		[-5,  -4,   -4,  4,  1.0,   0.5],\
 		[-5,  -3,   -3,  -3,  1.0,   0.5],\
 		[ -5, -2,  -2,  -2,  1.0,  0.5]])


	# stress_test_case(nn_rl, 100, test_case=test_case_1)
	# stress_test_case(nn_rl, 100)

	# test on a specific test case
	evaluate_on_test_case(test_cases[3], nn_rl, if_plot=True)

	# debug generate staticCase
	# test_case = nn_nav.generate_random_test_case(3, \
	# 				np.array([0.1,1.2]), np.array([0.3, 0.5]), if_static=True)
	# traj_raw, agent_1_time, agent_2_time, if_collided = \
	# 			nn_rl.staticCase_traj(test_case)

	plt.show()
