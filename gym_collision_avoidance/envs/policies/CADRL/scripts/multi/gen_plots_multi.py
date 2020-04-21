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
import matplotlib
import copy
import time

import nn_navigation_value_multi as nn_nav
import nn_rl_multi as nn_rl
import pedData_processing_multi as pedData
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
GAMMA = gb.RL_gamma
DT_NORMAL = gb.RL_dt_normal

''' genenerate plots (or instructions for how to generate plots) '''

# need to first generate test cases using gen_results.py
# then generate trajs using rvo (roslaunch rvo_ros rvo_traj_gen_multi.launch)
# then generate trajs using neural nets on the same test cases 
#   (nn_navigation_value_multi.py)

# plot trajectories to a number of test cases at various training episodes
def plot_training_process(file_dir, format_str, num_agents):
	plt.rcParams.update({'font.size': 30})
	save_folder_dir = file_dir + "/../../pickle_files/multi/results/figures/"

	# plot rvo trajs
	# rvo_trajs_filename = file_dir + \
	#   "/../../pickle_files/multi/results/hard_rvo_trajs_raw.p"
	# rvo_trajs = pickle.load(open(rvo_trajs_filename, "rb"))
	# for i, traj in enumerate(rvo_trajs):
	#   nn_nav_multi.plot_traj_raw_multi(traj, '')
	#   plt.title('')
	#   file_name = 'rvo_case_'+str(i)+format_str
	#   plt.savefig(save_folder_dir+file_name,bbox_inches='tight')
	#   print 'saved', file_name

	# plot neural network trajs
	test_cases = pickle.load(open(file_dir + \
		"/../../pickle_files/multi/results/%d_agents_hard_test_cases.p"%num_agents, "rb"))
	iterations = [0, 50, 500, 800, 1000]
		# load multiagent neural network
	# load nn_rl
	mode = 'no_constr'; passing_side = 'right'
	for iteration in iterations:
		# mode = 'rotate_constr'
		filename = "%d_agents_policy_iter_%d.p"%(num_agents, iteration)
		# filename=None
		NN_value_net_multi = nn_nav.load_NN_navigation_value(file_dir, mode, passing_side, filename=filename)

		nn_trajs_filename = file_dir + \
			"/../../pickle_files/multi/results/%d_agents_hard_nn_trajs_iter_%d.p"%(num_agents,iteration)
		for i, test_case in enumerate(test_cases):
			traj, time_to_complete = \
				NN_value_net_multi.generate_traj(test_case, figure_name='%_agents_network'%num_agents)
			pedData.plot_traj_raw_multi(traj, '', figure_name='training_process')
			plt.title('')
			file_name = '%d_agents_nn_iter_%d_case_%d'%(num_agents,iteration,i)+format_str
			plt.savefig(save_folder_dir+file_name,bbox_inches='tight')
			print('saved', file_name)

# load training score file and plot training score (value) as a function of episodes
def plot_convergence(file_dir, format_str, num_agents):
	plt.rcParams.update({'font.size': 30})

	save_folder_dir = file_dir + "/../../pickle_files/multi/results/figures/"

	score_fname = file_dir+"/../../pickle_files/multi/no_constr" \
		+"/RL_training_score.p"
	scores = pickle.load(open(score_fname,"rb"))
	stride = 5

	fig = plt.figure('training score', figsize=(10,8))
	plt.clf()

	test_cases = nn_rl.preset_testCases()
	episodes = stride * np.arange(len(scores))
	num_cases = scores[0].shape[0] / 3
	scores_np = np.asarray(scores)

	time_vec = scores_np[:,0:num_cases]
	collision_vec = scores_np[:,num_cases:2*num_cases]
	value_vec = scores_np[:,2*num_cases:3*num_cases]

	color_counter = 0
	for i in [6,1,7]:
		test_case = test_cases[i]
		dist_2_goal = np.linalg.norm(test_case[0, 0:2] - test_case[0, 2:4])
		upper_bnd = GAMMA ** (dist_2_goal / DT_NORMAL)
		color = plt_colors[color_counter]
		plt.plot(episodes, value_vec[:,i], c=color, linewidth=2)
		# print upper_bnd
		# plt.plot(episodes, upper_bnd * np.ones(episodes.shape), \
		#   c=color, ls='--', linewidth=2)
		color_counter += 1
		color_counter % 7 
		# plt.plot(episodes, )

	plt.xlabel('episode')
	plt.ylabel('value')
	# plotting style (only show axis on bottom and left)
	ax = plt.gca()
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

	# plt.xlim(0,16)
	# plt.ylim(0.25,0.5)
	plt.draw()
	plt.pause(0.0001)
	plt.savefig(save_folder_dir+"/convergence"+format_str,bbox_inches='tight')



# plot value function (test case)
# may need to comment out some lines in plot_ped_testCase()
def plot_value_function(file_dir, format_str):
	plt.rcParams.update({'font.size': 36})
	save_folder_dir = file_dir + "/../../pickle_files/multi/results/figures/"

	# define test case
	dist_to_goal = 2.5; pref_speed = 1.0; cur_speed= 1.0; cur_heading = np.pi/5.0;
	other_vx = 0.0; other_vy = 1.0; rel_pos_x = 1.5; rel_pos_y = -0.8;
	self_radius = 0.3; other_radius = 0.3; 
	vx = pref_speed * np.cos(cur_heading); 
	vy = pref_speed * np.sin(cur_heading); 
	dist_2_other = np.sqrt(np.array([rel_pos_x, rel_pos_y])) - \
					self_radius-other_radius
	x = [dist_to_goal, pref_speed, cur_speed, cur_heading, \
			other_vx, other_vy, rel_pos_x, rel_pos_y, self_radius, \
			other_radius, self_radius+other_radius, vx, vy, dist_2_other]
	y = 0.5


	# load 2 agent 'no rotate' neural network
	mode = 'no_constr'; passing_side = 'right'
	iteration = 1000
	filename = "twoAgents_policy_iter_%d.p"%iteration
	nn_navigation = nn_nav.load_NN_navigation_value(file_dir, mode, passing_side, filename)
	nn_navigation.plot_ped_testCase(x, y, ' ', \
						'test_case in no_constr')
	plt.subplot(121); plt.title('')
	plt.subplot(122); plt.title('')
	fig = plt.gcf()
	fig.tight_layout()
	file_name = 'value_func_no_constr' + format_str
	plt.savefig(save_folder_dir+file_name,bbox_inches='tight')

	# load 2 agent 'no rotate' neural network
	mode = 'rotate_constr'; passing_side = 'right'
	iteration = 500
	filename = "twoAgents_policy_iter_%d.p"%iteration
	nn_navigation = nn_nav.load_NN_navigation_value(file_dir, mode, passing_side, filename)
	nn_navigation.plot_ped_testCase(x, y, ' ', \
						'test_case rotate_constr')
	plt.subplot(121); plt.title('')
	plt.subplot(122); plt.title('')
	fig = plt.gcf()
	fig.tight_layout()
	file_name = 'value_func_rotate_constr' + format_str
	plt.savefig(save_folder_dir+file_name,bbox_inches='tight')
	

def plot_multi_agent_cases(file_dir, format_str):
	plt.rcParams.update({'font.size': 28})
	save_folder_dir = file_dir + "/../../pickle_files/multi/results/figures/"
	# load multiagent neural network
	# load nn_rl
	# mode = 'no_constr'
	mode = 'rotate_constr'; passing_side = 'right'
	iteration = 1000
	filename = "twoAgents_policy_iter_%d.p"%iteration
	# filename=None
	value_net = nn_nav.load_NN_navigation_value(file_dir, mode, passing_side, filename)
	NN_navigation_multi = nn_nav_multi.NN_navigation_value_multi(value_net)
	
	# six agent swap
	test_cases = nn_nav_multi.preset_testCases()
	traj_raw_multi, time_to_complete = \
		NN_navigation_multi.generate_traj(test_cases[2], figure_name='method 1', method=1)
	# raw_input()
	plt.title('')
	file_name = 'multi_traj_0' + format_str
	plt.savefig(save_folder_dir+file_name,bbox_inches='tight')

	traj_raw_multi, time_to_complete = \
		NN_navigation_multi.generate_traj(test_cases[3], figure_name='method 1', method=1)
	# raw_input()
	plt.title('')
	file_name = 'multi_traj_1' + format_str
	plt.savefig(save_folder_dir+file_name,bbox_inches='tight')
	
	# random test cases 
	for i in xrange(2,10):
		# np.random.seed(seed)
		# seed+=1
		# print 'seed', seed
		# is_end_near_bnd = np.random.binomial(1, 0.5)
		num_agents = 4
		side_length = 3
		test_case = nn_nav_multi.generate_rand_test_case_multi( num_agents, side_length,\
			np.array([0.5, 1.2]), \
			np.array([0.3, 0.5]), is_end_near_bnd=True)
		traj_raw_multi, time_to_complete = \
			NN_navigation_multi.generate_traj(test_case, figure_name='method 1', method=1)
		plt.title('')
		file_name = 'multi_traj_%d'%i + format_str
		plt.savefig(save_folder_dir+file_name,bbox_inches='tight')
		print('generated traj %d' %i)
		raw_input()

	pass

# may need to change color setting in global_var.py
# change all except the last one 
def plot_static_case(file_dir, format_str):
	plt.rcParams.update({'font.size': 34})
	save_folder_dir = file_dir + "/../../pickle_files/multi/results/figures/"
	# load multiagent neural network
	# load nn_rl
	# mode = 'no_constr'; passing_side = 'right'
	mode = 'rotate_constr'; passing_side = 'right'
	iteration = 1000
	filename = "twoAgents_policy_iter_%d.p"%iteration
	# filename=None
	value_net = nn_nav.load_NN_navigation_value(file_dir, mode, passing_side, filename)
	NN_navigation_multi = nn_nav_multi.NN_navigation_value_multi(value_net)

	test_case = np.array([[-3.0, -3.0, 3.0, 3.0, 1.0, 0.3],\
						[-2.0, -2.0, -2.0, -2.0, 1.0, 0.42],\
						[-3.0, -0.0, -3.0, -0.0, 1.0, 0.4],\
						[-1.5, 3.0, -1.5, 3.0, 1.0, 0.5],\
						[0.0, -0.5, 0.0, -0.5, 1.0, 0.4],\
						[0.5, 2.0, 0.5, 2.0, 1.0, 0.5],\
						[0.5, -1.8, 0.5, -1.8, 1.0, 0.41],\
						[3.0, 0.0, 3.0, 0.0, 1.0, 0.36],\
						[2.0, -3.0, 2.0, -3.0, 1.0, 0.37]])

	traj_raw_multi, time_to_complete = \
		NN_navigation_multi.generate_traj(test_case, figure_name='method 2', method=1)
	plt.title('')
	plt.locator_params(axis='y',nbins=4)
	plt.locator_params(axis='x',nbins=6)
	file_name = 'multi_traj_static' + format_str
	plt.savefig(save_folder_dir+file_name,bbox_inches='tight')

def plot_non_coop_case(file_dir, format_str):
	plt.rcParams.update({'font.size': 34})
	save_folder_dir = file_dir + "/../../pickle_files/multi/results/figures/"
	# load multiagent neural network
	# load nn_rl
	mode = 'no_constr'; passing_side = 'right'
	# mode = 'rotate_constr'; passing_side = 'right'
	iteration = 1000
	filename = "twoAgents_policy_iter_%d.p"%iteration
	# filename=None
	value_net = nn_nav.load_NN_navigation_value(file_dir, mode, passing_side, filename)
	NN_navigation_multi = nn_nav_multi.NN_navigation_value_multi(value_net)

	test_case = np.array([[-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],\
						[3.0, 0.0, -3.0, 0.0, 1.0, 0.5]])

	traj_raw_multi, time_to_complete = \
		NN_navigation_multi.generate_traj(test_case, figure_name='method 2', method=1)
	plt.title('')
	plt.locator_params(axis='y',nbins=5)
	plt.locator_params(axis='x',nbins=5)
	file_name = 'multi_traj_non_coop' + format_str
	plt.savefig(save_folder_dir+file_name,bbox_inches='tight')
	pass


def generate_trajs_for_comparison_cases(file_dir, format_str):
	save_folder_dir = file_dir + "/../../pickle_files/multi/results/figures/"
	# generate test cases
	num_agents_vec = [2, 4 ,6, 8]
	side_length_vec = [2.0, 2.5, 3.0, 3.5]
	num_test_cases = 100
	for i, num_agents in enumerate(num_agents_vec):
		np.random.seed(1)
		# side_length = 0.5 + num_agents/2.0
		# print side_length 
		side_length = side_length_vec[i]
		# print side_length
		
		speed_bnds = np.array([0.5, 1.2])
		radius_bnds = np.array([0.3, 0.5])
		test_cases = gen_results.generate_test_cases(num_test_cases, num_agents, \
			side_length, speed_bnds, radius_bnds, is_end_near_bnd=True)
		# print test_cases[85]
		# raw_input()
		# test_cases = nn_rl.preset_testCases() 
		filename = file_dir + "/../../pickle_files/multi/results/%d_agents_test_cases.p"%num_agents
		pickle.dump(test_cases, open(filename, "wb"))
		print('saved %s' %filename)

	# generate multiagent trajectories using neural networks
	mode = 'no_constr'; passing_side = 'right'
	# mode = 'rotate_constr'; passing_side = 'right'
	iteration = 1000
	filename = "twoAgents_policy_iter_%d.p"%iteration
	# filename=None
	value_net = nn_nav.load_NN_navigation_value(file_dir, mode, passing_side, filename)
	NN_navigation_multi = nn_nav_multi.NN_navigation_value_multi(value_net)

	# generate trajs
	for num_agents in num_agents_vec:
		tc_filename = file_dir+"/../../pickle_files/multi/results/%d_agents_test_cases.p" % num_agents
		save_filename = None
		test_cases = pickle.load(open(tc_filename, "rb"))
		# hard test cases for plotting
		# test_cases = pickle.load(open(file_dir + \
		#   "/../../pickle_files/multi/results/hard_test_cases.p", "rb"))
		# save_filename = 'hard_nn_trajs_iter_%d.p' % iteration
		NN_navigation_multi.generate_trajs_for_testcases(test_cases, file_dir=file_dir, \
			filename=save_filename)
		print("finished generating trajs for %s" % tc_filename)

def generate_trajs_for_training_multinet(file_dir, format_str, num_agents, num_test_cases):
	save_folder_dir = file_dir + "/../../pickle_files/multi/results/figures/"
	# generate test cases
	# np.random.seed(1)
	side_length = 2.5 + num_agents / 4.0
	speed_bnds = np.array([0.1, 1.5])
	radius_bnds = np.array([0.3, 0.5])
	test_cases = gen_results.generate_test_cases(num_test_cases, num_agents, \
		side_length, speed_bnds, radius_bnds, is_end_near_bnd=True)

	# generate multiagent trajectories using neural networks
	mode = 'no_constr'; passing_side = 'right'
	# mode = 'rotate_constr'; passing_side = 'right'
	iteration = 1000
	filename = "twoAgents_policy_iter_%d.p"%iteration
	# filename=None
	value_net = nn_nav.load_NN_navigation_value(file_dir, mode, passing_side, filename)
	NN_navigation_multi = nn_nav_multi.NN_navigation_value_multi(value_net)

	# generate trajs
	tc_filename = "%d_agents_cadrl_raw.p" % num_agents
	save_filename = 'multi_training_init'
	NN_navigation_multi.generate_trajs_for_testcases(test_cases, file_dir=file_dir, \
		filename=save_filename)
	print("finished generating trajs for %s" % tc_filename)


def plot_comparison_multi_agent_traj(file_dir, format_str):
	plt.rcParams.update({'font.size': 36})
	save_folder_dir = file_dir + "/../../pickle_files/multi/results/figures/"
	num_agents = 4

	# load 4 agent trajs
	traj_names = ['rvo', 'no_constr', 'rotate_constr']
	trajs = []
	for i in xrange(len(traj_names)):
		filename = file_dir + \
			"/../../pickle_files/multi/results/%d_agents_%s_trajs_raw.p"%(num_agents,traj_names[i])
		trajs.append(pickle.load(open(filename, "rb")))

	# find traj for plotting
	extra_time = pickle.load(open(file_dir + \
		"/../../pickle_files/multi/results/%d_agents_extra_time.p"%num_agents, "rb"))
	for i in xrange(len(extra_time[0])):
		if extra_time[0][i] > extra_time[1][i] + 2 and \
			extra_time[0][i] > extra_time[2][i] + 2:
			# plot trajs
			for j, traj_name in enumerate(traj_names):
				# rvo agent order flipped
				if traj_name == 'rvo':
					traj = []; traj.append(trajs[j][i][0])
					for kk in xrange(num_agents):
						traj.append(trajs[j][i][num_agents-kk])
				else:
					traj = trajs[j][i]
				nn_nav_multi.plot_traj_raw_multi(traj, traj_name)
				plt.title('')
				file_name = 'comp_multi_traj_%s'%traj_name + format_str
				plt.savefig(save_folder_dir+file_name,bbox_inches='tight')

			break



def plot_comparison_cases(file_dir, format_str):
	plt.rcParams.update({'font.size': 36})
	save_folder_dir = file_dir + "/../../pickle_files/multi/results/figures/"
	num_agents_vec = [2,4,6,8]
	# num_agents_vec = [4,6,8]
	rvo_offset_vec = [0.235, 0.498, 0.721, 0.925]
	# rvo_offset_vec = [0.498, 0.739, 0.925]

	for i, num_agents in enumerate(num_agents_vec):
		plt.rcParams.update({'font.size': 36})
		print('---')
		gen_results.plot_stats(file_dir, num_agents, rvo_offset_vec[i])
		plt.title('')
		file_name = '%d_agents_comparison'%num_agents
		plt.savefig(save_folder_dir+file_name,bbox_inches='tight')
		print('generated %s'%file_name) 

def generate_rvo_offset_cases(file_dir, format_str):
	# generate test cases
	num_agents_vec = [2,4,6,8]
	num_test_cases = 10
	for num_agents in num_agents_vec:
		test_case = []
		for i in xrange(num_agents):
			test_case.append(np.array([-2.0, i * 3.0, 2.0, i * 3.0, 1.0, 0.3])) 

		test_cases = []
		for i in xrange(num_test_cases):
			test_cases.append(test_case)
		# test_cases = nn_rl.preset_testCases() 
		filename = file_dir + "/../../pickle_files/multi/results/%d_agents_rvo_offset_test_cases.p"%num_agents
		pickle.dump(test_cases, open(filename, "wb"))
		print('saved %s' %filename)

def genenerate_intersection_cases(file_dir, format_str):
	# generate_test_cases
	sl = 2 # side_length
	angles_vec = np.linspace(-np.pi/6.0*5.0, 0.0, num=31)
	angles_vec = np.squeeze(np.matlib.repmat(angles_vec, 1, 5))
	print(angles_vec.shape)
	# print angles_vec
	test_cases = []
	for angle in angles_vec:
		test_case = np.array([[-sl, 0.0, sl, 0.0, 1.0, 0.3], \
								[sl*np.cos(angle), sl*np.sin(angle), \
								sl*np.cos(angle+np.pi), sl*np.sin(angle+np.pi),\
								1.0, 0.3]])
		test_cases.append(test_case)
	filename = file_dir + "/../../pickle_files/multi/results/intersection_test_cases.p"
	pickle.dump(test_cases, open(filename, "wb"))
	print('saved %s' %filename)

	# generate multiagent trajectories using neural networks
	mode_vec = ['no_constr', 'rotate_constr']
	passing_side = 'right'
	for mode in mode_vec:
		# load neural network
		iteration = 1000
		filename = "twoAgents_policy_iter_%d.p"%iteration
		# filename=None
		value_net = nn_nav.load_NN_navigation_value(file_dir, mode, passing_side, filename)
		NN_navigation_multi = nn_nav_multi.NN_navigation_value_multi(value_net)

		print(test_cases)
		# generate trajs
		save_filename = 'intersection_%s_trajs_raw.p' % mode
		NN_navigation_multi.generate_trajs_for_testcases(test_cases, file_dir=file_dir, \
			filename=save_filename)
		print("finished generating intersection trajs for %s" % mode)

def plot_compare_intersection_cases(file_dir, format_str):
	plt.rcParams.update({'font.size': 36})
	save_folder_dir = file_dir + "/../../pickle_files/multi/results/figures/"
	# load trajs
	traj_names = ['rvo', 'no_constr', 'rotate_constr']
	legend_traj_names = ['$ORCA$', '$CADRL$', '$CADRL\, w/\, cstr$']
	trajs = []
	trajs_stats = []
	for i in xrange(len(traj_names)):
		filename = file_dir + \
			"/../../pickle_files/multi/results/intersection_%s_trajs_raw.p"%(traj_names[i])
		trajs.append(pickle.load(open(filename, "rb")))
		stats = gen_results.compute_trajs_stats(trajs[-1])
		trajs_stats.append(stats)

	# compute stats
	# hardcoded to agree with genenerate_intersection_cases()
	angles_vec = np.linspace(-np.pi/6.0*5.0, 0.0, num=31)
	angles_vec = - angles_vec * 180 / np.pi
	num_test_cases = len(angles_vec)
	stats = np.zeros((num_test_cases, len(traj_names)+1))
	# test case i
	for i in xrange(num_test_cases):
		stats[i,0] = 4 #(4 meters / 1.0 m/s)
		# method j [e.g. rvo]
		for j in xrange(len(traj_names)):
			# print i,j
			total_mean = 0.0
			for k in xrange(5):
				total_mean += np.mean(trajs_stats[j][i+num_test_cases*k][0])
			stats[i,j+1] = total_mean / 5.0

	stats[:,1] -= 0.235 / 2.0  


	# plot time to reach goals
	fig = plt.figure(figsize=(10, 8))
	legend_lines = []
	legend_line, = plt.plot(angles_vec, stats[:,1]-stats[:,0], linewidth=2, color=plt_colors[0])
	legend_lines.append(legend_line)
	legend_line, = plt.plot(angles_vec, stats[:,2]-stats[:,0], linewidth=2, color=plt_colors[1])
	legend_lines.append(legend_line)
	legend_line, = plt.plot(angles_vec, stats[:,3]-stats[:,0], linewidth=2, color=plt_colors[2])
	legend_lines.append(legend_line)

	plt.xlabel('angle $\\alpha$ (deg)')
	plt.ylabel('extra time $\\bar{t}_e$ (s)')
	leg = plt.legend(legend_lines, legend_traj_names, numpoints=1, \
			loc='upper left',fontsize=26, frameon=False)
	# leg.get_frame().set_edgecolor([0.8, 0.8, 0.8])
	ax = plt.gca()
	plt.locator_params(axis='y',nbins=4)
	plt.locator_params(axis='x',nbins=4)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.draw()

	file_name = 'intersection_stats'+format_str
	plt.savefig(save_folder_dir+file_name, bbox_inches='tight')

def compute_rvo_offset(file_dir, format_str):
	save_folder_dir = file_dir + "/../../pickle_files/multi/results/figures/"
	# generate test cases
	num_agents_vec = [2,4,6,8]
	num_test_cases = 100

	# compute trajs stats
	trajs = []
	trajs_stats = []
	for num_agents in num_agents_vec:
		filename = file_dir + \
			"/../../pickle_files/multi/results/%d_agents_rvo_trajs_offset.p"%num_agents
		trajs.append(pickle.load(open(filename, "rb")))
		stats = gen_results.compute_trajs_stats(trajs[-1])
		trajs_stats.append(stats)
		num_cases = len(stats)
		offset = 0
		for j in xrange(num_cases):
			offset += np.sum(stats[j][0]) - 4.0 * num_agents # min time of each agent is 4.0
		offset = offset / num_cases 
		print('rvo_offset: %d agents with offset %.3f' % (num_agents, offset))

def plot_boxplot_extra_time(file_dir, format_str, plt_colors):
	plt.rcParams.update({'font.size': 36})
	save_folder_dir = file_dir + "/../../pickle_files/multi/results/figures/"
	# generate test cases
	num_agents_vec = [2,4,6,8]

	# import plotting functions
	# from pylab import plot, show, savefig, xlim, figure, \
	#                hold, ylim, legend, boxplot, setp, axes

	# load statistics
	trajs_extra_time = []
	for num_agents in num_agents_vec:
		extra_time_filename = file_dir + \
		"/../../pickle_files/multi/results/%d_agents_extra_time.p"%num_agents
		trajs_extra_time.append(pickle.load(open(extra_time_filename, "rb")))

	# box plot
	fig = plt.figure(figsize=(20, 8), frameon=False)
	ax = plt.axes()
	plt.hold(True)
	# first boxplot pair
	bp = plt.boxplot(trajs_extra_time[0], positions = [1, 2, 3], widths = 0.6)
	setBoxColors(bp, plt_colors)

	# second boxplot pair
	bp = plt.boxplot(trajs_extra_time[1], positions = [5, 6, 7], widths = 0.6)
	setBoxColors(bp, plt_colors)

	# thrid boxplot pair
	bp = plt.boxplot(trajs_extra_time[2], positions = [9, 10, 11], widths = 0.6)
	setBoxColors(bp, plt_colors)

	# fourth boxplot pair
	bp = plt.boxplot(trajs_extra_time[3], positions = [13, 14, 15], widths = 0.6)
	setBoxColors(bp, plt_colors)

	# set axes limits and labels
	plt.xlim(0,16)
	plt.ylim(0,5)
	ax.set_xticklabels(['2 agents', '4 agents', '6 agents', '8 agents'])
	ax.set_xticks([2, 6, 10, 14])
	plt.ylabel('extra time $\\bar{t}_e$ (s)')

	# draw temporary red and blue lines and use them to create a legend
	hB, = plt.plot([1],[1],c=plt_colors[0],linewidth=2)
	hR, = plt.plot([1],[1],c=plt_colors[1],linewidth=2)
	hG, = plt.plot([1],[1],c=plt_colors[2],linewidth=2)
	plt.legend((hB, hR, hG),('$ORCA$', '$CADRL$','$CADRL \, w/ \,cstr$'),fontsize=28,frameon=False)
	hB.set_visible(False)
	hR.set_visible(False)
	hG.set_visible(False)

	ax = plt.gca()
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

	file_name = 'extra_time'+format_str
	plt.savefig(save_folder_dir+file_name, bbox_inches='tight')


# function for setting the colors of the box plots pairs
def setBoxColors(bp, plt_colors):
	plt.setp(bp['boxes'][0], color=plt_colors[0], linewidth=2)
	plt.setp(bp['caps'][0], color=plt_colors[0], linewidth=2)
	plt.setp(bp['caps'][1], color=plt_colors[0], linewidth=2)
	plt.setp(bp['whiskers'][0], color=plt_colors[0], linewidth=2)
	plt.setp(bp['whiskers'][1], color=plt_colors[0], linewidth=2)
	plt.setp(bp['fliers'][0], color=plt_colors[0], ms=20, markeredgewidth=2)
	plt.setp(bp['fliers'][1], color=plt_colors[0], ms=20, markeredgewidth=2)
	plt.setp(bp['medians'][0], color=plt_colors[0], linewidth=2)

	plt.setp(bp['boxes'][1], color=plt_colors[1], linewidth=2)
	plt.setp(bp['caps'][2], color=plt_colors[1], linewidth=2)
	plt.setp(bp['caps'][3], color=plt_colors[1], linewidth=2)
	plt.setp(bp['whiskers'][2], color=plt_colors[1], linewidth=2)
	plt.setp(bp['whiskers'][3], color=plt_colors[1], linewidth=2)
	plt.setp(bp['fliers'][2], color=plt_colors[1], ms=20, markeredgewidth=2)
	plt.setp(bp['fliers'][3], color=plt_colors[1], ms=20, markeredgewidth=2)
	plt.setp(bp['medians'][1], color=plt_colors[1], linewidth=2)

	plt.setp(bp['boxes'][2], color=plt_colors[2], linewidth=2)
	plt.setp(bp['caps'][4], color=plt_colors[2], linewidth=2)
	plt.setp(bp['caps'][5], color=plt_colors[2], linewidth=2)
	plt.setp(bp['whiskers'][4], color=plt_colors[2], linewidth=2)
	plt.setp(bp['whiskers'][5], color=plt_colors[2], linewidth=2)
	plt.setp(bp['fliers'][4], color=plt_colors[2], ms=20, markeredgewidth=2)
	plt.setp(bp['fliers'][5], color=plt_colors[2], ms=20, markeredgewidth=2)
	plt.setp(bp['medians'][2], color=plt_colors[2], linewidth=2)

if __name__ == '__main__':
	print('hello world from gen_plots_multi.py')
	file_dir = os.path.dirname(os.path.realpath(__file__))
	# plt.rcParams.update({'font.size': 28}) # 28 for three figures
	# plt.rcParams.update({'font.size': 36}) # 36 for four figures
	# plt.rcParams.update({'font.size': 26})
	format_str = '.png'


	# if folder doesn't exist, create it
	directory = file_dir+"/../../pickle_files/multi/results/figures"
	if not os.path.exists(directory):
		os.makedirs(directory)

	# test cases at different iterations
	# plot_training_process(file_dir, format_str)

	# value as a function of score
	# plot_convergence(file_dir, format_str)

	# rotation example
	# plot_rotate_constr(file_dir, format_str)

	# plot value function
	# plot_value_function(file_dir, format_str)

	# plot passing side 
	# plot_passing_side(file_dir, format_str)

	# multiagent test cases
	# plot_multi_agent_cases(file_dir, format_str)

	# multiagent traj comparison
	# plot_comparison_multi_agent_traj(file_dir, format_str)

	# multiagent static test case
	# plot_static_case(file_dir, format_str)

	# multiagent non-cooperative test case
	# plot_non_coop_case(file_dir, format_str)

	# generate comparison plots
	# generate_trajs_for_comparison_cases(file_dir, format_str)

	# plot comparisonn cases
	# plot_comparison_cases(file_dir, format_str)

	# to determine rvo offset time due to using ROS
	# generate_rvo_offset_cases(file_dir, format_str)
	# compute_rvo_offset(file_dir, format_str)

	# plot box whisker of extra time 
	# plot_boxplot_extra_time(file_dir, format_str, plt_colors)

	# generate intersection test cases
	# genenerate_intersection_cases(file_dir, format_str)
	# plot_compare_intersection_cases(file_dir, format_str)

	plt.show()