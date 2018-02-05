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
NON_TERMINAL=gb.NON_TERMINAL
COLLIDED=gb.COLLIDED
REACHED_GOAL=gb.REACHED_GOAL
# plotting colors
plt_colors = gb.plt_colors
GAMMA = gb.RL_gamma
DT_NORMAL = gb.RL_dt_normal

def generate_test_cases(num_test_cases, num_agents, side_length, speed_bnds, \
						radius_bnds, is_end_near_bnd=False, is_static=False):
	test_cases = []
	counter = 0 # number of valid test cases
	while True:
		test_case = nn_nav.generate_rand_test_case_multi(num_agents, \
			side_length, speed_bnds, radius_bnds, is_end_near_bnd=is_end_near_bnd, \
			is_static = is_static)
		if if_test_case_permit_straight_line(test_case) == False:
			test_cases.append(test_case)
			counter += 1
		if counter >= num_test_cases:
			break
	return test_cases

def if_test_case_permit_straight_line(test_case):
	for i, agent in enumerate(test_case):
		for j in xrange(i):
			x1 = test_case[i][0:2]
			x2 = test_case[i][2:4]
			y1 = test_case[j][0:2]
			y2 = test_case[j][2:4]
			s1 = test_case[i][4]
			s2 = test_case[j][4]
			radius = test_case[i][5] + test_case[j][5]
			if not nn_nav.if_permitStraightLineSoln(x1,x2,s1,y1,y2,s2, radius):
				return False
	return True

def plot_traj(trajs_multi, number=None, title_string=None):
	num_trajs = len(trajs_multi)
	if number == None:
		number = np.random.randint(num_trajs)
	if title_string == None:
		title_string = 'from gen_results.py'
	# print 'number', number
	# print len(trajs_multi)
	# print len(trajs_multi[number])
	# print trajs_multi[number][0]
	# print trajs_multi[number][1]
	# print trajs_multi[number][2]
	# print trajs_multi[number][0].shape
	# print trajs_multi[number][1].shape
	# print trajs_multi[number][2].shape
	# print 'entering plot'
	pedData.plot_traj_raw_multi(trajs_multi[number], title_string)

def compute_test_cases_stats(test_cases):
	num_test_cases = len(test_cases)
	test_cases_stats = []
	for i, test_case in enumerate(test_cases):
		# compute lower bnd
		num_agents = len(test_case)
		lower_bnd = np.zeros((num_agents,))
		for j in xrange(len(test_case)):
			lower_bnd[j] = np.linalg.norm(test_case[j][2:4] \
				- test_case[j][0:2]) / test_case[j][4]
		test_cases_stats.append(lower_bnd)
		# if i == 0:
		# 	print test_case
		# 	print test_cases_stats[-1]
	return test_cases_stats

def compute_trajs_stats(trajs_multi):
	num_test_cases = len(trajs_multi)
	# print num_test_cases
	traj_stats = []
	for i, traj in enumerate(trajs_multi):
		# print i
		# print traj[0].shape
		# print traj[1].shape
		# print traj[2].shape
		# print 'traj_number', i, len(traj), len(traj[0]), len(traj[1])
		if len(traj[0]) == 0:
			print i, traj
		time_to_reach_goal, traj_lengths, min_sepDist, \
			if_completed_vec = pedData.computeStats(traj)
		traj_stats.append([time_to_reach_goal, traj_lengths, \
			min_sepDist, if_completed_vec])
		# if np.all(if_completed_vec) and np.sum(time_to_reach_goal)>4+0.235:
		# 	nn_nav.plot_traj_raw_multi(traj, 'hello')
		# 	raw_input()

	return traj_stats

def plot_stats(file_dir, num_agents, rvo_offset):
	# compute test case stats
	test_cases_filename = file_dir + \
		"/../pickle_files/results/%d_agents_test_cases.p"%num_agents
	test_cases = pickle.load(open(test_cases_filename, "rb"))
	test_cases_stats = compute_test_cases_stats(test_cases)

	# compute trajs stats
	traj_names = ['rvo', 'no_constr', 'rotate_constr', 'multi']
	# traj_names = ['rvo']
	legend_traj_names = ['$ORCA$', '$CADRL$', '$CADRL\, w/\, cstr$', 'multi']
	markers = ['o', '*', 's', '^', 'x', 'D']
	trajs = []
	trajs_stats = []
	for i in xrange(len(traj_names)):
		filename = file_dir + \
			"/../pickle_files/results/%d_agents_%s_trajs_raw.p"%(num_agents,traj_names[i])
		trajs.append(pickle.load(open(filename, "rb")))
		stats = compute_trajs_stats(trajs[-1])
		trajs_stats.append(stats)
		num_collided = 0; num_stuck = 0;
		for j in xrange(len(stats)):
			if stats[j][2] < 0:
				num_collided += 1
			if np.sum(stats[j][3]) < num_agents:
				num_stuck += 1
			#traj_stats.append([time_to_reach_goal, traj_lengths, \
			# min_sepDist, if_completed_vec])
		print(traj_names[i], ' collided %d, stuck %d, total %d ' \
			% (num_collided, num_stuck, len(stats)))

	# plotting
	num_test_cases = len(test_cases_stats)
	stats = np.zeros((num_test_cases, len(traj_names)+1))
	# test case i
	for i in xrange(num_test_cases):
		stats[i,0] = np.mean(test_cases_stats[i])
		# method j [e.g. rvo]
		for j in xrange(len(traj_names)):
			# print i,j
			stats[i,j+1] = np.mean(trajs_stats[j][i][0])
			# print test_cases[i]
			try:
				assert(stats[i,j+1]+EPS > stats[i,0])
			except:
				print('method: %s test case %d' % (traj_names[j], i))
				print('traj num_pts', len(trajs[j][i][0]))
				print(trajs[j][i][0])
				print(trajs[j][i][1])
				print(trajs[j][i][2])
				print(test_cases[i])
				print('a1_lb %.2f, a2_lb %.2f' % \
				(np.linalg.norm(test_cases[i][0][2:4] - test_cases[i][0][0:2])/test_cases[i][0][4], \
				np.linalg.norm(test_cases[i][1][2:4] - test_cases[i][1][0:2])/test_cases[i][1][4])
				time_to_reach_goal, traj_lengths, min_sepDist, \
					if_completed_vec = pedData.computeStats(trajs[j][i]))
				print('actual time', trajs_stats[j][i][0])
				print('lb: %.2f, method: %.2f' % (stats[i,0], stats[i,j+1]))
				plot_traj(trajs[j], number=i)
				raw_input()
				assert(0)

	# plot_traj(trajs[0], number=0, title_string = 'rvo')
	# plot_traj(trajs[1], number=0, title_string = 'no_constr')

	# zero out constant offset in rvo 
	# print 'minus offset', min(stats[:,1]-stats[:,0])
	# stats[:,1] -= min(stats[:,1]-stats[:,0])
	stats[:,1] -= min(rvo_offset/ num_agents,min(stats[:,1]-stats[:,0]))  


	fig = plt.figure(figsize=(10, 8))
	grey = [0.7, 0.7, 0.7]
	legend_lines = []
	print('method \t extra time \t 75th per \t 90th per \t average sepDist')

	trajs_extra_time = []
	for i in xrange(1, len(traj_names)+1):
		# where the agents reached goal and did not collide
		valid_inds = []
		avg_minSep_vec = []
		for j in xrange(len(trajs_stats[i-1])):
			# not stuck and not collided
			if trajs_stats[i-1][j][2] > 0 and np.sum(trajs_stats[i-1][j][3]) == num_agents: 
				valid_inds.append(j)
				avg_minSep_vec.append(trajs_stats[i-1][j][2])

		color_tmp = plt_colors[i-1]
		legend_line, = plt.plot(stats[valid_inds,0], stats[valid_inds,i]-stats[valid_inds,0], \
			color=color_tmp, \
			ls='None', marker=markers[i], mec=color_tmp, fillstyle='none', ms=15, markeredgewidth=1.5)
		legend_lines.append(legend_line)

		traj_names[0] = 'rvo-ocra' # for formatting
		extra_time = stats[valid_inds,i]-stats[valid_inds,0]
		print('%s \t %.2f \t %.2f \t %.2f \t %.3f' % \
			(traj_names[i-1], np.mean(stats[valid_inds,i]-stats[valid_inds,0]), \
				np.percentile(extra_time, 75), \
				np.percentile(extra_time, 90), \
				np.mean(avg_minSep_vec)))
		# plotting 
		max_ind = valid_inds[np.argmax(extra_time)]
		traj = trajs[i-1][max_ind]
		# print 'max_ind', max_ind
		# nn_nav.plot_traj_raw_multi(traj, traj_names[0])
		# raw_input()

		trajs_extra_time.append(extra_time)

	# save to file for later processing
	extra_time_filename = file_dir + \
		"/../pickle_files/results/%d_agents_extra_time.p"%num_agents
	pickle.dump(trajs_extra_time, open(extra_time_filename, "wb"))

	plt.xlabel('straight line time (s)')
	plt.ylabel('extra time $\\bar{t}_e$ (s)')
	plt.title('%d agents stats' % num_agents)
	leg = plt.legend(legend_lines, legend_traj_names, numpoints=1, \
		loc='upper left',fontsize=26)
	leg.get_frame().set_edgecolor([0.8, 0.8, 0.8])

	# plt.axis('equal')

	# plotting style (only show axis on bottom and left)
	ax = plt.gca()
	# plt.locator_params(axis='y',nbins=4)
	# plt.locator_params(axis='x',nbins=4)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.ylim(0,5)
	plt.draw()
	plt.pause(0.0001)


def hard_testCases():
	test_cases = []
	# fixed speed and radius
	test_cases.append(np.array([[-2.0, -1.5, 2.0, 1.5, 1.0, 0.5],\
								[-2.0, 1.5, 2.0, -1.5, 1.0, 0.5]]))
	test_cases.append(np.array([[-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],\
								[3.0, 0.0, -3.0, 0.0, 1.0, 0.5]]))
	# test_cases.append(np.array([[-2.0, 0.01, 2.0, 0.01, 1.0, 0.5],\
	# 							[2.0, 0.0, -2.0, 0.0, 0.5, 0.5]]))

	# test_cases.append(np.array([[-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],\
	# 							[3.0, 0.0, -3.0, 0.0, 1.0, 0.3],\
	# 							[0.0, 3.0, 0.0, -3.0, 1.0, 0.3],\
	# 							[0.0, -3.0, 0.0, 3.0, 1.0, 0.3]]))
	# test_cases.append(np.array([[-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],\
	# 							[3.0, 0.0, -3.0, 0.0, 1.0, 0.3],\
	# 							[3.0*0.5, 3.0*0.8660, -3.0*0.5, -3.0*0.8660, 1.0, 0.3],\
	# 							[-3.0*0.5, -3.0*0.8660, 3.0*0.5, 3.0*0.8660, 1.0, 0.3],\
	# 							[-3.0*0.5, 3.0*0.8660, 3.0*0.5, -3.0*0.8660, 1.0, 0.3],\
	# 							[3.0*0.5, -3.0*0.8660, -3.0*0.5, 3.0*0.8660, 1.0, 0.3]]))
	return test_cases


if __name__ == '__main__':
	print('hello world from gen_results.py')
	file_dir = os.path.dirname(os.path.realpath(__file__))
	plt.rcParams.update({'font.size': 18})

	''' generate dataset '''
	# generate test cases
	# np.random.seed(1)
	# num_test_cases = 100
	# num_agents = 2
	# side_length = 0.5 + num_agents/2.0
	# speed_bnds = np.array([0.5, 1.2])
	# radius_bnds = np.array([0.3, 0.5])
	# test_cases = generate_test_cases(num_test_cases, num_agents, side_length, speed_bnds, \
	# 					radius_bnds, is_end_near_bnd=True)
	# # test_cases = nn_rl.preset_testCases()	
	# filename = file_dir + "/../pickle_files/results/%d_agents_test_cases.p"%num_agents
	# pickle.dump(test_cases, open(filename, "wb"))
	# print 'saved %s' %filename

	# hard test cases for plotting
	test_cases = hard_testCases()
	filename = file_dir + "/../pickle_files/results/hard_test_cases.p"
	pickle.dump(test_cases, open(filename, "wb"))

	''' comparison '''
	# load trajectories
	# np.random.seed(1)
	# rvo_trajs_filename = file_dir + \
	# 	"/../pickle_files/results/%d_agents_rvo_trajs_raw.p"%num_agents
	# rvo_trajs_filename = file_dir + \
	# 	"/../pickle_files/results/hard_rvo_trajs_raw.p"
	# rvo_trajs = pickle.load(open(rvo_trajs_filename, "rb"))
	# plot_traj(rvo_trajs, number=None)
	# plot_traj(rvo_trajs, number=0)
	# plot_traj(rvo_trajs, number=1)
	# plot_traj(rvo_trajs, number=2)

	# # plot stats
	# num_agents = 2
	# plot_stats(file_dir, num_agents)
	plt.show()