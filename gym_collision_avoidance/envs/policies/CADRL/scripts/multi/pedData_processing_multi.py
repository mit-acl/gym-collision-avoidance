#!/usr/bin/env python
import sys
import os
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir+'/../neural_networks')

import numpy as np
import numpy.matlib
import pickle
import matplotlib.pyplot as plt
import time
from gym_collision_avoidance.envs.policies.CADRL.scripts.multi import global_var as gb
import copy

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


# angle_1 - angle_2
# contains direction in range [-3.14, 3.14]
def find_angle_diff(angle_1, angle_2):
	angle_diff_raw = angle_1 - angle_2
	angle_diff = (angle_diff_raw + np.pi) % (2 * np.pi) - np.pi
	return angle_diff

# time to reach goal
# length of each trajectory
# minimum separation distance
def computeStats(traj_raw_multi):
	time_vec = traj_raw_multi[0]
	num_agents = len(traj_raw_multi)-1
	num_pts = time_vec.shape[0]
	# initialize
	time_to_reach_goal = time_vec[-1] * np.ones((num_agents,))
	for i in range(num_agents):
		time_to_reach_goal[i] = max(time_vec[-1], np.linalg.norm(traj_raw_multi[i+1][0, 0:2] \
					- traj_raw_multi[i+1][0, 6:8])/traj_raw_multi[i+1][0, 5])
	traj_lengths = np.zeros((num_agents,))
	min_sepDist = np.linalg.norm(traj_raw_multi[1][0:2] - 
				traj_raw_multi[2][0:2])

	if_completed_vec = np.zeros((num_agents,),dtype=bool)


	# compute time to reach goal
	for kk in range(num_pts):
		for i in range(num_agents):
			# whether reached goal
			if (if_completed_vec[i] == False) and \
				(np.linalg.norm(traj_raw_multi[i+1][kk,0:2] - \
				traj_raw_multi[i+1][kk,6:8]) < DIST_2_GOAL_THRES):
				
				time_to_reach_goal[i] = time_vec[kk]
				traj_lengths[i] = np.sum(np.linalg.norm(traj_raw_multi[i+1][0:kk, 0:2] \
					- traj_raw_multi[i+1][1:kk+1, 0:2], axis=1))
					
				leftover_dist = np.linalg.norm(traj_raw_multi[i+1][kk, 0:2] \
					- traj_raw_multi[i+1][kk, 6:8])
				pref_speed = traj_raw_multi[i+1][0, 5]
				try:
					assert(pref_speed>0.05)
				except:
					print(traj_raw_multi[i+1][0,:])
					assert(0)
				leftover_time = leftover_dist / pref_speed
				
				time_to_reach_goal[i] += leftover_time
				traj_lengths[i] += leftover_dist

				if_completed_vec[i] = True

	# did not reach goals
	for i in range(num_agents):
		if if_completed_vec[i] == False:
			time_to_reach_goal[i] += 5.0

	# compute min_dist
	for i in range(num_agents):
		for j in range(i):
			dist_vec = np.linalg.norm(traj_raw_multi[i+1][:,0:2] - \
				traj_raw_multi[j+1][:,0:2], axis=1)
			min_dist_tmp = np.amin(dist_vec) - traj_raw_multi[i+1][0,8] \
					- traj_raw_multi[j+1][0,8]
			if min_dist_tmp < min_sepDist:
				min_sepDist = min_dist_tmp
	# debugging
	# for i in range(num_agents):
	# 	try:
	# 		assert(time_to_reach_goal[i] >= np.linalg.norm(traj_raw_multi[i+1][0,0:2]-\
	# 			traj_raw_multi[i+1][0,6:8])/traj_raw_multi[i+1][0,5])
	# 		assert(0)
	# 	except:
	# 		print 'time_to_reach_goal[i]', time_to_reach_goal[i]
	# 		print 'lb', np.linalg.norm(traj_raw_multi[i+1][0,0:2]-\
	# 			traj_raw_multi[i+1][0,6:8])/traj_raw_multi[i+1][0,5]
			# assert(0) 
	return time_to_reach_goal, traj_lengths, min_sepDist, if_completed_vec


def computeValue(end_time, cur_time, x, min_dist):
	gamma = GAMMA
	dt_normal = DT_NORMAL
	time_diff = end_time - cur_time
	bnd = x[0] / x[1] 
	try:
		assert(time_diff > bnd - 0.1)
	except:
		print('cur_time', cur_time)
		print('end_time', end_time)
		print('x', x)
		print('time_diff', time_diff)
		print('bnd', x[0]/x[1])
		assert(0)

	time_diff = max(time_diff, bnd)
	pref_speed = x[1]
	# assumed end reward = 1
	value = gamma ** (time_diff * pref_speed / dt_normal)
	if min_dist < 0:
		value = -0.25
	elif min_dist < GETTING_CLOSE_RANGE:
		value -= 0.1

	# value = gamma ** (x[0] / dt_normal)
	return value

# swap the ith agent to the front
def swap_OrderInTrajMulti(traj_raw_multi, i):
	num_agents = len(traj_raw_multi) - 1
	assert(i < num_agents)
	# print '-------'
	# print 'before, traj_raw_multi[1]', traj_raw_multi[1][0,:]
	# print 'before, traj_raw_multi[i+1]', traj_raw_multi[i+1][0,:]
	if i == 0:
		return traj_raw_multi
	else:
		traj_raw_multi_new = copy.deepcopy(traj_raw_multi)
		tmp = traj_raw_multi_new[1]
		traj_raw_multi_new[1] = traj_raw_multi_new[i+1] 
		traj_raw_multi_new[i+1] = tmp

	# print 'after, traj_raw_multi[1]', traj_raw_multi_new[1][0,:]
	# print 'after, traj_raw_multi[i+1]', traj_raw_multi_new[i+1][0,:]
	return traj_raw_multi_new


def findEndTime_first(traj_raw_multi):
	time_vec = traj_raw_multi[0]
	num_pts = time_vec.shape[0]
	traj_raw = traj_raw_multi[1]
	# print traj_raw.shape
	goal = traj_raw[0,6:8]
	for i in range(num_pts):
		dist = np.linalg.norm(traj_raw[i,0:2] - goal)
		# print dist, time_vec[i], dist/traj_raw[0,5], time_vec[i] + dist / traj_raw[0,5]
		if dist < DIST_2_GOAL_THRES:
			end_time = time_vec[i] + dist / traj_raw[0,5]
			return end_time

	dist = np.linalg.norm(traj_raw[-1,0:2] - goal)
	end_time = time_vec[-1] + dist / traj_raw[0,5]
	return end_time

# partition data into a training and test set
# convert into appropriate format
def process_raw_data(trajs_raw_multi, num_agents_in_network):
	# print trajs_raw_multi[0][0].shape
	# print trajs_raw_multi[0][1].shape
	# raw_input()

	num_pts = 0
	for i in range(len(trajs_raw_multi)):
		num_agents = len(trajs_raw_multi[i]) - 1
		try: 
			assert(num_agents <= num_agents_in_network)
		except AssertionError:
			print('more agents than network size')
			print('num_agents_in network', num_agents_in_network)
			print('num_agents', num_agents)
			assert(0)
		
		num_pts += num_agents * trajs_raw_multi[i][0].shape[0]
		# print 'trajs_raw[i].shape', trajs_raw[i].shape

	# x = 1 x 7 + n x 8 vector
	# 	[dist_to_goal, pref_speed, cur_speed, cur_heading, vx, vy, self_radius, \
	# 	 other_vx, other_vy, rel_pos_x, rel_pos_y, other_radius, self_radius+other_radius, dist_2_other, is_on]
	# y = [angle of a point on traj after 1sec (wrt desired velocity)]
	num_raw_states = 7 + 8 * (num_agents_in_network - 1)
	processed_trajs = np.zeros((num_pts, num_raw_states))
	processed_resp = np.zeros((num_pts, 5))
	ind = 0
	for traj in trajs_raw_multi:
		dt_thres = 0.1 + EPS
		time_vec = traj[0]
		traj_pts = time_vec.shape[0]
		# traj_end_time = traj[traj_pts-1, 0]
		num_agents = len(traj) - 1
		for aa in range(num_agents-1):
			traj_new = swap_OrderInTrajMulti(traj, aa)
			traj_end_time = findEndTime_first(traj_new)
			# plot_traj_raw_multi(traj_new, 'agent %d' %aa, str(aa))
			last_time = time_vec[0]
			for j in range(traj_pts):
				# if time_vec[j] < last_time + 0.1:
				# 	continue
				# if within one second of end, break
				if (traj_end_time - time_vec[j] - DIST_2_GOAL_THRES/traj_new[1][0,5]) < dt_thres:
					break
				np.set_printoptions(precision=2, edgeitems=500)
				assert(traj_new[1][j,5] > 0.1)
				
				agent_state = traj_new[1][j,:]
				others_state = [traj_new[tt][j,:] for tt in range(2, len(traj_new))]
				ref_prll, ref_orth, state_nn = \
					rawState_2_agentCentricState(agent_state, others_state, num_agents_in_network)
				min_dists = [np.linalg.norm(traj_new[1][j,0:2] - traj_new[tt][j,0:2]) for tt in range(2, len(traj_new))]
				min_dist = min(min_dists)

				# for computing px, py, angle_diff, speed_diff
				for k in range(j, traj_pts):
					if time_vec[k] - time_vec[j] > dt_thres:
						dist_traveled = traj_new[1][k,0:2] - traj_new[1][j,0:2]
						px = np.dot(dist_traveled, ref_prll) / (time_vec[k] - time_vec[j])
						py = np.dot(dist_traveled, ref_orth) / (time_vec[k] - time_vec[j])
						angle_diff = np.arctan2(py, px)
						speed_diff = np.linalg.norm(dist_traveled) / (time_vec[k] - time_vec[j])
						# print 'dist_travelled, dt, speed', np.linalg.norm(dist_traveled), \
						# (traj_new[k,0] - traj_new[j,0]), speed_diff
						break
				processed_trajs[ind,:] = state_nn
				value = computeValue(traj_end_time, time_vec[j], state_nn, min_dist)
				processed_resp[ind,:] = [px, py, angle_diff, speed_diff, value]
				last_time = time_vec[j]
				ind += 1
				# print 'state_nn', state_nn
				# print 'y', processed_resp[ind-1,:]
				# raw_input()
	
	X = processed_trajs[0:ind,:]
	Y = processed_resp[0:ind,:]
	# check
	for i in range(len(X)):
		try:
			assert(Y[i,4] < GAMMA ** (X[i,0]/DT_NORMAL) + EPS)
		except:
			print('value', Y[i,4])
			print('bnd', GAMMA ** (X[i,0]/DT_NORMAL))
			assert(0)
	print('finished processing inputs, has %d points' % X.shape[0])
	return X, Y


def reorder_other_agents_state(agent_state, others_state):
	num_agents = len(others_state)
	dist_2_others = np.zeros((num_agents))
	for i, other_state in enumerate(others_state):
		dist_2_others[i] = np.linalg.norm(other_state[0:2]-agent_state[0:2])
	closest_ind = np.argmin(dist_2_others)
	others_state_cp = copy.deepcopy(others_state)
	others_state_cp[0] = others_state[closest_ind]
	others_state_cp[closest_ind] = others_state[0]
	return others_state_cp

def rawState_2_agentCentricState(agent_state, others_state_in, num_agents_in_network):
	# print agent_state.shape
	# print len(others_state)
	others_state = reorder_other_agents_state(agent_state, others_state_in)
	num_agents = len(others_state) + 1
	try:
		assert(num_agents <= num_agents_in_network)
	except AssertionError:
		print('num_agents, num_agents_in_network', num_agents, num_agents_in_network)
		assert(0)
	state_nn = np.zeros((7+8*(num_agents_in_network-1),))
	for i in range(num_agents-1, num_agents_in_network-1):
		# state_nn[7+8*i:7+8*i+7] = [0.0, 0.0, -8.0, 0.0, 0.35, 0.70, 8.0]
		state_nn[7+8*i:7+8*i+7] = [-2.0, -2.0, -10, -10.0, -0.2, -0.2, -2.0]
	# print 'agent_state', agent_state
	
	# agent 
	# distance to goal
	goal_direction = agent_state[6:8]-agent_state[0:2]
	dist_to_goal = np.clip(np.linalg.norm(goal_direction), 0, 30)
	# desired speed
	pref_speed = agent_state[5]
	# new reference frame
	if dist_to_goal > EPS:
		ref_prll = goal_direction / dist_to_goal
	else:
		ref_prll = np.array([np.cos(agent_state[4]), np.sin(agent_state[4])])
	ref_orth = np.array([-ref_prll[1], ref_prll[0]]) # rotate by 90 deg

	# compute heading: method 1 (could be incorrect) 
	# v = 1.0
	# vel_vec = np.array([v*np.cos(agent_state[4]), v*np.sin(agent_state[4])])
	# heading_cos = np.dot(vel_vec, ref_prll)
	# heading_sin = np.dot(vel_vec, ref_orth)
	# heading = np.arctan2(heading_sin, heading_cos)

	# compute heading: method 2 
	ref_prll_angle = np.arctan2(ref_prll[1], ref_prll[0])
	heading = find_angle_diff(agent_state[4], ref_prll_angle)
	heading_cos = np.cos(heading)
	heading_sin = np.sin(heading)

	cur_speed = np.linalg.norm(agent_state[2:4])
	# cur_speed = 1.0
	# cur_speed = agent_state[9]

	vx = cur_speed * heading_cos
	vy = cur_speed * heading_sin
	# vx = 1.0 * heading_cos
	# vy = 1.0 * heading_sin
	# cur_speed = 1.0
	# vx = 0.0
	# vy = 0.0
	self_radius = agent_state[8]

	state_nn[0:7] = [dist_to_goal, pref_speed, cur_speed, heading, vx, vy, self_radius] 
	turning_dir = 0.0 #agent_state[9]
	# state_nn[0:7] = [dist_to_goal, pref_speed, cur_speed, turning_dir, vx, vy, self_radius] 


	# other agents
	for i, other_agent_state in enumerate(others_state):
		# project other elements onto the new reference frame
		rel_pos = other_agent_state[0:2] - agent_state[0:2]
		rel_pos_x = np.clip(np.dot(rel_pos, ref_prll), -8, 8)
		rel_pos_y = np.clip(np.dot(rel_pos, ref_orth), -8, 8)
		other_vx = np.dot(other_agent_state[2:4], ref_prll)
		other_vy = np.dot(other_agent_state[2:4], ref_orth)
		other_radius = other_agent_state[8]
		# tracking_time = other_agent_state[9]
		dist_2_other = np.clip(np.linalg.norm(agent_state[0:2]-other_agent_state[0:2]) \
				- self_radius - other_radius, -3, 10)
		is_on = 1
		# stationary
		if other_vx ** 2 + other_vy ** 2 < EPS:
			is_on = 2
		state_nn[7+8*i:7+8*(i+1)] = [other_vx, other_vy, rel_pos_x, rel_pos_y, other_radius, \
			self_radius+other_radius, dist_2_other, is_on]

	for i in range(num_agents-1, num_agents_in_network-1):
		state_nn[7+8*i:7+8*(i+1)-1] = state_nn[7:7+8-1]
	# for i in range(num_agents-1, num_agents_in_network-1):
	# 	state_nn[7+8*i:7+8*(i+1)-1] = state_nn[7+8*(i-1):7+8*i-1]

	# others_columns_inds = [7 + 6 + 8*(tt) for tt in range(num_agents_in_network-1)] 
	# min_dist_2_others = min(state_nn[others_columns_inds])
	# if_collide = min_dist_2_others < GETTING_CLOSE_RANGE
	# state_nn[3] = if_collide
	# state_nn[4:6] = 0

	return ref_prll, ref_orth, state_nn
	# x = 1 x 7 + n x 8 vector
	# 	[dist_to_goal, pref_speed, cur_speed, cur_heading, vx, vy, self_radius, \
	# 	 other_vx, other_vy, rel_pos_x, rel_pos_y, other_radius, self_radius+other_radius, dist_2_other, is_on]
	# y = [angle of a point on traj after 1sec (wrt desired velocity)]

def rawStates_2_agentCentricStates(agent_states, others_states_in, num_agents_in_network):
	# print agent_states.shape
	if agent_states.shape[0] >= 1:
		others_states = reorder_other_agents_state(agent_states[0,:], others_states_in)
	else: 
		others_states = others_states_in
	num_agents = len(others_states) + 1
	assert(num_agents <= num_agents_in_network)
	num_rawStates = agent_states.shape[0]
	states_nn = np.zeros((num_rawStates, 7+8*(num_agents_in_network-1)))
	for i in range(num_agents-1, num_agents_in_network-1):
		# states_nn[:,7+8*i:7+8*i+7] = np.matlib.repmat(\
			# np.array([0.0, 0.0, -8.0, 0.0, 0.35, 0.70, 8.0]), num_rawStates, 1)
		states_nn[:,7+8*i:7+8*i+7] = np.matlib.repmat(\
			np.array([-2.0, -2.0, -10, -10.0, -0.2, -0.2, -2.0]), num_rawStates, 1)

	# agent
	# distance to goal
	goal_direction = agent_states[:,6:8]-agent_states[:,0:2]
	dist_to_goal = np.clip(np.linalg.norm(goal_direction, axis=1), 0, 30)
	# desired speed
	pref_speed = agent_states[:,5]
	# new reference frame

	# compute ref_prll
	valid_inds = np.where(dist_to_goal>EPS)[0]
	ref_prll = np.vstack([np.cos(agent_states[:,4]), np.sin(agent_states[:,4])]).transpose()
	ref_prll[valid_inds,0] = goal_direction[valid_inds,0] / dist_to_goal[valid_inds]
	ref_prll[valid_inds,1] = goal_direction[valid_inds,1] / dist_to_goal[valid_inds]

	ref_orth = np.vstack([-ref_prll[:,1], ref_prll[:,0]]).transpose() # rotate by 90 deg

	# compute heading: method 1 (could be incorrect) 
	# v = 1.0
	# vel_vec = np.vstack((v*np.cos(agent_states[:,4]), v*np.sin(agent_states[:,4]))).transpose()
	# heading_cos = np.sum(vel_vec*ref_prll, axis=1)
	# heading_sin = np.sum(vel_vec*ref_orth, axis=1)
	# heading = np.arctan2(heading_sin, heading_cos)

	# compute heading: method 2 
	ref_prll_angle = np.arctan2(ref_prll[:,1], ref_prll[:,0])
	heading = find_angle_diff(agent_states[:,4], ref_prll_angle)
	heading_cos = np.cos(heading)
	heading_sin = np.sin(heading)

	cur_speed = np.linalg.norm(agent_states[:,2:4], axis=1)
	# cur_speed = agent_states[:,9]
	# cur_speed = np.ones((num_rawStates,))

	vx = cur_speed * heading_cos
	vy = cur_speed * heading_sin	
	# vx = 1.0 * heading_cos
	# vy = 1.0 * heading_sin
	# cur_speed = np.ones((num_rawStates,))
	# vx = np.zeros((num_rawStates, ))
	# vy = np.zeros((num_rawStates, ))
	self_radius = agent_states[:,8]

	# additional helper fields
	# vx = np.clip(cur_speed, 0.05, 100) * heading_cos
	# vy = np.clip(cur_speed, 0.05, 100) * heading_sin
	states_nn[:,0:7] = np.vstack((dist_to_goal, pref_speed, cur_speed, heading, vx, vy, self_radius)).transpose()
	# turning_dirs = agent_states[:,9]
	# states_nn[:,0:7] = np.vstack((dist_to_goal, pref_speed, cur_speed, turning_dirs, vx, vy, self_radius)).transpose()

	for i, other_agent_states in enumerate(others_states):
		# project other elements onto the new reference frame
		rel_pos = other_agent_states[0:2] - agent_states[:,0:2]
		rel_pos_x = np.clip(np.sum(rel_pos * ref_prll, axis=1), -8, 8)
		rel_pos_y = np.clip(np.sum(rel_pos * ref_orth, axis=1), -8, 8)
		other_vx = np.sum(other_agent_states[2:4] * ref_prll, axis=1)
		other_vy = np.sum(other_agent_states[2:4] * ref_orth, axis=1)
		other_radius = other_agent_states[8] * np.ones((num_rawStates,))
		# tracking_time = other_agent_states[9] * np.ones((num_rawStates,))
		is_on = np.ones((num_rawStates,))
		stat_inds = np.where(other_vx ** 2 + other_vy ** 2 < EPS)[0]
		is_on [stat_inds] = 2
		
		dist_2_other = np.clip(np.linalg.norm(agent_states[:,0:2]-other_agent_states[0:2], axis=1) - \
				self_radius - other_radius, -3, 10)
		# print other_vx.shape, other_vy.shape, rel_pos_x.shape, rel_pos_y.shape, 
		# print other_radius.shape, (self_radius+other_radius), dist_2_other.shape, is_on.shape
		states_nn[:,7+8*i:7+8*(i+1)] = np.vstack((other_vx, other_vy, rel_pos_x, rel_pos_y, other_radius, \
			self_radius+other_radius, dist_2_other, is_on)).transpose()

	for i in range(num_agents-1, num_agents_in_network-1):
		states_nn[:,7+8*i:7+8*(i+1)-1] = states_nn[:,7:7+8-1]

	# for i in range(num_agents-1, num_agents_in_network-1):
	# 	states_nn[:,7+8*i:7+8*(i+1)-1] = states_nn[:,7+8*(i-1):7+8*i-1]
	# states_nn[:,4:6] = 0
	# others_columns_inds = [7 + 6 + 8*(tt) for tt in range(num_agents_in_network-1)] 
	# min_dist_2_others = np.min(states_nn[:,others_columns_inds], axis = 1)
	# if_collide = min_dist_2_others < GETTING_CLOSE_RANGE
	# states_nn[:,3] = if_collide

	return ref_prll, ref_orth, states_nn
	# x = 1 x 7 + n x 8 vector
	# 	[dist_to_goal, pref_speed, cur_speed, cur_heading, vx, vy, self_radius, \
	# 	 other_vx, other_vy, rel_pos_x, rel_pos_y, other_radius, self_radius+other_radius, dist_2_other, is_on]
	# y = [angle of a point on traj after 1sec (wrt desired velocity)]

def agentCentricState_2_rawState_noRotate(agentCentricState):
	agent_state = np.zeros((10,))
	# see README.txt 
	agent_state[0] = 0.0; agent_state[1] = 0.0; 
	heading = agentCentricState[3]
	cur_speed = agentCentricState[2]
	agent_state[4] = heading; agent_state[5] = agentCentricState[1]
	agent_state[2] = cur_speed * np.cos(heading); agent_state[3] = cur_speed*np.sin(heading)
	agent_state[6] = agentCentricState[0]; agent_state[7] = 0.0
	agent_state[8] = agentCentricState[6]
	agent_state[9] = 0.0
	# print agent_state
	# raw_input()

	other_agent_states = [] #np.zeros((9,))
	num_other_agents = ((agentCentricState.shape[0] - 7) / 8)
	# print agentCentricState.shape
	# print num_other_agents
	# print 'here'
	# raw_input()
	for i in range(num_other_agents):
		other_agent_state = np.zeros((10,))
		partial_state = agentCentricState[7+8*i:7+8*(i+1)]
		# skip if not active
		if partial_state[7] == 0:
			continue
		other_agent_state[0] = partial_state[2]; other_agent_state[1] = partial_state[3]
		other_agent_state[2] = partial_state[0]; other_agent_state[3] = partial_state[1]
		heading = np.arctan2(other_agent_state[3], other_agent_state[2])
		speed = np.linalg.norm(other_agent_state[2:4]) 
		other_agent_state[4] = heading; other_agent_state[5] = max(speed, EPS)
		other_agent_state[6] = partial_state[2] + 1.0 * other_agent_state[2]; 
		other_agent_state[7] = partial_state[3] + 1.0 * other_agent_state[3];
		other_agent_state[8] = partial_state[4]
		other_agent_state[9] = partial_state[6]
		other_agent_states.append(other_agent_state)
	# prefered state and other state unknown
	return agent_state, other_agent_states

def plot_state_processed(x, y):
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(1, 1, 1)
	a_s, other_agent_states = agentCentricState_2_rawState_noRotate(x)

	# agent at (0,0)
	circ1 = plt.Circle((0.0, 0.0), radius=a_s[8], fc='w', ec=plt_colors[0])
	ax.add_patch(circ1)
	# goal
	plt.plot(a_s[0], 0.0, c=plt_colors[0], marker='*', markersize=20)
	# pref speed
	plt.arrow(0.0, 0.0, a_s[5], 0.0, fc='m', ec='m', head_width=0.05, head_length=0.1)
	vel_pref, = plt.plot([0.0, a_s[5]], [0.0, 0.0], 'm', linewidth=2)
	# current speed
	plt.arrow(0.0, 0.0, a_s[2], a_s[3], fc='k', ec='k', head_width=0.05, head_length=0.1)
	vel_cur, = plt.plot([0.0, a_s[2]], [0.0,  a_s[3]], 'k', linewidth=2)

	# actual speed (1 second after location)
	plt.arrow(0.0, 0.0, y[0], y[1],  fc=plt_colors[0], \
		ec=plt_colors[0], head_width=0.05, head_length=0.1)
	vel_select, = plt.plot([0.0, y[0]], [0.0, y[1]], \
		c=plt_colors[0], linewidth=2)
	
	# other agents
	for i, o_s in enumerate(other_agent_states):
		# plt.plot(x[6], x[7], 'b*', markersize=20)
		circ = plt.Circle((o_s[0], o_s[1]), radius=o_s[8], fc='w', ec=plt_colors[i+1])
		ax.add_patch(circ)
		# other agent's speed
		plt.arrow(o_s[0], o_s[1], o_s[2], o_s[3], fc=plt_colors[i+1], \
			ec=plt_colors[i+1], head_width=0.05, head_length=0.1)
		vel_other, = plt.plot([o_s[0], o_s[0]+x[2]], [o_s[1], o_s[1]+o_s[3]], \
			c=plt_colors[i+1], linewidth=2)

	# plt.title('test case')
	plt.xlabel('x (m)')
	plt.ylabel('y (m)')
	plt.legend([vel_pref,vel_cur,vel_other,vel_select], \
		['vel_pref','vel_cur','vel_other','vel_select'])
	plt.axis('equal')

# Y_raw is the heading angle
def find_bins(Y_raw, num_bins=11, center_value=0.001, lb=-1.0, ub=1.0):
	assert(num_bins % 2 == 1)
	# find angles in the center bin
	bins = np.zeros((num_bins+1,))
	center_inds = np.where((Y_raw<center_value) & (Y_raw>-center_value))[0]
	# print 'number of center elements', center_inds.shape[0]
	# print Y_raw.shape
	# equally divide the other datasets
	upper_inds = np.where((Y_raw > center_value))[0]
	lower_inds = np.where((Y_raw < -center_value))[0]
	
	''' symmetrical'''
	num_upper_bins = (num_bins - 1) / 2
	num_lower_bins = num_upper_bins

	Y_sort_high = np.sort(Y_raw[upper_inds])
	high_stride = np.floor(upper_inds.shape[0] / num_upper_bins)

	bins[-1] = max(ub, Y_sort_high[(num_upper_bins-1)*high_stride])
	bins[0] = min(lb, -Y_sort_high[(num_upper_bins-1)*high_stride])
	bins[num_lower_bins] = - center_value
	bins[num_lower_bins+1] = center_value
	for i in range(1, num_upper_bins):
		bins[num_lower_bins + 1 + i] = Y_sort_high[i*high_stride]
	# print 'num_lower_bins', num_lower_bins
	for i in range(1, num_lower_bins):
		bins[i] = -bins[num_bins-i]
	# print bins

	# not the most accurate because there are entries out of bound
	fig = plt.figure(figsize=(10, 8))
	plt.hist(Y_raw, bins)
	plt.title('raw')

	# compute other statistics
	bin_centers = np.zeros(num_bins)
	for i in range(num_bins):
		bins_centers = (bins[i] + bins[i+1]) / 2.0
	bin_centers[0] = lb
	bin_centers[-1] = ub 
	

	# might be inefficient / but ok b/c executed once
	bin_assignment = (num_bins-1) * np.ones((Y_raw.shape[0],))
	for i in range(Y_raw.shape[0]):
		for j in range(1,num_bins):
			if Y_raw[i] < bins[j]:
				bin_assignment[i] = j-1
				break
	# check
	# fig = plt.figure(figsize=(10, 8))
	# plt.hist(bin_assignment, range(num_bins+1))

	return bins, bin_assignment

# make sure the dominating class won't have too many sanples
def filterDominateClass(bins, bin_assignments):
	num_samples = bin_assignments.shape[0]
	num_classes = bins.shape[0] - 1
	counts = np.zeros((num_classes,))
	for i in range(num_samples):
		counts[bin_assignments[i]] += 1
	inds_all = np.empty((0,))
	num_data_pts = 0
	for i in range(num_classes):
		if counts[i] > 0.2 * num_samples:
			inds = np.where(bin_assignments == i)[0]
			rand_perm = np.random.permutation(np.arange(counts[i]))
			inds = inds[:int(0.2*num_samples)]
		else: 
			inds = np.where(bin_assignments == i)[0]
		num_data_pts += inds.shape[0]
		inds_all = np.hstack((inds_all, inds))
		# print 'num_data_pts', i, num_data_pts
	inds_all = inds_all.astype(int)
	# print inds_all
	fig = plt.figure(figsize=(10, 8))
	plt.hist(bin_assignments[inds_all], num_classes)
	plt.title('processed')
	# fig = plt.figure(figsize=(10, 8))
	# plt.hist(bin_assignments, num_classes)
	return inds_all

	
def plot_traj_raw_multi(traj_raw_multi, title_string, figure_name=None):
	if figure_name == None:
		fig = plt.figure(figsize=(10, 8))
	else:
		fig = plt.figure(figure_name,figsize=(10, 8))
		plt.clf()

	ax = fig.add_subplot(1, 1, 1)
	time_vec = traj_raw_multi[0]
	num_pts = time_vec.shape[0]
	# traj_raw_multi:
	# 	array of numpy arrays
	# 	traj_raw_multi[0] = time_vec (num_pts x 1)
	# 	traj_raw_multi[1] = agent1_raw_state (num_pts * 9)
	# 	...

	# compute stats
	time_to_reach_goal, traj_lengths, min_sepDist, if_completed_vec \
		= computeStats(traj_raw_multi)

	# plot traj and goal
	for i in range(1, len(traj_raw_multi)):
		# stationary (for plotting static case)
		# if traj_lengths[i-1] < EPS:
		# 	continue
		color_ind = (i - 1) % len(plt_colors)
		plt_color = plt_colors[color_ind]
		plt.plot(traj_raw_multi[i][:,0], traj_raw_multi[i][:,1],\
			color=plt_color, ls='-', linewidth=2)
		plt.plot(traj_raw_multi[i][0,6], traj_raw_multi[i][0,7],\
			color=plt_color, marker='*', markersize=20)


	# print title_string
	# print time_to_reach_goal
	# print traj_lengths

	# plot heading direction
	# for i in range(time_vec.shape[0]):
	# 	for j in range(1, len(traj_raw_multi)):
	# 		plt.plot([traj_raw_multi[j][i,0], traj_raw_multi[j][i,0]+0.3 * np.cos(traj_raw_multi[j][i,4])], \
	# 			[traj_raw_multi[j][i,1], traj_raw_multi[j][i,1]+0.3 * np.sin(traj_raw_multi[j][i,4])],\
	# 			c=plt_colors[j-1], ls='-',linewidth=1)

	# plot vehile position
	counter = 0
	cur_time = -100.0
	has_plotted_collision = False
	agent_plot_pos = np.random.rand(len(traj_raw_multi)-1,2)
	for i in range(time_vec.shape[0]):
		if (time_vec[i] - cur_time) >= 1.5 - EPS:
			cur_time = time_vec[i]
			for j in range(1, len(traj_raw_multi)):
				# if np.linalg.norm(agent_plot_pos[j-1, :] \
				# 	- traj_raw_multi[j][i, 0:2]) > EPS:
				if cur_time < time_to_reach_goal[j-1]-0.5:
					color_ind = (j - 1) % len(plt_colors)
					plt_color = plt_colors[color_ind]
					ax.add_patch( plt.Circle(traj_raw_multi[j][i, 0:2], \
						radius=traj_raw_multi[j][0, 8], fc='w', ec=plt_color) )
					if j % 2 == 0:
						y_text_offset = 0.05
					else:
						y_text_offset = 0.05
					ax.text(traj_raw_multi[j][i,0]-0.15, traj_raw_multi[j][i,1]+y_text_offset, \
						'%.1f'%cur_time, color=plt_color)
					agent_plot_pos[j-1, :] = traj_raw_multi[j][i, 0:2].copy()
			counter += 1
	# end pos
	for j in range(1, len(traj_raw_multi)):
		color_ind = (j - 1) % len(plt_colors)
		plt_color = plt_colors[color_ind]
		if j % 2 == 0:
			y_text_offset = 0.05
		else:
			y_text_offset = 0.05
		ax.add_patch( plt.Circle(traj_raw_multi[j][-1, 0:2], \
			radius=traj_raw_multi[j][0, 8], fc='w', ec=plt_color) )
		if traj_lengths[j-1] < EPS:
			continue
		ax.text(traj_raw_multi[j][-1,0], traj_raw_multi[j][-1,1]+y_text_offset, \
			'%.1f'%time_to_reach_goal[j-1], color=plt_color)


	string_tmp = '\n total length %.3f, total time %.3f, min dist %.3f' % \
		(np.sum(traj_lengths), np.sum(time_to_reach_goal), min_sepDist)
	title_string = title_string + string_tmp

	plt.title(title_string)
	plt.xlabel('x (m)')
	plt.ylabel('y (m)')
	plt.axis('equal')

	# plotting style (only show axis on bottom and left)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

	plt.draw()
	plt.pause(0.0001)


def generateNegValueSamples(dataset_value):
	num_neg_samples = dataset_value.shape[0]
	neg_dataset = dataset_value.copy()
	values = np.zeros((num_neg_samples,1))
	num_pts = int(num_neg_samples / 5.0)
	for i in range(num_pts):
		# just need to perturb first agent, b/c network symmetrical
		agent_state = neg_dataset[i,0:7]
		other_agent_state = neg_dataset[i,7:15]
		radius = agent_state[6] + other_agent_state[4]
		while True:
			rel_pos = np.random.rand(2) * 4.0 - 2.0
			neg_dataset[i,9] = rel_pos[0]
			neg_dataset[i,10] = rel_pos[1]
			# print 'rel_pos, np.linalg.norm(rel_pos)', rel_pos, np.linalg.norm(rel_pos)
			if np.linalg.norm(rel_pos) < radius:
				values[i,0] = -0.25 
				break
			else:
				other_vel = other_agent_state[0:2]
				agent_vel = agent_state[4:6]
				rel_vel = other_vel - agent_vel
				if np.linalg.norm(rel_pos + rel_vel * 0.5) < radius:
					values[i,0] = -0.25 * 3.0 / 4.0
					break
				elif np.linalg.norm(rel_pos + rel_vel * 0.5) < radius:
					values[i,0] = -0.25 * 2.0 / 4.0
					break
	return neg_dataset[:num_pts,:], values[:num_pts,:]

def reflectTraj(traj_raw_multi):
	traj_raw_multi_refl = copy.deepcopy(traj_raw_multi)
	for i in range(1, len(traj_raw_multi)):
		traj_raw_multi_refl[i][:,1] = -traj_raw_multi[i][:,1]
		traj_raw_multi_refl[i][:,3] = -traj_raw_multi[i][:,3]
		traj_raw_multi_refl[i][:,4] = -traj_raw_multi[i][:,4]
		traj_raw_multi_refl[i][:,7] = -traj_raw_multi[i][:,7]
	return traj_raw_multi_refl

if __name__ == '__main__':
	print('hello world from pedData_processing_multi.py')
	file_dir = os.path.dirname(os.path.realpath(__file__))
	plt.rcParams.update({'font.size': 18})
	# load raw trajs
	num_agents_in_network = 4
	trajs_raw_multi = pickle.load(open(file_dir+\
		"/../../pickle_files/multi/%d_agents_cadrl_raw.p"%num_agents_in_network,"rb"))
	# plot_traj_raw_multi(trajs_raw_multi[1], 'input raw trajectory (from pickle file)')
	# plot_traj_raw_multi(trajs_raw_multi[2], 'input raw trajectory (from pickle file)')
	# plot_traj_raw_multi(trajs_raw_multi[49], 'input raw trajectory (from pickle file)')

	# for i in range(25, 50):
		# plot_traj_raw_multi(trajs_raw_multi[i], 'input raw trajectory (from pickle file)')

	# load processed trajs
	try:
		assert(0)
		dataset_ped = pickle.load(open(file_dir+\
			"/../../pickle_files/multi/%d_agents_dataset.p"%num_agents_in_network, "rb"))
		print('load pedestrian dataset, has %d points' % dataset_ped[0].shape[0])
	except: # pickle.PickleError:
		X, Y_raw = process_raw_data(trajs_raw_multi, num_agents_in_network)
		bins, bin_assignments = find_bins(Y_raw[:,2])
		inds = filterDominateClass(bins, bin_assignments)
		dataset_ped = [X[inds,:], bin_assignments[inds], bins, Y_raw[inds,:]]
		pickle.dump(dataset_ped, open(file_dir+\
			"/../../pickle_files/multi/%d_agents_dataset.p"%num_agents_in_network, "wb"))
	
	X = dataset_ped[0]
	Y_raw = dataset_ped[3]
	
	# check plot one state
	# plot_state_processed(X[1,:], Y_raw[1,:])
	# plot_state_processed(X[1335,:], Y_raw[1335,:])
	# plot_state_processed(X[10054,:], Y_raw[10054,:])
	# plt.show()

	# dataset for classification / regression
	try:
		assert(0)
		# classification 
		dataset_ped_train = pickle.load(open(file_dir+\
			"/../../pickle_files/multi/%d_agents_dataset_train.p"%num_agents_in_network, "rb"))
		dataset_ped_test = pickle.load(open(file_dir+\
			"/../../pickle_files/multi/%d_agents_dataset_test.p"%num_agents_in_network, "rb"))
		print('loaded classification dataset')
		print('classification dataset contains %d pts, training set has %d pts, test set has %d pts' % \
				(dataset_ped[0].shape[0], dataset_ped_train[0].shape[0], dataset_ped_test[0].shape[0]))
		# regression
		dataset_ped_regr_train = pickle.load(open(file_dir+\
			"/../../pickle_files/multi/%d_agents_dataset_regr_train.p"%num_agents_in_network, "rb"))
		dataset_ped_regr_test = pickle.load(open(file_dir+\
			"/../../pickle_files/multi/%d_agents_dataset_regr_test.p"%num_agents_in_network, "rb"))
		print('loaded regression dataset')
		print('regression dataset contains %d pts, training set has %d pts, test set has %d pts' % \
				(dataset_ped[0].shape[0], dataset_ped_regr_train[0].shape[0], dataset_ped_regr_test[0].shape[0]))
	except:
		# partition into training and test sets
		rand_perm = np.random.permutation(np.arange(dataset_ped[0].shape[0]))
		dataset_ped[0] = dataset_ped[0][rand_perm, :]
		dataset_ped[1] = dataset_ped[1][rand_perm]
		dataset_ped[2] = dataset_ped[2]
		dataset_ped[3] = dataset_ped[3][rand_perm,:] 
		# training set 80%, test set 20%
		part_ind = int(0.8 * dataset_ped[0].shape[0])
		test_set_size = min(5000+part_ind, dataset_ped[0].shape[0])
		dataset_ped[0] = dataset_ped[0][:test_set_size, :]
		dataset_ped[1] = dataset_ped[1][:test_set_size]
		dataset_ped[2] = dataset_ped[2]
		dataset_ped[3] = dataset_ped[3][:test_set_size,:] 
		# classfication 
		dataset_ped_train = [dataset_ped[0][:part_ind,:], \
							dataset_ped[1][:part_ind], \
							dataset_ped[2], \
							dataset_ped[3][:part_ind,:] ] 

		dataset_ped_test =  [dataset_ped[0][part_ind:,:], \
							dataset_ped[1][part_ind:], \
							dataset_ped[2], \
							dataset_ped[3][part_ind:,:] ] 
		# print 'num of dataset_ped_train classes', np.amax(dataset_ped_train[1]) + 1
		# print 'num of dataset_ped_test classes', np.amax(dataset_ped_test[1]) + 1
		# print 'num of dataset_ped classes', np.amax(dataset_ped[1]) + 1
		# num_classes = int(np.amax(dataset_ped[1]) + 1)
		# fig = plt.figure(figsize=(10, 8))
		# plt.hist(dataset_ped_train[1], num_classes)
		# plt.title('training set')
		
		# fig = plt.figure(figsize=(10, 8))
		# plt.hist(dataset_ped_test[1], num_classes)
		# plt.title('test set')
		
		pickle.dump(dataset_ped_train, open(file_dir+\
			"/../../pickle_files/multi/%d_agents_dataset_train.p"%num_agents_in_network, "wb"))
		pickle.dump(dataset_ped_test, open(file_dir+\
			"/../../pickle_files/multi/%d_agents_dataset_test.p"%num_agents_in_network, "wb"))
		print('classification dataset contains %d pts, training set has %d pts, test set has %d pts' % \
				(dataset_ped[0].shape[0], dataset_ped_train[0].shape[0], dataset_ped_test[0].shape[0]))

		# regression
		dataset_ped_regr_train = [dataset_ped[0][:part_ind,:], \
							dataset_ped[3][:part_ind,0:2 ] ]
		dataset_ped_regr_test =  [dataset_ped[0][part_ind:,:], \
							dataset_ped[3][part_ind:,0:2 ] ]

		# value
		dataset_ped_value_train = [dataset_ped[0][:part_ind,:], \
							dataset_ped[3][:part_ind,4 ].reshape((-1,1)) ]
		dataset_ped_value_test =  [dataset_ped[0][part_ind:,:], \
							dataset_ped[3][part_ind:,4 ].reshape((-1,1)) ]
		# neg_dataset_train, values_train = generateNegValueSamples(dataset_ped_value_train[0])
		# neg_dataset_test, values_test = generateNegValueSamples(dataset_ped_value_test[0])
		# dataset_ped_value_train[0] = np.vstack((dataset_ped_value_train[0], neg_dataset_train))
		# dataset_ped_value_train[1] = np.vstack((dataset_ped_value_train[1], values_train))
		# dataset_ped_value_test[0] = np.vstack((dataset_ped_value_test[0], neg_dataset_test))
		# dataset_ped_value_test[1] = np.vstack((dataset_ped_value_test[1], values_test))


		pickle.dump(dataset_ped_regr_train, open(file_dir+\
			"/../../pickle_files/multi/%d_agents_dataset_regr_train.p"%num_agents_in_network, "wb"))
		pickle.dump(dataset_ped_regr_test, open(file_dir+\
			"/../../pickle_files/multi/%d_agents_dataset_regr_test.p"%num_agents_in_network, "wb"))
		pickle.dump(dataset_ped_value_train, open(file_dir+\
			"/../../pickle_files/multi/%d_agents_dataset_value_train.p"%num_agents_in_network, "wb"))
		pickle.dump(dataset_ped_value_test, open(file_dir+\
			"/../../pickle_files/multi/%d_agents_dataset_value_test.p"%num_agents_in_network, "wb"))
		print('dataset contains %d pts, training set has %d pts, test set has %d pts' % \
				(dataset_ped[0].shape[0], dataset_ped_regr_train[0].shape[0], dataset_ped_regr_test[0].shape[0]))

	plt.show()