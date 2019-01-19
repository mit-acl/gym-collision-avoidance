#!/usr/bin/env python
import sys
sys.path.append('../neural_networks')

import numpy as np
import numpy.matlib
import pickle
import copy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import os
import time
import copy

from gym_collision_avoidance.envs.policies.CADRL.scripts.neural_networks import neural_network_regr_multi as nn
from gym_collision_avoidance.envs.policies.CADRL.scripts.multi import pedData_processing_multi as pedData
from gym_collision_avoidance.envs.policies.CADRL.scripts.neural_networks.nn_training_param import NN_training_param
from gym_collision_avoidance.envs.policies.CADRL.scripts.neural_networks.multiagent_network_param import Multiagent_network_param
from gym_collision_avoidance.envs.policies.CADRL.scripts.multi import global_var as gb


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
SMOOTH_COST = gb.SMOOTH_COST

# for 'rotate_constr'
TURNING_LIMIT = np.pi/6.0

# neural network 
NN_ranges = gb.NN_ranges

# calculate the minimum distance between two line segments
# not counting the starting point
def find_dist_between_segs(x1, x2, y1, y2):
# x1.shape = (2,)
# x2.shape = (num_actions,2)
# y1.shape = (2,)
# y2.shape = (num_actions,2)
	if_one_pt = False
	if x2.shape == (2,):
		x2 = x2.reshape((1,2))
		y2 = y2.reshape((1,2))
		if_one_pt = True


	start_dist = np.linalg.norm(x1 - y1)
	end_dist = np.linalg.norm(x2 - y2, axis=1)
	critical_dist = end_dist.copy() 
	# start_dist * np.ones((num_pts,))   # initialize
	# critical points (where d/dt = 0)
	z_bar = (x2 - x1) - (y2 - y1)             # shape = (num_actions, 2)
	inds = np.where((np.linalg.norm(z_bar,axis=1)>0))[0]
	t_bar = - np.sum((x1-y1) * z_bar[inds,:], axis=1) \
			/ np.sum(z_bar[inds,:] * z_bar[inds,:], axis=1)
	t_bar_rep = np.matlib.repmat(t_bar, 2, 1).transpose()
	dist_bar = np.linalg.norm(x1 + (x2[inds,:]-x1) * t_bar_rep \
			  - y1 - (y2[inds,:]-y1) * t_bar_rep, axis=1)
	inds_2 = np.where((t_bar > 0) & (t_bar < 1.0))
	critical_dist[inds[inds_2]] = dist_bar[inds_2] 

	# end_dist = end_dist.clip(min=0, max=start_dist)
	min_dist = np.amin(np.vstack((end_dist, critical_dist)), axis=0)
	# print 'min_dist', min_dist

	if if_one_pt:
		return min_dist[0]
	else:
		return min_dist

''' calculate distance between point p3 and 
   line segment p1->p2'''
def distPointToSegment(p1, p2, p3):
    #print p1
    #print p2
    #print p3
    d = p2 - p1
    #print 'd', d
    #print '(p3-p1)', (p3-p1)
    #print 'linalg.norm(d) ** 2', linalg.norm(d) ** 2.0
    if np.linalg.norm(d) < EPS:
        u = 0.0
    else:
        u = np.dot(d, (p3-p1)) / (np.linalg.norm(d) ** 2.0)
    u = max(0.0, min(u, 1.0))

    inter = p1 + u * d
    dist = np.linalg.norm(p3 - inter)
    return dist


def generate_rand_test_case_multi(num_agents, side_length, speed_bnds, radius_bnds, \
	is_end_near_bnd = False, is_static = False):
	# num_agents_sampled = np.random.randint(2, high=num_agents+1)
	num_agents_sampled = num_agents
	# num_agents_sampled = 2

	random_case = np.random.rand()


	if is_static == True:
		test_case = generate_static_case(num_agents_sampled, side_length, \
			speed_bnds, radius_bnds)
	# else: 
	# 	if random_case < 0.5:
	# 		test_case = generate_swap_case(num_agents_sampled, side_length, \
	# 			speed_bnds, radius_bnds)
	# 	elif random_case > 0.5:
	# 		test_case = generate_circle_case(num_agents_sampled, side_length, \
	# 			speed_bnds, radius_bnds)
	else: 
		if random_case < 0.15:
			test_case = generate_swap_case(num_agents_sampled, side_length, \
				speed_bnds, radius_bnds)
		elif random_case > 0.15 and random_case < 0.3:
			test_case = generate_circle_case(num_agents_sampled, side_length, \
				speed_bnds, radius_bnds)
		else:
			# is_static == False:
			test_case = generate_rand_case(num_agents_sampled, side_length, speed_bnds, radius_bnds, \
				is_end_near_bnd = is_end_near_bnd)
	
	return test_case

def generate_rand_case(num_agents, side_length, speed_bnds, radius_bnds, \
	is_end_near_bnd=False):
	test_case = np.zeros((num_agents, 6))

	# if_oppo = np.random.rand() > 0.8

	for i in range(num_agents):
		# radius
		test_case[i,5] = (radius_bnds[1] - radius_bnds[0]) \
			* np.random.rand() + radius_bnds[0]
		counter = 0
		s1 = (speed_bnds[1] - speed_bnds[0]) * np.random.rand() + speed_bnds[0]
		s2 = (speed_bnds[1] - speed_bnds[0]) * np.random.rand() + speed_bnds[0]
		test_case[i,4] = max(s1, s2)


		while True:
		# generate random starting/ending points
			counter += 1
			side_length *= 1.01
			start = side_length * 2 * np.random.rand(2,) - side_length
			end = side_length * 2 * np.random.rand(2,) - side_length
			# make end point near the goal
			if is_end_near_bnd == True:
				# left, right, top, down
				random_side = np.random.randint(4)
				if random_side == 0:
					end[0] = np.random.rand() * 0.1 * \
						side_length - side_length
				elif random_side == 1:
					end[0] = np.random.rand() * 0.1 * \
						side_length + 0.9 * side_length
				elif random_side == 2:
					end[1] = np.random.rand() * 0.1 * \
						side_length - side_length
				elif random_side == 3:
					end[1] = np.random.rand() * 0.1 * \
						side_length + 0.9 * side_length
				else:
					assert(0)

			# agent 1 & 2 in opposite directions
			# if i == 0 and if_oppo == True:
			# 	start[0] = 0; start[1] = 0
			# 	end[0] = (5-1) * np.random.rand() + 1; end[1] = 0
			# elif i == 1 and if_oppo == True:
			# 	start[0] = (1-0.5) * np.random.rand() + 1.0; start[1] = np.random.rand() * 0.5 - 0.25
			# 	end[0] = (-1-(-5)) * np.random.rand() -5; end[1] = np.random.rand() * 0.5 - 0.25

			# if colliding with previous test cases
			if_collide = False
			for j in range(i):
				radius_start = test_case[j,5] + test_case[i,5] + GETTING_CLOSE_RANGE
				radius_end = test_case[j,5] + test_case[i,5] + GETTING_CLOSE_RANGE
				# start
				if np.linalg.norm(start - test_case[j,0:2] ) < radius_start:
					if_collide = True
					break
				# end
				if np.linalg.norm(end - test_case[j,2:4]) < radius_end:
					if_collide = True
					break
			if if_collide == True:
				continue

			# if straight line is permited
			if i >=1:
				if_straightLineSoln = True
				for j in range(0,i):
					x1 = test_case[j,0:2]; x2 = test_case[j,2:4]; 
					y1 = start; y2 = end; 
					s1 = test_case[j,4]; s2 = test_case[i,4]; 
					radius = test_case[j,5] + test_case[i,5] + GETTING_CLOSE_RANGE
					if if_permitStraightLineSoln(x1, x2, s1, y1, y2, s2, radius) == False:
						# print 'num_agents %d; i %d; j %d'%  (num_agents, i, j)
						if_straightLineSoln = False
						break
				if if_straightLineSoln == True:
					continue


			if np.linalg.norm(start-end) > side_length * 0.5:
				break

		# record test case
		test_case[i,0:2] = start
		test_case[i,2:4] = end
		# test_case[i,4] = (speed_bnds[1] - speed_bnds[0]) \
		# 	* np.random.rand() + speed_bnds[0]
	return test_case

def generate_easy_rand_case(num_agents, side_length, speed_bnds, radius_bnds, agent_separation, \
	is_end_near_bnd=False):
	test_case = np.zeros((num_agents, 6))

	# align agents so they just have to go approximately horizontal to their goal (above one another)
	agent_pos = agent_separation*np.arange(num_agents)
	np.random.shuffle(agent_pos)
	for i in range(num_agents):
		radius = np.random.uniform(radius_bnds[0], radius_bnds[1])
		speed = np.random.uniform(speed_bnds[0], speed_bnds[1])
		test_case[i,4] = speed
		test_case[i,5] = radius

		y = agent_pos[i]
		min_dist_to_others = -np.inf
		while min_dist_to_others < 1.0:
			start_x = np.random.uniform(-side_length/2.0,side_length/2.0)
			start_y = y + np.random.uniform(-0.5,0.5)
			if i == 0: min_dist_to_others = np.inf
			else:
				min_dist_to_others = min([np.linalg.norm([start_x - other_x, start_y - other_y]) for other_x, other_y in test_case[:i,0:2]])
		end_x = start_x + np.random.choice([-1,1])*side_length
		end_y = y + np.random.uniform(-0.5,0.5)
		test_case[i,0:2] = start_x, start_y
		test_case[i,2:4] = end_x, end_y

	return test_case

def generate_static_case(num_agents, side_length, speed_bnds, radius_bnds):
	test_case = np.zeros((num_agents, 6))

	# other agents
	for i in range(num_agents):
		# radius
		test_case[i,5] = (radius_bnds[1] - radius_bnds[0]) \
			* np.random.rand() + radius_bnds[0]
		counter = 0
		s1 = (speed_bnds[1] - speed_bnds[0]) * np.random.rand() + speed_bnds[0]
		s2 = (speed_bnds[1] - speed_bnds[0]) * np.random.rand() + speed_bnds[0]
		test_case[i,4] = max(s1, s2)

		# 0th agent
		if i == 0:
			start = side_length * 2.0 * np.random.rand(2,) - side_length
			end = side_length * 2.0 * np.random.rand(2,) - side_length
			start[0] = min(-1.5, -np.random.rand() * side_length)
			start[1] = np.random.rand() * 2.0 - 1.0
			end[0] = max(1.5, np.random.rand() * side_length)
			end[1] = np.random.rand() * 2.0 - 1.0
		
		elif i == 1:
			start = np.zeros((2,))
			end = start

		else:
			while True:
				# generate random starting/ending points
				start = (side_length * 2 * np.random.rand(2,) - side_length) / 2.0
				end = start
				
				# if colliding with previous test cases
				if_collide = False
				for j in range(i):
					radius_start = test_case[j,5] + test_case[i,5] + GETTING_CLOSE_RANGE
					radius_end = test_case[j,5] + test_case[i,5] + GETTING_CLOSE_RANGE
					# start
					if np.linalg.norm(start - test_case[j,0:2] ) < radius_start:
						if_collide = True
						break
					# end
					if np.linalg.norm(end - test_case[j,2:4]) < radius_end:
						if_collide = True
						break
				if if_collide == True:
					side_length *= 1.01
					continue
				else:
					break

		# record test case
		test_case[i,0:2] = start
		test_case[i,2:4] = end
		# test_case[i,4] = (speed_bnds[1] - speed_bnds[0]) \
		# 	* np.random.rand() + speed_bnds[0]
	return test_case

# two agents swapping position
def generate_swap_case(num_agents, side_length, speed_bnds, radius_bnds):
	r_min = num_agents / 2.0
	r = np.random.rand() * 2.0 + r_min
	test_case = np.zeros((num_agents, 6))
	counter = 0
	r_swap = 1.5 + np.random.rand() * 2.0
	offset = np.array([0,  1.0 + r_min + np.random.rand()*2.0])
	if np.random.rand() > 0.5:
		offset = -offset
	for i in range(num_agents):
		# radius
		test_case[i,5] = (radius_bnds[1] - radius_bnds[0]) \
			* np.random.rand() + radius_bnds[0]
		counter = 0
		s1 = (speed_bnds[1] - speed_bnds[0]) * np.random.rand() + speed_bnds[0]
		s2 = (speed_bnds[1] - speed_bnds[0]) * np.random.rand() + speed_bnds[0]
		test_case[i,4] = max(s1, s2)

		# first and second agent swap position, others 
		if i == 0:
			start = np.array([-r_swap, 0.0])
			end = np.array([r_swap, 0.0])
		elif i == 1:
			start = np.array([r_swap, 0.0])
			end = np.array([-r_swap, 0.0])
		else:
			while True:
				if counter > 10:
					r *= 1.01
					counter = 0
				start_angle = np.random.rand() * 2 * np.pi - np.pi  
				end_angle = np.pi + start_angle
				start = np.array([r*np.cos(start_angle), r*np.sin(start_angle)]) + offset
				end = np.array([r*np.cos(end_angle), r*np.sin(end_angle)]) + offset
				# if colliding with previous test cases
				if_collide = False
				for j in range(i):
					radius_start = test_case[j,5] + test_case[i,5] + GETTING_CLOSE_RANGE
					radius_end = test_case[j,5] + test_case[i,5] + GETTING_CLOSE_RANGE
					# start
					if np.linalg.norm(start - test_case[j,0:2] ) < radius_start:
						if_collide = True
						break
					# end
					if np.linalg.norm(end - test_case[j,2:4]) < radius_end:
						if_collide = True
						break
				if if_collide == True:
					counter += 1
					continue
				else:
					break

		test_case[i,0:2] = start
		test_case[i,2:4] = end
	return test_case

# multiple agents aranged on a circle
def generate_circle_case(num_agents, side_length, speed_bnds, radius_bnds):
	r_min = num_agents / 2.0
	r = np.random.rand() * 2.0 + r_min
	test_case = np.zeros((num_agents, 6))
	counter = 0
	for i in range(num_agents):
		# radius
		test_case[i,5] = (radius_bnds[1] - radius_bnds[0]) \
			* np.random.rand() + radius_bnds[0]
		counter = 0
		s1 = (speed_bnds[1] - speed_bnds[0]) * np.random.rand() + speed_bnds[0]
		s2 = (speed_bnds[1] - speed_bnds[0]) * np.random.rand() + speed_bnds[0]
		test_case[i,4] = max(s1, s2)

		while True:
			if counter > 10:
				r *= 1.01
				counter = 0
			start_angle = np.random.rand() * 2 * np.pi - np.pi  
			end_angle = np.pi + start_angle
			start = np.array([r*np.cos(start_angle), r*np.sin(start_angle)])
			end = np.array([r*np.cos(end_angle), r*np.sin(end_angle)])
			# if colliding with previous test cases
			if_collide = False
			for j in range(i):
				radius_start = test_case[j,5] + test_case[i,5] + GETTING_CLOSE_RANGE
				radius_end = test_case[j,5] + test_case[i,5] + GETTING_CLOSE_RANGE
				# start
				if np.linalg.norm(start - test_case[j,0:2] ) < radius_start:
					if_collide = True
					break
				# end
				if np.linalg.norm(end - test_case[j,2:4]) < radius_end:
					if_collide = True
					break
			if if_collide == True:
				counter += 1
				continue
			else:
				break

		test_case[i,0:2] = start
		test_case[i,2:4] = end
	return test_case

def if_permitStraightLineSoln(x1, x2, s1, y1, y2, s2, radius):
	t1 = np.linalg.norm(x2-x1)/s1
	t2 = np.linalg.norm(y2-y1)/s2
	if t1 < t2:
		x_crit = x2
		y_crit = y1+ t1 * (y2-y1) / t2
		if distPointToSegment(y_crit, y2, x_crit) < radius:
			return False
	else:
		x_crit = x1+ t2 * (x2-x1) / t1
		y_crit = y2
		if distPointToSegment(x_crit, x2, y_crit) < radius:
			return False
	start_dist = np.linalg.norm(x1-y1)
	end_dist = np.linalg.norm(x_crit-y_crit)
	mid_dist = find_dist_between_segs(x1, x_crit, y1, y_crit)
	dist = min(start_dist, end_dist, mid_dist)
	if dist<radius:
		return False
	return True



if __name__ == '__main__':
	speed_bnds = [0.5, 1.5]
	radius_bnds = [0.2, 0.8]

	num_test_cases = 100
	test_cases = []

	for i in range(num_test_cases):
		num_agents = np.random.randint(2, 4+1)
		side_length = np.random.uniform(4,8)
		# test_case = tc.generate_circle_case(num_agents, side_length, speed_bnds, radius_bnds)
		# test_case = tc.generate_swap_case(num_agents, side_length, speed_bnds, radius_bnds)
		# test_case = tc.generate_rand_case(num_agents, side_length, speed_bnds, radius_bnds, is_end_near_bnd = False)
		# test_case = tc.generate_easy_rand_case(num_agents, side_length, speed_bnds, radius_bnds, 4, is_end_near_bnd = False)
		test_case = generate_rand_test_case_multi(num_agents, side_length, speed_bnds, radius_bnds, is_end_near_bnd = False, is_static = False)
		test_cases.append(test_case)
	# print test_cases

	pickle.dump(test_cases, open("/home/mfe/ford_ws/src/2017-avrl/src/environment/Collision-Avoidance/test_cases/%s_agents_%i_cases.p" %("2_3_4", num_test_cases), "wb"))
