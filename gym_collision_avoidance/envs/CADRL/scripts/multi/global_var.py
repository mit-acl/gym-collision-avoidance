#!/usr/bin/env python
# global varibles
import numpy as np

COLLISION_COST = -0.25
DIST_2_GOAL_THRES = 0.05
GETTING_CLOSE_PENALTY = -0.05
GETTING_CLOSE_RANGE = 0.2
EPS = 1e-5
SMOOTH_COST = -0.5
# terminal states
NON_TERMINAL=0
COLLIDED=1
REACHED_GOAL=2
TRAINING_DT=1.0

# plotting colors
plt_colors = []
plt_colors.append([0.8500, 0.3250, 0.0980]) # red
plt_colors.append([0.0, 0.4470, 0.7410]) # blue 
plt_colors.append([0.4660, 0.6740, 0.1880]) # green 
plt_colors.append([0.4940, 0.1840, 0.5560]) # purple
plt_colors.append([0.9290, 0.6940, 0.1250]) # orange 
plt_colors.append([0.3010, 0.7450, 0.9330]) # cyan 
plt_colors.append([0.6350, 0.0780, 0.1840]) # chocolate 

# more plotting purpose
# plt_colors.append([0.0, 0.0, 0.0]) # black
# plt_colors.append([0.0, 0.0, 0.0]) # black
# plt_colors.append([0.0, 0.0, 0.0]) # black
# plt_colors.append([0.0, 0.0, 0.0]) # black
# plt_colors.append([0.0, 0.0, 0.0]) # black
# plt_colors.append([0.0, 0.0, 0.0]) # black
# plt_colors.append([0.0, 0.0, 0.0]) # black
# plt_colors.append([0.0, 0.0, 0.0]) # black
# plt_colors.append([0.0, 0.0, 0.0]) # black
# plt_colors.append([0.0, 0.0, 0.0]) # black
# plt_colors.append([0.0, 0.0, 0.0]) # black
# plt_colors.append([0.0, 0.0, 0.0]) # black

RL_gamma = 0.97
RL_dt_normal = 0.5

# neural network input/output vectors
# input_avg_vec = [np.array([10.0, 0.8, 0.7, 0.0, 0.65, 0.0, 0.35]), \
# 				np.array([0.0, 0.0, 0.0, 0.0, 0.35, 1.50, 8.0, 0.0])]
# input_std_vec = [np.array([8.0, 0.4, 0.4, 0.7, 0.5, 0.3, 0.2]), \
# 				np.array([0.8, 0.8, 8.0, 8.0, 0.2, 1.50, 6.0, 0.5])]
# output_avg_vec = np.array([0.5])
# output_std_vec = np.array([0.4])

input_avg_vec = [np.array([7.0, 0.8, 0.7, 0.0, 0.65, 0.0, 0.35]), \
				np.array([0.0, 0.0, 0.0, 0.0, 0.35, 0.7, 4.0, 0.5])]
input_std_vec = [np.array([5.0, 0.4, 0.4, 0.7, 0.5, 0.3, 0.2]), \
				np.array([0.8, 0.8, 4.0, 4.0, 0.2, 0.4, 4.0, 0.5])]
output_avg_vec = np.array([0.5])
output_std_vec = np.array([0.4])

NN_ranges = list()
NN_ranges.append(input_avg_vec); NN_ranges.append(input_std_vec)
NN_ranges.append(output_avg_vec); NN_ranges.append(output_std_vec)

# param computed from data
# input_avg_vec [ 3.6982656   0.83099974  0.7252522   0.0514733  -0.13318091  0.10315574
#   0.26505782 -0.11795232  0.39575324  0.40368014  0.79943338  0.65456946
#   0.02955785  2.90838236]
# input_std_vec [ 2.91446845  0.34856427  0.34699031  0.64742148  0.55879675  0.45500179
#   3.59396867  2.08934841  0.0561011   0.05553664  0.0812654   0.4022663
#   0.23505923  2.642269  ]
# out_avg_vec [ 0.43739441]
# output_std_vec [ 0.3618484]


