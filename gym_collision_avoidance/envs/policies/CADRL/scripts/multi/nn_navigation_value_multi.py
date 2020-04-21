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
from gym_collision_avoidance.envs.policies.CADRL.scripts.neural_networks.multiagent_network_param import Multiagent_network_param
from gym_collision_avoidance.envs.policies.CADRL.scripts.neural_networks.nn_training_param import NN_training_param
from gym_collision_avoidance.envs.policies.CADRL.scripts.multi import pedData_processing_multi as pedData
from gym_collision_avoidance.envs.policies.CADRL.scripts.multi import global_var as gb
from gym_collision_avoidance.envs.policies.CADRL.scripts.multi import gen_rand_testcases as gen_tc

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

# assume no kinematic constraints
def find_action_grids():
    nom_speed = 1.0
    num_angles = 18
    num_speeds = 5
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    speeds = np.linspace(nom_speed, 0.0, num_speeds, endpoint=False)
    
    angles, speeds = np.meshgrid(angles, speeds)
    angles = np.append([0.0], angles.flatten())
    speeds = np.append([0.0], speeds.flatten())

    actions = np.vstack((speeds, angles)).transpose()
    assert(actions.shape[0] == num_angles*num_speeds + 1)
    assert(actions.shape[1] == 2)

    # plot (for debugging)
    # fig = plt.figure(frameon=False)
    x = actions[:,0] * np.cos(actions[:,1])
    y = actions[:,0] * np.sin(actions[:,1])
    # plt.scatter(x, y, 50, cmap="rainbow")
    # plt.title('discrete actions')
    # plt.draw()
    # plt.show()

    return actions

# assume min/max acceleration over 1 seconds
def find_close_actions():
    num_angles = 6
    num_speeds = 4
    nom_speed = 1.0
    angles = np.linspace(-TURNING_LIMIT, TURNING_LIMIT, num_angles, endpoint=True)
    speeds = np.linspace(nom_speed, 0.0, num_speeds, endpoint=False)
    
    angles, speeds = np.meshgrid(angles, speeds)
    angles = np.append([0.0], angles.flatten())
    speeds = np.append([0.0], speeds.flatten())
    
    actions = np.vstack((speeds, angles)).transpose()
    # plot (for debugging)
    # fig = plt.figure(frameon=False)
    # x = actions[:,0] * np.cos(actions[:,1])
    # y = actions[:,0] * np.sin(actions[:,1])
    # plt.scatter(x, y, 50, cmap="rainbow")
    # plt.title('discrete actions')
    # plt.draw()
    # plt.show()
    return actions

# angle_1 - angle_2
# contains direction in range [-3.14, 3.14]
def find_angle_diff(angle_1, angle_2):
    angle_diff_raw = angle_1 - angle_2
    angle_diff = (angle_diff_raw + np.pi) % (2 * np.pi) - np.pi
    return angle_diff

def compute_multi_net_param(num_agents):
    num_others = num_agents - 1

    # no max layer
    # layers_type = []
    # layers_info = []
    # layers_info.append(np.array([[1, 7], [num_others, 8] ])); layers_type.append('conn')
    # layers_info.append(np.array([[1, 50], [num_others, 50]])); layers_type.append('conn')
    # layers_info.append(np.array([[1, 50], [num_others, 50]]));  layers_type.append('conn')
    # layers_info.append(np.array([[1, 50]])); layers_type.append('conn') 
    # layers_info.append(np.array([[1, 25]])); layers_type.append('conn') 
    # layers_info.append(np.array([[1, 1]]));

    # with max layer
    layers_type = []
    layers_info = []
    layers_info.append(np.array([[1, 7], [num_others, 8] ])); layers_type.append('conn')
    layers_info.append(np.array([[1, 50], [num_others, 50]])); layers_type.append('conn')
    layers_info.append(np.array([[1, 50], [num_others, 50]]));  layers_type.append('max')
    layers_info.append(np.array([[1, 50], [1, 50]]));  layers_type.append('conn')
    layers_info.append(np.array([[1, 50]]));  layers_type.append('conn')
    # layers_info.append(np.array([[1, 100]])); layers_type.append('conn') 
    # layers_info.append(np.array([[1, 25]])); layers_type.append('conn') 
    layers_info.append(np.array([[1, 1]]));

    # with max and self layer
    # layers_type = []
    # layers_info = []
    # layers_info.append(np.array([[1, 7], [num_others, 8] ])); layers_type.append('self')
    # layers_info.append(np.array([[1, 50], [num_others, 50] ])); layers_type.append('conn')
    # layers_info.append(np.array([[1, 50], [num_others, 50]])); layers_type.append('conn')
    # layers_info.append(np.array([[1, 50], [num_others, 50]]));  layers_type.append('max')
    # layers_info.append(np.array([[1, 50], [1, 50]]));  layers_type.append('conn')
    # layers_info.append(np.array([[1, 50]]));  layers_type.append('conn')
    # # layers_info.append(np.array([[1, 100]])); layers_type.append('conn') 
    # # layers_info.append(np.array([[1, 25]])); layers_type.append('conn') 
    # layers_info.append(np.array([[1, 1]]));

    multi_net_param = Multiagent_network_param(layers_info, layers_type)
    # for layer in range(len(layers_info)-1):
    #   print 'layer', layer
    #   print multi_net_param.symmetric_indices[layer]
    # raw_input()
    # print multi_net_param.symmetric_indices_b
    # raw_input()
    return layers_info, layers_type, multi_net_param

def find_nn_ranges(num_agents, NN_ranges):
    num_states = 7 + 8 * (num_agents-1)
    input_avg_vec = np.zeros((num_states,)); input_avg_vec[0:7] = NN_ranges[0][0].copy()
    input_std_vec = np.zeros((num_states,)); input_std_vec[0:7] = NN_ranges[1][0].copy()
    for i in range(num_agents-1):
        a = 7 + 8 * i
        b = 7 + 8 * (i+1)
        input_avg_vec[a:b] = NN_ranges[0][1].copy()
        input_std_vec[a:b] = NN_ranges[1][1].copy()
    output_avg_vec = NN_ranges[2]
    output_std_vec = NN_ranges[3]

    NN_ranges_processed = []
    NN_ranges_processed.append(input_avg_vec); NN_ranges_processed.append(input_std_vec)
    NN_ranges_processed.append(output_avg_vec); NN_ranges_processed.append(output_std_vec)
    return NN_ranges_processed


class NN_navigation_value:
    def __init__(self, num_agents, nn_training_param, mode='no_constr', passing_side='none'):
        self.num_agents = num_agents
        self.nn_training_param = nn_training_param
        self.nn = nn.Neural_network_regr_multi(self.nn_training_param)
        self.current_value = 0
        self.plot_actions = find_action_grids()
        self.close_actions = find_close_actions()
        self.test_vel_data = None
        self.dt_forward = 1.0
        self.radius_buffer = 0.0 # buffer around collision radius
        self.mode = mode
        self.passing_side = passing_side
        self.training_passing_side_weight = 0.5 #0.7 #0.2
        self.old_value_net = None

    # setup neural network
    def initialize_nn(self, num_agents):
        layers_info, layers_type, multi_net_param = compute_multi_net_param(num_agents)
        self.nn.initialize_network_param(layers_info, layers_type, multiagent_net_param=multi_net_param)


    def train_neural_network(self, training_data, test_data, test_vel_data=None):
        if not hasattr(self.nn, 'W'):
            self.initialize_nn(self.num_agents)

        # for plotting
        self.test_vel_data = test_vel_data
        self.nn.set_plotting_func(self.plot_ped_testCase_rand, test_data[0])
        # initialize neural network
        ERM = 0
        # self.nn.set_training_stepsize('sqrt_decay', 1.0, 0.1)
        # self.nn.set_training_stepsize('sum_of_grad', 0.1, 0.1)
        self.nn.set_training_stepsize('rmsprop', 0.1, 0.1)
        NN_ranges_processed = find_nn_ranges(self.num_agents, NN_ranges)
        self.nn.train_nn(training_data, ERM, test_data, input_output_ranges=NN_ranges_processed)
        self.test_vel_data = None

    def plot_ped_testCase_rand(self, X, Y_hat, title_string, figure_name=None):
        # Y_hat = [value]
        ind = np.random.randint(0, X.shape[0])
        # ind = 5
        x = X[ind,:]
        y_hat = Y_hat[ind,:]
        if (self.test_vel_data is None) or (ind > self.test_vel_data.shape[0]):
            y = None
        else:
            y = self.test_vel_data[ind,:]
        self.plot_ped_testCase(x, y_hat, title_string, figure_name, y)

    def plot_ped_testCase(self, x, y_hat, title_string, figure_name, y=None, plt_colors_custom=None):
        # print x.shape
        # print y_hat.shape
        # print y
        # raw_input()
        # new figure
        if figure_name is None:
            fig = plt.figure(figsize=(15, 6), frameon=False)
        else:
            fig = plt.figure(figure_name,figsize=(15, 6), frameon=False)
            plt.clf()

        if plt_colors_custom is None:
            plt_colors_local = plt_colors
        else:
            plt_colors_local = plt_colors_custom

        # convert to representation that's easier to plot
        a_s, other_agent_states = pedData.agentCentricState_2_rawState_noRotate(x)
        # print 'x', x
        # print 'a_s', a_s
        # print 'len(other_agent_states)', len(other_agent_states)
        # print a_s
        # print len(other_agent_states)
        # print other_agent_states
        # raw_input()

        # subfigure 1
        ax = fig.add_subplot(1, 2, 1)
        # agent at (0,0)
        circ1 = plt.Circle((0.0, 0.0), radius=a_s[8], fc='w', ec=plt_colors_local[0])
        ax.add_patch(circ1)
        # goal
        plt.plot(a_s[6], a_s[7], c=plt_colors_local[0], marker='*', markersize=20)
        # pref speed
        plt.arrow(0.0, 0.0, a_s[5], 0.0, fc='m', ec='m', head_width=0.05, head_length=0.1)
        vel_pref, = plt.plot([0.0, a_s[5]], [0.0, 0.0], 'm', linewidth=2)
        # current speed
        plt.arrow(0.0, 0.0, a_s[2], a_s[3], fc='k', ec='k', head_width=0.05, head_length=0.1)
        vel_cur, = plt.plot([0.0, a_s[2]], [0.0,  a_s[3]], 'k', linewidth=2)


        # actual speed
        if y != None:
            x_vel = y[0] 
            y_vel = y[1] 
            plt.arrow(0.0, 0.0, x_vel, y_vel,  fc=plt_colors_local[0], \
                ec=plt_colors_local[0], head_width=0.05, head_length=0.1)
            vel_select, = plt.plot([0.0, x_vel], [0.0, y_vel], \
                c=plt_colors_local[0], linewidth=2)

        # other agents
        for i, o_s in enumerate(other_agent_states):
            circ = plt.Circle((o_s[0], o_s[1]), radius=o_s[8], fc='w', ec=plt_colors_local[i+1])
            ax.add_patch(circ)
            # other agent's speed
            plt.arrow(o_s[0], o_s[1], o_s[2], o_s[3], fc=plt_colors_local[i+1], \
                ec=plt_colors_local[i+1], head_width=0.05, head_length=0.1)
            # vel_other, = plt.plot([o_s[0], o_s[0]+o_s[2]], [o_s[1], o_s[1]+o_s[3]], \
            #   c=plt_colors_local[i+1], linewidth=2)
        
        # meta data
        agent_state = a_s
        action = self.find_next_action(agent_state, other_agent_states)
        x_tmp = action[0] * np.cos(action[1]) 
        y_tmp = action[0] * np.sin(action[1])
        plt.arrow(0.0, 0.0, x_tmp, y_tmp, fc='g',\
            ec='g', head_width=0.05, head_length=0.1)
        vel_nn, = plt.plot([0.0, x_tmp], [0.0, y_tmp], 'g-', linewidth=2)
        value = self.find_next_states_values(agent_state, action.reshape([1,2]), other_agent_states)
        # plt.title(title_string + '; pred_value: %f, true_value: %f' % \
        #   (y_hat[0], y[0]))
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        dist_2_other_agents = np.zeros((len(other_agent_states), ))
        for i, other_agent_state in enumerate(other_agent_states):
            dist_2_other_agents[i] = np.linalg.norm(agent_state[0:2]-other_agent_state[0:2]) - \
                            agent_state[8] - other_agent_state[8]
        dist_2_other_agent = np.min(dist_2_other_agents)
        # print 'dist_2_other_agents', dist_2_other_agents
        # print 'dist_2_other_agent', dist_2_other_agent
        # raw_input()
        plt.title(title_string + '\n pref_speed: %.3f, min_dist_2_others: %.3f' % \
                                (a_s[5],dist_2_other_agent))
        # if y != None:
        #   plt.legend([vel_pref, vel_cur,vel_other, vel_select, vel_nn], \
        #       ['vel_pref', 'vel_cur', 'vel_other', 'vel_select', 'vel_nn'])
        # else:
        #   plt.legend([vel_pref, vel_cur, vel_other, vel_nn], \
        #       ['vel_pref', 'vel_cur', 'vel_other', 'vel_nn'])
        plt.legend([vel_cur, vel_pref, vel_nn], \
                ['${heading}$', '$v_{pref}$','$v_{select}$'], \
                loc='lower left', fontsize=30, frameon=False)
    
        ax.axis('equal')
        xlim = ax.get_xlim()
        new_xlim = np.array((xlim[0], xlim[1]+0.5))
        ax.set_xlim(new_xlim)
        # plotting style (only show axis on bottom and left)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # second subfigure
        # ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax = fig.add_subplot(1, 2, 2)
        # print 'before', agent_state
        # agent_state, other_agent_states = \
        #       pedData.agentCentricState_2_rawState_noRotate(x)
        # print 'after', agent_state
        if self.mode == 'no_constr':
            ''' all possible actions '''
            ''' actions choice 1 '''
            default_action_xy = agent_state[2:4]
            speed = np.linalg.norm(default_action_xy)
            angle_select = np.arctan2(default_action_xy[1], default_action_xy[0])

            # default_action_theta = np.array([speed, angle_select])
            # actions = self.find_actions_theta(agent_state, default_action_theta)
            
            # circular for plotting
            actions = self.plot_actions.copy()
            actions[:,1] = (actions[:,1] + np.pi) % (2 * np.pi) - np.pi
            actions[:,0] *= a_s[5]
            # turning needs to be slower
            # angle_diff_abs = abs(find_angle_diff(actions[:,1],agent_state[4]))
            # actions[:,0] *= (1 - angle_diff_abs / (2*np.pi))
        elif self.mode == 'rotate_constr':
            ''' dynamically feasible actions '''
            # print 'x', x
            # print 'agent_state', agent_state
            # print 'other_agent_state', other_agent_state
            cur_heading = agent_state[4]; desired_speed = agent_state[5]
            actions = self.close_actions.copy()
            actions[:,0] *= desired_speed
            print('rotate_constr')
            actions[:,1] = actions[:,1] + cur_heading
            actions[:,1] = (actions[:,1] + np.pi) % (2 * np.pi) - np.pi
            # actions = self.find_actions_theta_dynConstr(agent_state, 1.0)
            # print cur_heading
        else:
            assert(0)

        
        # agent_state, other_agent_state = \
            # pedData.agentCentricState_2_rawState_noRotate(x)
        # actions, accels_theta = self.find_actions_theta_dynConstr(agent_state, 1.0)
        
        plot_x = actions[:,0] * np.cos(actions[:,1])
        plot_y = actions[:,0] * np.sin(actions[:,1])
        # assert(0)
        # print 'before, agent state', agent_state
        # print 'before, other_agent state',other_agent_state

        # print 'after, agent state', agent_state
        # print 'after, other_agent state', other_agent_state
        plot_z = self.find_next_states_values(agent_state, actions, other_agent_states)
        # print actions.shape, plot_x.shape
        # print np.hstack((actions, plot_z.reshape(plot_z.shape[0], 1)))
        value = np.amax(plot_z)
        ind = np.argmax(plot_z)
        x_tmp = actions[ind, 0] * np.cos(actions[ind, 1])
        y_tmp = actions[ind, 0] * np.sin(actions[ind, 1])
        ''' plot using plot_trisurf (2D view of a 3D plot)'''
        # print plot_x.shape, plot_y.shape, plot_z.shape
        # ax = fig.gca(projection='3d')
        # im = ax.plot_trisurf(plot_x, plot_y, plot_z, cmap=cm.jet, linewidth=0.2)
        # ax.view_init(90.0, 270.0)
        # plt.title('value of best action: %.3f' % value)
        # fig.colorbar(im, shrink=0.5)
        
        ''' plot using tripcolor (2D plot) '''
        # triang = tri.Triangulation(plot_x, plot_y)
        color_min_inds = np.where(plot_z>0)[0]
        if len(color_min_inds) > 0:
            color_min = np.amin(plot_z[color_min_inds]) - 0.05
        else:
            color_min = 0.0
        color_max = max(np.amax(plot_z),0.0)
        # plt.tripcolor(plot_x, plot_y, plot_z, shading='flat', \
        #       cmap=plt.cm.rainbow, edgecolors='k',vmin=color_min, vmax=color_max)
        plt.tripcolor(plot_x, plot_y, plot_z, shading='flat', \
                cmap=plt.cm.rainbow, vmin=color_min, vmax=color_max)
        if actions[ind, 0] > EPS:
            plt.title('value of best action: %.3f \n action_x %.3f, action_y %.3f' \
                % (value, x_tmp, y_tmp))
        else:
            plt.title('value of best action: %.3f \n action_speed %.3f, action_angle %.3f' \
                % (value, actions[ind, 0], actions[ind, 1]))
        plt.xlabel('v_x (m/s)')
        plt.ylabel('v_y (m/s)')
        cbar = plt.colorbar()
        cbar.set_ticks([color_min,(color_min+color_max)/2.0,color_max])
        cbar.ax.set_yticklabels(['%.3f'%color_min, \
                            '%.3f'%((color_min+color_max)/2.0), \
                            '%.3f'%color_max])

        # plotting style (only show axis on bottom and left)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        plt.draw()
        plt.pause(0.0001)
        # raw_input()

    def find_actions_theta(self, agent_state, default_action_theta):
        # action = [speed theta]
        num_near_actions = 10
        num_rand_actions = 5

        # zero action
        zero_action = np.zeros((1,2))

        # desired action
        # cur_state = agent_state[0:2]
        # goal = agent_state[6:8]
        # nom_speed = agent_state[5]
        # desired_act = self.computePrefVel(cur_state, goal, nom_speed)
        desired_act = np.array([agent_state[5], \
                        np.arctan2(agent_state[7]-agent_state[1], \
                                agent_state[6]-agent_state[0])])
        desired_actions = np.matlib.repmat(desired_act, 5, 1)
        desired_actions[1,0] *= 0.80 
        desired_actions[2,0] *= 0.60
        desired_actions[3,0] *= 0.40 
        desired_actions[4,0] *= 0.20

        # near by actions: default action with perturbed action & reduced speed
        tmp_action_theta = default_action_theta.copy()
        # tmp_action_theta[0] = np.amax((0.75 * desired_act[0], default_action_theta[0]))
        tmp_action_theta[0] = agent_state[5]
        # tmp_action_theta[1] = desired_act[1]
        near_actions = np.matlib.repmat(tmp_action_theta, num_near_actions, 1)
        near_actions[:,1] += np.linspace(-np.pi/3.0, np.pi/3.0, num=num_near_actions)


        near_actions_reduced = near_actions.copy()
        near_actions_reduced_1 = near_actions.copy()
        near_actions_reduced_2 = near_actions.copy()
        near_actions_reduced[:,0] *= 0.75 #0.75
        near_actions_reduced_1[:,0] *= 0.50 #0.75
        near_actions_reduced_2[:,0] *= 0.25 #0.75

        # near_actions = np.vstack((near_actions, near_actions_reduced))
        near_actions = np.vstack((near_actions, near_actions_reduced, \
            near_actions_reduced_1, near_actions_reduced_2))

        # random actions: random actions with max speed at desired speed
        # pref_speed = agent_state[5]
        # rand_actions = np.zeros((num_rand_actions, 2))
        # rand_actions[:,0] = pref_speed * np.random.rand(num_rand_actions,)
        # rand_actions[:,1] = 2 * np.pi * np.random.rand(num_rand_actions,) - np.pi



        # put all actions together
        actions = np.vstack((default_action_theta, desired_actions, \
                            zero_action, near_actions)) #, rand_actions))

        # avoid oscillation
        # angles_diff = find_angle_diff(actions[:,1], agent_state[4])
        # if agent_state[9] > EPS:
        #   valid_inds = np.where(angles_diff > -EPS)[0]
        #   actions = actions[valid_inds,:]
        # elif agent_state[9] < -EPS:
        #   valid_inds = np.where(angles_diff < EPS)[0]
        #   actions = actions[valid_inds,:]


        # plot_actions = self.plot_actions.copy()
        # plot_actions[:,1] += default_action_theta[1]
        # plot_actions[:,0] *= agent_state[5]
        # actions = np.vstack((default_action_theta, desired_actions, \
                            # plot_actions))
        # actions = np.vstack((default_action_theta, zero_action, near_actions, rand_actions))
        actions[:,1] = (actions[:,1] + np.pi) % (np.pi * 2) - np.pi

        # turning needs to be slower
        # angle_diff_abs = abs(find_angle_diff(actions[:,1],agent_state[4]))
        # actions[:,0] *= (1 - angle_diff_abs / (2*np.pi))
        return actions

    def find_actions_theta_dynConstr(self, agent_state, dt):
        assert(dt > 0.9)
        angle_lim = TURNING_LIMIT*min(dt,1.0)
        # action = [speed theta]
        cur_heading = agent_state[4]
        desired_speed = agent_state[5]

        # near by actions: default action
        num_near_actions = 10
        cur_speed = np.linalg.norm(agent_state[2:4])
        actions = self.close_actions.copy()
        # actions[:,0] *= max(0.75 * desired_speed, cur_speed)
        actions[:,0] *= desired_speed
        actions[:,1] = actions[:,1] + cur_heading

        # desired action: add if desired_heading is within reachable range
        # of cur_heading
        desired_heading = np.arctan2(agent_state[7]-agent_state[1], \
                                agent_state[6]-agent_state[0])
        angle_diff_abs = abs(find_angle_diff(desired_heading, cur_heading))
        if  angle_diff_abs < angle_lim:
            desired_act = np.array([desired_speed, desired_heading])
            desired_actions = np.matlib.repmat(desired_act, 5, 1)
            desired_actions[1,0] *= 0.80 
            desired_actions[2,0] *= 0.60
            desired_actions[3,0] *= 0.40 
            desired_actions[4,0] *= 0.20
            # put all actions together
            actions = np.vstack((desired_actions, actions))
            # print '---- desired actions added'
        
        # default action: add if default_heading is within reachable range
        # of cur_heading
        default_heading = np.arctan2(agent_state[3], agent_state[2])
        angle_diff_abs = abs(find_angle_diff(default_heading, cur_heading))
        if  angle_diff_abs < angle_lim and cur_speed > 0.05:
            default_act = np.array([cur_speed, default_heading])
            default_actions = np.matlib.repmat(default_act, 2, 1)
            default_actions[1,0] *= 0.75 
            # put all actions together
            actions = np.vstack((default_actions, actions))
            # print '---- default actions added'

        # turning on spot
        # min_turning_radius = 0.5
        limit = TURNING_LIMIT
        # print 'desired_speed, dt, limit', desired_speed, dt, limit
        # print 'dt', dt
        added_actions = np.array([[0.0, limit + cur_heading], \
                [0.0, 0.66 * limit + cur_heading], \
                [0.0, 0.33 * limit + cur_heading], \
                [0.0, -0.33 * limit + cur_heading], \
                [0.0, -0.66 * limit + cur_heading], \
                [0.0, -limit + cur_heading]])
        actions = np.vstack((actions, added_actions))
        

        # getting unique rows
        # print 'before', actions
        actions = np.asarray(np.vstack([tuple(row) for row in actions]))
        # raw_input()
        # print 'after', actions
        # raw_input()
        actions[:,1] = (actions[:,1] + np.pi) % (np.pi * 2) - np.pi

        # turning needs to be slower
        # angle_diff_abs = abs(find_angle_diff(actions[:,1],agent_state[4]))
        # actions[:,0] *= (1 - angle_diff_abs / (2*np.pi))

        return actions

    # cost of action (smoothness)
    def find_state_action_cost(self, agent_state, actions_theta, dt_forward):
        cur_heading = agent_state[4]
        angle_diff = find_angle_diff(actions_theta[:,1], cur_heading)
        speed_diff = (actions_theta[:,0] - np.linalg.norm(agent_state[2:4]))
        
        # bias turning right early on
        # left_inds = np.where((angle_diff>0))[0]
        # angle_diff_abs = angle_diff
        # if np.linalg.norm(agent_state[0:2]-agent_state[6:8]) > 3.0:
        #   angle_diff_abs[left_inds] = 1.0 * abs(angle_diff[left_inds])
        # else:
        #   angle_diff_abs[left_inds] = abs(angle_diff[left_inds])

        # right_inds = np.where((angle_diff<0))[0]
        # angle_diff_abs[right_inds] = abs(angle_diff[right_inds])
        angle_diff_abs = abs(angle_diff) / np.pi
        zero_inds = np.where((actions_theta[:,1] < EPS) | (angle_diff_abs < np.pi/12.0) )[0]
        angle_diff_abs[zero_inds] = 0

        speed_diff_abs = abs(speed_diff) / agent_state[5]
        zero_inds = np.where( (speed_diff_abs < 0.5) )[0]
        speed_diff_abs[zero_inds] = 0


        # print 'angle_diff', angle_diff
        # print 'angle_diff_abs', angle_diff_abs
        # print 'speed_diff', speed_diff
        # print 'speed_diff_abs', speed_diff_abs
        assert(SMOOTH_COST < 0)

        cost = np.clip(angle_diff_abs * SMOOTH_COST, -0.25, 0) + \
            np.clip(speed_diff_abs * SMOOTH_COST, -0.25, 0)
        # cost = np.clip(cost, -0.5, 0.0)
        # print cost

        d = np.linalg.norm(agent_state[0:2] - agent_state[6:8])
        v = agent_state[5]
        getting_close_penalty = abs(GAMMA ** (d/DT_NORMAL) * (1.0 - GAMMA ** (-v/DT_NORMAL)))
        assert(np.all(cost <= 0))
        # return cost
        smoothness_cost = cost * getting_close_penalty
        # print smoothness_cost
        # raw_input()
        return smoothness_cost


    # limited to this application
    # future implementation should generalize 
    def find_action_rewards(self, agent_state, cur_dist, min_dists, dt_forward):
        # print 'cur_dist', cur_dist
        # print 'min_dists', min_dists
        rewards = np.zeros((len(min_dists),))
        # collision
        if cur_dist < 0:
            rewards[:] = COLLISION_COST
            return rewards

        d = np.linalg.norm(agent_state[0:2] - agent_state[6:8])
        v = agent_state[5]
        getting_close_penalty = GAMMA ** (d/DT_NORMAL) * (1.0 - GAMMA ** (-v/DT_NORMAL))
        
        # getting to close
        close_inds = np.where((min_dists > 0) & \
            (min_dists < GETTING_CLOSE_RANGE))[0]

        # current pos
        if cur_dist < GETTING_CLOSE_RANGE:
            assert(GETTING_CLOSE_RANGE - cur_dist > 0)
            rewards[:] = getting_close_penalty
        
        # future pos 
        rewards[close_inds] += getting_close_penalty

        collision_inds = np.where(min_dists < 0)[0]
        rewards[collision_inds] = COLLISION_COST
        

        # rewards[close_inds] = rewards[close_inds] + GETTING_CLOSE_PENALTY \
        #   - 0.3 * (GETTING_CLOSE_RANGE - min_dists[close_inds])

        scaling_cur = 2
        scaling_future = 5
        # scaling_future = 20
        rewards[close_inds] = scaling_cur * rewards[close_inds] \
            + scaling_future * getting_close_penalty * (GETTING_CLOSE_RANGE - min_dists[close_inds])

        rewards[close_inds] = np.clip(rewards[close_inds], COLLISION_COST+0.01, 0.0)
        assert(np.all(GETTING_CLOSE_RANGE - min_dists[close_inds]>0))

        # other states are 
        return rewards

    
    def find_passing_side_cost(self, agent_state, actions_theta, other_agents_state, \
        other_agents_action_theta, dt_forward):
        weight = self.training_passing_side_weight
        # print weight
        num_pts = len(actions_theta)


        num_states = 7 + 8 * (self.num_agents-1)
        agent_centric_states = np.zeros((num_pts, num_states))
        agent_next_states = self.update_states(agent_state, \
                            actions_theta, dt_forward)
        
        # only use the closest other agent
        dist_2_others = [(np.linalg.norm(other_agent_state[0:2]-agent_state[0:2]) - \
                other_agent_state[8] - agent_state[8]) for other_agent_state in other_agents_state]
        agent_num = np.argmin(np.array(dist_2_others))
        other_agent_next_state = self.update_state(other_agents_state[agent_num], \
                            other_agents_action_theta[agent_num], dt_forward)
        other_agents_next_state = [other_agent_next_state]
        

        ref_prll_vec, ref_orth_vec, agent_centric_states = \
            pedData.rawStates_2_agentCentricStates(\
                agent_next_states, other_agents_next_state, self.num_agents)

        # for i in range(num_pts):
        #   ref_prll, ref_orth, agent_centric_states[i,:] = \
        #       pedData.rawState_2_agentCentricState( \
        #       agent_next_states[i,:], other_agent_next_state)

        bad_inds_oppo, bad_inds_same, bad_inds_tangent = \
            self.find_bad_inds(agent_centric_states)
        #scaling factor
        d = np.linalg.norm(agent_state[0:2] - agent_state[6:8])
        v = agent_state[5]
        getting_close_penalty =np.ones((num_pts,)) \
            * GAMMA ** (d/DT_NORMAL) * (1.0 - GAMMA ** (-v/DT_NORMAL))
        penalty = np.zeros((num_pts,))
        penalty[bad_inds_oppo] = weight * getting_close_penalty[bad_inds_oppo]
        penalty[bad_inds_same] = 1.0 * weight * getting_close_penalty[bad_inds_same]
        penalty[bad_inds_tangent] = weight * getting_close_penalty[bad_inds_tangent]

        return penalty
        
        

    # for RL 
    # back out information required by find_next_states_values
    def find_next_state_pair_value_and_action_reward(self, agent_state, agent_next_state, \
        other_agents_state, other_agents_next_state, dt_forward):
        # agents
        # action_xy = agent_next_state[2:4]
        action_xy = (agent_next_state[0:2] - agent_state[0:2]) / dt_forward
        action_speed = np.linalg.norm(action_xy)
        if action_speed > EPS:
            action_angle = np.arctan2(action_xy[1], action_xy[0])
        else:
            action_angle = agent_next_state[4]
        action_theta = np.array([[action_speed, action_angle]])

        # other agents (TODO: vectorize)
        num_other_agents = len(other_agents_state)
        other_actions_theta = []
        for i, other_agent_next_state in enumerate(other_agents_next_state):
            # other_action_xy = other_agent_next_state[2:4]
            other_action_xy = (other_agent_next_state[0:2] - other_agents_state[i][0:2]) / dt_forward
            other_action_speed = np.linalg.norm(other_action_xy)
            if other_action_speed > EPS:
                other_action_angle = np.arctan2(other_action_xy[1], other_action_xy[0])
            else:
                other_action_angle = other_agent_next_state[4]
            other_actions_theta.append(np.array([other_action_speed, other_action_angle]))

        state_value, action_reward = \
            self.find_values_and_action_rewards(agent_state, action_theta, \
                            other_agents_state, other_actions_theta, dt_forward)

        return state_value, action_reward

    def check_collisions_and_get_action_rewards(self, agent_state, actions_theta, \
                            other_agents_state_in, other_agents_action=None, dt_forward=None):
        actions_theta_copy = actions_theta.copy()
        other_agents_state = copy.deepcopy(other_agents_state_in)

        # ref_prll, ref_orth, state_nn = \
        #   pedData.rawState_2_agentCentricState( \
        #   agent_state, other_agents_state, self.num_agents)
        num_actions = actions_theta.shape[0]
        # update other agent state
        num_other_agents = len(other_agents_state)
        if other_agents_action is None:
            other_agents_action = []
            for tt in range(num_other_agents):
                other_agent_speed = np.linalg.norm(other_agents_state_in[tt][2:4])
                other_agent_angle = np.arctan2(other_agents_state_in[tt][3], other_agents_state_in[tt][2])
                other_agents_action.append(np.array([other_agent_speed, other_agent_angle]))
        # update other agents' velocity
        for tt in range(num_other_agents):
            other_agents_state[tt][2] = other_agents_action[tt][0] * np.cos(other_agents_action[tt][1])
            other_agents_state[tt][3] = other_agents_action[tt][0] * np.sin(other_agents_action[tt][1])

        # assume other agent is heading toward the vehicle
        # rel_pos = agent_state[0:2] - other_agent_state[0:2]
        # rel_angle = np.arctan2(rel_pos[1], rel_pos[0])
        # angle_diff = find_angle_diff(rel_angle, other_agent_action[1])
        # other_agent_action[1] += max(-np.pi/9.0, min(np.pi/9.0, angle_diff))

        # print 'other_agent_state, other_agent_action', other_agent_state, other_agent_action
        other_agents_next_state = []
        for tt in range(num_other_agents):
            # dt_forward_other = min(1,0, dt_forward)
            other_agents_next_state.append(self.update_state(other_agents_state[tt], \
                            other_agents_action[tt], dt_forward))
        # other_agent_next_state = (other_agent_state + other_agent_next_state) / 2.0
        agent_desired_speed = agent_state[5]
        # print 'other_agent_action', other_agent_action
        # compute values for each state


        # collide (not just getting close)
        min_dists_mat = np.zeros((num_actions, num_other_agents))
        if_collide_mat = np.zeros((num_actions, num_other_agents))
        cur_dist_vec = np.zeros((num_other_agents,))
        for tt in range(num_other_agents):
            min_dists_mat[:,tt], if_collide_mat[:,tt] = self.if_actions_collide(agent_state, 
                        actions_theta, other_agents_state[tt], other_agents_action[tt], dt_forward)
        
            radius = agent_state[8] + other_agents_state[tt][8] + self.radius_buffer
            cur_dist_vec[tt] = np.linalg.norm(agent_state[0:2]-other_agents_state[tt][0:2]) - radius

        min_dists = np.min(min_dists_mat, axis=1)
        if_collide = np.max(if_collide_mat, axis=1)
        cur_dist = np.min(cur_dist_vec)

        # print 'min_dists_mat', min_dists_mat
        # print 'if_collide_mat', if_collide_mat
        # print 'cur_dist_vec', cur_dist_vec

        # print 'min_dists', min_dists
        # print 'if_collide', if_collide
        # print 'cur_dist', cur_dist
        # raw_input()

        action_rewards = self.find_action_rewards(agent_state, cur_dist, min_dists, dt_forward)
        # print 'action_rewards', action_rewards

        # print 'action_rewards', action_rewards
        return if_collide, action_rewards, min_dists, other_agents_next_state, num_actions, other_agents_state

    def find_values_and_action_rewards(self, agent_state, actions_theta, \
                            other_agents_state_in, other_agents_action=None, dt_forward=None):
        if_collide, action_rewards, min_dists, other_agents_next_state, num_actions, other_agents_state = self.check_collisions_and_get_action_rewards(agent_state, actions_theta, \
                            other_agents_state_in, other_agents_action, dt_forward)
        state_values = np.zeros((num_actions,))
        non_collision_inds = np.where(if_collide==False)[0]

        # find states_values in batch
        gamma = GAMMA
        dt_normal = DT_NORMAL
        if len(non_collision_inds) > 0:
            agent_next_states = self.update_states(agent_state, \
                    actions_theta[non_collision_inds,:], dt_forward)

            # find next_states that have reached the goal
            dists_to_goal = np.linalg.norm(agent_next_states[:,0:2]-agent_next_states[:,6:8],axis=1)

            reached_goals_inds = np.where((dists_to_goal < DIST_2_GOAL_THRES) & \
                                (min_dists[non_collision_inds]>GETTING_CLOSE_RANGE))[0]
            # not_reached_goals_inds = np.where(dists_to_goal >= DIST_2_GOAL_THRES)[0]
            not_reached_goals_inds = np.setdiff1d(np.arange(len(non_collision_inds)), reached_goals_inds)
            
            non_collision_reached_goals_inds = non_collision_inds[reached_goals_inds]
            non_collision_not_reached_goals_inds = non_collision_inds[not_reached_goals_inds]
            
            state_values[non_collision_not_reached_goals_inds] = \
                self.find_states_values(agent_next_states[not_reached_goals_inds], other_agents_next_state)


            # state_values[non_collision_reached_goals_inds] = gamma ** (dists_to_goal / dt_normal)
            # print 'dists_to_goal', dists_to_goal
            # print 'reached_goals_inds', reached_goals_inds
            # print 'non_collision_reached_goals_inds', non_collision_reached_goals_inds
            # print 'non_collision_not_reached_goals_inds', non_collision_not_reached_goals_inds
            state_values[non_collision_reached_goals_inds] = \
                gamma ** (dists_to_goal[reached_goals_inds] / dt_normal)

        try:
            assert(np.all(action_rewards + state_values < 1.0001))
        except:
            print('agent_state', agent_state)
            print('actions_theta', actions_theta)
            print('other_agent_state', other_agent_state)
            print('other_agent_action', other_agent_action)
            print('dt_forward', dt_forward)
            print('dists_to_goal', dists_to_goal)
            print('actions_rewerds', action_rewards)
            print('state_values', state_values)
            print('values', action_rewards + gamma ** (dt_forward * \
                agent_desired_speed / dt_normal) * state_values)
            assert(0)       
        
        # np.set_printoptions(precision=4,formatter={'float': '{: 0.3f}'.format})
        # if True:#np.max(actions_theta[:,1]) - np.min(actions_theta[:,1]) < 1.5 * np.pi:
        #   actions_theta[:,1] -= agent_state[4]
        #   print '---------------------------'
        #   dist_between_agents = (np.linalg.norm(agent_state[0:2]-other_agents_state[0][0:2]) - \
        #                       agent_state[8] - other_agents_state[0][8])
        #   print 'dist between agents %.3f' % dist_between_agents
            
        #   values = action_rewards + gamma ** (dt_forward * \
        #           agent_desired_speed/ dt_normal) *state_values
        #   print np.hstack((actions_theta, np.vstack((min_dists, action_rewards,\
        #       values, state_values)).transpose()))
        #   best_action_ind = np.argmax(values)
        #   print 'current heading', agent_state[4]
        #   print 'best_action', actions_theta[best_action_ind,:]
        #   print 'next_best_state', agent_next_states[best_action_ind,:]
        #   print 'other_agents_state[0]', other_agents_state[0]
        #   print 'other_agents_next_state[0]', other_agents_next_state[0]
        #   print 'other_agents_action[0]', other_agents_action[0]
        #   print other_agents_state_in[0]
        #   actions_theta[:,1] += agent_state[4]

        
        # # print gamma ** (dt_forward * \
        # #         agent_desired_speed/ dt_normal)
        # # print 'state_values', state_values
        # # print 'dt_forward', dt_forward
        # # print 'agent_state', agent_state 
        # # print 'other_agent_state', other_agent_state
        # # print 'other_agent_action', other_agent_action
        # # print 'other_agent_next_state', other_agent_next_state
        # # print agent_next_states.shape
        # # print actions_theta.shape
        # # print state_values[:,np.newaxis].shape
        # # print 'agent_next_states', np.hstack((agent_next_states, actions_theta[non_collision_inds], \
        # #     state_values[non_collision_inds,np.newaxis]))
        # print 'state_nn', state_nn
        # # print 'dt_forward', dt_forward
        # actions_theta[:,1] += agent_state[4]
        # # raw_input()
        # # if dist_between_agents < GETTING_CLOSE_RANGE:
        # #     raw_input()

        # assert(np.sum(actions_theta_copy - actions_theta) < 0.0001)
        # smoothness_cost = self.find_state_action_cost(agent_state, actions_theta, dt_forward)
        passing_side_cost = self.find_passing_side_cost(agent_state, actions_theta,\
            other_agents_state, other_agents_action, dt_forward)
        # print passing_side_cost
        # raw_input()
        # return state_values, action_rewards + smoothness_cost + passing_side_cost
        # if len(state_values)>5:
        #   print 'state_values', state_values
        #   print 'action_rewards', action_rewards
        #   print 'min_dists', min_dists
        #   num_states = len(state_values)
        #   # dt_forward_vec = 0.5 * np.ones((num_states,))
        #   # dt_forward_vec += 0.5 * actions_theta[:,0] / agent_desired_speed
        #   dt_forward_vec = np.ones((num_states, ))
        #   dt_forward_vec *= dt_forward
        #   total = action_rewards + gamma ** (dt_forward_vec * \
        #       agent_desired_speed / dt_normal) * state_values
        #   best_ind = np.argmax(total)
        #   d = np.linalg.norm(agent_state[0:2]-agent_state[6:8])
        #   v = agent_state[5]
        #   getting_close_penalty = GAMMA ** (d/DT_NORMAL) * (1.0 - GAMMA ** (-v/DT_NORMAL))
        #   print 'getting_close_penalty', getting_close_penalty
        #   print 'best_ind %d, best_value %.3f, min_dist[min_ind] %.3f, action_rewards[min_ind] %.3f, state_values[min_ind] %.3f\n' % \
        #       (best_ind, total[best_ind], min_dists[best_ind], \
        #       action_rewards[best_ind], state_values[best_ind])
        #   print 'dt_forward', dt_forward
        #   print 'current_dist', np.linalg.norm(agent_state[0:2] - agent_state[6:8])
        #   print 'next_dist_2_goal', np.linalg.norm(agent_next_states[:,0:2]-agent_next_states[:,6:8], axis=1)
        #   print 'speed', actions_theta[:,0]
        return state_values, action_rewards + passing_side_cost #+ smoothness_cost

    # assuming the other agent follows its current speed
    # agent_state: agent's current state
    # actions_theta: multiple actions considered
    # other_agent_state: other agent's current state
    def find_next_states_values(self, agent_state, actions_theta, \
                            other_agents_state, other_agents_action=None, dt_forward=None):
        values, state_values, action_rewards, dt_forward = self.find_next_states_values_and_components(agent_state, actions_theta, other_agents_state, other_agents_action, dt_forward)
        return values

    def find_next_states_values_and_components(self, agent_state, actions_theta, \
                            other_agents_state, other_agents_action=None, dt_forward=None):
        # making sure look ahead time is not too long (reaching goal)
        # or too short (if agent's desired speed is too slow)
        if dt_forward is None:
            agent_speed = agent_state[5]
            dt_forward_max = max(self.dt_forward, 0.5/agent_speed)
            # dt_forward_max = self.dt_forward
            dist_to_goal = np.linalg.norm(agent_state[6:8]- agent_state[0:2])
            time_to_goal = dist_to_goal / agent_speed
            dt_forward = min(dt_forward_max, time_to_goal) #1.0

        state_values, action_rewards = \
            self.find_values_and_action_rewards(agent_state, actions_theta, \
                            other_agents_state, other_agents_action, dt_forward)

        gamma = GAMMA
        dt_normal = DT_NORMAL
        agent_desired_speed = agent_state[5]

        num_states = len(actions_theta)
        # method 1
        # dt_forward_vec = np.ones((num_states,))
        # dt_forward_vec *= dt_forward
        # method 2
        dt_forward_vec = 0.2 * np.ones((num_states,)) * dt_forward
        dt_forward_vec += 0.8 * actions_theta[:,0] / agent_desired_speed * dt_forward
        # print 'dt_forwards', dt_forward
        # print 'dt_forwards_vec', dt_forward_vec
        # raw_input()
        values =  action_rewards + gamma ** (dt_forward_vec * \
            agent_desired_speed / dt_normal) * state_values

        return values, state_values, action_rewards, dt_forward

    def find_feasible_actions(self, agent_state, static_constraints=None):
        # print 'agent_state', agent_state
        if self.mode == 'no_constr':
            ''' actions choice 1 '''
            default_action_xy = agent_state[2:4]
            speed = np.linalg.norm(default_action_xy)
            # angle_select = np.arctan2(default_action_xy[1], default_action_xy[0])
            angle_select = agent_state[4]

            default_action_theta = np.array([speed, angle_select])
            actions_theta = self.find_actions_theta(agent_state, default_action_theta)

            # possibly stuck
            # if speed < agent_state[5] * 0.1:
            #   actions_theta_tmp = self.plot_actions.copy()    
            #   actions_theta_tmp[:,0] *= agent_state[5]
            #   actions_theta_tmp[:,1] = (actions_theta_tmp[:,1] - np.pi)  / 2.0
            #   actions_theta_tmp[:,1] += np.arctan2(agent_state[7]-agent_state[1], \
            #                   agent_state[6]-agent_state[0])
            #   actions_theta = np.vstack((actions_theta, actions_theta_tmp))
            #   actions_theta[:,1] = (actions_theta[:,1] + np.pi) % (np.pi * 2) - np.pi

            # print 'agent_state', agent_state
            # print 'actions_theta', actions_theta
        elif self.mode == 'rotate_constr':
            ''' actions choice 2 '''
            # need to handle acceleration better (limits on changing velocity as a function of time)
            actions_theta = self.find_actions_theta_dynConstr(agent_state, 1.0)
        else:
            assert(0)

        # print actions_theta
        # print '---'
        # print 'before, max_speed', np.max(actions_theta[:,0])
        # prune actions if there are static constraints (from static obstacles in map)
        # static_constraints: max_speeds, angles
        if static_constraints is not None:
            # print static_constraints
            angle_incr = abs(static_constraints[2,1] - static_constraints[1,1])
            # check
            # angle_incr_vec = abs(static_constraints[1:,1] - static_constraints[0:-1,1])
            # try:
            #   assert(np.all(angle_incr_vec - angle_incr<EPS))
            # except AssertionError:
            #   print 'static_constraints error (angle spacing)',
            #   print static_constraints
            #   assert(0)

            # min_angle = static_constraints[0,1]
            # lower_inds = ((actions_theta[:,1]-min_angle) / angle_incr).astype(int)
            # upper_inds = lower_inds + 1
            # print 'theta_col:',actions_theta[:,1]
            # print 'stat col:',static_constraints[:,1]
            # print 'len theta', len(actions_theta[:,1])
            # print 'len stat', len(static_constraints[:,1])
            upper_inds = np.digitize(actions_theta[:,1], static_constraints[:,1])
            # print 'lower_inds', lower_inds
            lower_inds = upper_inds - 1
            # print 'upper_inds', upper_inds
            try:
                assert(np.all(lower_inds<len(static_constraints)))
            except AssertionError:
                print('lower_inds:',lower_inds)
                print('len(actions_theta):', len(actions_theta))
                bad_ind = np.where(lower_inds>=len(actions_theta))[0]
                print('actions theta:',actions_theta)
                print('bad_ind', actions_theta[bad_ind,1])
                print('static_constraints', static_constraints)
                assert(0)

            alpha = (actions_theta[:,1] - static_constraints[lower_inds,1]) / angle_incr
            # print 'static_constraints', static_constraints
            # print 'before---', actions_theta
            # print static_constraints
            # print 'alpha', alpha
            max_speeds_int = alpha * static_constraints[upper_inds,0] + (1-alpha) * static_constraints[lower_inds,0]
            pref_speed = agent_state[5]
            # assert(np.all(max_speeds_int <=pref_speed))
            actions_theta[:,0] = actions_theta[:,0] * max_speeds_int / pref_speed
            #   unique rows
            actions_theta = np.asarray(np.vstack({tuple(row) for row in actions_theta}))
            # print 'after---', actions_theta
            # print 'cur_pos, goal', agent_state[0:2], agent_state[6:8]
        # print 'after, max_speed', np.max(actions_theta[:,0])

        return actions_theta

    # state raw is in global frame
    # velocity output should also be in global frame
    def find_next_action(self, agent_state, other_agents_state, other_agents_action=None, static_constraints=None):
        # state_raw =  2x [pos.x, pos.y, vel.x, vel.y,prefVelocity.x, \
        # prefVelocity.y, goals[0].x, goals[0].y]
        # state_nn = [dist_to_goal, pref_speed, vx, vy, \
        # other_vx, other_vx, rel_pos_x, rel_pos_y]

        actions_theta = self.find_feasible_actions(agent_state, static_constraints=static_constraints)

        state_values = self.find_next_states_values(agent_state, \
            actions_theta, other_agents_state, other_agents_action)

        # print("actions_theta:", actions_theta)
        # print("state_values:", state_values)

        best_action_ind = np.argmax(state_values)
        best_action = actions_theta[best_action_ind]
        # print '------'
        # print 'best_action_ind', best_action_ind
        # print 'best_action', best_action

        # best_action_xy = np.array([best_action[0] * np.cos(best_action[1]), best_action[0] * np.sin(best_action[1])])
        # select_action_xy = np.array([actions_theta[1,0] * np.cos(actions_theta[1,1]), actions_theta[1,0] * np.sin(actions_theta[1,1])])
        # next_state_best_action_dist = np.linalg.norm(agent_state[6:8] - agent_state[0:2] - 1.0 * best_action_xy)
        # next_state_straight_line_dist = np.linalg.norm(agent_state[6:8] - agent_state[0:2] - 1.0 * select_action_xy)
        # bnd_best = 0.97 ** (next_state_best_action_dist / 0.2)
        # bnd_straight = 0.97 ** (next_state_straight_line_dist / 0.2)
        # actions_theta[:,1] -= np.arctan2(agent_state[3], agent_state[2])
        # print 'best action: ', best_action, '; value: ', state_values[best_action_ind], 'bnd: ', bnd_best
        # print 'straight line action: ', actions_theta[1,:], '; value: ', state_values[1], 'bnd: ', bnd_straight
        # print 'dist_2_goal', np.linalg.norm(agent_state[6:8] - agent_state[0:2])
        # print 'dist_2_other_agent', np.linalg.norm(agent_state[0:2] - other_agents_state[0][0:2]) \
        #           - agent_state[8] - other_agents_state[0][8]
        # actions_theta[:,1] += np.arctan2(agent_state[3], agent_state[2])
        # print 'cur_value: %f, best_next_value: %f, best action (r, theta) %f, %f' % \
        #       (self.current_value, state_values[best_action_ind], best_action[0], best_action[1])
        if self.if_collide_with_other_agents(agent_state, other_agents_state):
            self.current_value = COLLISION_COST
        else:
            self.current_value = state_values[best_action_ind]

        # print 'best_action', best_action
        # raw_input()
        # print 'action_chosen', best_action[0] * np.cos(best_action[1]), \
        #   best_action[0] * np.sin(best_action[1])
        return best_action

    def find_subgoal(self, agent_state,\
     other_agents_state, min_dist, other_agents_action=None, static_constraints=None):

        # Find subgoals that are far away and have good diffusion costs
        diff_map_goal = agent_state[6:8]
        angle_to_diff_map_goal = np.arctan2(diff_map_goal[1]-agent_state[1],diff_map_goal[0]-agent_state[0])
        lower_cost_to_go_inds = np.where((static_constraints[:,2] > 6.5) & \
                    (abs(static_constraints[:,1]-angle_to_diff_map_goal) < np.pi/8))[0]
        print('MIN_DIST:', min_dist)
        # if len(lower_cost_to_go_inds)>0:# and min_dist < 7.0:
        if False:
            lower_cost_to_go_actions = static_constraints[lower_cost_to_go_inds,:]
            x = lower_cost_to_go_actions[:,2] * np.cos(lower_cost_to_go_actions[:,1]) + agent_state[0]
            y = lower_cost_to_go_actions[:,2] * np.sin(lower_cost_to_go_actions[:,1]) + agent_state[1]
            lower_cost_to_go_subgoals = np.column_stack((x,y))
            # print 'static_constraints:', static_constraints
            # print 'lower cost2go inds:', lower_cost_to_go_inds
            print('lower cost2go actions:', lower_cost_to_go_actions)
            # print 'lower cost2go subgoals:', lower_cost_to_go_subgoals

            agent_states = np.zeros((len(lower_cost_to_go_inds),len(agent_state)))
            for i in range(len(lower_cost_to_go_inds)):
                state = copy.deepcopy(agent_state)
                state[6:8] = lower_cost_to_go_subgoals[i,0:2]
                # state[2] = agent_state[5] * np.cos(agent_state[4])
                # state[3] = agent_state[5] * np.sin(agent_state[4])
                agent_states[i] = state
            # print 'states:', agent_states
            # Query the NN about every potential subgoal and pick the best one
            state_values = self.find_states_values(agent_states, other_agents_state)
            best_subgoal_ind = np.argmax(state_values)
            # print 'best_subgoal ind', best_subgoal_ind
            best_subgoal = lower_cost_to_go_subgoals[best_subgoal_ind]

            # print 'states_values:', state_values
            agent_state[6:8] = best_subgoal
            print('Good Subgoal Found')
            # print 'Best subgoal:', best_subgoal
        else:
            # use subgoal directly from diffusion map
            lower_cost_to_go_subgoals = None
            best_subgoal = agent_state[6:8]
            print('No good subgoal found.')

        return best_subgoal, lower_cost_to_go_subgoals

    # def find_next_action_using_cost_to_go(self, agent_state,\
    #        other_agents_state, min_dist, other_agents_action=None, static_constraints=None):

    #   # Use that best subgoal to query every feasible action
    #   actions_theta = self.find_feasible_actions(agent_state, static_constraints=static_constraints)
    #   state_values = self.find_next_states_values(agent_state, \
    #       actions_theta, other_agents_state, other_agents_action)

    #   best_action_ind = np.argmax(state_values)
    #   best_action = actions_theta[best_action_ind]
    #   if state_values[best_action_ind] < 0.2:
    #       best_action[0] = 0.0
    #       best_action[1] = agent_state[4]
    #   print '------'
    #   # print 'best_action_ind', best_action_ind
    #   print 'best_action', best_action

    #   if self.if_collide_with_other_agents(agent_state, other_agents_state):
    #       self.current_value = COLLISION_COST
    #   else:
    #       self.current_value = state_values[best_action_ind]

    #   return best_action

    

    def if_collide_with_other_agents(self, agent_state, other_agents_state):
        num_other_agents = len(other_agents_state)
        for other_agent_state in other_agents_state:
            if self.if_pos_collide(agent_state[0:2], other_agent_state[0:2], \
                agent_state[8] + other_agent_state[8]):
                return True
        return False

    def if_terminal_state(self, agent_state, other_agents_state):
        # check collision
        if self.if_collide_with_other_agents(agent_state, other_agents_state):
            return COLLIDED
        # check if reached goal
        for other_agent_state in other_agents_state:
            if not ((( np.linalg.norm(agent_state[0:2] - agent_state[6:8])<DIST_2_GOAL_THRES) \
                and (np.linalg.norm(agent_state[0:2] - other_agent_state[0:2]) >  \
                agent_state[8] + other_agent_state[8] + GETTING_CLOSE_RANGE or \
                 np.linalg.norm(other_agent_state[0:2] - other_agent_state[6:8])<DIST_2_GOAL_THRES))):
                break
            return REACHED_GOAL
        return NON_TERMINAL

    # find a random action
    def find_rand_action(self, agent_state, other_agents_state, \
            other_agents_action=None, isBestAction=False, static_constraints=None):
        rand_float = np.random.rand()
        cur_state = agent_state[0:2]
        goal = agent_state[6:8]
        nom_speed = agent_state[5]
        # if rand_float < 0.3: # prefered speed toward goal
        #   action = self.computePrefVel(cur_state, goal, nom_speed)
        # elif rand_float < 0.6: # zero action
        #   action = np.zeros((2,))
        # else: # random action
        #   action = np.zeros((2,))
        #   action[0] = np.random.uniform(0, nom_speed)
        #   action[1] = np.random.uniform(0, 2 * np.pi) - np.pi
        actions_theta = self.find_feasible_actions(agent_state, static_constraints=static_constraints)

        state_values = self.find_next_states_values(agent_state, \
            actions_theta, other_agents_state, other_agents_action)

        if isBestAction: 
            max_inds = np.argpartition(state_values, -4)[-1:]
        else: 
            max_inds = np.argpartition(state_values, -4)[-5:-2]

        # ind = np.random.randint(len(actions_theta))
        ind = np.random.randint(len(max_inds))
        ind = max_inds[ind]
        action = actions_theta[ind,:]
        return action

    def update_state(self, state, action_theta, dt):
        speed = action_theta[0]
        angle_select = action_theta[1]
        next_state = copy.deepcopy(state)
        pref_speed = state[5]
        # try:
        #   assert(speed < pref_speed + 0.0001)
        # except:
        #   print 'state', state
        #   print 'speed', speed
        #   print 'pref_speed', pref_speed
        #   assert(0)
        # print angle_select
        next_state[0] += speed * np.cos(angle_select) * dt
        next_state[1] += speed * np.sin(angle_select) * dt
        # next_state[2] = 0.5 * next_state[2] + 0.5 * speed * np.cos(angle_select)
        # next_state[3] = 0.5 * next_state[3] + 0.5 * speed * np.sin(angle_select)
        next_state[2] = speed * np.cos(angle_select)
        next_state[3] = speed * np.sin(angle_select)
        next_state[5] = pref_speed
        # next_state[9] = min(3.0, next_state[9] + dt)
        # turning direction and previous osscilation magnitude
        angle_diff = find_angle_diff(action_theta[1], state[4])
        # next_state[9] = angle_diff
        if abs(next_state[9]) < EPS:
            next_state[9] = 0.11 * np.sign(angle_diff)
        elif next_state[9] * angle_diff < 0:
            next_state[9] = max(-np.pi, min(np.pi, -next_state[9] + angle_diff))
        else:
            next_state[9] = np.sign(next_state[9]) * max(0.0, abs(next_state[9])-0.1)
        # print 'state[9]', state[9]
        # print 'angle_diff', angle_diff
        # print 'next_state[9]', next_state[9]
        # raw_input() 

        if self.mode == 'no_constr':
            next_state[4] = angle_select
        elif self.mode == 'rotate_constr':
            min_turning_radius = 0.5
            limit = pref_speed / min_turning_radius * dt  
            angle_diff = find_angle_diff(action_theta[1], state[4])
            delta_heading_lb = - limit
            delta_heading_ub = limit
            next_state[4] = state[4] + np.clip(angle_diff, \
                delta_heading_lb, delta_heading_ub)
            next_state[4] = (next_state[4] + np.pi) % (2 * np.pi) - np.pi
        else:
            assert(0)

        # if np.linalg.norm(next_state[0:2] - next_state[6:8]) == 0:
            # next_state[5] /= (np.linalg.norm(next_state[4:6]+EPS) * 100.0)
        # try:
        #   assert(next_state[4] > -np.pi-EPS and next_state[4] < np.pi+EPS)
        # except AssertionError:
        #   print action_theta
        #   assert(0)

        return next_state

    def update_states(self, state, actions_theta, dt):
        speeds = actions_theta[:,0]
        angles_select = actions_theta[:,1]
        num_actions = actions_theta.shape[0]
        next_states = np.matlib.repmat(state, num_actions, 1)
        pref_speed = state[5]
        # try:
        #   assert(np.all(actions_theta[:,0] < pref_speed + 0.0001))
        # except:
        #   print 'pref_speed', pref_speed
        #   print 'actions_theta[:,0]', actions_theta[:,0]
        #   assert(0)
        
        # print angle_select
        next_states[:,0] += speeds * np.cos(angles_select) * dt
        next_states[:,1] += speeds * np.sin(angles_select) * dt
        # next_states[:,2] = 0.5 * next_states[:,2] + 0.5 * speeds * np.cos(angles_select)
        # next_states[:,3] = 0.5 * next_states[:,3] + 0.5 * speeds * np.sin(angles_select)
        next_states[:,2] = speeds * np.cos(angles_select)
        next_states[:,3] = speeds * np.sin(angles_select)
        next_states[:,5] = pref_speed
        # next_states[:,9] = np.clip(next_states[:,9] + dt, 0.0, 3.0)
        # angles_diff = find_angle_diff(actions_theta[:,1], state[4])
        # next_states[:,9] = angles_diff

        angles_diff = find_angle_diff(actions_theta[:,1], state[4])
        # next_states[:,9] = angles_diff
        zero_inds = np.where(abs(next_states[:,9]) < EPS)[0]
        oscillation_inds = np.where(next_states[:,9] * angles_diff < 0)[0]
        oscillation_inds = np.setdiff1d(oscillation_inds, zero_inds)

        same_dir_inds = np.where(next_states[:,9] * angles_diff > -EPS)[0]
        same_dir_inds = np.setdiff1d(same_dir_inds, np.union1d(zero_inds, oscillation_inds))

        
        next_states[zero_inds,9] = 0.11 * np.sign(angles_diff[zero_inds])
        next_states[oscillation_inds,9] = np.clip(-next_states[oscillation_inds,9] \
            + angles_diff[oscillation_inds], -np.pi, np.pi)
        next_states[same_dir_inds,9] = np.sign(next_states[same_dir_inds,9]) * \
            np.clip(abs(next_states[same_dir_inds,9]) - 0.1, 0.0, np.pi)

        # print 'states[:,9], angles_diff, next_state[:,9]', 
        # print np.vstack((np.matlib.repmat(state[9], 1, len(actions_theta)),\
        #       angles_diff, next_states[:,9])).transpose()
        # raw_input() 

        if self.mode == 'no_constr':
            next_states[:,4] = angles_select
        elif self.mode == 'rotate_constr':
            min_turning_radius = 0.5
            limit = pref_speed / min_turning_radius * dt  
            angles_diff = find_angle_diff(actions_theta[:,1], state[4])
            # assert(len(angles_diff) == actions_theta.shape[0])
            max_angle_diff = np.amax(abs(angles_diff))
            # try:
            #   assert(max_angle_diff < limit/dt + 0.01)
            # except:
            #   print '---\n state[4]', state[4]
            #   print 'actions_theta', actions_theta
            #   print 'angles_diff', angles_diff
            #   print 'max_angle_diff', max_angle_diff
            #   print 'pref_speed, dt, limit', pref_speed, dt, limit/dt
            #   assert(0)
            delta_heading_lb = -limit
            delta_heading_ub = limit
            next_states[:,4] = state[4] + np.clip(angles_diff, \
                delta_heading_lb, delta_heading_ub)
            next_states[:,4] = (next_states[:,4] + np.pi) % (2 * np.pi) - np.pi
        else:
            assert(0)

        # compute pref_vecs
        # avoid divide by zero problem 
        # next_states[:,4:6] /= (np.linalg.norm(next_states[:,4:6], axis=1)[:,np.newaxis] * 100.0)
        # tmp_vec = next_states[:,6:8] - next_states[:,0:2]
        # pref_speeds = np.linalg.norm(tmp_vec, axis=1)[:,np.newaxis]
        # valid_inds = np.where(pref_speeds > 0)[0]
        # pref_vecs = tmp_vec[valid_inds] / pref_speeds[valid_inds] * pref_speed
        # next_states[valid_inds,4] = pref_vecs[:,0]
        # next_states[valid_inds,5] = pref_vecs[:,1]
        # try:
        #   assert(np.all(next_states[:,4] > -np.pi-EPS) and np.all(next_states[:,4] < np.pi+EPS))
        # except AssertionError:
        #   print actions_theta
        #   assert(0)

        return next_states

    # check collision 
    def if_action_collide(self, agent_state, agent_action, other_agent_state, \
                            other_agent_action, delta_t):
        # sampling version
        # t = 0.0
        # incr = np.min((delta_t, 0.25))
        # radius = agent_state[8] + other_agent_state[8] + self.radius_buffer
        # agent_v = np.zeros((2,))
        # agent_v[0] = agent_action[0] * np.cos(agent_action[1])
        # agent_v[1] = agent_action[0] * np.sin(agent_action[1])
        # other_agent_v = other_agent_state[2:4]
        # T = np.linspace(0, delta_t, num=np.ceil((delta_t+0.0001) / incr))
        # for t in T:
        #   agent_pos = agent_state[0:2] + agent_v * t
        #   other_agent_pos = other_agent_state[0:2] + other_agent_v * t
        #   if self.if_pos_collide(agent_pos, other_agent_pos, radius):
        #       return True
        # return False

        # analytical version
        agent_v = np.zeros((2,))
        agent_v[0] = agent_action[0] * np.cos(agent_action[1])
        agent_v[1] = agent_action[0] * np.sin(agent_action[1])
        other_agent_v = np.zeros((2,))
        other_agent_v[0] = other_agent_action[0] * np.cos(other_agent_action[1])
        other_agent_v[1] = other_agent_action[0] * np.sin(other_agent_action[1])
        # other_agent_v[0] = other_agent_action[0] * np.cos(other_agent_action[1] + np.random.randn()*np.pi/36)
        # other_agent_v[1] = other_agent_action[0] * np.sin(other_agent_action[1] + np.random.randn()*np.pi/36)

        # modifying distance calculation
        rel_pos_angle = np.arctan2(other_agent_state[1] - agent_state[1], \
                        other_agent_state[0] - agent_state[0])
        agent_speed_angle = np.arctan2(agent_v[1], agent_v[0])
        angle_diff = find_angle_diff(rel_pos_angle, agent_speed_angle)
        # other agent in front, zero out other agent's speed in the same direction
        if abs(angle_diff) < np.pi:
            dot_product = np.sum(agent_v * other_agent_v)
            if dot_product > EPS:
                other_agent_v = other_agent_v - dot_prouct / np.linalg.norm(agent_v) * agent_v 

        # blind spot
        # elif abs(angle_diff) > 3.0 * np.pi/4.0:


        radius = agent_state[8] + other_agent_state[8] + self.radius_buffer
        x1 = agent_state[0:2]
        x2 = agent_state[0:2] + delta_t * agent_v
        y1 = other_agent_state[0:2]
        y2 = other_agent_state[0:2] + delta_t * other_agent_v
        # return if_segs_collide(x1, x2, y1, y2, radius)
        min_dist = ind_dist_between_segs(x1, x2, y1, y2)
        # min_dist = min_dist - 0.03 * np.random.random()
        cur_dist = np.linalg.norm(x1-y1)
        if cur_dist < radius:
            if_collide = True
        else:
            if_collide = min_dists < radius
        # dist_future = x2 - (0.75*y1 + 0.25*y2)
        # dist_future = np.linalg.norm(dist_future)
        # min_dist = min(dist_future, min_dist)
        min_dist = min_dist - radius
        return min_dist, min_dist < 0

    # check collision 
    def if_actions_collide(self, agent_state, agent_actions, other_agent_state, \
                            other_agent_action, delta_t):
        # bnd
        agent_pref_speed = agent_state[5]
        other_agent_speed = other_agent_action[0]
        radius = agent_state[8] + other_agent_state[8] + self.radius_buffer
        num_actions = agent_actions.shape[0]
        if_collide = np.zeros((num_actions), dtype=bool)
        min_dists = (radius + GETTING_CLOSE_RANGE + EPS) * np.ones((num_actions), dtype=bool)
        # two agents are too far away for collision
        if np.linalg.norm(agent_state[0:2] - other_agent_state[0:2]) > \
            (agent_pref_speed+other_agent_speed) * delta_t + radius:
            return min_dists, if_collide # default to no collision

        # check for each action individually 
        agent_vels = np.zeros((num_actions,2))
        agent_vels[:,0] = agent_actions[:,0] * np.cos(agent_actions[:,1])
        agent_vels[:,1] = agent_actions[:,0] * np.sin(agent_actions[:,1])
        other_agent_v = np.zeros((2,))
        other_agent_v[0] = other_agent_action[0] * np.cos(other_agent_action[1])
        other_agent_v[1] = other_agent_action[0] * np.sin(other_agent_action[1])
        # other_agent_v[0] = other_agent_action[0] * np.cos(other_agent_action[1] + np.random.randn()*np.pi/36)
        # other_agent_v[1] = other_agent_action[0] * np.sin(other_agent_action[1] + np.random.randn()*np.pi/36)
        other_agent_vels = np.matlib.repmat(other_agent_v, num_actions, 1)

        # modifying distance calculation
        p_oa_angle = np.arctan2(other_agent_state[1] - agent_state[1], \
                        other_agent_state[0] - agent_state[0])
        agent_speed_angles = np.arctan2(agent_vels[:,1], agent_vels[:,0])
        other_speed_angle = np.arctan2(other_agent_v[1], other_agent_v[0])
        heading_diff = find_angle_diff(agent_speed_angles, other_speed_angle)
        agent_heading_2_other = find_angle_diff(agent_speed_angles, p_oa_angle)
        r = agent_state[8] + other_agent_state[8] + GETTING_CLOSE_RANGE
        coll_angle = abs(np.arcsin(min(0.95, r / np.linalg.norm(\
                        agent_state[0:2]-other_agent_state[0:2]))))
        
        # other agent in front, zero out other agent's speed in the same direction
        front_inds = np.where( (abs(agent_heading_2_other)<coll_angle) & \
            (abs(heading_diff) < np.pi/2.0))[0]
        if len(front_inds)>0:
            # print other_agent_vels[front_inds,:].shape
            # print other_agent_vels[front_inds,:].shape
            # print np.matlib.repmat(np.linalg.norm(agent_vels[front_inds,:], axis=1),2,1).transpose().shape
            # print agent_vels[front_inds,:].shape
            dot_product = np.sum(agent_vels * other_agent_vels, axis=1)
            valid_inds = np.where(agent_vels[:,0]>EPS)[0]
            dot_product[valid_inds] /= np.linalg.norm(agent_vels[valid_inds,:], axis=1)
            other_agent_vels[front_inds,:] = other_agent_vels[front_inds,:] -  \
                np.matlib.repmat(dot_product[front_inds], 2, 1).transpose()\
                * agent_vels[front_inds,:] / 2.0

        # analytical version
        x1 = agent_state[0:2]                      # shape = (2,)
        x2 = x1 + min(1.0, delta_t) * agent_vels             # shape = (num_actions, 2)
        y1 = other_agent_state[0:2]                # shape = (2,)
        y2 = y1 + min(1.0, delta_t) * other_agent_vels     # shape = (num_actions, 2)
        # end points
        # if np.linalg.norm(x1-y1) < radius:
        #   return np.ones((num_actions,), dtype=bool)
        # if_collide = np.linalg.norm(x2-y2, axis=1) < radius

        # # critical points (where d/dt = 0)
        # z_bar = (x2 - x1) - (y2 - y1)             # shape = (num_actions, 2)
        # inds = np.where((np.linalg.norm(z_bar,axis=1)>0) & (if_collide == False))[0]
        # t_bar = - np.sum((x1-y1) * z_bar[inds,:], axis=1) \
        #       / np.sum(z_bar[inds,:] * z_bar[inds,:], axis=1)
        # t_bar_rep = np.matlib.repmat(t_bar, 2, 1).transpose()
        # dist_bar = np.linalg.norm(x1 + (x2[inds,:]-x1) * t_bar_rep \
        #         - y1 - (y2-y1) * t_bar_rep, axis=1)
        # if_collide[inds] = np.logical_and(np.logical_and(dist_bar < radius,  t_bar > 0), t_bar < 1.0) 

        min_dists = gen_tc.find_dist_between_segs(x1, x2, y1, y2)

        # min_dists = min_dists - 0.03 * np.random.random(min_dists.shape)

        # if other agent in back, and will run into the agent
        # p_ao_angle = np.arctan2(agent_state[1] - other_agent_state[1], \
        #   agent_state[0] - other_agent_state[0])
        # other_heading_2_agent = find_angle_diff(other_speed_angle, p_ao_angle)
        # back_inds = np.where( (abs(other_heading_2_agent)<coll_angle) & \
        #   (abs(heading_diff) < np.pi/2.0))[0]
        # back_inds = np.setdiff1d(back_inds, front_inds)
        # min_dists[back_inds] += GETTING_CLOSE_RANGE



        # print find_dist_between_segs(x1, x2, y1, y2)
        # print if_collide_2
        # print radius
        # print if_collide
        # print if_collide - if_collide_2
        # assert(np.all(if_collide == if_collide_2))
        cur_dist = np.linalg.norm(x1-y1)
        if cur_dist < radius:
            if_collide[:] = True
        else:
            if_collide = min_dists < radius
        # dist_future = x2 - (0.75 * y1 + 0.25 * y2)
        # dist_future = np.linalg.norm(dist_future, axis=1)
        # min_dists = np.minimum(dist_future, min_dists)
        min_dists = min_dists - radius
        return min_dists, if_collide

        # sampling vector version
        # incr = 0.25 #np.min((0.25, delta_t))
        # T = np.linspace(0, delta_t, num=np.ceil((delta_t+0.0001) / incr))
        # for t in T:
        #   dist_vec = agent_vels * t + np.matlib.repmat(agent_state[0:2] - \
        #               (other_agent_state[0:2] + other_agent_v * t), num_actions, 1)
        #   if_collide_pt = np.linalg.norm(dist_vec, axis=1) < radius 
        #   if_collide = np.logical_or(if_collide, if_collide_pt)
        
        # sampling loop version
        # for i in range(num_actions):
        #   t = 0.0
        #   while t <= delta_t:
        #       agent_pos = agent_state[0:2] + agent_vels[i,:] * t
        #       other_agent_pos = other_agent_state[0:2] + other_agent_v * t
        #       if self.if_pos_collide(agent_pos, other_agent_pos, radius):
        #           if_collide[i] = True
        #           break
        #       t += np.min((incr, delta_t))
        # return if_collide

    # check collision between 2 pairs of initial and end states
    # def if_state_pairs_collide(self, agent_state, agent_next_state, \
    #                           other_agent_state, other_agent_next_state):
    #   radius = agent_state[8] + other_agent_state[8] + self.radius_buffer
    #   num_incr = 9
    #   agent1_incr = (agent_next_state[0:2] - agent_state[0:2]) / float(num_incr)
    #   agent2_incr = (other_agent_next_state[0:2] - other_agent_state[0:2]) / float(num_incr)
    #   for i in range(num_incr + 1):
    #       agent_pos = agent_state[0:2] + agent1_incr * i
    #       other_agent_pos = other_agent_state[0:2] + agent2_incr * i
    #       if self.if_pos_collide(agent_pos, other_agent_pos, radius):
    #           return True
    #   return False

    def if_pos_collide(self, agent1_pos, agent2_pos, radius):
        if np.linalg.norm(agent1_pos - agent2_pos) < radius:
            return True
        return False

    # query state values from the neural network
    # don't consider terminal state 
    # (expect the neural network to have learned values of the terminal states)
    def find_states_values(self, agent_states, other_agents_state):
        # print 'agent_states in find_states_values', agent_states
        # print 'other_agent_state', other_agent_state
        if agent_states.ndim == 1:
            ref_prll, ref_orth, state_nn = \
                pedData.rawState_2_agentCentricState( \
                agent_states, other_agents_state, self.num_agents)

            # print("state_nn:", state_nn)
            # try upper bounding with max
            value = np.squeeze(self.nn.make_prediction_raw(state_nn).clip(min=-0.25, max=1.0))
            # print 'state_nn', state_nn
            # print('before, value', value)
            upper_bnd = GAMMA ** (state_nn[0] / DT_NORMAL)
            value = min(upper_bnd, value)
            # print('after, value', value)

            return value
        else:
            num_states = agent_states.shape[0]
            # states_nn = np.zeros((num_states, 14))
            # for i in range(num_states):
            #   ref_prll, ref_orth, states_nn[i,:] = \
            #       pedData.rawState_2_agentCentricState( \
            #       agent_states[i,:], other_agent_state)
            ref_prll_vec, ref_orth_vec, states_nn = \
            pedData.rawStates_2_agentCentricStates(\
                agent_states, other_agents_state, self.num_agents)

            # print 'states_nn[0,:]', states_nn[0,:]
            values = np.squeeze(self.nn.make_prediction_raw(states_nn).clip(min=-0.25, max=1.0))
            # print 'states_nn', states_nn
            # print 'before, values', values
            upper_bnd = GAMMA ** (states_nn[:,0] / DT_NORMAL)
            # print 'upper_bnd', upper_bnd
            # print 'values', values
            values = np.minimum(upper_bnd, values)
            # print 'values[-1]', values
            # raw_input()

            return values


    def find_agent_next_state(self, agent_state, other_agents_state, \
            other_agents_action, dt):
        action_theta = self.find_next_action(agent_state, \
            other_agents_state, other_agents_action)
        next_state = self.update_state(agent_state, action_theta, dt)
        return next_state

    def find_agent_next_rand_state(self, agent_state, other_agents_state, random_action_theta, dt):
        # action_theta = self.find_rand_action(agent_state, other_agent_state)
        next_state = self.update_state(agent_state, random_action_theta, dt)
        return next_state



    def testcase_2_agentStates(self, test_case, ifRandSpeed=False, ifRandHeading=False):
        assert(test_case[0, 5] > 0.1 or ifRandSpeed==True)

        num_agents = len(test_case)
        agents_states = []

        for i in range(num_agents):
            agent_state = np.zeros((10,)); 
            agent_state[0:2] = test_case[i,0:2]; agent_state[6:8] = test_case[i,2:4];
            agent_state[9] = 0.0
            if ifRandSpeed == True:
                agent_state[5] = np.random.rand() * (1.5-0.1) + 0.1
                agent_state[8] = 0.3 + np.random.uniform(0,0.2)
            else:
                agent_state[5] = test_case[i,4]
                agent_state[8] = test_case[i,5]

            # assume originally facing the goal direction
            if ifRandHeading == True:
                agent_state[4] = np.random.rand() * 2 * np.pi - np.pi
            else:
                agent_state[4] = np.arctan2(test_case[i,3]-test_case[i,1], \
                        test_case[i,2]-test_case[i,0])

            # if agent is not stationary, assume it's travelling along its initially heading
            if np.linalg.norm(agent_state[0:2] - agent_state[6:8]) > EPS:
                heading_dir = np.array([np.cos(agent_state[4]), np.sin(agent_state[4])])
                # agent_state[2:4] = agent_state[5] * heading_dir

            try:
                assert(agent_state[5] > 0.05)
            except:
                print(agent_state)
                print(test_case)
                assert(0)
            agents_states.append(agent_state)

        return agents_states

    # generate a trajectory using the current value neural network
    # test_case: [ [start_x, start_y, end_x, end_y, desired_speed, radius],
    #              [start_x, start_y, end_x, end_y, desired_speed, radius] ]
    # rl_epsilon: explore a new action with rl_epsilon probability
    # ifRandSpeed: whether generate  random [desired_speed and radius] for test case
    # figure_name: name of figure, "no_plot" for no figure
    def generate_traj(self, test_case, rl_epsilon=0.0, \
            ifRandSpeed=False, figure_name=None, stopOnCollision=True, ifRandHeading=False, ifNonCoop=False):
        time_start = time.time()
        num_agents = test_case.shape[0]
        time_vec = []
        dt_vec = []
        traj_raw_multi = []
        agent_num_thres = self.num_agents / 4.0
        if_non_coop_vec = np.zeros((num_agents,), dtype=bool)
        if ifNonCoop:
            if_non_coop_vec = np.random.randint(2, size=(num_agents,))

        for i in range(num_agents):
            traj_raw_multi.append([])
        time_to_complete = np.zeros((num_agents,))
        agents_dt = np.ones((num_agents,))
        if_completed_vec = np.zeros((num_agents,), dtype=bool)
        time_past_one_ind = 0 # ind of time one second ago in time_vec
        filtered_actions_theta = [None] * num_agents
        filtered_actions_theta_close = [None] * num_agents
        filtered_actions_theta_far = [None] * num_agents
        dist_mat = np.zeros((num_agents, num_agents))

        # initialize
        agents_states = self.testcase_2_agentStates(test_case, ifRandSpeed, ifRandHeading)
        next_agents_states = copy.deepcopy(agents_states)
        agents_speeds = np.zeros((num_agents,))
        max_t_vec = np.zeros((num_agents,))
        for i in range(num_agents):
            agents_speeds[i] = agents_states[i][5]
            max_t_vec[i] = 3.0 * (np.linalg.norm(test_case[i, 2:4] - test_case[i, 0:2]) / agents_speeds[i])
        max_t = np.amax(max_t_vec)

        max_t = 0.2 ###################### delete later


        dt_nominal = 0.1
        t = 0

        # for epsilon random action
        rand_action_time_thres = 0.5
        if_agents_rand_action = np.zeros((num_agents,), dtype=bool)
        agents_rand_action_duration = np.zeros((num_agents, ))
        agents_rand_action_theta = np.zeros((num_agents,2))

        while True:
            # append to stats
            time_vec.append(t)
            for i in range(num_agents):
                traj_raw_multi[i].append(agents_states[i].copy())

            # compute if completed and dt
            # dt_nominal = 0.1
            for i in range(num_agents):
                dist_2_goal = np.linalg.norm(agents_states[i][0:2] - agents_states[i][6:8])
                if_completed_vec[i] = dist_2_goal < DIST_2_GOAL_THRES
                agents_dt[i] = dist_2_goal/agents_speeds[i]
                if if_completed_vec[i]:
                    agents_states[i][2:4] = 0
                    agents_dt[i] = 1
            dt = min(dt_nominal,np.amin(agents_dt))
            t += dt
            dt_vec.append(dt)
            
            # collision (TODO)
            for i in range(num_agents):
                for j in range(i):
                    dist_mat[i,j] = np.linalg.norm(agents_states[i][0:2] - agents_states[j][0:2]) - \
                                    agents_states[i][8] - agents_states[j][8]
                    dist_mat[j,i] = dist_mat[i,j]
            min_sepDist = np.min(dist_mat)
            if stopOnCollision and min_sepDist < 0:
                break

            # time too long, probably diverged
            if t > max_t:
                break 

            # completed
            if np.all(if_completed_vec):
                break

            # idling, not reaching goal (TODO)

            # filter velocities
            # while time_vec[-1] - time_vec[time_past_one_ind] > 0.45:
            while t - time_vec[time_past_one_ind] > 0.45:
                time_past_one_ind += 1
            dt_past_vec = np.asarray(dt_vec[time_past_one_ind:])

            for i in range(num_agents):
                if len(time_vec) != 0:
                    # if np.random.rand()>0.5 or filtered_actions_theta_close[i] is None:
                    past_vel = np.asarray(traj_raw_multi[i][time_past_one_ind:])[:,2:5]
                    # filtered_actions_theta[i] = nn_nav.filter_vel(dt_past_vec, past_vel)
                    filtered_actions_theta_close[i] = filter_vel(dt_past_vec, past_vel,ifClose=True)
                    filtered_actions_theta_far[i] = filter_vel(dt_past_vec, past_vel,ifClose=False)

            # propagate forward in time
            rl_random = np.random.rand()
            for i in range(num_agents):
                agent_state = agents_states[i]
                if not if_completed_vec[i]:
                    other_agents_states = [x for j, x in enumerate(agents_states) if j!=i]
                    # other_agents_actions = [x for j, x in enumerate(filtered_actions_theta) if i!=j]
                    other_agents_actions = []
                    for j in range(num_agents):
                        if j != i:
                            if dist_mat[i,j] < 0.5:
                                other_agents_actions.append(filtered_actions_theta_close[j])
                            else:
                                other_agents_actions.append(filtered_actions_theta_far[j])

                    # non-cooperative (for plotting)
                    if i > 0 and if_non_coop_vec[i] == True:
                        desired_velocity = self.computePrefVel(agent_state[0:2], \
                                    agent_state[6:8], agent_state[5])
                        next_agents_states[i] = agent_state.copy()
                        next_agents_states[i][2:4] = desired_velocity
                        next_agents_states[i][0:2] += dt * agent_state[2:4]
                        next_agents_states[i][4] = np.arctan2(desired_velocity[1], desired_velocity[0])

                    else:
                        # cooperative
                        if if_agents_rand_action[i] == False and rl_random < rl_epsilon \
                            and np.linalg.norm(agent_state[0:2] - agent_state[6:8]) > 1.0 and i < agent_num_thres: # only first agent freeze
                            if_agents_rand_action[i] = True
                            agents_rand_action_duration[i] = 0.0
                            if i < agent_num_thres:
                                agents_rand_action_theta[i] = self.find_rand_action(agent_state, other_agents_states, \
                                other_agents_action = other_agents_actions, isBestAction=False)
                            else: 
                                agents_rand_action_theta[i] = self.find_rand_action(agent_state, other_agents_states, \
                                other_agents_action = other_agents_actions, isBestAction=True)
                        elif agents_rand_action_duration[i] > rand_action_time_thres:
                            if_agents_rand_action[i] = False

                        if if_agents_rand_action[i] == True:
                            next_agents_states[i] = self.find_agent_next_rand_state(\
                                agent_state, other_agents_states, agents_rand_action_theta[i], dt)
                            agents_rand_action_duration[i] += dt
                        else:
                            next_agents_states[i] = self.find_agent_next_state(\
                                agent_state, other_agents_states, other_agents_actions, dt)

                    time_to_complete[i] = t
                else:
                    next_agents_states[i] = agent_state.copy()

            agents_states = copy.deepcopy(next_agents_states)

        # add time to trajectory list
        traj_raw_multi = [time_vec] + traj_raw_multi
        for i, vec in enumerate(traj_raw_multi):
            traj_raw_multi[i] = np.asarray(vec)


        # print 'traj_raw', traj_raw
        if figure_name != "no_plot":
            title_string = 'total time: %f' % (np.sum(time_to_complete))
            print('time_to_complete', time_to_complete)
            pedData.plot_traj_raw_multi(traj_raw_multi, title_string, figure_name)
            print('finished generating trajectory with %d points in %fs' % \
                (len(traj_raw_multi[0]), time.time()-time_start))
        return traj_raw_multi, time_to_complete

    # find preferred velocity vector (toward goal at desired(nominal) speed)
    def computePrefVel(self, cur_state, goal, nom_speed):
        pref_vec = goal - cur_state
        # print 'goal, cur_state', goal, cur_state
        # print 'nom_speed', nom_speed
        # print 'pref_vec before normalizing', pref_vec
        pref_speed = np.linalg.norm(pref_vec)
        if pref_speed > 0: 
            pref_vec = pref_vec / pref_speed * nom_speed
        else:
            pref_vec = np.array([EPS, 0])
        # print 'pref_vec', pref_vec
        return pref_vec

    # compute agent_centric_states
    def find_bad_inds(self, agent_centric_states):
        # closest other agent is the first agent
        # dist_2_others (see README.txt)
        agent_vel = agent_centric_states[:,4:6]
        agent_speed = np.linalg.norm(agent_vel, axis=1)
        agent_vx = agent_vel[:,0]
        agent_vy = agent_vel[:,1]
        agent_heading = agent_centric_states[:,3]

        dist_2_goal = agent_centric_states[:,0]
        other_pos = agent_centric_states[:,9:11]
        other_px = other_pos[:,0]
        other_py = other_pos[:,1]

        other_vel = agent_centric_states[:,7:9]
        other_vx = other_vel[:,0]
        other_vy = other_vel[:,1]
        other_speed = np.linalg.norm(other_vel, axis=1)
        other_heading = np.arctan2(other_vy, other_vx)

        rel_vel = agent_vel - other_vel
        rel_vel_angle = np.arctan2(rel_vel[:,1], rel_vel[:,0])
        rel_pos_angle = np.arctan2(0-other_py, 0-other_px)
        rot_angle = find_angle_diff(rel_vel_angle, rel_pos_angle)

        bad_inds_oppo = []
        bad_inds_same = []
        bad_inds_tangent = []
        if self.passing_side == 'right':
            # opposite  direction
            # same direction
            # faster
            bad_inds_same_fast = np.where( (dist_2_goal > 1) & (other_speed > EPS) & \
                (agent_speed > EPS) & \
                (agent_speed > other_speed + 0.1) & \
                (other_py > -0.5) & (other_py < 2) & \
                (other_px > 0) & (other_px < 3) & \
                (agent_heading < 0) & \
                (abs(other_heading) < np.pi/6.0))[0]
            # slower
            bad_inds_same_slow = np.where( (dist_2_goal > 1) & (other_speed > EPS) & \
                (agent_speed > EPS) & \
                (agent_speed < other_speed - 0.1) & \
                (other_py < 0) & (other_py > -2) & \
                (other_px < 0) & (other_px > -3) & \
                (agent_heading > 0) & \
                (abs(other_heading) < np.pi/6.0))[0]

            bad_inds_same = np.union1d(bad_inds_same_fast, bad_inds_same_slow)

            # opposite direction
            bad_inds_oppo = np.where( (dist_2_goal > 1) & (other_speed > EPS) & \
                (agent_speed > EPS) & \
                (other_py < 0) & (other_py > -2) & \
                (other_px > 0) & (other_px < 5) & \
                (agent_heading > EPS) & \
                (other_heading < -5.0*np.pi/6.0))[0]

            # tangent direction
            agent_speed = agent_centric_states[0,1]
            other_rel_dist = np.linalg.norm(other_pos, axis=1)
            bad_inds_tangent = np.where( (dist_2_goal > 1) & (other_speed > EPS) & \
                (agent_speed > EPS) & \
                # (other_py < 3 * agent_speed) & (other_py > -3 * agent_speed) & \
                # (other_px > 0 * agent_speed) & (other_px < 3 * agent_speed) & \
                (other_px > 0) & (other_rel_dist < 3) & \
                (rot_angle < 0) & \
                (abs(other_heading) > np.pi/4.0) & \
                (agent_speed > other_speed - 0.2)) [0]
                # & (abs(other_heading) < 5.0* np.pi/6.0) )[0]


        elif self.passing_side == 'left':
            # opposite  direction
            # same direction
            # faster
            bad_inds_same_fast = np.where( (dist_2_goal > 1) & (other_speed > EPS) & \
                (agent_speed > EPS) & \
                (agent_speed > other_speed + 0.1) & \
                (other_py > -2) & (other_py < 0.5) & \
                (other_px > 0) & (other_px < 3) & \
                (agent_heading > 0) & \
                (abs(other_heading) < np.pi/6.0))[0]
            # slower
            bad_inds_same_slow = np.where( (dist_2_goal > 1) & (other_speed > EPS) & \
                (agent_speed > EPS) & \
                (agent_speed < other_speed - 0.1) & \
                (other_py < 2) & (other_py > 0) & \
                (other_px < 0) & (other_px > -3) & \
                (agent_heading > 0) & \
                (abs(other_heading) < np.pi/6.0))[0]

            bad_inds_same = np.union1d(bad_inds_same_fast, bad_inds_same_slow)

            # opposite direction
            bad_inds_oppo = np.where( (dist_2_goal > 1) & (other_speed > EPS) & \
                (agent_speed > EPS) & \
                (other_py < 2) & (other_py > 0) & \
                (other_px > 0) & (other_px < 5) & \
                (agent_heading < EPS) & \
                (other_heading > 5.0*np.pi/6.0))[0]

            # tangent direction
            agent_speed = agent_centric_states[0,1]
            other_rel_dist = np.linalg.norm(other_pos, axis=1)
            bad_inds_tangent = np.where( (dist_2_goal > 1) & (other_speed > EPS) & \
                (agent_speed > EPS) & \
                # (other_py < 3* agent_speed) & (other_py > -3* agent_speed) & \
                # (other_px > 0* agent_speed) & (other_px < 3* agent_speed) & \
                (other_px > 0) & (other_rel_dist < 3) & \
                (rot_angle > 0) & \
                (abs(other_heading) > np.pi/4.0) & \
                (agent_speed > other_speed - 0.2))[0]
                # (abs(other_heading) < 5.0* np.pi/6.0) )[0]

        # return    [], bad_inds_same, []
        return  bad_inds_oppo, bad_inds_same, bad_inds_tangent

# speed filter
# input: past velocity in (x,y) at time_past 
# output: filtered velocity in (speed, theta)
def filter_vel(dt_vec, agent_past_vel, ifClose=False):
    # if np.sum(dt_vec) == 0:
    #   print 'dt_vec', dt_vec
    #   print 'agent_past_vel_xy', agent_past_vel_xy
    #   assert(0)

    # check if there is oscillation
    # num_pts = agent_past_vel.shape[0]
    # if_oscillate = False
    # if num_pts > 1:
    #   angle_diff_vec = find_angle_diff( agent_past_vel[1:,2], \
    #       agent_past_vel[0:-1,2] )
    #   if np.all(angle_diff_vec>-EPS) or np.all(angle_diff_vec<EPS):
    #       speed = np.linalg.norm(agent_past_vel[-1,0:2])
    #       angle = agent_past_vel[-1,2]
    #       return np.array([speed, angle])

    if True: #ifClose == False:
        agent_past_vel_xy = agent_past_vel[-2:,0:2]
        agent_heading = agent_past_vel[-2:,2]
        dt_vec = dt_vec[-2:]
    else:
        length = min(2, len(dt_vec)-1)
        # length = len(dt_vec)-1
        agent_past_vel_xy = agent_past_vel[-length:,0:2]
        agent_heading = agent_past_vel[-length:,2]
        dt_vec = dt_vec[-length:]
    # print 'len(dt_vec)', len(dt_vec)

    average_x = np.sum(dt_vec * agent_past_vel_xy[:,0]) / np.sum(dt_vec)
    average_y = np.sum(dt_vec * agent_past_vel_xy[:,1]) / np.sum(dt_vec)
    speeds = np.linalg.norm(agent_past_vel_xy, axis=1)
    # speed = np.mean(speeds)
    # speed = speeds[-1]
    # speed = np.linalg.norm(agent_past_vel_xy[-1,:])
    # assert(speed == speed1)
    speed = np.linalg.norm(np.array([average_x, average_y]))
    angle = np.arctan2(average_y, average_x)

    return np.array([speed, angle])


# load NN_navigation
def load_NN_navigation_value(file_dir, num_agents, mode, passing_side, filename=None, ifPrint=True):
    ''' initializing neural network ''' 
    try:
        # Messed up from transfer into this directory structure. I don't plan to ever change these settings.
        # nn_training_param_ = pickle.load(open(file_dir+"/../../pickle_files/multi/nn_training_param.p", "rb"))
        d = {'sgd_step_size': 0.5, 'sgd_step_c': 0.1, 'reg_lambda': 0.001, 'w_scale': 0.1, 'sgd_step_epsilon': 0.1, 'nb_iter': 1000, 'sgd_stepsize_mode': 'fixed_decay', 'sgd_batch_size': 500}
        nn_training_param = NN_training_param(d['sgd_step_size'],d['reg_lambda'], d['nb_iter'], d['sgd_batch_size'], d['w_scale'])
        assert(0)
    except AssertionError:
        sgd_step_size = 10.0/20.0
        reg_lambda = 1.0/1000.0
        nb_iter = 1000
        sgd_batch_size = 500
        w_scale = 0.1
        sgd_stepsize_mode = 'training data' 
        sgd_step_c = 0.1
        sgd_step_epsilon = 0.1

        nn_training_param = nn.NN_training_param(sgd_step_size, reg_lambda, nb_iter, sgd_batch_size, w_scale)
        with open(file_dir+"/../../pickle_files/multi/nn_training_param.p", "wb") as f:
            pickle.dump(nn_training_param, f)

    nn_navigation = NN_navigation_value(num_agents, nn_training_param, mode=mode, passing_side=passing_side)
    

    ''' load nn_navigation '''
    if filename is None:
        nn_filename = file_dir + "/../../pickle_files/multi/%d_agents_nn_regr_value_data.p" % num_agents
        print('loading NN trained on cadrl at ' + nn_filename)
    else:
        nn_filename = file_dir + "/../../pickle_files/multi/" + mode + "_" + passing_side + \
            "/RL_selfplay/" + filename
        if ifPrint:
            print('loading NN trained on RL self-play ' + nn_filename)

    # loading neural network
    try:
        # assert (0)
        nn_navigation.nn.load_neural_network(nn_filename)
    # train neural network for nn_navigation
    except AssertionError:
        print('failed to load navigation value network',  nn_filename)
        assert(0)
        # load data 
        dataset_ped_train, dataset_ped_test, dataset_ped_test_vel = load_ped_data(file_dir, num_agents)
        # train neural network
        nn_navigation.train_neural_network(dataset_ped_train, dataset_ped_test, dataset_ped_test_vel)
        nn_navigation.nn.save_neural_network(nn_filename)
        print('saved nn_navigation')

    return nn_navigation

def load_ped_data(file_dir, num_agents):
    try:
        # print 'num_agents', num_agents
        dataset_ped_train = pickle.load(open(file_dir+\
            "/../../pickle_files/multi/%d_agents_dataset_value_train.p"%num_agents, "rb"))
        # print 'radius', dataset_ped_train[0][:,8:10]
        dataset_ped_test = pickle.load(open(file_dir+\
            "/../../pickle_files/multi/%d_agents_dataset_value_test.p"%num_agents, "rb"))
        # print 'loaded dataset'
        num_train_pts = dataset_ped_train[0].shape[0]
        num_test_pts = dataset_ped_test[0].shape[0]
        # print 'here'
        dataset_ped_test_vel = pickle.load(open(file_dir+\
            "/../../pickle_files/multi/%d_agents_dataset_regr_test.p"%num_agents, "rb"))
        dataset_ped_test_vel = dataset_ped_test_vel[1]
        print('dataset contains %d pts, training set has %d pts, test set has %d pts' % \
                (num_train_pts+num_test_pts, num_train_pts, num_test_pts))
    except AssertionError:
        print('pickle file does not exist, exiting')
        assert(0)

    return dataset_ped_train, dataset_ped_test, dataset_ped_test_vel


if __name__ == '__main__':
    print('hello world from nn_navigation_value_multi.py')
    file_dir = os.path.dirname(os.path.realpath(__file__))
    plt.rcParams.update({'font.size': 18})
    
    ''' load nn_navigation_value and train if pickle file doesnot exist '''
    num_agents = 4
    mode = 'no_constr'
    # mode = 'rotate_constr'
    nn_navigation = load_NN_navigation_value(file_dir, num_agents, mode, 'none')

    ''' make prediction '''
    dataset_ped_train, dataset_ped_test, dataset_ped_test_vel = load_ped_data(file_dir, num_agents)
    scores = nn_navigation.nn.make_prediction_raw(dataset_ped_test[0])
    # nn_navigation.plot_ped_testCase_rand(dataset_ped_test[0], scores, 'testing')
    # nn_navigation.plot_ped_testCase_rand(dataset_ped_test[0], scores, 'testing')
    # nn_navigation.plot_ped_testCase_rand(dataset_ped_test[0], scores, 'testing')
    test_case = np.array([[-3.0, 0.0, 3.0, 0.0, 1.0, 0.4],\
                                        [0.0, 3.0, 0.0, -3.0, 1.0, 0.4], \
                                        [0.0, -3.0, 0.0, 3.0, 1.0, 0.4], \
                                        [3.0, 0.0, -3.0, 0.0, 1.0, 0.4]])
    nn_navigation.generate_traj(test_case[0:num_agents], ifRandSpeed=False)
    test_case = gen_tc.generate_rand_test_case_multi(num_agents, 4.0, np.array([0.5,1.2]), np.array([0.3, 0.5]))
    nn_navigation.generate_traj(test_case)

    plt.show()
