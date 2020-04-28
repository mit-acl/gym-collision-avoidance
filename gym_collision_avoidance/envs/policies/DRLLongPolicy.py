import numpy as np
import os
import operator
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs import util
from gym_collision_avoidance.envs import Config
from collections import deque


try:
    import torch
    import torch.nn as nn
    from gym_collision_avoidance.envs.policies.DRL_Long.model.ppo import generate_action_no_sampling
    from gym_collision_avoidance.envs.policies.DRL_Long.model.net import MLPPolicy, CNNPolicy
except:
    print("Torch not installed...")

LASER_HIST = 3

class DRLLongPolicy(InternalPolicy):
    """ Wrapper for an implementation of `Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep Reinforcement Learning <https://arxiv.org/abs/1709.10082>`_.

    Based on `this open-source implementation <https://github.com/Acmece/rl-collision-avoidance>`_.

    .. note::
        This policy is not fully working with this version of the code

    """
    def __init__(self):
        InternalPolicy.__init__(self, str="DRL_Long")
        self.obs_stack = None

    def initialize_network(self, **kwargs):
        if 'checkpt_name' in kwargs:
            checkpt_name = kwargs['checkpt_name']
        else:
            checkpt_name = 'stage2.pth'

        if 'checkpt_dir' in kwargs:
            checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/DRL_Long/policy/' + kwargs['checkpt_dir'] +'/'
        else:
            checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/DRL_Long/policy/'

        policy_path = checkpt_dir + checkpt_name

        policy = CNNPolicy(frames=LASER_HIST, action_space=2)

        if os.path.exists(policy_path):
            # print('Loading DRL_Long policy...')
            state_dict = torch.load(policy_path, map_location=torch.device('cpu'))
            policy.load_state_dict(state_dict)
        else:
            print ('Error: Policy File Cannot Find')
            exit()

        self.nn = policy
        self.action_bound = [[0, -1], [1, 1]]

        # self.count = 0 ### DELETE later

    def find_next_action(self, obs, agents, i):
        """ Normalize the laserscan, grab the goal position, query the NN, return the action.

        TODO

        """
        host_agent = agents[i]
        other_agents = agents[:i]+agents[i+1:]

        laserscan = obs['laserscan'] / 6.0 - 0.5

        x, y = host_agent.pos_global_frame
        goal_x, goal_y = host_agent.goal_global_frame
        theta = host_agent.heading_global_frame
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        goal = [local_x, local_y]

        # speed: [v.linear.x, v.angular.z]
        speed = host_agent.vel_global_frame[0]*np.array([np.cos(host_agent.heading_global_frame), np.sin(host_agent.heading_global_frame)])
        if self.obs_stack is None:
            self.obs_stack = deque([laserscan, laserscan, laserscan])
        else:
            self.obs_stack.popleft()
            self.obs_stack.append(laserscan)

        state = [self.obs_stack, goal, speed]
        state_list = [state]
        # print('---')
        # print(state[1])
        # print(obs['laserscan'])

        # self.count += 1
        # if self.count == 2:
        #     import pickle
        #     with open('/home/mfe/code/scan_gym.p', 'wb') as f:
        #         pickle.dump(self.obs_stack, f, protocol=2)
        #     assert(0)

        env_ = type('', (object,), {'index': 0})
        mean, scaled_action = generate_action_no_sampling(env=env_, state_list=state_list,
                                               policy=self.nn, action_bound=self.action_bound)

        [vx, vw] = scaled_action[0]
        delta_heading = vw*Config.DT
        action = np.array([vx, delta_heading])

        # action = self.near_goal_smoother(host_agent.dist_to_goal, host_agent.pref_speed, host_agent.heading_global_frame, action)

        # Note: I think it's impossible to account for pref_speed, as the network
        # is trained to output 0-1m/s. If we scaled it by pref_speed, it
        # would be unfair...
        # consider the example:
        # - agent is crossing perpindicular to ego agent
        # - network decides to go full speed to cut in front of agent
        # - we clip speed to be pref_speed (say 0.3m/s)
        # - collision
        # - network would have gone behind at low speed if it knew about clipping
        return action