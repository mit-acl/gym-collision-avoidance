import numpy as np
import os
import operator
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs import util
from gym_collision_avoidance.envs.policies.GA3C_CADRL import network
from gym_collision_avoidance.envs import Config

class GA3CCADRLPolicy(InternalPolicy):
    """ Pre-trained policy from `Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning <https://arxiv.org/pdf/1805.01956.pdf>`_

    By default, loads a pre-trained LSTM network (GA3C-CADRL-10-LSTM from the paper). There are 11 discrete actions with max heading angle change of $\pm \pi/6$.

    """
    def __init__(self):
        InternalPolicy.__init__(self, str="GA3C_CADRL")

        self.possible_actions = network.Actions()
        num_actions = self.possible_actions.num_actions
        self.device = '/cpu:0'
        self.nn = network.NetworkVP_rnn(self.device, 'network', num_actions)

    def initialize_network(self, **kwargs):
        """ Load the model parameters of either a default file, or if provided through kwargs, a specific path and/or tensorflow checkpoint.

        Args:
            kwargs['checkpt_name'] (str): name of checkpoint file to load (without file extension)
            kwargs['checkpt_dir'] (str): path to checkpoint

        """

        if 'checkpt_name' in kwargs:
            checkpt_name = kwargs['checkpt_name']
        else:
            checkpt_name = 'network_01900000'

        if 'checkpt_dir' in kwargs:
            if os.path.isabs(kwargs['checkpt_dir']):
                # Given an absolute path (use it exactly)
                checkpt_dir = kwargs['checkpt_dir']
            else:
                # Given a relative path (append it to normal path)
                checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/GA3C_CADRL/checkpoints/' + kwargs['checkpt_dir'] +'/'
        else:
            checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/GA3C_CADRL/checkpoints/IROS18/'

        self.nn.simple_load(checkpt_dir + checkpt_name)

    def find_next_action(self, obs, agents, i):
        """ Using only the dictionary obs, convert this to the vector needed for the GA3C-CADRL network, query the network, adjust the actions for this env.

        Args:
            obs (dict): this :class:`~gym_collision_avoidance.envs.agent.Agent` 's observation vector
            agents (list): [unused] of :class:`~gym_collision_avoidance.envs.agent.Agent` objects
            i (int): [unused] index of agents list corresponding to this agent
        
        Returns:
            [spd, heading change] command

        """

        pref_speed = obs['pref_speed']
        # host_agent = agents[i]
        # other_agents = agents[:i]+agents[i+1:]
        # new_obs = self.agents_to_ga3c_cadrl_state(host_agent, other_agents)
        # new_obs = np.expand_dims(new_obs[1:], axis=0)

        if type(obs) == dict:
            # Turn the dict observation into a flattened vector
            vec_obs = np.array([])
            for state in Config.STATES_IN_OBS:
                if state not in Config.STATES_NOT_USED_IN_POLICY:
                    vec_obs = np.hstack([vec_obs, obs[state].flatten()])
            vec_obs = np.expand_dims(vec_obs, axis=0)

        # print(obs)
        # print(vec_obs)
        # assert(0)

        predictions = self.nn.predict_p(vec_obs)[0]
        action_index = np.argmax(predictions)
        raw_action = self.possible_actions.actions[action_index]
        action = np.array([pref_speed*raw_action[0], raw_action[1]])
        return action

    # def agents_to_ga3c_cadrl_state(self, host_agent, other_agents):

    #     obs = np.zeros((network.Config.FULL_LABELED_STATE_LENGTH))

    #     # Own agent state (ID is removed before inputting to NN, num other agents is used to rearrange other agents into sequence by NN)
    #     obs[0] = host_agent.id 
    #     if network.Config.MULTI_AGENT_ARCH == 'RNN':
    #         obs[network.Config.AGENT_ID_LENGTH] = 0 
    #     obs[network.Config.AGENT_ID_LENGTH+network.Config.FIRST_STATE_INDEX:network.Config.AGENT_ID_LENGTH+network.Config.FIRST_STATE_INDEX+network.Config.HOST_AGENT_STATE_SIZE] = \
    #                          host_agent.dist_to_goal, host_agent.heading_ego_frame, host_agent.pref_speed, host_agent.radius

    #     other_agent_dists = []
    #     for i, other_agent in enumerate(other_agents):
    #         # project other elements onto the new reference frame
    #         rel_pos_to_other_global_frame = other_agent.pos_global_frame - host_agent.pos_global_frame
    #         p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame, host_agent.ref_orth)
    #         dist_between_agent_centers = np.linalg.norm(rel_pos_to_other_global_frame)
    #         dist_2_other = dist_between_agent_centers - host_agent.radius - other_agent.radius
    #         if dist_between_agent_centers > network.Config.SENSING_HORIZON:
    #             # print "Agent too far away"
    #             continue
    #         other_agent_dists.append([i,round(dist_2_other,2),p_orthog_ego_frame])
    #     sorted_dists = sorted(other_agent_dists, key = lambda x: (-x[1], x[2]))
    #     sorted_inds = [x[0] for x in sorted_dists]
    #     clipped_sorted_inds = sorted_inds[-network.Config.MAX_NUM_OTHER_AGENTS_OBSERVED:]
    #     clipped_sorted_agents = [other_agents[i] for i in clipped_sorted_inds]
    #     self.num_nearby_agents = len(clipped_sorted_inds)

    #     i = 0
    #     for other_agent in clipped_sorted_agents:
    #         # project other elements onto the new reference frame
    #         rel_pos_to_other_global_frame = other_agent.pos_global_frame - host_agent.pos_global_frame
    #         p_parallel_ego_frame = np.dot(rel_pos_to_other_global_frame, host_agent.ref_prll)
    #         p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame, host_agent.ref_orth)
    #         v_parallel_ego_frame = np.dot(other_agent.vel_global_frame, host_agent.ref_prll)
    #         v_orthog_ego_frame = np.dot(other_agent.vel_global_frame, host_agent.ref_orth)

    #         dist_2_other = np.linalg.norm(rel_pos_to_other_global_frame) - host_agent.radius - other_agent.radius
    #         combined_radius = host_agent.radius + other_agent.radius
    #         is_on = 1

    #         start_index = network.Config.AGENT_ID_LENGTH + network.Config.FIRST_STATE_INDEX + network.Config.HOST_AGENT_STATE_SIZE + network.Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i
    #         end_index = network.Config.AGENT_ID_LENGTH + network.Config.FIRST_STATE_INDEX + network.Config.HOST_AGENT_STATE_SIZE + network.Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*(i+1)
            
    #         other_obs = np.array([p_parallel_ego_frame, p_orthog_ego_frame, v_parallel_ego_frame, v_orthog_ego_frame, other_agent.radius, \
    #                                 combined_radius, dist_2_other])
    #         if network.Config.MULTI_AGENT_ARCH in ['WEIGHT_SHARING','VANILLA']:
    #             other_obs = np.hstack([other_obs, is_on])
    #         obs[start_index:end_index] = other_obs
    #         i += 1

            
    #     if network.Config.MULTI_AGENT_ARCH == 'RNN':
    #         obs[network.Config.AGENT_ID_LENGTH] = i # Will be used by RNN for seq_length
    #     if network.Config.MULTI_AGENT_ARCH in ['WEIGHT_SHARING','VANILLA']:
    #         for j in range(i,network.Config.MAX_NUM_OTHER_AGENTS_OBSERVED):
    #             start_index = network.Config.AGENT_ID_LENGTH + network.Config.FIRST_STATE_INDEX + network.Config.HOST_AGENT_STATE_SIZE + network.Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*j
    #             end_index = network.Config.AGENT_ID_LENGTH + network.Config.FIRST_STATE_INDEX + network.Config.HOST_AGENT_STATE_SIZE + network.Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*(j+1)
    #             other_obs[-1] = 0
    #             obs[start_index:end_index] = other_obs

    #     return obs

    #     #
    #     # Observation vector is as follows;
    #     # [<this_agent_info>, <other_agent_1_info>, <other_agent_2_info>, ... , <other_agent_n_info>] 
    #     # where <this_agent_info> = [id, dist_to_goal, heading (in ego frame)]
    #     # where <other_agent_i_info> = [pos in this agent's ego parallel coord, pos in this agent's ego orthog coord]
    #     #


if __name__ == '__main__':
    policy = GA3CCADRLPolicy()
    policy.initialize_network()