import numpy as np
import os
import operator
from gym_collision_avoidance.envs.policies.Policy import Policy
from gym_collision_avoidance.envs import util
from gym_collision_avoidance.envs.policies.GA3C_CADRL import network
from gym_collision_avoidance.envs.config import Config

class GA3CCADRLPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="GA3C_CADRL")
        self.is_still_learning = False

        self.possible_actions = network.Actions()
        num_actions = self.possible_actions.num_actions
        self.nn = network.NetworkVP_rnn(network.Config.DEVICE, 'network', num_actions)

    def initialize_network(self, **kwargs):
        if 'checkpt_name' in kwargs:
            checkpt_name = kwargs['checkpt_name']
        else:
            checkpt_name = 'network_01900000'

        if 'checkpt_dir' in kwargs:
            checkpt_dir = kwargs['checkpt_dir']
        else:
            checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/GA3C_CADRL/checkpoints/'

        self.nn.simple_load(checkpt_dir + checkpt_name)

    def find_next_action(self, obs, agents, i):
        host_agent = agents[i]
        other_agents = agents[:i]+agents[i+1:]
        obs = self.agents_to_ga3c_cadrl_state(host_agent, other_agents)
        obs = np.expand_dims(obs[1:], axis=0)
        predictions = self.nn.predict_p(obs)[0]
        action_index = np.argmax(predictions)
        raw_action = self.possible_actions.actions[action_index]
        action = np.array([host_agent.pref_speed*raw_action[0], raw_action[1]])
        return action

    def agents_to_ga3c_cadrl_state(self, host_agent, other_agents):

        obs = np.zeros((Config.FULL_LABELED_STATE_LENGTH))

        # Own agent state (ID is removed before inputting to NN, num other agents is used to rearrange other agents into sequence by NN)
        obs[0] = host_agent.id 
        if Config.MULTI_AGENT_ARCH == 'RNN':
            obs[Config.AGENT_ID_LENGTH] = 0 
        obs[Config.AGENT_ID_LENGTH+Config.FIRST_STATE_INDEX:Config.AGENT_ID_LENGTH+Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE] = \
                             host_agent.dist_to_goal, host_agent.heading_ego_frame, host_agent.pref_speed, host_agent.radius

        other_agent_dists = []
        for i, other_agent in enumerate(other_agents):
            # project other elements onto the new reference frame
            rel_pos_to_other_global_frame = other_agent.pos_global_frame - host_agent.pos_global_frame
            p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame, host_agent.ref_orth)
            dist_between_agent_centers = np.linalg.norm(rel_pos_to_other_global_frame)
            dist_2_other = dist_between_agent_centers - host_agent.radius - other_agent.radius
            if dist_between_agent_centers > Config.SENSING_HORIZON:
                # print "Agent too far away"
                continue
            other_agent_dists.append([i,round(dist_2_other,2),p_orthog_ego_frame])
        sorted_dists = sorted(other_agent_dists, key = lambda x: (-x[1], x[2]))
        sorted_inds = [x[0] for x in sorted_dists]
        clipped_sorted_inds = sorted_inds[-Config.MAX_NUM_OTHER_AGENTS_OBSERVED:]
        clipped_sorted_agents = [other_agents[i] for i in clipped_sorted_inds]
        self.num_nearby_agents = len(clipped_sorted_inds)

        i = 0
        for other_agent in clipped_sorted_agents:
            # project other elements onto the new reference frame
            rel_pos_to_other_global_frame = other_agent.pos_global_frame - host_agent.pos_global_frame
            p_parallel_ego_frame = np.dot(rel_pos_to_other_global_frame, host_agent.ref_prll)
            p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame, host_agent.ref_orth)
            v_parallel_ego_frame = np.dot(other_agent.vel_global_frame, host_agent.ref_prll)
            v_orthog_ego_frame = np.dot(other_agent.vel_global_frame, host_agent.ref_orth)

            dist_2_other = np.linalg.norm(rel_pos_to_other_global_frame) - host_agent.radius - other_agent.radius
            combined_radius = host_agent.radius + other_agent.radius
            is_on = 1

            start_index = Config.AGENT_ID_LENGTH + Config.FIRST_STATE_INDEX + Config.HOST_AGENT_STATE_SIZE + Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i
            end_index = Config.AGENT_ID_LENGTH + Config.FIRST_STATE_INDEX + Config.HOST_AGENT_STATE_SIZE + Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*(i+1)
            
            other_obs = np.array([p_parallel_ego_frame, p_orthog_ego_frame, v_parallel_ego_frame, v_orthog_ego_frame, other_agent.radius, \
                                    combined_radius, dist_2_other])
            if Config.MULTI_AGENT_ARCH in ['WEIGHT_SHARING','VANILLA']:
                other_obs = np.hstack([other_obs, is_on])
            obs[start_index:end_index] = other_obs
            i += 1

            
        if Config.MULTI_AGENT_ARCH == 'RNN':
            obs[Config.AGENT_ID_LENGTH] = i # Will be used by RNN for seq_length
        if Config.MULTI_AGENT_ARCH in ['WEIGHT_SHARING','VANILLA']:
            for j in range(i,Config.MAX_NUM_OTHER_AGENTS_OBSERVED):
                start_index = Config.AGENT_ID_LENGTH + Config.FIRST_STATE_INDEX + Config.HOST_AGENT_STATE_SIZE + Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*j
                end_index = Config.AGENT_ID_LENGTH + Config.FIRST_STATE_INDEX + Config.HOST_AGENT_STATE_SIZE + Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*(j+1)
                other_obs[-1] = 0
                obs[start_index:end_index] = other_obs

        return obs

        #
        # Observation vector is as follows;
        # [<this_agent_info>, <other_agent_1_info>, <other_agent_2_info>, ... , <other_agent_n_info>] 
        # where <this_agent_info> = [id, dist_to_goal, heading (in ego frame)]
        # where <other_agent_i_info> = [pos in this agent's ego parallel coord, pos in this agent's ego orthog coord]
        #

        # obs = np.zeros((network.Config.FULL_LABELED_STATE_LENGTH))

        # # Own agent state (ID is removed before inputting to NN, num other agents is used to rearrange other agents into sequence by NN)
        # obs[0] = host_agent.id
        # if network.Config.MULTI_AGENT_ARCH == 'RNN':
        #     obs[network.Config.AGENT_ID_LENGTH] = 0 
        # obs[network.Config.AGENT_ID_LENGTH+network.Config.FIRST_STATE_INDEX:network.Config.AGENT_ID_LENGTH+network.Config.FIRST_STATE_INDEX+network.Config.HOST_AGENT_STATE_SIZE] = \
        #                      host_agent.dist_to_goal, host_agent.heading_ego_frame, host_agent.pref_speed, host_agent.radius

        # other_agent_dists = {}
        # for i, other_agent in enumerate(other_agents):
        #     # project other elements onto the new reference frame
        #     rel_pos_to_other_global_frame = other_agent.pos_global_frame - host_agent.pos_global_frame
        #     dist_between_agent_centers = np.linalg.norm(rel_pos_to_other_global_frame)
        #     dist_2_other = dist_between_agent_centers - host_agent.radius - other_agent.radius
        #     if dist_between_agent_centers > network.Config.SENSING_HORIZON:
        #         # print "Agent too far away"
        #         continue
        #     other_agent_dists[i] = dist_2_other
        # # print "other_agent_dists:", other_agent_dists
        # sorted_pairs = sorted(other_agent_dists.items(), key=operator.itemgetter(1))
        # sorted_inds = [ind for (ind,pair) in sorted_pairs]
        # sorted_inds.reverse()
        # clipped_sorted_inds = sorted_inds[-network.Config.MAX_NUM_OTHER_AGENTS_OBSERVED:]
        # clipped_sorted_agents = [other_agents[i] for i in clipped_sorted_inds]

        # self.num_nearby_agents = len(clipped_sorted_inds)
        # # print "sorted_inds:", sorted_inds
        # # print "clipped_sorted_inds:", clipped_sorted_inds
        # # print "clipped_sorted_agents:", clipped_sorted_agents

        # i = 0
        # for other_agent in clipped_sorted_agents:
        #     # project other elements onto the new reference frame
        #     rel_pos_to_other_global_frame = other_agent.pos_global_frame - host_agent.pos_global_frame
        #     p_parallel_ego_frame = np.dot(rel_pos_to_other_global_frame, host_agent.ref_prll)
        #     p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame, host_agent.ref_orth)
        #     v_parallel_ego_frame = np.dot(other_agent.vel_global_frame, host_agent.ref_prll)
        #     v_orthog_ego_frame = np.dot(other_agent.vel_global_frame, host_agent.ref_orth)
        #     dist_2_other = np.linalg.norm(rel_pos_to_other_global_frame) - host_agent.radius - other_agent.radius
        #     combined_radius = host_agent.radius + other_agent.radius
        #     is_on = 1

        #     start_index = network.Config.AGENT_ID_LENGTH + network.Config.FIRST_STATE_INDEX + network.Config.HOST_AGENT_STATE_SIZE + network.Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i
        #     end_index = network.Config.AGENT_ID_LENGTH + network.Config.FIRST_STATE_INDEX + network.Config.HOST_AGENT_STATE_SIZE + network.Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*(i+1)
            
        #     other_obs = np.array([p_parallel_ego_frame, p_orthog_ego_frame, v_parallel_ego_frame, v_orthog_ego_frame, other_agent.radius, \
        #                             combined_radius, dist_2_other])
        #     if network.Config.MULTI_AGENT_ARCH in ['WEIGHT_SHARING','VANILLA']:
        #         other_obs = np.hstack([other_obs, is_on])
        #     obs[start_index:end_index] = other_obs
        #     i += 1

            
        # if network.Config.MULTI_AGENT_ARCH == 'RNN':
        #     obs[network.Config.AGENT_ID_LENGTH] = i # Will be used by RNN for seq_length
        # if network.Config.MULTI_AGENT_ARCH in ['WEIGHT_SHARING','VANILLA']:
        #     for j in range(i,network.Config.MAX_NUM_OTHER_AGENTS_OBSERVED):
        #         start_index = network.Config.AGENT_ID_LENGTH + network.Config.FIRST_STATE_INDEX + network.Config.HOST_AGENT_STATE_SIZE + network.Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*j
        #         end_index = network.Config.AGENT_ID_LENGTH + network.Config.FIRST_STATE_INDEX + network.Config.HOST_AGENT_STATE_SIZE + network.Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*(j+1)
        #         other_obs[-1] = 0
        #         obs[start_index:end_index] = other_obs

        # return obs
