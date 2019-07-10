import numpy as np
import os
from gym_collision_avoidance.envs.policies.Policy import Policy
from baselines.common.policies import build_policy
from gym_collision_avoidance.envs.config import Config
from baselines.ppo2.mfe_network import mfe_network
import tensorflow as tf

class PPOCADRLPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="PPO_CADRL")
        self.is_still_learning = False

    def initialize_network(self):
        network = "mfe_network"
        file_dir = os.path.dirname(os.path.realpath(__file__)) + '/PPO_CADRL/checkpoints/'
        load_path = file_dir + '2019-07-09-15-10-38-798399-00610'

        network_kwargs = {
            'states_in_obs': Config.STATES_IN_OBS,
            'states_not_used_in_network': Config.STATES_NOT_USED_IN_POLICY,
            'mean_obs': Config.MEAN_OBS,
            'std_obs': Config.STD_OBS,
            'normalize_input': Config.NORMALIZE_INPUT,
            'num_hidden': 64,
            'num_layers': 3,
            'lstm_hidden_size': 64
        }

        policy = build_policy(self.env, network, **network_kwargs)

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            self.trained_policy = policy(1, 1)
        self.trained_policy.load(load_path)

        self.one_env = self.env.envs[0]

    def find_next_action(self, dict_obs, agents, agent_index):

        network_output, _, _, _ = self.trained_policy.step([dict_obs], evaluate=True)

        agent = agents[agent_index]
        heading = agent.max_heading_change*(2.*network_output[0, 1] - 1.)
        speed = agent.pref_speed * network_output[0, 0]
        actions = np.array([speed, heading])
        return actions
