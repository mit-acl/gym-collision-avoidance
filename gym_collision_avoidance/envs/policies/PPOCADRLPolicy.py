import numpy as np
import os
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from stable_baselines.common.policies import build_policy
from gym_collision_avoidance.envs import Config
from stable_baselines.ppo2.mfe_network import mfe_network
import tensorflow as tf

class PPOCADRLPolicy(InternalPolicy):
    def __init__(self):
        InternalPolicy.__init__(self, str="PPO_CADRL")
        self.ppo_or_learning_policy = True

    def initialize_network(self, **kwargs):
        if 'checkpt_name' in kwargs:
            checkpoint_name = kwargs['checkpt_name']
        else:
            checkpoint_name = '2019-07-09-02-21-39-039832-00610'
        network = Config.NETWORK
        file_dir = os.path.dirname(os.path.realpath(__file__)) + '/PPO_CADRL/checkpoints/'
        load_path = file_dir + checkpoint_name
        load_path = "/tmp/wandb/run-20190712_184254-9ko7u0w0/checkpoints/00575.ckpt"
        load_path = "/tmp/wandb/run-20190712_210835-xos1fxrm/checkpoints/00410.ckpt"
        load_path = "/tmp/wandb/run-20190712_213449-3nteijf4/checkpoints/00545.ckpt"
        load_path = "/tmp/wandb/run-20190712_222413-ha8ghayz/checkpoints/00500.ckpt"
        load_path = "/tmp/wandb/run-20190714_183509-x6mb5po7/checkpoints/00545.ckpt"
        load_path = "/tmp/wandb/run-20190714_190552-nmamlb75/checkpoints/00610.ckpt"
        load_path = "/tmp/wandb/run-20190714_195544-ozrm02gt/checkpoints/00610.ckpt"
        load_path = "/tmp/wandb/run-20190714_203538-7peo10xo/checkpoints/00610.ckpt"
        load_path = "/tmp/wandb/run-20190715_200243-ngfnyie7/checkpoints/00610.ckpt"
        load_path = "/tmp/wandb/run-20190716_001241-4bw8vz63/checkpoints/02100.ckpt"


        network_kwargs = {
            'states_in_obs': Config.STATES_IN_OBS,
            'states_not_used_in_network': Config.STATES_NOT_USED_IN_POLICY,
            'mean_obs': Config.MEAN_OBS,
            'std_obs': Config.STD_OBS,
            'normalize_input': Config.NORMALIZE_INPUT,
            'num_hidden': Config.NUM_HIDDEN_UNITS,
            'num_layers': Config.NUM_LAYERS,
            'lstm_hidden_size': Config.LSTM_HIDDEN_SIZE
        }

        policy = build_policy(self.env, network, **network_kwargs)

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            self.trained_policy = policy(1, 1)
        self.trained_policy.load(load_path)

        self.one_env = self.env.envs[0]

    def find_next_action(self, dict_obs, agents, agent_index):

        network_output, _, _, _ = self.trained_policy.step([dict_obs], evaluate=True)

        print(network_output)

        agent = agents[agent_index]
        heading = agent.max_heading_change*(2.*network_output[0, 1] - 1.)
        speed = agent.pref_speed * network_output[0, 0]
        actions = np.array([speed, heading])
        return actions
