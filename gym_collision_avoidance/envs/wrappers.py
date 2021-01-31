import gym
import numpy as np
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv


__all__ = ['FlattenDictWrapper', 'MultiagentFlattenDictWrapper', 'MultiagentDummyVecEnv', 'MultiagentDictToMultiagentArrayWrapper']

class MultiagentFlattenDictWrapper(gym.ObservationWrapper):
    """Flattens selected keys of a Dict observation space into
    an array.
    """
    def __init__(self, env, dict_keys, max_num_agents):
        super(MultiagentFlattenDictWrapper, self).__init__(env)
        self.dict_keys = dict_keys
        self.max_num_agents = max_num_agents

        self.setup_obs(max_num_agents, dict_keys)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=self.obs_shape, dtype='float32')

        single_agent_size = self.observation_indices[0]['BOUNDS'][1] - self.observation_indices[0]['BOUNDS'][0]
        self.single_agent_observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(single_agent_size,), dtype='float32')

        self.dict_observation_space = self.env.observation_space

    def setup_obs(self, max_num_agents, dict_keys):
        self.observation_indices = {}
        size = 0
        for agent in range(max_num_agents):
            self.observation_indices[agent] = {}
            agent_lower_ind = size
            for key in dict_keys:
                shape = self.env.observation_space.spaces[key].shape
                prev_size = size
                size += np.prod(shape)
                self.observation_indices[agent][key] = [prev_size, size]
            agent_upper_ind = size
            self.observation_indices[agent]['BOUNDS'] = [agent_lower_ind, agent_upper_ind]

        self.obs_shape = (size,)

    def observation(self, observation):
        # Turn multiagent dict obs into a really long 1d array
        # with all agents & states concatenated
        assert isinstance(observation, dict)
        obs = []
        for agent in range(self.max_num_agents):
            for key in self.dict_keys:
                obs.append(observation[agent][key].ravel())
        return np.concatenate(obs)

    def observationArrayToDict(self, observation_array):
        assert isinstance(observation_array, np.ndarray)
        assert observation_array.shape == self.observation_space.shape
        obs = {}
        for agent in range(self.max_num_agents):
            obs[agent] = {}
            for key in self.dict_keys:
                obs[agent][key] = observation_array[self.observation_indices[agent][key][0]: self.observation_indices[agent][key][1]]
        return obs

    def multiEnvObservationArrayToDict(self, observation_array):
        assert isinstance(observation_array, np.ndarray)
        # assert observation_array.shape == self.observation_space.shape
        num_envs = observation_array.shape[0]
        dict_obs = np.empty((num_envs, self.max_num_agents), dtype=dict)
        for env in range(num_envs):
            for agent in range(self.max_num_agents):
                key = 'use_ppo'
                ppo = observation_array[env][self.observation_indices[agent][key][0]: self.observation_indices[agent][key][1]]
                if ppo == False:
                    continue
                dict_obs[env][agent] = {}
                for key in self.dict_keys:
                    dict_obs[env][agent][key] = observation_array[env][self.observation_indices[agent][key][0]: self.observation_indices[agent][key][1]].reshape(self.env.observation_space.spaces[key].shape)
        return dict_obs

    def singleAgentObservationArrayToDict(self, observation_array, agent):
        assert isinstance(observation_array, np.ndarray)
        # assert observation_array.shape == self.single_agent_observation_space.shape

        obs = []
        for env in range(observation_array.shape[0]):
            obs.append({})
            for key in self.dict_keys:
                obs[env][key] = observation_array[env, self.observation_indices[agent][key][0]: self.observation_indices[agent][key][1]].reshape(self.env.observation_space.spaces[key].shape)
        return obs

    def keyToArrayInds(self, key):
        inds = []
        for agent in range(self.max_num_agents):
            inds.append(self.observation_indices[agent][key])
        return inds

    def singleAgentObservationArray(self, observation_array, agent):
        return observation_array[self.observation_indices[agent]['BOUNDS'][0]:self.observation_indices[agent]['BOUNDS'][1]]

    def singleAgentObservationInds(self, agent):
        return self.observation_indices[agent]['BOUNDS']

class FlattenDictWrapper(MultiagentFlattenDictWrapper):
    def __init__(self, env, dict_keys):
        super(FlattenDictWrapper, self).__init__(env, dict_keys, max_num_agents=1)

class MultiagentDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        DummyVecEnv.__init__(self, env_fns)

        self.buf_dones = np.zeros((self.num_envs,), dtype=np.ndarray)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.ndarray)

class MultiagentDictToMultiagentArrayWrapper(MultiagentFlattenDictWrapper):
    def __init__(self, env, dict_keys, max_num_agents):
        super(MultiagentDictToMultiagentArrayWrapper, self).__init__(env, dict_keys, max_num_agents=max_num_agents)

    def setup_obs(self, max_num_agents, dict_keys):
        self.observation_indices = {}
        for agent in range(max_num_agents):
            self.observation_indices[agent] = {}
            size = 0
            for key in dict_keys:
                shape = self.env.observation_space.spaces[key].shape
                prev_size = size
                size += np.prod(shape)
                self.observation_indices[agent][key] = [prev_size, size]
            agent_upper_ind = size
            self.observation_indices[agent]['BOUNDS'] = [0, agent_upper_ind]

        self.obs_shape = (max_num_agents, size)

    def observation(self, observation):
        # Turn multiagent dict obs into a 2d array
        # with shape (max_num_agents, num_states_per_agent)
        assert isinstance(observation, dict)
        obs = np.zeros(shape=self.obs_shape)
        for agent in range(self.max_num_agents):
            for key in self.dict_keys:
                low, high = self.observation_indices[agent][key]
                obs[agent][low:high] = observation[agent][key].ravel()
        return obs