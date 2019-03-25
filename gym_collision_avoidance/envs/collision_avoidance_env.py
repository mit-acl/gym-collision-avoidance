'''
Collision Avoidance Environment
Author: Michael Everett
MIT Aerospace Controls Lab
'''

import gym
import gym.spaces
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import translate, rotate
import pickle
import time
from collections import OrderedDict
import itertools
import copy
import matplotlib.pyplot as plt

from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.util import find_nearest, rgba2rgb
from gym_collision_avoidance.envs.visualize import plot_episode
from gym_collision_avoidance.envs.agent import Agent
from gym_collision_avoidance.envs.Map import Map
from gym_collision_avoidance.envs import test_cases as tc

class CollisionAvoidanceEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        self.id = 0
        self.episode_step_number = 0

        # Initialize Rewards
        self._initialize_rewards()

        # Simulation Parameters
        self.num_agents = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT
        self.dt_nominal = Config.DT

        # Collision Parameters
        self.collision_dist = Config.COLLISION_DIST
        self.getting_close_range = Config.GETTING_CLOSE_RANGE

        self.evaluate = Config.EVALUATE_MODE
        self.plot_episodes = Config.PLOT_EPISODES
        self.test_case_index = -1

        # Size of domain (only used for viz)
        self.min_x = -10.0
        self.max_x = 10.0
        self.min_y = -10.0
        self.max_y = 10.0

        if Config.TRAIN_ON_MULTIPLE_AGENTS:
            self.low_state = np.zeros((Config.FULL_LABELED_STATE_LENGTH))
            self.high_state = np.zeros((Config.FULL_LABELED_STATE_LENGTH))
        else:
            self.low_state = np.zeros((Config.FULL_STATE_LENGTH))
            self.high_state = np.zeros((Config.FULL_STATE_LENGTH))

        self.viewer = None

        # Upper/Lower bounds on Actions
        self.max_heading_change = np.pi/3
        self.min_heading_change = -self.max_heading_change
        self.min_speed = 0.0
        self.max_speed = 1.0

        ### The gym.spaces library doesn't support Python2.7 (syntax of Super().__init__())
        self.action_space_type = Config.ACTION_SPACE_TYPE
        if self.action_space_type == Config.discrete:
            self.action_space = gym.spaces.Discrete(self.actions.num_actions, dtype=np.float32)
        elif self.action_space_type == Config.continuous:
            self.low_action = np.array([self.min_speed,
                                        self.min_heading_change])
            self.high_action = np.array([self.max_speed,
                                         self.max_heading_change])
            self.action_space = gym.spaces.Box(self.low_action, self.high_action, dtype=np.float32)
        self.observation_space = gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)


        # self.observation_space = np.array([gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
                                           # for _ in range(self.num_agents)])
        # observation_space = gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
        # self.observation_space = gym.spaces.Dict({})
        # for i in range(self.num_agents):
        #     self.observation_space.spaces["agent_"+str(i)] = observation_space

        self.agents = None
        self.default_agents = None

        self.static_map_filename = None
        self.map = None

        self.episode_step_number = None

    def step(self, actions, dt=None):
        ###############################
        # This is the main function. An external process will compute an action for every agent
        # then call env.step(actions). The agents take those actions,
        # then we check if any agents have earned a reward (collision/goal/...).
        # Then agents take an observation of the new world state. We compute whether each agent is done
        # (collided/reached goal/ran out of time) and if everyone's done, the episode ends.
        # We return the relevant info back to the process that called env.step(actions).
        #
        # Inputs
        # - actions: list of [delta heading angle, speed] commands (1 per agent in env)
        # Outputs
        # - next_observations: (obs_length x num_agents) np array with each agent's observation
        # - rewards: list with 1 scalar reward per agent in self.agents
        # - game_over: boolean, true if every agent is done
        # - info_dict: metadata (more details) that help in training, for example
        ###############################

        if dt is None:
            dt = self.dt_nominal

        self.episode_step_number += 1

        # Take action
        self._take_action(actions, dt)

        # Collect rewards
        rewards = self._compute_rewards()

        # Take observation
        next_observations = self._get_obs()

        # if self.episode_step_number % 5:
        #     plot_episode(self.agents, self.evaluate, self.map, self.test_case_index)

        # Check which agents' games are finished (at goal/collided/out of time)
        which_agents_done, game_over = self._check_which_agents_done()

        which_agents_done_dict = {}
        for i, agent in enumerate(self.agents):
            which_agents_done_dict[agent.id] = which_agents_done[i]

        return next_observations, rewards, game_over, \
            {'which_agents_done': which_agents_done_dict}

    def reset(self):
        if self.episode_step_number is not None and Config.PLOT_EPISODES:
            plot_episode(self.agents, self.evaluate, self.map, self.test_case_index)
        self.begin_episode = True
        self.episode_step_number = 0
        self._init_agents()
        self._init_static_map()
        self._init_env()
        return self._get_obs()

    def close(self):
        print("--- Closing CollisionAvoidanceEnv! ---")
        return

    def _take_action(self, actions, dt):
        ###############################
        # This function sends an action to each Agent object's take_action method.
        ###############################
        for i, agent in enumerate(self.agents):
            agent.take_action(actions[i,:], dt)

    def update_top_down_map(self):
        self.map.add_agents_to_map(self.agents)
        # plt.imshow(self.map.map)
        # plt.pause(0.1)

    def set_agents(self, agents):
        self.default_agents = agents

    def _init_agents(self):
        if self.default_agents is None:
            self.agents = tc.get_testcase_random()
        else:
            self.agents = self.default_agents
        for agent in self.agents:
            agent.max_heading_change = self.max_heading_change
            agent.max_speed = self.max_speed

    def set_static_map(self, map_filename):
        self.static_map_filename = map_filename
        
    def _init_static_map(self):
        x_width = 10 # meters
        y_width = 10 # meters
        grid_cell_size = 0.1 # meters/grid cell
        self.map = Map(x_width, y_width, grid_cell_size, self.static_map_filename)

    def _compute_rewards(self):
        ###############################
        # We check for collisions and reaching of the goal here, and also assign
        # the corresponding rewards based on those calculations.
        #
        # Outputs
        #   - rewards: is a scalar if we are only training on a single agent, or
        #               is a list of scalars if we are training on mult agents
        ###############################

        if Config.TRAIN_ON_MULTIPLE_AGENTS:
            agents = self.agents
        else:
            agents = [self.agents[0]]

        # if nothing noteworthy happened in that timestep, reward = -0.01
        rewards = self.reward_time_step*np.ones(len(agents))
        collision_with_agent, collision_with_wall, entered_norm_zone = \
            self._check_for_collisions()

        for i, agent in enumerate(agents):
            if agent.is_at_goal:
                if agent.was_at_goal_already is False:
                    # agents should only receive the goal reward once
                    rewards[i] = self.reward_at_goal
                    # print("Agent %i: Arrived at goal!"
                          # % agent.id)
            else:
                # agents at their goal shouldn't be penalized if someone else
                # bumps into them
                if agent.was_in_collision_already is False:
                    if collision_with_agent[i]:
                        rewards[i] = self.reward_collision_with_agent
                        agent.in_collision = True
                        # print("Agent %i: Collision with another agent!"
                              # % agent.id)
                    elif collision_with_wall[i]:
                        rewards[i] = self.reward_collision_with_wall
                        agent.in_collision = True
                        print("Agent %i: Collision with wall!"
                              % agent.id)
                    #  elif entered_norm_zone[i]:
                    #      rewards[i] = self.reward_entered_norm_zone
                    elif abs(agent.past_actions[0, 1]) > 0.4:
                        # Slightly penalize wiggly behavior
                        rewards[i] += -0.003
        rewards = np.clip(rewards, self.min_possible_reward,
                          self.max_possible_reward)
        if not Config.TRAIN_ON_MULTIPLE_AGENTS:
            rewards = rewards[0]
        return rewards

    def _check_for_collisions(self):
        # NOTE: This method doesn't compute social zones!!!!!
        collision_with_agent = [False for _ in self.agents]
        collision_with_wall = [False for _ in self.agents]
        entered_norm_zone = [False for _ in self.agents]
        agent_shapes = []
        agent_front_zones = []
        agent_inds = list(range(len(self.agents)))
        agent_pairs = list(itertools.combinations(agent_inds, 2))
        for i, j in agent_pairs:
            agent = self.agents[i]
            other_agent = self.agents[j]
            dist_btwn = np.linalg.norm(
                agent.pos_global_frame - other_agent.pos_global_frame)
            combined_radius = agent.radius + other_agent.radius
            if dist_btwn <= combined_radius:
                # Collision with another agent!
                collision_with_agent[i] = True
                collision_with_agent[j] = True
        for i in agent_inds:
            agent = self.agents[i]
            [pi, pj], in_map = self.map.world_coordinates_to_map_indices(agent.pos_global_frame)
            mask = self.map.get_agent_map_indices([pi, pj], agent.radius)
            if in_map and np.any(self.map.static_map[mask]):
            # if in_map and self.map.static_map[pi, pj] == 1:
                # Collision with wall!
                collision_with_wall[i] = True
        return collision_with_agent, collision_with_wall, entered_norm_zone

    def _check_which_agents_done(self):
        at_goal_condition = np.array(
                [a.is_at_goal for a in self.agents])
        ran_out_of_time_condition = np.array(
                [a.ran_out_of_time for a in self.agents])
        in_collision_condition = np.array(
                [a.in_collision for a in self.agents])
        which_agents_done = \
            np.logical_or(
                np.logical_or(at_goal_condition, ran_out_of_time_condition),
                in_collision_condition)
        if Config.EVALUATE_MODE:
            game_over = np.all(which_agents_done)
        elif Config.TRAIN_ON_MULTIPLE_AGENTS:
            game_over = np.all(which_agents_done)
            # game_over = which_agents_done[0]
        else:
            game_over = which_agents_done[0]
        return which_agents_done, game_over

    def _get_obs(self):
        ###############################
        # Each agent observes the other agents in the scene and returns an observation
        # vector in a standard format, defined in config.
        #
        # Outputs
        #   - next_observations: array with each agent's observation vector, stacked
        ###############################

        self.update_top_down_map()
        for i, agent in enumerate(self.agents):
            agent.sense(self.agents, i, self.map)

        if Config.TRAIN_ON_MULTIPLE_AGENTS:
            next_observations = np.empty([len(self.agents),
                                          Config.FULL_LABELED_STATE_LENGTH])
        else:
            next_observations = np.empty([len(self.agents),
                                          Config.FULL_STATE_LENGTH])
        for i, agent in enumerate(self.agents):
            agent_obs = agent.observe(self.agents)
            next_observations[i] = agent_obs
        return next_observations

    def _initialize_rewards(self):
        self.reward_at_goal = Config.REWARD_AT_GOAL
        self.reward_collision_with_agent = Config.REWARD_COLLISION_WITH_AGENT
        self.reward_collision_with_wall = Config.REWARD_COLLISION_WITH_WALL
        self.reward_getting_close = Config.REWARD_GETTING_CLOSE
        self.reward_entered_norm_zone = Config.REWARD_ENTERED_NORM_ZONE
        self.reward_time_step = Config.REWARD_TIME_STEP
        self.possible_reward_values = \
            np.array([self.reward_at_goal,
                      self.reward_collision_with_agent,
                      self.reward_time_step,
                      self.reward_collision_with_wall])
        self.min_possible_reward = np.min(self.possible_reward_values)
        self.max_possible_reward = np.max(self.possible_reward_values)

    def _init_env(self, test_case=None, alg='PPO'):
        # currently a useless method
        return
            

if __name__ == '__main__':
    print("See example.py for a minimum working example.")