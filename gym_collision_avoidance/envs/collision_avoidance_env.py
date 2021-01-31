'''
Collision Avoidance Environment
Author: Michael Everett
MIT Aerospace Controls Lab
'''

import gym
import gym.spaces
import numpy as np
import itertools
import copy
import os
import inspect
import sys

from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import find_nearest, rgba2rgb, l2norm, makedirs
from gym_collision_avoidance.envs.visualize import plot_episode, animate_episode
from gym_collision_avoidance.envs.agent import Agent
from gym_collision_avoidance.envs.Map import Map
from gym_collision_avoidance.envs import test_cases as tc

class CollisionAvoidanceEnv(gym.Env):
    """ Gym Environment for multiagent collision avoidance

    The environment contains a list of agents.

    :param agents: (list) A list of :class:`~gym_collision_avoidance.envs.agent.Agent` objects that represent the dynamic objects in the scene.
    :param num_agents: (int) The maximum number of agents in the environment.
    """

    # Attributes:
    #     agents: A list of :class:`~gym_collision_avoidance.envs.agent.Agent` objects that represent the dynamic objects in the scene.
    #     num_agents: The maximum number of agents in the environment.

    metadata = {
        # UNUSED !!
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        self.id = 0

        # Initialize Rewards
        self._initialize_rewards()

        # Simulation Parameters
        self.num_agents = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT
        self.dt_nominal = Config.DT

        # Collision Parameters
        self.collision_dist = Config.COLLISION_DIST
        self.getting_close_range = Config.GETTING_CLOSE_RANGE

        # Plotting Parameters
        self.evaluate = Config.EVALUATE_MODE

        self.plot_episodes = Config.SHOW_EPISODE_PLOTS or Config.SAVE_EPISODE_PLOTS
        self.plt_limits = Config.PLT_LIMITS
        self.plt_fig_size = Config.PLT_FIG_SIZE
        self.test_case_index = 0

        self.set_testcase(Config.TEST_CASE_FN, Config.TEST_CASE_ARGS)

        self.animation_period_steps = Config.ANIMATION_PERIOD_STEPS

        # if Config.TRAIN_ON_MULTIPLE_AGENTS:
        #     self.low_state = np.zeros((Config.FULL_LABELED_STATE_LENGTH))
        #     self.high_state = np.zeros((Config.FULL_LABELED_STATE_LENGTH))
        # else:
        #     self.low_state = np.zeros((Config.FULL_STATE_LENGTH))
        #     self.high_state = np.zeros((Config.FULL_STATE_LENGTH))

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
        

        # original observation space
        # self.observation_space = gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
        
        # not used...
        # self.observation_space = np.array([gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
                                           # for _ in range(self.num_agents)])
        # observation_space = gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
        
        # single agent dict obs
        self.observation = {}
        for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
            self.observation[agent] = {}

        # The observation returned by the environment is a Dict of Boxes, keyed by agent index.
        self.observation_space = gym.spaces.Dict({})
        for state in Config.STATES_IN_OBS:
            self.observation_space.spaces[state] = gym.spaces.Box(Config.STATE_INFO_DICT[state]['bounds'][0]*np.ones((Config.STATE_INFO_DICT[state]['size'])),
                Config.STATE_INFO_DICT[state]['bounds'][1]*np.ones((Config.STATE_INFO_DICT[state]['size'])),
                dtype=Config.STATE_INFO_DICT[state]['dtype'])
            for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
                self.observation[agent][state] = np.zeros((Config.STATE_INFO_DICT[state]['size']), dtype=Config.STATE_INFO_DICT[state]['dtype'])

        self.agents = None
        self.default_agents = None
        self.prev_episode_agents = None

        self.static_map_filename = None
        self.map = None

        self.episode_step_number = None
        self.episode_number = 0

        self.plot_save_dir = None
        self.plot_policy_name = None

        self.perturbed_obs = None

    def step(self, actions, dt=None):
        """ Run one timestep of environment dynamics.

        This is the main function. An external process will compute an action for every agent
        then call env.step(actions). The agents take those actions,
        then we check if any agents have earned a reward (collision/goal/...).
        Then agents take an observation of the new world state. We compute whether each agent is done
        (collided/reached goal/ran out of time) and if everyone's done, the episode ends.
        We return the relevant info back to the process that called env.step(actions).

        Args:
            actions (list): list of [delta heading angle, speed] commands (1 per agent in env)
            dt (float): time in seconds to run the simulation (defaults to :code:`self.dt_nominal`)

        Returns:
        4-element tuple containing

        - **next_observations** (*np array*): (obs_length x num_agents) with each agent's observation
        - **rewards** (*list*): 1 scalar reward per agent in self.agents
        - **game_over** (*bool*): true if every agent is done
        - **info_dict** (*dict*): metadata that helps in training

        """

        if dt is None:
            dt = self.dt_nominal

        self.episode_step_number += 1

        # Take action
        self._take_action(actions, dt)

        # Collect rewards
        rewards = self._compute_rewards()

        # Take observation
        next_observations = self._get_obs()

        if Config.ANIMATE_EPISODES and self.episode_step_number % self.animation_period_steps == 0:
            plot_episode(self.agents, False, self.map, self.test_case_index,
                circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ,
                plot_save_dir=self.plot_save_dir,
                plot_policy_name=self.plot_policy_name,
                save_for_animation=True,
                limits=self.plt_limits,
                fig_size=self.plt_fig_size,
                perturbed_obs=self.perturbed_obs,
                show=False,
                save=True)

        # Check which agents' games are finished (at goal/collided/out of time)
        which_agents_done, game_over = self._check_which_agents_done()

        which_agents_done_dict = {}
        which_agents_learning_dict = {}
        for i, agent in enumerate(self.agents):
            which_agents_done_dict[agent.id] = which_agents_done[i]
            which_agents_learning_dict[agent.id] = agent.policy.is_still_learning

        return next_observations, rewards, game_over, \
            {
                'which_agents_done': which_agents_done_dict,
                'which_agents_learning': which_agents_learning_dict,
            }

    def reset(self):
        """ Resets the environment, re-initializes agents, plots episode (if applicable) and returns an initial observation.

        Returns:
            initial observation (np array): each agent's observation given the initial configuration
        """
        if self.episode_step_number is not None and self.episode_step_number > 0 and self.plot_episodes and self.test_case_index >= 0:
            plot_episode(self.agents, self.evaluate, self.map, self.test_case_index, self.id, circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ, plot_save_dir=self.plot_save_dir, plot_policy_name=self.plot_policy_name, limits=self.plt_limits, fig_size=self.plt_fig_size, show=Config.SHOW_EPISODE_PLOTS, save=Config.SAVE_EPISODE_PLOTS)
            if Config.ANIMATE_EPISODES:
                animate_episode(num_agents=len(self.agents), plot_save_dir=self.plot_save_dir, plot_policy_name=self.plot_policy_name, test_case_index=self.test_case_index, agents=self.agents)
            self.episode_number += 1
        self.begin_episode = True
        self.episode_step_number = 0
        self._init_agents()
        if Config.USE_STATIC_MAP:
            self._init_static_map()
        for state in Config.STATES_IN_OBS:
            for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
                self.observation[agent][state] = np.zeros((Config.STATE_INFO_DICT[state]['size']), dtype=Config.STATE_INFO_DICT[state]['dtype'])
        return self._get_obs()

    def _take_action(self, actions, dt):
        """ Some agents' actions come externally through the actions arg, agents with internal policies query their policy here, 
        then each agent takes a step simultaneously.

        This makes it so an external script that steps through the environment doesn't need to
        be aware of internals of the environment, like ensuring RVO agents compute their RVO actions.
        Instead, all policies that are already trained/frozen are computed internally, and if an
        agent's policy is still being trained, it's convenient to isolate the training code from the environment this way.
        Or, if there's a real robot with its own planner on-board (thus, the agent should have an ExternalPolicy), 
        we don't bother computing its next action here and just take what the actions dict said.

        Args:
            actions (dict): keyed by agent indices, each value has a [delta heading angle, speed] command.
                Agents with an ExternalPolicy sub-class receive their actions through this dict.
                Other agents' indices shouldn't appear in this dict, but will be ignored if so, because they have 
                an InternalPolicy sub-class, meaning they can
                compute their actions internally given their observation (e.g., already trained CADRL, RVO, Non-Cooperative, etc.)
            dt (float): time in seconds to run the simulation (defaults to :code:`self.dt_nominal`)

        """
        num_actions_per_agent = 2  # speed, delta heading angle
        all_actions = np.zeros((len(self.agents), num_actions_per_agent), dtype=np.float32)

        # Agents set their action (either from external or w/ find_next_action)
        for agent_index, agent in enumerate(self.agents):
            if agent.is_done:
                continue
            elif agent.policy.is_external:
                all_actions[agent_index, :] = agent.policy.external_action_to_action(agent, actions[agent_index])
            else:
                dict_obs = self.observation[agent_index]
                all_actions[agent_index, :] = agent.policy.find_next_action(dict_obs, self.agents, agent_index)

        # After all agents have selected actions, run one dynamics update
        for i, agent in enumerate(self.agents):
            agent.take_action(all_actions[i,:], dt)

    def _update_top_down_map(self):
        """ After agents have moved, call this to update the map with their new occupancies. """
        self.map.add_agents_to_map(self.agents)
        # plt.imshow(self.map.map)
        # plt.pause(0.1)

    def set_agents(self, agents):
        """ Set the default agent configuration, which will get used at the start of each episode (and bypass calling self.test_case_fn)

        Args:
            agents (list): of :class:`~gym_collision_avoidance.envs.agent.Agent` objects that should become the self.default_agents
                and thus be loaded in that configuration every time the env resets.

        """
        self.default_agents = agents

    def _init_agents(self):
        """ Set self.agents (presumably at the start of a new episode) and set each agent's max heading change and speed based on env limits.

        self.agents gets set to self.default_agents if it exists.
        Otherwise, self.agents gets set to the result of self.test_case_fn(self.test_case_args).        
        """

        # The evaluation scripts need info about the previous episode's agents
        # (in case env.reset was called and thus self.agents was wiped)
        if self.evaluate and self.agents is not None:
            self.prev_episode_agents = copy.deepcopy(self.agents)

        # If nobody set self.default agents, query the test_case_fn
        if self.default_agents is None:
            self.agents = self.test_case_fn(**self.test_case_args)
        # Otherwise, somebody must want the agents to be reset in a certain way already
        else:
            self.agents = self.default_agents

        # Make every agent respect the same env-wide limits on actions (this probably should live elsewhere...)
        for agent in self.agents:
            agent.max_heading_change = self.max_heading_change
            agent.max_speed = self.max_speed

    def set_static_map(self, map_filename):
        """ If you want to have static obstacles, provide the path to the map image file that should be loaded.
        
        Args:
            map_filename (str or list): full path of a binary png file corresponding to the environment prior map 
                (or list of candidate map paths to randomly choose btwn each episode)
        """
        self.static_map_filename = map_filename

    def _init_static_map(self):
        """ Load the map based on its pre-provided filename, and initialize a :class:`~gym_collision_avoidance.envs.Map.Map` object

        Currently the dimensions of the world map are hard-coded.

        """
        if isinstance(self.static_map_filename, list):
            static_map_filename = np.random.choice(self.static_map_filename)
        else:
            static_map_filename = self.static_map_filename

        x_width = 16 # meters
        y_width = 16 # meters
        grid_cell_size = 0.1 # meters/grid cell
        self.map = Map(x_width, y_width, grid_cell_size, static_map_filename)

    def _compute_rewards(self):
        """ Check for collisions and reaching of the goal here, and also assign the corresponding rewards based on those calculations.
        
        Returns:
            rewards (scalar or list): is a scalar if we are only training on a single agent, or
                      is a list of scalars if we are training on mult agents
        """

        # if nothing noteworthy happened in that timestep, reward = -0.01
        rewards = self.reward_time_step*np.ones(len(self.agents))
        collision_with_agent, collision_with_wall, entered_norm_zone, dist_btwn_nearest_agent = \
            self._check_for_collisions()

        for i, agent in enumerate(self.agents):
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
                        #       % agent.id)
                    elif collision_with_wall[i]:
                        rewards[i] = self.reward_collision_with_wall
                        agent.in_collision = True
                        # print("Agent %i: Collision with wall!"
                              # % agent.id)
                    else:
                        # There was no collision
                        if dist_btwn_nearest_agent[i] <= Config.GETTING_CLOSE_RANGE:
                            rewards[i] = -0.1 - dist_btwn_nearest_agent[i]/2.
                            # print("Agent %i: Got close to another agent!"
                            #       % agent.id)
                        if abs(agent.past_actions[0, 1]) > self.wiggly_behavior_threshold:
                            # Slightly penalize wiggly behavior
                            rewards[i] += self.reward_wiggly_behavior
                        # elif entered_norm_zone[i]:
                        #     rewards[i] = self.reward_entered_norm_zone
        rewards = np.clip(rewards, self.min_possible_reward,
                          self.max_possible_reward)
        if Config.TRAIN_SINGLE_AGENT:
            rewards = rewards[0]
        return rewards

    def _check_for_collisions(self):
        """ Check whether each agent has collided with another agent or a static obstacle in the map 
        
        This method doesn't compute social zones currently!!!!!

        Returns:
            - collision_with_agent (list): for each agent, bool True if that agent is in collision with another agent
            - collision_with_wall (list): for each agent, bool True if that agent is in collision with object in map
            - entered_norm_zone (list): for each agent, bool True if that agent entered another agent's social zone
            - dist_btwn_nearest_agent (list): for each agent, float closest distance to another agent

        """
        collision_with_agent = [False for _ in self.agents]
        collision_with_wall = [False for _ in self.agents]
        entered_norm_zone = [False for _ in self.agents]
        dist_btwn_nearest_agent = [np.inf for _ in self.agents]
        agent_shapes = []
        agent_front_zones = []
        agent_inds = list(range(len(self.agents)))
        agent_pairs = list(itertools.combinations(agent_inds, 2))
        for i, j in agent_pairs:
            dist_btwn = l2norm(self.agents[i].pos_global_frame, self.agents[j].pos_global_frame)
            combined_radius = self.agents[i].radius + self.agents[j].radius
            dist_btwn_nearest_agent[i] = min(dist_btwn_nearest_agent[i], dist_btwn - combined_radius)
            if dist_btwn <= combined_radius:
                # Collision with another agent!
                collision_with_agent[i] = True
                collision_with_agent[j] = True
        if Config.USE_STATIC_MAP:
            for i in agent_inds:
                agent = self.agents[i]
                [pi, pj], in_map = self.map.world_coordinates_to_map_indices(agent.pos_global_frame)
                mask = self.map.get_agent_map_indices([pi, pj], agent.radius)
                # plt.figure('static map')
                # plt.imshow(self.map.static_map + mask)
                # plt.pause(0.1)
                if in_map and np.any(self.map.static_map[mask]):
                    # Collision with wall!
                    collision_with_wall[i] = True
        return collision_with_agent, collision_with_wall, entered_norm_zone, dist_btwn_nearest_agent

    def _check_which_agents_done(self):
        """ Check if any agents have reached goal, run out of time, or collided.

        Returns:
            - which_agents_done (list): for each agent, True if agent is done, o.w. False
            - game_over (bool): depending on mode, True if all agents done, True if 1st agent done, True if all learning agents done
        """
        at_goal_condition = np.array(
                [a.is_at_goal for a in self.agents])
        ran_out_of_time_condition = np.array(
                [a.ran_out_of_time for a in self.agents])
        in_collision_condition = np.array(
                [a.in_collision for a in self.agents])
        which_agents_done = np.logical_or.reduce((at_goal_condition, ran_out_of_time_condition, in_collision_condition))
        for agent_index, agent in enumerate(self.agents):
            agent.is_done = which_agents_done[agent_index]
        
        if Config.EVALUATE_MODE:
            # Episode ends when every agent is done
            game_over = np.all(which_agents_done)
        elif Config.TRAIN_SINGLE_AGENT:
            # Episode ends when ego agent is done
            game_over = which_agents_done[0]
        else:
            # Episode is done when all *learning* agents are done
            learning_agent_inds = [i for i in range(len(self.agents)) if self.agents[i].policy.is_still_learning]
            game_over = np.all(which_agents_done[learning_agent_inds])
        
        return which_agents_done, game_over

    def _get_obs(self):
        """ Update the map now that agents have moved, have each agent sense the world, and fill in their observations 

        Returns:
            observation (list): for each agent, a dictionary observation.

        """

        if Config.USE_STATIC_MAP:
            # Agents have moved (states have changed), so update the map view
            self._update_top_down_map()

        # Agents collect a reading from their map-based sensors
        for i, agent in enumerate(self.agents):
            agent.sense(self.agents, i, self.map)

        # Agents fill in their element of the multiagent observation vector
        for i, agent in enumerate(self.agents):
            self.observation[i] = agent.get_observation_dict(self.agents)

        return self.observation

    def _initialize_rewards(self):
        """ Set some class attributes regarding reward values based on Config """
        self.reward_at_goal = Config.REWARD_AT_GOAL
        self.reward_collision_with_agent = Config.REWARD_COLLISION_WITH_AGENT
        self.reward_collision_with_wall = Config.REWARD_COLLISION_WITH_WALL
        self.reward_getting_close = Config.REWARD_GETTING_CLOSE
        self.reward_entered_norm_zone = Config.REWARD_ENTERED_NORM_ZONE
        self.reward_time_step = Config.REWARD_TIME_STEP

        self.reward_wiggly_behavior = Config.REWARD_WIGGLY_BEHAVIOR
        self.wiggly_behavior_threshold = Config.WIGGLY_BEHAVIOR_THRESHOLD

        self.possible_reward_values = \
            np.array([self.reward_at_goal,
                      self.reward_collision_with_agent,
                      self.reward_time_step,
                      self.reward_collision_with_wall,
                      self.reward_wiggly_behavior
                      ])
        self.min_possible_reward = np.min(self.possible_reward_values)
        self.max_possible_reward = np.max(self.possible_reward_values)

    def set_plot_save_dir(self, plot_save_dir):
        """ Set where to save plots of trajectories (will get created if non-existent)
        
        Args:
            plot_save_dir (str): path to directory you'd like to save plots in

        """
        makedirs(plot_save_dir, exist_ok=True)
        self.plot_save_dir = plot_save_dir

    def set_perturbed_info(self, perturbed_obs):
        """ Used for robustness paper to pass info that could be visualized. Too hacky.
        """
        self.perturbed_obs = perturbed_obs

    def set_testcase(self, test_case_fn_str, test_case_args):
        """ 

        Args:
            test_case_fn_str (str): name of function in test_cases.py
        """

        # Provide a fn (which returns list of agents) and the fn's args,
        # to be called on each env.reset()
        test_case_fn = getattr(tc, test_case_fn_str, None)
        assert(callable(test_case_fn))

        # Before running test_case_fn, make sure we didn't provide any args it doesn't accept
        if sys.version[0] == '3':
            signature = inspect.signature(test_case_fn)
        elif sys.version[0] == '2':
            import funcsigs
            signature = funcsigs.signature(test_case_fn)
        test_case_fn_args = signature.parameters
        test_case_args_keys = list(test_case_args.keys())
        for key in test_case_args_keys:
            # print("checking if {} accepts {}".format(test_case_fn, key))
            if key not in test_case_fn_args:
                # print("{} doesn't accept {} -- removing".format(test_case_fn, key))
                del test_case_args[key]
        self.test_case_fn = test_case_fn
        self.test_case_args = test_case_args

if __name__ == '__main__':
    print("See example.py for a minimum working example.")