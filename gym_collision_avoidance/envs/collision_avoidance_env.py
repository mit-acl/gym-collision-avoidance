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

import matplotlib.pyplot as plt

from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.util import find_nearest, rgba2rgb

from gym_collision_avoidance.envs.agent import Agent
from gym_collision_avoidance.envs.static_agent import StaticAgent
from gym_collision_avoidance.envs.non_cooperative_agent \
        import NonCooperativeAgent
from gym_collision_avoidance.envs.cadrl_agent \
        import CADRLAgent
from gym_collision_avoidance.envs.CADRL.scripts.multi \
        import gen_rand_testcases as tc
from gym_collision_avoidance.envs.visualize import plot_episode
from gym_collision_avoidance.envs.test_cases import preset_testCases, gen_circle_test_case

if Config.USE_STAGE_ROS:
    from geometry_msgs.msg import Pose, Vector3
    import rospy
    import subprocess
    import rosgraph
    from stage_ros.srv import CmdPosesRecScans, \
        CmdPosesRecScansRequest, \
        NewEpisode, NewEpisodeRequest
    #  from sensor_msgs.msg import LaserScan

class CollisionAvoidanceEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        self.id = 0
        self.episode_step_number = 0
        if Config.USE_STAGE_ROS:
            self._setup_stage_ros()

        # Initialize Rewards
        self._initialize_rewards()

        # Simulation Parameters
        self.num_agents = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT
        self.dt = Config.DT

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
        #                                    for _ in range(self.num_agents)])
        # observation_space = gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
        # self.observation_space = gym.spaces.Dict({})
        # for i in range(self.num_agents):
        #     self.observation_space.spaces["agent_"+str(i)] = observation_space

        self.agents = None

    def step(self, actions):
        ###############################
        # This is the main function. An external process will compute an action for every agent
        # then call env.step(actions). The agents take those actions (or ignore them if running
        # a different policy), then we check if any agents have earned a reward (collision/goal/...).
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

        self.episode_step_number += 1

        # Take action
        self._take_action(actions)

        # Collect rewards
        rewards = self._compute_rewards()

        # Take observation
        next_observations = self._get_obs()

        # Check which agents' games are finished (at goal/collided/out of time)
        which_agents_done, game_over = self._check_which_agents_done()

        which_agents_done_dict = {}
        for i, agent in enumerate(self.agents):
            which_agents_done_dict[agent.id] = which_agents_done[i]

        return next_observations, rewards, game_over, \
            {'which_agents_done': which_agents_done_dict}

    def reset(self):
        if self.agents is not None and Config.PLOT_EPISODES:
            visualize.plot_episode(self.agents, self.evaluate, self.test_case_index)
        self.begin_episode = True
        self.episode_step_number = 0
        self._init_env(test_case=0)
        return self._get_obs()
        # TODO: for stage branch, confirm agent is getting correct scan on reset

    def close(self):
        print("--- Closing CollisionAvoidanceEnv! ---")
        if Config.USE_STAGE_ROS:
            subprocess.Popen(["rosnode", "kill",
                              self.stage_ros_env_ns+"/stage_ros"])
        return

    def _take_action(self, actions):
        ###############################
        # This function sends an action to each Agent object's update_state method.
        # Only the PPO agents should listen to the action coming from the external process;
        # everyone else (e.g. static, noncoop) has their own agent.find_next_action
        # method ==> for non-ppo agents we ignore the contents of their rows of "actions"!!
        ###############################

        action_vectors = [np.array([0.0, 0.0]) for i in range(self.num_agents)]

        # First extract next action of each agent from "actions"
        for i, agent in enumerate(self.agents):
            if agent.is_at_goal or agent.in_collision:
                action_vector = np.array([0.0, 0.0])
            elif agent.policy_type == 'CADRL':
                action_vector = agent.find_next_action(self.agents)
            elif agent.policy_type == 'DQN':  # TODO: Fix this up
                action = actions[0]  # probably doesnt work anymore
                action_vector = np.array([agent.pref_speed, action])
            elif agent.policy_type == 'PPO':
                # mfe's ppo network outputs actions btwn 0-1 (beta distr)
                heading = self.max_heading_change*(2.*actions[i, 1] - 1.)
                speed = agent.pref_speed * actions[i, 0]
                action_vector = np.array([speed, heading])
            else:
                action_vector = agent.find_next_action(self.agents)
            action_vectors[i] = action_vector

        # ...then update their states based on chosen actions
        for i, agent in enumerate(self.agents):
            agent.update_state(action_vectors[i], self.dt)

        if Config.USE_STAGE_ROS:
            update_agents_in_stage_ros()

    def _init_agents(self, test_case=None, alg='PPO'):
        ###############################
        # This function initializes the self.agents list.
        #
        # Outputs
        # - self.agents: list of Agent objects that have been initialized
        # - self.which_agents_running_ppo: list of T/F values for each agent in self.agents
        ###############################

        # Agents
        # easy_test_cases = True
        easy_test_cases = False
        random_test_cases = True
        if easy_test_cases:
            goal_x = 3
            goal_y = 3
            self.agents = np.array([
                Agent(goal_x, -goal_y, goal_x, goal_y, 0.5, 1.0, 0.5, 0),
                Agent(-goal_x, goal_y, -goal_x, -goal_y, 0.5, 1.0, 0.5, 1)])
        else:
            if self.evaluate:
                print("self.test_case_index:", self.test_case_index)
                self.test_case_index = 1
                # self.test_case_index += 1
                self.test_case_num_agents = 2
                # self.full_test_suite = True
                self.full_test_suite = False
                test_cases = \
                    preset_testCases(self.test_case_num_agents,
                                     full_test_suite=self.full_test_suite)
                test_case = test_cases[self.test_case_index]

            elif random_test_cases:
                num_agents = 2
                # num_agents = np.random.randint(2, Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)
                side_length = 4
                #  side_length = np.random.uniform(4, 8)
                speed_bnds = [0.5, 1.5]
                radius_bnds = [0.2, 0.8]

                test_case = \
                    tc.generate_rand_test_case_multi(num_agents,
                                                     side_length,
                                                     speed_bnds,
                                                     radius_bnds,
                                                     is_end_near_bnd=False,
                                                     is_static=False)

            self.agents = self._cadrl_test_case_to_agents(test_case, alg=alg)
        self.which_agents_running_ppo = \
            [agent.id for agent in self.agents if agent.policy_type == 'PPO']
        self.num_agents_running_ppo = len(self.which_agents_running_ppo)

    def _cadrl_test_case_to_agents(self, test_case, alg='PPO'):
        ###############################
        # This function accepts a test_case in legacy cadrl format and converts it
        # into our new list of Agent objects. The legacy cadrl format is a list of
        # [start_x, start_y, goal_x, goal_y, pref_speed, radius] for each agent.
        ###############################

        agents = []
        policies = [Agent, NonCooperativeAgent, StaticAgent, CADRLAgent]
        if self.evaluate:
            if alg == 'PPO':
                # All PPO agents
                agent_policy_list = [0 for _ in range(np.shape(test_case)[0])]
            elif alg == 'CADRL':
                # All CADRL agents
                agent_policy_list = [-1 for _ in range(np.shape(test_case)[0])]
        else:
            # Random mix of agents following various policies
            agent_policy_list = np.random.choice(len(policies),
                                                 np.shape(test_case)[0],
                                                 p=[0.9, 0.05, 0.05, 0.0])
            if 0 not in agent_policy_list:
                # Make sure at least one agent is following PPO
                #  (otherwise waste of time...)
                random_agent_id = np.random.randint(len(agent_policy_list))
                agent_policy_list[random_agent_id] = 0
        for i, agent in enumerate(test_case):
            px = agent[0]
            py = agent[1]
            gx = agent[2]
            gy = agent[3]
            pref_speed = agent[4]
            radius = agent[5]
            if self.evaluate:
                # initial heading is pointed toward the goal
                vec_to_goal = np.array([gx, gy]) - np.array([px, py])
                heading = np.arctan2(vec_to_goal[1], vec_to_goal[0])
            else:
                heading = np.random.uniform(-np.pi, np.pi)

            agents.append(policies[agent_policy_list[i]](px, py, gx, gy,
                                                         radius, pref_speed,
                                                         heading, i))
        return agents

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
                    #       % agent.id)
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
        for i in agent_inds:
            agent = self.agents[i]
            if min(agent.latest_laserscan.ranges) - agent.radius < 0.1:
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
        # mostly a useless method, except in stage_ros mode
        self._init_agents(test_case=test_case, alg=alg)
        if Config.USE_STAGE_ROS:
            self._init_stage_env()

    def _update_static_world_id(self, id=None):
        # only used in stage_ros mode
        if id is None:
            # self.static_world_id = 2
            self.static_world_id = np.random.randint(0,3)
        else:
            self.static_world_id = 0
        self.world_name = "cadrl_{id}".format(id=self.static_world_id)
        self.world_bitmap_filename = "/home/mfe/code/Stage/worlds/bitmaps/{world_name}.png".format(world_name=self.world_name)


    def _setup_stage_ros(self):
        try:
            rosgraph.Master('/rostopic').getPid()
        except:
            raise Exception("No ROS Master exists! Please start \
                one externally and re-start this script.")
        rospy.init_node('collision_avoidance_env_%i' % self.id)

        self._update_static_world_id(id=0)
        self.world_filename = "/home/mfe/code/Stage/worlds/{world_name}.world".format(world_name=self.world_name)
        self.stage_ros_env_ns = "/stage_env_%i" % self.id
        subprocess.Popen(["rosnode", "kill",
                          self.stage_ros_env_ns+"/stage_ros"])
        self.stage_ros_process = \
            subprocess.Popen(["roslaunch", "stage_ros",
                              "stage_ros_node.launch",
                              "ns:=%s" % self.stage_ros_env_ns,
                              "world_filename:=%s"
                              % self.world_filename])
        # Connect to service
        srv_name = '{}/command_poses_receive_scans'.format(self.stage_ros_env_ns)
        rospy.wait_for_service(srv_name, timeout=5.0)
        self.srv_cmd_poses_rec_scans = rospy.ServiceProxy(srv_name,
                                                          CmdPosesRecScans)
        rospy.loginfo("[{}] Srv: {} available.\n".format(rospy.get_name(),
                                                     srv_name))
        # Connect to service
        srv_name = '{}/new_episode'.format(self.stage_ros_env_ns)
        rospy.wait_for_service(srv_name, timeout=5.0)
        self.srv_new_episode = rospy.ServiceProxy(srv_name, NewEpisode)
        rospy.loginfo("[{}] Srv: {} available.\n".format(rospy.get_name(),
                                                     srv_name))
    def _update_agents_in_stage_ros(self):
        # Update each agent's position in Stage simulator
        req = CmdPosesRecScansRequest()
        for agent in self.agents:
            pose_msg = Pose()
            pose_msg.position.x = agent.pos_global_frame[0]
            pose_msg.position.y = agent.pos_global_frame[1]
            pose_msg.orientation.w = agent.heading_global_frame
            req.poses.append(pose_msg)
        resp = self.srv_cmd_poses_rec_scans(req)
        # Then update each agent's laserscan
        for i, agent in enumerate(self.agents):
            agent.latest_laserscan = resp.laserscans[i]

    def _init_stage_env(self):
        req = NewEpisodeRequest()
        for agent in self.agents:
            # Update each agent's radius in Stage simulator
            req.sizes.append(Vector3(x=agent.radius, y=agent.radius,
                                     z=0.1))
            # Update each agent's position in Stage simulator
            pose_msg = Pose()
            pose_msg.position.x = agent.pos_global_frame[0]
            pose_msg.position.y = agent.pos_global_frame[1]
            pose_msg.orientation.w = agent.heading_global_frame
            req.poses.append(pose_msg)
        self._update_static_world_id()
        req.bitmap_path = self.world_bitmap_filename
        resp = self.srv_new_episode(req)
        # Then update each agent's laserscan
        for i, agent in enumerate(self.agents):
            agent.latest_laserscan = resp.laserscans[i]
            

if __name__ == '__main__':
    ## Minimum working example
    env = CollisionAvoidanceEnv()
    print("Created environment.")
    env.reset()
    print("Reset environment.")
    num_agents = len(env.agents)
    actions = np.zeros((num_agents,2), dtype=np.float32)
    num_steps = 10
    for i in range(num_steps):
        env.step(actions)
    print("Sent {steps} steps to environment.".format(steps=num_steps))