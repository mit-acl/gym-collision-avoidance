'''
Collision Avoidance Environement
Author: Michael Everett
MIT Aerospace Controls Lab
'''

import gym
import gym.spaces
# from gym import spaces
#  from gym.utils import seeding
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import translate, rotate
import pickle
import time
from collections import OrderedDict


import matplotlib.pyplot as plt
#  import matplotlib.patches as ptch
#  import matplotlib.colors

from gym.envs.classic_control import rendering

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

if Config.USE_ROS:
    import rospy
    from stage_ros.srv import CmdPosesRecScans, \
        CmdPosesRecScansRequest, \
        NewEpisode, NewEpisodeRequest
    from geometry_msgs.msg import Pose, Vector3
    import subprocess
    #  from sensor_msgs.msg import LaserScan


class CollisionAvoidanceEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        self.id = 0
        self.episode_step_number = 0
        if Config.USE_ROS:
            stage_ros_env_ns = "/stage_env_%i" % self.id
            subprocess.Popen(["rosnode", "kill",
                              stage_ros_env_ns+"/stage_ros"])
            self.stage_ros_process = \
                subprocess.Popen(["roslaunch", "stage_ros",
                                  "stage_ros_node.launch",
                                  "ns:=%s" % stage_ros_env_ns])
            rospy.init_node('collision_avoidance_env_%i' % self.id)
            # Connect to service
            srv_name = '%s/command_poses_receive_scans' % stage_ros_env_ns
            rospy.wait_for_service(srv_name, timeout=5.0)
            self.srv_cmd_poses_rec_scans = rospy.ServiceProxy(srv_name,
                                                              CmdPosesRecScans)
            rospy.loginfo("[%s] Srv: %s available.\n" % (rospy.get_name(),
                                                         srv_name))
            # Connect to service
            srv_name = '%s/new_episode' % stage_ros_env_ns
            rospy.wait_for_service(srv_name, timeout=5.0)
            self.srv_new_episode = rospy.ServiceProxy(srv_name, NewEpisode)
            rospy.loginfo("[%s] Srv: %s available.\n" % (rospy.get_name(),
                                                         srv_name))

        # Initialize Rewards
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

        # Simulation Parameters
        self.num_agents = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT
        self.dt = Config.DT

        # Collision Parameters
        self.collision_dist = Config.COLLISION_DIST
        self.getting_close_range = Config.GETTING_CLOSE_RANGE

        self.evaluate = Config.EVALUATE_MODE
        self.plot_episodes = Config.PLOT_EPISODES
        self.test_case_index = -1

        self.first = True

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

        # self.reset()

    def _init_env(self, test_case=None, alg='PPO'):
        self._init_agents(test_case=test_case, alg=alg)
        if Config.USE_ROS:
            req = NewEpisodeRequest()
            for agent in self.agents:
                req.sizes.append(Vector3(x=agent.radius, y=agent.radius,
                                         z=0.1))
            self.srv_new_episode(req)

    def _init_agents(self, test_case=None, alg='PPO'):
        # Agents
        #  easy_test_cases = True
        easy_test_cases = False
        random_test_cases = True
        if easy_test_cases:
            goal_x = np.random.uniform(-3, 3)
            goal_y = np.random.uniform(1, 5)
            self.agents = np.array([
                Agent(goal_x, -goal_y, goal_x, goal_y, 0.5, 1.0, 0.5, 0),
                Agent(-goal_x, goal_y, -goal_x, -goal_y, 0.5, 1.0, 0.5, 1)])
            #  goal_x = np.random.choice([-1, 1])*np.random.uniform(2, 5)
            #  goal_y = np.random.uniform(-2, 2)
            #  self.agents = np.array([
            #      Agent(0, 0, goal_x, goal_y, 0.5, 1.0, 0.5, 0),
            #      Agent(goal_x, goal_y+5, 0, 5, 0.5, 1.0, 0.5, 1),
            #      Agent(goal_x, goal_y-5, 0, -5, 0.5, 1.0, np.pi, 2)])
        else:
            if self.evaluate:
                print("self.test_case_index:", self.test_case_index)
                self.test_case_index += 1
                self.test_case_num_agents = 2
                # self.full_test_suite = True
                self.full_test_suite = False
                test_cases = \
                    preset_testCases(self.test_case_num_agents,
                                     full_test_suite=self.full_test_suite)
                test_case = test_cases[self.test_case_index]

            elif random_test_cases:
                num_agents = 2
                # num_agents = random.randint(2, 3)
                # num_agents = random.randint(2, Config.MAX_NUM_AGENTS)
                # num_agents = np.random.randint(2, 4)
                side_length = 4
                #  side_length = np.random.uniform(4, 8)
                speed_bnds = [0.5, 1.5]
                radius_bnds = [0.2, 0.8]

                #  test_case = tc.generate_circle_case(num_agents,
                #                                      side_length,
                #                                      speed_bnds,
                #                                      radius_bnds)
                #  test_case = tc.generate_swap_case(num_agents,
                #                                      side_length,
                #                                      speed_bnds,
                #                                      radius_bnds)
                #  test_case = tc.generate_rand_case(num_agents,
                #                                      side_length,
                #                                      speed_bnds,
                #                                      radius_bnds,
                #                                      is_end_near_bnd=False)
                #  test_case = tc.generate_easy_rand_case(num_agents,
                #                                         side_length,
                #                                         speed_bnds,
                #                                         radius_bnds,
                #                                         2,
                #                                         is_end_near_bnd=False)
                test_case = \
                    tc.generate_rand_test_case_multi(num_agents,
                                                     side_length,
                                                     speed_bnds,
                                                     radius_bnds,
                                                     is_end_near_bnd=False,
                                                     is_static=False)
                # test_cases = preset_testCases(2, full_test_suite=False)
                # test_case = test_cases[0]
                # print test_case

            self.agents = self.cadrl_test_case_to_agents(test_case, alg=alg)
        self.which_agents_running_ppo = \
            [agent.id for agent in self.agents if agent.policy_type == 'PPO']
        self.num_agents_running_ppo = len(self.which_agents_running_ppo)

    def cadrl_test_case_to_agents(self, test_case, alg='PPO'):
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
                                                 p=[1.0, 0.0, 0.0, 0.0])
            #  agent_policy_list = np.random.choice(len(policies),
            #                                       np.shape(test_case)[0],
            #                                       p=[0.9, 0.05, 0.05, 0.0])
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

    def _take_action(self, actions):
        action_vectors = [np.array([0.0, 0.0]) for i in range(self.num_agents)]
        # First find next action of each agent
        # Importantly done before agents update their state
        # (mostly only impt for CADRL)
        for i, agent in enumerate(self.agents):
            heading = self.max_heading_change*(2.*actions[i, 1] - 1.)
            speed = agent.pref_speed * actions[i, 0]
            if agent.is_at_goal or agent.in_collision:
                action_vector = np.array([0.0, 0.0])
            elif agent.policy_type == 'CADRL':
                action_vector = agent.find_next_action(self.agents)
            elif agent.policy_type == 'DQN':  # TODO: Fix this up
                action = actions[0]  # probably doesnt work anymore
                action_vector = np.array([agent.pref_speed, action])
            elif agent.policy_type == 'PPO':
                # action_vector = np.array([1.0, 0.0])
                 action_vector = np.array([speed, heading])
            else:
                action_vector = agent.find_next_action(self.agents)
            action_vectors[i] = action_vector

        # ...then update their states based on chosen actions
        for i, agent in enumerate(self.agents):
            agent.update_state(action_vectors[i], self.dt)

        if Config.USE_ROS:
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

    def _compute_rewards(self):
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
                        # print("Agent %i: Collision with wall!"
                              # % agent.id)
                    #  elif entered_norm_zone[i]:
                    #      rewards[i] = self.reward_entered_norm_zone
                    #  elif abs(agent.past_actions[0, 1]) > 0.4:
                    #      rewards[i] += -0.003
        rewards = np.clip(rewards, self.min_possible_reward,
                          self.max_possible_reward)
        if not Config.TRAIN_ON_MULTIPLE_AGENTS:
            rewards = rewards[0]
        return rewards

    def _check_for_collisions(self):
        collision_with_agent = [False for _ in self.agents]
        collision_with_wall = [False for _ in self.agents]
        entered_norm_zone = [False for _ in self.agents]
        agent_shapes = []
        agent_front_zones = []
        for agent in self.agents:
            width = agent.radius
            length = agent.radius + 0.5
            agent_shapes.append(Point(agent.pos_global_frame[0],
                                      agent.pos_global_frame[1]).buffer(
                                          agent.radius))
            agent_front_zones.append(rotate(translate(
                Polygon([(0, -width), (length, -width),
                         (length, width), (0, width)]),
                xoff=agent.pos_global_frame[0],
                yoff=agent.pos_global_frame[1]),
                angle=agent.heading_global_frame,
                origin=Point(agent.pos_global_frame[0],
                             agent.pos_global_frame[1]),
                use_radians=True))
        agent_inds = list(range(len(self.agents)))
        for i in agent_inds:
            other_inds = agent_inds[:i] + agent_inds[i+1:]
            for j in other_inds:
                if agent_shapes[i].intersects(agent_shapes[j]):
                    # Collision with another agent!
                    collision_with_agent[i] = True
                if agent_shapes[i].intersects(agent_front_zones[j]):
                    # Entered another agent's social zone!
                    entered_norm_zone[i] = True
            agent = self.agents[i]
            if min(agent.latest_laserscan.ranges) - agent.radius < 1.0:
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
        if Config.TRAIN_ON_MULTIPLE_AGENTS or Config.EVALUATE_MODE:
            game_over = np.all(which_agents_done)
        else:
            game_over = which_agents_done[0]
        return which_agents_done, game_over

    def step(self, actions):
        self.episode_step_number += 1
        # time.sleep(0.05)
        #  print("-----------\n Step \n ----------------")
        # Take action
        self._take_action(actions)

        # Collect rewards
        rewards = self._compute_rewards()
        #  print("rewards:", rewards)

        # Take observation
        next_observations = self._get_obs()
        #  print("obs:", next_observations)

        # Visualize env
        #  if self.display_screen: self.visualize_ego_frame(next_observations)

        # Check which agents' games are finished (at goal/collided/out of time)
        which_agents_done, game_over = self._check_which_agents_done()
        #  print("which_agents_done:", which_agents_done)
        #  print("game_over:", game_over)

        which_agents_done_dict = {}
        for i, agent in enumerate(self.agents):
            which_agents_done_dict[agent.id] = which_agents_done[i]

        #  if self.episode_step_number % 5 == 0:
        # self._plot_episode()

        return next_observations, rewards, game_over, \
            {'which_agents_done': which_agents_done_dict}

    def _get_obs(self):
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

    def reset(self):
        #  print("--- Reset ---")
        if self.agents is not None and Config.PLOT_EPISODES:
            self._plot_episode()
        self.begin_episode = True
        self.episode_step_number = 0
        self._init_env(test_case=0)
        return self._get_obs()
        # TODO: confirm agent is getting correct scan on reset

    def render(self, mode='human', close=False):
        if not Config.ANIMATE_EPISODES:
            return
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_x - self.min_x
        world_height = self.max_y - self.min_y
        scale_x = screen_width/world_width
        scale_y = screen_height/world_height

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        if self.begin_episode:
            self.begin_episode = False
            self.goaltrans = []
            self.agenttrans = []
            self.viewer.geoms = []
            '''
            xs = np.linspace(self.min_dist_to_goal, self.max_dist_to_goal, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_dist_to_goal)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            '''

            for i, agent in enumerate(self.agents):
                goal_icon = rendering.make_circle(10)
                goal_icon.add_attr(rendering.Transform(translation=(0, 10)))
                self.goaltrans.append(rendering.Transform())
                goal_icon.add_attr(self.goaltrans[i])
                goal_icon.set_color(plt_colors[i][0],
                                    plt_colors[i][1],
                                    plt_colors[i][2])
                self.viewer.add_geom(goal_icon)

                agent_icon = rendering.make_circle(scale_x*agent.radius)
                agent_icon.set_color(plt_colors[i][0],
                                     plt_colors[i][1],
                                     plt_colors[i][2])
                agent_icon.add_attr(rendering.Transform(translation=(0, 0)))
                self.agenttrans.append(rendering.Transform())
                agent_icon.add_attr(self.agenttrans[i])
                self.viewer.add_geom(agent_icon)

            # flagx = (agent.dist_to_goal-self.min_position)*scale
            # flagy1 = self._height(self.goal_position)*scale
            # flagy2 = flagy1 + 50
            # flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            # self.viewer.add_geom(flagpole)

        #  flag = rendering.FilledPolygon([(flagx, flagy2),
        #                                 (flagx, flagy2-10),
        #                                 (flagx+25, flagy2-5)])
        #     flag.set_color(.8,.8,0)
        #     self.viewer.add_geom(flag)

        else:
            for i, agent in enumerate(self.agents):
                self.goaltrans[i].set_translation(
                    (agent.goal_global_frame[0] - self.min_x) * scale_x,
                    (agent.goal_global_frame[1] - self.min_y) * scale_y)
                self.agenttrans[i].set_translation(
                    (agent.pos_global_frame[0] - self.min_x) * scale_x,
                    (agent.pos_global_frame[1] - self.min_y) * scale_y)

                agent_traj = rendering.make_circle(1)
                agent_traj.add_attr(rendering.Transform(translation=(0, 0)))
                agent_traj.set_color(plt_colors[i][0],
                                     plt_colors[i][1],
                                     plt_colors[i][2])
                agenttrans = rendering.Transform()
                agent_traj.add_attr(agenttrans)
                agenttrans.set_translation(
                    (agent.pos_global_frame[0] - self.min_x) * scale_x,
                    (agent.pos_global_frame[1] - self.min_y) * scale_y)
                self.viewer.add_geom(agent_traj)

        rgb_array = mode == 'rgb_array'
        return self.viewer.render(return_rgb_array=rgb_array)

    def _plot_episode(self):
        
        fig = plt.figure(self.id, figsize=(10, 8))
        plt.clf()

        plt_colors = []
        plt_colors.append([0.8500, 0.3250, 0.0980])  # red
        plt_colors.append([0.0, 0.4470, 0.7410])  # blue
        plt_colors.append([0.4660, 0.6740, 0.1880])  # green
        plt_colors.append([0.4940, 0.1840, 0.5560])  # purple
        plt_colors.append([0.9290, 0.6940, 0.1250])  # orange
        plt_colors.append([0.3010, 0.7450, 0.9330])  # cyan
        plt_colors.append([0.6350, 0.0780, 0.1840])  # chocolate

        ax = fig.add_subplot(1, 1, 1)

        img = plt.imread("/home/mfe/code/Stage/worlds/bitmaps/mfe_hallway.png")
        ax.imshow(img, extent=[-8,8,-8,8])

        max_time = max(
                [agent.global_state_history[-1, 0] for agent in self.agents])
        max_time_alpha_scalar = 1.2
        for i, agent in enumerate(self.agents):
            if agent.global_state_history.ndim == 1:
                # if there's only 1 timestep in state history
                agent.global_state_history = np.expand_dims(
                        agent.global_state_history, axis=0)
                agent.body_state_history = np.expand_dims(
                        agent.body_state_history, axis=0)
            color_ind = i % len(plt_colors)
            plt_color = plt_colors[color_ind]
            plt.plot(agent.global_state_history[:, 1],
                     agent.global_state_history[:, 2],
                     color=plt_color, ls='-', linewidth=2)
            plt.plot(agent.global_state_history[0, 3],
                     agent.global_state_history[0, 4],
                     color=plt_color, marker='*', markersize=20)

            # Display circle at agent pos every circle_spacing (nom 1.5 sec)
            circle_spacing = 0.4
            circle_times = np.arange(0.0, agent.global_state_history[-1, 0],
                                     circle_spacing)
            _, circle_inds = find_nearest(agent.global_state_history[:, 0],
                                          circle_times)
            for ind in circle_inds:
                alpha = 1 - \
                        agent.global_state_history[ind, 0] / \
                        (max_time_alpha_scalar*max_time)
                c = rgba2rgb(plt_color+[float(alpha)])
                ax.add_patch(plt.Circle(agent.global_state_history[ind, 1:3],
                             radius=agent.radius, fc=c, ec=plt_color,
                             fill=True))

            text_spacing = 1.5
            text_times = np.arange(0.0, agent.global_state_history[-1, 0],
                                   text_spacing)
            _, text_inds = find_nearest(agent.global_state_history[:, 0],
                                        text_times)
            for ind in text_inds:
                y_text_offset = 0.1
                alpha = agent.global_state_history[ind, 0] / \
                    (max_time_alpha_scalar*max_time)
                if alpha < 0.5:
                    alpha = 0.3
                else:
                    alpha = 0.9
                c = rgba2rgb(plt_color+[float(alpha)])
                ax.text(agent.global_state_history[ind, 1]-0.15,
                        agent.global_state_history[ind, 2]+y_text_offset,
                        '%.1f' % agent.global_state_history[ind, 0], color=c)
            # Also display circle at agent position at end of trajectory
            ind = -1
            alpha = 1 - \
                agent.global_state_history[ind, 0] / \
                (max_time_alpha_scalar*max_time)
            c = rgba2rgb(plt_color+[float(alpha)])
            ax.add_patch(plt.Circle(agent.global_state_history[ind, 1:3],
                         radius=agent.radius, fc=c, ec=plt_color))
            y_text_offset = 0.1
            ax.text(agent.global_state_history[ind, 1] - 0.15,
                    agent.global_state_history[ind, 2] + y_text_offset,
                    '%.1f' % agent.global_state_history[ind, 0],
                    color=plt_color)

        # title_string = "Episode"
        # plt.title(title_string)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.axis('equal')

        # plotting style (only show axis on bottom and left)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        plt.draw()
        if self.evaluate:
            fig_dir = '/home/mfe/code/'\
                    'openai_baselines/baselines/ppo2/logs/test_cases/'
            fig_name = self.agents[0].policy_type + '_' + \
                str(self.test_case_num_agents) + 'agents_' + \
                str(self.test_case_index) + '.png'
            plt.savefig(fig_dir+fig_name)
        plt.pause(0.0001)
        # plt.pause(5.0)


plt_colors = []
plt_colors.append([0.8500, 0.3250, 0.0980])  # red
plt_colors.append([0.0, 0.4470, 0.7410])  # blue
plt_colors.append([0.4660, 0.6740, 0.1880])  # green
plt_colors.append([0.4940, 0.1840, 0.5560])  # purple
plt_colors.append([0.9290, 0.6940, 0.1250])  # orange
plt_colors.append([0.3010, 0.7450, 0.9330])  # cyan
plt_colors.append([0.6350, 0.0780, 0.1840])  # chocolate


def preset_testCases(test_case_num_agents, full_test_suite=False):
    if full_test_suite:
        num_test_cases = 500
        test_cases = pickle.load(open(
            "/home/mfe/ford_ws/src/2017-avrl/src/environment/\
            Collision-Avoidance/test_cases/%s_agents_%i_cases_short.p"
            % (test_case_num_agents, num_test_cases), "rb"))
    else:

        if test_case_num_agents == 2:
            test_cases = []
            # fixed speed and radius
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0/1.4, -3.0/1.4, -3.0/1.4, 3.0/1.4, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-2.0, -1.5, 2.0, 1.5, 1.0, 0.5],
                [-2.0, 1.5, 2.0, -1.5, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [0.0, -3.0, 0.0, 3.0, 1.0, 0.5]
                ]))
            # variable speed and radius
            test_cases.append(np.array([
                [-2.5, 0.0, 2.5, 0.0, 1.0, 0.3],
                [2.5, 0.0, -2.5, 0.0, 0.8, 0.4]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 0.6, 0.5],
                [3.0/1.4, -3.0/1.4, -3.0/1.4, 3.0/1.4, 1.0, 0.4]
                ]))
            test_cases.append(np.array([
                [-2.0, 0.0, 2.0, 0.0, 0.9, 0.35],
                [2.0, 0.0, -2.0, 0.0, 0.85, 0.45]
                ]))
            test_cases.append(np.array([
                [-4.0, 0.0, 4.0, 0.0, 1.0, 0.4],
                [-2.0, 0.0, 2.0, 0.0, 0.5, 0.4]
                ]))

        elif test_case_num_agents == 3 or test_case_num_agents == 4:
            test_cases = []
            # hardcoded to be 3 agents for now
            d = 3.0
            l1 = d*np.cos(np.pi/6)
            l2 = d*np.sin(np.pi/6)
            test_cases.append(np.array([
                [0.0, d, 0.0, -d, 1.0, 0.5],
                [l1, -l2, -l1, l2, 1.0, 0.5],
                [-l1, -l2, l1, l2, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, 1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, 1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, -1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, 1.5, 1.0, 0.5]
                ]))
            # hardcoded to be 4 agents for now
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.3],
                [3.0, -1.5, -3.0, -1.5, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3],
                [-3.0, -3.0, 3.0, -3.0, 1.0, 0.3],
                [3.0, -3.0, -3.0, -3.0, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [0.0, -3.0, 0.0, 3.0, 1.0, 0.5],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.5],
                [0.0, 3.0, 0.0, -3.0, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [-2.0, -1.5, 2.0, 1.5, 1.0, 0.5],
                [-2.0, 1.5, 2.0, -1.5, 1.0, 0.5],
                [-2.0, -4.0, 2.0, -4.0, 0.9, 0.35],
                [2.0, -4.0, -2.0, -4.0, 0.85, 0.45]
                ]))
            test_cases.append(np.array([
                [-4.0, 0.0, 4.0, 0.0, 1.0, 0.4],
                [-2.0, 0.0, 2.0, 0.0, 0.5, 0.4],
                [-4.0, -4.0, 4.0, -4.0, 1.0, 0.4],
                [-2.0, -4.0, 2.0, -4.0, 0.5, 0.4]
                ]))

        elif test_case_num_agents == 5:
            test_cases = []

            radius = 4
            tc = gen_circle_test_case(test_case_num_agents, radius)
            test_cases.append(tc)

            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, 1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.5],
                [-3.0, 3.0, 3.0, 3.0, 1.0, 0.5],
                [-3.0, -3.0, 3.0, -3.0, 1.0, 0.5]
                ]))

        elif test_case_num_agents == 6:
            test_cases = []

            radius = 5
            tc = gen_circle_test_case(test_case_num_agents, radius)
            test_cases.append(tc)

            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.5],
                [-3.0, 1.5, 3.0, 1.5, 1.0, 0.5],
                [-3.0, -1.5, 3.0, -1.5, 1.0, 0.5],
                [-3.0, 3.0, 3.0, 3.0, 1.0, 0.5],
                [-3.0, -3.0, 3.0, -3.0, 1.0, 0.5],
                [-3.0, -4.5, 3.0, -4.5, 1.0, 0.5]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3],
                [-3.0, 0.7, 3.0, 0.7, 1.0, 0.3],
                [3.0, 0.7, -3.0, 0.7, 1.0, 0.3],
                [-3.0, -0.7, 3.0, -0.7, 1.0, 0.3],
                [3.0, -0.7, -3.0, -0.7, 1.0, 0.3]
                ]))
            test_cases.append(np.array([
                [-3.0, 0.0, 3.0, 0.0, 1.0, 0.3],
                [3.0, 0.0, -3.0, 0.0, 1.0, 0.3],
                [-3.0, 1.0, 3.0, 1.0, 1.0, 0.3],
                [3.0, 1.0, -3.0, 1.0, 1.0, 0.3],
                [-3.0, -1.0, 3.0, -1.0, 1.0, 0.3],
                [3.0, -1.0, -3.0, -1.0, 1.0, 0.3]
                ]))

        elif test_case_num_agents == 10:
            test_cases = []

            radius = 7
            tc = gen_circle_test_case(test_case_num_agents, radius)
            test_cases.append(tc)

        elif test_case_num_agents == 20:
            test_cases = []

            radius = 10
            tc = gen_circle_test_case(test_case_num_agents, radius)
            test_cases.append(tc)

        else:
            print("[preset_testCases in Collision_Avoidance.py]\
                    invalid test_case_num_agents")
            assert(0)
    return test_cases


def gen_circle_test_case(num_agents, radius):
    tc = np.zeros((num_agents, 6))
    for i in range(num_agents):
        tc[i, 4] = 1.0
        tc[i, 5] = 0.5
        theta_start = (2*np.pi/num_agents)*i
        theta_end = theta_start + np.pi
        tc[i, 0] = radius*np.cos(theta_start)
        tc[i, 1] = radius*np.sin(theta_start)
        tc[i, 2] = radius*np.cos(theta_end)
        tc[i, 3] = radius*np.sin(theta_end)
    return tc
