'''
Collision Avoidance Environement
Author: Michael Everett
MIT Aerospace Controls Lab
'''

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import matplotlib.colors

from gym.envs.classic_control import rendering
from pyglet import gl

from gym_collision_avoidance.envs.agent import Agent
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.util import *

class CollisionAvoidanceEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        # Initialize Rewards
        self.reward_at_goal = Config.REWARD_AT_GOAL
        self.reward_collision = Config.REWARD_COLLISION
        self.reward_getting_close = Config.REWARD_GETTING_CLOSE
            
        # Simulation Parameters
        self.num_agents      = Config.MAX_NUM_AGENTS
        self.dt              = Config.DT

        # Collision Parameters
        self.collision_dist  = Config.COLLISION_DIST
        self.getting_close_range  = Config.GETTING_CLOSE_RANGE

        self.evaluate        = Config.EVALUATE_MODE
        self.plot_episodes   = Config.PLOT_EPISODES
        self.test_case       = 0

        self.display_screen = Config.DISPLAY_SCREEN

        self.first = True
        
        self.min_x = -10.0
        self.max_x = 10.0
        self.min_y = -10.0
        self.max_y = 10.0
        self.min_dist_to_goal = 0.0
        self.max_dist_to_goal = 10.0
        self.min_heading = -np.pi
        self.max_heading = np.pi
        self.min_pref_speed = 0.0
        self.max_pref_speed = 2.0
        self.min_radius = 0.3
        self.max_radius = 2.0

        
        self.low_state = np.zeros((Config.FULL_STATE_LENGTH))
        self.high_state = np.zeros((Config.FULL_STATE_LENGTH))
        # self.low_state = np.array([self.min_dist_to_goal, self.min_heading, self.min_pref_speed, self.min_radius])
        # self.high_state = np.array([self.max_dist_to_goal, self.max_heading, self.max_pref_speed, self.max_radius])

        self.viewer = None

        self.max_heading_change = np.pi/3
        self.min_heading_change = -self.max_heading_change
        self.min_speed = 0.0
        self.max_speed = 1.0

        self.action_space_type = Config.ACTION_SPACE_TYPE
        if self.action_space_type == Config.discrete:
            self.action_space = spaces.Discrete(self.actions.num_actions)
        elif self.action_space_type == Config.continuous:
            self.low_action = np.array([self.min_speed, self.min_heading_change])
            self.high_action = np.array([self.max_speed, self.max_heading_change])
            self.action_space = spaces.Box(self.low_action, self.high_action)
        self.observation_space = spaces.Box(self.low_state, self.high_state)

        self.reset()

    def _step(self, action):
        # TODO: make sure actions are valid
        # assert self.action_space.contains(np.tanh(action)), "%r (%s) invalid" % (action, type(action))
        self._take_action(action)

        reward = -0.01
        collision = False
        agent = self.agents[0]
        if agent.is_at_goal and agent.was_at_goal_already == False:
            reward = self.reward_at_goal
        else:
            dist_between = np.inf
            for other_agent in self.agents:
                if agent.id != other_agent.id:
                    is_collision, dist_between = self.check_collision(agent, other_agent) 
                    agent.min_dist_to_other_agents = min(agent.min_dist_to_other_agents, dist_between)
                    if is_collision and agent.was_in_collision_already == False:
                        reward = self.reward_collision
                        collision = True
                        agent.in_collision = True
        #             # elif dist_between <= self.getting_close_range:
        #             #     rewards[i] = min(self.reward_getting_close + 0.5*dist_between, rewards[i])
        reward = np.clip(reward, self.reward_collision, self.reward_at_goal)

        # # Reward
        # rewards = -0.01*np.ones(self.num_agents)
        # collision = False
        # for i,agent in enumerate(self.agents):
        #     if agent.is_at_goal and agent.was_at_goal_already == False:
        #         rewards[i] = self.reward_at_goal
        #     else:
        #         dist_between = np.inf
        #         # for other_agent in self.agents:
        #             # if agent.id != other_agent.id:
        #             #     is_collision, dist_between = self.check_collision(agent, other_agent) 
        #             #     agent.min_dist_to_other_agents = min(agent.min_dist_to_other_agents, dist_between)
        #             #     if is_collision and agent.was_in_collision_already == False:
        #             #         rewards[i] = self.reward_collision
        #             #         collision = True
        #             #         agent.in_collision = True
        #             #     # elif dist_between <= self.getting_close_range:
        #             #     #     rewards[i] = min(self.reward_getting_close + 0.5*dist_between, rewards[i])
        # rewards = np.clip(rewards, self.reward_collision, self.reward_at_goal)
        # # print rewards

        next_observations = self._get_obs()
        # Game over
        at_goal_condition = np.array([a.is_at_goal for a in self.agents])
        ran_out_of_time_condition = np.array([a.ran_out_of_time for a in self.agents])
        ran_out_of_time_condition = np.array([a.ran_out_of_time for a in self.agents])
        in_collision_condition = np.array([a.in_collision for a in self.agents])
        which_agents_done = np.logical_or(np.logical_or(at_goal_condition,ran_out_of_time_condition),in_collision_condition)
        game_over = np.all(which_agents_done)
        done = which_agents_done[0]
        # if done:
        #     agent = self.agents[0]
        #     if agent.is_at_goal: print("------ Made it to goal!! ------")
        #     if agent.in_collision: print("collision")
        #     if agent.ran_out_of_time: print("ran out of time")
        # print 'which_agents_done:', which_agents_done
        # print 'game_over:', game_over
        return next_observations, reward, done, {}
        # return next_observations, rewards, which_agents_done, game_over, {}

        # return next_observation, reward, done, {}

    def _get_obs(self):
        # obs = self.agents[0].observe(self.agents)
        # return obs
        next_observations = np.empty([len(self.agents), Config.FULL_STATE_LENGTH])
        for i, agent in enumerate(self.agents):
            agent_obs = agent.observe(self.agents)
            next_observations[i] = agent_obs
        return next_observations

    def _take_action(self, actions):
        # print("[env] raw actions:", actions)
        action_vectors = [np.array([0.0, 0.0]) for i in range(self.num_agents)]
        # First find next action of each agent -- Importantly done before agents update their state, mostly only impt for CADRL
        agent = self.agents[0]
        for i, agent in enumerate(self.agents):
            heading = (2.0*self.max_heading_change*actions[i,1]) - self.max_heading_change
            speed = agent.pref_speed * actions[i,0]
            if agent.is_at_goal or agent.in_collision:
                action_vector = np.array([0.0, 0.0]) # TODO Confirm this works?
            elif agent.policy_type == 'CADRL':
                action_vector = agent.find_next_action(self.agents)
            elif agent.policy_type == 'DQN': # TODO: Fix this up
                action_vector = np.array([agent.pref_speed, action])
            elif agent.policy_type == 'PPO':
                action_vector = np.array([speed, heading])
            else:
                print("INVALID AGENT POLICY TYPE")
                assert(0)
            action_vectors[i] = action_vector

        # ...then update their states based on chosen actions
        for i, agent in enumerate(self.agents):
            agent.update_state(action_vectors[i], self.dt)

    def _reset(self):
        self.begin_episode = True
        # print("RESET")
        # if self.first == True:
        #     self.first = False
        # else:
        #     self._plot_episode()
        other_x = np.random.choice([-1,1])*np.random.uniform(1.5,3)
        other_y = np.random.choice([-1,1])*np.random.uniform(1.5,3)

        goal_x = np.random.choice([-1,1])*np.random.uniform(1,7)
        goal_y = np.random.uniform(-2,2)
        initial_heading = np.random.uniform(-np.pi, np.pi)
        self.agents = np.array([Agent(0,0,goal_x,goal_y,0.5,1.0,initial_heading,0), Agent(goal_x,goal_y,0,0,0.5,1.0,np.pi, 1)])

        return self._get_obs()

    def _render(self, mode='human', close=False):
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
            # xs = np.linspace(self.min_dist_to_goal, self.max_dist_to_goal, 100)
            # ys = self._height(xs)
            # xys = list(zip((xs-self.min_dist_to_goal)*scale, ys*scale))


            # self.track = rendering.make_polyline(xys)
            # self.track.set_linewidth(4)
            # self.viewer.add_geom(self.track)

            # clearance = 10

            # l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            # car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            # car.add_attr(rendering.Transform(translation=(0, clearance)))
            # self.cartrans = rendering.Transform()
            # car.add_attr(self.cartrans)
            # self.viewer.add_geom(car)

            for i, agent in enumerate(self.agents):
                goal_icon = rendering.make_circle(10)
                goal_icon.add_attr(rendering.Transform(translation=(0, 10)))
                self.goaltrans.append(rendering.Transform())
                goal_icon.add_attr(self.goaltrans[i])
                goal_icon.set_color(plt_colors[i][0],plt_colors[i][1],plt_colors[i][2])
                self.viewer.add_geom(goal_icon)

                agent_icon = rendering.make_circle(scale_x*agent.radius)
                agent_icon.set_color(plt_colors[i][0],plt_colors[i][1],plt_colors[i][2])
                agent_icon.add_attr(rendering.Transform(translation=(0, 0)))
                self.agenttrans.append(rendering.Transform())
                agent_icon.add_attr(self.agenttrans[i])
                self.viewer.add_geom(agent_icon)

            # flagx = (agent.dist_to_goal-self.min_position)*scale
            # flagy1 = self._height(self.goal_position)*scale
            # flagy2 = flagy1 + 50
            # flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            # self.viewer.add_geom(flagpole)


        #     flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
        #     flag.set_color(.8,.8,0)
        #     self.viewer.add_geom(flag)

        else:
            for i, agent in enumerate(self.agents):
                self.goaltrans[i].set_translation((agent.goal_global_frame[0]-self.min_x)*scale_x, (agent.goal_global_frame[1]-self.min_y)*scale_y)
                self.agenttrans[i].set_translation((agent.pos_global_frame[0]-self.min_x)*scale_x, (agent.pos_global_frame[1]-self.min_y)*scale_y)

                agent_traj = rendering.make_circle(1)
                agent_traj.add_attr(rendering.Transform(translation=(0, 0)))
                agent_traj.set_color(plt_colors[i][0],plt_colors[i][1],plt_colors[i][2])
                agenttrans = rendering.Transform()
                agent_traj.add_attr(agenttrans)
                agenttrans.set_translation((agent.pos_global_frame[0]-self.min_x)*scale_x, (agent.pos_global_frame[1]-self.min_y)*scale_y)
                self.viewer.add_geom(agent_traj)
        # dist_to_goal = agent.dist_to_goal
        # self.agenttrans.set_translation((dist_to_goal-self.min_dist_to_goal)*scale, screen_height/2)
        # self.goaltrans.set_translation(100, 200)
        # self.cartrans.set_translation((dist_to_goal-self.min_dist_to_goal)*scale, self._height(pos)*scale)
        # self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')



    def check_collision(self, agent_a, agent_b):
        dist_between = self.dist_euclid(agent_a.pos_global_frame, agent_b.pos_global_frame) - agent_a.radius - agent_b.radius
        is_collision = dist_between <= self.collision_dist
        return is_collision, dist_between

    def dist_manhat(self, loc_a, loc_b):
        return abs(loc_a[0] - loc_b[0]) + abs(loc_a[1] - loc_b[1])

    def dist_euclid(self, loc_a, loc_b):
       return np.linalg.norm(loc_a - loc_b)

    def _plot_episode(self):
        print("plotting ep")
        fig = plt.figure(figsize=(10, 8))

        plt_colors = []
        plt_colors.append([0.8500, 0.3250, 0.0980]) # red
        plt_colors.append([0.0, 0.4470, 0.7410]) # blue 
        plt_colors.append([0.4660, 0.6740, 0.1880]) # green 
        plt_colors.append([0.4940, 0.1840, 0.5560]) # purple
        plt_colors.append([0.9290, 0.6940, 0.1250]) # orange 
        plt_colors.append([0.3010, 0.7450, 0.9330]) # cyan 
        plt_colors.append([0.6350, 0.0780, 0.1840]) # chocolate 

        ax = fig.add_subplot(1, 1, 1)


        max_time = max([agent.global_state_history[-1,0] for agent in self.agents])
        max_time_alpha_scalar = 1.2
        for i, agent in enumerate(self.agents):
            if agent.global_state_history.ndim == 1: # if there's only 1 timestep in state history
                agent.global_state_history = np.expand_dims(agent.global_state_history, axis=0)
                agent.body_state_history = np.expand_dims(agent.body_state_history, axis=0)
            color_ind = i % len(plt_colors)
            plt_color = plt_colors[color_ind]
            plt.plot(agent.global_state_history[:,1], agent.global_state_history[:,2],\
                color=plt_color, ls='-', linewidth=2)
            plt.plot(agent.global_state_history[0,3], agent.global_state_history[0,4],\
                color=plt_color, marker='*', markersize=20)

            # Display circle at agent position every circle_spacing (nominally 1.5 sec)
            circle_spacing = 0.4
            circle_times = np.arange(0.0,agent.global_state_history[-1,0],circle_spacing)
            _, circle_inds = find_nearest(agent.global_state_history[:,0], circle_times)
            # circle_inds = np.where((agent.global_state_history[:,0] % circle_spacing < Config.DT) | (agent.global_state_history[:,0] % circle_spacing > circle_spacing - Config.DT))[0]
            for ind in circle_inds:
                alpha = (1-agent.global_state_history[ind,0]/(max_time_alpha_scalar*max_time))
                c = rgba2rgb(plt_color+[float(alpha)])
                ax.add_patch( plt.Circle(agent.global_state_history[ind,1:3], \
                        radius=agent.radius, fc=c, ec=plt_color, fill=True) )
            
            text_spacing = 1.5
            text_times = np.arange(0.0,agent.global_state_history[-1,0],text_spacing)
            _, text_inds = find_nearest(agent.global_state_history[:,0], text_times)
            for ind in text_inds:
                y_text_offset = 0.1
                alpha = (agent.global_state_history[ind,0]/(max_time_alpha_scalar*max_time))
                if alpha < 0.5: alpha = 0.3
                else: c = alpha = 0.9
                c = rgba2rgb(plt_color+[float(alpha)])
                ax.text(agent.global_state_history[ind,1]-0.15, agent.global_state_history[ind,2]+y_text_offset, \
                            '%.1f'%agent.global_state_history[ind,0], color=c)
            # Also display circle at agent position at end of trajectory
            ind = -1
            alpha = (1-agent.global_state_history[ind,0]/(max_time_alpha_scalar*max_time))
            c = rgba2rgb(plt_color+[float(alpha)])
            ax.add_patch( plt.Circle(agent.global_state_history[ind,1:3], \
                        radius=agent.radius, fc=c, ec=plt_color) )
            y_text_offset = 0.1
            ax.text(agent.global_state_history[ind,1]-0.15, agent.global_state_history[ind,2]+y_text_offset, \
                            '%.1f'%agent.global_state_history[ind,0], color=plt_color)
            
            # arrow_times = np.arange(circle_spacing/2.0,agent.global_state_history[-2,0],circle_spacing)
            # _, arrow_inds = util.find_nearest(agent.global_state_history[:,0], arrow_times)
            # num_arrows = np.shape(arrow_inds)[0]
            # num_pts = np.shape(agent.global_state_history)[0]
            # if num_arrows == 0:
            #     arrow_inds = [min(int(num_pts/2.0),num_pts-2)]
            # arrow_length = 0.3
            # for ind in arrow_inds:
            #     arrow_start = agent.global_state_history[ind,1:3]
            #     arrow_end = agent.global_state_history[ind+1,1:3]
            #     arrow_dxdy = arrow_end - arrow_start
            #     arrow_dxdy = arrow_dxdy / (np.linalg.norm(arrow_dxdy) / arrow_length)
            #     arrow_end = arrow_start + arrow_dxdy
            #     style="Wedge,tail_width=20"
            #     ax.add_patch( ptch.FancyArrowPatch(arrow_start, arrow_end, arrowstyle=style, color=plt_color))

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
        # if self.evaluate:
        #     fig_dir = 'logs/test_cases/'
        #     fig_name = self.agents[0].policy_type+'_'+str(self.test_case_num_agents)+'agents_'+str(self.test_case)+'.png'
        #     plt.savefig(fig_dir+fig_name)
        plt.pause(0.0001)
        # plt.pause(5.0)


plt_colors = []
plt_colors.append([0.8500, 0.3250, 0.0980]) # red
plt_colors.append([0.0, 0.4470, 0.7410]) # blue 
plt_colors.append([0.4660, 0.6740, 0.1880]) # green 
plt_colors.append([0.4940, 0.1840, 0.5560]) # purple
plt_colors.append([0.9290, 0.6940, 0.1250]) # orange 
plt_colors.append([0.3010, 0.7450, 0.9330]) # cyan 
plt_colors.append([0.6350, 0.0780, 0.1840]) # chocolate 

