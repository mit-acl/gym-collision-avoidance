"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class Actions():
    def __init__(self):
        # self.actions = np.linspace(-np.pi/3, np.pi/3, 9)
        self.actions = np.mgrid[1.0:1.1:0.5, -np.pi/3:np.pi/3+0.01:np.pi/9].reshape(2, -1).T
        self.actions = np.vstack([self.actions,np.mgrid[0.5:0.6:0.5, -np.pi/3:np.pi/3+0.01:np.pi/6].reshape(2, -1).T])
        self.actions = np.vstack([self.actions, np.array([0.0,0.0])])

        # self.actions = np.mgrid[0.5:1.1:0.5, -np.pi/3:np.pi/3+0.01:np.pi/9].reshape(2, -1).T
        # self.actions = np.mgrid[0.25:1.1:0.25, -np.pi/3:np.pi/3+0.01:np.pi/9].reshape(2, -1).T
        
        self.num_actions = len(self.actions)

class CollisionAvoidanceEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        # Initialize Rewards
        reward_at_goal = 1.0
        reward_collision = -0.25
        num_agents = 2
        dt = 0.2
        self.reward_at_goal = reward_at_goal
        self.reward_collision = reward_collision
        # self.reward_getting_close = reward_getting_close
        
        self.actions         = Actions()

        # Simulation Parameters
        self.num_agents      = num_agents
        self.dt              = dt
        self.prev_action     = 0

        # Collision Parameters
        collision_dist = -10000
        self.collision_dist  = collision_dist
        # self.getting_close_range  = getting_close_range

        # self.evaluate        = evaluate
        # self.plot_episodes   = plot_episodes

        self.min_x = -10.0
        self.max_x = 10.0
        self.min_y = -10.0
        self.max_y = 10.0
        self.min_dist_to_goal = 0.0
        self.max_dist_to_goal = 10.0
        self.min_heading = -np.pi
        self.max_heading = np.pi

        self.low_state = np.array([self.min_dist_to_goal, self.min_heading])
        self.high_state = np.array([self.max_dist_to_goal, self.max_heading])

        self.viewer = None

        self.min_action = -1.0
        self.max_action = 1.0

        self.action_space = spaces.Discrete(self.actions.num_actions)
        # self.action_space = spaces.Box(self.min_action, self.max_action, shape = (1,))
        # self.action_space = spaces.Box(self.min_action, self.max_action, shape = (1,))
        self.observation_space = spaces.Box(self.low_state, self.high_state)

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Take action for each agent
        # for i, action in enumerate(actions):
        agent = self.agents[0]
        if agent.is_at_goal or agent.in_collision:
            action_vector = np.array([0.0, 0.0]) # TODO Confirm this works?
        else:
            speed_multiplier, selected_heading = self.actions.actions[action]
            action_vector = np.array([agent.pref_speed*speed_multiplier, selected_heading])
        agent.update_state(action_vector, self.dt)

        # Reward
        reward = -0.01
        collision = False
        agent = self.agents[0]
        if agent.is_at_goal:
            reward = self.reward_at_goal
        else:
            dist_between = np.inf
            for other_agent in self.agents:
                if agent.id != other_agent.id:
                    is_collision, dist_between = self.check_collision(agent, other_agent) 
                    agent.min_dist_to_other_agents = min(agent.min_dist_to_other_agents, dist_between)
                    if is_collision:
                        reward = self.reward_collision
                        collision = True
                        agent.in_collision = True
        reward = np.clip(reward, self.reward_collision, self.reward_at_goal)

        done = False

        at_goal_condition = self.agents[0].is_at_goal
        ran_out_of_time_condition = self.agents[0].ran_out_of_time
        done = at_goal_condition or ran_out_of_time_condition

        # reward = -1.0

        next_observation = self.agents[0].observe(self.agents)

        return next_observation, reward, done, {}

    def _reset(self):
        goal_x = np.random.choice([-1,1])*np.random.uniform(-7,7)
        goal_y = np.random.uniform(-5,5)
        initial_heading = np.random.uniform(-np.pi, np.pi)
        self.agents = np.array([Agent(0,0,goal_x,goal_y,0.5,1.0,initial_heading,0), Agent(goal_x,goal_y+5,0,5,0.5,1.0,np.pi, 1)])
        return self.agents[0].observe(self.agents)

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


        agent = self.agents[0]
        # if agent.t = 0.0:
            # from gym.envs.classic_control import rendering
            

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
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

            for agent in self.agents:
                goal_icon = rendering.make_circle(10)
                goal_icon.add_attr(rendering.Transform(translation=(0, 10)))
                self.goaltrans = rendering.Transform()
                goal_icon.add_attr(self.goaltrans)
                goal_icon.set_color(.8,.8,0)
                print(agent.goal_global_frame)
                self.viewer.add_geom(goal_icon)

                agent_icon = rendering.make_circle(20)
                agent_icon.set_color(.8,.8,0)
                agent_icon.add_attr(rendering.Transform(translation=(0, 0)))
                self.agenttrans = rendering.Transform()
                agent_icon.add_attr(self.agenttrans)
                self.viewer.add_geom(agent_icon)

            # flagx = (agent.dist_to_goal-self.min_position)*scale
            # flagy1 = self._height(self.goal_position)*scale
            # flagy2 = flagy1 + 50
            # flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            # self.viewer.add_geom(flagpole)


        #     flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
        #     flag.set_color(.8,.8,0)
        #     self.viewer.add_geom(flag)

        self.goaltrans.set_translation((agent.goal_global_frame[0]-self.min_x)*scale_x, (agent.goal_global_frame[1]-self.min_y)*scale_y)
        self.agenttrans.set_translation((agent.pos_global_frame[0]-self.min_x)*scale_x, (agent.pos_global_frame[1]-self.min_y)*scale_y)

        agent_traj = rendering.make_circle(5)
        agent_traj.add_attr(rendering.Transform(translation=(0, 0)))
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



class Agent():
    def __init__(self, start_x, start_y, goal_x, goal_y, radius, pref_speed, initial_heading, id):

        self.policy_type = "A3C"

        # Global Frame states
        self.pos_global_frame = np.array([start_x, start_y], dtype='float64')
        self.goal_global_frame = np.array([goal_x, goal_y], dtype='float64')
        self.vel_global_frame = np.array([0.0, 0.0], dtype='float64')
        self.speed_global_frame = 0.0 
        self.heading_global_frame = initial_heading
        
        # Ego Frame states
        self.speed_ego_frame = 0.0
        self.heading_ego_frame = 0.0 
        self.vel_ego_frame = np.array([0.0, 0.0])
        self.goal_ego_frame = np.array([0.0, 0.0]) # xy coords of goal position
        
        # Other parameters
        self.radius = radius
        self.pref_speed = pref_speed
        self.id = id
        self.dist_to_goal = 0.0

        self.time_remaining_to_reach_goal = 2*np.linalg.norm(self.pos_global_frame - self.goal_global_frame)/self.pref_speed
        self.t = 0.0

        self.is_at_goal = False
        self.in_collision = False
        self.ran_out_of_time = False

        self.ref_prll, self.ref_orth = self.get_ref()
        self.global_state_history, self.ego_state_history = self.to_vector()
        self.global_state_history = np.hstack([self.t, self.global_state_history])

        self.min_dist_to_other_agents = np.inf


    def _check_if_at_goal(self):
        near_goal_threshold = 0.5
        is_near_goal = np.linalg.norm([self.pos_global_frame - self.goal_global_frame]) <= near_goal_threshold
        self.is_at_goal = is_near_goal

    def update_state(self, action, dt):
        if self.is_at_goal or self.ran_out_of_time:
            self.vel_global_frame = np.array([0.0, 0.0])
            return

        selected_speed = action[0]*self.pref_speed
        selected_heading = wrap(action[1] + self.heading_global_frame) # in global frame

        dx = selected_speed * np.cos(selected_heading) * dt
        dy = selected_speed * np.sin(selected_heading) * dt
        self.pos_global_frame += np.array([dx, dy])
        self.vel_global_frame[0] = selected_speed * np.cos(selected_heading)
        self.vel_global_frame[1] = selected_speed * np.sin(selected_heading)
        self.speed_global_frame = selected_speed
        self.heading_global_frame = selected_heading

        # Compute heading w.r.t. ref_prll, ref_orthog coordinate axes
        self.ref_prll, self.ref_orth = self.get_ref()
        ref_prll_angle_global_frame = np.arctan2(self.ref_prll[1], self.ref_prll[0])
        self.heading_ego_frame = wrap(self.heading_global_frame - ref_prll_angle_global_frame)

        # Compute velocity w.r.t. ref_prll, ref_orthog coordinate axes
        cur_speed = np.linalg.norm(self.vel_global_frame)
        v_prll = cur_speed * np.cos(self.heading_ego_frame)
        v_orthog = cur_speed * np.sin(self.heading_ego_frame)    
        self.vel_ego_frame = np.array([v_prll, v_orthog])

        # Update time left so agent does not run around forever
        self.time_remaining_to_reach_goal -= dt
        self.t += dt
        if self.time_remaining_to_reach_goal <= 0.0:
            self.ran_out_of_time = True

        self._update_state_history()

        self._check_if_at_goal()

        return

    def _update_state_history(self):
        global_state, ego_state = self.to_vector()
        self.global_state_history = np.vstack([self.global_state_history, np.hstack([self.t, global_state])])
        self.ego_state_history = np.vstack([self.ego_state_history, ego_state])

    # def print_agent_info(self):
    #     print '----------'
    #     print 'Global Frame:'
    #     print '(px,py):', self.pos_global_frame
    #     print '(vx,vy):', self.vel_global_frame
    #     print 'speed:', self.speed_global_frame
    #     print 'heading:', self.heading_global_frame
    #     print 'Body Frame:'
    #     print '(vx,vy):', self.vel_ego_frame
    #     print 'heading:', self.heading_ego_frame
    #     print '----------'

    def to_vector(self):
        global_state = np.array([self.pos_global_frame[0], self.pos_global_frame[1], \
            self.goal_global_frame[0], self.goal_global_frame[1], self.radius, self.pref_speed, \
            self.vel_global_frame[0], self.vel_global_frame[1], self.speed_global_frame, self.heading_global_frame])
        ego_state = np.array([self.dist_to_goal, self.heading_ego_frame])
        return global_state, ego_state

    def observe(self, agents):
        #
        # Observation vector is as follows;
        # [<this_agent_info>, <other_agent_1_info>, <other_agent_2_info>, ... , <other_agent_n_info>] 
        # where <this_agent_info> = [id, dist_to_goal, heading (in ego frame)]
        # where <other_agent_i_info> = [pos in this agent's ego parallel coord, pos in this agent's ego orthog coord]
        #

        # Own agent state
        obs = np.array([self.dist_to_goal, self.heading_ego_frame])
        # # obs = np.array([self.id, self.dist_to_goal, self.heading_ego_frame, self.pref_speed, self.radius])

        # for i, other_agent in enumerate(agents):
        #     if other_agent.id == self.id:
        #         continue
        #     # project other elements onto the new reference frame
        #     rel_pos_to_other_global_frame = other_agent.pos_global_frame - self.pos_global_frame
        #     p_parallel_ego_frame = np.dot(rel_pos_to_other_global_frame, self.ref_prll)
        #     p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame, self.ref_orth)
        #     v_parallel_ego_frame = np.dot(other_agent.vel_global_frame, self.ref_prll)
        #     v_orthog_ego_frame = np.dot(other_agent.vel_global_frame, self.ref_orth)
        #     dist_2_other = np.linalg.norm(rel_pos_to_other_global_frame)
        #     other_agent_obs = np.array([p_parallel_ego_frame, p_orthog_ego_frame])
        #     # other_agent_obs = np.array([p_parallel_ego_frame, p_orthog_ego_frame, v_parallel_ego_frame, v_orthog_ego_frame])
        #     # other_agent_obs = np.array([p_parallel_ego_frame, p_orthog_ego_frame, v_parallel_ego_frame, v_orthog_ego_frame, other_agent.radius])
        #     obs = np.hstack([obs, other_agent_obs])

            # is_on = np.ones((HIST_NUMPTS,))
            # is_history_on = other_stateHist[:,9] > -EPS
            # dist_2_other = np.clip(np.linalg.norm(agent_stateHist[:,0:2]-other_stateHist[:,0:2], axis=1) \
            # - self_radius - other_radius, 0, 10)
            # state_nn_history[:,9+9*i:9+9*(i+1)] = np.vstack((rel_pos_x, rel_pos_y, other_vx, other_vy, other_radius, \
            # self_radius+other_radius, dist_2_other, is_history_on, is_on)).transpose()



        # obs = np.array([a.pos_global_frame[0], a.pos_global_frame[1], a.heading_global_frame])
        # obs = np.hstack([agents[0].to_vector(), agents[1].to_vector()])
        return obs

    def get_ref(self):
        #
        # Using current and goal position of agent in global frame,
        # compute coordinate axes of ego frame
        #
        # Returns:
        # ref_prll: vector pointing from agent position -> goal
        # ref_orthog: vector orthogonal to ref_prll
        #
        goal_direction = self.goal_global_frame - self.pos_global_frame
        self.dist_to_goal = np.linalg.norm(goal_direction)
        if self.dist_to_goal > 1e-8:
            ref_prll = goal_direction / self.dist_to_goal
        else:
            ref_prll = goal_direction
        ref_orth = np.array([-ref_prll[1], ref_prll[0]]) # rotate by 90 deg
        return ref_prll, ref_orth


##################
# Utils
################

def rad2deg(rad):
    return rad*180/np.pi
# angle_1 - angle_2
# contains direction in range [-3.14, 3.14]
def find_angle_diff(angle_1, angle_2):
    angle_diff_raw = angle_1 - angle_2
    angle_diff = (angle_diff_raw + np.pi) % (2 * np.pi) - np.pi
    return angle_diff

# keep angle between [-pi, pi]
def wrap(angle):
    while angle >= np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle

