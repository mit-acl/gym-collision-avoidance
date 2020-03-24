import numpy as np
from gym_collision_avoidance.envs.config import Config as EnvConfig; Config = EnvConfig()
from gym_collision_avoidance.envs.util import wrap, find_nearest
import operator
import math

class Agent(object):
    def __init__(self, start_x, start_y, goal_x, goal_y, radius,
                 pref_speed, initial_heading, policy, dynamics_model, sensors, id):
        self.policy = policy()
        self.dynamics_model = dynamics_model(self)
        self.sensors = [sensor() for sensor in sensors]

        # Global Frame states
        self.pos_global_frame = np.array([start_x, start_y], dtype='float64')
        self.goal_global_frame = np.array([goal_x, goal_y], dtype='float64')
        self.vel_global_frame = np.array([0.0, 0.0], dtype='float64')
        self.speed_global_frame = 0.0

        if initial_heading is None:
            vec_to_goal = self.goal_global_frame - self.pos_global_frame
            self.heading_global_frame = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        else:
            self.heading_global_frame = initial_heading
        self.delta_heading_global_frame = 0.0

        # Ego Frame states
        self.speed_ego_frame = 0.0
        self.heading_ego_frame = 0.0
        self.vel_ego_frame = np.array([0.0, 0.0])

        # Store past selected actions
        self.chosen_action_dict = {}

        self.num_actions_to_store = 2
        self.action_dim = 2
        self.past_actions = np.zeros((self.num_actions_to_store,
                                      self.action_dim))

        # Other parameters
        self.radius = radius
        self.pref_speed = pref_speed
        self.id = id
        self.dist_to_goal = 0.0
        self.near_goal_threshold = Config.NEAR_GOAL_THRESHOLD

        self.straight_line_time_to_reach_goal = (np.linalg.norm(self.pos_global_frame - self.goal_global_frame) - self.near_goal_threshold)/self.pref_speed
        if Config.EVALUATE_MODE or Config.PLAY_MODE:
            self.time_remaining_to_reach_goal = Config.MAX_TIME_RATIO*self.straight_line_time_to_reach_goal
        else:
            self.time_remaining_to_reach_goal = Config.MAX_TIME_RATIO*self.straight_line_time_to_reach_goal
        self.t = 0.0
        self.t_offset = None
        self.step_num = 0

        self.is_at_goal = False
        self.was_at_goal_already = False
        self.was_in_collision_already = False
        self.in_collision = False
        self.ran_out_of_time = False

        self.min_x = -20.0
        self.max_x = 20.0
        self.min_y = -20.0
        self.max_y = 20.0

        self.num_states_in_history = 1000
        self.global_state_dim = 11
        self.global_state_history = np.empty((self.num_states_in_history, self.global_state_dim))
        self.ego_state_dim = 3
        self.ego_state_history = np.empty((self.num_states_in_history, self.ego_state_dim))

        # self.past_actions = np.zeros((self.num_actions_to_store,2))
        self.past_global_velocities = np.zeros((self.num_actions_to_store,2))
        self.past_global_velocities = self.vel_global_frame * np.ones((self.num_actions_to_store,2))

        self.other_agent_states = np.zeros((7,))

        self.dynamics_model.update_ego_frame()
        # self._update_state_history()
        # self._check_if_at_goal()
        # self.take_action([0.0, 0.0])

        self.dt_nominal = Config.DT

        self.min_dist_to_other_agents = np.inf

        self.turning_dir = 0.0

        # self.latest_laserscan = LaserScan()
        # self.latest_laserscan.ranges = 10*np.ones(Config.LASERSCAN_LENGTH)

        self.is_done = False

    def __deepcopy__(self, memo):
        # Copy every attribute about the agent except its policy
        cls = self.__class__
        obj = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k != 'policy':
                setattr(obj, k, v)
        return obj

    def _check_if_at_goal(self):
        is_near_goal = (self.pos_global_frame[0] - self.goal_global_frame[0])**2 + (self.pos_global_frame[1] - self.goal_global_frame[1])**2 <= self.near_goal_threshold**2
        self.is_at_goal = is_near_goal

    def set_state(self, px, py, vx=None, vy=None, heading=None):
        if vx is None or vy is None:
            if self.step_num == 0:
                # On first timestep, just set to zero
                self.vel_global_frame = np.array([0,0])
            else:
                # Interpolate velocity from last pos
                self.vel_global_frame = (np.array([px, py]) - self.pos_global_frame) / self.dt_nominal
        else:
            self.vel_global_frame = np.array([vx, vy])

        if heading is None:
            # Estimate heading to be the direction of the velocity vector
            heading = np.arctan2(self.vel_global_frame[1], self.vel_global_frame[0])
            self.delta_heading_global_frame = wrap(heading - self.heading_global_frame)
        else:
            self.delta_heading_global_frame = wrap(heading - self.heading_global_frame)

        self.pos_global_frame = np.array([px, py])
        self.speed_global_frame = np.linalg.norm(self.vel_global_frame)
        self.heading_global_frame = heading

    def take_action(self, action, dt):
        if self.is_at_goal or self.ran_out_of_time or self.in_collision:
            if self.is_at_goal:
                self.was_at_goal_already = True
            if self.in_collision:
                self.was_in_collision_already = True
            self.vel_global_frame = np.array([0.0, 0.0])
            self._store_past_velocities()
            return

        # Store past actions
        self.past_actions = np.roll(self.past_actions, 1, axis=0)
        self.past_actions[0, :] = action

        # Store info about the TF btwn the ego frame and global frame before moving agent
        goal_direction = self.goal_global_frame - self.pos_global_frame 
        theta = np.arctan2(goal_direction[1], goal_direction[0])
        self.T_global_ego = np.array([[np.cos(theta), -np.sin(theta), self.pos_global_frame[0]], [np.sin(theta), np.cos(theta), self.pos_global_frame[1]], [0,0,1]])
        self.ego_to_global_theta = theta

        # In the case of ExternalDynamics, this call does nothing,
        # but set_state was called instead
        self.dynamics_model.step(action, dt)

        self.dynamics_model.update_ego_frame()

        self._update_state_history()

        self._check_if_at_goal()

        self._store_past_velocities()
        
        # Update time left so agent does not run around forever
        self.time_remaining_to_reach_goal -= dt
        self.t += dt
        self.step_num += 1
        if self.time_remaining_to_reach_goal <= 0.0:
            self.ran_out_of_time = True

        return

    def sense(self, agents, agent_index, top_down_map):
        self.sensor_data = {}
        for sensor in self.sensors:
            sensor_data = sensor.sense(agents, agent_index, top_down_map)
            self.sensor_data[sensor.name] = sensor_data

    def _update_state_history(self):
        global_state, ego_state = self.to_vector()
        self.global_state_history[self.step_num, :] = global_state
        self.ego_state_history[self.step_num, :] = ego_state

    def print_agent_info(self):
        print('----------')
        print('Global Frame:')
        print('(px,py):', self.pos_global_frame)
        print('(vx,vy):', self.vel_global_frame)
        print('speed:', self.speed_global_frame)
        print('heading:', self.heading_global_frame)
        print('Body Frame:')
        print('(vx,vy):', self.vel_ego_frame)
        print('heading:', self.heading_ego_frame)
        print('----------')

    def to_vector(self):
        global_state = np.array([self.t,
                                 self.pos_global_frame[0],
                                 self.pos_global_frame[1],
                                 self.goal_global_frame[0],
                                 self.goal_global_frame[1],
                                 self.radius,
                                 self.pref_speed,
                                 self.vel_global_frame[0],
                                 self.vel_global_frame[1],
                                 self.speed_global_frame,
                                 self.heading_global_frame])
        ego_state = np.array([self.t, self.dist_to_goal, self.heading_ego_frame])
        return global_state, ego_state

    def get_sensor_data(self, sensor_name):
        if sensor_name in self.sensor_data:
            return self.sensor_data[sensor_name]

    def get_agent_data(self, attribute):
        return getattr(self, attribute)

    def get_agent_data_equiv(self, attribute, value):
        return eval("self."+attribute) == value

    def get_observation_dict(self, agents):
        observation = {}
        for state in Config.STATES_IN_OBS:
            observation[state] = np.array(eval("self." + Config.STATE_INFO_DICT[state]['attr']))
        return observation

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
        self.dist_to_goal = math.sqrt(goal_direction[0]**2 + goal_direction[1]**2)
        if self.dist_to_goal > 1e-8:
            ref_prll = goal_direction / self.dist_to_goal
        else:
            ref_prll = goal_direction
        ref_orth = np.array([-ref_prll[1], ref_prll[0]])  # rotate by 90 deg
        return ref_prll, ref_orth

    def _store_past_velocities(self):
        self.past_global_velocities = np.roll(self.past_global_velocities,1,axis=0)
        self.past_global_velocities[0,:] = self.vel_global_frame

    def ego_pos_to_global_pos(self, ego_pos):
        # goal_direction = self.goal_global_frame - self.pos_global_frame 
        # theta = np.arctan2(goal_direction[1], goal_direction[0])
        # T_global_ego = np.array([[np.cos(theta), -np.sin(theta), self.pos_global_frame[0]], [np.sin(theta), np.cos(theta), self.pos_global_frame[1]], [0,0,1]])
        if ego_pos.ndim == 1:
            ego_pos_ = np.array([ego_pos[0], ego_pos[1], 1])
            global_pos = np.dot(self.T_global_ego, ego_pos_)
            return global_pos[:2]
        else:
            ego_pos_ = np.hstack([ego_pos, np.ones((ego_pos.shape[0],1))])
            global_pos = np.dot(self.T_global_ego, ego_pos_.T).T
            return global_pos[:,:2]

    def global_pos_to_ego_pos(self, global_pos):
        # goal_direction = self.goal_global_frame - self.pos_global_frame 
        # theta = np.arctan2(goal_direction[1], goal_direction[0])
        # T_ego_global = np.linalg.inv(np.array([[np.cos(theta), -np.sin(theta), self.pos_global_frame[0]], [np.sin(theta), np.cos(theta), self.pos_global_frame[1]], [0,0,1]]))
        ego_pos = np.dot(np.linalg.inv(self.T_global_ego), np.array([global_pos[0], global_pos[1], 1]))
        return ego_pos[:2]

if __name__ == '__main__':
    start_x = -3
    start_y = 1
    goal_x = 3
    goal_y = 0
    radius = 0.5
    pref_speed = 1.2
    initial_heading = 0.0
    from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
    from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
    policy = GA3CCADRLPolicy
    dynamics_model = UnicycleDynamics
    sensors = []
    id = 0
    agent = Agent(start_x, start_y, goal_x, goal_y, radius,
                 pref_speed, initial_heading, policy, dynamics_model, sensors, id)
    print(agent.ego_pos_to_global_pos(np.array([1,0.5])))
    print(agent.global_pos_to_ego_pos(np.array([-1.93140658, 1.32879797])))
    # agents = [Agent(start_x, start_y, goal_x, goal_y, radius,
    #              pref_speed, initial_heading, i) for i in range(4)]
    # agents[0].observe(agents)
    print("Created Agent.")