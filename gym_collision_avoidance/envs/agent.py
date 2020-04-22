import numpy as np
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import wrap, find_nearest
import operator
import math

class Agent(object):
    """ A disc-shaped object that has a policy, dynamics, sensors, and can move through the environment

    :param start_x: (float or int) x position of agent in global frame at start of episode
    :param start_y: (float or int) y position of agent in global frame at start of episode
    :param goal_x: (float or int) desired x position of agent in global frame by end of episode
    :param goal_y: (float or int) desired y position of agent in global frame by end of episode
    :param radius: (float or int) radius of circle describing disc-shaped agent's boundaries in meters
    :param pref_speed: (float or int) maximum speed of agent in m/s
    :param initial_heading: (float) angle of agent in global frame at start of episode
    :param policy: (:class:`~gym_collision_avoidance.envs.policies.Policy.Policy`) computes agent's action from its state
    :param dynamics_model: (:class:`~gym_collision_avoidance.envs.dynamics.Dynamics.Dynamics`) computes agent's new state from its state and action
    :param sensors: (list) of :class:`~gym_collision_avoidance.envs.sensors.Sensor.Sensor` measures the environment for use by the policy
    :param id: (int) not sure how much it's used, but uniquely identifies each agent

    :param action_dim: (int) number of actions on each timestep (e.g., 2 because of speed, heading cmds)
    :param num_actions_to_store: (int) number of past action vectors to remember (I think just used by CADRL to compute turning_dir?)
    
    :param near_goal_threshold: (float) once within this distance to goal, say that agent has reached goal
    :param dt_nominal: (float) time in seconds of each simulation step

    """
    def __init__(self, start_x, start_y, goal_x, goal_y, radius,
                 pref_speed, initial_heading, policy, dynamics_model, sensors, id):
        self.policy = policy()
        self.dynamics_model = dynamics_model(self)
        self.sensors = [sensor() for sensor in sensors]

        # Store past selected actions
        self.chosen_action_dict = {}

        self.num_actions_to_store = 2
        self.action_dim = 2
        
        self.id = id
        self.dist_to_goal = 0.0
        self.near_goal_threshold = Config.NEAR_GOAL_THRESHOLD
        self.dt_nominal = Config.DT

        self.min_x = -20.0
        self.max_x = 20.0
        self.min_y = -20.0
        self.max_y = 20.0

        self.t_offset = None
        self.global_state_dim = 11
        self.ego_state_dim = 3
        
        self.reset(px=start_x, py=start_y, gx=goal_x, gy=goal_y, pref_speed=pref_speed, radius=radius, heading=initial_heading)

    def reset(self, px=None, py=None, gx=None, gy=None, pref_speed=None, radius=None, heading=None):
        """ Reset an agent with different states/goal, delete history and reset timer (but keep its dynamics, policy, sensors)

        :param px: (float or int) x position of agent in global frame at start of episode
        :param py: (float or int) y position of agent in global frame at start of episode
        :param gx: (float or int) desired x position of agent in global frame by end of episode
        :param gy: (float or int) desired y position of agent in global frame by end of episode
        :param pref_speed: (float or int) maximum speed of agent in m/s
        :param radius: (float or int) radius of circle describing disc-shaped agent's boundaries in meters
        :param heading: (float) angle of agent in global frame at start of episode

        """
        # Global Frame states
        if px is not None and py is not None:
            self.pos_global_frame = np.array([px, py], dtype='float64')
        if gx is not None and gy is not None:
            self.goal_global_frame = np.array([gx, gy], dtype='float64')
        self.vel_global_frame = np.array([0.0, 0.0], dtype='float64')
        self.speed_global_frame = 0.0

        if heading is None:
            vec_to_goal = self.goal_global_frame - self.pos_global_frame
            self.heading_global_frame = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        else:
            self.heading_global_frame = heading
        self.delta_heading_global_frame = 0.0

        # Ego Frame states
        self.speed_ego_frame = 0.0
        self.heading_ego_frame = 0.0
        self.vel_ego_frame = np.array([0.0, 0.0])

        self.past_actions = np.zeros((self.num_actions_to_store,
                                      self.action_dim))

        # Other parameters
        if radius is not None:
            self.radius = radius
        if pref_speed is not None:
            self.pref_speed = pref_speed

        self.straight_line_time_to_reach_goal = (np.linalg.norm(self.pos_global_frame - self.goal_global_frame) - self.near_goal_threshold)/self.pref_speed
        if Config.EVALUATE_MODE or Config.PLAY_MODE:
            self.time_remaining_to_reach_goal = Config.MAX_TIME_RATIO*self.straight_line_time_to_reach_goal
        else:
            self.time_remaining_to_reach_goal = Config.MAX_TIME_RATIO*self.straight_line_time_to_reach_goal
        self.time_remaining_to_reach_goal = max(self.time_remaining_to_reach_goal, self.dt_nominal)
        self.t = 0.0

        self.step_num = 0

        self.is_at_goal = False
        self.was_at_goal_already = False
        self.was_in_collision_already = False
        self.in_collision = False
        self.ran_out_of_time = False

        self.num_states_in_history = int(1.2*self.time_remaining_to_reach_goal / self.dt_nominal)
        self.global_state_history = np.empty((self.num_states_in_history, self.global_state_dim))
        self.ego_state_history = np.empty((self.num_states_in_history, self.ego_state_dim))

        # self.past_actions = np.zeros((self.num_actions_to_store,2))
        self.past_global_velocities = np.zeros((self.num_actions_to_store,2))
        self.past_global_velocities = self.vel_global_frame * np.ones((self.num_actions_to_store,2))

        self.other_agent_states = np.zeros((7,))

        self.dynamics_model.update_ego_frame()
        # self._update_state_history()
        # self._check_if_at_goal()
        # self.take_action([0.0, 0.0])

        self.min_dist_to_other_agents = np.inf

        self.turning_dir = 0.0

        # self.latest_laserscan = LaserScan()
        # self.latest_laserscan.ranges = 10*np.ones(Config.LASERSCAN_LENGTH)

        self.is_done = False


    def __deepcopy__(self, memo):
        """ Copy every attribute about the agent except its policy (since that may contain MBs of DNN weights) """
        cls = self.__class__
        obj = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k != 'policy':
                setattr(obj, k, v)
        return obj

    def _check_if_at_goal(self):
        """ Set :code:`self.is_at_goal` if norm(pos_global_frame - goal_global_frame) <= near_goal_threshold """
        is_near_goal = (self.pos_global_frame[0] - self.goal_global_frame[0])**2 + (self.pos_global_frame[1] - self.goal_global_frame[1])**2 <= self.near_goal_threshold**2
        self.is_at_goal = is_near_goal

    def set_state(self, px, py, vx=None, vy=None, heading=None):
        """ Without doing a full reset, update the agent's current state (pos, vel, heading).

        This is useful in conjunction with (:class:`~gym_collision_avoidance.envs.dynamics.ExternalDynamics.ExternalDynamics`).
        For instance, if Agents in this environment should be aware of a real robot or one from a more realistic simulator (e.g., Gazebo),
        you could just call set_state each sim step based on what the robot/Gazebo tells you.

        If vx, vy not provided, will interpolate based on new&old position. Same for heading.

        Args:
            px (float or int): x position of agent in global frame right now
            py (float or int): y position of agent in global frame right now
            vx (float or int): x velocity of agent in global frame right now
            vy (float or int): y velocity of agent in global frame right now
            heading (float): angle of agent in global frame right now
        """
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
        """ If not yet done, take action for dt seconds, check if done.

        Args:
            action (list): nominally a [delta heading angle, speed] command for this agent (but probably could be anything that the dynamics_model.step can handle)
            dt (float): time in seconds to execute :code:`action`

        """

        # Agent is done if any of these conditions hold (at goal, out of time, in collision). Stop moving if so & ignore the action.
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
        # but set_state should have been called instead
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
        """ Call the sense method of each Sensor in self.sensors, store in self.sensor_data dict keyed by sensor.name.

        Args:
            agents (list): all :class:`~gym_collision_avoidance.envs.agent.Agent` in the environment
            agent_index (int): index of this agent (the one with this sensor) in :code:`agents`
            top_down_map (2D np array): binary image with 0 if that pixel is free space, 1 if occupied

        """
        self.sensor_data = {}
        for sensor in self.sensors:
            sensor_data = sensor.sense(agents, agent_index, top_down_map)
            self.sensor_data[sensor.name] = sensor_data

    def _update_state_history(self):
        global_state, ego_state = self.to_vector()
        self.global_state_history[self.step_num, :] = global_state
        self.ego_state_history[self.step_num, :] = ego_state

    def print_agent_info(self):
        """ Print out a summary of the agent's current state. """
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
        """ Convert the agent's attributes to a single global state vector. """
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
        """ Extract the latest measurement from the sensor by looking up in the self.sensor_data dict (which is populated by the self.sense method. 

        Args:
            sensor_name (str): name of the sensor (e.g., 'laserscan', I think from Sensor.str?)
    
        """
        if sensor_name in self.sensor_data:
            return self.sensor_data[sensor_name]

    def get_agent_data(self, attribute):
        """ Grab the value of self.attribute (useful to define which states sensor uses from config file).
    
        Args:
            attribute (str): which attribute of this agent to look up (e.g., "pos_global_frame")
    
        """
        return getattr(self, attribute)

    def get_agent_data_equiv(self, attribute, value):
        """ Grab the value of self.attribute and return whether it's equal to value (useful to define states sensor uses from config file). 
        
        Args:
            attribute (str): which attribute of this agent to look up (e.g., "radius")
            value (anything): thing to compare self.attribute to (e.g., 0.23)

        Returns:
            result of self.attribute and value comparison (bool)

        """
        return eval("self."+attribute) == value

    def get_observation_dict(self, agents):
        observation = {}
        for state in Config.STATES_IN_OBS:
            observation[state] = np.array(eval("self." + Config.STATE_INFO_DICT[state]['attr']))
        return observation

    def get_ref(self):
        """ Using current and goal position of agent in global frame, compute coordinate axes of ego frame.

        Ego frame is defined as: origin at center of agent, x-axis pointing from agent's center to agent's goal (right-hand rule, z axis upward).
        This is a useful representation for goal-conditioned tasks, since many configurations of agent-pos-and-goal in the global frame map to the same ego setup. 

        Returns:
        2-element tuple containing

        - **ref_prll** (*np array*): (2,) with vector corresponding to ego-x-axis (pointing from agent_position->goal)
        - **ref_orth** (*np array*): (2,) with vector corresponding to ego-y-axis (orthogonal to ref_prll)

        """
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
        """ Convert a position in the ego frame to the global frame.
    
        This might be useful for plotting some of the perturbation stuff.

        Args:
            ego_pos (np array): if (2,), it represents one (x,y) position in ego frame
                if (n,2), it represents n (x,y) positions in ego frame

        Returns:
            global_pos (np array): either (2,) (x,y) position in global frame or (n,2) n (x,y) positions in global frame

        """
        if ego_pos.ndim == 1:
            ego_pos_ = np.array([ego_pos[0], ego_pos[1], 1])
            global_pos = np.dot(self.T_global_ego, ego_pos_)
            return global_pos[:2]
        else:
            ego_pos_ = np.hstack([ego_pos, np.ones((ego_pos.shape[0],1))])
            global_pos = np.dot(self.T_global_ego, ego_pos_.T).T
            return global_pos[:,:2]

    def global_pos_to_ego_pos(self, global_pos):
        """ Convert a position in the global frame to the ego frame.

        Args:
            global_pos (np array): one (x,y) position in global frame

        Returns:
            ego_pos (np array): (2,) (x,y) position in ego frame

        """
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