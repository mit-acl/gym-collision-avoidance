import numpy as np
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.util import wrap, find_nearest
import operator

from sensor_msgs.msg import LaserScan


class Agent():
    def __init__(self, start_x, start_y, goal_x, goal_y, radius,
                 pref_speed, initial_heading, id):
        # self.policy_type = "A3C"
        self.policy_type = "PPO"

        # Global Frame states
        self.pos_global_frame = np.array([start_x, start_y], dtype='float64')
        self.goal_global_frame = np.array([goal_x, goal_y], dtype='float64')
        self.vel_global_frame = np.array([0.0, 0.0], dtype='float64')
        self.speed_global_frame = 0.0
        self.heading_global_frame = initial_heading
        self.delta_heading_global_frame = 0.0

        # Ego Frame states
        self.speed_ego_frame = 0.0
        self.heading_ego_frame = 0.0
        self.vel_ego_frame = np.array([0.0, 0.0])

        # Store past selected actions
        self.chosen_action_dict = {}
        self.action_time_lag = 0.0

        self.num_actions_to_store = 3
        self.action_dim = 2
        self.past_actions = np.zeros((self.num_actions_to_store,
                                      self.action_dim))

        # Other parameters
        self.radius = radius
        self.pref_speed = pref_speed
        self.id = id
        self.dist_to_goal = 0.0

        self.time_remaining_to_reach_goal = \
            5*np.linalg.norm(self.pos_global_frame -
                             self.goal_global_frame)/self.pref_speed
        self.t = 0.0

        self.is_at_goal = False
        self.was_at_goal_already = False
        self.was_in_collision_already = False
        self.in_collision = False
        self.ran_out_of_time = False

        self.min_x = -20.0
        self.max_x = 20.0
        self.min_y = -20.0
        self.max_y = 20.0

        self.global_state_history = None
        self.ego_state_history = None
        self.update_state([0.0, 0.0], 0.0)

        self.min_dist_to_other_agents = np.inf

        self.latest_laserscan = LaserScan()
        self.latest_laserscan.ranges = 10*np.ones(Config.LASERSCAN_LENGTH)

    def _check_if_at_goal(self):
        near_goal_threshold = 0.2
        is_near_goal = np.linalg.norm([self.pos_global_frame -
                                       self.goal_global_frame]) \
            <= near_goal_threshold
        self.is_at_goal = is_near_goal

    def update_state(self, action, dt):
        if self.is_at_goal or self.ran_out_of_time or self.in_collision:
            if self.is_at_goal:
                self.was_at_goal_already = True
            if self.in_collision:
                self.was_in_collision_already = True
            self.vel_global_frame = np.array([0.0, 0.0])
            return

        self.past_actions = np.roll(self.past_actions, 1, axis=0)
        self.past_actions[0, :] = action

        if self.action_time_lag > 0:
            # This is a future feature... action_time_lag = 0 for now.
            # Store current action in dictionary,
            # then look up the past action that should be executed this step
            self.chosen_action_dict[self.t] = action
            timestamp_of_action_to_execute = self.t - self.action_time_lag
            if timestamp_of_action_to_execute < 0:
                action_to_execute = np.array([0.0, 0.0])
            else:
                nearest_timestamp, _ = find_nearest(np.array(
                    self.chosen_action_dict.keys()),
                    timestamp_of_action_to_execute)
                action_to_execute = \
                    self.chosen_action_dict[nearest_timestamp[0]]
        else:
            action_to_execute = action

        selected_speed = action_to_execute[0]
        selected_heading = wrap(action_to_execute[1] +
                                self.heading_global_frame)  # in global frame

        dx = selected_speed * np.cos(selected_heading) * dt
        dy = selected_speed * np.sin(selected_heading) * dt
        self.pos_global_frame += np.array([dx, dy])

        # Constrain xy position (global) to be within some rect from origin
        #  self.pos_global_frame[0] = np.clip(self.pos_global_frame[0],
        #                                     self.min_x, self.max_x)
        #  self.pos_global_frame[1] = np.clip(self.pos_global_frame[1],
        #                                     self.min_y, self.max_y)

        self.vel_global_frame[0] = selected_speed * np.cos(selected_heading)
        self.vel_global_frame[1] = selected_speed * np.sin(selected_heading)
        self.speed_global_frame = selected_speed
        self.delta_heading_global_frame = wrap(selected_heading -
                                               self.heading_global_frame)
        self.heading_global_frame = selected_heading

        # Compute heading w.r.t. ref_prll, ref_orthog coordinate axes
        self.ref_prll, self.ref_orth = self.get_ref()
        ref_prll_angle_global_frame = np.arctan2(self.ref_prll[1],
                                                 self.ref_prll[0])
        self.heading_ego_frame = wrap(self.heading_global_frame -
                                      ref_prll_angle_global_frame)

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
        if self.global_state_history is None or self.ego_state_history is None:
            self.global_state_history = np.expand_dims(
                    np.hstack([self.t, global_state]), axis=0)
            self.ego_state_history = np.expand_dims(ego_state, axis=0)
        else:
            self.global_state_history = np.vstack([
                self.global_state_history, np.hstack([self.t, global_state])])
            self.ego_state_history = np.vstack([self.ego_state_history,
                                                ego_state])

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
        global_state = np.array([self.pos_global_frame[0],
                                 self.pos_global_frame[1],
                                 self.goal_global_frame[0],
                                 self.goal_global_frame[1],
                                 self.radius,
                                 self.pref_speed,
                                 self.vel_global_frame[0],
                                 self.vel_global_frame[1],
                                 self.speed_global_frame,
                                 self.heading_global_frame])
        ego_state = np.array([self.dist_to_goal, self.heading_ego_frame])
        return global_state, ego_state

    def observe(self, agents):
        #
        # Observation vector is as follows;
        # [<this_agent_info>, <other_agent_1_info>, ... , <other_agent_n_info>]
        # where <this_agent_info> = [id, dist_to_goal, heading (in ego frame)]
        # where <other_agent_i_info> =
        #  [pos in this agent's ego parallel coord, pos in
        #       this agent's ego orthog coord]

        obs = np.zeros((Config.FULL_LABELED_STATE_LENGTH))

        # Own agent state
        # ID is removed before inputting to NN
        # num other agents is used to rearrange other agents into seq. by NN
        obs[0] = self.id
        # obs[1] should be 1 if using PPO, 0 otherwise,
        #  so that RL trainer can make policy updates on PPO agents only
        obs[1] = self.policy_type == "PPO"
        if Config.MULTI_AGENT_ARCH == 'RNN':
            obs[Config.AGENT_ID_LENGTH] = 0
        obs[Config.AGENT_ID_LENGTH + Config.FIRST_STATE_INDEX:
            Config.AGENT_ID_LENGTH + Config.FIRST_STATE_INDEX +
            Config.HOST_AGENT_STATE_SIZE] = \
            self.dist_to_goal, self.heading_ego_frame, self.pref_speed, \
            self.radius

        other_agent_dists = {}
        for i, other_agent in enumerate(agents):
            if other_agent.id == self.id:
                continue
            # project other elements onto the new reference frame
            rel_pos_to_other_global_frame = other_agent.pos_global_frame - \
                self.pos_global_frame
            dist_between_agent_centers = np.linalg.norm(
                    rel_pos_to_other_global_frame)
            dist_2_other = dist_between_agent_centers - self.radius - \
                other_agent.radius
            if dist_between_agent_centers > Config.SENSING_HORIZON:
                # print "Agent too far away"
                continue
            other_agent_dists[i] = dist_2_other
        sorted_pairs = sorted(other_agent_dists.items(),
                              key=operator.itemgetter(1))
        sorted_inds = [ind for (ind, pair) in sorted_pairs]
        sorted_inds.reverse()
        clipped_sorted_inds = \
            sorted_inds[-Config.MAX_NUM_OTHER_AGENTS_OBSERVED:]
        clipped_sorted_agents = [agents[i] for i in clipped_sorted_inds]

        i = 0
        for other_agent in clipped_sorted_agents:
            if other_agent.id == self.id:
                continue
            # project other elements onto the new reference frame
            rel_pos_to_other_global_frame = other_agent.pos_global_frame - \
                self.pos_global_frame
            p_parallel_ego_frame = np.dot(rel_pos_to_other_global_frame,
                                          self.ref_prll)
            p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame,
                                        self.ref_orth)
            v_parallel_ego_frame = np.dot(other_agent.vel_global_frame,
                                          self.ref_prll)
            v_orthog_ego_frame = np.dot(other_agent.vel_global_frame,
                                        self.ref_orth)
            dist_2_other = np.linalg.norm(rel_pos_to_other_global_frame) - \
                self.radius - other_agent.radius
            combined_radius = self.radius + other_agent.radius
            is_on = 1

            start_index = Config.AGENT_ID_LENGTH + Config.FIRST_STATE_INDEX + \
                Config.HOST_AGENT_STATE_SIZE + \
                Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i
            end_index = Config.AGENT_ID_LENGTH + Config.FIRST_STATE_INDEX + \
                Config.HOST_AGENT_STATE_SIZE + \
                Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*(i+1)

            other_obs = np.array([p_parallel_ego_frame,
                                  p_orthog_ego_frame,
                                  v_parallel_ego_frame,
                                  v_orthog_ego_frame,
                                  other_agent.radius,
                                  combined_radius,
                                  dist_2_other])
            if Config.MULTI_AGENT_ARCH in ['WEIGHT_SHARING', 'VANILLA']:
                other_obs = np.hstack([other_obs, is_on])
            obs[start_index:end_index] = other_obs
            i += 1

        if Config.MULTI_AGENT_ARCH == 'RNN':
            # will be used by RNN for seq_length
            obs[Config.AGENT_ID_LENGTH] = i
        if Config.MULTI_AGENT_ARCH in ['WEIGHT_SHARING', 'VANILLA']:
            for j in range(i, Config.MAX_NUM_OTHER_AGENTS_OBSERVED):
                start_index = Config.AGENT_ID_LENGTH + \
                        Config.FIRST_STATE_INDEX + \
                        Config.HOST_AGENT_STATE_SIZE + \
                        Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*j
                end_index = Config.AGENT_ID_LENGTH + \
                    Config.FIRST_STATE_INDEX + \
                    Config.HOST_AGENT_STATE_SIZE + \
                    Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*(j+1)
                other_obs[-1] = 0
                obs[start_index:end_index] = other_obs

        if Config.USE_LASERSCAN_IN_OBSERVATION:
            start_index = end_index
            end_index = start_index + Config.LASERSCAN_LENGTH
            obs[start_index:end_index] = self.latest_laserscan.ranges

        #  Only adds previous 1 action to state vector
        # past_actions = self.past_actions[1:3,:].flatten()
        # obs = np.hstack([obs, past_actions])

        if Config.TRAIN_ON_MULTIPLE_AGENTS:
            return obs
        else:
            # if only one agent is being trained on,
            # the agent's ID is irrelevant (always 0), so cut it off obs vector
            return obs[2:]

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
        ref_orth = np.array([-ref_prll[1], ref_prll[0]])  # rotate by 90 deg
        return ref_prll, ref_orth
