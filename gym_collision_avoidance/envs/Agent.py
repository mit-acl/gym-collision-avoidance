import numpy as np
import util

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
        selected_heading = util.wrap(action[1] + self.heading_global_frame) # in global frame

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
        self.heading_ego_frame = util.wrap(self.heading_global_frame - ref_prll_angle_global_frame)

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

    def print_agent_info(self):
        print '----------'
        print 'Global Frame:'
        print '(px,py):', self.pos_global_frame
        print '(vx,vy):', self.vel_global_frame
        print 'speed:', self.speed_global_frame
        print 'heading:', self.heading_global_frame
        print 'Body Frame:'
        print '(vx,vy):', self.vel_ego_frame
        print 'heading:', self.heading_ego_frame
        print '----------'

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
        obs = np.array([self.id, self.dist_to_goal, self.heading_ego_frame, self.pref_speed, self.radius])

        for i, other_agent in enumerate(agents):
            if other_agent.id == self.id:
                continue
            # project other elements onto the new reference frame
            rel_pos_to_other_global_frame = other_agent.pos_global_frame - self.pos_global_frame
            p_parallel_ego_frame = np.dot(rel_pos_to_other_global_frame, self.ref_prll)
            p_orthog_ego_frame = np.dot(rel_pos_to_other_global_frame, self.ref_orth)
            v_parallel_ego_frame = np.dot(other_agent.vel_global_frame, self.ref_prll)
            v_orthog_ego_frame = np.dot(other_agent.vel_global_frame, self.ref_orth)
            dist_2_other = np.linalg.norm(rel_pos_to_other_global_frame)
            other_agent_obs = np.array([p_parallel_ego_frame, p_orthog_ego_frame])
            # other_agent_obs = np.array([p_parallel_ego_frame, p_orthog_ego_frame, v_parallel_ego_frame, v_orthog_ego_frame])
            # other_agent_obs = np.array([p_parallel_ego_frame, p_orthog_ego_frame, v_parallel_ego_frame, v_orthog_ego_frame, other_agent.radius])
            obs = np.hstack([obs, other_agent_obs])

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





