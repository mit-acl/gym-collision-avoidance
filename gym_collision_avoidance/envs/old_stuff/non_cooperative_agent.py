from gym_collision_avoidance.envs.agent import Agent
import numpy as np

class NonCooperativeAgent(Agent):
    def __init__(self, start_x, start_y, goal_x, goal_y, radius, pref_speed, initial_heading, id):
        Agent.__init__(self, start_x, start_y, goal_x, goal_y, radius, pref_speed, initial_heading, id)
        self.policy_type = "Non_Cooperative"

    def find_next_action(self, agents):
    	# Non Cooperative Agents simply drive at pref speed toward the goal, ignoring other agents.
        action = np.array([self.pref_speed, -self.heading_ego_frame])
        return action