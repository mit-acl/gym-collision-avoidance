from gym_collision_avoidance.envs.agent import Agent
import numpy as np

class StaticAgent(Agent):
    def __init__(self, start_x, start_y, goal_x, goal_y, radius, pref_speed, initial_heading, id):
    	# Set goal to be start position
        Agent.__init__(self, start_x, start_y, start_x, start_y, radius, pref_speed, initial_heading, id)
        self.policy_type = "Static"

    def find_next_action(self, agents):
    	# Static Agents do not move
        action = np.array([0.0, 0.0])
        return action