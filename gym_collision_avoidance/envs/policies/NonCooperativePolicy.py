import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy

class NonCooperativePolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="NonCooperativePolicy")

    def find_next_action(self, obs, agents, i):
        # Non Cooperative Agents simply drive at pref speed toward the goal, ignoring other agents.
        action = np.array([agents[i].pref_speed, -agents[i].heading_ego_frame])
        return action
