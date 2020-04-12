import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy

class StaticPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="Static")
        pass

    def find_next_action(self, obs, agents, i):
        # Static Agents do not move.
        agents[i].goal_global_frame = agents[i].pos_global_frame
        action = np.array([0.0, 0.0])
        return action
