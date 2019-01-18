import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy

class ExternalPolicy(Policy):
    def __init__(self):
        super().__init__()

    def find_next_action(self, obs, agents, i):
        # External Policy agents don't set their actions, someone else does (e.g. real human)
        action = np.array([None, None])
        return action
