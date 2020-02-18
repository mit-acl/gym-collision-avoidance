import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy

class ExternalPolicy(Policy):
    def __init__(self, str="External"):
        Policy.__init__(self, str=str)
        self.is_external = True
        self.is_still_learning = False

    def find_next_action(self, obs, agents, i):
        # External Policy agents don't set their actions, someone else does (e.g. real human)
        raise NotImplementedError