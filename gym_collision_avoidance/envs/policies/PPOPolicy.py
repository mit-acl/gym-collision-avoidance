import numpy as np

from gym_collision_avoidance.envs.policies.Policy import Policy

class PPOPolicy(Policy):
    def __init__(self):
        Policy.__init__(self)

    def find_next_action(self, obs, agents, i):
        raise NotImplementedError
