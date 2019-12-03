import numpy as np
from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy

class CARRLPolicy(ExternalPolicy):
    def __init__(self):
        ExternalPolicy.__init__(self, str="CARRL")
        num_actions = 11
        max_heading_change = np.pi / 6
        self.actions = np.zeros((num_actions, 2)) # two for vel, heading 
        self.actions[:, 0] = np.ones((num_actions,))  # vel
        self.actions[:, 1] = np.linspace(
            -max_heading_change, max_heading_change, num_actions)  # vel

    def convert_to_action(self, discrete_action):
        return self.actions[discrete_action,:]