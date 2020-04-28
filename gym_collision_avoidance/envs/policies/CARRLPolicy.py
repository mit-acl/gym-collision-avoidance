import numpy as np
from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy

class CARRLPolicy(ExternalPolicy):
    """ Wrapper for the policy related to `Certified Adversarial Robustness for Deep Reinforcement Learning <https://arxiv.org/abs/2004.06496>`_

    .. note::
        None of the interesting aspects of the policy are implemented here, as that software is under IP protection currently.
    """
    def __init__(self):
        ExternalPolicy.__init__(self, str="CARRL")
        num_actions = 11
        max_heading_change = np.pi / 6
        self.actions = np.zeros((num_actions, 2)) # two for vel, heading 
        self.actions[:, 0] = np.ones((num_actions,))  # vel
        self.actions[:, 1] = np.linspace(
            -max_heading_change, max_heading_change, num_actions)  # vel

    def convert_to_action(self, discrete_action):
        """ The CARRL code (external) provides the index of the desired action (but doesn't need to know the details of what that means in this environment),
        so we convert that index to an environment-specific action here.

        Args:
            discrete_action (int): index corresponding to the desired element of self.actions

        Returns:
            [speed, heading delta] corresponding to the provided index

        """
        return self.actions[discrete_action,:]