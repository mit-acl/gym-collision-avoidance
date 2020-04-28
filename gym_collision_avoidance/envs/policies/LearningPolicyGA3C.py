import numpy as np

from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from gym_collision_avoidance.envs.policies.GA3C_CADRL import network

class LearningPolicyGA3C(LearningPolicy):
    """ The GA3C-CADRL policy while it's still being trained (an external process provides a discrete action input)
    """
    def __init__(self):
        LearningPolicy.__init__(self)
        self.possible_actions = network.Actions()

    def external_action_to_action(self, agent, external_action):
        """ Convert the discrete external_action into an action for this environment using properties about the agent.

        Args:
            agent (:class:`~gym_collision_avoidance.envs.agent.Agent`): the agent who has this policy
            external_action (int): discrete action between 0-11 directly from the network output

        Returns:
            [speed, heading_change] command

        """

        raw_action = self.possible_actions.actions[int(external_action)]
        action = np.array([agent.pref_speed*raw_action[0], raw_action[1]])
        return action