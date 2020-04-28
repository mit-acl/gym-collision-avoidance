import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy

class InternalPolicy(Policy):
    """ Convert an observation to an action completely within the environment (for model-based/pre-trained, simulated agents).

    Please see the possible subclasses at :ref:`all_internal_policies`.
    """
    def __init__(self, str="Internal"):
        Policy.__init__(self, str=str)

    def find_next_action(self, obs, agents, i):
        """ Use the provided inputs to select a commanded action [heading delta, speed]

        Args:
            obs (dict): this :class:`~gym_collision_avoidance.envs.agent.Agent` 's observation vector
            agents (list): of :class:`~gym_collision_avoidance.envs.agent.Agent` objects
            i (int): index of agents list corresponding to this agent

        Returns:
            To be implemented by children.
        """
        raise NotImplementedError