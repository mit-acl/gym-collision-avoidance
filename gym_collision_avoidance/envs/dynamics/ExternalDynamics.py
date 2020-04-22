import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics

class ExternalDynamics(Dynamics):
    """ For Agents who are not controlled by the simulation (e.g., real robots), but the simulated Agents should be aware of.
    """
    def __init__(self, agent):
        Dynamics.__init__(self, agent)

    def step(self, action, dt):
        """ Return with no changes, since the agent's state was already updated
        """
        return
